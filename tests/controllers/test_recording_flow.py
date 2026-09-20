"""Tests for RecordingFlow — coroutine-based recording controller."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import wenzi.async_loop as async_loop
from wenzi.audio.system_volume import SystemVolumeBusyError
from wenzi.controllers.recording_flow import Action, RecordingFlow

_FILE = str(Path(__file__).resolve())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_loop():
    """Ensure a fresh asyncio loop for each test."""
    async_loop.shutdown_sync(timeout=2)
    yield
    async_loop.shutdown_sync(timeout=2)


@pytest.fixture(autouse=True)
def mock_type_text(monkeypatch):
    """Prevent type_text from typing into the real cursor position.

    Every test gets this automatically.  Tests that need to assert on
    type_text calls can use the ``mock_type_text`` fixture directly.
    """
    mock = MagicMock()
    monkeypatch.setattr("wenzi.controllers.recording_flow.type_text", mock)
    return mock


@pytest.fixture
def mock_app(tmp_path):
    """Create a mock WenZiApp with all attributes used by RecordingFlow."""
    app = MagicMock()
    app._busy = False

    def _try_begin(name):
        if app._busy:
            return None
        app._busy = True
        return object()

    app._try_begin_op = MagicMock(side_effect=_try_begin)
    app._end_op = MagicMock(
        side_effect=lambda owner: setattr(app, "_busy", False)
    )
    app._config_degraded = False
    app._voice_input_available = True
    app._shutdown_started = False
    app._config = {
        "audio": {
            "duck_system_audio": False,
            "duck_volume_ratio": 0.25,
            "duck_max_volume": 0.05,
        },
        "feedback": {"sound_enabled": True, "visual_indicator": True},
    }
    app._config_path = str(tmp_path / "config.json")
    app._sound_manager = MagicMock()
    app._sound_manager.enabled = True
    app._sound_manager.should_play_start.return_value = True
    app._recording_indicator = MagicMock()
    app._recording_indicator.enabled = True
    app._recording_indicator.show_device_name = False
    app._recording_indicator.current_frame = MagicMock()
    app._recorder = MagicMock()
    app._recorder.is_recording = False
    app._recorder.current_level = 0.5
    app._recorder.last_device_name = "MacBook Pro Microphone"
    app._recorder.device = None

    def _recorder_start(*args, **kwargs):
        # Mirror the real contract: a successful start() flips is_recording
        app._recorder.is_recording = True
        return "MacBook Pro Microphone"

    def _recorder_stop():
        app._recorder.is_recording = False
        return b"fake_wav_data"

    app._recorder.start.side_effect = _recorder_start
    app._recorder.stop.side_effect = _recorder_stop
    app._system_output_ducker = MagicMock()
    app._system_output_ducker.begin.return_value = object()
    app._transcriber = MagicMock()
    app._transcriber.supports_streaming = False
    app._transcriber.transcribe.return_value = f"[mock from {_FILE}::mock_app fixture]"
    app._enhancer = MagicMock()
    app._enhancer.is_active = False
    app._enhancer.mode = "proofread"
    app._enhancer.input_context_level = "basic"
    app._enhance_mode = "proofread"
    app._preview_enabled = False
    app._streaming_overlay = MagicMock()
    app._live_overlay = None
    app._usage_stats = MagicMock()
    app._conversation_history = MagicMock()
    app._enhance_menu_items = {}
    app._enhance_controller = MagicMock()
    app._append_newline = False
    app._output_method = "type"
    app._current_stt_model = MagicMock(return_value="FunASR")
    app._current_llm_model = MagicMock(return_value="openai / gpt-4o")
    app._last_audio_duration = 0.0
    app._build_dynamic_hotwords = MagicMock(return_value=([], None))
    app._preview_controller = MagicMock()
    return app


@pytest.fixture
def flow(mock_app):
    return RecordingFlow(mock_app)


def run(coro):
    """Run a coroutine on the shared loop and return the result."""
    return async_loop.submit(coro).result(timeout=10)


async def _wait_thread_event(
    event: threading.Event,
    *,
    timeout: float = 2.0,
) -> None:
    """Wait for a thread-side barrier without occupying an executor worker."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise TimeoutError("thread event was not set")
        await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLevelPolling:
    def test_constant_level_advances_indicator_ema_on_every_tick(
        self,
        flow,
        mock_app,
    ):
        sleep = AsyncMock(
            side_effect=[None, None, asyncio.CancelledError()]
        )

        with patch(
            "PyObjCTools.AppHelper.callAfter",
            side_effect=lambda callback, *args: callback(*args),
        ), patch(
            "wenzi.controllers.recording_flow.asyncio.sleep",
            sleep,
        ):
            run(flow._poll_level())

        assert mock_app._recording_indicator.update_level.call_count == 3
        mock_app._recording_indicator.update_level.assert_has_calls(
            [
                call(0.5),
                call(0.5),
                call(0.5),
            ]
        )


class TestIsNotBusy:
    def test_initially_not_busy(self, flow):
        assert not flow.is_busy

    def test_busy_returns_early(self, flow, mock_app):
        """A second press while busy should be ignored."""
        # Simulate a long-running task
        async def _block():
            flow._current_task = asyncio.current_task()
            await asyncio.sleep(3600)

        task = async_loop.submit(_block())
        # Give it a moment to start
        async_loop.submit(asyncio.sleep(0.01)).result(timeout=2)

        assert flow.is_busy

        # Second press should be ignored
        run(flow._handle_press("fn"))
        # The original task should still be the current one
        task.cancel()


class TestSoundDelay:
    @pytest.fixture(autouse=True)
    def _fast_delay(self, monkeypatch):
        monkeypatch.setattr(RecordingFlow, "_DELAYED_START_SECS", 0.1)

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_cancel_during_delay(self, mock_ah, _mock_ic, flow, mock_app):
        """Cancel sent during the sound delay should abort immediately."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True

        async def _test():
            # Start press (will begin the recording session)
            await flow._handle_press("fn")
            # Give session time to reach the delay
            await asyncio.sleep(0.05)
            assert flow.is_busy
            # Send cancel
            flow._actions.put_nowait(Action.CANCEL)
            # Wait for session to complete
            await flow._current_task

        run(_test())

        # A quick tap must never touch the microphone — that keeps the
        # reset instant and avoids any system-audio hiccup on press.
        mock_app._recorder.start.assert_not_called()
        # Overlay and indicator must be cleaned up
        mock_app._recording_indicator.hide.assert_called()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_release_during_delay(self, mock_ah, _mock_ic, flow, mock_app):
        """Release during sound delay should abort without recording."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        mock_app._recorder.start.assert_not_called()
        # Overlay and indicator must be cleaned up
        mock_app._recording_indicator.hide.assert_called()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_preview_history_during_delay(self, mock_ah, _mock_ic, flow, mock_app):
        """PREVIEW_HISTORY during sound delay should abort and show history."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.PREVIEW_HISTORY)
            await flow._current_task

        run(_test())

        mock_app._recorder.start.assert_not_called()
        mock_app._recording_indicator.hide.assert_called()
        mock_app._preview_controller.on_show_last_preview.assert_called_once()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_restart_during_delay(self, mock_ah, _mock_ic, flow, mock_app):
        """RESTART during sound delay should restart the session."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True
        mock_app._recorder.start.return_value = None

        restart_seen = asyncio.Event()

        def _start_side_effect(*a, **kw):
            restart_seen.set()
            mock_app._recorder.is_recording = True
            return None

        mock_app._recorder.start.side_effect = _start_side_effect

        async def _test():
            # Trigger press, then restart during delay, then release
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RESTART)
            await asyncio.wait_for(restart_seen.wait(), timeout=2.0)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        mock_app._recorder.start.assert_called()


class TestWarmupStart:
    """Tap grace + gated warm-up: a press that survives _TAP_GRACE_SECS
    starts the mic concurrently with the rest of the sound window
    (frames gated), so speech is captured from the window's end.  A tap
    inside the grace never touches the engine."""

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_tap_within_grace_never_touches_engine(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True
        monkeypatch.setattr(RecordingFlow, "_TAP_GRACE_SECS", 30.0)
        monkeypatch.setattr(RecordingFlow, "_DELAYED_START_SECS", 60.0)

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        mock_app._recorder.start.assert_not_called()
        mock_app._recorder.stop.assert_not_called()
        mock_app._system_output_ducker.begin.assert_not_called()
        mock_app._recording_indicator.hide.assert_called()
        assert not flow.is_busy

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_hold_past_grace_warms_mic_gated_during_guard(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        """After the grace, start() launches with armed=False while the
        sound window is still pending."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True
        monkeypatch.setattr(RecordingFlow, "_TAP_GRACE_SECS", 0.02)
        monkeypatch.setattr(RecordingFlow, "_DELAYED_START_SECS", 30.0)

        async def _test():
            started = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _start(*a, **kw):
                mock_app._recorder.is_recording = True
                loop.call_soon_threadsafe(started.set)
                return "MacBook Pro Microphone"

            mock_app._recorder.start.side_effect = _start
            await flow._handle_press("fn")
            # The guard window (30s) is still pending — the start already
            # ran concurrently with it.
            await asyncio.wait_for(started.wait(), timeout=5.0)
            assert flow.is_busy
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        assert mock_app._recorder.start.call_args.kwargs == {"armed": False}
        mock_app._recorder.arm.assert_not_called()  # released before window end
        mock_app._recorder.stop.assert_called_once()
        assert mock_app._recorder.is_recording is False
        mock_app._recording_indicator.hide.assert_called()
        assert not flow.is_busy

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_bluetooth_skips_chime_and_starts_armed_after_tap_grace(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True
        mock_app._sound_manager.should_play_start.return_value = False
        monkeypatch.setattr(RecordingFlow, "_TAP_GRACE_SECS", 0.02)
        monkeypatch.setattr(RecordingFlow, "_DELAYED_START_SECS", 30.0)

        async def _test():
            started = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _start(*a, **kw):
                mock_app._recorder.is_recording = True
                loop.call_soon_threadsafe(started.set)
                return "AirPods Max"

            mock_app._recorder.start.side_effect = _start
            await flow._handle_press("fn")
            # Bluetooth does not wait for the 30-second sound window.
            await asyncio.wait_for(started.wait(), timeout=2.0)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        mock_app._sound_manager.play.assert_not_called()
        mock_app._usage_stats.record_sound_feedback.assert_not_called()
        assert mock_app._recorder.start.call_args.kwargs == {"armed": True}
        mock_app._recorder.arm.assert_called_once()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_early_release_settles_inflight_start_and_stops_mic(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        """A RELEASE while start() is STILL RUNNING on the executor must
        wait the start out and then close the mic via the single-flight
        shutdown."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True
        monkeypatch.setattr(RecordingFlow, "_TAP_GRACE_SECS", 0.02)
        monkeypatch.setattr(RecordingFlow, "_DELAYED_START_SECS", 30.0)

        async def _test():
            hold = threading.Event()
            started = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _start(*a, **kw):
                loop.call_soon_threadsafe(started.set)
                assert hold.wait(timeout=5)
                mock_app._recorder.is_recording = True
                return "MacBook Pro Microphone"

            mock_app._recorder.start.side_effect = _start
            await flow._handle_press("fn")
            await asyncio.wait_for(started.wait(), timeout=5.0)
            # start() is deterministically in flight (blocked on hold)
            flow._actions.put_nowait(Action.RELEASE)
            hold.set()
            await flow._current_task

        run(_test())

        mock_app._recorder.stop.assert_called_once()
        assert mock_app._recorder.is_recording is False
        mock_app._recorder.arm.assert_not_called()
        assert not flow.is_busy

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_early_cancel_with_hung_start_marks_tainted(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        """When the in-flight start() hangs past _START_TIMEOUT, the early
        cancel abandons it through the taint mechanism (exactly once)."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True
        monkeypatch.setattr(RecordingFlow, "_TAP_GRACE_SECS", 0.02)
        monkeypatch.setattr(RecordingFlow, "_DELAYED_START_SECS", 30.0)
        monkeypatch.setattr(RecordingFlow, "_START_TIMEOUT", 0.2)

        hold = threading.Event()

        async def _test():
            started = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _start(*a, **kw):
                loop.call_soon_threadsafe(started.set)
                hold.wait(timeout=5)
                return None  # abandoned: the recorder never commits

            mock_app._recorder.start.side_effect = _start
            await flow._handle_press("fn")
            await asyncio.wait_for(started.wait(), timeout=5.0)
            flow._actions.put_nowait(Action.CANCEL)
            for _ in range(100):
                if mock_app._recorder.mark_tainted.called:
                    break
                await asyncio.sleep(0.01)
            assert mock_app._recorder.mark_tainted.called
            assert flow.is_busy
            hold.set()
            await flow._current_task

        try:
            run(_test())
        finally:
            hold.set()

        mock_app._recorder.mark_tainted.assert_called_once_with(
            stop_async=False
        )
        mock_app._recorder.stop.assert_not_called()
        mock_app._recording_indicator.hide.assert_called()
        assert not flow.is_busy

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_arm_after_commit_and_before_streaming_attach(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        """Gate order on the happy path: start commits → arm opens the
        gate → indicator activates → streaming attaches.  Gated frames
        can therefore never reach a streaming backend."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True
        monkeypatch.setattr(RecordingFlow, "_TAP_GRACE_SECS", 0.02)
        monkeypatch.setattr(RecordingFlow, "_DELAYED_START_SECS", 0.1)
        mock_app._transcriber.supports_streaming = True
        mock_app._transcriber.stop_streaming.return_value = (
            f"[mock from {_FILE}::test_arm_after_commit]"
        )
        order: list[str] = []

        def _start(*a, **kw):
            order.append("start")
            mock_app._recorder.is_recording = True
            return "MacBook Pro Microphone"

        mock_app._recorder.start.side_effect = _start
        mock_app._recorder.arm.side_effect = lambda: order.append("arm")
        mock_app._recording_indicator.set_recording_active.side_effect = (
            lambda: order.append("active")
        )

        async def _test():
            attached = asyncio.Event()

            def _attach(cb):
                order.append("attach")
                attached.set()

            mock_app._recorder.set_on_audio_chunk.side_effect = _attach

            await flow._handle_press("fn")
            # Streaming attach ⇒ the sound window elapsed and arm() ran;
            # only then may the RELEASE be delivered.
            await asyncio.wait_for(attached.wait(), timeout=5.0)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        assert order == ["start", "arm", "active", "attach"]
        assert mock_app._recorder.start.call_args.kwargs == {"armed": False}

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_sound_disabled_starts_armed_without_guard(
        self, mock_ah, _mock_ic, flow, mock_app, mock_type_text
    ):
        """With sound feedback off there is no guard window: start() runs
        armed and audio flows immediately."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        _text = f"[mock from {_FILE}::test_sound_disabled_starts_armed]"
        mock_app._transcriber.transcribe.return_value = _text

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        assert mock_app._recorder.start.call_args.kwargs == {"armed": True}
        mock_app._recorder.arm.assert_called_once()
        mock_type_text.assert_called_once_with(
            _text, append_newline=False, method="type"
        )

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_session_exception_while_start_inflight_still_closes_mic(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        """A crash between launching start() and awaiting it must settle
        the in-flight start and close the mic it opened."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = True
        monkeypatch.setattr(RecordingFlow, "_TAP_GRACE_SECS", 0.02)
        monkeypatch.setattr(RecordingFlow, "_DELAYED_START_SECS", 30.0)

        async def _test():
            hold = threading.Event()
            started = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _start(*a, **kw):
                loop.call_soon_threadsafe(started.set)
                assert hold.wait(timeout=5)
                mock_app._recorder.is_recording = True
                return "MacBook Pro Microphone"

            mock_app._recorder.start.side_effect = _start
            # Grace passes normally; the warm-up-phase wait crashes while
            # the start is still in flight.
            monkeypatch.setattr(
                flow, "_wait_action",
                AsyncMock(side_effect=[None, RuntimeError("boom")]),
            )
            await flow._handle_press("fn")
            await asyncio.wait_for(started.wait(), timeout=5.0)
            hold.set()
            await flow._current_task

        run(_test())

        mock_app._recorder.stop.assert_called_once()
        assert mock_app._recorder.is_recording is False
        mock_app._recording_indicator.hide.assert_called()
        assert not flow.is_busy


class TestSystemAudioDucking:
    def test_route_refresh_retries_until_updated_output_is_available(
        self, flow, mock_app
    ):
        token = object()
        mock_app._system_output_ducker.refresh.side_effect = [
            False,
            False,
            True,
        ]

        with patch(
            "wenzi.controllers.recording_flow.time.sleep"
        ) as mock_sleep:
            assert run(flow._refresh_output_duck(token)) is True

        assert mock_app._system_output_ducker.refresh.call_args_list == [
            call(token),
            call(token),
            call(token),
        ]
        assert mock_sleep.call_args_list == [
            call(flow._OUTPUT_REFRESH_RETRY_DELAY),
            call(flow._OUTPUT_REFRESH_RETRY_DELAY),
        ]

    def test_recorder_recovery_receives_same_route_refresh_token(
        self, flow, mock_app
    ):
        token = object()
        mock_app._system_output_ducker.refresh.return_value = True

        async def _test():
            return await flow._launch_recorder_start(
                armed=True,
                duck_token=token,
            )

        assert run(_test()) == "MacBook Pro Microphone"
        callback = mock_app._recorder.start.call_args.kwargs[
            "on_route_recovered"
        ]

        assert callback() is True
        mock_app._system_output_ducker.refresh.assert_called_once_with(token)

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_unconfirmed_updated_output_aborts_before_audio_gate_opens(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token
        mock_app._system_output_ducker.refresh.return_value = False
        mock_app._system_output_ducker.end.return_value = True

        async def _test():
            await flow._handle_press("fn")
            task = flow._current_task
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=1)
                return True
            except TimeoutError:
                # Clean up the old behavior deterministically: implementations
                # that continue recording after an unconfirmed refresh wait
                # for RELEASE instead of hanging this regression test.
                flow._actions.put_nowait(Action.RELEASE)
                await asyncio.wait_for(task, timeout=2)
                return False

        with patch("wenzi.controllers.recording_flow.time.sleep"):
            assert run(_test()) is True

        assert mock_app._system_output_ducker.refresh.call_count == (
            flow._OUTPUT_REFRESH_ATTEMPTS
        )
        mock_app._recorder.arm.assert_not_called()
        mock_app._recorder.stop.assert_called_once()
        mock_app._system_output_ducker.end.assert_called_once_with(token)
        assert mock_app._recorder.is_recording is False
        assert not flow.is_busy

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_release_during_duck_ramp_never_starts_microphone(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        entered = threading.Event()
        release_begin = threading.Event()

        def _begin(**_kwargs):
            entered.set()
            assert release_begin.wait(timeout=5)
            return token

        mock_app._system_output_ducker.begin.side_effect = _begin

        async def _test():
            await flow._handle_press("fn")
            await asyncio.get_running_loop().run_in_executor(None, entered.wait)
            flow._actions.put_nowait(Action.RELEASE)
            release_begin.set()
            await flow._current_task

        run(_test())

        mock_app._recorder.start.assert_not_called()
        mock_app._system_output_ducker.end.assert_called_once_with(token)

    def test_cancelled_duck_begin_settles_token_and_restores(
        self, flow, mock_app, monkeypatch
    ):
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        entered = threading.Event()
        release_begin = threading.Event()
        second_shield_entered = threading.Event()
        restore_entered = threading.Event()
        release_restore = threading.Event()
        original_shield = asyncio.shield
        shield_calls = 0

        def _begin(**_kwargs):
            entered.set()
            assert release_begin.wait(timeout=5)
            return token

        def _restore(_token):
            restore_entered.set()
            assert release_restore.wait(timeout=5)
            return True

        def _tracking_shield(awaitable):
            nonlocal shield_calls
            shield_calls += 1
            if shield_calls == 2:
                second_shield_entered.set()
            return original_shield(awaitable)

        mock_app._system_output_ducker.begin.side_effect = _begin
        mock_app._system_output_ducker.end.side_effect = _restore
        monkeypatch.setattr(asyncio, "shield", _tracking_shield)

        async def _test():
            task = asyncio.create_task(flow._begin_output_duck())
            try:
                await _wait_thread_event(entered)
                task.cancel()
                await _wait_thread_event(second_shield_entered)

                # A second cancel used to escape and permanently lose the
                # token returned later by the executor job.
                task.cancel()
                await asyncio.sleep(0.01)
                assert not task.done()
                mock_app._system_output_ducker.end.assert_not_called()

                release_begin.set()
                await _wait_thread_event(restore_entered)
                assert flow._output_restore_task is not None
                assert mock_app._system_output_ducker.end.call_count == 1

                # Restoration itself is also cancellation-resistant.
                task.cancel()
                await asyncio.sleep(0.01)
                assert not task.done()
                release_restore.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release_begin.set()
                release_restore.set()
                if not task.done():
                    await asyncio.gather(task, return_exceptions=True)

        run(_test())

        mock_app._system_output_ducker.end.assert_called_once_with(token)

    def test_cancelled_duck_begin_failure_preserves_cancellation(
        self, flow, mock_app, monkeypatch
    ):
        mock_app._config["audio"]["duck_system_audio"] = True
        entered = threading.Event()
        release_begin = threading.Event()
        second_shield_entered = threading.Event()
        original_shield = asyncio.shield
        shield_calls = 0

        def _begin(**_kwargs):
            entered.set()
            assert release_begin.wait(timeout=5)
            raise RuntimeError("injected begin failure")

        def _tracking_shield(awaitable):
            nonlocal shield_calls
            shield_calls += 1
            if shield_calls == 2:
                second_shield_entered.set()
            return original_shield(awaitable)

        mock_app._system_output_ducker.begin.side_effect = _begin
        monkeypatch.setattr(asyncio, "shield", _tracking_shield)

        async def _test():
            task = asyncio.create_task(flow._begin_output_duck())
            try:
                await _wait_thread_event(entered)
                task.cancel()
                await _wait_thread_event(second_shield_entered)
                release_begin.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release_begin.set()
                if not task.done():
                    await asyncio.gather(task, return_exceptions=True)

        run(_test())

        mock_app._system_output_ducker.end.assert_not_called()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_quick_release_restore_stays_single_flight_when_waiter_cancelled(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        begin_entered = threading.Event()
        allow_begin = threading.Event()
        restore_entered = threading.Event()
        allow_restore = threading.Event()

        def _begin(**_kwargs):
            begin_entered.set()
            assert allow_begin.wait(timeout=5)
            return token

        def _restore(_token):
            restore_entered.set()
            assert allow_restore.wait(timeout=5)
            return True

        mock_app._system_output_ducker.begin.side_effect = _begin
        mock_app._system_output_ducker.end.side_effect = _restore

        async def _test():
            try:
                await flow._handle_press("fn")
                await _wait_thread_event(begin_entered)
                flow._actions.put_nowait(Action.RELEASE)
                allow_begin.set()
                await _wait_thread_event(restore_entered)

                task = flow._current_task
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done()
                assert mock_app._system_output_ducker.end.call_count == 1
                mock_app._end_op.assert_not_called()
            finally:
                allow_begin.set()
                allow_restore.set()
                task = flow._current_task
                if task is not None:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5)

        run(_test())

        mock_app._recorder.start.assert_not_called()
        mock_app._system_output_ducker.end.assert_called_once_with(token)
        mock_app._end_op.assert_called_once()
        assert not mock_app._busy

    def test_pending_start_timeout_retains_duck_until_future_settles(
        self, flow, mock_app, monkeypatch
    ):
        token = object()
        monkeypatch.setattr(flow, "_START_TIMEOUT", 0.01)
        mock_app._system_output_ducker.end.return_value = True

        async def _test():
            pending = asyncio.get_running_loop().create_future()
            flow._pending_start = pending
            settle = asyncio.create_task(
                flow._settle_pending_start(token)
            )
            for _ in range(100):
                if mock_app._recorder.mark_tainted.called:
                    break
                await asyncio.sleep(0.005)
            assert mock_app._recorder.mark_tainted.called
            mock_app._system_output_ducker.end.assert_not_called()

            settle.cancel()
            await asyncio.sleep(0.01)
            settle.cancel()
            await asyncio.sleep(0.01)
            assert not settle.done()
            assert not pending.cancelled()
            assert flow._pending_start is pending
            mock_app._system_output_ducker.end.assert_not_called()

            pending.set_result(None)
            outcome = await asyncio.wait_for(settle, timeout=2)
            assert outcome.releases_owner
            assert (await flow._settle_output_restore()) is outcome

        run(_test())

        mock_app._recorder.mark_tainted.assert_called_once_with(
            stop_async=False
        )
        mock_app._system_output_ducker.end.assert_called_once_with(token)

    def test_end_exception_is_durably_deferred_once(self, flow, mock_app):
        token = object()
        mock_app._system_output_ducker.end.side_effect = RuntimeError("HAL")
        mock_app._system_output_ducker.defer_failed_restore.return_value = True

        outcome = run(flow._restore_output_duck(token))
        settled = run(flow._settle_output_restore())

        assert outcome.value == "deferred"
        assert settled is outcome
        mock_app._system_output_ducker.end.assert_called_once_with(token)
        mock_app._system_output_ducker.defer_failed_restore.assert_called_once_with(
            token
        )

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_transient_defer_failure_recovers_before_op_release(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token
        monkeypatch.setattr(flow, "_OUTPUT_RESTORE_BACKGROUND_DELAY", 0.0)
        monkeypatch.setattr(flow, "_OUTPUT_RESTORE_RETRY_DELAY", 0.0)
        second_cycle_entered = threading.Event()
        allow_second_cycle = threading.Event()
        end_calls = 0
        defer_calls = 0

        def _end(_token):
            nonlocal end_calls
            end_calls += 1
            if end_calls == 4:
                second_cycle_entered.set()
                assert allow_second_cycle.wait(timeout=5)
            return False

        def _defer(_token):
            nonlocal defer_calls
            defer_calls += 1
            return defer_calls == 2

        mock_app._system_output_ducker.end.side_effect = _end
        mock_app._system_output_ducker.defer_failed_restore.side_effect = _defer

        async def _test():
            try:
                await flow._handle_press("fn")
                while not mock_app._recorder.is_recording:
                    await asyncio.sleep(0.005)
                flow._actions.put_nowait(Action.RELEASE)
                await _wait_thread_event(second_cycle_entered)

                mock_app._end_op.assert_not_called()
                assert mock_app._busy
                assert not flow._current_task.done()
            finally:
                allow_second_cycle.set()
                task = flow._current_task
                if task is not None:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5)

        run(_test())

        assert mock_app._system_output_ducker.end.call_count == 6
        assert mock_app._system_output_ducker.defer_failed_restore.call_count == 2
        mock_app._end_op.assert_called_once()
        assert not mock_app._busy

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_exhausted_defer_failure_keeps_owner_without_stale_end(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token
        mock_app._system_output_ducker.end.return_value = False
        mock_app._system_output_ducker.defer_failed_restore.return_value = False
        monkeypatch.setattr(flow, "_OUTPUT_RESTORE_BACKGROUND_ATTEMPTS", 1)
        monkeypatch.setattr(flow, "_OUTPUT_RESTORE_BACKGROUND_DELAY", 0.0)
        monkeypatch.setattr(flow, "_OUTPUT_RESTORE_RETRY_DELAY", 0.0)

        async def _test():
            await flow._handle_press("fn")
            while not mock_app._recorder.is_recording:
                await asyncio.sleep(0.005)
            flow._actions.put_nowait(Action.RELEASE)
            await asyncio.wait_for(flow._current_task, timeout=5)

        run(_test())

        assert mock_app._system_output_ducker.end.call_count == 6
        assert mock_app._system_output_ducker.defer_failed_restore.call_count == 2
        mock_app._end_op.assert_not_called()
        assert mock_app._busy
        assert flow._op_token is not None

        outcome = run(flow._settle_output_restore())
        assert outcome.value == "retained"
        assert mock_app._system_output_ducker.end.call_count == 6
        assert mock_app._system_output_ducker.defer_failed_restore.call_count == 2

    @patch(
        "wenzi.controllers.recording_flow.capture_input_context",
        return_value=None,
    )
    @patch("PyObjCTools.AppHelper")
    def test_busy_volume_recovery_aborts_before_microphone_start(
        self,
        mock_ah,
        _mock_ic,
        flow,
        mock_app,
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        mock_app._system_output_ducker.begin.side_effect = (
            SystemVolumeBusyError("restore busy")
        )

        async def _test():
            await flow._handle_press("fn")
            await flow._current_task

        run(_test())

        mock_app._recorder.start.assert_not_called()
        mock_app._system_output_ducker.end.assert_not_called()
        assert not flow.is_busy

    def test_explicit_restore_failure_retries_are_bounded(
        self, flow, mock_app, caplog
    ):
        token = object()
        mock_app._system_output_ducker.end.return_value = False

        with (
            caplog.at_level(
                "ERROR", logger="wenzi.controllers.recording_flow"
            ),
            patch("wenzi.controllers.recording_flow.time.sleep") as mock_sleep,
        ):
            run(flow._restore_output_duck(token))

        assert mock_app._system_output_ducker.end.call_count == 3
        mock_app._system_output_ducker.end.assert_has_calls(
            [call(token), call(token), call(token)]
        )
        mock_sleep.assert_has_calls(
            [
                call(flow._OUTPUT_RESTORE_RETRY_DELAY),
                call(flow._OUTPUT_RESTORE_RETRY_DELAY),
            ]
        )
        assert mock_sleep.call_count == 2
        mock_app._system_output_ducker.defer_failed_restore.assert_called_once_with(
            token
        )
        assert "restore returned False after 3 attempts" in caplog.text

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_ducks_before_start_and_restores_after_stop(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        order: list[str] = []
        ready = threading.Event()

        mock_app._system_output_ducker.begin.side_effect = (
            lambda **_kwargs: order.append("duck") or token
        )

        def _start(*_args, **_kwargs):
            order.append("start")
            mock_app._recorder.is_recording = True
            return "MacBook Pro Microphone"

        def _refresh(_token):
            order.append("refresh")
            ready.set()

        def _stop():
            order.append("stop")
            mock_app._recorder.is_recording = False
            return b"fake_wav_data"

        mock_app._recorder.start.side_effect = _start
        mock_app._recorder.stop.side_effect = _stop
        mock_app._system_output_ducker.refresh.side_effect = _refresh
        mock_app._system_output_ducker.end.side_effect = (
            lambda _token: order.append("restore")
        )

        async def _test():
            await flow._handle_press("fn")
            await asyncio.get_running_loop().run_in_executor(None, ready.wait)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        assert order[:3] == ["duck", "start", "refresh"]
        assert order.index("stop") < order.index("restore")
        mock_app._system_output_ducker.begin.assert_called_once_with(
            factor=0.25,
            max_volume=0.05,
        )
        mock_app._system_output_ducker.end.assert_called_once_with(token)

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_batch_stt_overlaps_restore_but_op_waits_for_restore(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """Batch STT may run during the restore ramp, but a new operation
        must not enter until the ramp has settled."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token

        first_restore_failed = threading.Event()
        retry_sleep_entered = threading.Event()
        allow_retry = threading.Event()
        transcribe_entered = threading.Event()
        direct_flow_done = threading.Event()
        restore_attempts = 0

        def _restore(_token):
            nonlocal restore_attempts
            restore_attempts += 1
            if restore_attempts == 1:
                first_restore_failed.set()
                return False
            return True

        def _retry_sleep(delay):
            assert delay == flow._OUTPUT_RESTORE_RETRY_DELAY
            retry_sleep_entered.set()
            assert allow_retry.wait(timeout=5)

        def _transcribe(*_args, **_kwargs):
            transcribe_entered.set()
            return "transcribed"

        mock_app._system_output_ducker.end.side_effect = _restore
        mock_app._transcriber.transcribe.side_effect = _transcribe
        mock_app._conversation_history.log.side_effect = (
            lambda **_kwargs: direct_flow_done.set()
        )

        async def _test():
            try:
                await flow._handle_press("fn")
                while not mock_app._recorder.is_recording:
                    await asyncio.sleep(0.005)
                flow._actions.put_nowait(Action.RELEASE)

                await _wait_thread_event(first_restore_failed)
                await _wait_thread_event(retry_sleep_entered)
                await _wait_thread_event(transcribe_entered)
                await _wait_thread_event(direct_flow_done)

                assert mock_app._system_output_ducker.end.call_count == 1
                mock_app._end_op.assert_not_called()
                assert mock_app._busy
                assert not flow._current_task.done()
            finally:
                allow_retry.set()
                task = flow._current_task
                if task is not None:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5)

        with patch(
            "wenzi.controllers.recording_flow.time.sleep",
            side_effect=_retry_sleep,
        ):
            run(_test())

        assert mock_app._system_output_ducker.end.call_count == 2
        mock_app._system_output_ducker.end.assert_has_calls(
            [call(token), call(token)]
        )
        mock_app._end_op.assert_called_once()
        assert not mock_app._busy

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_preview_cannot_release_op_before_restore_settles(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._preview_enabled = True
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token

        restore_entered = threading.Event()
        allow_restore = threading.Event()
        preview_returned = threading.Event()

        def _restore(_token):
            restore_entered.set()
            assert allow_restore.wait(timeout=5)

        mock_app._system_output_ducker.end.side_effect = _restore
        mock_app._do_transcribe_with_preview.side_effect = (
            lambda **_kwargs: preview_returned.set()
        )

        async def _test():
            try:
                await flow._handle_press("fn")
                while not mock_app._recorder.is_recording:
                    await asyncio.sleep(0.005)
                flow._actions.put_nowait(Action.RELEASE)

                await _wait_thread_event(restore_entered)
                await _wait_thread_event(preview_returned)
                mock_app._end_op.assert_not_called()
                assert mock_app._busy
                assert not flow._current_task.done()
            finally:
                allow_restore.set()
                task = flow._current_task
                if task is not None:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5)

        run(_test())

        mock_app._system_output_ducker.end.assert_called_once_with(token)
        mock_app._end_op.assert_called_once()
        assert not mock_app._busy

    @patch("wenzi.scripting.api.alert.alert")
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_start_failure_restores_volume(
        self, mock_ah, _mock_ic, _mock_alert, flow, mock_app
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token
        mock_app._recorder.start.side_effect = lambda **_kwargs: None

        async def _test():
            await flow._handle_press("fn")
            await flow._current_task

        run(_test())

        mock_app._recorder.stop.assert_not_called()
        mock_app._system_output_ducker.end.assert_called_once_with(token)

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_restart_reuses_one_duck_session_without_volume_pulse(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token
        starts = 0
        first_started = threading.Event()
        second_started = threading.Event()

        def _start(*_args, **_kwargs):
            nonlocal starts
            starts += 1
            mock_app._recorder.is_recording = True
            if starts == 1:
                first_started.set()
            elif starts == 2:
                second_started.set()
            return "MacBook Pro Microphone"

        mock_app._recorder.start.side_effect = _start

        async def _test():
            await flow._handle_press("fn")
            await _wait_thread_event(first_started)
            flow._actions.put_nowait(Action.RESTART)
            await _wait_thread_event(second_started)
            mock_app._system_output_ducker.begin.assert_called_once()
            mock_app._system_output_ducker.end.assert_not_called()
            flow._actions.put_nowait(Action.RELEASE)
            await TestQuickRelease._wait_session_done(flow)

        run(_test())

        assert starts == 2
        mock_app._system_output_ducker.begin.assert_called_once()
        mock_app._system_output_ducker.end.assert_called_once_with(token)

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_failed_restart_handoff_restores_volume(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token
        started = threading.Event()

        def _start(*_args, **_kwargs):
            mock_app._recorder.is_recording = True
            started.set()
            return "MacBook Pro Microphone"

        mock_app._recorder.start.side_effect = _start
        flow._hide_live_overlay = MagicMock(
            side_effect=[RuntimeError("UI failed"), None]
        )

        async def _test():
            await flow._handle_press("fn")
            await asyncio.get_running_loop().run_in_executor(None, started.wait)
            flow._actions.put_nowait(Action.RESTART)
            await flow._current_task

        run(_test())

        mock_app._system_output_ducker.end.assert_called_once_with(token)
        assert not flow.is_busy


class TestQuickRelease:
    @staticmethod
    async def _wait_session_done(flow):
        """Let on_press's loop callback run, then wait out the session."""
        await asyncio.sleep(0)
        for _ in range(200):
            task = flow._current_task
            if task is not None and task.done() and not flow.is_busy:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("session did not finish in time")

    @patch("wenzi.controllers.recording_flow.capture_input_context")
    @patch("PyObjCTools.AppHelper")
    def test_release_during_context_capture_not_lost(
        self, mock_ah, mock_ic, flow, mock_app
    ):
        """A RELEASE arriving while the press awaits the input-context
        capture must stop the session promptly, not be dropped."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"] = {"max_recording_seconds": 60}
        _text = f"[mock from {_FILE}::test_release_during_context_capture_not_lost]"
        mock_app._transcriber.transcribe.return_value = _text

        def _release_during_capture(_level):
            # Runs on the executor thread — same as the real quick-tap race
            flow.send_action(Action.RELEASE)
            return None

        mock_ic.side_effect = _release_during_capture

        flow.on_press("fn")
        run(self._wait_session_done(flow))

        mock_app._recorder.start.assert_called_once()
        mock_app._recorder.stop.assert_called_once()
        mock_app._transcriber.transcribe.assert_called_once()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_release_immediately_after_press_not_lost(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """A RELEASE enqueued before the press coroutine's first step
        (extremely fast tap) must still stop the session."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"] = {"max_recording_seconds": 60}
        _text = f"[mock from {_FILE}::test_release_immediately_after_press_not_lost]"
        mock_app._transcriber.transcribe.return_value = _text

        # Press and release queued back-to-back from the hotkey thread:
        # the release callback lands on the loop before the press
        # coroutine's first step runs.
        flow.on_press("fn")
        flow.send_action(Action.RELEASE)
        run(self._wait_session_done(flow))

        mock_app._recorder.start.assert_called_once()
        mock_app._recorder.stop.assert_called_once()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_stray_actions_dropped_while_idle(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """Actions sent while idle are dropped at enqueue time and cannot
        poison the next session."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        _text = f"[mock from {_FILE}::test_stray_actions_dropped_while_idle]"
        mock_app._transcriber.transcribe.return_value = _text

        flow.send_action(Action.CANCEL)  # idle → dropped at enqueue

        async def _test():
            await asyncio.sleep(0.01)
            assert flow._actions.empty()
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await asyncio.wait_for(flow._current_task, timeout=5)

        run(_test())

        # The stray CANCEL must not abort the new session
        mock_app._transcriber.transcribe.assert_called_once()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_press_refused_while_model_switch(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """Recording must not start while another op owns the app slot."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._busy = True  # e.g. a model switch in progress
        flow._show_error_alert = MagicMock()

        run(flow._handle_press("fn"))

        mock_app._recorder.start.assert_not_called()
        flow._show_error_alert.assert_called_once()
        assert not flow.is_busy

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_refused_press_does_not_poison_next_session(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """A RELEASE queued for a refused press must be discarded, and
        the next session must run normally."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        _text = f"[mock from {_FILE}::test_refused_press_does_not_poison_next_session]"
        mock_app._transcriber.transcribe.return_value = _text
        flow._show_error_alert = MagicMock()

        mock_app._busy = True  # switch in progress → press refused
        flow.on_press("fn")
        flow.send_action(Action.RELEASE)

        async def _settle():
            await asyncio.sleep(0.1)
            assert flow._actions.empty()
            assert not flow.is_busy

        run(_settle())
        mock_app._recorder.start.assert_not_called()

        # The next press must record and stop on ITS OWN release
        mock_app._busy = False
        flow.on_press("fn")
        flow.send_action(Action.RELEASE)
        run(self._wait_session_done(flow))

        mock_app._recorder.start.assert_called_once()
        mock_app._transcriber.transcribe.assert_called_once()


class TestPressContextCapture:
    @patch("wenzi.controllers.recording_flow.capture_input_context")
    @patch("wenzi.controllers.recording_flow.get_frontmost_app")
    def test_frontmost_app_captured_before_input_context(
        self, mock_get_frontmost_app, mock_capture_input_context, flow
    ):
        """The original target app must be saved before slow AX context lookup."""
        call_order: list[str] = []
        target_app = object()

        mock_get_frontmost_app.side_effect = (
            lambda: call_order.append("frontmost") or target_app
        )
        mock_capture_input_context.side_effect = (
            lambda _level: call_order.append("context") or None
        )

        async def _noop_session(_key_name: str) -> None:
            return None

        flow._recording_session = _noop_session

        async def _test():
            await flow._handle_press("fn")
            await flow._current_task

        run(_test())

        assert call_order == ["frontmost", "context"]
        assert flow._target_app is target_app

    def test_shutdown_during_context_capture_never_starts_late_session(
        self, flow, mock_app
    ):
        entered = threading.Event()
        release = threading.Event()

        def _blocked_capture(_level):
            entered.set()
            assert release.wait(timeout=2.0)
            return None

        flow._recording_session = AsyncMock()
        with (
            patch(
                "wenzi.controllers.recording_flow.capture_input_context",
                side_effect=_blocked_capture,
            ),
            patch(
                "wenzi.controllers.recording_flow.get_frontmost_app",
                return_value=None,
            ),
        ):
            async def _install_stale_done_task():
                task = asyncio.create_task(asyncio.sleep(0))
                await task
                flow._current_task = task

            run(_install_stale_done_task())
            assert flow._current_task.done()
            flow.on_press("fn")
            assert entered.wait(timeout=2.0)
            mock_app._shutdown_started = True
            release.set()

            async def _wait_until_idle():
                for _ in range(200):
                    if not flow.is_busy:
                        return
                    await asyncio.sleep(0.01)
                raise AssertionError("press did not leave the pending state")

            run(_wait_until_idle())

        flow._recording_session.assert_not_awaited()
        mock_app._recorder.start.assert_not_called()
        mock_app._system_output_ducker.begin.assert_not_called()
        assert mock_app._busy is False


class TestRecordAndRelease:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_full_flow_no_enhance(
        self, mock_ah, _mock_ic, flow, mock_app, mock_type_text
    ):
        """Full flow: press → record → release → transcribe → type."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False  # Skip delay
        mock_app._enhancer.is_active = False
        _text = f"[mock from {_FILE}::test_full_flow_no_enhance]"
        mock_app._transcriber.transcribe.return_value = _text
        mock_app._recorder.stop.return_value = b"fake_wav"

        async def _test():
            await flow._handle_press("fn")
            # Give it time to reach the wait_action
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        mock_app._recorder.start.assert_called_once()
        mock_app._recorder.stop.assert_called_once()
        mock_app._transcriber.transcribe.assert_called_once()
        mock_type_text.assert_called_once_with(
            _text, append_newline=False, method="type"
        )

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_empty_transcription(self, mock_ah, _mock_ic, flow, mock_app):
        """Empty transcription should show empty status."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.transcribe.return_value = ""

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        mock_app._set_status.assert_any_call("statusbar.status.empty")


class TestReleaseHidesIndicatorImmediately:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_release_hides_indicator_before_audio_shutdown(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """The orb must vanish at the moment of release — not after the
        blocking recorder.stop() (which takes ~0.3s of engine teardown)."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        _text = f"[mock from {_FILE}::test_release_hides_indicator]"
        mock_app._transcriber.transcribe.return_value = _text
        order: list[str] = []
        mock_app._recording_indicator.hide.side_effect = (
            lambda: order.append("hide")
        )

        def _stop():
            order.append("stop")
            mock_app._recorder.is_recording = False
            return b"fake_wav_data"

        mock_app._recorder.stop.side_effect = _stop

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        assert "hide" in order and "stop" in order
        assert order.index("hide") < order.index("stop")

    @patch("wenzi.scripting.api.alert.alert")
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_empty_recording_still_hides_indicator_at_release(
        self, mock_ah, _mock_ic, _mock_alert, flow, mock_app
    ):
        """The empty-recording path must not keep the orb up while the
        engine tears down."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        order: list[str] = []
        mock_app._recording_indicator.hide.side_effect = (
            lambda: order.append("hide")
        )

        def _stop_empty():
            order.append("stop")
            mock_app._recorder.is_recording = False
            return None

        mock_app._recorder.stop.side_effect = _stop_empty

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        assert order.index("hide") < order.index("stop")
        assert not flow.is_busy


class TestCancel:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_cancel_during_recording(self, mock_ah, _mock_ic, flow, mock_app):
        """Cancel during recording should stop and not transcribe."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        # is_recording starts False (no orphan), becomes True after start()
        mock_app._recorder.is_recording = False

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.CANCEL)
            await flow._current_task

        run(_test())

        mock_app._recorder.stop.assert_called_once()
        mock_app._transcriber.transcribe.assert_not_called()


class TestRestart:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_release_right_after_restart_not_lost(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """A RELEASE queued immediately behind a RESTART belongs to the
        restarted session and must stop it — not be drained away."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"] = {"max_recording_seconds": 60}
        _text = f"[mock from {_FILE}::test_release_right_after_restart_not_lost]"
        mock_app._transcriber.transcribe.return_value = _text

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            # Restart and release enqueued back-to-back
            flow._actions.put_nowait(Action.RESTART)
            flow._actions.put_nowait(Action.RELEASE)
            for _ in range(200):
                if not flow.is_busy:
                    return
                await asyncio.sleep(0.05)
            raise AssertionError("restarted session did not stop on RELEASE")

        run(_test())

        assert mock_app._recorder.start.call_count == 2
        assert mock_app._recorder.stop.call_count == 2
        mock_app._transcriber.transcribe.assert_called_once()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_restart_creates_new_session(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """Restart should stop current recording and start a new session."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.transcribe.return_value = f"[mock from {_FILE}::test_restart_creates_new_session]"

        call_count = [0]

        def counting_start(*a, **kw):
            call_count[0] += 1
            mock_app._recorder.is_recording = True
            return "mic"

        mock_app._recorder.start.side_effect = counting_start

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            # First: restart
            flow._actions.put_nowait(Action.RESTART)
            await asyncio.sleep(0.1)
            # Then: release the restarted session
            flow._actions.put_nowait(Action.RELEASE)
            # Wait for the restarted session to complete
            for _ in range(100):
                if not flow.is_busy:
                    break
                await asyncio.sleep(0.05)

        run(_test())

        # Recorder should have been started twice (original + restart)
        assert call_count[0] == 2
        # And stopped twice
        assert mock_app._recorder.stop.call_count == 2


class TestPreviewHistory:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_preview_history_action(self, mock_ah, _mock_ic, flow, mock_app):
        """Preview history action should stop recording and show preview."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token
        order: list[str] = []

        mock_app._system_output_ducker.end.side_effect = (
            lambda _token: order.append("restore")
        )

        def _end_op(_token):
            order.append("end_op")
            mock_app._busy = False

        def _show_history():
            assert not mock_app._busy
            order.append("show_history")

        mock_app._end_op.side_effect = _end_op
        mock_app._preview_controller.on_show_last_preview.side_effect = (
            _show_history
        )

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.PREVIEW_HISTORY)
            await flow._current_task

        run(_test())

        mock_app._recorder.stop.assert_called_once()
        mock_app._preview_controller.on_show_last_preview.assert_called_once()
        assert order == ["restore", "end_op", "show_history"]


class TestWatchdogTimeout:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_timeout_auto_stops(self, mock_ah, _mock_ic, flow, mock_app):
        """Recording should auto-stop on timeout (acts like release)."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"] = {"max_recording_seconds": 0.1}
        mock_app._transcriber.transcribe.return_value = f"[mock from {_FILE}::test_timeout_auto_stops]"

        async def _test():
            await flow._handle_press("fn")
            # Don't send any action — let it timeout
            for _ in range(100):
                if not flow.is_busy:
                    break
                await asyncio.sleep(0.05)

        run(_test())

        mock_app._recorder.stop.assert_called_once()
        mock_app._transcriber.transcribe.assert_called_once()


class TestModeNav:
    def test_build_mode_list(self, flow, mock_app):
        mock_app._enhancer.available_modes = [
            ("proofread", "Proofread"),
            ("translate_en", "Translate EN"),
        ]
        modes = flow._build_mode_list()
        assert modes[0] == ("off", "Off")
        assert modes[1] == ("proofread", "Proofread")

    @patch("PyObjCTools.AppHelper")
    def test_mode_nav_inline(self, mock_ah, flow, mock_app):
        """Mode navigation should be handled inline without interrupting wait."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._enhancer.available_modes = [
            ("proofread", "Proofread"),
            ("translate_en", "Translate EN"),
        ]
        mock_app._enhance_mode = "proofread"

        async def _test():
            # Test inline handling
            action = Action.MODE_NEXT
            flow._handle_inline_action(action)

        run(_test())

        assert mock_app._enhance_mode == "translate_en"


class TestFeedbackToggles:
    @patch("wenzi.controllers.recording_flow.save_config")
    def test_sound_toggle(self, mock_save, flow, mock_app):
        sender = MagicMock()
        flow.on_sound_feedback_toggle(sender)
        assert mock_app._sound_manager.enabled is False
        assert sender.state == 0
        mock_save.assert_called_once()

    @patch("wenzi.controllers.recording_flow.save_config")
    def test_visual_toggle(self, mock_save, flow, mock_app):
        sender = MagicMock()
        flow.on_visual_indicator_toggle(sender)
        assert mock_app._recording_indicator.enabled is False
        assert sender.state == 0
        mock_save.assert_called_once()


class TestStreaming:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_streaming_starts_when_supported(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """Streaming transcription should start if transcriber supports it."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.supports_streaming = True
        mock_app._transcriber.stop_streaming.return_value = f"[mock from {_FILE}::test_streaming_starts_when_supported]"

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        mock_app._transcriber.start_streaming.assert_called_once()
        mock_app._recorder.set_on_audio_chunk.assert_called_once()

    @patch("wenzi.scripting.api.alert.alert")
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_empty_wav_cancels_streaming(
        self, mock_ah, _mock_ic, _mock_alert, flow, mock_app
    ):
        """An empty recording on the streaming path must still cancel the
        streaming session — Apple/Sherpa otherwise keep a background
        recognizer session alive."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.supports_streaming = True

        def _stop_empty():
            mock_app._recorder.is_recording = False
            return None

        mock_app._recorder.stop.side_effect = _stop_empty

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await asyncio.wait_for(flow._current_task, timeout=5)

        run(_test())

        mock_app._transcriber.cancel_streaming.assert_called_once()
        mock_app._transcriber.stop_streaming.assert_not_called()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_start_streaming_failure_best_effort_cancels(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """start_streaming raising AFTER allocating backend resources must
        still trigger a best-effort cancel — no half-started recognizer
        session may leak."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.supports_streaming = True
        mock_app._transcriber.start_streaming.side_effect = RuntimeError(
            "thread failed after allocation"
        )
        _text = f"[mock from {_FILE}::test_start_streaming_failure]"
        mock_app._transcriber.transcribe.return_value = _text

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await asyncio.wait_for(flow._current_task, timeout=5)

        run(_test())

        mock_app._transcriber.cancel_streaming.assert_called_once()
        mock_app._transcriber.transcribe.assert_called_once()  # batch fallback

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_streaming_fallback_on_error(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """If streaming init fails, should fall back to batch mode."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.supports_streaming = True
        mock_app._transcriber.start_streaming.side_effect = RuntimeError("fail")

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        # Should fall back to batch transcription
        mock_app._transcriber.transcribe.assert_called_once()


class TestStartTimeout:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_start_timeout_resets_to_idle(
        self, mock_ah, _mock_ic, flow, mock_app, monkeypatch
    ):
        """A timed-out native start keeps output owned until teardown."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._config["audio"]["duck_system_audio"] = True
        token = object()
        mock_app._system_output_ducker.begin.return_value = token
        mock_app._system_output_ducker.end.return_value = True
        monkeypatch.setattr(RecordingFlow, "_START_TIMEOUT", 0.05)
        started = threading.Event()
        allow_start = threading.Event()
        order: list[str] = []

        def hanging_start(*a, **kw):
            started.set()
            assert allow_start.wait(timeout=5)
            order.append("late-start-commit")
            mock_app._recorder.is_recording = True
            return "AirPods Max"

        def _stop():
            order.append("recorder-stop")
            mock_app._recorder.is_recording = False
            return None

        def _restore(_token):
            order.append("output-restore")
            return True

        mock_app._recorder.start.side_effect = hanging_start
        mock_app._recorder.stop.side_effect = _stop
        mock_app._system_output_ducker.end.side_effect = _restore

        async def _test():
            await flow._handle_press("fn")
            await _wait_thread_event(started)
            for _ in range(100):
                if mock_app._recorder.mark_tainted.called:
                    break
                await asyncio.sleep(0.01)
            assert mock_app._recorder.mark_tainted.called
            assert flow.is_busy
            mock_app._system_output_ducker.end.assert_not_called()
            mock_app._end_op.assert_not_called()
            mock_app._recording_indicator.hide.assert_called()

            allow_start.set()
            await asyncio.wait_for(flow._current_task, timeout=2)

        try:
            run(_test())
        finally:
            allow_start.set()

        mock_app._recorder.mark_tainted.assert_called_once_with(
            stop_async=False
        )
        assert order == [
            "late-start-commit",
            "recorder-stop",
            "output-restore",
        ]
        mock_app._recorder.stop.assert_called_once()
        mock_app._system_output_ducker.end.assert_called_once_with(token)
        assert not flow.is_busy


class TestSessionExceptionCleanup:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_exception_after_start_stops_recorder(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """A crash after recording started must stop the recorder — the
        microphone must never stay open behind a reset UI."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._recording_indicator.set_recording_active.side_effect = (
            RuntimeError("boom")
        )

        flow.on_press("fn")
        run(TestQuickRelease._wait_session_done(flow))

        mock_app._recorder.stop.assert_called_once()
        assert mock_app._recorder.is_recording is False
        assert not flow.is_busy

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_cancel_stops_recorder_before_streaming(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """On cancel the recorder must stop BEFORE streaming finalizes —
        the reverse order can feed tail audio into a recognizer that has
        already finished."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.supports_streaming = True
        order: list[str] = []

        def _stop_recorder():
            order.append("recorder.stop")
            mock_app._recorder.is_recording = False
            return b"fake_wav_data"

        mock_app._recorder.stop.side_effect = _stop_recorder
        mock_app._transcriber.cancel_streaming.side_effect = (
            lambda: order.append("cancel_streaming")
        )

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.CANCEL)
            await asyncio.wait_for(flow._current_task, timeout=5)

        run(_test())

        # Abort paths cancel (not stop) streaming — stop would block for
        # a final result nobody consumes
        assert order == ["recorder.stop", "cancel_streaming"]
        mock_app._transcriber.stop_streaming.assert_not_called()

    def test_shutdown_finalizes_streaming_when_recorder_stop_raises(
        self, flow, mock_app
    ):
        """recorder.stop() raising must not skip streaming finalization."""
        mock_app._recorder.stop.side_effect = RuntimeError("encode failed")
        token = object()

        async def _test():
            with pytest.raises(RuntimeError):
                await asyncio.shield(
                    flow._ensure_audio_shutdown(
                        True,
                        cancel=True,
                        duck_token=token,
                    )
                )
            assert await flow._settle_output_restore()

        run(_test())
        mock_app._system_output_ducker.end.assert_called_once_with(token)
        mock_app._transcriber.cancel_streaming.assert_called_once()

    def test_restore_waits_for_mic_and_overlaps_streaming_finalize(
        self, flow, mock_app
    ):
        """Mic stop is a hard barrier, then restore and recognizer cleanup
        must be able to progress independently."""
        token = object()
        stop_entered = threading.Event()
        allow_stop = threading.Event()
        restore_entered = threading.Event()
        allow_restore = threading.Event()
        finalize_entered = threading.Event()
        allow_finalize = threading.Event()

        def _stop():
            stop_entered.set()
            assert allow_stop.wait(timeout=5)
            mock_app._recorder.is_recording = False
            return b"wav"

        def _restore(_token):
            restore_entered.set()
            assert allow_restore.wait(timeout=5)

        def _finalize():
            finalize_entered.set()
            assert allow_finalize.wait(timeout=5)
            return "final"

        mock_app._recorder.stop.side_effect = _stop
        mock_app._system_output_ducker.end.side_effect = _restore
        mock_app._transcriber.stop_streaming.side_effect = _finalize

        async def _test():
            try:
                shutdown = flow._ensure_audio_shutdown(
                    True,
                    cancel=False,
                    duck_token=token,
                )
                await _wait_thread_event(stop_entered)
                assert not restore_entered.is_set()
                assert not finalize_entered.is_set()

                allow_stop.set()
                await _wait_thread_event(restore_entered)
                await _wait_thread_event(finalize_entered)

                allow_finalize.set()
                wav, text = await asyncio.wait_for(
                    asyncio.shield(shutdown),
                    timeout=5,
                )
                assert (wav, text) == (b"wav", "final")
                assert not flow._output_restore_task.done()

                allow_restore.set()
                assert await flow._settle_output_restore()
            finally:
                allow_stop.set()
                allow_finalize.set()
                allow_restore.set()

        run(_test())

        mock_app._system_output_ducker.end.assert_called_once_with(token)
        mock_app._transcriber.stop_streaming.assert_called_once_with()

    def test_audio_shutdown_single_flight_and_shielded(self, flow, mock_app):
        """All callers share ONE shutdown task; cancelling an awaiter must
        not cancel the underlying work nor allow a second cleanup."""
        token = object()
        block = threading.Event()
        restore_entered = threading.Event()
        allow_restore = threading.Event()
        calls = {"stop": 0, "concurrent": 0, "max_concurrent": 0}
        lock = threading.Lock()

        def _blocking_stop():
            with lock:
                calls["stop"] += 1
                calls["concurrent"] += 1
                calls["max_concurrent"] = max(
                    calls["max_concurrent"], calls["concurrent"]
                )
            block.wait(5)
            with lock:
                calls["concurrent"] -= 1
            mock_app._recorder.is_recording = False
            return b"wav"

        mock_app._recorder.stop.side_effect = _blocking_stop
        mock_app._system_output_ducker.end.side_effect = lambda _token: (
            restore_entered.set(),
            allow_restore.wait(5),
        )

        async def _test():
            t1 = flow._ensure_audio_shutdown(
                False,
                cancel=True,
                duck_token=token,
            )
            t2 = flow._ensure_audio_shutdown(False, cancel=False)
            assert t1 is t2  # single flight

            try:
                waiter = asyncio.ensure_future(asyncio.shield(t1))
                await asyncio.sleep(0.05)
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter

                block.set()
                wav, text = await asyncio.shield(t1)
                assert wav == b"wav"
                assert text is None
                await _wait_thread_event(restore_entered)
                assert not flow._output_restore_task.cancelled()

                allow_restore.set()
                assert await flow._settle_output_restore()
            finally:
                block.set()
                allow_restore.set()

        run(_test())
        assert calls["stop"] == 1
        assert calls["max_concurrent"] == 1
        mock_app._system_output_ducker.end.assert_called_once_with(token)

    def test_stop_streaming_failure_falls_back_to_cancel(self, flow, mock_app):
        """stop_streaming failing must fall back to cancel exactly once."""
        mock_app._transcriber.stop_streaming.side_effect = RuntimeError("x")

        async def _test():
            wav, text = await asyncio.shield(
                flow._ensure_audio_shutdown(True, cancel=False)
            )
            assert text is None  # fallback cancelled → no final text

        run(_test())
        mock_app._transcriber.stop_streaming.assert_called_once()
        mock_app._transcriber.cancel_streaming.assert_called_once()

    def test_cancel_streaming_failure_falls_back_to_stop(self, flow, mock_app):
        """cancel_streaming failing must fall back to stop exactly once."""
        mock_app._transcriber.cancel_streaming.side_effect = RuntimeError("x")

        async def _test():
            wav, text = await asyncio.shield(
                flow._ensure_audio_shutdown(True, cancel=True)
            )
            assert text is None

        run(_test())
        mock_app._transcriber.cancel_streaming.assert_called_once()
        mock_app._transcriber.stop_streaming.assert_called_once()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_both_streaming_cleanups_failing_still_resets_ui(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """When stop AND cancel both fail, the error is contained, the UI
        resets, and no second cleanup is ever started."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.supports_streaming = True
        mock_app._transcriber.stop_streaming.side_effect = RuntimeError("s")
        mock_app._transcriber.cancel_streaming.side_effect = RuntimeError("c")

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.CANCEL)
            await asyncio.wait_for(flow._current_task, timeout=5)

        run(_test())

        assert not flow.is_busy
        mock_app._recorder.stop.assert_called_once()
        mock_app._transcriber.cancel_streaming.assert_called_once()
        mock_app._transcriber.stop_streaming.assert_called_once()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_streaming_attach_failure_cancels_started_stream(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """start_streaming succeeding but the chunk attach failing must
        cancel the already-started stream — no background session leaks —
        and fall back to batch transcription."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.supports_streaming = True
        mock_app._recorder.set_on_audio_chunk.side_effect = RuntimeError("attach")
        _text = f"[mock from {_FILE}::test_streaming_attach_failure]"
        mock_app._transcriber.transcribe.return_value = _text

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await asyncio.wait_for(flow._current_task, timeout=5)

        run(_test())

        mock_app._transcriber.cancel_streaming.assert_called_once()
        mock_app._transcriber.transcribe.assert_called_once()

    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_exception_with_mic_closed_still_cancels_streaming(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """Even when the recorder already closed the mic (stop() raised
        after flipping state), an active streaming session must still be
        finalized — gating cleanup on is_recording alone missed this."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._transcriber.supports_streaming = True

        def _stop_raises():
            mock_app._recorder.is_recording = False
            raise RuntimeError("WAV encode failed")

        mock_app._recorder.stop.side_effect = _stop_raises

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await asyncio.wait_for(flow._current_task, timeout=5)

        run(_test())

        mock_app._transcriber.cancel_streaming.assert_called()
        assert not flow.is_busy


class TestStartFailure:
    @patch("wenzi.scripting.api.alert.alert")
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_start_failure_aborts_session_and_drains_release(
        self, mock_ah, _mock_ic, mock_alert, flow, mock_app
    ):
        """start() failing without raising (engine failure) must abort the
        session — no live recording UI — and its queued RELEASE must not
        leak into the next session."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        # start() "fails": returns None without flipping is_recording
        mock_app._recorder.start.side_effect = lambda *a, **kw: None

        flow.on_press("fn")
        flow.send_action(Action.RELEASE)  # released during startup
        run(TestQuickRelease._wait_session_done(flow))

        mock_alert.assert_called_once()
        mock_app._transcriber.transcribe.assert_not_called()
        mock_app._recorder.stop.assert_not_called()
        assert flow._actions.empty()  # leftover RELEASE drained
        assert not flow.is_busy


class TestStreamingTimeoutFallback:
    def test_single_stream_timeout_replaces_partial_output(
        self, flow, mock_app
    ):
        """Timeout fallback must replace partial streamed output."""

        async def _gen():
            yield "partial", None, False
            yield "original text", None, "timeout"

        mock_app._enhancer.enhance_stream.return_value = _gen()
        flow._show_error_alert = MagicMock()

        result, fell_back = run(
            flow._run_direct_single_stream("original text", asyncio.Event())
        )

        assert fell_back is True
        assert result == "original text"
        mock_app._streaming_overlay.clear_text.assert_called_once()
        assert mock_app._streaming_overlay.append_text.call_args_list[-1].args == (
            "original text",
        )
        assert (
            mock_app._streaming_overlay.append_text.call_args_list[-1].kwargs
            == {"completion_tokens": len("original text")}
        )
        mock_app._streaming_overlay.set_status.assert_called_with(
            "\u26a0\ufe0f AI enhancement failed, using original text"
        )
        flow._show_error_alert.assert_called_once_with(
            "AI enhancement failed, original text used"
        )

    def test_chain_stream_timeout_aborts_remaining_steps(
        self, flow, mock_app
    ):
        """A timed-out step must abort the chain — later steps would burn
        tokens on stale input — and fall back to the original text."""
        step1_def = MagicMock()
        step1_def.label = "Proofread"
        step2_def = MagicMock()
        step2_def.label = "Translate"
        mock_app._enhancer.get_mode_definition.side_effect = (
            lambda mode_id: {
                "proofread": step1_def,
                "translate": step2_def,
            }.get(mode_id)
        )

        inputs: list[str] = []

        def _make_stream(text: str, input_context=None):
            inputs.append(text)

            async def _gen():
                yield "partial", None, False
                yield "original text", None, "timeout"

            return _gen()

        mock_app._enhancer.enhance_stream.side_effect = _make_stream
        flow._show_error_alert = MagicMock()

        result, fell_back = run(
            flow._run_direct_chain_stream(
                "original text",
                ["proofread", "translate"],
                asyncio.Event(),
            )
        )

        assert fell_back is True
        assert result == "original text"
        assert inputs == ["original text"]  # step 2 never ran
        # The overlay must display exactly what will be typed — the
        # fallback replaces the partial output with the original text
        assert mock_app._streaming_overlay.append_text.call_args_list[-1].args == (
            "original text",
        )
        flow._show_error_alert.assert_called_once_with(
            "AI enhancement failed, original text used"
        )


class TestOrphanedRecording:
    @patch("wenzi.controllers.recording_flow.capture_input_context", return_value=None)
    @patch("PyObjCTools.AppHelper")
    def test_orphaned_recording_cleaned_up(
        self, mock_ah, _mock_ic, flow, mock_app
    ):
        """An orphaned active recording should be stopped before starting."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        mock_app._recorder.is_recording = True
        mock_app._recorder.start.return_value = "TestMic"
        mock_app._transcriber.transcribe.return_value = (
            f"[mock from {_FILE}::test_orphaned_recording_cleaned_up]"
        )

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.RELEASE)
            await flow._current_task

        run(_test())

        # stop() should be called twice: once for orphan cleanup,
        # once for the normal release
        assert mock_app._recorder.stop.call_count == 2


class TestConfigDegraded:
    @patch("PyObjCTools.AppHelper")
    def test_config_degraded_shows_alert(self, mock_ah, flow, mock_app):
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._config_degraded = True

        run(flow._handle_press("fn"))

        mock_app._show_config_error_alert.assert_called_once()
        assert not flow.is_busy
