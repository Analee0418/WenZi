"""Tests for RecordingFlow — coroutine-based recording controller."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import wenzi.async_loop as async_loop
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
    app._config = {
        "feedback": {"sound_enabled": True, "visual_indicator": True},
    }
    app._config_path = str(tmp_path / "config.json")
    app._sound_manager = MagicMock()
    app._sound_manager.enabled = True
    app._recording_indicator = MagicMock()
    app._recording_indicator.enabled = True
    app._recording_indicator.show_device_name = False
    app._recording_indicator.current_frame = MagicMock()
    app._recorder = MagicMock()
    app._recorder.is_recording = False
    app._recorder.current_level = 0.5
    app._recorder.last_device_name = "MacBook Pro Microphone"

    def _recorder_start():
        # Mirror the real contract: a successful start() flips is_recording
        app._recorder.is_recording = True
        return "MacBook Pro Microphone"

    def _recorder_stop():
        app._recorder.is_recording = False
        return b"fake_wav_data"

    app._recorder.start.side_effect = _recorder_start
    app._recorder.stop.side_effect = _recorder_stop
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

        # Recorder should never have been started
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

        def _start_side_effect():
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

        def counting_start():
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

        async def _test():
            await flow._handle_press("fn")
            await asyncio.sleep(0.05)
            flow._actions.put_nowait(Action.PREVIEW_HISTORY)
            await flow._current_task

        run(_test())

        mock_app._recorder.stop.assert_called_once()
        mock_app._preview_controller.on_show_last_preview.assert_called_once()


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
        """When recorder.start() times out, session should reset to idle."""
        mock_ah.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        mock_app._sound_manager.enabled = False
        monkeypatch.setattr(RecordingFlow, "_START_TIMEOUT", 0.1)

        def hanging_start():
            # Block until cancelled — simulates a hung PortAudio call
            import time
            time.sleep(5)

        mock_app._recorder.start.side_effect = hanging_start

        async def _test():
            await flow._handle_press("fn")
            # Wait for session to finish (via timeout)
            for _ in range(100):
                if not flow.is_busy:
                    break
                await asyncio.sleep(0.05)

        run(_test())

        mock_app._recorder.mark_tainted.assert_called_once()
        mock_app._recording_indicator.hide.assert_called()
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

        async def _test():
            with pytest.raises(RuntimeError):
                await asyncio.shield(
                    flow._ensure_audio_shutdown(True, cancel=True)
                )

        run(_test())
        mock_app._transcriber.cancel_streaming.assert_called_once()

    def test_audio_shutdown_single_flight_and_shielded(self, flow, mock_app):
        """All callers share ONE shutdown task; cancelling an awaiter must
        not cancel the underlying work nor allow a second cleanup."""
        block = threading.Event()
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

        async def _test():
            t1 = flow._ensure_audio_shutdown(False, cancel=True)
            t2 = flow._ensure_audio_shutdown(False, cancel=False)
            assert t1 is t2  # single flight

            waiter = asyncio.ensure_future(asyncio.shield(t1))
            await asyncio.sleep(0.05)
            waiter.cancel()
            try:
                await waiter
            except asyncio.CancelledError:
                pass

            block.set()
            wav, text = await asyncio.shield(t1)
            assert wav == b"wav"
            assert text is None

        run(_test())
        assert calls["stop"] == 1
        assert calls["max_concurrent"] == 1

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
        mock_app._recorder.start.side_effect = lambda: None

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
