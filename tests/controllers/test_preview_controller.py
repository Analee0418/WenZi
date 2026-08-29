"""Tests for PreviewController enhance mode debounce and history caching."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from wenzi.controllers.preview_controller import PreviewController
from wenzi.enhance.preview_history import PreviewRecord

_DEBOUNCE_POLL_INTERVAL = 0.005
_DEBOUNCE_POLL_TIMEOUT = 0.5


def _wait_for_debounce(ctrl, timeout=_DEBOUNCE_POLL_TIMEOUT):
    """Wait for the debounce timer to fire (timer becomes None)."""
    deadline = time.monotonic() + timeout
    while ctrl._enhance_debounce_timer is not None and time.monotonic() < deadline:
        time.sleep(_DEBOUNCE_POLL_INTERVAL)


@pytest.fixture
def mock_app():
    """Create a mock WenZiApp for PreviewController."""
    app = MagicMock()
    app._enhance_mode = "proofread"
    app._enhance_controller = MagicMock()
    app._enhance_controller.get_cached.return_value = None
    app._enhancer = MagicMock()
    app._enhancer._enabled = True
    app._enhancer.mode = "proofread"
    app._enhance_menu_items = {"off": MagicMock(), "proofread": MagicMock(), "translate": MagicMock()}
    app._preview_panel = MagicMock()
    app._preview_panel.enhance_request_id = 0
    app._current_preview_asr_text = "hello"
    app._config = {"ai_enhance": {"enabled": True, "mode": "proofread"}}
    app._config_path = "/tmp/test.json"
    return app


@pytest.fixture
def ctrl(mock_app):
    return PreviewController(mock_app)


class TestPreviewSttLateCommit:
    @patch("wenzi.controllers.preview_controller.save_config")
    def test_stale_switch_result_discarded(self, _mock_save, ctrl, mock_app):
        """A switch finishing after the panel closed must not replace the
        app-wide transcriber (a new recording or switch may be using it)."""
        mock_app._preview_stt_keys = [
            ("preset", "funasr-zh"),
            ("preset", "mlx-whisper-large-v3-turbo"),
        ]
        mock_app._current_preset_id = "funasr-zh"
        mock_app._current_remote_asr = None
        mock_app._config["asr"] = {}
        mock_app._preview_panel.hotwords_detail = None
        mock_app._preview_panel.asr_request_id = 7
        mock_app._preview_panel.is_visible = False  # panel already closed
        mock_app._preview_audio_duration = 1.5
        old_transcriber = mock_app._transcriber
        new_transcriber = MagicMock()
        mock_app._create_transcriber_for_preset.return_value = new_transcriber

        mock_helper = MagicMock()
        mock_helper.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        with patch.dict("sys.modules", {
            "PyObjCTools": MagicMock(AppHelper=mock_helper),
            "PyObjCTools.AppHelper": mock_helper,
        }):
            ctrl.on_preview_stt_change(1)
            deadline = time.monotonic() + 5
            while (
                not new_transcriber.cleanup.called
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)

        assert mock_app._transcriber is old_transcriber
        old_transcriber.cleanup.assert_not_called()
        new_transcriber.cleanup.assert_called_once()
        assert ctrl._stt_switching is False

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_superseded_switch_result_discarded(self, _mock_save, ctrl, mock_app):
        """A switch whose request id was superseded (new preview session,
        monotonic ids) must be discarded even if a panel is visible."""
        mock_app._preview_stt_keys = [
            ("preset", "funasr-zh"),
            ("preset", "mlx-whisper-large-v3-turbo"),
        ]
        mock_app._current_preset_id = "funasr-zh"
        mock_app._current_remote_asr = None
        mock_app._config["asr"] = {}
        mock_app._preview_panel.hotwords_detail = None
        mock_app._preview_panel.asr_request_id = 7
        mock_app._preview_panel.is_visible = True
        mock_app._preview_audio_duration = 1.5
        old_transcriber = mock_app._transcriber
        new_transcriber = MagicMock()
        release = threading.Event()
        new_transcriber.transcribe.side_effect = (
            lambda *a, **k: release.wait(5) or "new text"
        )
        mock_app._create_transcriber_for_preset.return_value = new_transcriber

        mock_helper = MagicMock()
        mock_helper.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        try:
            with patch.dict("sys.modules", {
                "PyObjCTools": MagicMock(AppHelper=mock_helper),
                "PyObjCTools.AppHelper": mock_helper,
            }):
                ctrl.on_preview_stt_change(1)
                # A newer preview session bumps the monotonic id while
                # the switch worker is still transcribing
                mock_app._preview_panel.asr_request_id = 8
                release.set()
                deadline = time.monotonic() + 5
                while (
                    not new_transcriber.cleanup.called
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
        finally:
            release.set()

        assert mock_app._transcriber is old_transcriber
        old_transcriber.cleanup.assert_not_called()
        new_transcriber.cleanup.assert_called_once()


def _tiny_wav() -> bytes:
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<8h", *([100] * 8)))
    return buf.getvalue()


class TestPreviewSttSessionLifecycle:
    def test_cancelled_before_start_skips_transcribe(self, ctrl, mock_app):
        """A session cancelled before the worker starts must not even
        call transcribe."""
        from wenzi.controllers.preview_controller import _PreviewSttSession

        mock_app._preview_panel.asr_request_id = 5
        session = _PreviewSttSession(5, MagicMock())
        session.cancel_event.set()

        ctrl._run_preview_stt(session, _tiny_wav(), [], False, None)

        session.transcriber.transcribe.assert_not_called()
        assert session.done_event.is_set()

    def test_cancel_mid_transcribe_discards_result_and_event(
        self, ctrl, mock_app
    ):
        """Cancelling (panel close) while transcribe blocks must discard
        the result AND the scripting event."""
        import threading as _threading

        from wenzi.controllers.preview_controller import _PreviewSttSession

        mock_app._preview_panel.asr_request_id = 5
        session = _PreviewSttSession(5, MagicMock())
        gate = _threading.Event()
        session.transcriber.transcribe.side_effect = (
            lambda *a, **k: gate.wait(5) or "late text"
        )
        mock_app._build_dynamic_hotwords = MagicMock(return_value=([], []))
        ctrl._fire_scripting_event = MagicMock()

        worker = _threading.Thread(
            target=ctrl._run_preview_stt,
            args=(session, _tiny_wav(), [], False, None),
            daemon=True,
        )
        worker.start()
        time.sleep(0.05)
        session.cancel_event.set()  # panel closed
        gate.set()
        worker.join(5)

        assert session.done_event.is_set()
        ctrl._fire_scripting_event.assert_not_called()
        mock_app._preview_panel.set_asr_result.assert_not_called()

    def test_new_panel_generation_discards_old_result(self, ctrl, mock_app):
        """A result finishing after the panel was reopened (generation
        advanced) must be discarded."""
        import threading as _threading

        from wenzi.controllers.preview_controller import _PreviewSttSession

        mock_app._preview_panel.asr_request_id = 5
        session = _PreviewSttSession(5, MagicMock())
        gate = _threading.Event()
        session.transcriber.transcribe.side_effect = (
            lambda *a, **k: gate.wait(5) or "old result"
        )
        mock_app._build_dynamic_hotwords = MagicMock(return_value=([], []))
        ctrl._fire_scripting_event = MagicMock()

        worker = _threading.Thread(
            target=ctrl._run_preview_stt,
            args=(session, _tiny_wav(), [], False, None),
            daemon=True,
        )
        worker.start()
        time.sleep(0.05)
        mock_app._preview_panel.asr_request_id = 6  # panel reopened
        gate.set()
        worker.join(5)

        ctrl._fire_scripting_event.assert_not_called()
        mock_app._preview_panel.set_asr_result.assert_not_called()

    def test_finish_never_releases_before_worker_done(self, ctrl, mock_app):
        """_finish_stt_session must NEVER time out into a release: while
        the worker is not done, the session stays set and the caller's
        OpGuard stays busy — across multiple wait intervals."""
        import threading as _threading

        from wenzi.controllers.preview_controller import _PreviewSttSession

        ctrl._STT_FINISH_WAIT_INTERVAL = 0.05  # several intervals elapse
        session = _PreviewSttSession(5, MagicMock())
        ctrl._stt_session = session
        mock_app._busy = True  # the caller holds the op slot
        released = _threading.Event()

        def _close_and_release():
            # Mirrors the production ordering: finish first, only then
            # release the exclusive-op slot.
            ctrl._finish_stt_session()
            mock_app._busy = False
            released.set()

        waiter = _threading.Thread(target=_close_and_release, daemon=True)
        waiter.start()
        time.sleep(0.3)  # ≥ 5 wait intervals — the old code would give up
        assert not released.is_set()          # still blocked on the worker
        assert mock_app._busy is True         # OpGuard never released early
        assert ctrl._stt_session is session   # session NOT cleared
        assert session.cancel_event.is_set()  # close cancels the worker

        session.done_event.set()              # worker actually exits
        waiter.join(5)
        assert released.is_set()
        assert mock_app._busy is False        # released only after done
        assert ctrl._stt_session is None


class TestPreviewSttCommitTransaction:
    def _setup_switch(self, ctrl, mock_app):
        mock_app._preview_stt_keys = [
            ("preset", "funasr-zh"),
            ("preset", "mlx-whisper-large-v3-turbo"),
        ]
        mock_app._current_preset_id = "funasr-zh"
        mock_app._current_remote_asr = None
        mock_app._config["asr"] = {}
        mock_app._preview_panel.hotwords_detail = None
        mock_app._preview_panel.asr_request_id = 7
        mock_app._preview_panel.is_visible = True
        mock_app._preview_audio_duration = 1.5
        new_transcriber = MagicMock()
        new_transcriber.transcribe.return_value = "new text"
        mock_app._create_transcriber_for_preset.return_value = new_transcriber
        return new_transcriber

    def _run_switch(self, ctrl, mock_app, new_transcriber):
        mock_helper = MagicMock()
        mock_helper.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        with patch.dict("sys.modules", {
            "PyObjCTools": MagicMock(AppHelper=mock_helper),
            "PyObjCTools.AppHelper": mock_helper,
        }):
            ctrl.on_preview_stt_change(1)
            deadline = time.monotonic() + 5
            while ctrl._stt_switching and time.monotonic() < deadline:
                time.sleep(0.01)

    @patch(
        "wenzi.controllers.preview_controller.save_config",
        side_effect=OSError("disk full"),
    )
    def test_save_config_failure_keeps_live_state(
        self, _mock_save, ctrl, mock_app
    ):
        """Persistence failing must not leave panel/live state on the old
        model, and the loading state must resolve."""
        new_transcriber = self._setup_switch(ctrl, mock_app)
        old_transcriber = mock_app._transcriber

        self._run_switch(ctrl, mock_app, new_transcriber)

        assert mock_app._transcriber is new_transcriber
        old_transcriber.cleanup.assert_called_once()
        assert mock_app._current_preview_asr_text == "new text"
        mock_app._enhance_controller.clear_cache.assert_called_once()
        calls = [
            c for c in mock_app._preview_panel.set_asr_result.call_args_list
            if c.args and c.args[0] == "new text"
        ]
        assert calls  # panel resolved with the NEW transcription

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_menu_update_failure_keeps_live_state(
        self, _mock_save, ctrl, mock_app
    ):
        new_transcriber = self._setup_switch(ctrl, mock_app)
        mock_app._menu_builder.update_model_checkmarks.side_effect = (
            RuntimeError("menu")
        )

        self._run_switch(ctrl, mock_app, new_transcriber)

        assert mock_app._transcriber is new_transcriber
        assert mock_app._current_preview_asr_text == "new text"
        mock_app._enhance_controller.clear_cache.assert_called_once()
        assert mock_app._current_preset_id == "mlx-whisper-large-v3-turbo"


class TestEnhanceModeDebounce:
    """Tests for debounced enhancement on mode switch."""

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_single_switch_fires_after_debounce(self, mock_save, ctrl, mock_app):
        """A single mode switch should fire enhancement after debounce delay."""
        # Use a very short debounce for testing
        ctrl._ENHANCE_DEBOUNCE_SECONDS = 0.05

        ctrl.on_preview_mode_change("translate")

        # Enhancement should NOT have started yet
        mock_app._enhance_controller.run.assert_not_called()
        # A debounce timer should be active
        assert ctrl._enhance_debounce_timer is not None

        _wait_for_debounce(ctrl)

        mock_app._enhance_controller.run.assert_called_once()

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_rapid_switches_fire_only_once(self, mock_save, ctrl, mock_app):
        """Rapid mode switches should only fire one enhancement request."""
        ctrl._ENHANCE_DEBOUNCE_SECONDS = 0.05

        ctrl.on_preview_mode_change("translate")
        ctrl.on_preview_mode_change("proofread")
        ctrl.on_preview_mode_change("translate")

        # Nothing fired yet
        mock_app._enhance_controller.run.assert_not_called()
        # Cancel should have been called for each switch
        assert mock_app._enhance_controller.cancel.call_count == 3

        _wait_for_debounce(ctrl)

        # Only one call should have fired (for the last mode)
        mock_app._enhance_controller.run.assert_called_once()

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_switch_to_off_cancels_immediately(self, mock_save, ctrl, mock_app):
        """Switching to Off should cancel immediately without debounce."""
        ctrl._ENHANCE_DEBOUNCE_SECONDS = 0.05

        ctrl.on_preview_mode_change("off")

        mock_app._enhance_controller.cancel.assert_called_once()
        # No debounce timer should be active
        assert ctrl._enhance_debounce_timer is None

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_switch_off_cancels_pending_debounce(self, mock_save, ctrl, mock_app):
        """Switching to Off should cancel any pending debounce timer."""
        ctrl._ENHANCE_DEBOUNCE_SECONDS = 0.05

        ctrl.on_preview_mode_change("translate")
        assert ctrl._enhance_debounce_timer is not None

        ctrl.on_preview_mode_change("off")
        assert ctrl._enhance_debounce_timer is None

        # Wait past debounce period to ensure cancelled timer doesn't fire
        time.sleep(0.1)
        mock_app._enhance_controller.run.assert_not_called()

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_cached_result_replays_immediately(self, mock_save, ctrl, mock_app):
        """If cache hit, replay immediately without debounce."""
        cached = MagicMock()
        mock_app._enhance_controller.get_cached.return_value = cached

        ctrl.on_preview_mode_change("translate")

        mock_app._preview_panel.replay_cached_result.assert_called_once()
        # No debounce timer needed
        assert ctrl._enhance_debounce_timer is None
        # No enhancement run needed
        mock_app._enhance_controller.run.assert_not_called()

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_config_saved_immediately(self, mock_save, ctrl, mock_app):
        """Config should be saved immediately, not debounced."""
        ctrl._ENHANCE_DEBOUNCE_SECONDS = 0.05

        ctrl.on_preview_mode_change("translate")

        mock_save.assert_called_once()
        assert mock_app._config["ai_enhance"]["mode"] == "translate"

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_stale_timer_does_not_fire(self, mock_save, ctrl, mock_app):
        """A stale debounce timer should not fire if request_id changed."""
        ctrl._ENHANCE_DEBOUNCE_SECONDS = 0.05

        ctrl.on_preview_mode_change("translate")
        # Simulate another component bumping the request_id
        mock_app._preview_panel.enhance_request_id = 999

        _wait_for_debounce(ctrl)

        # Timer fired but guard prevented run()
        mock_app._enhance_controller.run.assert_not_called()


def _make_record(**overrides) -> PreviewRecord:
    defaults = dict(
        timestamp="2026-01-01T00:00:00+00:00",
        created_at="2026-01-01T08:00:00",
        action="confirm",
        asr_text="hello",
        enhanced_text="Hello.",
        final_text="Hello.",
        enhance_mode="proofread",
        stt_model="funasr",
        llm_model="openai/gpt-4o",
        wav_data=b"\x00" * 100,
        audio_duration=2.5,
        source="voice",
    )
    defaults.update(overrides)
    return PreviewRecord(**defaults)


class TestSaveToPreviewHistory:
    """Tests for _save_to_preview_history including system_prompt/thinking_text."""

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_save_includes_system_prompt_and_thinking(self, _mock_save, ctrl, mock_app):
        result_holder = {
            "enhanced_text": "Hello.",
            "system_prompt": "Be helpful.",
            "thinking_text": "Hmm...",
        }
        mock_app._current_preview_asr_text = "hello"
        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "openai/gpt-4o"

        ctrl._save_to_preview_history(
            "ts1", "confirm", result_holder, None, 0.0, "voice",
        )

        rec = ctrl._preview_history.get(0)
        assert rec.system_prompt == "Be helpful."
        assert rec.thinking_text == "Hmm..."

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_save_defaults_when_keys_missing(self, _mock_save, ctrl, mock_app):
        result_holder = {"enhanced_text": "Hello."}
        mock_app._current_preview_asr_text = "hello"
        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "openai/gpt-4o"

        ctrl._save_to_preview_history(
            "ts1", "confirm", result_holder, None, 0.0, "voice",
        )

        rec = ctrl._preview_history.get(0)
        assert rec.system_prompt == ""
        assert rec.thinking_text == ""

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_save_includes_token_usage(self, _mock_save, ctrl, mock_app):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        result_holder = {
            "enhanced_text": "Hello.",
            "token_usage": usage,
        }
        mock_app._current_preview_asr_text = "hello"
        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "openai/gpt-4o"

        ctrl._save_to_preview_history(
            "ts1", "confirm", result_holder, None, 0.0, "voice",
        )

        rec = ctrl._preview_history.get(0)
        assert rec.token_usage == usage

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_save_token_usage_none_when_missing(self, _mock_save, ctrl, mock_app):
        result_holder = {"enhanced_text": "Hello."}
        mock_app._current_preview_asr_text = "hello"
        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "openai/gpt-4o"

        ctrl._save_to_preview_history(
            "ts1", "confirm", result_holder, None, 0.0, "voice",
        )

        rec = ctrl._preview_history.get(0)
        assert rec.token_usage is None


class TestHandleHistoryConfirm:
    """Tests for _handle_history_confirm system_prompt/thinking_text/token_usage update."""

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_updates_prompt_and_thinking_when_present(self, _mock_save, ctrl, mock_app):
        record = _make_record(system_prompt="old", thinking_text="old think")
        ctrl._preview_history.add(record)

        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "openai/gpt-4o"

        result_holder = {
            "text": "Hello.",
            "enhanced_text": "Hello.",
            "system_prompt": "new prompt",
            "thinking_text": "new think",
        }

        ctrl._handle_history_confirm(0, result_holder, None, 0.0, "voice")
        assert record.system_prompt == "new prompt"
        assert record.thinking_text == "new think"

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_preserves_prompt_and_thinking_when_absent(self, _mock_save, ctrl, mock_app):
        record = _make_record(system_prompt="keep", thinking_text="keep think")
        ctrl._preview_history.add(record)

        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "openai/gpt-4o"

        result_holder = {"text": "Hello.", "enhanced_text": "Hello."}

        ctrl._handle_history_confirm(0, result_holder, None, 0.0, "voice")
        assert record.system_prompt == "keep"
        assert record.thinking_text == "keep think"

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_updates_token_usage_when_present(self, _mock_save, ctrl, mock_app):
        old_usage = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        new_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        record = _make_record(token_usage=old_usage)
        ctrl._preview_history.add(record)

        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "openai/gpt-4o"

        result_holder = {
            "text": "Hello.",
            "enhanced_text": "Hello.",
            "token_usage": new_usage,
        }

        ctrl._handle_history_confirm(0, result_holder, None, 0.0, "voice")
        assert record.token_usage == new_usage

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_preserves_token_usage_when_absent(self, _mock_save, ctrl, mock_app):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        record = _make_record(token_usage=usage)
        ctrl._preview_history.add(record)

        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "openai/gpt-4o"

        result_holder = {"text": "Hello.", "enhanced_text": "Hello."}

        ctrl._handle_history_confirm(0, result_holder, None, 0.0, "voice")
        assert record.token_usage == usage


class TestSelectHistory:
    """Tests for on_select_history passing fields and syncing result_holder."""

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_passes_prompt_and_thinking_to_panel(self, _mock_save, ctrl, mock_app):
        record = _make_record(
            system_prompt="sys prompt",
            thinking_text="think text",
            wav_data=None,
            audio_duration=0.0,
        )
        ctrl._preview_history.add(record)

        ctrl.on_select_history(0)

        mock_app._preview_panel.load_history_record.assert_called_once()
        call_kwargs = mock_app._preview_panel.load_history_record.call_args
        assert call_kwargs.kwargs.get("system_prompt") == "sys prompt" or \
            call_kwargs[1].get("system_prompt") == "sys prompt"
        assert call_kwargs.kwargs.get("thinking_text") == "think text" or \
            call_kwargs[1].get("thinking_text") == "think text"

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_passes_token_usage_to_panel(self, _mock_save, ctrl, mock_app):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        record = _make_record(
            token_usage=usage,
            wav_data=None,
            audio_duration=0.0,
        )
        ctrl._preview_history.add(record)

        ctrl.on_select_history(0)

        call_kwargs = mock_app._preview_panel.load_history_record.call_args
        assert call_kwargs.kwargs.get("token_usage") == usage or \
            call_kwargs[1].get("token_usage") == usage

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_syncs_result_holder_with_record(self, _mock_save, ctrl, mock_app):
        """Selecting history syncs result_holder to prevent stale data."""
        usage = {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110}
        record = _make_record(
            enhanced_text="Record enhanced.",
            system_prompt="record prompt",
            thinking_text="record think",
            token_usage=usage,
            wav_data=None,
            audio_duration=0.0,
        )
        ctrl._preview_history.add(record)

        # Simulate an active preview session with stale data
        ctrl._result_holder = {
            "text": None,
            "confirmed": False,
            "enhanced_text": "Stale enhanced.",
            "system_prompt": "stale prompt",
            "thinking_text": "stale think",
            "token_usage": {"prompt_tokens": 999, "total_tokens": 999},
        }

        ctrl.on_select_history(0)

        assert ctrl._result_holder["enhanced_text"] == "Record enhanced."
        assert ctrl._result_holder["system_prompt"] == "record prompt"
        assert ctrl._result_holder["thinking_text"] == "record think"
        assert ctrl._result_holder["token_usage"] == usage

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_sync_skipped_when_result_holder_is_none(self, _mock_save, ctrl, mock_app):
        """No crash when result_holder is None (e.g. direct test call)."""
        record = _make_record(wav_data=None, audio_duration=0.0)
        ctrl._preview_history.add(record)
        ctrl._result_holder = None

        ctrl.on_select_history(0)  # should not raise

        mock_app._preview_panel.load_history_record.assert_called_once()

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_pushes_asr_diffs_to_panel(self, _mock_save, ctrl, mock_app):
        """Selecting history pushes ASR→Enhanced diffs to the side panel."""
        record = _make_record(
            asr_text="hello world",
            enhanced_text="Hello World",
            final_text="Hello World",
            wav_data=None, audio_duration=0.0,
        )
        ctrl._preview_history.add(record)

        ctrl.on_select_history(0)

        # clear_diffs() clears all, then set_asr_diffs pushes new ones
        mock_app._preview_panel.clear_diffs.assert_called_once()
        mock_app._preview_panel.set_asr_diffs.assert_called_once()
        pairs = mock_app._preview_panel.set_asr_diffs.call_args[0][0]
        assert len(pairs) > 0

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_pushes_user_diffs_when_final_differs(self, _mock_save, ctrl, mock_app):
        """Selecting history pushes Enhanced→Final diffs when they differ."""
        record = _make_record(
            asr_text="hello",
            enhanced_text="Hello good.",
            final_text="Hello great.",
            wav_data=None, audio_duration=0.0,
        )
        ctrl._preview_history.add(record)

        ctrl.on_select_history(0)

        # clear_diffs() clears all, then set_user_diffs pushes new diffs
        mock_app._preview_panel.clear_diffs.assert_called_once()
        mock_app._preview_panel.set_user_diffs.assert_called_once()

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_no_user_diffs_when_final_equals_enhanced(self, _mock_save, ctrl, mock_app):
        """No user diffs pushed when final_text == enhanced_text."""
        record = _make_record(
            asr_text="hello",
            enhanced_text="Hello.",
            final_text="Hello.",
            wav_data=None, audio_duration=0.0,
        )
        ctrl._preview_history.add(record)

        ctrl.on_select_history(0)

        # clear_diffs() handles clearing; no separate set_user_diffs call
        mock_app._preview_panel.clear_diffs.assert_called_once()
        mock_app._preview_panel.set_user_diffs.assert_not_called()

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_pushes_manual_vocab_state(self, _mock_save, ctrl, mock_app):
        """Selecting history syncs manual vocab state to the panel."""
        record = _make_record(wav_data=None, audio_duration=0.0)
        ctrl._preview_history.add(record)
        mock_app._manual_vocab_store = MagicMock()
        mock_app._manual_vocab_store.get_all_for_state.return_value = [{"v": "a"}]

        ctrl.on_select_history(0)

        mock_app._preview_panel.set_manual_vocab_state.assert_called_once_with([{"v": "a"}])


class TestLogWithChainSteps:
    """Tests for _log_with_chain_steps per-mode history logging."""

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_non_chain_mode_logs_single_entry(self, _mock_save, ctrl, mock_app):
        """Without chain_step_results, logs a single entry under current mode."""
        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "gpt-4o"
        mock_app._conversation_history.log.return_value = "ts-001"

        result_holder = {"enhanced_text": "Hello.", "user_corrected": False}

        ts = ctrl._log_with_chain_steps(
            mock_app,
            result_holder=result_holder,
            asr_text="hello",
            final_text="Hello.",
        )

        assert ts == "ts-001"
        mock_app._conversation_history.log.assert_called_once()
        call_kwargs = mock_app._conversation_history.log.call_args.kwargs
        assert call_kwargs["enhance_mode"] == "proofread"
        assert call_kwargs["asr_text"] == "hello"
        assert call_kwargs["final_text"] == "Hello."

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_chain_mode_skips_logging(self, _mock_save, ctrl, mock_app):
        """Chain mode (is_chain=True) does not log to conversation history."""
        result_holder = {
            "enhanced_text": "Hello World",
            "user_corrected": True,
            "is_chain": True,
        }

        ts = ctrl._log_with_chain_steps(
            mock_app,
            result_holder=result_holder,
            asr_text="你好试解",
            final_text="Hello World!",
            audio_duration=2.5,
        )

        assert ts is None
        mock_app._conversation_history.log.assert_not_called()

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_non_chain_without_flag_logs_normally(self, _mock_save, ctrl, mock_app):
        """result_holder without is_chain flag logs normally."""
        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "gpt-4o"
        mock_app._conversation_history.log.return_value = "ts-001"

        result_holder = {"enhanced_text": "Hello."}

        ctrl._log_with_chain_steps(
            mock_app,
            result_holder=result_holder,
            asr_text="hello",
            final_text="Hello.",
        )

        mock_app._conversation_history.log.assert_called_once()
        assert mock_app._conversation_history.log.call_args.kwargs["enhance_mode"] == "proofread"

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_is_chain_false_logs_normally(self, _mock_save, ctrl, mock_app):
        """is_chain=False still logs normally."""
        mock_app._enhance_mode = "proofread"
        mock_app._current_stt_model.return_value = "funasr"
        mock_app._current_llm_model.return_value = "gpt-4o"
        mock_app._conversation_history.log.return_value = "ts-001"

        result_holder = {"enhanced_text": "Hello.", "is_chain": False}

        ctrl._log_with_chain_steps(
            mock_app,
            result_holder=result_holder,
            asr_text="hello",
            final_text="Hello.",
        )

        mock_app._conversation_history.log.assert_called_once()


class TestModeChangeResultHolder:
    """Tests for on_preview_mode_change updating result_holder."""

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_mode_off_clears_result_holder(self, _mock_save, ctrl, mock_app):
        ctrl._result_holder = {
            "enhanced_text": "Hello.",
            "system_prompt": "old",
            "thinking_text": "old think",
            "token_usage": {"total_tokens": 100},
        }

        ctrl.on_preview_mode_change("off")

        assert ctrl._result_holder["enhanced_text"] is None
        assert ctrl._result_holder["system_prompt"] == ""
        assert ctrl._result_holder["thinking_text"] == ""
        assert ctrl._result_holder["token_usage"] is None

    @patch("wenzi.controllers.preview_controller.save_config")
    def test_cache_hit_updates_result_holder(self, _mock_save, ctrl, mock_app):
        cached_usage = {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110}
        cached = MagicMock()
        cached.display_text = "Hello."
        cached.usage = cached_usage
        cached.system_prompt = "cached prompt"
        cached.thinking_text = "cached think"
        cached.final_text = "Hello."
        mock_app._enhance_controller.get_cached.return_value = cached

        ctrl._result_holder = {"enhanced_text": None}

        ctrl.on_preview_mode_change("translate")

        assert ctrl._result_holder["enhanced_text"] == "Hello."
        assert ctrl._result_holder["system_prompt"] == "cached prompt"
        assert ctrl._result_holder["thinking_text"] == "cached think"
        assert ctrl._result_holder["token_usage"] == cached_usage
