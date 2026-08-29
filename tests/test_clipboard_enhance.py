"""Tests for clipboard AI enhancement feature."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# Mock AppKit/Foundation before importing modules that use them
@pytest.fixture(autouse=True)
def mock_appkit(monkeypatch):
    """Provide mock AppKit and Foundation modules for headless testing."""
    mock_appkit_mod = MagicMock()
    mock_appkit_mod.NSCommandKeyMask = 1 << 20
    mock_appkit_mod.NSShiftKeyMask = 1 << 17
    mock_appkit_mod.NSDeviceIndependentModifierFlagsMask = 0xFFFF0000
    mock_appkit_mod.NSKeyDownMask = 1 << 10

    modules = {
        "AppKit": mock_appkit_mod,
        "Foundation": MagicMock(),
        "objc": MagicMock(),
        "PyObjCTools": MagicMock(),
        "PyObjCTools.AppHelper": MagicMock(),
    }

    for name, mod in modules.items():
        monkeypatch.setitem(__import__("sys").modules, name, mod)


class TestClipboardPublicFunctions:
    """Test public clipboard read/write functions in input.py."""

    @patch("wenzi.input.NSPasteboard")
    def test_get_clipboard_text(self, mock_pb_cls):
        from wenzi.input import get_clipboard_text

        mock_pb = MagicMock()
        mock_pb_cls.generalPasteboard.return_value = mock_pb
        mock_pb.stringForType_.return_value = "hello clipboard"

        result = get_clipboard_text()
        assert result == "hello clipboard"

    @patch("wenzi.input.NSPasteboard")
    def test_get_clipboard_text_empty(self, mock_pb_cls):
        from wenzi.input import get_clipboard_text

        mock_pb = MagicMock()
        mock_pb_cls.generalPasteboard.return_value = mock_pb
        mock_pb.stringForType_.return_value = None

        result = get_clipboard_text()
        assert result is None

    @patch("wenzi.input.NSString")
    @patch("wenzi.input.NSPasteboard")
    def test_set_clipboard_text(self, mock_pb_cls, mock_nsstr):
        from wenzi.input import set_clipboard_text

        mock_pb = MagicMock()
        mock_pb_cls.generalPasteboard.return_value = mock_pb
        mock_nsstr.stringWithString_.return_value = "enhanced text"

        set_clipboard_text("enhanced text")

        mock_pb.clearContents.assert_called_once()
        # Should set string without concealed markers
        assert mock_pb.setString_forType_.call_count == 1

    @patch("wenzi.input.NSPasteboardTypeString", "public.utf8-plain-text")
    @patch("wenzi.input.NSPasteboard")
    def test_has_clipboard_text_true(self, mock_pb_cls):
        from wenzi.input import has_clipboard_text

        mock_pb = MagicMock()
        mock_pb_cls.generalPasteboard.return_value = mock_pb
        mock_pb.availableTypeFromArray_.return_value = "public.utf8-plain-text"

        assert has_clipboard_text() is True

    @patch("wenzi.input.NSPasteboardTypeString", "public.utf8-plain-text")
    @patch("wenzi.input.NSPasteboard")
    def test_has_clipboard_text_false_for_image(self, mock_pb_cls):
        from wenzi.input import has_clipboard_text

        mock_pb = MagicMock()
        mock_pb_cls.generalPasteboard.return_value = mock_pb
        mock_pb.availableTypeFromArray_.return_value = None

        assert has_clipboard_text() is False


class TestCopySelectionToClipboard:
    """Test copy_selection_to_clipboard() function."""

    @patch("wenzi.input._has_text_selection", return_value=True)
    @patch("wenzi.input.time.sleep")
    @patch("wenzi.input._send_cmd_c")
    @patch("wenzi.input.get_clipboard_text")
    def test_selection_copied_successfully(self, mock_get, mock_send, mock_sleep, _):
        from wenzi.input import copy_selection_to_clipboard

        # Clipboard changes after Cmd+C
        mock_get.side_effect = ["old text", "new selected text"]

        result = copy_selection_to_clipboard()

        assert result is True
        mock_send.assert_called_once()
        # Two sleeps: 0.05 before Cmd+C and 0.15 after
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.05)
        mock_sleep.assert_any_call(0.15)

    @patch("wenzi.input._has_text_selection", return_value=False)
    def test_no_selection_skips_cmd_c(self, _):
        from wenzi.input import copy_selection_to_clipboard

        result = copy_selection_to_clipboard()

        assert result is False

    @patch("wenzi.input._has_text_selection", return_value=True)
    @patch("wenzi.input.time.sleep")
    @patch("wenzi.input._send_cmd_c")
    @patch("wenzi.input.get_clipboard_text")
    def test_clipboard_unchanged_returns_false(self, mock_get, mock_send, mock_sleep, _):
        from wenzi.input import copy_selection_to_clipboard

        # Clipboard stays the same (nothing selected per clipboard check)
        mock_get.side_effect = ["same text", "same text"]

        result = copy_selection_to_clipboard()

        assert result is False

    @patch("wenzi.input._has_text_selection", return_value=True)
    @patch("wenzi.input.time.sleep")
    @patch("wenzi.input._send_cmd_c")
    @patch("wenzi.input.get_clipboard_text")
    def test_send_cmd_c_failure_returns_false(self, mock_get, mock_send, mock_sleep, _):
        from wenzi.input import copy_selection_to_clipboard

        mock_get.return_value = "old text"
        mock_send.side_effect = OSError("failed")

        result = copy_selection_to_clipboard()

        assert result is False


class TestClipboardEnhanceValidation:
    """Test clipboard content validation before enhancement."""

    @pytest.fixture(autouse=True)
    def mock_appkit(self):
        """Override the module-level autouse fixture — not needed here."""
        pass

    def _make_app_and_ctrl(self):
        """Create a minimal mock app and a real PreviewController."""
        from wenzi.controllers.preview_controller import PreviewController

        app = MagicMock(spec=[])
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
        ctrl = PreviewController(app)
        return app, ctrl

    def test_non_text_clipboard_shows_alert(self):
        with patch("wenzi.controllers.preview_controller.copy_selection_to_clipboard"), \
             patch("wenzi.controllers.preview_controller.has_clipboard_text", return_value=False), \
             patch("wenzi.controllers.preview_controller.topmost_alert") as mock_alert, \
             patch("wenzi.controllers.preview_controller.restore_accessory") as mock_restore:
            app, ctrl = self._make_app_and_ctrl()

            mock_helper = MagicMock()
            mock_helper.callAfter = lambda fn, *a: fn(*a)
            with patch.dict("sys.modules", {
                "PyObjCTools": MagicMock(AppHelper=mock_helper),
                "PyObjCTools.AppHelper": mock_helper,
            }):
                ctrl._on_clipboard_enhance_worker()

            mock_alert.assert_called_once()
            title = mock_alert.call_args[1]["title"]
            assert "Not Supported" in title or "not_supported" in title
            mock_restore.assert_called_once()

    def test_empty_text_clipboard_shows_alert(self):
        with patch("wenzi.controllers.preview_controller.copy_selection_to_clipboard"), \
             patch("wenzi.controllers.preview_controller.has_clipboard_text", return_value=True), \
             patch("wenzi.controllers.preview_controller.get_clipboard_text", return_value=""), \
             patch("wenzi.controllers.preview_controller.topmost_alert") as mock_alert, \
             patch("wenzi.controllers.preview_controller.restore_accessory") as mock_restore:
            app, ctrl = self._make_app_and_ctrl()

            mock_helper = MagicMock()
            mock_helper.callAfter = lambda fn, *a: fn(*a)
            with patch.dict("sys.modules", {
                "PyObjCTools": MagicMock(AppHelper=mock_helper),
                "PyObjCTools.AppHelper": mock_helper,
            }):
                ctrl._on_clipboard_enhance_worker()

            mock_alert.assert_called_once()
            title = mock_alert.call_args[1]["title"]
            assert "Empty" in title or "empty" in title
            mock_restore.assert_called_once()

    def test_long_text_shows_alert_and_aborts(self):
        with patch("wenzi.controllers.preview_controller.copy_selection_to_clipboard"), \
             patch("wenzi.controllers.preview_controller.has_clipboard_text", return_value=True), \
             patch("wenzi.controllers.preview_controller.get_clipboard_text", return_value="x" * 2001), \
             patch("wenzi.controllers.preview_controller.topmost_alert") as mock_alert, \
             patch("wenzi.controllers.preview_controller.restore_accessory") as mock_restore:
            app, ctrl = self._make_app_and_ctrl()

            mock_helper = MagicMock()
            mock_helper.callAfter = lambda fn, *a: fn(*a)
            with patch.dict("sys.modules", {
                "PyObjCTools": MagicMock(AppHelper=mock_helper),
                "PyObjCTools.AppHelper": mock_helper,
            }):
                ctrl._on_clipboard_enhance_worker()

            mock_alert.assert_called_once()
            message = mock_alert.call_args[1]["message"]
            assert "2001" in message or "too_long" in message
            mock_restore.assert_called_once()
            assert not app._busy

    def test_normal_text_proceeds_without_alert(self):
        with patch("wenzi.controllers.preview_controller.copy_selection_to_clipboard"), \
             patch("wenzi.controllers.preview_controller.has_clipboard_text", return_value=True), \
             patch("wenzi.controllers.preview_controller.get_clipboard_text", return_value="short text"):
            app, ctrl = self._make_app_and_ctrl()
            app._set_status = MagicMock()
            ctrl._do_clipboard_with_preview = MagicMock()

            mock_helper = MagicMock()
            mock_helper.callAfter = lambda fn, *a: fn(*a)
            with patch.dict("sys.modules", {
                "PyObjCTools": MagicMock(AppHelper=mock_helper),
                "PyObjCTools.AppHelper": mock_helper,
            }):
                ctrl._on_clipboard_enhance_worker()

            ctrl._do_clipboard_with_preview.assert_called_once_with("short text")
            assert app._busy is False  # busy is reset in finally block

    def test_busy_skips(self):
        app, ctrl = self._make_app_and_ctrl()
        app._busy = True

        mock_helper = MagicMock()
        with patch.dict("sys.modules", {
            "PyObjCTools": MagicMock(AppHelper=mock_helper),
            "PyObjCTools.AppHelper": mock_helper,
        }):
            ctrl._on_clipboard_enhance_worker()

    def test_run_clipboard_preview_refused_keeps_mode(self):
        """A refused Universal-Action preview must not change the global
        enhance mode — the mode switch happens only after the claim."""
        app, ctrl = self._make_app_and_ctrl()
        app._busy = True
        app._enhance_mode = "proofread"

        ctrl.run_clipboard_preview("some text", mode_id="translate")

        assert app._enhance_mode == "proofread"

    def test_run_clipboard_preview_applies_mode_before_preview(self):
        """The mode (incl. enabling the enhancer when it was Off) must be
        applied before the preview computes use_enhance."""
        app, ctrl = self._make_app_and_ctrl()
        app._enhance_mode = "off"
        app._enhancer = MagicMock()
        app._enhancer._enabled = False
        app._enhance_controller = MagicMock()
        seen: dict = {}
        ctrl._do_clipboard_with_preview = MagicMock(
            side_effect=lambda text: seen.update(
                mode=app._enhance_mode,
                controller_mode=app._enhance_controller.enhance_mode,
                enabled=app._enhancer._enabled,
                enhancer_mode=app._enhancer.mode,
            )
        )

        mock_helper = MagicMock()
        mock_helper.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        with patch.dict("sys.modules", {
            "PyObjCTools": MagicMock(AppHelper=mock_helper),
            "PyObjCTools.AppHelper": mock_helper,
        }):
            ctrl.run_clipboard_preview("some text", mode_id="translate")

        assert seen["mode"] == "translate"
        # EnhanceController.enhance_mode drives the chain/single split
        # and the cache key — it must move together with the app mode
        assert seen["controller_mode"] == "translate"
        assert seen["enabled"] is True
        assert seen["enhancer_mode"] == "translate"

    def test_ua_mode_apply_timeout_aborts_and_late_apply_is_noop(self):
        """When the main thread is too slow, the preview must abort AND
        the already-queued mode callback must not fire late — it would
        mutate the global mode under a newer, unrelated operation."""
        app, ctrl = self._make_app_and_ctrl()
        app._enhance_mode = "proofread"
        app._enhancer = MagicMock()
        app._enhance_controller = MagicMock()
        ctrl._do_clipboard_with_preview = MagicMock()
        ctrl._MODE_APPLY_TIMEOUT = 0.05

        captured: list = []
        mock_helper = MagicMock()
        mock_helper.callAfter = lambda fn, *a, **kw: captured.append(fn)
        with patch.dict("sys.modules", {
            "PyObjCTools": MagicMock(AppHelper=mock_helper),
            "PyObjCTools.AppHelper": mock_helper,
        }):
            ctrl.run_clipboard_preview("some text", mode_id="translate")

        ctrl._do_clipboard_with_preview.assert_not_called()
        assert app._busy is False  # op slot released on abort

        # The queued callback fires late — it must be a no-op
        assert captured
        captured[0]()
        assert app._enhance_mode == "proofread"

    def test_ua_opguard_held_while_apply_in_progress(self):
        """Once the callback entered 'applying', the worker must wait for
        it — the OpGuard cannot be released mid-apply."""
        import threading
        import time

        app, ctrl = self._make_app_and_ctrl()
        app._enhance_mode = "proofread"
        app._enhancer = MagicMock()
        ctrl._do_clipboard_with_preview = MagicMock()
        ctrl._MODE_APPLY_TIMEOUT = 0.05

        gate = threading.Event()
        entered = threading.Event()

        class _BlockyCtrl:
            def __init__(self):
                self._mode = "proofread"

            @property
            def enhance_mode(self):
                return self._mode

            @enhance_mode.setter
            def enhance_mode(self, value):
                entered.set()
                gate.wait(5)
                self._mode = value

        app._enhance_controller = _BlockyCtrl()

        mock_helper = MagicMock()
        mock_helper.callAfter = lambda fn, *a, **kw: threading.Thread(
            target=fn, daemon=True
        ).start()

        with patch.dict("sys.modules", {
            "PyObjCTools": MagicMock(AppHelper=mock_helper),
            "PyObjCTools.AppHelper": mock_helper,
        }):
            worker = threading.Thread(
                target=lambda: ctrl.run_clipboard_preview(
                    "text", mode_id="translate"
                ),
                daemon=True,
            )
            worker.start()
            assert entered.wait(5)
            time.sleep(0.2)  # well past the 0.05s timeout
            # Applying in progress → the op slot must still be held
            assert app._busy is True
            gate.set()
            worker.join(5)

        assert not worker.is_alive()
        assert app._busy is False
        assert app._enhance_controller.enhance_mode == "translate"
        ctrl._do_clipboard_with_preview.assert_called_once()

    def test_ua_apply_failure_rolls_back_everything(self):
        """A setter raising mid-apply must restore app, controller and
        enhancer state on the main thread, then abort the preview."""
        app, ctrl = self._make_app_and_ctrl()
        app._enhance_mode = "proofread"
        app._enhancer = MagicMock()
        app._enhancer._enabled = False
        app._enhancer.mode = "old-mode"
        ctrl._do_clipboard_with_preview = MagicMock()

        class _RaisingCtrl:
            def __init__(self):
                self._mode = "ctrl-old"

            @property
            def enhance_mode(self):
                return self._mode

            @enhance_mode.setter
            def enhance_mode(self, value):
                raise RuntimeError("setter blew up")

        app._enhance_controller = _RaisingCtrl()

        mock_helper = MagicMock()
        mock_helper.callAfter = lambda fn, *a, **kw: fn(*a, **kw)
        with patch.dict("sys.modules", {
            "PyObjCTools": MagicMock(AppHelper=mock_helper),
            "PyObjCTools.AppHelper": mock_helper,
        }):
            ctrl.run_clipboard_preview("text", mode_id="translate")

        ctrl._do_clipboard_with_preview.assert_not_called()
        assert app._busy is False
        # Full rollback: app mode, enhancer enabled + mode
        assert app._enhance_mode == "proofread"
        assert app._enhancer._enabled is False
        assert app._enhancer.mode == "old-mode"

    def test_dispatches_to_worker_thread(self):
        """Verify on_clipboard_enhance starts a worker thread."""
        from wenzi.controllers.preview_controller import PreviewController

        app = MagicMock()
        ctrl = PreviewController(app)

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            ctrl.on_clipboard_enhance()
            mock_thread.assert_called_once()
            assert mock_thread.call_args[1]["target"] == ctrl._on_clipboard_enhance_worker


class TestPreviewPanelClipboardSource:
    """Test Preview panel behavior with source='clipboard'."""

    def _setup_panel(self):
        from wenzi.ui.result_window_web import ResultPreviewPanel

        panel = ResultPreviewPanel()
        panel._build_panel = MagicMock()
        panel._panel = MagicMock()
        panel._webview = MagicMock()
        panel._page_loaded = True
        return panel

    def test_source_defaults_to_voice(self):
        panel = self._setup_panel()

        panel.show(
            asr_text="hello",
            show_enhance=False,
            on_confirm=MagicMock(),
            on_cancel=MagicMock(),
        )

        assert panel._source == "voice"

    def test_source_clipboard_stored(self):
        panel = self._setup_panel()

        panel.show(
            asr_text="clipboard text",
            show_enhance=True,
            on_confirm=MagicMock(),
            on_cancel=MagicMock(),
            source="clipboard",
        )

        assert panel._source == "clipboard"

    def test_clipboard_source_no_wav_data(self):
        panel = self._setup_panel()

        panel.show(
            asr_text="clipboard text",
            show_enhance=True,
            on_confirm=MagicMock(),
            on_cancel=MagicMock(),
            source="clipboard",
            asr_wav_data=None,
        )

        assert panel._asr_wav_data is None
        assert panel._source == "clipboard"


class TestClipboardEnhanceConfig:
    """Test clipboard_enhance config defaults."""

    def test_default_config_has_clipboard_enhance(self):
        from wenzi.config import DEFAULT_CONFIG

        assert "clipboard_enhance" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["clipboard_enhance"]["hotkey"] == "ctrl+cmd+v"
        assert "output" not in DEFAULT_CONFIG["clipboard_enhance"]

    def test_config_merge_preserves_clipboard_enhance(self):
        from wenzi.config import DEFAULT_CONFIG, _merge_dict

        overrides = {
            "clipboard_enhance": {
                "hotkey": "ctrl+shift+v",
            }
        }
        result = _merge_dict(DEFAULT_CONFIG, overrides)
        assert result["clipboard_enhance"]["hotkey"] == "ctrl+shift+v"
