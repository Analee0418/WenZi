"""Tests for app menu structure and Show Config functionality."""

from __future__ import annotations

import asyncio
import os
import signal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from wenzi.controllers.config_controller import ConfigController
from wenzi.controllers.menu_builder import MenuBuilder
from wenzi.enhance.enhancer import MODE_OFF


def _make_mock_app():
    """Create a minimal mock of WenZiApp for testing _build_config_info."""
    app = MagicMock(spec=[])
    app._current_remote_asr = None
    app._current_preset_id = "funasr-zh"
    app._enhance_mode = "proofread"
    app._preview_enabled = True
    app._output_method = "clipboard"
    app._config = {
        "hotkeys": {"right_cmd": True, "fn": True},
        "logging": {"level": "INFO"},
    }
    app._config_path = "/tmp/test_config.yaml"

    app._enhancer = MagicMock()
    app._enhancer.provider_name = "my-provider"
    app._enhancer.model_name = "gpt-4o"
    app._enhancer.thinking = True

    app._enhance_vocab_item = MagicMock()
    app._enhance_vocab_item.state = 1
    app._enhance_history_item = MagicMock()
    app._enhance_history_item.state = 0

    return app


def _get_info(app):
    """Call build_config_info via ConfigController with PRESET_BY_ID patched."""
    ctrl = ConfigController(app)
    preset_map = {"funasr-zh": MagicMock(display_name="FunASR 中文")}
    with patch("wenzi.controllers.config_controller.PRESET_BY_ID", preset_map):
        return ctrl.build_config_info()


class TestAutomaticInputMigration:
    def test_legacy_device_uid_becomes_automatic(self):
        from wenzi.app import _migrate_input_device_to_automatic

        config = {"audio": {"device": "airpods-uid", "sample_rate": 16000}}

        assert _migrate_input_device_to_automatic(config)
        assert config["audio"] == {"device": None, "sample_rate": 16000}

    def test_automatic_device_config_is_unchanged(self):
        from wenzi.app import _migrate_input_device_to_automatic

        config = {"audio": {"device": None}}

        assert not _migrate_input_device_to_automatic(config)
        assert config["audio"]["device"] is None


class TestBuildConfigInfo:
    """Tests for _build_config_info."""

    def test_all_fields_present(self):
        app = _make_mock_app()
        info = _get_info(app)

        assert "FunASR" in info
        assert "proofread" in info
        assert "my-provider" in info
        assert "gpt-4o" in info
        assert "Thinking:       \u2705" in info
        assert "Preview:        \u2705" in info
        assert "Vocabulary:     \u2705" in info
        assert "History:        \u274C" in info
        assert "clipboard" in info
        assert "right_cmd" in info
        assert "INFO" in info
        assert "test_config.yaml" in info

    def test_default_config_path(self):
        app = _make_mock_app()
        app._config_path = None
        info = _get_info(app)

        assert "None" not in info
        # Config path may be patched by conftest; check the live value
        import wenzi.config as _cfg
        expected = os.path.expanduser(_cfg.DEFAULT_CONFIG_PATH)
        assert expected in info

    def test_no_enhancer(self):
        app = _make_mock_app()
        app._enhancer = None
        info = _get_info(app)

        assert "AI Provider:    N/A" in info
        assert "AI Model:       N/A" in info
        assert "Thinking:       N/A" in info

    def test_toggle_states_off(self):
        app = _make_mock_app()
        app._preview_enabled = False
        app._enhancer.thinking = False
        app._enhance_vocab_item.state = 0
        app._enhance_history_item.state = 0
        info = _get_info(app)

        assert "Thinking:       \u274C" in info
        assert "Preview:        \u274C" in info
        assert "Vocabulary:     \u274C" in info
        assert "History:        \u274C" in info

    def test_enhance_mode_off(self):
        app = _make_mock_app()
        app._enhance_mode = MODE_OFF
        info = _get_info(app)

        assert f"AI Enhance:     {MODE_OFF}" in info

    def test_unknown_preset(self):
        app = _make_mock_app()
        app._current_remote_asr = None
        app._current_preset_id = "unknown-preset"

        ctrl = ConfigController(app)
        with patch("wenzi.controllers.config_controller.PRESET_BY_ID", {}):
            info = ctrl.build_config_info()

        assert "unknown-preset" in info

    def test_remote_asr_active(self):
        app = _make_mock_app()
        app._current_remote_asr = ("groq", "whisper-large-v3-turbo")
        info = _get_info(app)

        assert "groq / whisper-large-v3-turbo (remote)" in info


class TestHelpMenu:
    """Tests for the help menu functionality."""

    @patch("webbrowser.open")
    @patch("wenzi.i18n.get_locale", return_value="zh")
    def test_help_click_chinese_locale(self, mock_locale, mock_open):
        app = MagicMock()
        builder = MenuBuilder(app)
        builder.on_help_click(MagicMock())

        mock_open.assert_called_once()
        url = mock_open.call_args[0][0]
        assert url == "https://airead.github.io/WenZi/zh/docs/user-guide.html"

    @patch("webbrowser.open")
    @patch("wenzi.i18n.get_locale", return_value="zh")
    def test_help_click_chinese_traditional_locale(self, mock_locale, mock_open):
        app = MagicMock()
        builder = MenuBuilder(app)
        builder.on_help_click(MagicMock())

        mock_open.assert_called_once()
        url = mock_open.call_args[0][0]
        assert url == "https://airead.github.io/WenZi/zh/docs/user-guide.html"

    @patch("webbrowser.open")
    @patch("wenzi.i18n.get_locale", return_value="en")
    def test_help_click_english_locale(self, mock_locale, mock_open):
        app = MagicMock()
        builder = MenuBuilder(app)
        builder.on_help_click(MagicMock())

        mock_open.assert_called_once()
        url = mock_open.call_args[0][0]
        assert url == "https://airead.github.io/WenZi/docs/user-guide.html"

    @patch("webbrowser.open")
    @patch("wenzi.i18n.get_locale", return_value="en")
    def test_help_click_no_locale(self, mock_locale, mock_open):
        app = MagicMock()
        builder = MenuBuilder(app)
        builder.on_help_click(MagicMock())

        mock_open.assert_called_once()
        url = mock_open.call_args[0][0]
        assert url == "https://airead.github.io/WenZi/docs/user-guide.html"


class TestSystemOutputExitRestore:
    def test_restore_helper_is_best_effort_and_handles_partial_init(self):
        from wenzi.app import WenZiApp

        app = object.__new__(WenZiApp)
        WenZiApp._restore_system_output_on_exit(app)

        app._system_output_ducker = MagicMock()
        app._system_output_ducker.restore_all.side_effect = RuntimeError("restore")
        WenZiApp._restore_system_output_on_exit(app)

        app._system_output_ducker.restore_all.assert_called_once_with()
        app._system_output_ducker.recover_stale.assert_called_once_with(
            start_deferred=False
        )
        app._system_output_ducker.stop_background_workers.assert_called_once_with()

    @patch("wenzi.statusbar.restart_application")
    def test_restart_restores_before_relaunch(self, mock_restart):
        from wenzi.app import WenZiApp

        app = object.__new__(WenZiApp)
        order: list[str] = []
        app._shutdown_runtime = MagicMock(
            side_effect=lambda: order.append("shutdown") or True
        )
        mock_restart.side_effect = lambda: order.append("restart")

        WenZiApp._on_restart(app, None)

        assert order == ["shutdown", "restart"]

    @patch("wenzi.app.quit_application")
    @patch("wenzi.app.async_loop.shutdown_sync")
    def test_normal_quit_restores_output(
        self,
        mock_shutdown_loop,
        mock_quit,
    ):
        from wenzi.app import WenZiApp

        app = object.__new__(WenZiApp)
        app._update_controller = MagicMock()
        app._script_engine = None
        app._hotkey_listener = None
        app._app_hotkey_tap = MagicMock()
        app._settings_panel = MagicMock(is_visible=False)
        app._vocab_controller = None
        app._recording_indicator = MagicMock()
        app._streaming_overlay = MagicMock()
        app._transcriber = MagicMock()
        app._preview_panel = MagicMock()
        app._history_browser = None
        app._screenshot_annotation = None
        app._preview_controller = MagicMock()
        app._usage_stats = MagicMock()
        app._manual_vocab_store = MagicMock()
        app._enhancer = None
        order: list[str] = []
        app._stop_hotkey_sources_on_exit = MagicMock(
            side_effect=lambda: order.append("hotkeys")
        )
        app._stop_recording_on_exit = MagicMock(
            side_effect=lambda: order.append("recording")
        )
        app._restore_system_output_on_exit = MagicMock(
            side_effect=lambda: order.append("restore")
        )
        app._update_controller.stop.side_effect = lambda: order.append("cleanup")
        mock_shutdown_loop.side_effect = lambda **_kwargs: order.append("loop")
        mock_quit.side_effect = lambda: order.append("quit")

        with (
            patch("wenzi.input_context.shutdown_input_context"),
            patch("wenzi.vault.shutdown_vault"),
            patch("wenzi.hotkey.shutdown_hotkey_executor"),
            patch("wenzi.statusbar.cleanup_callbacks"),
        ):
            WenZiApp._on_quit_click(app, None)

        mock_shutdown_loop.assert_called_once_with(timeout=5)
        app._restore_system_output_on_exit.assert_called_once_with()
        mock_quit.assert_called_once_with()
        assert order == [
            "hotkeys",
            "recording",
            "restore",
            "cleanup",
            "loop",
            "quit",
        ]

        WenZiApp._on_quit_click(app, None)
        mock_quit.assert_called_once_with()
        app._stop_recording_on_exit.assert_called_once_with()

    @patch("wenzi.app.async_loop.submit")
    def test_recording_shutdown_waits_for_loop_cleanup(self, mock_submit):
        from wenzi.app import WenZiApp

        app = object.__new__(WenZiApp)
        app._recording_controller = MagicMock()
        app._recorder = MagicMock(is_recording=False)
        order: list[str] = []
        app._recording_controller.on_cancel_recording.side_effect = (
            lambda: order.append("cancel")
        )
        future = MagicMock()
        future.result.side_effect = lambda timeout: order.append(f"wait:{timeout}")

        def _submit(coro):
            order.append("submit")
            coro.close()
            return future

        mock_submit.side_effect = _submit

        WenZiApp._stop_recording_on_exit(app)

        assert order == ["cancel", "submit", "wait:7.0"]

    @patch("wenzi.app.quit_application")
    @patch("wenzi.app.async_loop.shutdown_sync")
    def test_non_audio_cleanup_errors_do_not_block_quit(
        self,
        mock_shutdown_loop,
        mock_quit,
    ):
        from wenzi.app import WenZiApp

        app = object.__new__(WenZiApp)
        app._stop_hotkey_sources_on_exit = MagicMock()
        app._stop_recording_on_exit = MagicMock()
        app._restore_system_output_on_exit = MagicMock()
        app._update_controller = MagicMock()
        app._update_controller.stop.side_effect = RuntimeError("update")
        app._settings_panel = MagicMock(is_visible=True)
        app._settings_panel.close.side_effect = RuntimeError("settings")
        app._vocab_controller = MagicMock()
        app._vocab_controller.close_panel.side_effect = RuntimeError("vocab")
        app._recording_indicator = MagicMock()
        app._streaming_overlay = MagicMock()
        app._transcriber = MagicMock()
        app._preview_panel = MagicMock()
        app._history_browser = None
        app._screenshot_annotation = None
        app._preview_controller = MagicMock()
        app._usage_stats = MagicMock()
        app._manual_vocab_store = MagicMock()
        app._enhancer = None

        with (
            patch("wenzi.input_context.shutdown_input_context"),
            patch("wenzi.vault.shutdown_vault"),
            patch("wenzi.statusbar.cleanup_callbacks"),
        ):
            WenZiApp._on_quit_click(app, None)

        mock_shutdown_loop.assert_called_once_with(timeout=5)
        mock_quit.assert_called_once_with()

    def test_recording_shutdown_awaits_shielded_audio_jobs(self):
        from wenzi.app import WenZiApp

        async def _exercise() -> list[str]:
            app = object.__new__(WenZiApp)
            order: list[str] = []
            recorder = SimpleNamespace(is_recording=True, stop=MagicMock())
            app._recorder = recorder
            controller = SimpleNamespace(
                _press_pending=False,
                _current_task=None,
                _audio_shutdown_task=None,
                _output_restore_task=None,
                _loop=asyncio.get_running_loop(),
            )
            app._recording_controller = controller

            async def _audio_cleanup() -> None:
                await asyncio.sleep(0)
                order.append("audio")
                recorder.is_recording = False

            async def _output_cleanup() -> None:
                await controller._audio_shutdown_task
                order.append("output")

            async def _session() -> None:
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    controller._audio_shutdown_task = asyncio.create_task(
                        _audio_cleanup()
                    )
                    controller._output_restore_task = asyncio.create_task(
                        _output_cleanup()
                    )
                    raise

            controller._current_task = asyncio.create_task(_session())
            await asyncio.sleep(0)
            await WenZiApp._cancel_recording_session_on_exit(app)
            recorder.stop.assert_not_called()
            return order

        assert asyncio.run(_exercise()) == ["audio", "output"]

    def test_recording_shutdown_waits_past_stale_done_session_task(self):
        from wenzi.app import WenZiApp

        async def _exercise() -> bool:
            app = object.__new__(WenZiApp)
            old_task = asyncio.create_task(asyncio.sleep(0))
            await old_task
            controller = SimpleNamespace(
                _press_pending=True,
                _current_task=old_task,
                _audio_shutdown_task=None,
                _output_restore_task=None,
                _loop=asyncio.get_running_loop(),
            )
            app._recording_controller = controller
            app._recorder = SimpleNamespace(
                is_recording=False,
                stop=MagicMock(),
            )
            cancelled = asyncio.Event()
            session_started = asyncio.Event()

            async def _new_session() -> None:
                try:
                    session_started.set()
                    await asyncio.sleep(60)
                finally:
                    cancelled.set()

            async def _publish_session() -> None:
                await asyncio.sleep(0.02)
                controller._current_task = asyncio.create_task(_new_session())
                await session_started.wait()
                controller._press_pending = False

            publisher = asyncio.create_task(_publish_session())
            await WenZiApp._cancel_recording_session_on_exit(app)
            await publisher
            return cancelled.is_set()

        assert asyncio.run(_exercise()) is True

    @patch("wenzi.app.async_loop.submit")
    def test_context_only_shutdown_timeout_cancels_waiter(self, mock_submit):
        from wenzi.app import WenZiApp

        app = object.__new__(WenZiApp)
        stale_task = MagicMock()
        stale_task.done.return_value = True
        controller = MagicMock()
        controller._press_pending = True
        controller._current_task = stale_task
        app._recording_controller = controller
        app._recorder = MagicMock(is_recording=False)
        future = MagicMock()
        future.result.side_effect = TimeoutError
        future.done.return_value = False

        def _submit(coro):
            coro.close()
            return future

        mock_submit.side_effect = _submit

        WenZiApp._stop_recording_on_exit(app, timeout=0.01)

        future.cancel.assert_called_once_with()
        app._recorder.mark_tainted.assert_not_called()
        app._recorder.stop.assert_not_called()

    def test_shutdown_rejects_new_operations(self):
        from wenzi.app import WenZiApp

        app = object.__new__(WenZiApp)
        app._shutdown_started = True
        app._op_guard = MagicMock()

        assert WenZiApp._try_begin_op(app, "recording") is None
        app._op_guard.try_begin.assert_not_called()

    @patch("wenzi.app.quit_application")
    @patch("os.kill")
    @patch("PyObjCTools.AppHelper.callAfter")
    @patch("atexit.register")
    @patch("signal.signal")
    @patch("faulthandler.enable")
    @patch("wenzi.app.WenZiApp")
    def test_main_registers_signal_and_atexit_restores(
        self,
        mock_app_class,
        _mock_faulthandler,
        mock_signal,
        mock_atexit,
        mock_call_after,
        mock_kill,
        mock_quit,
    ):
        from wenzi.app import main

        app = mock_app_class.return_value
        order: list[str] = []
        app._restore_system_output_on_exit.side_effect = (
            lambda: order.append("restore")
        )
        mock_quit.side_effect = lambda: order.append("quit")

        main()

        handlers = {entry.args[0]: entry.args[1] for entry in mock_signal.call_args_list}
        assert set(handlers) == {signal.SIGINT, signal.SIGTERM}
        mock_atexit.assert_called_once_with(app._restore_system_output_on_exit)
        app.run.assert_called_once_with()

        handlers[signal.SIGINT](signal.SIGINT, None)
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        mock_call_after.assert_called_once_with(app._on_quit_click, None)
        mock_signal.assert_any_call(signal.SIGTERM, signal.SIG_DFL)
        mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
        app._on_quit_click.assert_not_called()

        callback, arg = mock_call_after.call_args.args
        callback(arg)
        app._on_quit_click.assert_called_once_with(None)

        mock_atexit.call_args.args[0]()

        assert order == ["restore"]
        mock_quit.assert_not_called()
