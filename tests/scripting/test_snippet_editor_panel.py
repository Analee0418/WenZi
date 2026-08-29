"""Tests for the snippet editor panel keyboard handling."""

from __future__ import annotations

from unittest.mock import MagicMock

from wenzi.scripting.ui.snippet_editor_panel import SnippetEditorPanel


def _make_editor():
    editor = SnippetEditorPanel.__new__(SnippetEditorPanel)
    editor._panel = MagicMock()
    editor._panel.isKeyWindow.return_value = True
    editor._content_view = None
    editor._name_field = None
    editor._error_label = None
    editor._do_save = MagicMock()
    return editor


def _key_event(char: str, flags: int = 0):
    event = MagicMock()
    event.modifierFlags.return_value = flags
    event.charactersIgnoringModifiers.return_value = char
    return event


class TestSnippetEditorEnterHandling:
    def test_plain_enter_not_consumed(self):
        """Plain Enter must fall through (inserts a newline in Content)."""
        editor = _make_editor()
        event = _key_event("\r")

        assert editor._handle_key_event(event) is event
        editor._do_save.assert_not_called()

    def test_shift_enter_not_consumed(self):
        from AppKit import NSShiftKeyMask

        editor = _make_editor()
        event = _key_event("\r", NSShiftKeyMask)

        assert editor._handle_key_event(event) is event
        editor._do_save.assert_not_called()

    def test_cmd_enter_saves(self):
        from AppKit import NSCommandKeyMask

        editor = _make_editor()
        event = _key_event("\r", NSCommandKeyMask)

        assert editor._handle_key_event(event) is None
        editor._do_save.assert_called_once()
