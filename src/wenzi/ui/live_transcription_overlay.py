"""Floating overlay panel that displays partial transcription during recording."""

from __future__ import annotations

import logging
import weakref

logger = logging.getLogger(__name__)

# Panel dimensions
_PANEL_WIDTH = 390
_PANEL_MIN_HEIGHT = 70
_PANEL_MAX_HEIGHT = 190
_PADDING = 14
_HEADER_HEIGHT = 16
_HEADER_GAP = 6
_CORNER_RADIUS = 12
_SCREEN_Y_OFFSET = 80  # offset below center (below recording indicator)


# Module-level NSView subclass for drawRect_-based background
try:
    from AppKit import NSAppearanceNameAqua as _AQUA
    from AppKit import NSAppearanceNameDarkAqua as _DARK_AQUA
    from AppKit import NSBezierPath as _BP
    from AppKit import NSColor as _NC
    from AppKit import NSView as _NV

    def _make_dynamic_color(name, light_rgba, dark_rgba):
        def _provider(appearance):
            match_name = appearance.bestMatchFromAppearancesWithNames_([_AQUA, _DARK_AQUA])
            if match_name == _DARK_AQUA:
                return _NC.colorWithSRGBRed_green_blue_alpha_(*dark_rgba)
            return _NC.colorWithSRGBRed_green_blue_alpha_(*light_rgba)

        return _NC.colorWithName_dynamicProvider_(name, _provider)

    _BG_COLOR = _make_dynamic_color(
        "WenZiLiveBg",
        (0.965, 0.975, 0.98, 0.96),
        (0.075, 0.085, 0.095, 0.96),
    )
    _BORDER_COLOR = _make_dynamic_color(
        "WenZiLiveBorder",
        (0.04, 0.54, 0.60, 0.28),
        (0.30, 0.88, 0.88, 0.30),
    )
    _ACCENT_COLOR = _make_dynamic_color(
        "WenZiLiveAccent",
        (0.02, 0.55, 0.61, 0.82),
        (0.28, 0.88, 0.88, 0.88),
    )

    class _LiveBgView(_NV):
        def isOpaque(self):
            return False

        def drawRect_(self, rect):
            bounds = self.bounds()
            background = _BP.bezierPathWithRoundedRect_xRadius_yRadius_(
                bounds, _CORNER_RADIUS, _CORNER_RADIUS
            )
            _BG_COLOR.setFill()
            background.fill()

            _BORDER_COLOR.setStroke()
            background.setLineWidth_(1.0)
            background.stroke()

            accent = _BP.alloc().init()
            accent.moveToPoint_((1.5, 12.0))
            accent.lineToPoint_((1.5, max(12.0, bounds.size.height - 12.0)))
            accent.setLineWidth_(3.0)
            accent.setLineCapStyle_(1)
            _ACCENT_COLOR.setStroke()
            accent.stroke()

        def viewDidChangeEffectiveAppearance(self):
            # Appearance changes are event-driven; no background polling timer.
            self.setNeedsDisplay_(True)

except Exception:
    _LiveBgView = None


class LiveTranscriptionOverlay:
    """Non-interactive floating overlay that shows partial STT text during recording.

    Must be created, shown, updated, and hidden on the main thread
    (or via AppHelper.callAfter).
    """

    # Alpha value for the inactive (waiting-for-recording) state
    _INACTIVE_ALPHA = 0.62

    # Track all live instances for bulk cleanup (weak references to allow GC)
    _instances: weakref.WeakSet[LiveTranscriptionOverlay] = weakref.WeakSet()

    def __init__(self) -> None:
        self._panel: object = None
        self._text_field: object = None
        self._status_field: object = None
        self._content_view: object = None
        self._screen_center_y: float = 0  # cached for repositioning
        self._active: bool = True
        LiveTranscriptionOverlay._instances.add(self)

    _TEXT_COLOR = None

    @classmethod
    def _dynamic_text_color(cls):
        """Return a cached dynamic text color that contrasts with the background."""
        if cls._TEXT_COLOR is None:
            from AppKit import NSAppearanceNameAqua, NSAppearanceNameDarkAqua, NSColor

            def _provider(appearance):
                name = appearance.bestMatchFromAppearancesWithNames_(
                    [NSAppearanceNameAqua, NSAppearanceNameDarkAqua]
                )
                if name == NSAppearanceNameDarkAqua:
                    return NSColor.colorWithSRGBRed_green_blue_alpha_(0.95, 0.95, 0.95, 1.0)
                return NSColor.colorWithSRGBRed_green_blue_alpha_(0.1, 0.1, 0.1, 1.0)

            cls._TEXT_COLOR = NSColor.colorWithName_dynamicProvider_(
                "WenZiLiveText", _provider
            )
        return cls._TEXT_COLOR

    @staticmethod
    def _status_text(active: bool) -> str:
        from wenzi.i18n import t

        key = "live_overlay.listening" if active else "live_overlay.preparing"
        return f"\u25cf  {t(key)}"

    @staticmethod
    def _animate_alpha(panel, target: float, duration: float = 0.14) -> None:
        """Run a one-shot opacity transition without a repeating timer."""
        from AppKit import NSAnimationContext

        NSAnimationContext.beginGrouping()
        try:
            NSAnimationContext.currentContext().setDuration_(duration)
            panel.animator().setAlphaValue_(target)
        finally:
            NSAnimationContext.endGrouping()

    def _set_status(self, active: bool) -> None:
        if self._status_field is None:
            return
        from AppKit import NSColor

        self._status_field.setStringValue_(self._status_text(active))
        color = NSColor.systemTealColor() if active else NSColor.secondaryLabelColor()
        self._status_field.setTextColor_(color)

    def _layout_content(self, height: float) -> None:
        """Lay out the fixed header and the flexible transcript body."""
        from Foundation import NSMakeRect

        inner_width = _PANEL_WIDTH - 2 * _PADDING
        body_height = max(
            1.0,
            height - 2 * _PADDING - _HEADER_HEIGHT - _HEADER_GAP,
        )
        if self._text_field is not None:
            self._text_field.setFrame_(
                NSMakeRect(_PADDING, _PADDING, inner_width, body_height)
            )
        if self._status_field is not None:
            self._status_field.setFrame_(
                NSMakeRect(
                    _PADDING,
                    _PADDING + body_height + _HEADER_GAP,
                    inner_width,
                    _HEADER_HEIGHT,
                )
            )

    def show(self, active: bool = True) -> None:
        """Create and show the overlay panel. Must be called on the main thread.

        Args:
            active: If False, the panel is shown in a faded state (waiting for
                recording to start). Call ``set_active()`` to switch to full
                opacity later.
        """
        LiveTranscriptionOverlay._instances.add(self)
        try:
            from AppKit import (
                NSColor,
                NSFont,
                NSPanel,
                NSScreen,
                NSStatusWindowLevel,
                NSTextField,
            )
            from Foundation import NSMakeRect

            if self._panel is not None:
                self.hide()

            init_h = _PANEL_MIN_HEIGHT
            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, _PANEL_WIDTH, init_h),
                0,  # NSBorderlessWindowMask
                2,  # NSBackingStoreBuffered
                False,
            )
            panel.setLevel_(NSStatusWindowLevel + 1)
            panel.setOpaque_(False)
            panel.setBackgroundColor_(NSColor.clearColor())
            panel.setIgnoresMouseEvents_(True)
            panel.setHasShadow_(True)
            panel.setHidesOnDeactivate_(False)
            panel.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 8))  # canJoinAllSpaces | stationary | fullScreenAuxiliary
            self._panel = panel

            # Content view with drawRect_-based rounded background
            content = _LiveBgView.alloc().initWithFrame_(
                NSMakeRect(0, 0, _PANEL_WIDTH, init_h)
            )

            # Compact state header. It only changes on the existing
            # preparing -> listening transition.
            status = NSTextField.labelWithString_(self._status_text(active))
            status.setFont_(NSFont.systemFontOfSize_weight_(10.5, 0.23))
            status.setAlignment_(0)  # NSTextAlignmentLeft
            content.addSubview_(status)

            # Text field for partial transcription (no line limit, wraps freely)
            tf = NSTextField.wrappingLabelWithString_("")
            tf.setFont_(NSFont.systemFontOfSize_(15.0))
            tf.setTextColor_(self._dynamic_text_color())
            tf.setMaximumNumberOfLines_(0)  # unlimited lines
            tf.setAlignment_(0)  # NSTextAlignmentLeft
            content.addSubview_(tf)
            self._text_field = tf
            self._status_field = status
            self._content_view = content
            self._layout_content(init_h)
            self._set_status(active)

            panel.setContentView_(content)

            # Position at screen center, offset below the recording indicator
            screen = NSScreen.mainScreen()
            if screen:
                sf = screen.visibleFrame()
                x = sf.origin.x + (sf.size.width - _PANEL_WIDTH) / 2
                self._screen_center_y = (
                    sf.origin.y + (sf.size.height - init_h) / 2 - _SCREEN_Y_OFFSET
                )
                panel.setFrameOrigin_((x, self._screen_center_y))

            self._active = active
            panel.setAlphaValue_(0.0)
            panel.orderFront_(None)
            target_alpha = 1.0 if active else self._INACTIVE_ALPHA
            self._animate_alpha(panel, target_alpha)

            logger.debug("Live transcription overlay shown")
        except Exception:
            logger.error("Failed to show live transcription overlay", exc_info=True)
            self.hide()

    def set_active(self) -> None:
        """Switch the overlay from faded to full opacity.

        Must be called on the main thread.
        """
        if self._active:
            return
        self._active = True
        self._set_status(True)
        if self._panel is not None:
            self._animate_alpha(self._panel, 1.0)

    def hide(self) -> None:
        """Hide and clean up the overlay panel. Must be called on the main thread."""
        try:
            if self._panel is not None:
                self._panel.orderOut_(None)
                self._panel = None
            self._text_field = None
            self._status_field = None
            self._content_view = None
            LiveTranscriptionOverlay._instances.discard(self)
            logger.debug("Live transcription overlay hidden")
        except Exception as e:
            logger.warning("Failed to hide live transcription overlay: %s", e)

    def update_text(self, text: str) -> None:
        """Update the displayed partial transcription text. Must be called on the main thread."""
        if self._text_field is None or self._panel is None:
            return

        self._text_field.setStringValue_(text)
        self._resize_panel()

    def _resize_panel(self) -> None:
        """Resize the panel to fit the current text content, up to _PANEL_MAX_HEIGHT."""
        from Foundation import NSMakeRect

        tf = self._text_field
        panel = self._panel
        content = self._content_view
        if tf is None or panel is None or content is None:
            return

        # Calculate the height the text needs at the available width
        inner_width = _PANEL_WIDTH - 2 * _PADDING
        # cellSizeForBounds_ returns the size needed to render the text
        needed = tf.cell().cellSizeForBounds_(
            NSMakeRect(0, 0, inner_width, 10000)
        )
        text_h = needed.height
        chrome_h = 2 * _PADDING + _HEADER_HEIGHT + _HEADER_GAP
        new_h = min(
            max(text_h + chrome_h, _PANEL_MIN_HEIGHT),
            _PANEL_MAX_HEIGHT,
        )

        old_frame = panel.frame()
        if abs(old_frame.size.height - new_h) < 1:
            return  # no meaningful change

        # Grow upward: keep the top edge stable by adjusting y origin
        new_y = old_frame.origin.y + old_frame.size.height - new_h
        panel.setFrame_display_(
            NSMakeRect(old_frame.origin.x, new_y, _PANEL_WIDTH, new_h), True
        )
        content.setFrame_(NSMakeRect(0, 0, _PANEL_WIDTH, new_h))
        self._layout_content(new_h)

    def close(self) -> None:
        """Alias for hide() for consistency with other panels."""
        self.hide()
        LiveTranscriptionOverlay._instances.discard(self)

    @classmethod
    def close_all(cls) -> None:
        """Close every live overlay instance. Must be called on the main thread."""
        for inst in list(cls._instances):
            inst.close()
