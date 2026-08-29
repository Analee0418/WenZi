"""Lightweight, audio-reactive recording orb for macOS."""

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)

# Asymmetric EMA smoothing: speech should light the orb quickly and decay softly.
_EMA_ATTACK = 0.7
_EMA_RELEASE = 0.25

# The transparent host is wider than the orb so the lower waveform can taper.
# Its visual center remains close to the panel center for preview transitions.
_PANEL_WIDTH = 220
_PANEL_HEIGHT = 208
_PANEL_CENTER_X = _PANEL_WIDTH / 2.0
_PANEL_CENTER_Y = _PANEL_HEIGHT / 2.0
_ORB_CENTER_X = _PANEL_CENTER_X
_ORB_CENTER_Y = 115.0
_ORB_RADIUS_BASE = 50.0
_ORB_BREATH_GAIN = 1.6
_OUTLINE_POINT_COUNT = 24
_OUTLINE_ANGLES = tuple(math.tau * index / _OUTLINE_POINT_COUNT for index in range(_OUTLINE_POINT_COUNT))
_PULL_DIRECTIONS = tuple(math.radians(angle) for angle in (80.0, 205.0, -15.0))
_PULL_WIDTHS = tuple(math.radians(width) for width in (32.0, 25.0, 29.0))
_PULL_GAINS = (5.0, 8.0, 6.3)
_PULL_SPEEDS = (0.43, 0.36, 0.51)
_PULL_PHASES = (0.2, 2.2, 4.1)
_PULL_DRIFT = math.radians(10.0)
_VOICE_FLOOR = 0.22
_VOICE_FULL = 0.55
_ORB_LEVEL_SCALE_GAIN = 0.055
_RIM_OUTER_WIDTH = 10.0
_RIM_OUTER_LEVEL_GAIN = 4.0
_RIM_MIDDLE_WIDTH = 4.2
_RIM_MIDDLE_LEVEL_GAIN = 1.6
_RIM_INNER_WIDTH = 1.35
_RIM_INNER_LEVEL_GAIN = 0.55
_HALO_BASE_RADIUS = 72.0
_HALO_LEVEL_GAIN = 11.0
_HALO_BREATH_GAIN = 4.0
_CORONA_POINT_COUNT = 20
_CORONA_ANGLES = tuple(math.tau * index / _CORONA_POINT_COUNT for index in range(_CORONA_POINT_COUNT))
_CORONA_BASE_RADII = (66.0, 71.0, 76.0)
_CORONA_IDLE_AMPLITUDES = (1.8, 1.5, 1.2)
_CORONA_LEVEL_AMPLITUDES = (7.0, 6.1, 5.25)
_CORONA_PHASES = (0.2, 2.3, 4.7)
_CORONA_SPEEDS = (1.35, -1.15, 1.55)
_CORONA_STROKE_WIDTHS = (
    (7.4, 3.8, 1.45),
    (6.8, 3.5, 1.33),
    (6.2, 3.2, 1.21),
)
_WAVE_POINT_COUNT = 24
_WAVE_POSITIONS = tuple(index / (_WAVE_POINT_COUNT - 1) for index in range(_WAVE_POINT_COUNT))
_WAVE_X_MIN = 15.0
_WAVE_X_MAX = _PANEL_WIDTH - _WAVE_X_MIN
_WAVE_BASE_Y = 26.0
_WAVE_OFFSETS = (-3.0, 0.0, 3.0)
_WAVE_IDLE_AMPLITUDES = (2.2, 2.8, 1.9)
_WAVE_LEVEL_AMPLITUDES = (12.0, 14.5, 11.5)
_WAVE_PHASES = (0.1, 2.2, 4.1)
_WAVE_SPEEDS = (1.45, -1.20, 1.70)
_WAVE_STROKE_WIDTHS = (
    (7.0, 5.2, 1.65),
    (6.65, 4.9, 1.53),
    (6.3, 4.6, 1.41),
)
_WAVE_CLIP_HEIGHT = 55.0

# One coalescible 20 Hz timer drives both the idle breath and audio response.
_REFRESH_INTERVAL = 0.05
_ENTRY_DURATION = 0.24
_EXIT_DURATION = 0.20
_ENTRY_INITIAL_ALPHA = 0.62


def _halo_radius(level: float, breathe: float) -> float:
    """Return a bounded halo radius for normalized audio activity."""
    pull = _gravity_pull(level)
    breath = max(0.0, min(1.0, breathe))
    return _HALO_BASE_RADIUS + _HALO_LEVEL_GAIN * pull + _HALO_BREATH_GAIN * breath


def _gravity_pull(level: float) -> float:
    """Separate room noise from speech with a smooth, bounded response."""
    activity = max(0.0, min(1.0, level))
    normalized = max(
        0.0,
        min(1.0, (activity - _VOICE_FLOOR) / (_VOICE_FULL - _VOICE_FLOOR)),
    )
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _orb_scale(level: float) -> float:
    """Make active speech visibly expand the whole orb."""
    return 1.0 + _ORB_LEVEL_SCALE_GAIN * _gravity_pull(level)


def _angular_distance(first: float, second: float) -> float:
    """Return the shortest signed distance between two angles."""
    return (first - second + math.pi) % math.tau - math.pi


def _orb_outline_points(
    elapsed: float,
    activity: float,
    breathe: float,
) -> tuple[tuple[float, float], ...]:
    """Return a smoothable, audio-reactive polar outline around the orb."""
    normalized = max(0.0, min(1.0, activity))
    pull = _gravity_pull(normalized)
    breath = max(0.0, min(1.0, breathe))
    directions = tuple(
        direction + math.sin(elapsed * speed + phase) * _PULL_DRIFT
        for direction, speed, phase in zip(
            _PULL_DIRECTIONS,
            _PULL_SPEEDS,
            _PULL_PHASES,
            strict=True,
        )
    )
    base_radius = _ORB_RADIUS_BASE + breath * _ORB_BREATH_GAIN + pull * 2.8

    points = []
    for angle in _OUTLINE_ANGLES:
        directional_pull = sum(
            gain * math.exp(-0.5 * (_angular_distance(angle, direction) / width) ** 2)
            for direction, width, gain in zip(
                directions,
                _PULL_WIDTHS,
                _PULL_GAINS,
                strict=True,
            )
        )
        irregularity = 1.15 * math.sin(angle * 2.0 + elapsed * 0.83) + 0.70 * math.sin(angle * 5.0 - elapsed * 0.57)
        radius = base_radius + pull * (directional_pull + irregularity)
        points.append(
            (
                _ORB_CENTER_X + math.cos(angle) * radius,
                _ORB_CENTER_Y + math.sin(angle) * radius,
            )
        )
    return tuple(points)


def _corona_points(
    index: int,
    elapsed: float,
    activity: float,
) -> tuple[tuple[float, float], ...]:
    """Return one organic energy ring around the orb."""
    pull = _gravity_pull(activity)
    phase = _CORONA_PHASES[index]
    motion = elapsed * _CORONA_SPEEDS[index] + phase
    amplitude = _CORONA_IDLE_AMPLITUDES[index] + pull * _CORONA_LEVEL_AMPLITUDES[index]
    points = []
    for angle in _CORONA_ANGLES:
        ripple = (
            0.56 * math.sin(angle * 3.0 + motion)
            + 0.29 * math.sin(angle * 7.0 - motion * 0.73)
            + 0.15 * math.sin(angle * 11.0 + motion * 0.41)
        )
        flame = (
            max(
                0.0,
                math.sin(angle * (5.0 + index) - motion * 1.17),
            )
            ** 3
        )
        radius = _CORONA_BASE_RADII[index] + amplitude * ripple + pull * (4.2 - index * 0.5) * flame
        points.append(
            (
                _ORB_CENTER_X + math.cos(angle) * radius,
                _ORB_CENTER_Y + math.sin(angle) * radius,
            )
        )
    return tuple(points)


def _wave_points(
    index: int,
    elapsed: float,
    activity: float,
) -> tuple[tuple[float, float], ...]:
    """Return one tapered lower waveform driven by the voice envelope."""
    pull = _gravity_pull(activity)
    phase = _WAVE_PHASES[index]
    motion = elapsed * _WAVE_SPEEDS[index] + phase
    amplitude = _WAVE_IDLE_AMPLITUDES[index] + pull * _WAVE_LEVEL_AMPLITUDES[index]
    points = []
    for position in _WAVE_POSITIONS:
        envelope = math.sin(math.pi * position) ** 1.7
        carrier = 0.68 * math.sin(math.tau * position * (1.35 + index * 0.16) + motion) + 0.32 * math.sin(
            math.tau * position * (2.65 - index * 0.12) - motion * 0.71
        )
        points.append(
            (
                _WAVE_X_MIN + (_WAVE_X_MAX - _WAVE_X_MIN) * position,
                _WAVE_BASE_Y + _WAVE_OFFSETS[index] + envelope * amplitude * carrier,
            )
        )
    return tuple(points)


def _rim_widths(level: float) -> tuple[float, float, float]:
    """Return cached-color stroke widths for the current audio activity."""
    pull = _gravity_pull(level)
    return (
        _RIM_OUTER_WIDTH + pull * _RIM_OUTER_LEVEL_GAIN,
        _RIM_MIDDLE_WIDTH + pull * _RIM_MIDDLE_LEVEL_GAIN,
        _RIM_INNER_WIDTH + pull * _RIM_INNER_LEVEL_GAIN,
    )


def _entry_scale(elapsed: float) -> float:
    """Ease from a visible compact orb into a subtle overshoot."""
    progress = max(0.0, min(1.0, elapsed / _ENTRY_DURATION))
    shifted = progress - 1.0
    eased = 1.0 + 2.2 * shifted**3 + 1.2 * shifted**2
    return 0.78 + 0.22 * eased


def _exit_scale(elapsed: float) -> float:
    """Shrink smoothly while the native panel fades out."""
    progress = max(0.0, min(1.0, elapsed / _EXIT_DURATION))
    eased = progress * progress
    return 1.0 - 0.24 * eased


class RecordingIndicatorView:
    """Draw a compact voice orb whose glow follows microphone activity."""

    _view: object = None

    def __init__(self) -> None:
        self._level = 0.0
        self._start_time = 0.0
        self._view = None
        self._recording_active = False
        self._exit_started_at: float | None = None

        # Native colors, gradients, and the mutable path are prewarmed once.
        self._halo_gradient = None
        self._orb_body_gradient = None
        self._orb_accent_gradients = None
        self._orb_depth_gradient = None
        self._orb_bloom_gradient = None
        self._orb_path = None
        self._rim_colors = None
        self._corona_paths = None
        self._corona_colors = None
        self._wave_paths = None
        self._wave_colors = None
        self._wave_edge_gradient = None

    def create_view(self, width: int, height: int) -> object:
        """Create and return the transparent NSView host."""
        from Foundation import NSMakeRect

        view = _IndicatorNSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        view._indicator = self
        self._view = view
        if self._start_time == 0.0:
            self._start_time = time.monotonic()
        return view

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))

    def prepare_for_display(self) -> None:
        """Create all drawing resources before the panel becomes visible."""
        self._ensure_active_colors()

    def begin_exit(self) -> None:
        """Start the exit scale animation on the existing refresh timer."""
        if self._exit_started_at is None:
            self._exit_started_at = time.monotonic()

    def _ensure_active_colors(self) -> None:
        if self._orb_body_gradient is not None:
            return

        from AppKit import NSBezierPath, NSColor, NSColorSpace, NSGradient

        def _radial_gradient(
            red: float,
            green: float,
            blue: float,
            alpha: float,
            middle_alpha: float,
        ):
            start = NSColor.colorWithSRGBRed_green_blue_alpha_(red, green, blue, alpha)
            middle = NSColor.colorWithSRGBRed_green_blue_alpha_(red, green, blue, middle_alpha)
            end = NSColor.colorWithSRGBRed_green_blue_alpha_(red, green, blue, 0.0)
            return NSGradient.alloc().initWithColors_([start, middle, end])

        halo_gradient = NSGradient.alloc().initWithColors_atLocations_colorSpace_(
            [
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.18, 0.78, 1.00, 0.22),
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.42, 0.24, 1.00, 0.14),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.12, 0.68, 0.055),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.12, 0.68, 0.0),
            ],
            (0.0, 0.38, 0.72, 1.0),
            NSColorSpace.sRGBColorSpace(),
        )
        body_gradient = NSGradient.alloc().initWithColors_(
            [
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.18, 0.08, 0.72, 0.96),
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.32, 0.16, 1.00, 0.95),
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.22, 0.34, 0.98, 0.92),
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.12, 0.78, 0.96, 0.84),
            ]
        )
        accent_gradients = (
            _radial_gradient(0.28, 0.98, 1.00, 0.42, 0.16),
            _radial_gradient(0.86, 0.16, 1.00, 0.30, 0.10),
        )
        depth_gradient = _radial_gradient(0.015, 0.025, 0.16, 0.58, 0.20)
        bloom_gradient = _radial_gradient(0.96, 0.99, 1.00, 0.52, 0.12)
        rim_colors = (
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.20, 1.00, 1.00, 0.16),
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.36, 1.00, 1.00, 0.52),
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.88, 1.00, 1.00, 0.92),
        )
        orb_path = NSBezierPath.alloc().init()
        orb_path.setLineJoinStyle_(1)
        corona_paths = tuple(NSBezierPath.alloc().init() for _ in _CORONA_BASE_RADII)
        wave_paths = tuple(NSBezierPath.alloc().init() for _ in _WAVE_OFFSETS)
        for path in (*corona_paths, *wave_paths):
            path.setLineJoinStyle_(1)
            path.setLineCapStyle_(1)
        corona_colors = (
            (
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.72, 0.20, 0.08),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.78, 0.34, 0.26),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.91, 0.64, 0.74),
            ),
            (
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.18, 0.38, 0.07),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.27, 0.45, 0.23),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.53, 0.62, 0.66),
            ),
            (
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.08, 0.62, 0.06),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.16, 0.70, 0.20),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.48, 0.82, 0.60),
            ),
        )
        wave_colors = (
            (
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.12, 0.95, 1.00, 0.06),
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.12, 0.95, 1.00, 0.22),
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.40, 1.00, 1.00, 0.86),
            ),
            (
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.36, 0.18, 1.00, 0.055),
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.36, 0.18, 1.00, 0.20),
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.60, 0.42, 1.00, 0.82),
            ),
            (
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.12, 0.78, 0.05),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.12, 0.78, 0.18),
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.00, 0.40, 0.88, 0.76),
            ),
        )
        edge_mask_colors = [NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, alpha) for alpha in (0.0, 1.0, 1.0, 0.0)]
        wave_edge_gradient = NSGradient.alloc().initWithColors_atLocations_colorSpace_(
            edge_mask_colors,
            (0.0, 0.16, 0.84, 1.0),
            NSColorSpace.sRGBColorSpace(),
        )

        # Publish the cache only after every native resource exists. A failed
        # prewarm can then retry cleanly instead of retaining a partial palette.
        self._halo_gradient = halo_gradient
        self._orb_accent_gradients = accent_gradients
        self._orb_depth_gradient = depth_gradient
        self._orb_bloom_gradient = bloom_gradient
        self._orb_path = orb_path
        self._rim_colors = rim_colors
        self._corona_paths = corona_paths
        self._corona_colors = corona_colors
        self._wave_paths = wave_paths
        self._wave_colors = wave_colors
        self._wave_edge_gradient = wave_edge_gradient
        self._orb_body_gradient = body_gradient

    def _rebuild_orb_path(
        self,
        points: tuple[tuple[float, float], ...],
    ) -> None:
        """Rebuild the cached path as a closed Catmull-Rom curve."""
        self._rebuild_closed_path(self._orb_path, points)

    @staticmethod
    def _rebuild_closed_path(
        path: object,
        points: tuple[tuple[float, float], ...],
    ) -> None:
        """Rebuild a cached path as a closed Catmull-Rom curve."""
        path.removeAllPoints()
        path.moveToPoint_(points[0])
        point_count = len(points)
        for index in range(point_count):
            previous = points[(index - 1) % point_count]
            current = points[index]
            following = points[(index + 1) % point_count]
            after_following = points[(index + 2) % point_count]
            control_one = (
                current[0] + (following[0] - previous[0]) / 6.0,
                current[1] + (following[1] - previous[1]) / 6.0,
            )
            control_two = (
                following[0] - (after_following[0] - current[0]) / 6.0,
                following[1] - (after_following[1] - current[1]) / 6.0,
            )
            path.curveToPoint_controlPoint1_controlPoint2_(
                following,
                control_one,
                control_two,
            )
        path.closePath()

    @staticmethod
    def _rebuild_open_path(
        path: object,
        points: tuple[tuple[float, float], ...],
    ) -> None:
        """Rebuild a cached path as an open Catmull-Rom curve."""
        path.removeAllPoints()
        path.moveToPoint_(points[0])
        last_index = len(points) - 1
        for index in range(last_index):
            previous = points[max(0, index - 1)]
            current = points[index]
            following = points[index + 1]
            after_following = points[min(last_index, index + 2)]
            control_one = (
                current[0] + (following[0] - previous[0]) / 6.0,
                current[1] + (following[1] - previous[1]) / 6.0,
            )
            control_two = (
                following[0] - (after_following[0] - current[0]) / 6.0,
                following[1] - (after_following[1] - current[1]) / 6.0,
            )
            path.curveToPoint_controlPoint1_controlPoint2_(
                following,
                control_one,
                control_two,
            )

    def _draw_bottom_waves(self, elapsed: float, activity: float) -> None:
        """Draw three cached waveforms with cheap layered glow and edge fade."""
        from AppKit import (
            NSCompositingOperationDestinationIn,
            NSGradientDrawsAfterEndingLocation,
            NSGradientDrawsBeforeStartingLocation,
            NSGraphicsContext,
            NSRectClip,
        )
        from Foundation import NSMakePoint, NSMakeRect

        NSGraphicsContext.saveGraphicsState()
        try:
            NSRectClip(NSMakeRect(0.0, 0.0, _PANEL_WIDTH, _WAVE_CLIP_HEIGHT))
            for index, path in enumerate(self._wave_paths):
                self._rebuild_open_path(
                    path,
                    _wave_points(index, elapsed, activity),
                )
                for color, width in zip(
                    self._wave_colors[index],
                    _WAVE_STROKE_WIDTHS[index],
                    strict=True,
                ):
                    color.setStroke()
                    path.setLineWidth_(width)
                    path.stroke()

            context = NSGraphicsContext.currentContext()
            context.setCompositingOperation_(NSCompositingOperationDestinationIn)
            self._wave_edge_gradient.drawFromPoint_toPoint_options_(
                NSMakePoint(_WAVE_X_MIN, _WAVE_BASE_Y),
                NSMakePoint(_WAVE_X_MAX, _WAVE_BASE_Y),
                NSGradientDrawsBeforeStartingLocation | NSGradientDrawsAfterEndingLocation,
            )
        finally:
            NSGraphicsContext.restoreGraphicsState()

    def _draw_corona(self, elapsed: float, activity: float) -> None:
        """Draw layered audio-reactive ripples around the orb."""
        from AppKit import NSGraphicsContext

        NSGraphicsContext.saveGraphicsState()
        try:
            for index in reversed(range(len(self._corona_paths))):
                path = self._corona_paths[index]
                self._rebuild_closed_path(
                    path,
                    _corona_points(index, elapsed, activity),
                )
            # Paint every broad glow before any bright core so neighboring
            # rings cannot wash out one another at their intersections.
            for layer_index in range(3):
                for index in reversed(range(len(self._corona_paths))):
                    path = self._corona_paths[index]
                    color = self._corona_colors[index][layer_index]
                    width = _CORONA_STROKE_WIDTHS[index][layer_index]
                    color.setStroke()
                    path.setLineWidth_(width)
                    path.stroke()
        finally:
            NSGraphicsContext.restoreGraphicsState()

    def _draw_active_orb(self, elapsed: float, breathe: float) -> None:
        from AppKit import NSAffineTransform, NSGraphicsContext
        from Foundation import NSMakePoint

        self._ensure_active_colors()
        # The full color palette is visible on the first frame. Audio only
        # starts influencing motion after the recorder confirms it is active.
        activity = self._level if self._recording_active else 0.0
        self._draw_bottom_waves(elapsed, activity)
        orb_center = NSMakePoint(_ORB_CENTER_X, _ORB_CENTER_Y)
        halo_radius = _halo_radius(activity, breathe)
        self._halo_gradient.drawFromCenter_radius_toCenter_radius_options_(orb_center, 0.0, orb_center, halo_radius, 0)
        self._draw_corona(elapsed, activity)

        NSGraphicsContext.saveGraphicsState()
        try:
            voice_transform = NSAffineTransform.transform()
            voice_transform.translateXBy_yBy_(_ORB_CENTER_X, _ORB_CENTER_Y)
            voice_transform.scaleBy_(_orb_scale(activity))
            voice_transform.translateXBy_yBy_(-_ORB_CENTER_X, -_ORB_CENTER_Y)
            voice_transform.concat()
            self._draw_orb_body(elapsed, breathe, activity, orb_center)
        finally:
            NSGraphicsContext.restoreGraphicsState()

    def _draw_orb_body(
        self,
        elapsed: float,
        breathe: float,
        activity: float,
        orb_center: object,
    ) -> None:
        """Draw the clipped glass body and its cached luminous rim."""
        from AppKit import NSGraphicsContext
        from Foundation import NSMakePoint

        points = _orb_outline_points(elapsed, activity, breathe)
        self._rebuild_orb_path(points)
        path = self._orb_path

        # The vivid body is clipped to the same path later used for the rim.
        # Layered strokes emulate glow without a real-time blur filter.
        NSGraphicsContext.saveGraphicsState()
        try:
            path.addClip()
            body_start = NSMakePoint(
                _ORB_CENTER_X - 8.0,
                _ORB_CENTER_Y + 11.0,
            )
            self._orb_body_gradient.drawFromCenter_radius_toCenter_radius_options_(
                body_start,
                0.0,
                orb_center,
                _ORB_RADIUS_BASE + 10.0,
                0,
            )
            for gradient, center, radius in (
                (
                    self._orb_accent_gradients[0],
                    NSMakePoint(_ORB_CENTER_X - 8.0, _ORB_CENTER_Y - 31.0),
                    32.0,
                ),
                (
                    self._orb_accent_gradients[1],
                    NSMakePoint(_ORB_CENTER_X + 25.0, _ORB_CENTER_Y + 10.0),
                    32.0,
                ),
            ):
                gradient.drawFromCenter_radius_toCenter_radius_options_(center, 0.0, center, radius, 0)
            depth_center = NSMakePoint(
                _ORB_CENTER_X + 22.0,
                _ORB_CENTER_Y - 18.0,
            )
            self._orb_depth_gradient.drawFromCenter_radius_toCenter_radius_options_(depth_center, 0.0, depth_center, 39.0, 0)
            bloom_center = NSMakePoint(
                _ORB_CENTER_X - 13.0,
                _ORB_CENTER_Y + 17.0,
            )
            bloom_radius = 13.0 + 4.0 * _gravity_pull(activity)
            self._orb_bloom_gradient.drawFromCenter_radius_toCenter_radius_options_(bloom_center, 0.0, bloom_center, bloom_radius, 0)
            glint_center = NSMakePoint(
                _ORB_CENTER_X - 23.0,
                _ORB_CENTER_Y + 27.0,
            )
            self._orb_bloom_gradient.drawFromCenter_radius_toCenter_radius_options_(glint_center, 0.0, glint_center, 7.5, 0)
        finally:
            NSGraphicsContext.restoreGraphicsState()

        NSGraphicsContext.saveGraphicsState()
        try:
            for color, width in zip(
                self._rim_colors,
                _rim_widths(activity),
                strict=True,
            ):
                color.setStroke()
                path.setLineWidth_(width)
                path.stroke()
        finally:
            NSGraphicsContext.restoreGraphicsState()

    def draw(self, rect: object) -> None:
        """Draw only the orb and its transparent, audio-reactive glow."""
        from AppKit import NSAffineTransform, NSGraphicsContext

        now = time.monotonic()
        elapsed = max(0.0, now - self._start_time)
        breathe = 0.5 + 0.5 * math.sin(elapsed * 2.4)
        if self._exit_started_at is None:
            scale = _entry_scale(elapsed)
        else:
            scale = _exit_scale(max(0.0, now - self._exit_started_at))

        NSGraphicsContext.saveGraphicsState()
        try:
            transform = NSAffineTransform.transform()
            transform.translateXBy_yBy_(_PANEL_CENTER_X, _PANEL_CENTER_Y)
            transform.scaleBy_(scale)
            transform.translateXBy_yBy_(-_PANEL_CENTER_X, -_PANEL_CENTER_Y)
            transform.concat()

            self._draw_active_orb(elapsed, breathe)
        finally:
            NSGraphicsContext.restoreGraphicsState()


try:
    import objc
    from AppKit import NSView

    class _IndicatorNSView(NSView):
        _indicator = objc.ivar()

        def drawRect_(self, rect):
            if self._indicator:
                self._indicator.draw(rect)

        def isOpaque(self):
            return False

        def refresh_(self, timer):
            self.setNeedsDisplay_(True)

except Exception:
    _IndicatorNSView = None


class RecordingIndicatorPanel:
    """Own the transparent native host for the centered recording orb."""

    def __init__(self) -> None:
        self._panel: object = None
        self._timer: object = None
        self._indicator_view: RecordingIndicatorView | None = None
        self._smoothed_level = 0.0
        self._enabled = True
        self._show_device_name = False
        # Kept for caller compatibility; the orb intentionally displays no text.
        self._mode_name: str | None = None
        self._device_name: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            self.hide()

    @property
    def show_device_name(self) -> bool:
        return self._show_device_name

    @show_device_name.setter
    def show_device_name(self, value: bool) -> None:
        self._show_device_name = value

    @staticmethod
    def _animate_alpha(panel, target: float, duration: float = 0.14) -> None:
        """Run a one-shot opacity transition without another timer."""
        from AppKit import NSAnimationContext

        NSAnimationContext.beginGrouping()
        try:
            NSAnimationContext.currentContext().setDuration_(duration)
            panel.animator().setAlphaValue_(target)
        finally:
            NSAnimationContext.endGrouping()

    @staticmethod
    def _configure_refresh_timer(timer) -> None:
        """Allow macOS to coalesce redraw wakeups where possible."""
        try:
            timer.setTolerance_(0.01)
        except Exception:
            logger.debug("NSTimer tolerance is unavailable", exc_info=True)

    def show(
        self,
        device_name: str | None = None,
        mode_name: str | None = None,
    ) -> None:
        """Show a centered orb; legacy text arguments are stored but not drawn."""
        if not self._enabled:
            return

        try:
            from AppKit import NSColor, NSPanel, NSScreen, NSStatusWindowLevel
            from Foundation import NSMakeRect, NSTimer

            if self._panel is not None:
                self.hide()

            self._smoothed_level = 0.0
            self._mode_name = mode_name
            self._device_name = device_name
            self._indicator_view = RecordingIndicatorView()
            self._indicator_view.prepare_for_display()

            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, _PANEL_WIDTH, _PANEL_HEIGHT),
                0,
                2,
                False,
            )
            panel.setLevel_(NSStatusWindowLevel + 1)
            panel.setOpaque_(False)
            panel.setBackgroundColor_(NSColor.clearColor())
            panel.setIgnoresMouseEvents_(True)
            panel.setHasShadow_(False)
            panel.setHidesOnDeactivate_(False)
            panel.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 8))
            self._panel = panel

            screen = NSScreen.mainScreen()
            if screen:
                frame = screen.visibleFrame()
                x = frame.origin.x + (frame.size.width - _PANEL_WIDTH) / 2.0
                y = frame.origin.y + (frame.size.height - _PANEL_HEIGHT) / 2.0
                panel.setFrameOrigin_((x, y))

            indicator = self._indicator_view.create_view(_PANEL_WIDTH, _PANEL_HEIGHT)
            panel.setContentView_(indicator)
            # Attach the window invisibly so AppKit creates a backing store,
            # then synchronously rasterize the colored first frame before reveal.
            panel.setAlphaValue_(0.0)
            panel.orderFront_(None)
            indicator.setNeedsDisplay_(True)
            indicator.displayIfNeededIgnoringOpacity()
            panel.setAlphaValue_(_ENTRY_INITIAL_ALPHA)
            self._animate_alpha(panel, 1.0, duration=0.18)

            self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                _REFRESH_INTERVAL,
                indicator,
                b"refresh:",
                None,
                True,
            )
            self._configure_refresh_timer(self._timer)
            logger.debug("Recording orb shown")
        except Exception:
            logger.error("Failed to show recording orb", exc_info=True)
            self.hide()

    def set_recording_active(self) -> None:
        """Allow microphone activity to influence the already-visible orb."""
        if self._indicator_view is not None:
            self._indicator_view._recording_active = True
            if self._indicator_view._view is not None:
                self._indicator_view._view.setNeedsDisplay_(True)

    @staticmethod
    def _clear_view_backref(indicator_view: RecordingIndicatorView | None) -> None:
        view = indicator_view._view if indicator_view else None
        if view is not None:
            try:
                view._indicator = None
            except Exception:
                logger.debug("Failed to clear recording view back-reference", exc_info=True)

    def hide(self) -> None:
        """Hide the orb and release its only repeating timer."""
        timer = self._timer
        panel = self._panel
        indicator_view = self._indicator_view

        # Detach ownership first: native cleanup failures must not leave stale
        # state that can overwrite or interfere with the next recording.
        self._timer = None
        self._panel = None
        self._indicator_view = None
        self._smoothed_level = 0.0
        self._mode_name = None
        self._device_name = None
        self._clear_view_backref(indicator_view)

        if timer is not None:
            try:
                timer.invalidate()
            except Exception:
                logger.warning("Failed to invalidate recording orb timer", exc_info=True)
        if panel is not None:
            try:
                panel.orderOut_(None)
            except Exception:
                logger.warning("Failed to close recording orb panel", exc_info=True)
        logger.debug("Recording orb hidden")

    def update_mode(self, name: str) -> None:
        """Remember the active mode without adding text around the orb."""
        self._mode_name = name

    def clear_mode(self) -> None:
        """Clear the compatibility-only active mode state."""
        self._mode_name = None

    def update_device_name(self, device_name: str) -> None:
        """Remember the input name without adding text around the orb."""
        if self._panel is not None:
            self._device_name = device_name

    @property
    def current_frame(self) -> object | None:
        """Return the orb host frame, or None while it is hidden."""
        return self._panel.frame() if self._panel is not None else None

    def animate_out(self, completion: callable = None) -> None:
        """Shrink and fade out the orb, then clean up and call completion."""
        if self._panel is None:
            if completion:
                completion()
            return

        try:
            from AppKit import NSAnimationContext

            panel = self._panel
            indicator_view = self._indicator_view
            if indicator_view is not None:
                indicator_view.begin_exit()
                if indicator_view._view is not None:
                    indicator_view._view.setNeedsDisplay_(True)

            def _on_complete():
                if self._panel is not panel or self._indicator_view is not indicator_view:
                    return

                timer = self._timer
                self._timer = None
                self._panel = None
                self._indicator_view = None
                self._smoothed_level = 0.0
                self._mode_name = None
                self._device_name = None
                self._clear_view_backref(indicator_view)
                if timer is not None:
                    try:
                        timer.invalidate()
                    except Exception:
                        logger.warning(
                            "Failed to invalidate recording orb timer",
                            exc_info=True,
                        )
                try:
                    panel.orderOut_(None)
                except Exception:
                    logger.warning(
                        "Failed to close recording orb panel",
                        exc_info=True,
                    )
                logger.debug("Recording orb animated out")
                if completion:
                    completion()

            NSAnimationContext.beginGrouping()
            try:
                context = NSAnimationContext.currentContext()
                context.setDuration_(_EXIT_DURATION)
                context.setCompletionHandler_(_on_complete)
                panel.animator().setAlphaValue_(0.0)
            finally:
                NSAnimationContext.endGrouping()
        except Exception:
            logger.error("Failed to animate recording orb out", exc_info=True)
            self.hide()
            if completion:
                completion()

    def update_level(self, level: float) -> None:
        """Apply a fast-attack, soft-release level to the visible glow."""
        normalized = max(0.0, min(1.0, level))
        alpha = _EMA_ATTACK if normalized > self._smoothed_level else _EMA_RELEASE
        self._smoothed_level = alpha * normalized + (1.0 - alpha) * self._smoothed_level
        if self._indicator_view is not None:
            self._indicator_view.set_level(self._smoothed_level)
