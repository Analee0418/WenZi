"""Lightweight, audio-reactive recording indicator (black hole) for macOS."""

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)

try:
    import objc
    from AppKit import (
        NSAffineTransform,
        NSAnimationContext,
        NSBezierPath,
        NSColor,
        NSColorSpace,
        NSGradient,
        NSGraphicsContext,
        NSPanel,
        NSRectClip,
        NSScreen,
        NSStatusWindowLevel,
        NSView,
    )
    from Foundation import (
        NSMakePoint,
        NSMakeRect,
        NSRunLoop,
        NSRunLoopCommonModes,
        NSTimer,
        NSValue,
    )
    from Quartz import (
        CABasicAnimation,
        CAMediaTimingFunction,
        CATransform3DMakeTranslation,
        CATransform3DScale,
        CATransform3DTranslate,
        kCAFillModeForwards,
        kCAMediaTimingFunctionEaseIn,
        kCAMediaTimingFunctionEaseOut,
    )

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

# Asymmetric EMA smoothing: speech should light the glow quickly and decay softly.
_EMA_ATTACK = 0.7
_EMA_RELEASE = 0.25
# Speech is amplitude-modulated at syllable rate; steady room noise is
# not.  The indicator therefore follows the level's MODULATION around
# its own local mean: continuous speech keeps it rippling indefinitely,
# while any steady signal — silence, fans, music — settles the mean and
# lets the disk calm down, regardless of how loud that steady signal is.
_MOD_MEAN_ALPHA = 0.12  # local-mean EMA per 50 ms poll (~1 s window)
_MOD_GAIN = 3.5  # maps typical speech swings into the response band

# The transparent host is wider than the black hole so the accretion
# disk can extend sideways.  Its visual center matches the panel center
# for preview transitions.
_PANEL_WIDTH = 240
_PANEL_HEIGHT = 236
_PANEL_CENTER_X = _PANEL_WIDTH / 2.0
_PANEL_CENTER_Y = _PANEL_HEIGHT / 2.0
_BH_CENTER_X = _PANEL_CENTER_X
_BH_CENTER_Y = _PANEL_CENTER_Y
_VOICE_FLOOR = 0.22
_VOICE_FULL = 0.55
_ACTIVITY_HANDOFF_DURATION = 0.30

# Black-hole anatomy.  Every luminous element is a radial gradient with
# a ring-shaped alpha profile — drawn inside a vertically squashed
# coordinate system it becomes a soft elliptical band of light, so the
# picture has no hard vector edges anywhere.  The only per-frame
# rebuilt geometry is the three streak rings flowing inside the disk.
# The void and its glowing rim are SEPARATE layers: the black core is
# a full feathered circle, but the rim glow is painted only over the
# hole's UPPER region — the bottom stays dark, so the light truly
# wraps AROUND the hole instead of growing out of its lower edge.
_SHADOW_RADIUS = 33.0
_SHADOW_CORE_STOP = 0.91  # pure black out to radius × stop (≈30 pt)
_SHADOW_LEVEL_GAIN = 0.05
_RIM_RADIUS = 46.0
_RIM_CLIP_Y = -10.0  # rim glow only above this offset from center
_DOME_LEVEL_GAIN = 0.06
# The top halo is alive: a barely-there sideways drift plus a gentle
# IRREGULAR pulse — two incommensurate sine waves interfering, so the
# breathing never repeats, like water.  Everything stays subtle: a
# visibly swinging or throbbing halo breaks the illusion immediately.
_DOME_SWAY = 1.5
_DOME_SWAY_RATE = 0.4
_DOME_PULSE = 0.03
_DOME_PULSE_RATE = 0.9
_DOME_PULSE_B = 0.022
_DOME_PULSE_RATE_B = 0.37
# The halo's SHAPE wobbles too (aspect ratio, third frequency) —
# uniform scaling alone always reads as mechanical breathing.
_DOME_WOBBLE = 0.03
_DOME_WOBBLE_RATE = 0.83
# The luminous disk band: hottest against the hole, red fade at the rim.
_DISK_BAND_RADIUS = 112.0
_DISK_BAND_SQUASH = 0.27
_DISK_POINT_COUNT = 28
_DISK_ANGLES = tuple(math.tau * index / _DISK_POINT_COUNT for index in range(_DISK_POINT_COUNT))
_DISK_FLATTEN = 0.22
_DISK_PHASES = (0.3, 2.1, 4.4)
_DISK_PHASES_B = (1.7, 3.9, 0.8)
# Each ring's wave crests travel at their own speed/direction.
_DISK_SPEEDS = (1.5, -1.2, 0.9)
_DISK_IDLE_RIPPLE = 0.05
_DISK_LEVEL_RIPPLE = 0.08
# Wide soft halo + thin core: each ring is a WAVE of light undulating
# through the band (soundwave-like), not a wire outline.
_DISK_STREAK_WIDTHS = (12.0, 2.6)
# The disk's INNER edge burns around the hole: a fire-colored collar
# lying in the disk plane (same squash), part of both disk passes —
# the far arc hides behind the void, the near arc draws the half-ring
# of fire around the hole.  The collar must stay NARROW: below the
# equator it paints on top of the stack, so a wide bright zone would
# recolor the flanks and step against the rim light above the line.
# The collar breathes in lockstep with the void (same voice gain) and,
# like the top rim's ember (31.3), may ride the core's feathered edge
# (30–33) — but never the opaque black:
#   R × ember stop (≈32.4) ≥ core × core stop (≈30),
# pinned by test_disk_fire_collar_clears_the_swollen_core.
_DISK_FIRE_RADIUS = 38.0
_DISK_FIRE_EMBER_STOP = 0.853
# Matter spirals INTO the hole: each streak ring falls from the outer
# disk toward the rim — accelerating, as infall does — is swallowed by
# the shadow, and respawns outside.  Three rings a third of a cycle
# apart make a continuous stream of light being sucked in, and the
# whole stream speeds up with the voice (it shares the flow phase).
_DISK_RING_COUNT = 3
_INFALL_OUTER = 99.0
_INFALL_INNER = 30.0
_INFALL_RATE = 0.12  # infall cycles per flow-phase radian
_INFALL_ACCEL = 1.6  # radius exponent: faster as it nears the hole
_INFALL_FADE_IN = 5.0  # width ramp after respawning outside
# Rings dissolve as they touch the rim — they must never survive into
# the shadow itself (the near-disk pass would show them over the void).
_INFALL_FADE_OUT = 7.0
# Bent-light domes over and under the hole: clipped soft ring gradients.
# The top dome's clip sits exactly on the equator, where the disk band
# is at its brightest — the cut line lands on saturated light and the
# halo blends seamlessly into the disk below it.
# Sizing is solved, not eyeballed: a compact top halo whose bright
# peak (radius × 0.88 ≈ 39) hugs the rim's white-hot peak (36.6), with
# the hollow at its mathematical minimum — the worst-case clearance
#   R × stop × (1 − pulses) × squash × (1 − wobble) ≥ black core (30)
# → R × stop ≥ 34.34 (here 34.8), pinned by
# test_halo_hollows_clear_the_black_core.
_DOME_TOP_RADIUS = 44.0
_DOME_TOP_INNER_STOP = 0.79
_DOME_TOP_PEAK_STOP = 0.88
_DOME_TOP_SQUASH = 0.95
# Like the rim, the top dome's cut hides 10 pt BELOW the equator: the
# fire collar spreads bright light past the band's saturated core, so
# a cut exactly on the equator shows as a brightness step beside the
# hole.  At −10 the cut lands back on saturated disk light.
_DOME_TOP_CLIP_Y = -10.0
# The bottom halo is concentric with the top one but LARGER and more
# diffuse: hollow 53.3 (well clear of the 30 pt core), glow spread over
# a wide 21.3 pt falloff (peak at 0.74 × R ≈ 61 → zero at 82).
_DOME_BOTTOM_RADIUS = 82.0
_DOME_BOTTOM_SQUASH = 0.95
_DOME_BOTTOM_INNER_STOP = 0.65
_DOME_BOTTOM_PEAK_STOP = 0.74
_DOME_BOTTOM_CLIP_Y = -10.0
# The bottom halo answers the voice by TREMBLING, not swelling: its
# radius carries no pulse at all — instead the aspect ratio and the
# center jitter on three fast incommensurate rates, like ripples on
# water.  Everything is gated by the pull: silent, it rests.
_DOME_BOTTOM_WOBBLE = 0.06
_DOME_BOTTOM_WOBBLE_RATE = 2.9
_DOME_BOTTOM_JITTER_X = 3.5
_DOME_BOTTOM_JITTER_X_RATE = 3.7
_DOME_BOTTOM_JITTER_Y = 2.8
_DOME_BOTTOM_JITTER_Y_RATE = 5.3
_GLOW_BASE_RADIUS = 116.0
_GLOW_LEVEL_GAIN = 10.0
# The panel sits in the upper part of the screen so it never covers
# content the user is dictating into (usually mid/lower screen).
_SCREEN_VERTICAL_BIAS = 0.70
# The "breathing" is light flow, not size: the streak phase advances
# continuously (accumulated, so speed changes never snap the texture)
# and speech accelerates it.
_FLOW_BASE_SPEED = 1.6  # radians per second at rest
_FLOW_LEVEL_GAIN = 1.5
_FLOW_MAX_FRAME_SECS = 0.1

# One coalescible 30 Hz timer drives the disk flow and audio response.
# Entry/exit scale+fade are Core Animation transforms interpolated by the
# system at display refresh, so drawRect_ always draws resting geometry.
_REFRESH_INTERVAL = 1.0 / 30.0
_ENTRY_DURATION = 0.22
_ENTRY_SCALE_FROM = 0.86
_EXIT_DURATION = 0.20
_EXIT_SCALE_TO = 0.76


def _center_scale_transform(scale: float) -> object:
    """Compose a scale about the panel center as an NSValue<CATransform3D>.

    Composing translate → scale → translate-back avoids touching
    ``layer.anchorPoint``: AppKit pins backing-layer anchor points at
    (0, 0), and mutating them shifts the view frame.

    The NSValue wrap is mandatory: a raw PyObjC struct bridges into
    Core Animation as OC_PythonObject and crashes the app at the next
    CATransaction commit (CA_prepareRenderValue → unrecognized selector).
    """
    transform = CATransform3DMakeTranslation(_PANEL_CENTER_X, _PANEL_CENTER_Y, 0.0)
    transform = CATransform3DScale(transform, scale, scale, 1.0)
    transform = CATransform3DTranslate(transform, -_PANEL_CENTER_X, -_PANEL_CENTER_Y, 0.0)
    return NSValue.valueWithCATransform3D_(transform)


def _gravity_pull(level: float) -> float:
    """Separate room noise from speech with a smooth, bounded response."""
    activity = max(0.0, min(1.0, level))
    normalized = max(
        0.0,
        min(1.0, (activity - _VOICE_FLOOR) / (_VOICE_FULL - _VOICE_FLOOR)),
    )
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _handoff_activity(level: float, progress: float) -> float:
    """Ease initial microphone activity through the existing response band."""
    activity = max(0.0, min(1.0, level))
    handoff = max(0.0, min(1.0, progress))
    if handoff >= 1.0 or activity <= _VOICE_FLOOR:
        return activity
    normalized = min(
        1.0,
        (activity - _VOICE_FLOOR) / (_VOICE_FULL - _VOICE_FLOOR),
    )
    return _VOICE_FLOOR + normalized * handoff * (_VOICE_FULL - _VOICE_FLOOR)


def _infall_cycle(index: int, phase: float) -> float:
    """Return this ring's position in its infall cycle (0 outer → 1 rim)."""
    return (phase * _INFALL_RATE + index / _DISK_RING_COUNT) % 1.0


def _disk_points(
    index: int,
    phase: float,
    activity: float,
) -> tuple[tuple[float, float], ...]:
    """Return one accretion-disk streak ring spiralling into the hole.

    A flattened ellipse that falls inward with *phase* (accelerating
    toward the rim) while its radius is modulated by two travelling
    waves: the same phase swirls the streaks around the ring and the
    voice pull deepens them (speech ripples).
    """
    pull = _gravity_pull(activity)
    amplitude = _DISK_IDLE_RIPPLE + pull * _DISK_LEVEL_RIPPLE
    motion = phase * _DISK_SPEEDS[index]
    cycle = _infall_cycle(index, phase)
    radius_x = _INFALL_OUTER - (
        (_INFALL_OUTER - _INFALL_INNER) * cycle**_INFALL_ACCEL
    )
    radius_y = radius_x * _DISK_FLATTEN
    points = []
    for angle in _DISK_ANGLES:
        ripple = (
            0.6 * math.sin(angle * 3.0 + _DISK_PHASES[index] + motion)
            + 0.4 * math.sin(angle * 7.0 - motion * 0.6 + _DISK_PHASES_B[index])
        )
        scale = 1.0 + amplitude * ripple
        points.append(
            (
                _BH_CENTER_X + math.cos(angle) * radius_x * scale,
                _BH_CENTER_Y + math.sin(angle) * radius_y * scale,
            )
        )
    return tuple(points)


class RecordingIndicatorView:
    """Draw a black hole whose accretion disk flows with the voice."""

    _view: object = None

    def __init__(self) -> None:
        self._level = 0.0
        self._start_time = 0.0
        self._view = None
        self._recording_active = False
        self._recording_active_at: float | None = None
        # Accumulated streak phase; advancing it per frame (instead of
        # deriving it from elapsed time) keeps the texture continuous
        # when the flow speed changes with the voice.
        self._flow_phase = 0.0
        self._last_flow_time: float | None = None

        # Native colors, gradients, and paths are prewarmed once.
        self._glow_gradient = None
        self._band_gradient = None
        self._shadow_gradient = None
        self._rim_gradient = None
        self._dome_top_gradient = None
        self._dome_bottom_gradient = None
        self._disk_fire_gradient = None
        self._disk_paths = None
        self._disk_colors = None

    def create_view(self, width: int, height: int) -> object:
        """Create and return the transparent, layer-backed NSView host."""
        view = _IndicatorNSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        view._indicator = self
        # Layer-backed so the entry/exit scale runs as a Core Animation
        # transform over the rasterized contents instead of CPU redraws.
        view.setWantsLayer_(True)
        self._view = view
        if self._start_time == 0.0:
            self._start_time = time.monotonic()
        return view

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))

    def reset_session(self) -> None:
        """Reset per-recording state on the reused view."""
        self._level = 0.0
        self._recording_active = False
        self._recording_active_at = None
        self._flow_phase = 0.0
        self._last_flow_time = None
        # Restart the clock so every session opens on the same calm
        # frame, and elapsed stays small across app lifetime.
        self._start_time = time.monotonic()

    def activate_recording(self) -> None:
        """Begin a one-shot visual handoff from idle to live microphone level."""
        if self._recording_active:
            return
        self._recording_active = True
        self._recording_active_at = time.monotonic()

    def _effective_activity(self, elapsed: float) -> float:
        """Return live activity with a smooth first-sample handoff."""
        if not self._recording_active:
            return 0.0
        if self._recording_active_at is None:
            return self._level
        active_elapsed = max(
            0.0,
            self._start_time + elapsed - self._recording_active_at,
        )
        if active_elapsed + 1e-9 >= _ACTIVITY_HANDOFF_DURATION:
            self._recording_active_at = None
            return self._level
        progress = active_elapsed / _ACTIVITY_HANDOFF_DURATION
        return _handoff_activity(self._level, progress)

    def prepare_for_display(self) -> None:
        """Create all drawing resources before the panel becomes visible."""
        self._ensure_active_colors()

    def _ensure_active_colors(self) -> None:
        if self._glow_gradient is not None:
            return

        def _srgb(red: float, green: float, blue: float, alpha: float):
            return NSColor.colorWithSRGBRed_green_blue_alpha_(red, green, blue, alpha)

        def _ring_gradient(stops):
            """A radial gradient with a ring-shaped alpha profile."""
            return NSGradient.alloc().initWithColors_atLocations_colorSpace_(
                [_srgb(*color) for color, _ in stops],
                tuple(location for _, location in stops),
                NSColorSpace.sRGBColorSpace(),
            )

        # The three disk streak rings are the only per-frame rebuilt
        # paths; create them first so a failed prewarm retries cleanly.
        disk_paths = tuple(
            NSBezierPath.alloc().init() for _ in range(_DISK_RING_COUNT)
        )
        for path in disk_paths:
            path.setLineJoinStyle_(1)
            path.setLineCapStyle_(1)

        glow_gradient = _ring_gradient(
            (
                ((1.00, 0.40, 0.10, 0.07), 0.0),
                ((1.00, 0.22, 0.05, 0.025), 0.55),
                ((1.00, 0.22, 0.05, 0.0), 1.0),
            )
        )
        # The luminous band: transparent over the hole, a hot white-amber
        # inner edge, cooling to deep red and fading at the rim.
        band_gradient = _ring_gradient(
            (
                ((1.00, 0.90, 0.66, 0.0), 0.0),
                ((1.00, 0.90, 0.66, 0.0), 0.26),
                ((1.00, 0.90, 0.66, 0.95), 0.33),
                ((1.00, 0.66, 0.20, 0.80), 0.50),
                ((0.95, 0.36, 0.07, 0.50), 0.75),
                ((0.85, 0.20, 0.04, 0.0), 1.0),
            )
        )
        # The void: an absolute black feathered circle, nothing more.
        shadow_gradient = _ring_gradient(
            (
                ((0.0, 0.0, 0.0, 1.0), 0.0),
                ((0.0, 0.0, 0.0, 1.0), _SHADOW_CORE_STOP),
                ((0.0, 0.0, 0.0, 0.0), 1.0),
            )
        )
        # Its photon rim: fire rising straight out of the void into a
        # white-hot peak that melts outward — painted only over the
        # hole's upper region, so the light wraps around the void from
        # above.  No dark stop is allowed between the black core and the
        # bright light: any opaque dark band here stacks on the core's
        # silhouette and reads as a second, larger sphere behind it.
        rim_gradient = _ring_gradient(
            (
                ((0.98, 0.45, 0.10, 0.0), 0.0),
                ((0.98, 0.45, 0.10, 0.0), 0.615),
                ((0.98, 0.45, 0.10, 1.0), 0.68),
                ((1.00, 0.62, 0.20, 0.95), 0.74),
                ((1.00, 0.85, 0.55, 0.85), 0.795),
                ((1.00, 0.68, 0.28, 0.45), 0.87),
                ((1.00, 0.55, 0.20, 0.0), 1.0),
            )
        )
        dome_top_gradient = _ring_gradient(
            (
                ((1.00, 0.84, 0.52, 0.0), 0.0),
                ((1.00, 0.84, 0.52, 0.0), _DOME_TOP_INNER_STOP),
                ((1.00, 0.86, 0.55, 0.90), _DOME_TOP_PEAK_STOP),
                ((1.00, 0.65, 0.28, 0.0), 1.0),
            )
        )
        dome_bottom_gradient = _ring_gradient(
            (
                ((1.00, 0.66, 0.28, 0.0), 0.0),
                ((1.00, 0.66, 0.28, 0.0), _DOME_BOTTOM_INNER_STOP),
                ((1.00, 0.68, 0.30, 0.30), _DOME_BOTTOM_PEAK_STOP),
                ((1.00, 0.55, 0.20, 0.0), 1.0),
            )
        )
        # The fire collar on the disk's inner edge: the same hot orange
        # as the photon rim, so the half-ring the near disk draws around
        # the hole reads as one continuous circle of fire with the top.
        disk_fire_gradient = _ring_gradient(
            (
                ((0.98, 0.45, 0.10, 0.0), 0.0),
                ((0.98, 0.45, 0.10, 0.0), 0.776),
                ((0.98, 0.45, 0.10, 0.72), _DISK_FIRE_EMBER_STOP),
                ((1.00, 0.62, 0.20, 0.52), 0.905),
                ((1.00, 0.72, 0.35, 0.18), 0.953),
                ((1.00, 0.60, 0.25, 0.0), 1.0),
            )
        )
        # Shimmer streaks flowing inside the band (halo, core) per ring,
        # hottest innermost.
        disk_colors = (
            (_srgb(1.00, 0.80, 0.45, 0.10), _srgb(1.00, 0.88, 0.60, 0.38)),
            (_srgb(1.00, 0.70, 0.32, 0.09), _srgb(1.00, 0.78, 0.45, 0.34)),
            (_srgb(1.00, 0.55, 0.20, 0.08), _srgb(1.00, 0.62, 0.28, 0.30)),
        )

        # Publish the cache only after every native resource exists. A failed
        # prewarm can then retry cleanly instead of retaining a partial palette.
        self._band_gradient = band_gradient
        self._shadow_gradient = shadow_gradient
        self._rim_gradient = rim_gradient
        self._dome_top_gradient = dome_top_gradient
        self._dome_bottom_gradient = dome_bottom_gradient
        self._disk_fire_gradient = disk_fire_gradient
        self._disk_paths = disk_paths
        self._disk_colors = disk_colors
        self._glow_gradient = glow_gradient

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

    def _draw_squashed_ring(
        self,
        gradient,
        radius: float,
        squash: float,
        center_x: float = _BH_CENTER_X,
        center_y_offset: float = 0.0,
    ) -> None:
        """Draw a ring-profile radial gradient in a y-squashed frame —
        a soft elliptical band of light with no hard edges.

        *center_y_offset* is the VISUAL offset from the hole center;
        the squash scales y about the hole center, so the offset is
        divided back out for the in-frame coordinate.
        """
        NSGraphicsContext.saveGraphicsState()
        try:
            transform = NSAffineTransform.transform()
            transform.translateXBy_yBy_(_BH_CENTER_X, _BH_CENTER_Y)
            transform.scaleXBy_yBy_(1.0, squash)
            transform.translateXBy_yBy_(-_BH_CENTER_X, -_BH_CENTER_Y)
            transform.concat()
            center = NSMakePoint(
                center_x, _BH_CENTER_Y + center_y_offset / squash
            )
            gradient.drawFromCenter_radius_toCenter_radius_options_(
                center, 0.0, center, radius, 0
            )
        finally:
            NSGraphicsContext.restoreGraphicsState()

    def _draw_disk(self, pull: float) -> None:
        """The luminous band plus the light waves undulating through it
        while spiralling into the hole, and the fire collar burning on
        the band's inner edge.  The collar is part of BOTH the far and
        near passes — painted once only, it would end on a hard seam at
        the equator beside the hole — and it breathes with the void
        (same voice gain), like the rim above."""
        self._draw_squashed_ring(
            self._band_gradient, _DISK_BAND_RADIUS, _DISK_BAND_SQUASH
        )
        for index in reversed(range(_DISK_RING_COUNT)):
            # Rings fade in as they respawn outside and dissolve again
            # as they reach the rim and are swallowed.
            cycle = _infall_cycle(index, self._flow_phase)
            fade = min(
                1.0,
                cycle * _INFALL_FADE_IN,
                (1.0 - cycle) * _INFALL_FADE_OUT,
            )
            if fade < 0.05:
                continue
            path = self._disk_paths[index]
            halo, core = self._disk_colors[index]
            halo.setStroke()
            path.setLineWidth_(_DISK_STREAK_WIDTHS[0] * fade)
            path.stroke()
            core.setStroke()
            path.setLineWidth_(_DISK_STREAK_WIDTHS[1] * fade)
            path.stroke()
        self._draw_squashed_ring(
            self._disk_fire_gradient,
            _DISK_FIRE_RADIUS * (1.0 + _SHADOW_LEVEL_GAIN * pull),
            _DISK_BAND_SQUASH,
        )

    def _draw_black_hole(self, activity: float, pull: float) -> None:
        self._ensure_active_colors()
        center = NSMakePoint(_BH_CENTER_X, _BH_CENTER_Y)

        # Ambient glow
        self._glow_gradient.drawFromCenter_radius_toCenter_radius_options_(
            center,
            0.0,
            center,
            _GLOW_BASE_RADIUS + _GLOW_LEVEL_GAIN * pull,
            0,
        )

        # Painter's algorithm: the disk's FAR half first (behind the
        # hole), then the full shadow, then the domes, and finally the
        # disk's NEAR half over everything — every overlap is gradient
        # on gradient, no cut lines anywhere.
        for index, path in enumerate(self._disk_paths):
            self._rebuild_closed_path(
                path,
                _disk_points(index, self._flow_phase, activity),
            )
        NSGraphicsContext.saveGraphicsState()
        try:
            NSRectClip(
                NSMakeRect(
                    0.0,
                    _BH_CENTER_Y,
                    _PANEL_WIDTH,
                    _PANEL_HEIGHT - _BH_CENTER_Y,
                )
            )
            self._draw_disk(pull)
        finally:
            NSGraphicsContext.restoreGraphicsState()

        # The black void, then its photon rim — painted only over the
        # hole's upper region (the cut hides under the near disk), so
        # the glow wraps the void from above and the bottom stays dark.
        # Both breathe slightly outward with the voice.
        self._shadow_gradient.drawFromCenter_radius_toCenter_radius_options_(
            center,
            0.0,
            center,
            _SHADOW_RADIUS * (1.0 + _SHADOW_LEVEL_GAIN * pull),
            0,
        )
        NSGraphicsContext.saveGraphicsState()
        try:
            NSRectClip(
                NSMakeRect(
                    0.0,
                    _BH_CENTER_Y + _RIM_CLIP_Y,
                    _PANEL_WIDTH,
                    _PANEL_HEIGHT - (_BH_CENTER_Y + _RIM_CLIP_Y),
                )
            )
            self._rim_gradient.drawFromCenter_radius_toCenter_radius_options_(
                center,
                0.0,
                center,
                _RIM_RADIUS * (1.0 + _SHADOW_LEVEL_GAIN * pull),
                0,
            )
        finally:
            NSGraphicsContext.restoreGraphicsState()

        # Gravitationally bent light: soft ring gradients clipped to the
        # regions over and under the hole.  The bright top halo is alive —
        # it sways and pulses with the flow.
        dome_gain = 1.0 + _DOME_LEVEL_GAIN * pull
        NSGraphicsContext.saveGraphicsState()
        try:
            NSRectClip(
                NSMakeRect(
                    0.0,
                    _BH_CENTER_Y + _DOME_TOP_CLIP_Y,
                    _PANEL_WIDTH,
                    _PANEL_HEIGHT - (_BH_CENTER_Y + _DOME_TOP_CLIP_Y),
                )
            )
            self._draw_squashed_ring(
                self._dome_top_gradient,
                _DOME_TOP_RADIUS
                * dome_gain
                * (
                    1.0
                    + _DOME_PULSE
                    * math.sin(self._flow_phase * _DOME_PULSE_RATE)
                    + _DOME_PULSE_B
                    * math.sin(self._flow_phase * _DOME_PULSE_RATE_B + 1.7)
                ),
                _DOME_TOP_SQUASH
                * (
                    1.0
                    + _DOME_WOBBLE
                    * math.sin(self._flow_phase * _DOME_WOBBLE_RATE + 0.5)
                ),
                center_x=(
                    _BH_CENTER_X
                    + _DOME_SWAY
                    * math.sin(self._flow_phase * _DOME_SWAY_RATE)
                ),
            )
        finally:
            NSGraphicsContext.restoreGraphicsState()
        NSGraphicsContext.saveGraphicsState()
        try:
            NSRectClip(
                NSMakeRect(
                    0.0,
                    0.0,
                    _PANEL_WIDTH,
                    _BH_CENTER_Y + _DOME_BOTTOM_CLIP_Y,
                )
            )
            self._draw_squashed_ring(
                self._dome_bottom_gradient,
                _DOME_BOTTOM_RADIUS * dome_gain,
                _DOME_BOTTOM_SQUASH
                * (
                    1.0
                    + _DOME_BOTTOM_WOBBLE
                    * pull
                    * math.sin(self._flow_phase * _DOME_BOTTOM_WOBBLE_RATE + 1.9)
                ),
                center_x=(
                    _BH_CENTER_X
                    + _DOME_BOTTOM_JITTER_X
                    * pull
                    * math.sin(self._flow_phase * _DOME_BOTTOM_JITTER_X_RATE + 0.7)
                ),
                center_y_offset=(
                    _DOME_BOTTOM_JITTER_Y
                    * pull
                    * math.sin(self._flow_phase * _DOME_BOTTOM_JITTER_Y_RATE + 2.6)
                ),
            )
        finally:
            NSGraphicsContext.restoreGraphicsState()

        # The disk's NEAR half, in front of the hole and its rim.
        NSGraphicsContext.saveGraphicsState()
        try:
            NSRectClip(NSMakeRect(0.0, 0.0, _PANEL_WIDTH, _BH_CENTER_Y))
            self._draw_disk(pull)
        finally:
            NSGraphicsContext.restoreGraphicsState()

    def draw(self, rect: object) -> None:
        """Draw the black hole; the disk streaks flow, nothing scales.

        Entry/exit scaling happens on the backing layer, never inside
        the raster loop.
        """
        now = time.monotonic()
        elapsed = max(0.0, now - self._start_time)
        activity = self._effective_activity(elapsed)
        pull = _gravity_pull(activity)
        if self._last_flow_time is None:
            frame_secs = 0.0
        else:
            frame_secs = min(
                max(0.0, now - self._last_flow_time),
                _FLOW_MAX_FRAME_SECS,
            )
        self._last_flow_time = now
        self._flow_phase += (
            _FLOW_BASE_SPEED * (1.0 + _FLOW_LEVEL_GAIN * pull) * frame_secs
        )
        self._draw_black_hole(activity, pull)


class RecordingIndicatorPanel:
    """Own the transparent native host for the centered recording orb.

    The NSPanel, the layer-backed view and the drawing palette are built
    once (``prewarm()`` or the first ``show()``) and reused across
    recordings — a cold build plus first-frame rasterization on the main
    thread is exactly the hitch users see on the first show.
    """

    def __init__(self) -> None:
        self._panel: object = None
        self._timer: object = None
        self._indicator_view: RecordingIndicatorView | None = None
        self._smoothed_level = 0.0
        self._level_mean: float | None = None
        self._enabled = True
        self._show_device_name = False
        self._visible = False
        # Bumped on every show(): the reused panel's identity can no
        # longer distinguish sessions, so animate_out's async completion
        # must check the generation instead.
        self._show_gen = 0
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
            # A disabled indicator should not keep native objects alive.
            self.hide()
            self._teardown()

    @property
    def show_device_name(self) -> bool:
        return self._show_device_name

    @show_device_name.setter
    def show_device_name(self, value: bool) -> None:
        self._show_device_name = value

    @staticmethod
    def _configure_refresh_timer(timer) -> None:
        """Allow macOS to coalesce redraw wakeups where possible."""
        try:
            timer.setTolerance_(_REFRESH_INTERVAL * 0.2)
        except Exception:
            logger.debug("NSTimer tolerance is unavailable", exc_info=True)

    def prewarm(self) -> None:
        """Build the native host ahead of the first recording (main thread)."""
        if not self._enabled:
            return
        try:
            self._ensure_built()
        except Exception:
            logger.debug("Recording orb prewarm failed", exc_info=True)
            self._teardown()

    def _ensure_built(self) -> None:
        """Create the panel, layer-backed view and palette exactly once."""
        if self._panel is not None:
            return

        indicator_view = RecordingIndicatorView()
        indicator_view.prepare_for_display()

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
        panel.setContentView_(indicator_view.create_view(_PANEL_WIDTH, _PANEL_HEIGHT))

        # Publish only the fully-built pair (all-or-nothing, like the palette).
        self._indicator_view = indicator_view
        self._panel = panel

    def _run_entry_animation(self, panel, layer) -> None:
        """Zoom+fade the whole orb in, interpolated by CA at display refresh."""
        scale = CABasicAnimation.animationWithKeyPath_("transform")
        scale.setFromValue_(_center_scale_transform(_ENTRY_SCALE_FROM))
        scale.setDuration_(_ENTRY_DURATION)
        scale.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionEaseOut)
        )
        layer.addAnimation_forKey_(scale, "wenzi.entry")
        NSAnimationContext.beginGrouping()
        try:
            context = NSAnimationContext.currentContext()
            context.setDuration_(_ENTRY_DURATION)
            context.setTimingFunction_(
                CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionEaseOut)
            )
            panel.animator().setAlphaValue_(1.0)
        finally:
            NSAnimationContext.endGrouping()

    def show(
        self,
        device_name: str | None = None,
        mode_name: str | None = None,
    ) -> None:
        """Show a centered orb; legacy text arguments are stored but not drawn."""
        if not self._enabled:
            return

        try:
            self._ensure_built()
            if self._visible or self._timer is not None:
                # Re-show over a live orb: stop its timer and order out
                # before restarting the session cleanly.
                self.hide()
            self._show_gen += 1
            self._smoothed_level = 0.0
            self._level_mean = None
            self._mode_name = mode_name
            self._device_name = device_name

            panel = self._panel
            view = self._indicator_view._view
            self._indicator_view.reset_session()
            # Drop the previous exit animation's held end state
            # (fillMode=forwards) before the orb becomes visible again.
            layer = view.layer()
            layer.removeAllAnimations()

            # Re-position on the CURRENT main screen — it changes between
            # shows.  Horizontally centered, vertically in the upper part
            # of the screen so dictated content below stays visible.
            screen = NSScreen.mainScreen()
            if screen:
                frame = screen.visibleFrame()
                x = frame.origin.x + (frame.size.width - _PANEL_WIDTH) / 2.0
                y = frame.origin.y + (
                    (frame.size.height - _PANEL_HEIGHT) * _SCREEN_VERTICAL_BIAS
                )
                panel.setFrameOrigin_((x, y))

            # Attach the window invisibly so AppKit creates a backing store,
            # then synchronously rasterize the colored first frame before reveal.
            panel.setAlphaValue_(0.0)
            panel.orderFront_(None)
            view.setNeedsDisplay_(True)
            view.displayIfNeededIgnoringOpacity()
            self._run_entry_animation(panel, layer)
            self._visible = True

            timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                _REFRESH_INTERVAL,
                view,
                b"refresh:",
                None,
                True,
            )
            self._configure_refresh_timer(timer)
            # Common modes: the orb must keep breathing during menu tracking.
            NSRunLoop.currentRunLoop().addTimer_forMode_(
                timer, NSRunLoopCommonModes
            )
            self._timer = timer
            logger.debug("Recording orb shown")
        except Exception:
            logger.error("Failed to show recording orb", exc_info=True)
            self._teardown()

    def set_recording_active(self) -> None:
        """Allow microphone activity to influence the already-visible orb."""
        if self._indicator_view is not None:
            self._indicator_view.activate_recording()
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
        """Hide the orb and stop its timer, keeping the built host for reuse."""
        timer = self._timer
        self._timer = None
        self._visible = False
        self._smoothed_level = 0.0
        self._level_mean = None
        self._mode_name = None
        self._device_name = None

        if timer is not None:
            try:
                timer.invalidate()
            except Exception:
                logger.warning("Failed to invalidate recording orb timer", exc_info=True)
        if self._panel is not None:
            try:
                self._panel.orderOut_(None)
            except Exception:
                logger.warning("Failed to close recording orb panel", exc_info=True)
                # Broken native state cannot be trusted for reuse.
                self._teardown()
        logger.debug("Recording orb hidden")

    def _teardown(self) -> None:
        """Destroy the native host entirely; the next show() cold-builds it."""
        timer = self._timer
        panel = self._panel
        indicator_view = self._indicator_view

        # Detach ownership first: native cleanup failures must not leave stale
        # state that can overwrite or interfere with the next recording.
        self._timer = None
        self._panel = None
        self._indicator_view = None
        self._visible = False
        self._smoothed_level = 0.0
        self._level_mean = None
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
        logger.debug("Recording orb host destroyed")

    def update_mode(self, name: str) -> None:
        """Remember the active mode without adding text around the orb."""
        self._mode_name = name

    def clear_mode(self) -> None:
        """Clear the compatibility-only active mode state."""
        self._mode_name = None

    def update_device_name(self, device_name: str) -> None:
        """Remember the input name without adding text around the orb."""
        if self._visible:
            self._device_name = device_name

    @property
    def current_frame(self) -> object | None:
        """Return the orb host frame, or None while it is hidden."""
        if self._panel is None or not self._visible:
            return None
        return self._panel.frame()

    def animate_out(self, completion: callable = None) -> None:
        """Shrink and fade out the orb, then clean up and call completion."""
        if self._panel is None or not self._visible:
            if completion:
                completion()
            return

        try:
            panel = self._panel
            view = self._indicator_view._view
            gen = self._show_gen

            # The last rasterized frame is frozen under the fade — stop
            # paying for redraws during the exit.
            timer = self._timer
            self._timer = None
            if timer is not None:
                try:
                    timer.invalidate()
                except Exception:
                    logger.warning(
                        "Failed to invalidate recording orb timer",
                        exc_info=True,
                    )

            shrink = CABasicAnimation.animationWithKeyPath_("transform")
            shrink.setToValue_(_center_scale_transform(_EXIT_SCALE_TO))
            shrink.setDuration_(_EXIT_DURATION)
            shrink.setTimingFunction_(
                CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionEaseIn)
            )
            # Hold the shrunken end state until orderOut; the next show()
            # clears it via removeAllAnimations().
            shrink.setFillMode_(kCAFillModeForwards)
            shrink.setRemovedOnCompletion_(False)
            view.layer().addAnimation_forKey_(shrink, "wenzi.exit")

            def _on_complete():
                if self._show_gen != gen:
                    # A newer show() owns the reused panel now.
                    return
                self._visible = False
                self._smoothed_level = 0.0
                self._level_mean = None
                self._mode_name = None
                self._device_name = None
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
        """Apply a fast-attack, soft-release level to the visible glow.

        The disk follows the level's modulation around its local mean:
        speech pulses at syllable rate and keeps the streaks rippling
        for as long as the user talks, while any steady signal (silence
        or constant room noise at any loudness) reads as zero swing and
        lets the disk calm down.
        """
        normalized = max(0.0, min(1.0, level))
        if self._level_mean is None:
            # Seed with the first sample so a session opening onto steady
            # room noise does not read as one large artificial swing.
            self._level_mean = normalized
        self._level_mean += (normalized - self._level_mean) * _MOD_MEAN_ALPHA
        swing = min(1.0, abs(normalized - self._level_mean) * _MOD_GAIN)
        alpha = _EMA_ATTACK if swing > self._smoothed_level else _EMA_RELEASE
        self._smoothed_level = alpha * swing + (1.0 - alpha) * self._smoothed_level
        if self._indicator_view is not None:
            self._indicator_view.set_level(self._smoothed_level)
