"""Tests for the lightweight recording indicator (black hole)."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

import wenzi.audio.recording_indicator as ri
from wenzi.audio.recording_indicator import (
    _ACTIVITY_HANDOFF_DURATION,
    _BH_CENTER_X,
    _BH_CENTER_Y,
    _DISK_BAND_RADIUS,
    _DISK_BAND_SQUASH,
    _DISK_FIRE_EMBER_STOP,
    _DISK_FIRE_RADIUS,
    _DISK_POINT_COUNT,
    _DISK_STREAK_WIDTHS,
    _DOME_BOTTOM_INNER_STOP,
    _DOME_BOTTOM_JITTER_X,
    _DOME_BOTTOM_JITTER_X_RATE,
    _DOME_BOTTOM_JITTER_Y,
    _DOME_BOTTOM_RADIUS,
    _DOME_BOTTOM_SQUASH,
    _DOME_BOTTOM_WOBBLE,
    _DOME_LEVEL_GAIN,
    _DOME_PULSE,
    _DOME_PULSE_B,
    _DOME_PULSE_RATE,
    _DOME_PULSE_RATE_B,
    _DOME_TOP_INNER_STOP,
    _DOME_TOP_RADIUS,
    _DOME_TOP_SQUASH,
    _DOME_WOBBLE,
    _ENTRY_DURATION,
    _ENTRY_SCALE_FROM,
    _EXIT_DURATION,
    _EXIT_SCALE_TO,
    _FLOW_BASE_SPEED,
    _FLOW_LEVEL_GAIN,
    _INFALL_RATE,
    _PANEL_CENTER_X,
    _PANEL_CENTER_Y,
    _PANEL_HEIGHT,
    _PANEL_WIDTH,
    _REFRESH_INTERVAL,
    _RIM_RADIUS,
    _SCREEN_VERTICAL_BIAS,
    _SHADOW_CORE_STOP,
    _SHADOW_LEVEL_GAIN,
    _SHADOW_RADIUS,
    RecordingIndicatorPanel,
    RecordingIndicatorView,
    _center_scale_transform,
    _disk_points,
    _gravity_pull,
    _handoff_activity,
)

# The module binds its PyObjC symbols at import time; tests replace the
# bound names with fake namespaces instead of patching sys.modules.
_APPKIT_NAMES = (
    "NSAnimationContext",
    "NSBezierPath",
    "NSColor",
    "NSColorSpace",
    "NSGradient",
    "NSGraphicsContext",
    "NSPanel",
    "NSRectClip",
    "NSScreen",
    "NSStatusWindowLevel",
)
_FOUNDATION_NAMES = (
    "NSMakePoint",
    "NSMakeRect",
    "NSRunLoop",
    "NSRunLoopCommonModes",
    "NSTimer",
    "NSValue",
)
_QUARTZ_NAMES = (
    "CABasicAnimation",
    "CAMediaTimingFunction",
    "CATransform3DMakeTranslation",
    "CATransform3DScale",
    "CATransform3DTranslate",
    "kCAFillModeForwards",
    "kCAMediaTimingFunctionEaseIn",
    "kCAMediaTimingFunctionEaseOut",
)


def _install_fake_native(monkeypatch):
    """Bind the module's native names to fake AppKit/Foundation/Quartz."""
    appkit = MagicMock(name="AppKit")
    appkit.NSStatusWindowLevel = 100
    foundation = MagicMock(name="Foundation")
    quartz = MagicMock(name="Quartz")
    for name in _APPKIT_NAMES:
        monkeypatch.setattr(ri, name, getattr(appkit, name))
    for name in _FOUNDATION_NAMES:
        monkeypatch.setattr(ri, name, getattr(foundation, name))
    for name in _QUARTZ_NAMES:
        monkeypatch.setattr(ri, name, getattr(quartz, name))
    monkeypatch.setattr(ri, "_IndicatorNSView", appkit._IndicatorNSView)
    return appkit, foundation, quartz


def _average_distance(first, second):
    return sum(math.dist(first_point, second_point) for first_point, second_point in zip(first, second, strict=True)) / len(first)


def _curve_extent_points(points, *, closed):
    """Include the Catmull-Rom controls that can overshoot sampled points."""
    extent_points = list(points)
    point_count = len(points)
    segment_count = point_count if closed else point_count - 1
    for index in range(segment_count):
        if closed:
            previous = points[(index - 1) % point_count]
            following = points[(index + 1) % point_count]
            after_following = points[(index + 2) % point_count]
        else:
            previous = points[max(0, index - 1)]
            following = points[index + 1]
            after_following = points[min(point_count - 1, index + 2)]
        current = points[index]
        extent_points.extend(
            (
                (
                    current[0] + (following[0] - previous[0]) / 6.0,
                    current[1] + (following[1] - previous[1]) / 6.0,
                ),
                (
                    following[0] - (after_following[0] - current[0]) / 6.0,
                    following[1] - (after_following[1] - current[1]) / 6.0,
                ),
            )
        )
    return extent_points


def _install_mock_resources(view):
    """Install a complete prewarmed cache without allocating AppKit objects."""
    view._glow_gradient = MagicMock(name="glow_gradient")
    view._band_gradient = MagicMock(name="band_gradient")
    view._shadow_gradient = MagicMock(name="shadow_gradient")
    view._rim_gradient = MagicMock(name="rim_gradient")
    view._dome_top_gradient = MagicMock(name="dome_top_gradient")
    view._dome_bottom_gradient = MagicMock(name="dome_bottom_gradient")
    view._disk_fire_gradient = MagicMock(name="disk_fire_gradient")
    view._disk_paths = tuple(MagicMock(name=f"disk_path_{index}") for index in range(3))
    view._disk_colors = tuple(
        (
            MagicMock(name=f"disk_halo_{index}"),
            MagicMock(name=f"disk_core_{index}"),
        )
        for index in range(3)
    )
    return view


def _cached_resources(view):
    return (
        view._glow_gradient,
        view._band_gradient,
        view._shadow_gradient,
        view._rim_gradient,
        view._dome_top_gradient,
        view._dome_bottom_gradient,
        view._disk_fire_gradient,
        view._disk_paths,
        view._disk_colors,
    )


class TestRecordingIndicatorView:
    def test_set_level_clamps_to_normalized_range(self):
        view = RecordingIndicatorView()

        view.set_level(-1.0)
        assert view._level == 0.0
        view.set_level(0.5)
        assert view._level == 0.5
        view.set_level(2.0)
        assert view._level == 1.0

    def test_initial_state_is_idle(self):
        view = RecordingIndicatorView()

        assert view._recording_active is False
        assert view._recording_active_at is None
        assert view._level == 0.0
        assert view._flow_phase == 0.0
        assert view._last_flow_time is None

    def test_reset_session_restores_idle_state_and_restarts_clock(self):
        view = RecordingIndicatorView()
        view._level = 0.7
        view._recording_active = True
        view._recording_active_at = 5.0
        view._start_time = 5.0
        view._flow_phase = 9.5
        view._last_flow_time = 6.0

        with patch(
            "wenzi.audio.recording_indicator.time.monotonic",
            return_value=42.0,
        ):
            view.reset_session()

        assert view._level == 0.0
        assert view._recording_active is False
        assert view._recording_active_at is None
        assert view._start_time == 42.0
        assert view._flow_phase == 0.0
        assert view._last_flow_time is None

    def test_disk_rings_are_flattened_finite_and_distinct(self):
        rings = tuple(_disk_points(index, 1.0, 0.5) for index in range(3))

        for ring in rings:
            assert len(ring) == _DISK_POINT_COUNT == 28
            assert all(math.isfinite(coordinate) for point in ring for coordinate in point)
            width = max(x for x, _ in ring) - min(x for x, _ in ring)
            height = max(y for _, y in ring) - min(y for _, y in ring)
            # Edge-on accretion disk: much wider than tall.
            assert width > height * 3.0
        assert all(
            _average_distance(rings[first], rings[second]) >= 10.0
            for first, second in ((0, 1), (0, 2), (1, 2))
        )

    def test_disk_ripple_grows_with_voice_and_ignores_room_noise(self):
        for index in range(3):
            idle = _disk_points(index, 1.0, 0.0)
            noise = _disk_points(index, 1.0, 0.22)
            speech = _disk_points(index, 1.0, 1.0)

            assert _average_distance(idle, noise) == pytest.approx(0.0)
            assert _average_distance(idle, speech) >= 1.0

    def test_flow_phase_moves_the_streaks(self):
        for index in range(3):
            before = _disk_points(index, 0.0, 0.0)
            after = _disk_points(index, 0.8, 0.0)
            assert _average_distance(before, after) >= 0.15

    def test_voice_accelerates_the_flow(self):
        def _advance(view):
            with (
                patch.object(view, "_draw_black_hole"),
                patch(
                    "wenzi.audio.recording_indicator.time.monotonic",
                    side_effect=(10.0, 10.05),
                ),
            ):
                view.draw(None)
                view.draw(None)
            return view._flow_phase

        quiet = RecordingIndicatorView()
        quiet._start_time = 10.0
        quiet_phase = _advance(quiet)

        loud = RecordingIndicatorView()
        loud._start_time = 10.0
        loud._recording_active = True
        loud._recording_active_at = None
        loud.set_level(1.0)
        loud_phase = _advance(loud)

        assert quiet_phase == pytest.approx(_FLOW_BASE_SPEED * 0.05)
        assert loud_phase == pytest.approx(
            _FLOW_BASE_SPEED * (1.0 + _FLOW_LEVEL_GAIN) * 0.05
        )
        assert loud_phase > quiet_phase

    def test_initial_activity_handoff_advances_smoothly_through_response_band(self):
        pulls = tuple(_gravity_pull(_handoff_activity(1.0, step / 6.0)) for step in range(7))

        assert pulls == tuple(sorted(pulls))
        assert pulls[0] == 0.0
        assert pulls[-1] == 1.0
        assert _gravity_pull(_handoff_activity(0.2, 0.5)) == 0.0
        assert _handoff_activity(0.4, 1.0) == 0.4
        assert _handoff_activity(1.0, -1.0) == pytest.approx(0.22)
        assert _handoff_activity(1.0, 2.0) == 1.0

    def test_recording_activation_handoff_is_timed_and_idempotent(self):
        view = RecordingIndicatorView()
        view._start_time = 10.0
        view.set_level(1.0)

        with patch(
            "wenzi.audio.recording_indicator.time.monotonic",
            side_effect=(10.4, 10.6),
        ) as now:
            view.activate_recording()
            view.activate_recording()

        assert now.call_count == 1
        assert _ACTIVITY_HANDOFF_DURATION == pytest.approx(0.30)
        assert _gravity_pull(view._effective_activity(0.4)) == 0.0
        assert _gravity_pull(
            view._effective_activity(
                0.4 + _ACTIVITY_HANDOFF_DURATION / 2.0,
            )
        ) == pytest.approx(0.5)
        assert _gravity_pull(view._effective_activity(0.4 + _ACTIVITY_HANDOFF_DURATION)) == 1.0
        assert view._recording_active_at is None

    def test_all_elements_stay_inside_rectangular_host(self):
        # The entry/exit layer scale is always <= 1.0, so the resting
        # geometry sampled here is the worst case for the host bounds.
        margin = 1.0
        for activity in (0.0, 0.22, 0.375, 0.55, 1.0):
            for step in range(40):
                phase = step * 0.37
                for index in range(3):
                    points = _curve_extent_points(
                        _disk_points(index, phase, activity),
                        closed=True,
                    )
                    padding = max(_DISK_STREAK_WIDTHS) / 2.0
                    assert min(x for x, _ in points) - padding >= margin
                    assert min(y for _, y in points) - padding >= margin
                    assert max(x for x, _ in points) + padding <= _PANEL_WIDTH - margin
                    assert max(y for _, y in points) + padding <= _PANEL_HEIGHT - margin

        # Static elements at the maximum voice gain (gradient rings fade
        # to zero alpha AT their radius, so the radius is the extent).
        shadow_extent = max(_SHADOW_RADIUS, _RIM_RADIUS) * (1.0 + _SHADOW_LEVEL_GAIN)
        top_extent = (
            _DOME_TOP_RADIUS
            * (1.0 + _DOME_LEVEL_GAIN)
            * (1.0 + _DOME_PULSE + _DOME_PULSE_B)
        )
        bottom_extent = _DOME_BOTTOM_RADIUS * (1.0 + _DOME_LEVEL_GAIN)
        horizontal_extent = max(
            shadow_extent,
            top_extent,
            bottom_extent + _DOME_BOTTOM_JITTER_X,
            _DISK_BAND_RADIUS,
        )
        assert _BH_CENTER_X - horizontal_extent >= margin
        assert _BH_CENTER_X + horizontal_extent <= _PANEL_WIDTH - margin
        vertical_extent = max(
            shadow_extent,
            top_extent * _DOME_TOP_SQUASH * (1.0 + _DOME_WOBBLE),
            bottom_extent * _DOME_BOTTOM_SQUASH * (1.0 + _DOME_BOTTOM_WOBBLE)
            + _DOME_BOTTOM_JITTER_Y,
            _DISK_BAND_RADIUS * _DISK_BAND_SQUASH,
        )
        assert _BH_CENTER_Y - vertical_extent >= margin
        assert _BH_CENTER_Y + vertical_extent <= _PANEL_HEIGHT - margin

    def test_audio_helpers_clamp_and_remain_monotonic(self):
        assert _gravity_pull(-1.0) == 0.0
        assert _gravity_pull(0.0) == 0.0
        assert _gravity_pull(0.22) == 0.0
        assert _gravity_pull(0.385) == pytest.approx(0.5)
        assert _gravity_pull(0.5) > 0.93
        assert _gravity_pull(0.55) == 1.0
        assert _gravity_pull(2.0) == 1.0
        assert _gravity_pull(0.02) < _gravity_pull(0.45) < 1.0

    def test_entry_and_exit_animation_constants(self):
        assert _ENTRY_DURATION == pytest.approx(0.22)
        assert _ENTRY_SCALE_FROM == pytest.approx(0.86)
        assert _EXIT_DURATION == pytest.approx(0.20)
        assert _EXIT_SCALE_TO == pytest.approx(0.76)
        assert _REFRESH_INTERVAL == pytest.approx(1.0 / 30.0)

    def test_center_scale_transform_composes_about_panel_center(self, monkeypatch):
        _appkit, foundation, quartz = _install_fake_native(monkeypatch)

        result = _center_scale_transform(0.86)

        quartz.CATransform3DMakeTranslation.assert_called_once_with(
            _PANEL_CENTER_X, _PANEL_CENTER_Y, 0.0
        )
        quartz.CATransform3DScale.assert_called_once_with(
            quartz.CATransform3DMakeTranslation.return_value, 0.86, 0.86, 1.0
        )
        quartz.CATransform3DTranslate.assert_called_once_with(
            quartz.CATransform3DScale.return_value,
            -_PANEL_CENTER_X,
            -_PANEL_CENTER_Y,
            0.0,
        )
        foundation.NSValue.valueWithCATransform3D_.assert_called_once_with(
            quartz.CATransform3DTranslate.return_value
        )
        assert result is foundation.NSValue.valueWithCATransform3D_.return_value

    def test_center_scale_transform_returns_nsvalue_for_core_animation(self):
        # REAL PyObjC, no fakes: CABasicAnimation from/to values must be
        # NSValue-wrapped — a raw PyObjC struct bridges as OC_PythonObject
        # and crashes the app at the next CATransaction commit
        # (CA_prepareRenderValue → unrecognized selector).
        value = _center_scale_transform(0.86)

        transform = value.CATransform3DValue()
        assert transform.m11 == pytest.approx(0.86)
        assert transform.m22 == pytest.approx(0.86)
        assert transform.m33 == pytest.approx(1.0)
        # Scaling about the panel center leaves tx = cx·(1−s), ty = cy·(1−s)
        assert transform.m41 == pytest.approx(_PANEL_CENTER_X * (1.0 - 0.86))
        assert transform.m42 == pytest.approx(_PANEL_CENTER_Y * (1.0 - 0.86))

    def test_prepare_for_display_caches_paths_colors_and_gradient(self):
        view = RecordingIndicatorView()

        view.prepare_for_display()
        initial_resources = _cached_resources(view)
        view.prepare_for_display()

        assert len(view._disk_paths) == len(view._disk_colors) == 3
        assert view._shadow_gradient is not None
        assert view._rim_gradient is not None
        assert view._band_gradient is not None
        assert view._dome_top_gradient is not None
        assert view._dome_bottom_gradient is not None
        assert view._disk_fire_gradient is not None
        assert view._glow_gradient is not None
        assert tuple(
            tuple(round(color.alphaComponent(), 3) for color in layers) for layers in view._disk_colors
        ) == (
            (0.10, 0.38),
            (0.09, 0.34),
            (0.08, 0.30),
        )
        assert all(resource is not None for resource in initial_resources)
        for current, initial in zip(
            _cached_resources(view),
            initial_resources,
            strict=True,
        ):
            assert current is initial

    def test_path_prewarm_failure_retries_without_partial_cache(self, monkeypatch):
        appkit, _foundation, _quartz = _install_fake_native(monkeypatch)
        view = RecordingIndicatorView()
        native_path = MagicMock()
        appkit.NSBezierPath.alloc.return_value.init.return_value = native_path
        native_path.setLineJoinStyle_.side_effect = RuntimeError("path setup failed")

        with pytest.raises(RuntimeError, match="path setup failed"):
            view.prepare_for_display()

        assert view._glow_gradient is None
        assert view._band_gradient is None
        assert view._shadow_gradient is None
        assert view._rim_gradient is None
        assert view._dome_top_gradient is None
        assert view._dome_bottom_gradient is None
        assert view._disk_fire_gradient is None
        assert view._disk_paths is None
        assert view._disk_colors is None

        native_path.setLineJoinStyle_.side_effect = None
        view.prepare_for_display()

        # The failed attempt allocated the three disk paths; the retry
        # allocates them again (the other elements are gradients).
        assert appkit.NSBezierPath.alloc.call_count == 6
        assert view._glow_gradient is not None
        assert len(view._disk_paths) == 3
        assert view._band_gradient is not None

    def test_rebuild_path_uses_closed_catmull_rom_sequence(self):
        path = MagicMock()
        points = _disk_points(0, 1.0, 0.5)
        expected_calls = [
            call.removeAllPoints(),
            call.moveToPoint_(points[0]),
        ]

        for index, current in enumerate(points):
            previous = points[(index - 1) % len(points)]
            following = points[(index + 1) % len(points)]
            after_following = points[(index + 2) % len(points)]
            control_one = (
                current[0] + (following[0] - previous[0]) / 6.0,
                current[1] + (following[1] - previous[1]) / 6.0,
            )
            control_two = (
                following[0] - (after_following[0] - current[0]) / 6.0,
                following[1] - (after_following[1] - current[1]) / 6.0,
            )
            expected_calls.append(
                call.curveToPoint_controlPoint1_controlPoint2_(
                    following,
                    control_one,
                    control_two,
                )
            )
        expected_calls.append(call.closePath())

        RecordingIndicatorView._rebuild_closed_path(path, points)

        assert path.mock_calls == expected_calls
        assert path.curveToPoint_controlPoint1_controlPoint2_.call_count == _DISK_POINT_COUNT

    def test_idle_first_frame_draws_resting_black_hole(self, monkeypatch):
        appkit, _foundation, _quartz = _install_fake_native(monkeypatch)
        view = RecordingIndicatorView()
        view._start_time = 10.0

        with (
            patch(
                "wenzi.audio.recording_indicator.time.monotonic",
                return_value=10.0,
            ),
            patch.object(view, "_draw_black_hole") as draw_black_hole,
        ):
            view.draw(None)

        draw_black_hole.assert_called_once_with(0.0, 0.0)
        # The very first frame has no dt yet — the flow starts from rest.
        assert view._flow_phase == 0.0
        appkit.NSGraphicsContext.saveGraphicsState.assert_not_called()

    def test_draw_uses_handoff_activity_during_microphone_takeover(self):
        view = RecordingIndicatorView()
        view._start_time = 10.0
        view._recording_active = True
        view._recording_active_at = 10.4
        view.set_level(1.0)
        expected_activity = _handoff_activity(1.0, 0.5)

        with (
            patch(
                "wenzi.audio.recording_indicator.time.monotonic",
                return_value=10.55,
            ),
            patch.object(view, "_draw_black_hole") as draw_black_hole,
        ):
            view.draw(None)

        draw_black_hole.assert_called_once()
        activity, pull = draw_black_hole.call_args.args
        assert activity == pytest.approx(expected_activity)
        assert pull == pytest.approx(_gravity_pull(expected_activity))

    def test_draw_order_layers_disk_behind_then_in_front_of_the_hole(self, monkeypatch):
        appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        view = _install_mock_resources(RecordingIndicatorView())
        view._recording_active = True
        view.set_level(1.0)
        # A phase where every infall ring is fully faded in.
        view._flow_phase = 2.0

        state_depth = [0]
        events = []

        appkit.NSGraphicsContext.saveGraphicsState.side_effect = lambda: state_depth.__setitem__(0, state_depth[0] + 1)
        appkit.NSGraphicsContext.restoreGraphicsState.side_effect = lambda: state_depth.__setitem__(0, state_depth[0] - 1)
        appkit.NSRectClip.side_effect = lambda rect: events.append("clip")
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)
        foundation.NSMakeRect.side_effect = lambda x, y, width, height: SimpleNamespace(
            x=x,
            y=y,
            width=width,
            height=height,
        )

        view._glow_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("glow")
        view._band_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("band")
        view._shadow_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("shadow")
        view._rim_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("rim")
        view._dome_top_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("dome-top")
        view._dome_bottom_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("dome-bottom")
        view._disk_fire_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("fire")
        for index, path in enumerate(view._disk_paths):
            path.stroke.side_effect = lambda ring=index: events.append(f"disk-{ring}")

        view._draw_black_hole(activity=1.0, pull=1.0)

        assert state_depth[0] == 0
        band_events = [index for index, event in enumerate(events) if event == "band"]
        disk_events = [index for index, event in enumerate(events) if event.startswith("disk-")]
        shadow_index = events.index("shadow")

        # Painter's algorithm: far half of the disk, void, upper rim
        # glow, domes, then the near half of the disk over everything.
        assert len(band_events) == 2
        assert len(disk_events) == 12
        assert events.count("shadow") == 1
        assert events.count("rim") == 1
        assert events.count("clip") == 5  # far half, rim, dome ×2, near half
        assert events.count("dome-top") == events.count("dome-bottom") == 1
        # The fire collar is part of BOTH disk passes — over the far
        # streaks (then swallowed by the void) and over the near ones.
        fire_events = [index for index, event in enumerate(events) if event == "fire"]
        assert len(fire_events) == 2
        assert events.index("glow") < band_events[0] < disk_events[0]
        # Far half of the disk is under the void...
        assert disk_events[5] < fire_events[0] < shadow_index
        # ...the rim glow wraps the void from above...
        assert shadow_index < events.index("rim")
        # ...the bent-light domes dissolve it into the disk...
        assert events.index("rim") < events.index("dome-top") < events.index("dome-bottom")
        # ...and the near half of the disk crosses in front of it all,
        # its inner edge burning around the hole.
        assert events.index("dome-bottom") < band_events[1] < disk_events[6]
        assert disk_events[11] < fire_events[1]

        # Voice pushes the void, its rim and the domes outward; the top
        # halo additionally pulses with the flow phase.
        shadow_radius = view._shadow_gradient.drawFromCenter_radius_toCenter_radius_options_.call_args.args[3]
        assert shadow_radius == pytest.approx(_SHADOW_RADIUS * (1.0 + _SHADOW_LEVEL_GAIN))
        rim_radius = view._rim_gradient.drawFromCenter_radius_toCenter_radius_options_.call_args.args[3]
        assert rim_radius == pytest.approx(_RIM_RADIUS * (1.0 + _SHADOW_LEVEL_GAIN))
        dome_radius = view._dome_top_gradient.drawFromCenter_radius_toCenter_radius_options_.call_args.args[3]
        expected_dome_radius = (
            _DOME_TOP_RADIUS
            * (1.0 + _DOME_LEVEL_GAIN)
            * (
                1.0
                + _DOME_PULSE * math.sin(2.0 * _DOME_PULSE_RATE)
                + _DOME_PULSE_B * math.sin(2.0 * _DOME_PULSE_RATE_B + 1.7)
            )
        )
        assert dome_radius == pytest.approx(expected_dome_radius)
        # The bottom halo TREMBLES with the voice instead of swelling:
        # no pulse on its radius, but the center jitters (pull is 1
        # here, so the full amplitude applies).
        bottom_call = view._dome_bottom_gradient.drawFromCenter_radius_toCenter_radius_options_.call_args
        assert bottom_call.args[3] == pytest.approx(
            _DOME_BOTTOM_RADIUS * (1.0 + _DOME_LEVEL_GAIN)
        )
        assert bottom_call.args[0].x == pytest.approx(
            _BH_CENTER_X
            + _DOME_BOTTOM_JITTER_X * math.sin(2.0 * _DOME_BOTTOM_JITTER_X_RATE + 0.7)
        )
        # The fire collar breathes with the void, like the rim.
        fire_radius = view._disk_fire_gradient.drawFromCenter_radius_toCenter_radius_options_.call_args.args[3]
        assert fire_radius == pytest.approx(_DISK_FIRE_RADIUS * (1.0 + _SHADOW_LEVEL_GAIN))

    def test_streak_rings_spiral_into_the_hole(self):
        """The rings fall from the outer disk to the rim (accelerating)
        and end up small enough for the shadow to swallow them."""
        rate_phase = 1.0 / _INFALL_RATE  # one full infall cycle for ring 0

        def extent(cycle):
            points = _disk_points(0, cycle * rate_phase, 0.0)
            return max(abs(x - _BH_CENTER_X) for x, _ in points)

        extents = [extent(cycle) for cycle in (0.05, 0.3, 0.6, 0.9)]
        assert extents == sorted(extents, reverse=True)
        # Accelerating: the drop over the last third beats the first third
        assert (extents[2] - extents[3]) > (extents[0] - extents[1])
        # Swallowed: by the end of the cycle the ring is inside the rim
        assert extent(0.97) < _RIM_RADIUS

    def test_halo_hollows_clear_the_black_core(self):
        """The transparent middles of both halos must be at least as
        large as the black core on BOTH axes — otherwise their glow
        bleeds inside the void.  For the top halo the binding case is
        the pulse/wobble MINIMUM."""
        core = _SHADOW_RADIUS * _SHADOW_CORE_STOP

        # The bottom halo trembles (center jitter + aspect wobble); the
        # hollow must clear the core even at the jittered extreme.
        bottom_inner = _DOME_BOTTOM_RADIUS * _DOME_BOTTOM_INNER_STOP
        assert bottom_inner - _DOME_BOTTOM_JITTER_X >= core
        assert (
            bottom_inner * _DOME_BOTTOM_SQUASH * (1.0 - _DOME_BOTTOM_WOBBLE)
            - _DOME_BOTTOM_JITTER_Y
            >= core
        )

        top_inner_worst = (
            _DOME_TOP_RADIUS
            * _DOME_TOP_INNER_STOP
            * (1.0 - _DOME_PULSE - _DOME_PULSE_B)
        )
        assert top_inner_worst >= core
        assert top_inner_worst * _DOME_TOP_SQUASH * (1.0 - _DOME_WOBBLE) >= core

        # The domes grow at least as fast as the shadow with the voice,
        # so voice gain can never invert the clearance.
        assert _DOME_LEVEL_GAIN >= _SHADOW_LEVEL_GAIN

    def test_disk_fire_collar_clears_the_swollen_core(self):
        """The fire collar on the disk's inner edge may ride the core's
        feathered edge (like the top rim's ember does) but must stay
        outside the OPAQUE black — deeper in, the near pass paints a
        fire ring floating on the void's face.  It breathes in lockstep
        with the void (same gain, asserted in the draw-order test), so
        the resting geometry is the only case."""
        ember = _DISK_FIRE_RADIUS * _DISK_FIRE_EMBER_STOP
        assert ember >= _SHADOW_RADIUS * _SHADOW_CORE_STOP
        # And the collar never outgrows the band it lives on.
        assert _DISK_FIRE_RADIUS <= _DISK_BAND_RADIUS

    def test_ring_dissolves_before_entering_the_shadow(self, monkeypatch):
        """A ring at the very end of its infall must not be stroked at
        all — the near-disk pass would otherwise show it hovering over
        the void."""
        _appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        view = _install_mock_resources(RecordingIndicatorView())
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)
        view._flow_phase = 0.995 / _INFALL_RATE  # ring 0 nearly swallowed

        view._draw_black_hole(activity=0.0, pull=0.0)

        assert view._disk_paths[0].stroke.call_count == 0
        assert view._disk_paths[1].stroke.call_count == 4
        assert view._disk_paths[2].stroke.call_count == 4

    def test_draw_loop_reuses_paths_and_allocates_no_native_resources(self, monkeypatch):
        appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        view = _install_mock_resources(RecordingIndicatorView())
        initial_resources = _cached_resources(view)
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)

        view._flow_phase = 2.0  # every infall ring fully faded in
        view._draw_black_hole(activity=0.5, pull=_gravity_pull(0.5))
        view._draw_black_hole(activity=0.55, pull=_gravity_pull(0.55))

        for current, initial in zip(
            _cached_resources(view),
            initial_resources,
            strict=True,
        ):
            assert current is initial
        assert not appkit.NSColor.mock_calls
        assert not appkit.NSGradient.mock_calls
        assert not appkit.NSBezierPath.mock_calls
        assert not appkit.NSShadow.mock_calls
        assert not appkit.CIFilter.mock_calls

        # Only the disk streak rings are rebuilt per frame (once each,
        # then stroked in both the far and near passes).
        for path in view._disk_paths:
            assert path.removeAllPoints.call_count == 2
            assert path.moveToPoint_.call_count == 2
            assert path.curveToPoint_controlPoint1_controlPoint2_.call_count == _DISK_POINT_COUNT * 2
            assert path.closePath.call_count == 2
            assert path.stroke.call_count == 8  # halo+core × 2 passes × 2 frames
        assert view._glow_gradient.drawFromCenter_radius_toCenter_radius_options_.call_count == 2
        assert view._band_gradient.drawFromCenter_radius_toCenter_radius_options_.call_count == 4
        assert view._shadow_gradient.drawFromCenter_radius_toCenter_radius_options_.call_count == 2
        assert view._rim_gradient.drawFromCenter_radius_toCenter_radius_options_.call_count == 2
        assert view._dome_top_gradient.drawFromCenter_radius_toCenter_radius_options_.call_count == 2
        assert view._dome_bottom_gradient.drawFromCenter_radius_toCenter_radius_options_.call_count == 2
        assert view._disk_fire_gradient.drawFromCenter_radius_toCenter_radius_options_.call_count == 4

        path_segments_per_frame = len(view._disk_paths) * _DISK_POINT_COUNT
        strokes_per_frame = 12  # wave halo+core × 3 rings × 2 passes
        # glow + (band + fire collar)×2 + void + rim + domes
        gradient_draws_per_frame = 1 + 4 + 1 + 1 + 2
        assert path_segments_per_frame == 84
        assert strokes_per_frame == 12
        assert gradient_draws_per_frame == 9

    def test_clipped_dome_failure_restores_graphics_state(self, monkeypatch):
        appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        view = _install_mock_resources(RecordingIndicatorView())
        state_depth = [0]

        appkit.NSGraphicsContext.saveGraphicsState.side_effect = lambda: state_depth.__setitem__(0, state_depth[0] + 1)
        appkit.NSGraphicsContext.restoreGraphicsState.side_effect = lambda: state_depth.__setitem__(0, state_depth[0] - 1)
        appkit.NSRectClip.side_effect = RuntimeError("clip failed")
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)

        with pytest.raises(RuntimeError, match="clip failed"):
            view._draw_black_hole(activity=0.5, pull=0.5)

        assert state_depth[0] == 0

    def test_failed_disk_rebuild_recovers_next_frame(self, monkeypatch):
        _appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        view = _install_mock_resources(RecordingIndicatorView())
        view._start_time = 10.0
        view._flow_phase = 2.0  # every infall ring fully faded in
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)
        failing_path = view._disk_paths[0]
        failing_path.curveToPoint_controlPoint1_controlPoint2_.side_effect = RuntimeError("curve failed")

        with patch(
            "wenzi.audio.recording_indicator.time.monotonic",
            return_value=11.0,
        ):
            with pytest.raises(RuntimeError, match="curve failed"):
                view.draw(None)

            # The rebuild died before any disk stroke or the shadow draw.
            assert all(path.stroke.call_count == 0 for path in view._disk_paths)
            view._shadow_gradient.drawFromCenter_radius_toCenter_radius_options_.assert_not_called()

            failing_path.curveToPoint_controlPoint1_controlPoint2_.side_effect = None
            view.draw(None)

        assert failing_path.removeAllPoints.call_count == 2
        assert view._shadow_gradient.drawFromCenter_radius_toCenter_radius_options_.call_count == 1
        assert all(path.stroke.call_count == 4 for path in view._disk_paths)


def _host_mocks(appkit, foundation):
    """Prepare fake NSPanel/NSScreen/NSTimer plumbing for panel tests."""
    native_panel = MagicMock(name="native_panel")
    timer = MagicMock(name="timer")
    appkit.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = native_panel
    screen_frame = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=50.0),
        size=SimpleNamespace(width=1000.0, height=700.0),
    )
    appkit.NSScreen.mainScreen.return_value.visibleFrame.return_value = screen_frame
    foundation.NSMakeRect.side_effect = lambda *args: args
    foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_.return_value = timer
    return native_panel, timer


def _patched_view(indicator):
    """Patch RecordingIndicatorView natives so a real view instance is used."""

    def _create_view(indicator_view, width, height):
        indicator_view._view = indicator
        indicator._indicator = indicator_view
        return indicator

    return (
        patch.object(RecordingIndicatorView, "prepare_for_display"),
        patch.object(
            RecordingIndicatorView,
            "create_view",
            autospec=True,
            side_effect=_create_view,
        ),
    )


class TestRecordingIndicatorPanel:
    def test_initial_state(self):
        panel = RecordingIndicatorPanel()

        assert panel.enabled is True
        assert panel.show_device_name is False
        assert panel._panel is None
        assert panel._timer is None
        assert panel._visible is False
        assert panel._show_gen == 0

    def test_enabled_and_show_device_name_toggles(self):
        panel = RecordingIndicatorPanel()

        panel.show_device_name = True
        assert panel.show_device_name is True
        panel.enabled = False
        assert panel.enabled is False

    def test_show_builds_transparent_host_and_runs_entry_animation(self, monkeypatch):
        appkit, foundation, quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        native_panel, timer = _host_mocks(appkit, foundation)
        indicator = MagicMock(name="indicator_ns_view")
        layer = indicator.layer.return_value

        events = []
        native_panel.setContentView_.side_effect = lambda view: events.append("content")
        native_panel.setAlphaValue_.side_effect = lambda value: events.append(f"alpha-{value}")
        native_panel.animator.return_value.setAlphaValue_.side_effect = lambda value: events.append(f"alpha-animator-{value}")
        indicator.setNeedsDisplay_.side_effect = lambda value: events.append("needs-display")
        indicator.displayIfNeededIgnoringOpacity.side_effect = lambda: events.append("first-frame")
        native_panel.orderFront_.side_effect = lambda sender: events.append("order-front")
        layer.addAnimation_forKey_.side_effect = lambda anim, key: events.append(f"layer-{key}")
        foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_.side_effect = lambda *args: (
            events.append("timer") or timer
        )

        prepare_patch, create_patch = _patched_view(indicator)
        with prepare_patch as prepare, create_patch as create_view:
            prepare.side_effect = lambda: events.append("prepare")
            panel.show("MacBook Pro Microphone", "Proofread")

        prepare.assert_called_once()
        appkit.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.assert_called_once_with(
            (0, 0, _PANEL_WIDTH, _PANEL_HEIGHT), 0, 2, False
        )
        native_panel.setOpaque_.assert_called_once_with(False)
        native_panel.setBackgroundColor_.assert_called_once_with(appkit.NSColor.clearColor.return_value)
        native_panel.setIgnoresMouseEvents_.assert_called_once_with(True)
        native_panel.setHasShadow_.assert_called_once_with(False)
        native_panel.setFrameOrigin_.assert_called_once_with(
            (480.0, 50.0 + (700.0 - _PANEL_HEIGHT) * _SCREEN_VERTICAL_BIAS)
        )
        assert create_view.call_args.args[1:] == (_PANEL_WIDTH, _PANEL_HEIGHT)
        native_panel.setContentView_.assert_called_once_with(indicator)
        indicator.setNeedsDisplay_.assert_called_once_with(True)
        indicator.displayIfNeededIgnoringOpacity.assert_called_once_with()
        native_panel.orderOut_.assert_not_called()

        # The hard alpha cut is gone: 0.0 is set directly, 1.0 only ever
        # through the animator inside the entry animation.
        assert native_panel.setAlphaValue_.call_args_list == [((0.0,), {})]
        native_panel.animator.return_value.setAlphaValue_.assert_called_once_with(1.0)
        assert events == [
            "prepare",
            "content",
            "alpha-0.0",
            "order-front",
            "needs-display",
            "first-frame",
            "layer-wenzi.entry",
            "alpha-animator-1.0",
            "timer",
        ]

        layer.removeAllAnimations.assert_called_once_with()
        quartz.CABasicAnimation.animationWithKeyPath_.assert_called_once_with("transform")
        entry = quartz.CABasicAnimation.animationWithKeyPath_.return_value
        entry.setFromValue_.assert_called_once_with(
            foundation.NSValue.valueWithCATransform3D_.return_value
        )
        entry.setDuration_.assert_called_once_with(pytest.approx(_ENTRY_DURATION))
        quartz.CAMediaTimingFunction.functionWithName_.assert_called_with(quartz.kCAMediaTimingFunctionEaseOut)
        appkit.NSAnimationContext.currentContext.return_value.setDuration_.assert_called_once_with(
            pytest.approx(_ENTRY_DURATION)
        )

        foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once_with(
            pytest.approx(_REFRESH_INTERVAL), indicator, b"refresh:", None, True
        )
        timer.setTolerance_.assert_called_once_with(pytest.approx(_REFRESH_INTERVAL * 0.2))
        foundation.NSRunLoop.currentRunLoop.return_value.addTimer_forMode_.assert_called_once_with(
            timer, foundation.NSRunLoopCommonModes
        )
        assert panel._visible is True
        assert panel._show_gen == 1
        assert panel._mode_name == "Proofread"
        assert panel._device_name == "MacBook Pro Microphone"

    def test_show_disabled_does_nothing(self):
        panel = RecordingIndicatorPanel()
        panel.enabled = False

        with patch("wenzi.audio.recording_indicator.RecordingIndicatorView") as view_type:
            panel.show()

        assert panel._panel is None
        assert panel._timer is None
        view_type.assert_not_called()

    def test_prewarm_builds_panel_and_palette_without_ordering_front(self, monkeypatch):
        appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        native_panel, _timer = _host_mocks(appkit, foundation)
        indicator = MagicMock(name="indicator_ns_view")

        prepare_patch, create_patch = _patched_view(indicator)
        with prepare_patch as prepare, create_patch:
            panel.prewarm()

        prepare.assert_called_once()
        assert panel._panel is native_panel
        assert panel._indicator_view is not None
        assert panel._visible is False
        assert panel.current_frame is None
        native_panel.orderFront_.assert_not_called()
        native_panel.setAlphaValue_.assert_not_called()
        foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_.assert_not_called()

    def test_prewarm_failure_leaves_clean_state_for_cold_show(self, monkeypatch):
        appkit, _foundation, _quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        appkit.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.side_effect = RuntimeError(
            "panel init failed"
        )

        prepare_patch, create_patch = _patched_view(MagicMock())
        with prepare_patch, create_patch:
            panel.prewarm()  # must not raise

        assert panel._panel is None
        assert panel._indicator_view is None
        assert panel._timer is None
        assert panel._visible is False

    def test_prewarm_disabled_builds_nothing(self):
        panel = RecordingIndicatorPanel()
        panel.enabled = False

        with patch("wenzi.audio.recording_indicator.RecordingIndicatorView") as view_type:
            panel.prewarm()

        assert panel._panel is None
        view_type.assert_not_called()

    def test_show_after_prewarm_reuses_panel_view_and_palette(self, monkeypatch):
        appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        native_panel, timer = _host_mocks(appkit, foundation)
        indicator = MagicMock(name="indicator_ns_view")

        prepare_patch, create_patch = _patched_view(indicator)
        with prepare_patch as prepare, create_patch as create_view:
            panel.prewarm()
            built_view = panel._indicator_view
            panel.show("Mic", "Proofread")
            panel.hide()
            panel.show("Mic", "Proofread")

        # One cold build, then pure reuse.
        appkit.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.assert_called_once()
        prepare.assert_called_once()
        assert create_view.call_count == 1
        assert panel._panel is native_panel
        assert panel._indicator_view is built_view
        assert native_panel.orderFront_.call_count == 2
        assert panel._show_gen == 2
        assert timer.invalidate.call_count == 1  # invalidated by hide()

    def test_show_resets_session_state_and_removes_stale_layer_animations(self, monkeypatch):
        appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        _host_mocks(appkit, foundation)
        indicator = MagicMock(name="indicator_ns_view")
        layer = indicator.layer.return_value

        prepare_patch, create_patch = _patched_view(indicator)
        with prepare_patch, create_patch:
            panel.show()
            view = panel._indicator_view
            view._level = 0.7
            view._recording_active = True
            view._recording_active_at = 1.0
            view._flow_phase = 7.0
            panel._smoothed_level = 0.5
            panel._level_mean = 0.4
            panel.show()

        assert view._level == 0.0
        assert view._recording_active is False
        assert view._recording_active_at is None
        assert view._flow_phase == 0.0
        assert panel._smoothed_level == 0.0
        assert panel._level_mean is None
        # Once per show: clears the exit animation's held fillMode state.
        assert layer.removeAllAnimations.call_count == 2

    def test_show_recenters_on_current_main_screen_each_show(self, monkeypatch):
        appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        native_panel, _timer = _host_mocks(appkit, foundation)
        indicator = MagicMock(name="indicator_ns_view")
        frames = [
            SimpleNamespace(
                origin=SimpleNamespace(x=100.0, y=50.0),
                size=SimpleNamespace(width=1000.0, height=700.0),
            ),
            SimpleNamespace(
                origin=SimpleNamespace(x=0.0, y=0.0),
                size=SimpleNamespace(width=2000.0, height=1200.0),
            ),
        ]
        appkit.NSScreen.mainScreen.return_value.visibleFrame.side_effect = frames

        prepare_patch, create_patch = _patched_view(indicator)
        with prepare_patch, create_patch:
            panel.show()
            panel.hide()
            panel.show()

        assert native_panel.setFrameOrigin_.call_args_list == [
            call((480.0, 50.0 + (700.0 - _PANEL_HEIGHT) * _SCREEN_VERTICAL_BIAS)),
            call(
                (
                    (2000.0 - _PANEL_WIDTH) / 2.0,
                    (1200.0 - _PANEL_HEIGHT) * _SCREEN_VERTICAL_BIAS,
                )
            ),
        ]

    def test_show_failure_tears_down_for_cold_rebuild(self, monkeypatch):
        appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        native_panel, _timer = _host_mocks(appkit, foundation)
        indicator = MagicMock(name="indicator_ns_view")
        indicator.displayIfNeededIgnoringOpacity.side_effect = RuntimeError("draw failed")

        prepare_patch, create_patch = _patched_view(indicator)
        with prepare_patch, create_patch:
            panel.show("Microphone", "Proofread")

        native_panel.setAlphaValue_.assert_called_once_with(0.0)
        native_panel.orderFront_.assert_called_once_with(None)
        native_panel.orderOut_.assert_called_once_with(None)
        foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_.assert_not_called()
        assert indicator._indicator is None
        assert panel._panel is None
        assert panel._timer is None
        assert panel._indicator_view is None
        assert panel._visible is False

    def test_disabling_indicator_tears_down_cached_panel(self, monkeypatch):
        appkit, foundation, _quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        native_panel, _timer = _host_mocks(appkit, foundation)
        indicator = MagicMock(name="indicator_ns_view")

        prepare_patch, create_patch = _patched_view(indicator)
        with prepare_patch, create_patch:
            panel.prewarm()
            assert panel._panel is native_panel
            panel.enabled = False

        assert panel._panel is None
        assert panel._indicator_view is None
        assert indicator._indicator is None
        native_panel.orderOut_.assert_called()

    def test_update_level_uses_asymmetric_ema_and_clamps(self):
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()

        # The first sample seeds the local mean: no artificial swing.
        panel.update_level(0.05)
        assert panel._smoothed_level == pytest.approx(0.0)
        # A jump far above the mean saturates the swing → fast attack.
        panel.update_level(1.0)
        assert panel._smoothed_level == pytest.approx(0.7)
        panel.update_level(1.0)
        assert panel._smoothed_level == pytest.approx(0.91)
        # Falling back near the (risen) mean → soft release.
        panel.update_level(0.2)
        assert panel._smoothed_level == pytest.approx(0.732, abs=1e-3)
        assert panel._indicator_view._level == pytest.approx(0.732, abs=1e-3)

    def test_shape_rests_after_speech_despite_room_noise(self):
        """Stopping speech must relax the disk even when steady ambient
        noise (music, fans) keeps the absolute level above the voice
        floor: a flat signal has no modulation."""
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()

        for _ in range(10):
            panel.update_level(0.9)  # syllable peaks over room noise
            panel.update_level(0.45)
        assert _gravity_pull(panel._indicator_view._level) > 0.9

        for _ in range(40):
            panel.update_level(0.35)  # speech stopped; noise persists
        assert _gravity_pull(panel._indicator_view._level) == 0.0

    def test_continuous_speech_keeps_disk_rippling(self):
        """As long as the signal is modulated (someone is talking), the
        disk must keep rippling — it must NOT fade out mid-sentence."""
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()

        panel.update_level(0.1)  # settle the local mean while quiet
        for _ in range(3):  # speech onset
            panel.update_level(1.0)
            panel.update_level(0.5)
        for _ in range(60):  # a long uninterrupted sentence
            panel.update_level(1.0)
            assert _gravity_pull(panel._indicator_view._level) == 1.0
            panel.update_level(0.5)
            assert _gravity_pull(panel._indicator_view._level) == 1.0

    def test_pause_then_resume_deforms_again(self):
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()

        panel.update_level(0.1)
        for _ in range(4):
            panel.update_level(1.0)
            panel.update_level(0.5)
        assert _gravity_pull(panel._indicator_view._level) == 1.0

        for _ in range(40):
            panel.update_level(0.35)  # pause: steady room noise
        assert _gravity_pull(panel._indicator_view._level) == 0.0

        panel.update_level(1.0)  # resume speaking
        assert _gravity_pull(panel._indicator_view._level) == 1.0

    def test_first_speech_sample_ripples_the_disk(self):
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()

        # Quiet room seeds the mean; recorder RMS 1500 maps to 0.625 —
        # one attack tick must already be fully visible.
        panel.update_level(0.05)
        panel.update_level(0.625)
        activity = panel._indicator_view._level

        assert activity == pytest.approx(0.7)
        assert _gravity_pull(activity) > 0.7
        for index in range(3):
            assert (
                _average_distance(
                    _disk_points(index, 1.0, 0.0),
                    _disk_points(index, 1.0, activity),
                )
                >= 1.0
            )

        noise_panel = RecordingIndicatorPanel()
        noise_panel._indicator_view = RecordingIndicatorView()
        noise_panel.update_level(0.25)  # steady noise seeds the mean
        assert noise_panel._indicator_view._level == pytest.approx(0.0)
        assert _gravity_pull(noise_panel._indicator_view._level) == 0.0

    def test_repeated_level_updates_converge_and_return_to_quiet(self):
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()

        for _ in range(5):
            panel.update_level(1.0)
            panel.update_level(0.4)
        assert panel._smoothed_level > 0.9
        assert _gravity_pull(panel._indicator_view._level) > 0.99

        for _ in range(40):
            panel.update_level(0.0)
        assert panel._smoothed_level < 0.05
        assert _gravity_pull(panel._indicator_view._level) == 0.0

    def test_set_recording_active_updates_view(self):
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()
        panel._indicator_view._view = MagicMock()

        panel.set_recording_active()

        assert panel._indicator_view._recording_active is True
        assert panel._indicator_view._recording_active_at is not None
        panel._indicator_view._view.setNeedsDisplay_.assert_called_once_with(True)

    def test_legacy_mode_and_device_updates_add_no_visual_content(self):
        panel = RecordingIndicatorPanel()
        panel._panel = MagicMock()
        panel._visible = True
        panel._indicator_view = RecordingIndicatorView()

        panel.update_mode("Translate EN")
        panel.update_device_name("MacBook Pro Microphone")

        assert panel._mode_name == "Translate EN"
        assert panel._device_name == "MacBook Pro Microphone"
        assert not hasattr(panel._indicator_view, "_subtitle")
        assert not hasattr(panel._indicator_view, "_status_text")

    def test_update_device_name_ignored_while_hidden(self):
        panel = RecordingIndicatorPanel()
        panel._panel = MagicMock()
        panel._visible = False

        panel.update_device_name("MacBook Pro Microphone")

        assert panel._device_name is None

    def test_clear_mode_keeps_orb_state(self):
        panel = RecordingIndicatorPanel()
        view = RecordingIndicatorView()
        panel._indicator_view = view
        panel._mode_name = "Proofread"

        panel.clear_mode()

        assert panel._mode_name is None
        assert panel._indicator_view is view

    def test_hide_orders_out_and_keeps_reusable_host(self):
        panel = RecordingIndicatorPanel()
        timer = MagicMock()
        native_panel = MagicMock()
        native_view = MagicMock()
        indicator_view = RecordingIndicatorView()
        indicator_view._view = native_view
        native_view._indicator = indicator_view
        panel._timer = timer
        panel._panel = native_panel
        panel._indicator_view = indicator_view
        panel._visible = True
        panel._smoothed_level = 0.5
        panel._mode_name = "Proofread"
        panel._device_name = "Mic"

        panel.hide()

        timer.invalidate.assert_called_once()
        native_panel.orderOut_.assert_called_once_with(None)
        assert panel._timer is None
        # The host survives for the next show()
        assert panel._panel is native_panel
        assert panel._indicator_view is indicator_view
        assert native_view._indicator is indicator_view
        assert panel._visible is False
        assert panel._smoothed_level == 0.0
        assert panel._mode_name is None
        assert panel._device_name is None

    def test_hide_tears_down_when_native_cleanup_fails(self):
        panel = RecordingIndicatorPanel()
        timer = MagicMock()
        timer.invalidate.side_effect = RuntimeError("timer failed")
        native_panel = MagicMock()
        native_panel.orderOut_.side_effect = RuntimeError("panel failed")
        native_view = MagicMock()
        indicator_view = RecordingIndicatorView()
        indicator_view._view = native_view
        native_view._indicator = indicator_view
        panel._timer = timer
        panel._panel = native_panel
        panel._indicator_view = indicator_view
        panel._visible = True

        panel.hide()

        timer.invalidate.assert_called_once()
        # Broken native state must not be kept for reuse.
        assert panel._timer is None
        assert panel._panel is None
        assert panel._indicator_view is None
        assert native_view._indicator is None
        assert panel._visible is False

    def test_current_frame(self):
        panel = RecordingIndicatorPanel()
        assert panel.current_frame is None

        native_panel = MagicMock()
        panel._panel = native_panel
        # Built but hidden: still no frame (preview morphs only from a
        # visible orb).
        assert panel.current_frame is None

        panel._visible = True
        assert panel.current_frame is native_panel.frame.return_value

    def test_animate_out_calls_completion_when_hidden(self):
        panel = RecordingIndicatorPanel()
        completion = MagicMock()

        panel.animate_out(completion)
        completion.assert_called_once()

        # Built-but-hidden panel behaves the same
        panel._panel = MagicMock()
        panel._visible = False
        second_completion = MagicMock()
        panel.animate_out(second_completion)
        second_completion.assert_called_once()

    def test_animate_out_stops_timer_and_orders_out_on_completion(self, monkeypatch):
        appkit, foundation, quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        native_panel = MagicMock()
        native_view = MagicMock()
        indicator_view = RecordingIndicatorView()
        indicator_view._view = native_view
        panel._panel = native_panel
        panel._indicator_view = indicator_view
        panel._visible = True
        panel._show_gen = 3
        timer = MagicMock()
        panel._timer = timer
        completion = MagicMock()

        captured = {}
        context = appkit.NSAnimationContext.currentContext.return_value
        context.setCompletionHandler_.side_effect = lambda callback: captured.setdefault("callback", callback)

        panel.animate_out(completion)

        # The refresh timer stops immediately: the exit is pure CA.
        timer.invalidate.assert_called_once()
        assert panel._timer is None

        quartz.CABasicAnimation.animationWithKeyPath_.assert_called_once_with("transform")
        shrink = quartz.CABasicAnimation.animationWithKeyPath_.return_value
        shrink.setToValue_.assert_called_once_with(
            foundation.NSValue.valueWithCATransform3D_.return_value
        )
        shrink.setDuration_.assert_called_once_with(pytest.approx(_EXIT_DURATION))
        shrink.setFillMode_.assert_called_once_with(quartz.kCAFillModeForwards)
        shrink.setRemovedOnCompletion_.assert_called_once_with(False)
        native_view.layer.return_value.addAnimation_forKey_.assert_called_once_with(shrink, "wenzi.exit")
        quartz.CAMediaTimingFunction.functionWithName_.assert_called_with(quartz.kCAMediaTimingFunctionEaseIn)
        native_panel.animator.return_value.setAlphaValue_.assert_called_once_with(0.0)
        context.setDuration_.assert_called_once_with(pytest.approx(_EXIT_DURATION))
        native_panel.orderOut_.assert_not_called()
        completion.assert_not_called()

        captured["callback"]()

        native_panel.orderOut_.assert_called_once_with(None)
        completion.assert_called_once()
        assert panel._visible is False
        # The host survives for the next show()
        assert panel._panel is native_panel
        assert panel._indicator_view is indicator_view

    def test_stale_animate_completion_cannot_clear_new_orb(self, monkeypatch):
        appkit, _foundation, _quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        native_panel = MagicMock()
        native_view = MagicMock()
        indicator_view = RecordingIndicatorView()
        indicator_view._view = native_view
        panel._panel = native_panel
        panel._indicator_view = indicator_view
        panel._visible = True
        panel._show_gen = 1
        completion = MagicMock()

        captured = {}
        context = appkit.NSAnimationContext.currentContext.return_value
        context.setCompletionHandler_.side_effect = lambda callback: captured.setdefault("callback", callback)

        panel.animate_out(completion)

        # A newer show() takes over the REUSED panel before the fade ends
        panel._show_gen += 1
        panel._visible = True
        new_timer = MagicMock()
        panel._timer = new_timer

        captured["callback"]()

        native_panel.orderOut_.assert_not_called()
        assert panel._visible is True
        assert panel._timer is new_timer
        new_timer.invalidate.assert_not_called()
        completion.assert_not_called()

    def test_animate_out_failure_falls_back_to_hide(self, monkeypatch):
        appkit, _foundation, quartz = _install_fake_native(monkeypatch)
        panel = RecordingIndicatorPanel()
        native_panel = MagicMock()
        native_view = MagicMock()
        indicator_view = RecordingIndicatorView()
        indicator_view._view = native_view
        panel._panel = native_panel
        panel._indicator_view = indicator_view
        panel._visible = True
        completion = MagicMock()
        quartz.CABasicAnimation.animationWithKeyPath_.side_effect = RuntimeError("no CA")

        panel.animate_out(completion)

        native_panel.orderOut_.assert_called_once_with(None)
        assert panel._visible is False
        completion.assert_called_once()
