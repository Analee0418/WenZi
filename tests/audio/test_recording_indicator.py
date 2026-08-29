"""Tests for the lightweight recording orb."""

import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from wenzi.audio.recording_indicator import (
    _CORONA_POINT_COUNT,
    _CORONA_STROKE_WIDTHS,
    _ENTRY_DURATION,
    _ORB_CENTER_X,
    _ORB_CENTER_Y,
    _OUTLINE_POINT_COUNT,
    _PANEL_CENTER_X,
    _PANEL_CENTER_Y,
    _PANEL_HEIGHT,
    _PANEL_WIDTH,
    _WAVE_BASE_Y,
    _WAVE_CLIP_HEIGHT,
    _WAVE_OFFSETS,
    _WAVE_POINT_COUNT,
    _WAVE_STROKE_WIDTHS,
    RecordingIndicatorPanel,
    RecordingIndicatorView,
    _corona_points,
    _entry_scale,
    _exit_scale,
    _gravity_pull,
    _halo_radius,
    _orb_outline_points,
    _orb_scale,
    _rim_widths,
    _wave_points,
)


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


def _scale_about(point, scale, center):
    return (
        center[0] + (point[0] - center[0]) * scale,
        center[1] + (point[1] - center[1]) * scale,
    )


def _install_mock_resources(view):
    """Install a complete prewarmed cache without allocating AppKit objects."""
    view._halo_gradient = MagicMock(name="halo_gradient")
    view._orb_body_gradient = MagicMock(name="orb_body_gradient")
    view._orb_accent_gradients = (
        MagicMock(name="orb_accent_0"),
        MagicMock(name="orb_accent_1"),
    )
    view._orb_depth_gradient = MagicMock(name="orb_depth_gradient")
    view._orb_bloom_gradient = MagicMock(name="orb_bloom_gradient")
    view._orb_path = MagicMock(name="orb_path")
    view._rim_colors = tuple(MagicMock(name=f"rim_color_{index}") for index in range(3))
    view._corona_paths = tuple(MagicMock(name=f"corona_path_{index}") for index in range(3))
    view._corona_colors = tuple(
        (
            MagicMock(name=f"corona_far_glow_{index}"),
            MagicMock(name=f"corona_near_glow_{index}"),
            MagicMock(name=f"corona_core_{index}"),
        )
        for index in range(3)
    )
    view._wave_paths = tuple(MagicMock(name=f"wave_path_{index}") for index in range(3))
    view._wave_colors = tuple(
        (
            MagicMock(name=f"wave_far_glow_{index}"),
            MagicMock(name=f"wave_near_glow_{index}"),
            MagicMock(name=f"wave_core_{index}"),
        )
        for index in range(3)
    )
    view._wave_edge_gradient = MagicMock(name="wave_edge_gradient")
    return view


def _cached_resources(view):
    return (
        view._halo_gradient,
        view._orb_body_gradient,
        view._orb_accent_gradients,
        view._orb_depth_gradient,
        view._orb_bloom_gradient,
        view._orb_path,
        view._rim_colors,
        view._corona_paths,
        view._corona_colors,
        view._wave_paths,
        view._wave_colors,
        view._wave_edge_gradient,
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
        assert view._level == 0.0
        assert view._exit_started_at is None

    def test_outline_has_exactly_24_finite_points(self):
        points = _orb_outline_points(1.0, 0.5, 0.5)

        assert len(points) == _OUTLINE_POINT_COUNT == 24
        assert all(math.isfinite(coordinate) for point in points for coordinate in point)
        assert all(math.hypot(x - _ORB_CENTER_X, y - _ORB_CENTER_Y) > 0.0 for x, y in points)

    def test_noise_stays_quiet_while_speech_has_clear_visual_response(self):
        idle = _orb_outline_points(1.0, 0.0, 0.5)
        room_noise = _orb_outline_points(1.0, 0.22, 0.5)
        speech = _orb_outline_points(1.0, 0.6, 0.5)

        noise_deformation = (
            sum(math.dist(idle_point, noise_point) for idle_point, noise_point in zip(idle, room_noise, strict=True)) / _OUTLINE_POINT_COUNT
        )
        speech_deformation = (
            sum(math.dist(idle_point, speech_point) for idle_point, speech_point in zip(idle, speech, strict=True)) / _OUTLINE_POINT_COUNT
        )

        assert noise_deformation == pytest.approx(0.0)
        assert speech_deformation >= 3.0
        assert _orb_scale(0.22) == 1.0
        assert _orb_scale(0.6) > 1.05
        for idle_width, noise_width, speech_width in zip(
            _rim_widths(0.0),
            _rim_widths(0.22),
            _rim_widths(0.6),
            strict=True,
        ):
            assert idle_width == noise_width
            assert speech_width > idle_width

    def test_idle_breath_visibly_expands_orb_and_halo(self):
        contracted = _orb_outline_points(1.0, 0.0, 0.0)
        expanded = _orb_outline_points(1.0, 0.0, 1.0)

        assert _average_distance(contracted, expanded) == pytest.approx(1.6)
        assert _halo_radius(0.0, 1.0) - _halo_radius(0.0, 0.0) == pytest.approx(4.0)

    @pytest.mark.parametrize("elapsed", (0.0, 1.0))
    def test_wave_and_corona_geometry_is_distinct_and_audio_reactive(self, elapsed):
        idle_waves = tuple(_wave_points(index, elapsed, 0.0) for index in range(3))
        speech_waves = tuple(_wave_points(index, elapsed, 0.4375) for index in range(3))

        assert all(len(points) == _WAVE_POINT_COUNT == 24 for points in idle_waves)
        assert all(all(math.isfinite(coordinate) for point in points for coordinate in point) for points in (*idle_waves, *speech_waves))
        for index, (idle, speech) in enumerate(zip(idle_waves, speech_waves, strict=True)):
            baseline = _WAVE_BASE_Y + _WAVE_OFFSETS[index]
            assert [point[0] for point in speech] == sorted(point[0] for point in speech)
            assert speech[0][1] == pytest.approx(baseline)
            assert speech[-1][1] == pytest.approx(baseline)
            idle_span = max(point[1] for point in idle) - min(point[1] for point in idle)
            speech_span = max(point[1] for point in speech) - min(point[1] for point in speech)
            assert idle_span >= 2.0
            assert (
                _average_distance(
                    idle,
                    _wave_points(index, elapsed + 0.25, 0.0),
                )
                >= 0.12
            )
            assert speech_span - idle_span >= 8.0
            assert (
                _average_distance(
                    speech,
                    _wave_points(index, elapsed + 0.05, 0.4375),
                )
                >= 0.1
            )

        assert all(_average_distance(speech_waves[first], speech_waves[second]) >= 3.0 for first, second in ((0, 1), (0, 2), (1, 2)))

        for index in range(3):
            idle = _corona_points(index, elapsed, 0.0)
            noise = _corona_points(index, elapsed, 0.22)
            speech = _corona_points(index, elapsed, 0.4375)
            idle_radii = tuple(math.hypot(x - _ORB_CENTER_X, y - _ORB_CENTER_Y) for x, y in idle)
            radii = tuple(math.hypot(x - _ORB_CENTER_X, y - _ORB_CENTER_Y) for x, y in speech)

            assert len(speech) == _CORONA_POINT_COUNT == 20
            assert _average_distance(noise, idle) == pytest.approx(0.0)
            assert max(idle_radii) - min(idle_radii) >= 1.25
            assert (
                _average_distance(
                    idle,
                    _corona_points(index, elapsed + 0.25, 0.0),
                )
                >= 0.12
            )
            assert _average_distance(idle, speech) >= 0.9
            assert max(radii) - min(radii) >= 4.0
            assert (
                _average_distance(
                    speech,
                    _corona_points(index, elapsed + 0.05, 0.4375),
                )
                >= 0.025
            )

    def test_all_curves_and_strokes_stay_inside_rectangular_host(self):
        activities = (0.0, 0.22, 0.25, 0.375, 0.55, 1.0)
        maximum_entry_scale = max(_entry_scale(step * _ENTRY_DURATION / 1000.0) for step in range(1001))
        panel_center = (_PANEL_CENTER_X, _PANEL_CENTER_Y)
        orb_center = (_ORB_CENTER_X, _ORB_CENTER_Y)
        bounds = {name: [math.inf, math.inf, -math.inf, -math.inf] for name in ("orb", "corona", "waves", "halo")}
        minimum_wave_gap = math.inf
        maximum_wave_top = -math.inf

        def _record(name, points, padding):
            component = bounds[name]
            component[0] = min(component[0], min(x for x, _ in points) - padding)
            component[1] = min(component[1], min(y for _, y in points) - padding)
            component[2] = max(component[2], max(x for x, _ in points) + padding)
            component[3] = max(component[3], max(y for _, y in points) + padding)

        for activity in activities:
            voice_scale = _orb_scale(activity)
            rim_padding = max(_rim_widths(activity)) * voice_scale * maximum_entry_scale / 2.0
            for breathe in (0.0, 0.5, 1.0):
                for step in range(401):
                    elapsed = step * 0.25
                    orb_points = tuple(
                        _scale_about(
                            _scale_about(point, voice_scale, orb_center),
                            maximum_entry_scale,
                            panel_center,
                        )
                        for point in _curve_extent_points(
                            _orb_outline_points(elapsed, activity, breathe),
                            closed=True,
                        )
                    )
                    _record("orb", orb_points, rim_padding)

                    wave_top = -math.inf
                    for index in range(3):
                        corona_points = tuple(
                            _scale_about(
                                point,
                                maximum_entry_scale,
                                panel_center,
                            )
                            for point in _curve_extent_points(
                                _corona_points(index, elapsed, activity),
                                closed=True,
                            )
                        )
                        corona_padding = max(_CORONA_STROKE_WIDTHS[index]) * maximum_entry_scale / 2.0
                        _record("corona", corona_points, corona_padding)

                        wave_points = tuple(
                            _scale_about(
                                point,
                                maximum_entry_scale,
                                panel_center,
                            )
                            for point in _curve_extent_points(
                                _wave_points(index, elapsed, activity),
                                closed=False,
                            )
                        )
                        wave_padding = max(_WAVE_STROKE_WIDTHS[index]) * maximum_entry_scale / 2.0
                        _record("waves", wave_points, wave_padding)
                        wave_top = max(
                            wave_top,
                            max(y for _, y in wave_points) + wave_padding,
                        )
                        maximum_wave_top = max(
                            maximum_wave_top,
                            max(y for _, y in wave_points) + wave_padding,
                        )
                    orb_bottom = min(y for _, y in orb_points) - rim_padding
                    minimum_wave_gap = min(
                        minimum_wave_gap,
                        orb_bottom - wave_top,
                    )

            transformed_center = _scale_about(
                orb_center,
                maximum_entry_scale,
                panel_center,
            )
            halo_extent = _halo_radius(activity, 1.0) * maximum_entry_scale
            _record(
                "halo",
                (
                    (
                        transformed_center[0] - halo_extent,
                        transformed_center[1] - halo_extent,
                    ),
                    (
                        transformed_center[0] + halo_extent,
                        transformed_center[1] + halo_extent,
                    ),
                ),
                0.0,
            )

        for name, (left, bottom, right, top) in bounds.items():
            assert left >= 1.0, f"{name} clips the left edge: {left}"
            assert bottom >= 1.0, f"{name} clips the bottom edge: {bottom}"
            assert right <= _PANEL_WIDTH - 1.0, f"{name} clips the right edge: {right}"
            assert top <= _PANEL_HEIGHT - 1.0, f"{name} clips the top edge: {top}"
        assert minimum_wave_gap >= 1.0
        assert maximum_wave_top <= _WAVE_CLIP_HEIGHT

    def test_audio_helpers_clamp_and_remain_monotonic(self):
        assert _halo_radius(-1.0, -1.0) == 72.0
        assert _halo_radius(4.0, 4.0) == 87.0
        assert _gravity_pull(-1.0) == 0.0
        assert _gravity_pull(0.0) == 0.0
        assert _gravity_pull(0.22) == 0.0
        assert _gravity_pull(0.385) == pytest.approx(0.5)
        assert _gravity_pull(0.5) > 0.93
        assert _gravity_pull(0.55) == 1.0
        assert _gravity_pull(2.0) == 1.0
        assert _gravity_pull(0.02) < _gravity_pull(0.45) < 1.0

    def test_entry_and_exit_scales_are_bounded(self):
        assert abs(_entry_scale(0.0) - 0.78) < 0.001
        assert 1.0 < _entry_scale(0.14) < 1.03
        assert _entry_scale(1.0) == 1.0
        assert _exit_scale(0.0) == 1.0
        assert abs(_exit_scale(1.0) - 0.76) < 0.001

    def test_begin_exit_is_idempotent(self):
        view = RecordingIndicatorView()

        with patch("wenzi.audio.recording_indicator.time.monotonic") as now:
            now.side_effect = [10.0, 20.0]
            view.begin_exit()
            view.begin_exit()

        assert view._exit_started_at == 10.0
        now.assert_called_once()

    def test_prepare_for_display_caches_gradients_colors_and_mutable_path(self):
        view = RecordingIndicatorView()

        view.prepare_for_display()
        initial_resources = _cached_resources(view)
        view.prepare_for_display()

        assert len(view._orb_accent_gradients) == 2
        assert view._orb_depth_gradient is not None
        assert len(view._rim_colors) == 3
        assert len(view._corona_paths) == len(view._corona_colors) == 3
        assert len(view._wave_paths) == len(view._wave_colors) == 3
        assert view._wave_edge_gradient is not None
        assert tuple(tuple(round(color.alphaComponent(), 3) for color in layers) for layers in view._corona_colors) == (
            (0.08, 0.26, 0.74),
            (0.07, 0.23, 0.66),
            (0.06, 0.20, 0.60),
        )
        assert tuple(
            tuple(round(component, 2) for component in (layers[-1].redComponent(), layers[-1].greenComponent(), layers[-1].blueComponent()))
            for layers in view._corona_colors
        ) == (
            (1.0, 0.91, 0.64),
            (1.0, 0.53, 0.62),
            (1.0, 0.48, 0.82),
        )
        assert all(resource is not None for resource in initial_resources)
        for current, initial in zip(
            _cached_resources(view),
            initial_resources,
            strict=True,
        ):
            assert current is initial

    def test_path_prewarm_failure_retries_without_partial_cache(self):
        view = RecordingIndicatorView()
        appkit = MagicMock()
        native_path = MagicMock()
        appkit.NSBezierPath.alloc.return_value.init.return_value = native_path
        native_path.setLineJoinStyle_.side_effect = RuntimeError("path setup failed")

        with patch.dict(sys.modules, {"AppKit": appkit}):
            with pytest.raises(RuntimeError, match="path setup failed"):
                view.prepare_for_display()

            assert view._halo_gradient is None
            assert view._orb_body_gradient is None
            assert view._orb_accent_gradients is None
            assert view._orb_depth_gradient is None
            assert view._orb_bloom_gradient is None
            assert view._orb_path is None
            assert view._rim_colors is None
            assert view._corona_paths is None
            assert view._corona_colors is None
            assert view._wave_paths is None
            assert view._wave_colors is None
            assert view._wave_edge_gradient is None

            native_path.setLineJoinStyle_.side_effect = None
            view.prepare_for_display()

        # The failed orb allocation is followed by a complete seven-path retry.
        assert appkit.NSBezierPath.alloc.call_count == 8
        assert view._orb_path is native_path
        assert view._orb_body_gradient is not None
        assert view._orb_depth_gradient is not None
        assert len(view._orb_accent_gradients) == 2
        assert len(view._rim_colors) == 3
        assert len(view._corona_paths) == 3
        assert len(view._wave_paths) == 3
        assert view._wave_edge_gradient is not None

    def test_idle_first_frame_uses_full_color_orb(self):
        view = RecordingIndicatorView()
        view._start_time = 10.0
        appkit = MagicMock()

        with (
            patch.dict(sys.modules, {"AppKit": appkit}),
            patch(
                "wenzi.audio.recording_indicator.time.monotonic",
                return_value=10.0,
            ),
            patch.object(view, "_draw_active_orb") as draw_active,
        ):
            view.draw(None)

        draw_active.assert_called_once_with(0.0, 0.5)
        applied_scale = appkit.NSAffineTransform.transform.return_value.scaleBy_.call_args.args[0]
        assert abs(applied_scale - 0.78) < 0.001
        appkit.NSAffineTransform.transform.return_value.translateXBy_yBy_.assert_has_calls(
            [
                call(_PANEL_CENTER_X, _PANEL_CENTER_Y),
                call(-_PANEL_CENTER_X, -_PANEL_CENTER_Y),
            ]
        )
        appkit.NSGraphicsContext.restoreGraphicsState.assert_called_once()

    def test_layer_order_keeps_waves_and_corona_outside_voice_transform(self):
        view = _install_mock_resources(RecordingIndicatorView())
        view._recording_active = True
        view.set_level(1.0)

        state_depth = [0]
        events = []
        rim_count = [0]

        def _save_state():
            state_depth[0] += 1

        def _restore_state():
            state_depth[0] -= 1

        view._halo_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("halo")
        view._orb_path.addClip.side_effect = lambda: events.append("clip")
        view._orb_body_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("body")
        for index, gradient in enumerate(view._orb_accent_gradients):
            gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args, layer=index: events.append(
                f"accent-{layer}"
            )
        view._orb_depth_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("depth")
        view._orb_bloom_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = lambda *args: events.append("bloom")
        view._wave_edge_gradient.drawFromPoint_toPoint_options_.side_effect = lambda *args: events.append("wave-mask")
        for index, path in enumerate(view._wave_paths):
            path.stroke.side_effect = lambda layer=index: events.append(f"wave-{layer}")
        corona_counts = [0, 0, 0]

        def _record_corona(path_index):
            def _stroke():
                events.append(f"corona-{corona_counts[path_index]}-{path_index}")
                corona_counts[path_index] += 1

            return _stroke

        for index, path in enumerate(view._corona_paths):
            path.stroke.side_effect = _record_corona(index)

        def _stroke():
            events.append(f"rim-{rim_count[0]}")
            rim_count[0] += 1

        view._orb_path.stroke.side_effect = _stroke

        appkit = MagicMock()
        appkit.NSGraphicsContext.saveGraphicsState.side_effect = _save_state
        appkit.NSGraphicsContext.restoreGraphicsState.side_effect = _restore_state
        appkit.NSAffineTransform.transform.return_value.concat.side_effect = lambda: events.append("voice-transform")
        foundation = MagicMock()
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)
        foundation.NSMakeRect.side_effect = lambda x, y, width, height: SimpleNamespace(
            x=x,
            y=y,
            width=width,
            height=height,
        )

        with patch.dict(
            sys.modules,
            {"AppKit": appkit, "Foundation": foundation},
        ):
            view._draw_active_orb(elapsed=1.0, breathe=0.5)

        assert state_depth[0] == 0
        wave_events = [index for index, event in enumerate(events) if event.startswith("wave-") and event != "wave-mask"]
        corona_events = [index for index, event in enumerate(events) if event.startswith("corona-")]
        rim_events = [index for index, event in enumerate(events) if event.startswith("rim-")]
        assert len(wave_events) == 9
        assert len(corona_events) == 9
        assert len(rim_events) == 3
        assert [events[index] for index in corona_events] == [
            "corona-0-2",
            "corona-0-1",
            "corona-0-0",
            "corona-1-2",
            "corona-1-1",
            "corona-1-0",
            "corona-2-2",
            "corona-2-1",
            "corona-2-0",
        ]
        assert max(wave_events) < events.index("wave-mask") < events.index("halo")
        appkit.NSRectClip.assert_called_once_with(SimpleNamespace(x=0.0, y=0.0, width=_PANEL_WIDTH, height=_WAVE_CLIP_HEIGHT))
        appkit.NSGraphicsContext.currentContext.return_value.setCompositingOperation_.assert_called_once_with(
            appkit.NSCompositingOperationDestinationIn
        )
        view._wave_edge_gradient.drawFromPoint_toPoint_options_.assert_called_once_with(
            SimpleNamespace(x=15.0, y=_WAVE_BASE_Y),
            SimpleNamespace(x=205.0, y=_WAVE_BASE_Y),
            appkit.NSGradientDrawsBeforeStartingLocation | appkit.NSGradientDrawsAfterEndingLocation,
        )
        assert events.index("halo") < min(corona_events)
        assert max(corona_events) < events.index("voice-transform")
        assert events.index("voice-transform") < events.index("body")
        assert events.index("body") < events.index("depth")
        assert events.index("depth") < events.index("bloom")
        assert events.index("bloom") < min(rim_events)
        view._orb_path.addClip.assert_called_once_with()
        view._orb_path.setLineWidth_.assert_has_calls([call(width) for width in _rim_widths(1.0)])
        assert view._orb_path.stroke.call_count == 3
        view._orb_path.fill.assert_not_called()
        appkit.NSRectFill.assert_not_called()
        assert not appkit.CIFilter.mock_calls
        for color in view._rim_colors:
            color.setStroke.assert_called_once_with()

        depth_center = view._orb_depth_gradient.drawFromCenter_radius_toCenter_radius_options_.call_args.args[0]
        bloom_center = view._orb_bloom_gradient.drawFromCenter_radius_toCenter_radius_options_.call_args_list[0].args[0]
        assert depth_center.x > _ORB_CENTER_X
        assert depth_center.y < _ORB_CENTER_Y
        assert bloom_center.x < _ORB_CENTER_X
        assert bloom_center.y > _ORB_CENTER_Y
        assert (
            math.dist(
                (depth_center.x, depth_center.y),
                (bloom_center.x, bloom_center.y),
            )
            >= 45.0
        )
        voice_transform = appkit.NSAffineTransform.transform.return_value
        voice_transform.translateXBy_yBy_.assert_has_calls(
            [
                call(_ORB_CENTER_X, _ORB_CENTER_Y),
                call(-_ORB_CENTER_X, -_ORB_CENTER_Y),
            ]
        )
        voice_transform.scaleBy_.assert_called_once_with(_orb_scale(1.0))

    def test_rebuild_path_uses_closed_catmull_rom_sequence(self):
        view = RecordingIndicatorView()
        view._orb_path = MagicMock()
        points = _orb_outline_points(1.0, 0.5, 0.5)
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

        view._rebuild_orb_path(points)

        assert view._orb_path.mock_calls == expected_calls
        assert view._orb_path.curveToPoint_controlPoint1_controlPoint2_.call_count == _OUTLINE_POINT_COUNT

    def test_draw_loop_reuses_path_and_allocates_no_native_resources(self):
        view = _install_mock_resources(RecordingIndicatorView())
        initial_resources = _cached_resources(view)

        appkit = MagicMock()
        foundation = MagicMock()
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)
        with patch.dict(
            sys.modules,
            {"AppKit": appkit, "Foundation": foundation},
        ):
            view._draw_active_orb(elapsed=1.0, breathe=0.5)
            view._draw_active_orb(elapsed=1.05, breathe=0.55)

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
        assert view._orb_path.removeAllPoints.call_count == 2
        assert view._orb_path.moveToPoint_.call_count == 2
        assert view._orb_path.curveToPoint_controlPoint1_controlPoint2_.call_count == _OUTLINE_POINT_COUNT * 2
        assert view._orb_path.closePath.call_count == 2
        assert view._orb_path.addClip.call_count == 2
        assert view._orb_path.stroke.call_count == 6

        for path in view._corona_paths:
            assert path.removeAllPoints.call_count == 2
            assert path.moveToPoint_.call_count == 2
            assert path.curveToPoint_controlPoint1_controlPoint2_.call_count == _CORONA_POINT_COUNT * 2
            assert path.closePath.call_count == 2
            assert path.stroke.call_count == 6
        for path in view._wave_paths:
            assert path.removeAllPoints.call_count == 2
            assert path.moveToPoint_.call_count == 2
            assert path.curveToPoint_controlPoint1_controlPoint2_.call_count == (_WAVE_POINT_COUNT - 1) * 2
            path.closePath.assert_not_called()
            assert path.stroke.call_count == 6

        assert view._wave_edge_gradient.drawFromPoint_toPoint_options_.call_count == 2
        assert appkit.NSRectClip.call_count == 2

        path_segments_per_frame = (
            _OUTLINE_POINT_COUNT + len(view._corona_paths) * _CORONA_POINT_COUNT + len(view._wave_paths) * (_WAVE_POINT_COUNT - 1)
        )
        strokes_per_frame = 3 + len(view._corona_paths) * 3 + len(view._wave_paths) * 3
        assert path_segments_per_frame == 153
        assert path_segments_per_frame <= 160
        assert strokes_per_frame == 21

    def test_wave_mask_failure_restores_graphics_state(self):
        view = _install_mock_resources(RecordingIndicatorView())
        view._wave_edge_gradient.drawFromPoint_toPoint_options_.side_effect = RuntimeError("mask failed")
        state_depth = [0]

        appkit = MagicMock()
        appkit.NSGraphicsContext.saveGraphicsState.side_effect = lambda: state_depth.__setitem__(0, state_depth[0] + 1)
        appkit.NSGraphicsContext.restoreGraphicsState.side_effect = lambda: state_depth.__setitem__(0, state_depth[0] - 1)
        foundation = MagicMock()
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)

        with (
            patch.dict(
                sys.modules,
                {"AppKit": appkit, "Foundation": foundation},
            ),
            pytest.raises(RuntimeError, match="mask failed"),
        ):
            view._draw_bottom_waves(elapsed=1.0, activity=0.5)

        assert state_depth[0] == 0
        assert sum(path.stroke.call_count for path in view._wave_paths) == 9

    def test_clipped_body_failure_restores_body_and_outer_graphics_state(self):
        view = _install_mock_resources(RecordingIndicatorView())
        view._orb_body_gradient.drawFromCenter_radius_toCenter_radius_options_.side_effect = RuntimeError("body failed")
        view._start_time = 10.0

        state_depth = [0]
        state_events = []

        def _save_state():
            state_depth[0] += 1
            state_events.append(("save", state_depth[0]))

        def _restore_state():
            state_events.append(("restore", state_depth[0]))
            state_depth[0] -= 1

        appkit = MagicMock()
        appkit.NSGraphicsContext.saveGraphicsState.side_effect = _save_state
        appkit.NSGraphicsContext.restoreGraphicsState.side_effect = _restore_state
        foundation = MagicMock()
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)

        with (
            patch.dict(
                sys.modules,
                {"AppKit": appkit, "Foundation": foundation},
            ),
            patch(
                "wenzi.audio.recording_indicator.time.monotonic",
                return_value=11.0,
            ),
            pytest.raises(RuntimeError, match="body failed"),
        ):
            view.draw(None)

        assert state_depth[0] == 0
        assert sum(event[0] == "save" for event in state_events) == sum(event[0] == "restore" for event in state_events)
        view._orb_path.addClip.assert_called_once_with()
        view._orb_path.stroke.assert_not_called()

    def test_rim_failure_restores_rim_and_outer_graphics_state(self):
        view = _install_mock_resources(RecordingIndicatorView())
        view._orb_path.stroke.side_effect = [None, RuntimeError("rim failed")]
        view._start_time = 10.0

        state_depth = [0]
        state_events = []

        def _save_state():
            state_depth[0] += 1
            state_events.append(("save", state_depth[0]))

        def _restore_state():
            state_events.append(("restore", state_depth[0]))
            state_depth[0] -= 1

        appkit = MagicMock()
        appkit.NSGraphicsContext.saveGraphicsState.side_effect = _save_state
        appkit.NSGraphicsContext.restoreGraphicsState.side_effect = _restore_state
        foundation = MagicMock()
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)

        with (
            patch.dict(
                sys.modules,
                {"AppKit": appkit, "Foundation": foundation},
            ),
            patch(
                "wenzi.audio.recording_indicator.time.monotonic",
                return_value=11.0,
            ),
            pytest.raises(RuntimeError, match="rim failed"),
        ):
            view.draw(None)

        assert state_depth[0] == 0
        assert sum(event[0] == "save" for event in state_events) == sum(event[0] == "restore" for event in state_events)
        view._orb_path.addClip.assert_called_once_with()
        assert view._orb_path.stroke.call_count == 2

    def test_failed_path_rebuild_clears_again_and_recovers_next_frame(self):
        view = _install_mock_resources(RecordingIndicatorView())
        view._orb_path.curveToPoint_controlPoint1_controlPoint2_.side_effect = RuntimeError("curve failed")
        view._start_time = 10.0

        state_depth = [0]

        def _save_state():
            state_depth[0] += 1

        def _restore_state():
            state_depth[0] -= 1

        appkit = MagicMock()
        appkit.NSGraphicsContext.saveGraphicsState.side_effect = _save_state
        appkit.NSGraphicsContext.restoreGraphicsState.side_effect = _restore_state
        foundation = MagicMock()
        foundation.NSMakePoint.side_effect = lambda x, y: SimpleNamespace(x=x, y=y)

        with (
            patch.dict(
                sys.modules,
                {"AppKit": appkit, "Foundation": foundation},
            ),
            patch(
                "wenzi.audio.recording_indicator.time.monotonic",
                return_value=11.0,
            ),
        ):
            with pytest.raises(RuntimeError, match="curve failed"):
                view.draw(None)

            assert state_depth[0] == 0
            view._orb_path.curveToPoint_controlPoint1_controlPoint2_.side_effect = None
            view.draw(None)

        assert state_depth[0] == 0
        assert view._orb_path.removeAllPoints.call_count == 2
        assert view._orb_path.closePath.call_count == 1
        view._orb_path.addClip.assert_called_once_with()
        assert view._orb_path.stroke.call_count == 3


class TestRecordingIndicatorPanel:
    def test_initial_state(self):
        panel = RecordingIndicatorPanel()

        assert panel.enabled is True
        assert panel.show_device_name is False
        assert panel._panel is None
        assert panel._timer is None

    def test_enabled_and_show_device_name_toggles(self):
        panel = RecordingIndicatorPanel()

        panel.show_device_name = True
        assert panel.show_device_name is True
        panel.enabled = False
        assert panel.enabled is False

    def test_show_builds_only_a_transparent_centered_indicator_host(self):
        panel = RecordingIndicatorPanel()
        native_panel = MagicMock()
        timer = MagicMock()
        indicator = MagicMock()

        appkit = MagicMock()
        appkit.NSStatusWindowLevel = 100
        appkit.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = native_panel
        screen_frame = SimpleNamespace(
            origin=SimpleNamespace(x=100.0, y=50.0),
            size=SimpleNamespace(width=1000.0, height=700.0),
        )
        appkit.NSScreen.mainScreen.return_value.visibleFrame.return_value = screen_frame

        foundation = MagicMock()
        foundation.NSMakeRect.side_effect = lambda *args: args
        foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.return_value = timer

        events = []
        native_panel.setContentView_.side_effect = lambda view: events.append("content")
        native_panel.setAlphaValue_.side_effect = lambda value: events.append(f"alpha-{value}")
        indicator.setNeedsDisplay_.side_effect = lambda value: events.append("needs-display")
        indicator.displayIfNeededIgnoringOpacity.side_effect = lambda: events.append("first-frame")
        native_panel.orderFront_.side_effect = lambda sender: events.append("order-front")
        foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.side_effect = lambda *args: (
            events.append("timer") or timer
        )

        with (
            patch.dict(
                sys.modules,
                {"AppKit": appkit, "Foundation": foundation},
            ),
            patch.object(
                RecordingIndicatorView,
                "prepare_for_display",
                side_effect=lambda: events.append("prepare"),
            ) as prepare,
            patch.object(
                RecordingIndicatorView,
                "create_view",
                return_value=indicator,
            ) as create_view,
            patch.object(
                RecordingIndicatorPanel,
                "_animate_alpha",
            ) as animate_alpha,
        ):
            panel.show("MacBook Pro Microphone", "Proofread")

        prepare.assert_called_once()
        appkit.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.assert_called_once_with(
            (0, 0, _PANEL_WIDTH, _PANEL_HEIGHT), 0, 2, False
        )
        native_panel.setOpaque_.assert_called_once_with(False)
        native_panel.setBackgroundColor_.assert_called_once_with(appkit.NSColor.clearColor.return_value)
        native_panel.setIgnoresMouseEvents_.assert_called_once_with(True)
        native_panel.setHasShadow_.assert_called_once_with(False)
        native_panel.setFrameOrigin_.assert_called_once_with((490.0, 296.0))
        create_view.assert_called_once_with(_PANEL_WIDTH, _PANEL_HEIGHT)
        native_panel.setContentView_.assert_called_once_with(indicator)
        indicator.setNeedsDisplay_.assert_called_once_with(True)
        indicator.displayIfNeededIgnoringOpacity.assert_called_once_with()
        assert native_panel.setAlphaValue_.call_args_list == [
            ((0.0,), {}),
            ((0.62,), {}),
        ]
        assert events == [
            "prepare",
            "content",
            "alpha-0.0",
            "order-front",
            "needs-display",
            "first-frame",
            "alpha-0.62",
            "timer",
        ]
        animate_alpha.assert_called_once_with(native_panel, 1.0, duration=0.18)
        foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once_with(
            0.05, indicator, b"refresh:", None, True
        )
        timer.setTolerance_.assert_called_once_with(0.01)
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

    def test_first_frame_failure_closes_invisible_panel_without_timer(self):
        panel = RecordingIndicatorPanel()
        native_panel = MagicMock()
        indicator = MagicMock()
        indicator.displayIfNeededIgnoringOpacity.side_effect = RuntimeError("draw failed")

        appkit = MagicMock()
        appkit.NSStatusWindowLevel = 100
        appkit.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = native_panel
        appkit.NSScreen.mainScreen.return_value = None
        foundation = MagicMock()
        foundation.NSMakeRect.side_effect = lambda *args: args

        def _create_view(indicator_view, width, height):
            indicator_view._view = indicator
            indicator._indicator = indicator_view
            return indicator

        with (
            patch.dict(
                sys.modules,
                {"AppKit": appkit, "Foundation": foundation},
            ),
            patch.object(
                RecordingIndicatorView,
                "prepare_for_display",
            ),
            patch.object(
                RecordingIndicatorView,
                "create_view",
                autospec=True,
                side_effect=_create_view,
            ),
        ):
            panel.show("Microphone", "Proofread")

        native_panel.setAlphaValue_.assert_called_once_with(0.0)
        native_panel.orderFront_.assert_called_once_with(None)
        native_panel.orderOut_.assert_called_once_with(None)
        foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_not_called()
        assert indicator._indicator is None
        assert panel._panel is None
        assert panel._timer is None
        assert panel._indicator_view is None

    def test_update_level_uses_asymmetric_ema_and_clamps(self):
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()

        panel.update_level(4.0)
        assert panel._smoothed_level == pytest.approx(0.7)
        panel.update_level(1.0)
        assert panel._smoothed_level == pytest.approx(0.91)
        panel.update_level(-2.0)
        assert panel._smoothed_level == pytest.approx(0.6825)
        assert panel._indicator_view._level == pytest.approx(0.6825)

    def test_first_real_speech_sample_drives_waves_and_corona(self):
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()

        # Recorder RMS 500 maps to 0.625; one attack tick must already be visible.
        panel.update_level(0.625)
        activity = panel._indicator_view._level

        assert activity == pytest.approx(0.4375)
        assert _gravity_pull(activity) > 0.7
        for index in range(3):
            idle_wave = _wave_points(index, 1.0, 0.0)
            speech_wave = _wave_points(index, 1.0, activity)
            idle_span = max(y for _, y in idle_wave) - min(y for _, y in idle_wave)
            speech_span = max(y for _, y in speech_wave) - min(y for _, y in speech_wave)
            assert speech_span - idle_span >= 8.0
            assert (
                _average_distance(
                    _corona_points(index, 1.0, 0.0),
                    _corona_points(index, 1.0, activity),
                )
                >= 0.9
            )

        noise_panel = RecordingIndicatorPanel()
        noise_panel._indicator_view = RecordingIndicatorView()
        noise_panel.update_level(0.25)
        assert noise_panel._indicator_view._level == pytest.approx(0.175)
        assert _gravity_pull(noise_panel._indicator_view._level) == 0.0

    def test_repeated_level_updates_converge_and_return_to_quiet(self):
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()

        for _ in range(10):
            panel.update_level(1.0)
        assert panel._smoothed_level > 0.99
        assert _gravity_pull(panel._indicator_view._level) > 0.99

        for _ in range(20):
            panel.update_level(0.0)
        assert panel._smoothed_level < 0.01
        assert _gravity_pull(panel._indicator_view._level) == 0.0

    def test_set_recording_active_updates_view(self):
        panel = RecordingIndicatorPanel()
        panel._indicator_view = RecordingIndicatorView()
        panel._indicator_view._view = MagicMock()

        panel.set_recording_active()

        assert panel._indicator_view._recording_active is True
        panel._indicator_view._view.setNeedsDisplay_.assert_called_once_with(True)

    def test_legacy_mode_and_device_updates_add_no_visual_content(self):
        panel = RecordingIndicatorPanel()
        panel._panel = MagicMock()
        panel._indicator_view = RecordingIndicatorView()

        panel.update_mode("Translate EN")
        panel.update_device_name("MacBook Pro Microphone")

        assert panel._mode_name == "Translate EN"
        assert panel._device_name == "MacBook Pro Microphone"
        assert not hasattr(panel._indicator_view, "_subtitle")
        assert not hasattr(panel._indicator_view, "_status_text")

    def test_clear_mode_keeps_orb_state(self):
        panel = RecordingIndicatorPanel()
        view = RecordingIndicatorView()
        panel._indicator_view = view
        panel._mode_name = "Proofread"

        panel.clear_mode()

        assert panel._mode_name is None
        assert panel._indicator_view is view

    def test_hide_invalidates_timer_and_clears_state(self):
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
        panel._smoothed_level = 0.5
        panel._mode_name = "Proofread"
        panel._device_name = "Mic"

        panel.hide()

        timer.invalidate.assert_called_once()
        native_panel.orderOut_.assert_called_once_with(None)
        assert panel._timer is None
        assert panel._panel is None
        assert panel._indicator_view is None
        assert panel._smoothed_level == 0.0
        assert panel._mode_name is None
        assert panel._device_name is None
        assert native_view._indicator is None

    def test_hide_clears_ownership_when_native_cleanup_fails(self):
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

        panel.hide()

        timer.invalidate.assert_called_once()
        native_panel.orderOut_.assert_called_once_with(None)
        assert panel._timer is None
        assert panel._panel is None
        assert panel._indicator_view is None
        assert native_view._indicator is None

    def test_current_frame(self):
        panel = RecordingIndicatorPanel()
        assert panel.current_frame is None

        native_panel = MagicMock()
        panel._panel = native_panel
        assert panel.current_frame is native_panel.frame.return_value

    def test_animate_out_calls_completion_when_hidden(self):
        panel = RecordingIndicatorPanel()
        completion = MagicMock()

        panel.animate_out(completion)

        completion.assert_called_once()

    def test_animate_out_keeps_timer_until_scale_animation_completes(self):
        panel = RecordingIndicatorPanel()
        panel._panel = MagicMock()
        panel._indicator_view = RecordingIndicatorView()
        panel._indicator_view._view = MagicMock()
        panel._indicator_view._view._indicator = panel._indicator_view
        native_panel = panel._panel
        native_view = panel._indicator_view._view
        timer = MagicMock()
        panel._timer = timer

        captured = {}
        context = MagicMock()
        context.setCompletionHandler_.side_effect = lambda callback: captured.setdefault("callback", callback)
        animation_context = MagicMock()
        animation_context.currentContext.return_value = context

        with patch.dict(
            sys.modules,
            {"AppKit": MagicMock(NSAnimationContext=animation_context)},
        ):
            panel.animate_out()

        assert panel._indicator_view._exit_started_at is not None
        panel._indicator_view._view.setNeedsDisplay_.assert_called_once_with(True)
        timer.invalidate.assert_not_called()
        assert panel._timer is timer

        captured["callback"]()

        timer.invalidate.assert_called_once()
        native_panel.orderOut_.assert_called_once_with(None)
        assert native_view._indicator is None
        assert panel._timer is None
        assert panel._panel is None

    def test_stale_animate_completion_cannot_clear_new_orb(self):
        panel = RecordingIndicatorPanel()
        old_panel = MagicMock()
        old_view = RecordingIndicatorView()
        panel._panel = old_panel
        panel._indicator_view = old_view
        completion = MagicMock()

        captured = {}
        context = MagicMock()
        context.setCompletionHandler_.side_effect = lambda callback: captured.setdefault("callback", callback)
        animation_context = MagicMock()
        animation_context.currentContext.return_value = context

        with patch.dict(
            sys.modules,
            {"AppKit": MagicMock(NSAnimationContext=animation_context)},
        ):
            panel.animate_out(completion)

        new_panel = MagicMock()
        new_view = RecordingIndicatorView()
        new_timer = MagicMock()
        panel._panel = new_panel
        panel._indicator_view = new_view
        panel._timer = new_timer
        captured["callback"]()

        assert panel._panel is new_panel
        assert panel._indicator_view is new_view
        assert panel._timer is new_timer
        new_timer.invalidate.assert_not_called()
        old_panel.orderOut_.assert_not_called()
        completion.assert_not_called()

    def test_alpha_animation_always_ends_group(self):
        native_panel = MagicMock()
        native_panel.animator().setAlphaValue_.side_effect = RuntimeError("animation failed")
        animation_context = MagicMock()

        with patch.dict(
            sys.modules,
            {"AppKit": MagicMock(NSAnimationContext=animation_context)},
        ):
            try:
                RecordingIndicatorPanel._animate_alpha(native_panel, 1.0)
            except RuntimeError:
                pass

        animation_context.endGrouping.assert_called_once()
