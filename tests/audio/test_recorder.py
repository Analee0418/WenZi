"""Tests for the recorder module (AVAudioEngine backend)."""

import struct
import threading
import time
from unittest.mock import MagicMock

import pytest

from wenzi.audio.recorder import (
    Recorder,
    _InputRoute,
    _resample_linear,
    _rms_int16,
    _select_input_route,
    _TapSession,
)


def _int16_bytes(value: int, count: int = 320) -> bytes:
    """Create raw int16 PCM bytes filled with a constant value."""
    return struct.pack(f"<{count}h", *([value] * count))


def _silence_bytes(count: int = 320) -> bytes:
    return b"\x00" * (count * 2)


def _new_mock_engine():
    engine = MagicMock()
    input_node = MagicMock()
    hw_format = MagicMock()
    hw_format.sampleRate.return_value = 48000.0
    input_node.outputFormatForBus_.return_value = hw_format
    engine.inputNode.return_value = input_node
    engine.startAndReturnError_.return_value = (True, None)
    return engine


def _mock_engine(monkeypatch):
    """Patch AVAudioEngine and friends so start() succeeds without hardware."""
    mock_engine = _new_mock_engine()

    monkeypatch.setattr(
        "wenzi.audio.recorder.AVAudioEngine",
        MagicMock(alloc=MagicMock(return_value=MagicMock(init=MagicMock(return_value=mock_engine)))),
    )
    monkeypatch.setattr(
        "wenzi.audio.recorder.NSNotificationCenter",
        MagicMock(defaultCenter=MagicMock(return_value=MagicMock(
            addObserverForName_object_queue_usingBlock_=MagicMock(return_value="observer"),
            removeObserver_=MagicMock(),
        ))),
    )
    monkeypatch.setattr(
        "wenzi.audio.recorder.default_input_device_name",
        lambda: "TestMic",
    )
    monkeypatch.setattr(
        "wenzi.audio.recorder.list_input_devices",
        lambda: [{"uid": "test-uid", "name": "TestMic"}],
    )
    monkeypatch.setattr(
        "wenzi.audio.recorder._resolve_device_id",
        lambda uid: None,
    )
    monkeypatch.setattr(
        "wenzi.audio.recorder._select_input_route",
        lambda uid: _InputRoute("test-uid", "TestMic", None, False),
    )
    return mock_engine


def _mock_engine_sequence(monkeypatch, engines, *, transport=b"blue"):
    """Install normal mocks while returning AVAudioEngine objects in order."""
    _mock_engine(monkeypatch)
    allocator = MagicMock()
    allocator.alloc.return_value.init.side_effect = engines
    monkeypatch.setattr("wenzi.audio.recorder.AVAudioEngine", allocator)
    route = _InputRoute(
        "test-uid",
        "TestMic",
        int.from_bytes(transport, "big"),
        False,
    )
    monkeypatch.setattr(
        "wenzi.audio.recorder._select_input_route",
        lambda uid: route,
    )


def _capture_config_callbacks(monkeypatch):
    callbacks = []
    center = MagicMock()

    def _add_observer(_name, _engine, _queue, callback):
        callbacks.append(callback)
        return object()

    center.addObserverForName_object_queue_usingBlock_.side_effect = (
        _add_observer
    )
    notification_center = MagicMock()
    notification_center.defaultCenter.return_value = center
    monkeypatch.setattr(
        "wenzi.audio.recorder.NSNotificationCenter",
        notification_center,
    )
    return callbacks


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def _capture_device(name: str, uid: str, transport: bytes):
    device = MagicMock()
    device.localizedName.return_value = name
    device.uniqueID.return_value = uid
    device.transportType.return_value = int.from_bytes(transport, "big")
    return device


class TestInputRouteSelection:
    def test_automatic_keeps_bluetooth_system_default(self, monkeypatch):
        bluetooth = _capture_device("AirPods Max", "airpods", b"blue")
        capture = MagicMock()
        capture.defaultDeviceWithMediaType_.return_value = bluetooth
        monkeypatch.setattr("wenzi.audio.recorder.AVCaptureDevice", capture)

        route = _select_input_route(None)

        assert route.uid == "airpods"
        assert route.name == "AirPods Max"
        assert route.bind is False
        capture.devicesWithMediaType_.assert_not_called()

    def test_automatic_keeps_non_bluetooth_system_default(self, monkeypatch):
        usb = _capture_device("USB Microphone", "usb", b"usb ")
        capture = MagicMock()
        capture.defaultDeviceWithMediaType_.return_value = usb
        monkeypatch.setattr("wenzi.audio.recorder.AVCaptureDevice", capture)

        route = _select_input_route(None)

        assert route.uid == "usb"
        assert route.name == "USB Microphone"
        assert route.bind is False
        capture.devicesWithMediaType_.assert_not_called()

    def test_explicit_uid_still_follows_system_default(self, monkeypatch):
        bluetooth = _capture_device("AirPods Max", "airpods", b"blue")
        built_in = _capture_device(
            "MacBook Pro Microphone", "builtin", b"bltn"
        )
        capture = MagicMock()
        capture.defaultDeviceWithMediaType_.return_value = built_in
        capture.devicesWithMediaType_.return_value = [bluetooth, built_in]
        monkeypatch.setattr("wenzi.audio.recorder.AVCaptureDevice", capture)

        route = _select_input_route("airpods")

        assert route.uid == "builtin"
        assert route.name == "MacBook Pro Microphone"
        assert route.bind is False
        capture.defaultDeviceWithMediaType_.assert_called_once()
        capture.devicesWithMediaType_.assert_not_called()


class TestRecorder:
    def test_init_defaults(self):
        r = Recorder()
        assert r.sample_rate == 16000
        assert r.is_recording is False

    def test_stop_without_start_returns_none(self):
        r = Recorder()
        assert r.stop() is None

    def test_start_stop_cycle(self, monkeypatch):
        _mock_engine(monkeypatch)

        r = Recorder(sample_rate=16000, block_ms=20)
        r.start()
        assert r.is_recording is True

        # Simulate audio frames with enough energy to pass silence check
        frame = _int16_bytes(500)
        r._queue.put(frame)
        r._queue.put(frame)

        wav_data = r.stop()
        assert r.is_recording is False
        assert wav_data is not None
        assert len(wav_data) > 0

    def test_automatic_device_does_not_rebind_input_node(self, monkeypatch):
        engine = _mock_engine(monkeypatch)

        recorder = Recorder(sample_rate=16000, block_ms=20)
        recorder.start()

        audio_unit = engine.inputNode.return_value.AUAudioUnit.return_value
        audio_unit.setDeviceID_error_.assert_not_called()
        recorder.stop()

    def test_silence_detection_discards_quiet_audio(self, monkeypatch):
        _mock_engine(monkeypatch)

        r = Recorder(sample_rate=16000, block_ms=20, silence_rms=20)
        r.start()

        # Simulate silent audio (all zeros -> RMS=0)
        r._queue.put(_silence_bytes())
        r._queue.put(_silence_bytes())

        wav_data = r.stop()
        assert wav_data is None

    def test_silence_detection_passes_loud_audio(self, monkeypatch):
        _mock_engine(monkeypatch)

        r = Recorder(sample_rate=16000, block_ms=20, silence_rms=20)
        r.start()

        r._queue.put(_int16_bytes(1000))

        wav_data = r.stop()
        assert wav_data is not None

    def test_double_start_is_noop(self):
        r = Recorder()
        r._recording = True
        r.start()  # Should not raise

    def test_current_level_initial_zero(self):
        r = Recorder()
        assert r.current_level == 0.0

    def test_current_level_after_rms_set(self):
        r = Recorder(sample_rate=16000, block_ms=20)
        # RMS lives on the session: 1500 → level = 1500/2400 = 0.625
        r._session = _TapSession()
        r._session.rms = 1500.0
        assert abs(r.current_level - 0.625) < 0.01

    def test_current_level_capped_at_one(self):
        r = Recorder(sample_rate=16000, block_ms=20)
        r._session = _TapSession()
        r._session.rms = 10000.0
        assert r.current_level == 1.0

    def test_max_session_bytes(self, monkeypatch):
        """tap_callback drops frames once the session byte cap is hit."""
        _mock_engine(monkeypatch)

        r = Recorder(sample_rate=16000, block_ms=20, max_session_bytes=2)
        r._query_device_name_enabled = False
        r.start()
        buf = _voice_buffer()

        # First frame (2 bytes) fits exactly; the second exceeds the cap
        r._tap_callback(buf, r._active_gen, 3.0, r._session)
        r._tap_callback(buf, r._active_gen, 3.0, r._session)

        assert r._session.queue.qsize() == 1
        assert r._session.total_bytes == 2
        r.stop()

    def test_start_queries_device_name(self, monkeypatch):
        _mock_engine(monkeypatch)

        r = Recorder(sample_rate=16000, block_ms=20)
        assert r._query_device_name_enabled is True

        name = r.start()
        assert name == "TestMic"
        r.stop()

    def test_start_skips_device_query_when_disabled(self, monkeypatch):
        _mock_engine(monkeypatch)

        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False

        r.start()
        assert r.is_recording is True
        assert r._last_device_name is None
        r.stop()

    def test_concurrent_start_raises(self):
        """A start() while another is in flight must fail loudly instead
        of pretending recording started."""
        r = Recorder(sample_rate=16000, block_ms=20)
        r._starting_since = time.monotonic()
        with pytest.raises(RuntimeError):
            r.start()
        assert not r._recording

    def test_stale_starting_flag_is_reset(self, monkeypatch):
        monkeypatch.setattr(Recorder, "_STARTING_STALE_SECS", 0.0)
        _mock_engine(monkeypatch)

        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r._starting_since = time.monotonic() - 1.0  # already stale
        r.start()
        assert r.is_recording is True
        r.stop()

    def test_explicit_device_config_is_ignored_without_binding(
        self,
        monkeypatch,
    ):
        engine = _mock_engine(monkeypatch)
        select = MagicMock(
            return_value=_InputRoute(
                "system-default",
                "Default Mic",
                None,
                True,
            )
        )
        resolve = MagicMock(return_value=42)
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            select,
        )
        monkeypatch.setattr(
            "wenzi.audio.recorder._resolve_device_id",
            resolve,
        )

        r = Recorder(sample_rate=16000, block_ms=20, device="old-uid")
        r._query_device_name_enabled = False
        assert r.device is None
        assert r.start() is None
        assert r.is_recording is True
        select.assert_called_once_with(None)
        resolve.assert_not_called()
        engine.inputNode.return_value.AUAudioUnit.return_value.setDeviceID_error_.assert_not_called()
        r.stop()

    def test_start_revalidates_sound_preflight_route(self, monkeypatch):
        engine = _mock_engine(monkeypatch)
        route = _InputRoute(
            "airpods",
            "AirPods Max",
            int.from_bytes(b"blue", "big"),
            False,
        )
        select = MagicMock(return_value=route)
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            select,
        )

        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        assert r.preflight_input_route() is route
        assert r.start() is None

        assert r.is_recording is True
        assert r._session.is_bluetooth is True
        assert select.call_args_list == [((None,), {}), ((None,), {})]
        engine.inputNode.return_value.AUAudioUnit.return_value.setDeviceID_error_.assert_not_called()
        r.stop()

    def test_chime_approved_then_bluetooth_default_aborts_start(
        self,
        monkeypatch,
    ):
        engine = _mock_engine(monkeypatch)
        built_in = _InputRoute(
            "built-in",
            "Built-in Microphone",
            int.from_bytes(b"bltn", "big"),
            False,
        )
        airpods = _InputRoute(
            "airpods",
            "AirPods Max",
            int.from_bytes(b"blue", "big"),
            False,
        )
        select = MagicMock(side_effect=[built_in, airpods])
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            select,
        )
        recorder = Recorder(sample_rate=16000, block_ms=20)
        recorder.preflight_input_route()
        recorder.mark_preflight_chime_allowed()

        assert recorder.start() is None

        assert not recorder.is_recording
        engine.inputNode.assert_not_called()

    def test_unknown_transport_enables_bluetooth_recovery(self, monkeypatch):
        _mock_engine(monkeypatch)
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            lambda _uid: _InputRoute("unknown", "Unknown Mic", None, False),
        )
        recorder = Recorder(sample_rate=16000, block_ms=20)

        recorder.start()

        assert recorder.is_recording
        assert recorder._session.is_bluetooth is True
        recorder.stop()


class TestMarkTainted:
    def test_taint_when_idle_is_noop(self):
        r = Recorder()
        r.mark_tainted()  # Should not raise
        assert r.is_recording is False

    def test_taint_during_start_tears_down_late_engine(self, monkeypatch):
        """A start() finishing after the caller gave up must not leave the
        microphone open: it tears the engine down instead of committing."""
        engine = _mock_engine(monkeypatch)
        started = threading.Event()
        release = threading.Event()

        def _blocking_start(_err):
            started.set()
            assert release.wait(timeout=5)
            return (True, None)

        engine.startAndReturnError_.side_effect = _blocking_start

        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False

        result: list = []
        t = threading.Thread(target=lambda: result.append(r.start()))
        t.start()
        try:
            assert started.wait(timeout=5)
            r.mark_tainted()  # caller timed out and gave up
        finally:
            release.set()  # start() now finishes late
            t.join(timeout=5)
        assert not t.is_alive()

        assert result == [None]
        assert r.is_recording is False
        engine.stop.assert_called_once()
        engine.inputNode.return_value.removeTapOnBus_.assert_called_with(0)

        # A fresh start() must succeed afterwards
        engine.startAndReturnError_.side_effect = None
        engine.startAndReturnError_.return_value = (True, None)
        r.start()
        assert r.is_recording is True
        r.stop()

    def test_taint_after_commit_stops_recording(self, monkeypatch):
        """If start() committed just before the taint, the engine is stopped."""
        engine = _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start()
        assert r.is_recording is True

        r.mark_tainted()  # stop happens on a helper thread
        for _ in range(200):
            if not r.is_recording:
                break
            time.sleep(0.01)
        assert r.is_recording is False
        engine.stop.assert_called_once()

    def test_taint_after_commit_can_leave_teardown_to_owner(
        self,
        monkeypatch,
    ):
        engine = _mock_engine(monkeypatch)
        recorder = Recorder(sample_rate=16000, block_ms=20)
        recorder.start()

        recorder.mark_tainted(stop_async=False)

        assert recorder.is_recording is True
        engine.stop.assert_not_called()
        recorder.stop()
        engine.stop.assert_called_once()

    def test_superseded_stale_start_does_not_commit(self, monkeypatch):
        """A stuck start() that finishes after a newer one committed must
        tear down its own engine and leave the new session untouched."""
        monkeypatch.setattr(Recorder, "_STARTING_STALE_SECS", 0.0)

        def _basic_engine():
            e = MagicMock()
            node = MagicMock()
            fmt = MagicMock()
            fmt.sampleRate.return_value = 48000.0
            node.outputFormatForBus_.return_value = fmt
            e.inputNode.return_value = node
            e.startAndReturnError_.return_value = (True, None)
            return e

        eng_old = _basic_engine()
        eng_new = _basic_engine()

        old_started = threading.Event()
        release = threading.Event()

        def _blocked(_err):
            old_started.set()
            assert release.wait(timeout=5)
            return (True, None)

        eng_old.startAndReturnError_.side_effect = _blocked

        av = MagicMock()
        av.alloc.return_value.init.side_effect = [eng_old, eng_new]
        monkeypatch.setattr("wenzi.audio.recorder.AVAudioEngine", av)
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            lambda uid: _InputRoute("test-uid", "TestMic", None, False),
        )
        monkeypatch.setattr(
            "wenzi.audio.recorder.NSNotificationCenter",
            MagicMock(defaultCenter=MagicMock(return_value=MagicMock(
                addObserverForName_object_queue_usingBlock_=MagicMock(
                    return_value="observer"
                ),
                removeObserver_=MagicMock(),
            ))),
        )

        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False

        results: list = []
        t = threading.Thread(target=lambda: results.append(r.start()))
        t.start()
        try:
            assert old_started.wait(timeout=5)
            # Stale threshold is 0 → this start() takes over immediately
            r.start()
            assert r.is_recording is True
            assert r._engine is eng_new
        finally:
            release.set()  # the stuck start() now finishes late
            t.join(timeout=5)
        assert not t.is_alive()

        assert results == [None]
        assert r.is_recording is True
        assert r._engine is eng_new
        eng_old.stop.assert_called_once()
        eng_new.stop.assert_not_called()

        r.stop()
        assert r.is_recording is False
        eng_new.stop.assert_called_once()


def _voice_buffer():
    """A fake AVAudioPCMBuffer with three float32 samples."""
    buf = MagicMock()
    buf.frameLength.return_value = 3
    ch = MagicMock()
    ch.as_buffer.return_value = struct.pack("<3f", 0.5, 0.5, 0.5)
    buf.floatChannelData.return_value = [ch]
    return buf


def _zero_buffer(frame_count=3):
    """A fake AVAudioPCMBuffer containing bit-exact float32 zeros."""
    buf = MagicMock()
    buf.frameLength.return_value = frame_count
    channel = MagicMock()
    channel.as_buffer.return_value = b"\x00" * (frame_count * 4)
    buf.floatChannelData.return_value = [channel]
    return buf


class TestGenerationIsolation:
    def test_zombie_tap_audio_discarded(self, monkeypatch):
        """A tap from a superseded generation must not feed the queue."""
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start()
        buf = _voice_buffer()

        r._tap_callback(buf, r._active_gen, 3.0, r._session)
        assert not r._queue.empty()

        r._flush()
        r._tap_callback(buf, r._active_gen - 1, 3.0, r._session)  # zombie
        assert r._queue.empty()
        r.stop()

    def test_zombie_cannot_touch_new_session_state(self, monkeypatch):
        """Even a zombie callback that raced past the generation check can
        only write its own session — the new session's rms, byte count,
        queue and chunk callback stay untouched."""
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start()
        old_session = r._session
        r.stop()
        r.start()
        new_session = r._session
        received: list = []
        r.set_on_audio_chunk(lambda b: received.append(b))
        buf = _voice_buffer()

        # Zombie with a stale generation: dropped at the gate
        r._tap_callback(buf, r._active_gen - 1, 3.0, old_session)
        # Zombie racing PAST the gate (gen matches, but it carries its
        # own session): writes land only in old_session
        r._tap_callback(buf, r._active_gen, 3.0, old_session)

        assert not received
        assert new_session.rms == 0.0
        assert new_session.total_bytes == 0
        assert new_session.queue.empty()
        assert not old_session.queue.empty()
        r.stop()

    def test_taint_stop_only_stops_its_generation(self, monkeypatch):
        """A late taint-triggered stop must not kill a newer session."""
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start()
        stale_gen = r._active_gen
        r.stop()
        r.start()  # newer session
        assert r.is_recording is True

        assert r._stop_generation(stale_gen) is None  # late taint stop
        assert r.is_recording is True

        r.stop()
        assert r.is_recording is False

    def test_each_session_gets_a_fresh_queue(self, monkeypatch):
        """A (late) stop can only drain its own generation's queue —
        never a newer session's audio."""
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start()
        q1 = r._queue
        r.stop()
        r.start()
        assert r._queue is not q1

        # Frames of the new session survive a drain of the old queue
        r._queue.put(_int16_bytes(1000))
        while not q1.empty():
            q1.get_nowait()
        assert not r._queue.empty()
        assert r.stop() is not None

    def test_stop_clears_chunk_callback(self, monkeypatch):
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start()
        session = r._session
        r.set_on_audio_chunk(lambda b: None)
        assert session.on_chunk is not None
        r.stop()
        assert session.on_chunk is None
        assert r._session is None

    def test_prepare_failure_tears_down_engine(self, monkeypatch):
        """An exception after tap install must remove the tap and stop
        the engine (mic must not stay open)."""
        engine = _mock_engine(monkeypatch)
        engine.prepare.side_effect = RuntimeError("boom")
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False

        assert r.start() is None
        assert r.is_recording is False
        engine.stop.assert_called_once()
        engine.inputNode.return_value.removeTapOnBus_.assert_called_with(0)

    def test_finalization_failure_tears_down(self, monkeypatch):
        """A failure between engine start and commit must close the mic."""
        engine = _mock_engine(monkeypatch)
        monkeypatch.setattr(
            "wenzi.audio.recorder.NSNotificationCenter",
            MagicMock(defaultCenter=MagicMock(side_effect=RuntimeError("boom"))),
        )
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False

        assert r.start() is None
        assert r.is_recording is False
        engine.stop.assert_called_once()

        # A later start() must succeed once the failure is gone
        monkeypatch.setattr(
            "wenzi.audio.recorder.NSNotificationCenter",
            MagicMock(defaultCenter=MagicMock(return_value=MagicMock(
                addObserverForName_object_queue_usingBlock_=MagicMock(
                    return_value="obs"
                ),
                removeObserver_=MagicMock(),
            ))),
        )
        r.start()
        assert r.is_recording is True
        r.stop()


class TestArmedGate:
    def test_start_unarmed_tap_discards_frames(self, monkeypatch):
        """Unarmed frames (the start-sound window) must not reach rms,
        buffers or on_chunk."""
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start(armed=False)
        received: list = []
        r.set_on_audio_chunk(lambda b: received.append(b))
        buf = _voice_buffer()

        r._tap_callback(buf, r._active_gen, 3.0, r._session)

        assert r._session.queue.empty()
        assert r._session.rms == 0.0
        assert r._session.total_bytes == 0
        assert not received
        assert r.current_level == 0.0
        r.stop()

    def test_arm_opens_gate_for_buffers_and_chunks(self, monkeypatch):
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start(armed=False)
        received: list = []
        r.set_on_audio_chunk(lambda b: received.append(b))
        buf = _voice_buffer()

        r._tap_callback(buf, r._active_gen, 3.0, r._session)  # gated
        r.arm()
        r._tap_callback(buf, r._active_gen, 3.0, r._session)  # flows

        assert r._session.queue.qsize() == 1
        assert r._session.rms > 0.0
        assert len(received) == 1
        r.stop()

    def test_start_default_is_armed(self, monkeypatch):
        """Without the keyword the sequential behavior is unchanged."""
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start()
        buf = _voice_buffer()

        r._tap_callback(buf, r._active_gen, 3.0, r._session)

        assert r._session.armed is True
        assert r._session.queue.qsize() == 1
        r.stop()

    def test_arm_without_session_is_noop(self):
        r = Recorder()
        r.arm()  # must not raise
        assert r.is_recording is False


class TestBluetoothRecovery:
    @pytest.mark.parametrize("transport", [b"blue", b"blea"])
    def test_exact_zero_rebuilds_engine_and_preserves_session(
        self,
        monkeypatch,
        transport,
    ):
        old_engine = _new_mock_engine()
        replacement = _new_mock_engine()
        _mock_engine_sequence(
            monkeypatch,
            [old_engine, replacement],
            transport=transport,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )

        recorder = Recorder(sample_rate=16000, block_ms=20)
        recorder.start()
        session = recorder._session
        original_queue = session.queue
        received = []

        def callback(data):
            received.append(data)

        recorder.set_on_audio_chunk(callback)

        recorder._tap_callback(
            _voice_buffer(),
            recorder._active_gen,
            3.0,
            session,
            engine_epoch=1,
        )
        recorder._tap_callback(
            _zero_buffer(),
            recorder._active_gen,
            3.0,
            session,
            engine_epoch=1,
        )
        _wait_until(lambda: recorder._engine is replacement)

        assert recorder._session is session
        assert session.queue is original_queue
        assert session.on_chunk is callback
        assert session.engine_epoch == 2
        assert session.saw_nonzero is True
        assert session.recovery_count == 1
        old_engine.stop.assert_called_once()

        received_before_stale_callback = len(received)
        recorder._tap_callback(
            _voice_buffer(),
            recorder._active_gen,
            3.0,
            session,
            engine_epoch=1,
        )
        assert len(received) == received_before_stale_callback

        recorder._tap_callback(
            _voice_buffer(),
            recorder._active_gen,
            3.0,
            session,
            engine_epoch=2,
        )
        assert len(received) == received_before_stale_callback + 1
        assert recorder.stop() is not None
        replacement.stop.assert_called_once()

    def test_short_startup_zeros_and_non_bluetooth_silence_do_not_recover(
        self,
        monkeypatch,
    ):
        recorder = Recorder(sample_rate=16000)
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_ZERO_RECOVERY_SECS",
            0.0,
        )
        schedule = MagicMock()
        monkeypatch.setattr(
            recorder,
            "_schedule_bluetooth_recovery",
            schedule,
        )
        zero = b"\x00" * 12
        voice = struct.pack("<3f", 0.1, 0.0, 0.0)

        bluetooth = _TapSession(is_bluetooth=True)
        recorder._track_bluetooth_zero_audio(
            zero, 3, 3.0, 1, bluetooth
        )
        schedule.assert_not_called()

        built_in = _TapSession(is_bluetooth=False)
        recorder._track_bluetooth_zero_audio(
            voice, 3, 3.0, 1, built_in
        )
        recorder._track_bluetooth_zero_audio(
            zero, 3, 3.0, 1, built_in
        )
        schedule.assert_not_called()

        recorder._track_bluetooth_zero_audio(
            voice, 3, 3.0, 1, bluetooth
        )
        recorder._track_bluetooth_zero_audio(
            zero, 3, 3.0, 1, bluetooth
        )
        schedule.assert_called_once_with(1, bluetooth)

    def test_subsecond_bluetooth_zero_run_does_not_recover(
        self,
        monkeypatch,
    ):
        recorder = Recorder(sample_rate=100)
        schedule = MagicMock()
        monkeypatch.setattr(
            recorder,
            "_schedule_bluetooth_recovery",
            schedule,
        )
        session = _TapSession(is_bluetooth=True)
        session.saw_nonzero = True

        # Successful AirPods recordings can end with about 0.72 seconds of
        # exact zeros.  Keep that valid tail below the recovery boundary.
        recorder._track_bluetooth_zero_audio(
            b"\x00" * (75 * 4), 75, 1.0, 1, session
        )
        schedule.assert_not_called()

        recorder._track_bluetooth_zero_audio(
            b"\x00" * (25 * 4), 25, 1.0, 1, session
        )
        schedule.assert_called_once_with(1, session)

    def test_bluetooth_startup_exact_zero_eventually_recovers(
        self,
        monkeypatch,
    ):
        old_engine = _new_mock_engine()
        replacement = _new_mock_engine()
        _mock_engine_sequence(monkeypatch, [old_engine, replacement])
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_STARTUP_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )

        recorder = Recorder(sample_rate=16000)
        recorder.start()
        session = recorder._session
        assert session.saw_nonzero is False

        recorder._tap_callback(
            _zero_buffer(), recorder._active_gen, 3.0, session
        )
        _wait_until(lambda: recorder._engine is replacement)

        assert session.recovery_count == 1
        assert session.engine_epoch == 2
        recorder._queue.put(_int16_bytes(1000))
        assert recorder.stop() is not None

    def test_recovery_refreshes_output_after_commit_without_recorder_lock(
        self,
        monkeypatch,
    ):
        old_engine = _new_mock_engine()
        replacement = _new_mock_engine()
        _mock_engine_sequence(monkeypatch, [old_engine, replacement])
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_STARTUP_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )

        recorder = Recorder(sample_rate=16000)
        callback_done = threading.Event()
        observations = {}

        def _on_route_recovered():
            acquired = recorder._lock.acquire(blocking=False)
            observations["lock_was_free"] = acquired
            if acquired:
                recorder._lock.release()
            observations["engine"] = recorder._engine
            observations["pending"] = recorder._session.recovery_pending
            callback_done.set()
            return True

        recorder.start(on_route_recovered=_on_route_recovered)
        session = recorder._session

        recorder._tap_callback(
            _zero_buffer(), recorder._active_gen, 3.0, session
        )
        assert callback_done.wait(timeout=2)
        _wait_until(lambda: session.recovery_pending is False)

        assert observations == {
            "lock_was_free": True,
            "engine": replacement,
            "pending": False,
        }
        recorder._queue.put(_int16_bytes(1000))
        assert recorder.stop() is not None

    def test_stop_during_route_refresh_waits_without_discarding_audio(
        self,
        monkeypatch,
    ):
        old_engine = _new_mock_engine()
        replacement = _new_mock_engine()
        _mock_engine_sequence(monkeypatch, [old_engine, replacement])
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_STARTUP_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )

        recorder = Recorder(sample_rate=16000)
        refresh_entered = threading.Event()
        allow_refresh = threading.Event()
        stop_reached = threading.Event()
        replacement.stop.side_effect = stop_reached.set

        def _on_route_recovered():
            refresh_entered.set()
            assert allow_refresh.wait(timeout=5)
            return True

        recorder.start(on_route_recovered=_on_route_recovered)
        session = recorder._session
        recorder._tap_callback(
            _zero_buffer(), recorder._active_gen, 3.0, session
        )
        assert refresh_entered.wait(timeout=2)
        assert session.recovery_pending is False
        recorder._queue.put(_int16_bytes(1000))

        result = []
        stop_thread = threading.Thread(
            target=lambda: result.append(recorder.stop())
        )
        stop_thread.start()
        assert stop_reached.wait(timeout=2)
        stop_thread.join(timeout=0.05)
        assert stop_thread.is_alive()
        assert session.capture_failed is False

        allow_refresh.set()
        stop_thread.join(timeout=2)

        assert not stop_thread.is_alive()
        assert result[0] is not None
        replacement.stop.assert_called_once()

    def test_route_refresh_exception_keeps_committed_recovery_usable(
        self,
        monkeypatch,
    ):
        old_engine = _new_mock_engine()
        replacement = _new_mock_engine()
        _mock_engine_sequence(monkeypatch, [old_engine, replacement])
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_STARTUP_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )

        def _on_route_recovered():
            raise RuntimeError("CoreAudio route is still publishing")

        recorder = Recorder(sample_rate=16000)
        recorder.start(on_route_recovered=_on_route_recovered)
        session = recorder._session
        recorder._tap_callback(
            _zero_buffer(), recorder._active_gen, 3.0, session
        )
        assert session.recovery_done.wait(timeout=2)

        assert recorder._engine is replacement
        assert session.recovery_pending is False
        assert session.capture_failed is False
        assert recorder._recovery_done.is_set()
        recorder._queue.put(_int16_bytes(1000))
        assert recorder.stop() is not None

    def test_old_config_observer_cannot_invalidate_recovered_engine(
        self,
        monkeypatch,
    ):
        old_engine = _new_mock_engine()
        replacement = _new_mock_engine()
        _mock_engine_sequence(monkeypatch, [old_engine, replacement])
        callbacks = _capture_config_callbacks(monkeypatch)
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_STARTUP_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )

        recorder = Recorder(sample_rate=16000)
        recorder.start()
        session = recorder._session
        assert len(callbacks) == 1
        old_callback = callbacks[0]
        old_engine.stop.side_effect = lambda: old_callback(None)

        recorder._tap_callback(
            _zero_buffer(), recorder._active_gen, 3.0, session
        )
        _wait_until(lambda: recorder._engine is replacement)
        assert len(callbacks) == 2
        assert session.capture_failed is False

        old_callback(None)
        assert session.capture_failed is False

        callbacks[1](None)
        assert session.capture_failed is True
        assert recorder.stop() is None

    def test_unknown_replacement_keeps_bluetooth_recovery_armed(
        self,
        monkeypatch,
    ):
        old_engine = _new_mock_engine()
        replacement = _new_mock_engine()
        _mock_engine_sequence(monkeypatch, [old_engine, replacement])
        bluetooth = _InputRoute(
            "airpods",
            "AirPods Max",
            int.from_bytes(b"blue", "big"),
            False,
        )
        unknown = _InputRoute("airpods", "AirPods Max", None, False)
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            MagicMock(side_effect=[bluetooth, unknown]),
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_STARTUP_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )
        recorder = Recorder(sample_rate=16000)
        recorder.start()
        session = recorder._session

        recorder._tap_callback(
            _zero_buffer(), recorder._active_gen, 3.0, session
        )
        _wait_until(lambda: recorder._engine is replacement)

        assert session.is_bluetooth is True
        recorder._queue.put(_int16_bytes(1000))
        assert recorder.stop() is not None

    def test_failed_recovery_discards_truncated_prefix(
        self,
        monkeypatch,
    ):
        old_engine = _new_mock_engine()
        failed_engines = [_new_mock_engine() for _ in range(3)]
        for engine in failed_engines:
            engine.startAndReturnError_.return_value = (False, "failed")
        _mock_engine_sequence(
            monkeypatch,
            [old_engine, *failed_engines],
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_RETRY_DELAY_SECS",
            0.0,
        )

        recorder = Recorder(sample_rate=16000)
        recorder.start()
        session = recorder._session
        recorder._tap_callback(
            _voice_buffer(), recorder._active_gen, 3.0, session
        )
        recorder._tap_callback(
            _zero_buffer(), recorder._active_gen, 3.0, session
        )
        _wait_until(lambda: recorder._recovery_done.is_set())

        assert session.capture_failed is True
        assert recorder.is_recording is True
        assert recorder.stop() is None
        old_engine.stop.assert_called_once()
        for engine in failed_engines:
            engine.stop.assert_called_once()

    def test_stop_then_start_waits_for_stale_recovery_cleanup(
        self,
        monkeypatch,
    ):
        old_engine = _new_mock_engine()
        stale_replacement = _new_mock_engine()
        next_engine = _new_mock_engine()
        build_entered = threading.Event()
        allow_build = threading.Event()

        def _blocking_start(_error):
            build_entered.set()
            assert allow_build.wait(timeout=5)
            return True, None

        stale_replacement.startAndReturnError_.side_effect = _blocking_start
        _mock_engine_sequence(
            monkeypatch,
            [old_engine, stale_replacement, next_engine],
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )

        recorder = Recorder(sample_rate=16000)
        recorder.start()
        old_session = recorder._session
        recorder._tap_callback(
            _voice_buffer(), recorder._active_gen, 3.0, old_session
        )
        recorder._tap_callback(
            _zero_buffer(), recorder._active_gen, 3.0, old_session
        )
        assert build_entered.wait(timeout=2)

        stop_result = []
        stop_done = threading.Event()

        def _stop_old():
            stop_result.append(recorder.stop())
            stop_done.set()

        stop_thread = threading.Thread(target=_stop_old)
        stop_thread.start()
        _wait_until(lambda: not recorder.is_recording)
        assert not stop_done.wait(timeout=0.1)

        start_result = []
        start_done = threading.Event()

        def _start_next():
            start_result.append(recorder.start())
            start_done.set()

        start_thread = threading.Thread(target=_start_next)
        start_thread.start()
        assert not start_done.wait(timeout=0.1)
        next_engine.startAndReturnError_.assert_not_called()

        allow_build.set()
        stop_thread.join(timeout=2)
        start_thread.join(timeout=2)
        assert not stop_thread.is_alive()
        assert not start_thread.is_alive()
        assert stop_result == [None]
        assert start_result == ["TestMic"]
        assert recorder._engine is next_engine
        assert recorder._session is not old_session
        stale_replacement.stop.assert_called_once()

        recorder._queue.put(_int16_bytes(1000))
        assert recorder.stop() is not None
        next_engine.stop.assert_called_once()

    def test_start_abandoned_while_waiting_never_builds_engine(
        self,
        monkeypatch,
    ):
        engine = _mock_engine(monkeypatch)
        recorder = Recorder()
        recorder._recovery_done.clear()
        result = []
        thread = threading.Thread(target=lambda: result.append(recorder.start()))
        thread.start()
        _wait_until(lambda: recorder._starting_since is not None)

        recorder.mark_tainted()
        recorder._recovery_done.set()
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert result == [None]
        assert recorder.is_recording is False
        engine.startAndReturnError_.assert_not_called()

    def test_recovery_dispatch_failure_reopens_both_gates(
        self,
        monkeypatch,
    ):
        _mock_engine(monkeypatch)
        recorder = Recorder()
        recorder.start()
        session = recorder._session
        monkeypatch.setattr(
            "wenzi.audio.recorder.threading.Thread",
            MagicMock(side_effect=RuntimeError("cannot create thread")),
        )

        recorder._schedule_bluetooth_recovery(
            recorder._active_gen,
            session,
        )

        assert session.recovery_pending is False
        assert session.recovery_count == 0
        assert session.recovery_done.is_set()
        assert recorder._recovery_done.is_set()
        recorder._queue.put(_int16_bytes(1000))
        assert recorder.stop() is not None

    def test_recovery_limit_marks_repeated_zero_stream_failed(
        self,
        monkeypatch,
    ):
        engines = [_new_mock_engine() for _ in range(3)]
        _mock_engine_sequence(monkeypatch, engines)
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_ZERO_RECOVERY_SECS",
            0.0,
        )
        monkeypatch.setattr(
            Recorder,
            "_BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS",
            0.0,
        )

        recorder = Recorder(sample_rate=16000)
        recorder.start()
        session = recorder._session
        recorder._tap_callback(
            _voice_buffer(), recorder._active_gen, 3.0, session
        )

        for expected_engine in engines[1:]:
            recorder._tap_callback(
                _zero_buffer(),
                recorder._active_gen,
                3.0,
                session,
                engine_epoch=session.engine_epoch,
            )
            _wait_until(lambda: recorder._engine is expected_engine)

        recorder._tap_callback(
            _zero_buffer(),
            recorder._active_gen,
            3.0,
            session,
            engine_epoch=session.engine_epoch,
        )

        assert session.recovery_count == 2
        assert session.capture_failed is True
        assert recorder.stop() is None


class TestRouteCache:
    def test_legacy_explicit_config_follows_each_new_default(
        self,
        monkeypatch,
    ):
        engine = _mock_engine(monkeypatch)
        built_in = _InputRoute(
            "built-in",
            "Built-in",
            int.from_bytes(b"bltn", "big"),
            False,
        )
        airpods = _InputRoute(
            "airpods",
            "AirPods Max",
            int.from_bytes(b"blue", "big"),
            False,
        )
        select = MagicMock(side_effect=[built_in, airpods])
        monkeypatch.setattr("wenzi.audio.recorder._select_input_route", select)
        au = engine.inputNode.return_value.AUAudioUnit.return_value

        r = Recorder(sample_rate=16000, block_ms=20, device="uid-1")
        r._query_device_name_enabled = False
        r.start()
        assert r.is_recording is True
        assert r._session.is_bluetooth is False
        r.stop()
        r.start()
        assert r.is_recording is True
        assert r._session.is_bluetooth is True
        r.stop()

        assert [call.args for call in select.call_args_list] == [(None,), (None,)]
        au.setDeviceID_error_.assert_not_called()

    def test_automatic_route_never_cached(self, monkeypatch):
        _mock_engine(monkeypatch)
        select = MagicMock(
            return_value=_InputRoute("test-uid", "TestMic", None, False)
        )
        monkeypatch.setattr("wenzi.audio.recorder._select_input_route", select)

        r = Recorder(sample_rate=16000, block_ms=20)
        r._query_device_name_enabled = False
        r.start()
        r.stop()
        r.start()
        r.stop()

        assert select.call_count == 2
        assert r._route_cache is None

    def test_device_setter_keeps_automatic_and_drops_preflight(
        self,
        monkeypatch,
    ):
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20, device="uid-1")
        route = r.preflight_input_route()
        assert r._preflight_route is route

        r.device = "uid-2"
        assert r._route_cache is None
        assert r._preflight_route is None
        assert r.device is None

    def test_config_change_invalidates_preflight(self, monkeypatch):
        _mock_engine(monkeypatch)
        r = Recorder(sample_rate=16000, block_ms=20)
        r.preflight_input_route()
        assert r._preflight_route is not None

        r._on_config_change(None, None, None)
        assert r._route_cache is None
        assert r._preflight_route is None

    def test_config_change_discards_active_capture(self, monkeypatch):
        _mock_engine(monkeypatch)
        recorder = Recorder(sample_rate=16000, block_ms=20)
        recorder.start()
        recorder._queue.put(_int16_bytes(1000))

        recorder._on_config_change(
            recorder._active_gen,
            recorder._session,
            recorder._session.engine_epoch,
        )

        assert recorder._session.capture_failed is True
        assert recorder.stop() is None

    def test_start_failure_invalidates_preflight(self, monkeypatch):
        engine = _mock_engine(monkeypatch)
        engine.startAndReturnError_.return_value = (False, "boom")
        r = Recorder(sample_rate=16000, block_ms=20)
        r.preflight_input_route()
        assert r.start() is None
        assert r._route_cache is None
        assert r._preflight_route is None

    def test_coreaudio_bindings_created_once(self, monkeypatch):
        import ctypes.util

        from wenzi.audio import recorder as recorder_module

        monkeypatch.setattr(recorder_module, "_COREAUDIO_BINDINGS", None)
        calls: list = []
        real_find = ctypes.util.find_library

        def _counting_find(name):
            calls.append(name)
            return real_find(name)

        monkeypatch.setattr(ctypes.util, "find_library", _counting_find)

        first = recorder_module._coreaudio_bindings()
        second = recorder_module._coreaudio_bindings()

        assert first is second
        assert calls == ["CoreAudio", "CoreFoundation"]


class TestRmsInt16:
    def test_silence(self):
        assert _rms_int16(b"\x00" * 640) == 0

    def test_constant_value(self):
        data = _int16_bytes(500, 100)
        assert _rms_int16(data) == 500

    def test_empty(self):
        assert _rms_int16(b"") == 0


class TestResampleLinear:
    def test_identity_ratio(self):
        """Ratio 1.0 should return the same samples."""
        src = (0.1, 0.2, 0.3, 0.4)
        result = _resample_linear(src, 4, 4, 1.0)
        assert len(result) == 4
        for a, b in zip(result, src):
            assert abs(a - b) < 1e-6

    def test_3to1_decimation(self):
        """48kHz→16kHz: ratio=3, 9 input → 3 output."""
        src = tuple(float(i) for i in range(9))
        result = _resample_linear(src, 9, 3, 3.0)
        assert len(result) == 3
        assert result[0] == 0.0
        assert result[1] == 3.0
        assert result[2] == 6.0

    def test_empty_input(self):
        assert _resample_linear((), 0, 0, 3.0) == []

    def test_fractional_ratio(self):
        """Non-integer ratio (e.g. 44.1→16) should interpolate."""
        src = (0.0, 1.0, 0.0)
        ratio = 3.0 / 2.0  # 1.5
        result = _resample_linear(src, 3, 2, ratio)
        assert len(result) == 2
        assert result[0] == 0.0  # src[0]
        assert abs(result[1] - 0.5) < 1e-6  # interpolated between src[1] and src[2]
