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


def _mock_engine(monkeypatch):
    """Patch AVAudioEngine and friends so start() succeeds without hardware."""
    mock_engine = MagicMock()
    mock_input_node = MagicMock()
    mock_hw_fmt = MagicMock()
    mock_hw_fmt.sampleRate.return_value = 48000.0
    mock_input_node.outputFormatForBus_.return_value = mock_hw_fmt
    mock_engine.inputNode.return_value = mock_input_node
    mock_engine.startAndReturnError_.return_value = (True, None)

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


def _capture_device(name: str, uid: str, transport: bytes):
    device = MagicMock()
    device.localizedName.return_value = name
    device.uniqueID.return_value = uid
    device.transportType.return_value = int.from_bytes(transport, "big")
    return device


class TestInputRouteSelection:
    def test_automatic_uses_built_in_for_bluetooth_default(self, monkeypatch):
        bluetooth = _capture_device("AirPods Max", "airpods", b"blue")
        built_in = _capture_device(
            "MacBook Pro Microphone", "builtin", b"bltn"
        )
        capture = MagicMock()
        capture.defaultDeviceWithMediaType_.return_value = bluetooth
        capture.devicesWithMediaType_.return_value = [bluetooth, built_in]
        monkeypatch.setattr("wenzi.audio.recorder.AVCaptureDevice", capture)

        route = _select_input_route(None)

        assert route.uid == "builtin"
        assert route.name == "MacBook Pro Microphone"
        assert route.bind is True

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

    def test_automatic_keeps_bluetooth_when_no_built_in_exists(
        self, monkeypatch
    ):
        bluetooth = _capture_device("AirPods Max", "airpods", b"blue")
        capture = MagicMock()
        capture.defaultDeviceWithMediaType_.return_value = bluetooth
        capture.devicesWithMediaType_.return_value = [bluetooth]
        monkeypatch.setattr("wenzi.audio.recorder.AVCaptureDevice", capture)

        route = _select_input_route(None)

        assert route.uid == "airpods"
        assert route.name == "AirPods Max"
        assert route.bind is False

    def test_automatic_keeps_bluetooth_when_listing_fails(self, monkeypatch):
        bluetooth = _capture_device("AirPods Max", "airpods", b"blue")
        capture = MagicMock()
        capture.defaultDeviceWithMediaType_.return_value = bluetooth
        capture.devicesWithMediaType_.side_effect = RuntimeError("unavailable")
        monkeypatch.setattr("wenzi.audio.recorder.AVCaptureDevice", capture)

        route = _select_input_route(None)

        assert route.uid == "airpods"
        assert route.bind is False

    def test_explicit_airpods_is_honored(self, monkeypatch):
        bluetooth = _capture_device("AirPods Max", "airpods", b"blue")
        built_in = _capture_device(
            "MacBook Pro Microphone", "builtin", b"bltn"
        )
        capture = MagicMock()
        capture.devicesWithMediaType_.return_value = [bluetooth, built_in]
        monkeypatch.setattr("wenzi.audio.recorder.AVCaptureDevice", capture)

        route = _select_input_route("airpods")

        assert route.uid == "airpods"
        assert route.name == "AirPods Max"
        assert route.bind is True
        capture.defaultDeviceWithMediaType_.assert_not_called()


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
        # RMS lives on the session: 500 → level = 500/800 = 0.625
        r._session = _TapSession()
        r._session.rms = 500.0
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

    def test_explicit_device_not_found_fails_start(self, monkeypatch):
        engine = _mock_engine(monkeypatch)
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            lambda uid: _InputRoute(uid, "Missing Mic", None, True),
        )

        r = Recorder(sample_rate=16000, block_ms=20, device="missing")
        r._query_device_name_enabled = False
        assert r.start() is None
        assert r.is_recording is False
        engine.stop.assert_called_once()

    def test_device_bind_false_result_fails_start(self, monkeypatch):
        engine = _mock_engine(monkeypatch)
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            lambda uid: _InputRoute(uid, "Selected Mic", None, True),
        )
        monkeypatch.setattr("wenzi.audio.recorder._resolve_device_id", lambda uid: 42)
        au = engine.inputNode.return_value.AUAudioUnit.return_value
        au.setDeviceID_error_.return_value = False

        r = Recorder(sample_rate=16000, block_ms=20, device="selected")
        r._query_device_name_enabled = False
        assert r.start() is None
        assert r.is_recording is False
        engine.stop.assert_called_once()

    def test_device_bind_mismatched_readback_fails_start(self, monkeypatch):
        engine = _mock_engine(monkeypatch)
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            lambda uid: _InputRoute(uid, "Selected Mic", None, True),
        )
        monkeypatch.setattr("wenzi.audio.recorder._resolve_device_id", lambda uid: 42)
        au = engine.inputNode.return_value.AUAudioUnit.return_value
        au.setDeviceID_error_.return_value = True
        au.deviceID.return_value = 41

        r = Recorder(sample_rate=16000, block_ms=20, device="selected")
        r._query_device_name_enabled = False
        assert r.start() is None
        assert r.is_recording is False
        engine.stop.assert_called_once()

    def test_device_bind_verified_before_recording(self, monkeypatch):
        engine = _mock_engine(monkeypatch)
        monkeypatch.setattr(
            "wenzi.audio.recorder._select_input_route",
            lambda uid: _InputRoute(uid, "Selected Mic", None, True),
        )
        monkeypatch.setattr("wenzi.audio.recorder._resolve_device_id", lambda uid: 42)
        au = engine.inputNode.return_value.AUAudioUnit.return_value
        au.setDeviceID_error_.return_value = True
        au.deviceID.return_value = 42

        r = Recorder(sample_rate=16000, block_ms=20, device="selected")
        r._query_device_name_enabled = False
        r.start()

        assert r.is_recording is True
        au.setDeviceID_error_.assert_called_once_with(42, None)
        r.stop()


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
