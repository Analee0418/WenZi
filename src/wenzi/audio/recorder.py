"""Audio recording using AVAudioEngine (macOS AVFoundation)."""

from __future__ import annotations

import array
import io
import logging
import queue
import struct
import threading
import time
import wave
from dataclasses import dataclass

from AVFoundation import AVAudioEngine, AVCaptureDevice, AVMediaTypeAudio
from Foundation import NSNotificationCenter

logger = logging.getLogger(__name__)

# Notification name (string constant; not always in the PyObjC bindings).
_ENGINE_CONFIG_CHANGE = "AVAudioEngineConfigurationChangeNotification"

# CoreAudio transport FourCC values.  AVCaptureDevice.transportType() exposes
# these values even though the constants are not consistently exported by
# every PyObjC build.
_TRANSPORT_BLUETOOTH = int.from_bytes(b"blue", "big")
_TRANSPORT_BUILT_IN = int.from_bytes(b"bltn", "big")


@dataclass(frozen=True)
class _InputRoute:
    """Resolved capture route for one Recorder.start() attempt."""

    uid: str | None
    name: str | None
    transport_type: int | None
    bind: bool


def list_input_devices() -> list[dict]:
    """Return a list of available audio input devices.

    Each dict has keys: ``uid`` (str) and ``name`` (str).
    The ``uid`` is stable across reboots and suitable for config storage.
    """
    try:
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeAudio)
    except Exception:
        logger.warning("Failed to enumerate audio devices", exc_info=True)
        return []
    result: list[dict] = []
    for d in devices:
        uid = str(d.uniqueID())
        name = str(d.localizedName())
        if uid and name:
            result.append({"uid": uid, "name": name})
    return result


def default_input_device_name() -> str | None:
    """Return the localized name of the current system default input device."""
    try:
        d = AVCaptureDevice.defaultDeviceWithMediaType_(AVMediaTypeAudio)
        return str(d.localizedName()) if d else None
    except Exception:
        return None


def _capture_device_route(device, *, bind: bool) -> _InputRoute:
    """Build an input route from an AVCaptureDevice."""
    try:
        uid = str(device.uniqueID())
    except Exception:
        uid = None
    try:
        name = str(device.localizedName())
    except Exception:
        name = None
    try:
        transport_type = int(device.transportType())
    except Exception:
        transport_type = None
    return _InputRoute(uid or None, name or None, transport_type, bind)


def _select_input_route(configured_uid: str | None) -> _InputRoute:
    """Resolve the effective input route without activating the microphone.

    An explicit UID is always honored.  In automatic mode, a Bluetooth
    system default is replaced with the Mac's built-in microphone so opening
    the input stream does not force Bluetooth playback into its call profile.
    Other system defaults are left untouched.
    """
    if configured_uid:
        try:
            devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeAudio)
        except Exception:
            devices = []
        for device in devices:
            route = _capture_device_route(device, bind=True)
            if route.uid == configured_uid:
                return route
        # CoreAudio remains the source of truth for whether an explicit UID
        # exists.  Keeping the UID here lets _resolve_device_id() either bind
        # it or fail closed instead of silently changing the user's choice.
        return _InputRoute(configured_uid, None, None, True)

    try:
        default_device = AVCaptureDevice.defaultDeviceWithMediaType_(
            AVMediaTypeAudio
        )
    except Exception:
        logger.warning("Failed to query the default input device", exc_info=True)
        return _InputRoute(None, None, None, False)
    if default_device is None:
        return _InputRoute(None, None, None, False)

    default_route = _capture_device_route(default_device, bind=False)
    if default_route.transport_type != _TRANSPORT_BLUETOOTH:
        return default_route

    try:
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeAudio)
    except Exception:
        logger.warning(
            "The default input is Bluetooth, but input devices could not be "
            "listed; keeping the system default",
            exc_info=True,
        )
        return default_route
    for device in devices:
        route = _capture_device_route(device, bind=True)
        if route.transport_type == _TRANSPORT_BUILT_IN:
            logger.info(
                "Automatic input switched from Bluetooth default %s to built-in %s",
                default_route.name or default_route.uid or "unknown",
                route.name or route.uid or "unknown",
            )
            return route

    logger.warning(
        "The default input is Bluetooth and no built-in microphone is available; "
        "keeping the system default"
    )
    return default_route


def automatic_input_device_name() -> str | None:
    """Return the device name automatic routing would use, if available."""
    try:
        return _select_input_route(None).name
    except Exception:
        logger.warning("Failed to resolve the automatic input device", exc_info=True)
        return None


def _resolve_device_id(uid: str) -> int | None:
    """Find the CoreAudio AudioDeviceID for an AVCaptureDevice UID.

    AVAudioEngine's input node uses CoreAudio device IDs internally.
    We bridge from AVCaptureDevice UID → AudioDeviceID via the
    ``transportType`` + private ``_audioDeviceID`` selector, falling
    back to a CoreAudio property lookup.
    """
    import ctypes
    import ctypes.util

    _ca = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio"))

    # AudioObjectPropertyAddress
    class _Addr(ctypes.Structure):
        _fields_ = [
            ("mSelector", ctypes.c_uint32),
            ("mScope", ctypes.c_uint32),
            ("mElement", ctypes.c_uint32),
        ]

    _ca.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
    _ca.AudioObjectGetPropertyDataSize.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Addr),
        ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
    ]
    _ca.AudioObjectGetPropertyData.restype = ctypes.c_int32
    _ca.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Addr),
        ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]

    _cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
    _cf.CFStringGetLength.restype = ctypes.c_long
    _cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
    _cf.CFStringGetCString.restype = ctypes.c_bool
    _cf.CFStringGetCString.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32,
    ]
    _cf.CFRelease.restype = None
    _cf.CFRelease.argtypes = [ctypes.c_void_p]

    kSys = 1  # kAudioObjectSystemObject
    kDevs = 0x64657623  # 'dev#'
    kUID = 0x75696420  # 'uid '
    kGlob = 0x676C6F62  # 'glob'
    kUTF8 = 0x08000100

    # Get all device IDs
    addr = _Addr(kDevs, kGlob, 0)
    size = ctypes.c_uint32(0)
    if _ca.AudioObjectGetPropertyDataSize(kSys, ctypes.byref(addr), 0, None, ctypes.byref(size)) != 0:
        return None
    n = size.value // 4
    ids = (ctypes.c_uint32 * n)()
    if _ca.AudioObjectGetPropertyData(kSys, ctypes.byref(addr), 0, None, ctypes.byref(size), ids) != 0:
        return None

    # Match UID
    for did in ids:
        addr2 = _Addr(kUID, kGlob, 0)
        sz = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        cf_str = ctypes.c_void_p()
        if _ca.AudioObjectGetPropertyData(did, ctypes.byref(addr2), 0, None, ctypes.byref(sz), ctypes.byref(cf_str)) != 0:
            continue
        if not cf_str.value:
            continue
        try:
            length = _cf.CFStringGetLength(cf_str) * 4 + 1
            buf = ctypes.create_string_buffer(length)
            if _cf.CFStringGetCString(cf_str, buf, length, kUTF8):
                if buf.value.decode("utf-8") == uid:
                    return int(did)
        finally:
            _cf.CFRelease(cf_str)
    return None


class _TapSession:
    """Mutable per-start() recording state, owned by that start's tap.

    Everything a tap callback writes (rms, byte count, frame queue, chunk
    callback) lives here, so even a zombie callback that raced past the
    generation check can only ever touch its own session — never a newer
    one's shared state.
    """

    __slots__ = ("queue", "total_bytes", "rms", "on_chunk")

    def __init__(self) -> None:
        self.queue: queue.Queue[bytes] = queue.Queue()
        self.total_bytes = 0
        self.rms = 0.0
        self.on_chunk = None


class Recorder:
    """Record audio from the microphone using AVAudioEngine. Thread-safe start/stop."""

    # RMS threshold for silence detection (int16 range: 0-32768).
    # Typical quiet room noise is ~100-300, speech is ~1000+.
    DEFAULT_SILENCE_RMS = 20
    # Reference RMS for normalizing current_level to 0.0-1.0 range.
    # Normal speech (~1000-3000 RMS) maps to roughly 0.5-1.0.
    _LEVEL_REFERENCE_RMS = 800.0
    # Max seconds _starting may remain True before it is considered stuck
    # and forcibly reset, allowing a new start() to proceed.
    _STARTING_STALE_SECS = 10.0

    def __init__(
        self,
        sample_rate: int = 16000,
        block_ms: int = 20,
        device: str | None = None,
        max_session_bytes: int = 20 * 1024 * 1024,
        silence_rms: int = DEFAULT_SILENCE_RMS,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.device = device
        self.max_session_bytes = max_session_bytes
        self.silence_rms = silence_rms

        self._queue: queue.Queue[bytes] = queue.Queue()
        self._engine: AVAudioEngine | None = None
        self._hw_sample_rate: float = 0.0
        self._resample_ratio: float = 0.0
        self._lock = threading.Lock()
        self._recording = False
        # Non-None while start() is in progress (value = monotonic timestamp).
        self._starting_since: float | None = None
        # Start-attempt generation counter.  Each start() claims a new
        # generation; it may only commit while it is still the current
        # generation and not abandoned (see mark_tainted()).
        self._start_gen = 0
        self._abandoned_gen = 0
        # Generation of the committed (currently recording) engine; 0 when
        # idle.  Tap callbacks carry their own generation and are ignored
        # unless it matches, so a zombie engine can never feed audio into
        # a newer session.
        self._active_gen = 0
        # The committed (currently recording) session's mutable state.
        # self._queue stays as an alias of _session.queue for stop().
        self._session: _TapSession | None = None
        self._last_device_name: str | None = None
        self._query_device_name_enabled: bool = True
        self._config_observer = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def last_device_name(self) -> str | None:
        """Return the last known input device name, or None."""
        return self._last_device_name

    @property
    def current_level(self) -> float:
        """Return current audio level normalized to 0.0-1.0.

        Uses ``_LEVEL_REFERENCE_RMS`` (800) as reference so normal
        speech (~1000-3000 RMS) maps to roughly 0.5-1.0.
        """
        session = self._session
        if session is None:
            return 0.0
        return min(1.0, session.rms / self._LEVEL_REFERENCE_RMS)

    def start(self) -> str | None:
        """Start recording. Returns the input device name, or None.

        Engine creation happens **outside** the lock so that a hung
        AVFoundation call cannot deadlock subsequent ``stop()`` /
        ``is_recording`` calls.  ``_starting_since`` prevents
        concurrent ``start()`` calls from racing.
        """
        # --- Phase 1: claim the "starting" slot (lock held briefly) -----
        with self._lock:
            if self._recording:
                return self._last_device_name
            if self._starting_since is not None:
                elapsed = time.monotonic() - self._starting_since
                if elapsed > self._STARTING_STALE_SECS:
                    logger.warning(
                        "Previous start() appears stuck (%.0fs), resetting",
                        elapsed,
                    )
                    # Abandon the stuck attempt: if it ever finishes, it
                    # must tear its engine down instead of committing.
                    self._abandoned_gen = self._start_gen
                    self._starting_since = None
                else:
                    # Refuse instead of pretending success: returning the
                    # device name here made callers believe recording had
                    # started while no engine existed.
                    raise RuntimeError(
                        "Recorder.start() already in progress"
                    )
            self._start_gen += 1
            gen = self._start_gen
            self._starting_since = time.monotonic()

        # --- Phase 2: create AVAudioEngine and audio graph (lock free) --
        engine = None
        try:
            configured_device = self.device
            route = _select_input_route(configured_device)
            engine = AVAudioEngine.alloc().init()
            input_node = engine.inputNode()

            if route.bind:
                if not route.uid:
                    raise RuntimeError("Selected input device has no UID")
                dev_id = _resolve_device_id(route.uid)
                if dev_id is None:
                    raise RuntimeError(
                        f"Input device uid={route.uid!r} was not found"
                    )
                au = input_node.AUAudioUnit()
                changed = au.setDeviceID_error_(dev_id, None)
                if not changed:
                    raise RuntimeError(
                        f"CoreAudio rejected input device uid={route.uid!r}"
                    )
                actual_dev_id = int(au.deviceID())
                if actual_dev_id != dev_id:
                    raise RuntimeError(
                        "CoreAudio input route mismatch: "
                        f"requested id={dev_id}, actual id={actual_dev_id}"
                    )
            else:
                dev_id = None

            logger.info(
                "Input route configured=%s effective_uid=%s name=%s "
                "transport=%s coreaudio_id=%s",
                configured_device or "automatic",
                route.uid or "system-default",
                route.name or "unknown",
                route.transport_type if route.transport_type is not None else "unknown",
                dev_id if dev_id is not None else "system-default",
            )
            hw_fmt = input_node.outputFormatForBus_(0)
            hw_sample_rate = hw_fmt.sampleRate()
            resample_ratio = hw_sample_rate / self.sample_rate

            # Each start() gets its own session state, captured by the
            # tap closure: audio, rms, byte counts and the chunk callback
            # of a zombie engine can only ever land in its own session —
            # never a newer one's.
            session = _TapSession()

            # Install tap on input node at its native format; resampling
            # happens in _tap_callback with the closure-captured ratio.
            def tap_block(buf, when):
                self._tap_callback(buf, gen, resample_ratio, session)

            input_node.installTapOnBus_bufferSize_format_block_(
                0,
                int(hw_sample_rate * self.block_ms / 1000),
                hw_fmt,
                tap_block,
            )

            engine.prepare()
            ok, err = engine.startAndReturnError_(None)
            if not ok:
                logger.error("AVAudioEngine start failed: %s", err)
                self._teardown_engine(engine, None)
                with self._lock:
                    if gen == self._start_gen:
                        self._starting_since = None
                return None

        except Exception:
            logger.error("Failed to create audio engine", exc_info=True)
            # The tap may already be installed or the engine started —
            # tear down whatever exists so the mic cannot stay open.
            if engine is not None:
                self._teardown_engine(engine, None)
            with self._lock:
                if gen == self._start_gen:
                    self._starting_since = None
            return None

        # --- Phases 3-5: any failure below must tear the engine down ----
        observer = None
        try:
            # Phase 3: device name query
            device_name = route.name if self._query_device_name_enabled else None

            # Phase 4: register for config change notifications
            observer = (
                NSNotificationCenter.defaultCenter()
                .addObserverForName_object_queue_usingBlock_(
                    _ENGINE_CONFIG_CHANGE,
                    engine,
                    None,
                    lambda note: self._on_config_change(),
                )
            )

            # Phase 5: commit (lock held briefly)
            with self._lock:
                if gen == self._start_gen and gen > self._abandoned_gen:
                    self._engine = engine
                    self._session = session
                    self._queue = session.queue
                    self._recording = True
                    self._active_gen = gen
                    self._hw_sample_rate = hw_sample_rate
                    self._resample_ratio = resample_ratio
                    self._starting_since = None
                    self._last_device_name = device_name
                    self._config_observer = observer
                    logger.info(
                        "Recording started (sr=%d, hw=%.0f Hz, device=%s)",
                        self.sample_rate,
                        hw_sample_rate,
                        device_name or "unknown",
                    )
                    return device_name
                if gen == self._start_gen:
                    self._starting_since = None
        except Exception:
            logger.error("start() finalization failed", exc_info=True)
            with self._lock:
                if gen == self._start_gen:
                    self._starting_since = None

        # Abandoned, superseded by a newer start(), or finalization
        # failed: tear the engine down instead of committing, otherwise
        # the microphone stays open with no session owning it.
        logger.warning(
            "start() did not commit (abandoned/superseded/failed); "
            "tearing down engine"
        )
        self._teardown_engine(engine, observer)
        return None

    def stop(self) -> bytes | None:
        """Stop recording and return WAV data as bytes, or None if nothing recorded."""
        return self._stop_generation(None)

    def _stop_generation(self, only_gen: int | None) -> bytes | None:
        """Stop recording; with *only_gen*, only if that generation is active.

        The generation filter lets a late taint-triggered stop never kill
        a newer session that started in the meantime.
        """
        with self._lock:
            if not self._recording:
                return None
            if only_gen is not None and self._active_gen != only_gen:
                return None

            self._recording = False
            self._active_gen = 0
            engine = self._engine
            self._engine = None
            observer = self._config_observer
            self._config_observer = None
            session = self._session
            self._session = None
            # Break the transcriber reference on OUR session only — a
            # stop racing a new session can never clear the new session's
            # callback, because that one lives on a different object.
            if session is not None:
                session.on_chunk = None

        # Stop engine, remove tap, detach notification observer
        self._teardown_engine(engine, observer)

        if session is None:
            return None

        # Collect this generation's buffered frames.  The queue is
        # per-session: a new session's audio lives in its own queue and
        # is untouched by this (possibly late) stop.
        frames: list[bytes] = []
        while not session.queue.empty():
            try:
                frames.append(session.queue.get_nowait())
            except queue.Empty:
                break

        if not frames:
            logger.warning("No audio frames captured")
            return None

        audio_bytes = b"".join(frames)
        n_samples = len(audio_bytes) // 2
        duration = n_samples / self.sample_rate
        rms = _rms_int16(audio_bytes)
        logger.info(
            "Recording stopped, captured %d samples (%.1fs), RMS=%d",
            n_samples,
            duration,
            rms,
        )

        if rms < self.silence_rms:
            logger.warning(
                "Audio below silence threshold (RMS=%d < %d), discarding",
                rms,
                self.silence_rms,
            )
            return None

        # Encode as WAV in memory
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)
        return buf.getvalue()

    def set_on_audio_chunk(self, cb: callable) -> None:
        """Set the current session's audio chunk callback (raw int16 PCM).

        The callback is stored on the active session object, so a zombie
        tap from a superseded start() can never invoke it.
        """
        session = self._session
        if session is not None:
            session.on_chunk = cb

    def clear_on_audio_chunk(self) -> None:
        """Remove the current session's audio chunk callback."""
        session = self._session
        if session is not None:
            session.on_chunk = None

    def _tap_callback(self, buffer, gen: int, ratio: float, session) -> None:
        """Process audio from the AVAudioEngine tap.

        Called on a real-time audio thread.  Exceptions MUST be caught
        because an unhandled exception propagates through PyObjC and
        crashes the process.  *gen*, *ratio* and *session* come from the
        start() that installed the tap: a stale generation is discarded
        up front, and every write below goes to *session* — this tap's
        own state — so even a callback that raced past the check can
        never touch a newer session's rms, byte count, queue or chunk
        callback.
        """
        try:
            if not self._recording or gen != self._active_gen:
                return

            in_frames = buffer.frameLength()
            if in_frames == 0:
                return

            # Read float32 samples from the native-rate input buffer
            channel0 = buffer.floatChannelData()[0]
            raw = bytes(channel0.as_buffer(in_frames))
            floats = struct.unpack(f"<{in_frames}f", raw)

            # Resample to target rate via linear interpolation
            out_count = int(in_frames / ratio)
            resampled = _resample_linear(floats, in_frames, out_count, ratio)

            # RMS from float32 data
            if out_count > 0:
                sum_sq = sum(s * s for s in resampled)
                session.rms = (sum_sq / out_count) ** 0.5 * 32768

            # Convert float32 → int16 bytes
            int16_data = struct.pack(
                f"<{out_count}h",
                *(max(-32768, min(32767, int(s * 32768))) for s in resampled),
            )

            byte_len = len(int16_data)
            if session.total_bytes + byte_len > self.max_session_bytes:
                logger.warning("Max session size reached, dropping frames")
                return

            session.total_bytes += byte_len
            try:
                session.queue.put_nowait(int16_data)
            except queue.Full:
                logger.warning("Audio queue full, dropping frame")

            cb = session.on_chunk
            # Re-check liveness right before invoking: stop() may have
            # cleared the callback while this frame was in flight.
            if cb is not None and self._recording and gen == self._active_gen:
                try:
                    cb(int16_data)
                except Exception:
                    logger.debug("Audio chunk callback error", exc_info=True)

        except Exception:
            logger.debug("Tap callback error", exc_info=True)

    def mark_tainted(self) -> None:
        """Abandon an in-flight start() whose caller gave up waiting.

        An abandoned start() tears its engine down when it eventually
        finishes instead of committing — otherwise the microphone would
        stay open with no recording session owning it.  If start()
        committed just before the taint arrived, stop the engine on a
        helper thread (stop() blocks on AVFoundation and the caller may
        be on the asyncio loop thread).
        """
        with self._lock:
            if self._starting_since is not None:
                self._abandoned_gen = self._start_gen
                logger.warning(
                    "In-flight start() abandoned; "
                    "late engine will be torn down"
                )
                return
            if not self._recording:
                return
            gen = self._active_gen
        logger.warning("start() committed before taint; stopping engine")
        threading.Thread(
            target=lambda: self._stop_generation(gen),
            name="recorder-taint-stop", daemon=True,
        ).start()

    def _teardown_engine(self, engine, observer) -> None:
        """Stop an engine and detach its tap and config-change observer.

        Every step is individually guarded — teardown must always run to
        completion, or a failed step leaves the microphone open.
        """
        if engine is not None:
            try:
                engine.inputNode().removeTapOnBus_(0)
            except Exception as e:
                logger.warning("Error removing tap: %s", e)
            try:
                engine.stop()
            except Exception as e:
                logger.warning("Error stopping engine: %s", e)
        if observer is not None:
            try:
                NSNotificationCenter.defaultCenter().removeObserver_(observer)
            except Exception as e:
                logger.warning("Error removing observer: %s", e)

    def _on_config_change(self) -> None:
        """Handle AVAudioEngine configuration change (device added/removed)."""
        logger.info("Audio engine configuration changed")
        # If not recording, nothing to do — next start() creates a fresh engine.
        # If recording, the engine has already stopped; we cannot seamlessly
        # restart mid-session without losing audio.  Log it and let the
        # current session end naturally when stop() is called.
        if self._recording:
            logger.warning(
                "Audio configuration changed during recording; "
                "current session may have gaps"
            )

    def _flush(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


def _rms_int16(data: bytes) -> int:
    """Compute RMS of raw int16 PCM bytes."""
    n = len(data) // 2
    if n == 0:
        return 0
    arr = array.array("h", data)
    sum_sq = sum(s * s for s in arr)
    return int((sum_sq / n) ** 0.5)


def _resample_linear(
    samples: tuple[float, ...],
    in_count: int,
    out_count: int,
    ratio: float,
) -> list[float]:
    """Resample float32 audio via linear interpolation.

    Works for any sample-rate ratio (integer or fractional).
    """
    if out_count == 0 or in_count == 0:
        return []
    last = in_count - 1
    result: list[float] = []
    for i in range(out_count):
        src = i * ratio
        idx = int(src)
        if idx >= last:
            result.append(samples[last])
        else:
            frac = src - idx
            result.append(samples[idx] + (samples[idx + 1] - samples[idx]) * frac)
    return result
