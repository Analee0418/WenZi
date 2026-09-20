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
from collections.abc import Callable
from dataclasses import dataclass

from AVFoundation import AVAudioEngine, AVCaptureDevice, AVMediaTypeAudio
from Foundation import NSNotificationCenter

logger = logging.getLogger(__name__)

# Notification name (string constant; not always in the PyObjC bindings).
_ENGINE_CONFIG_CHANGE = "AVAudioEngineConfigurationChangeNotification"
_BLUETOOTH_TRANSPORTS = {
    int.from_bytes(b"blue", "big"),
    int.from_bytes(b"blea", "big"),
}


def _input_route_has_bluetooth_risk(route: _InputRoute) -> bool:
    transport_type = route.transport_type
    return (
        transport_type is None
        or transport_type in _BLUETOOTH_TRANSPORTS
    )

@dataclass(frozen=True)
class _InputRoute:
    """Resolved capture route for one Recorder.start() attempt."""

    uid: str | None
    name: str | None
    transport_type: int | None
    bind: bool


@dataclass(frozen=True)
class _BuiltEngine:
    """An AVAudioEngine graph that has started but is not yet committed."""

    engine: object
    observer: object | None
    route: _InputRoute
    device_id: int | None
    hardware_sample_rate: float
    resample_ratio: float


def list_input_devices() -> list[dict]:
    """Return a list of available audio input devices.

    Each dict has keys: ``uid`` (str) and ``name`` (str).
    UIDs are informational only; capture always follows the macOS default.
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

    WenZi always follows the current macOS default input. ``configured_uid``
    remains in the signature only so older callers and configuration can be
    ignored without changing the system route or binding AVAudioEngine.
    """
    del configured_uid

    try:
        default_device = AVCaptureDevice.defaultDeviceWithMediaType_(
            AVMediaTypeAudio
        )
    except Exception:
        logger.warning("Failed to query the default input device", exc_info=True)
        return _InputRoute(None, None, None, False)
    if default_device is None:
        return _InputRoute(None, None, None, False)

    return _capture_device_route(default_device, bind=False)


def automatic_input_device_name() -> str | None:
    """Return the current macOS default input device name, if available."""
    try:
        return _select_input_route(None).name
    except Exception:
        logger.warning("Failed to resolve the automatic input device", exc_info=True)
        return None


_COREAUDIO_BINDINGS = None


def _coreaudio_bindings():
    """Load CoreAudio/CoreFoundation via ctypes once per process.

    ``find_library`` walks the dyld search paths — too slow to repeat on
    every recording start.  Assignment is idempotent, so a benign race
    between two first callers needs no lock.
    """
    global _COREAUDIO_BINDINGS
    if _COREAUDIO_BINDINGS is None:
        import ctypes
        import ctypes.util

        # AudioObjectPropertyAddress
        class _Addr(ctypes.Structure):
            _fields_ = [
                ("mSelector", ctypes.c_uint32),
                ("mScope", ctypes.c_uint32),
                ("mElement", ctypes.c_uint32),
            ]

        ca = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio"))
        ca.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyDataSize.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_Addr),
            ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
        ]
        ca.AudioObjectGetPropertyData.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_Addr),
            ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]

        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        cf.CFStringGetLength.restype = ctypes.c_long
        cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32,
        ]
        cf.CFRelease.restype = None
        cf.CFRelease.argtypes = [ctypes.c_void_p]

        _COREAUDIO_BINDINGS = (ca, cf, _Addr)
    return _COREAUDIO_BINDINGS


def _resolve_device_id(uid: str) -> int | None:
    """Find the CoreAudio AudioDeviceID for an AVCaptureDevice UID.

    AVAudioEngine's input node uses CoreAudio device IDs internally.
    We bridge from AVCaptureDevice UID → AudioDeviceID via a CoreAudio
    property lookup over all audio objects.
    """
    import ctypes

    _ca, _cf, _Addr = _coreaudio_bindings()

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

    __slots__ = (
        "queue",
        "total_bytes",
        "rms",
        "on_chunk",
        "armed",
        "configured_device",
        "is_bluetooth",
        "engine_epoch",
        "saw_nonzero",
        "exact_zero_frames",
        "recovery_pending",
        "recovery_count",
        "last_recovery_at",
        "capture_failed",
        "recovery_done",
        "on_route_recovered",
    )

    def __init__(
        self,
        armed: bool = True,
        *,
        configured_device: str | None = None,
        is_bluetooth: bool = False,
        on_route_recovered: Callable[[], bool] | None = None,
    ) -> None:
        self.queue: queue.Queue[bytes] = queue.Queue()
        self.total_bytes = 0
        self.rms = 0.0
        self.on_chunk = None
        # While False the tap discards every frame (see Recorder.arm()):
        # the engine can spin up during the start-sound guard window
        # without the sound leaking into the recording.
        self.armed = armed
        self.configured_device = configured_device
        self.is_bluetooth = is_bluetooth
        # Recovery reuses this session and generation.  An engine epoch
        # keeps callbacks from the replaced engine from writing after the
        # replacement commits.
        self.engine_epoch = 1
        self.saw_nonzero = False
        self.exact_zero_frames = 0
        self.recovery_pending = False
        self.recovery_count = 0
        self.last_recovery_at = 0.0
        self.capture_failed = False
        self.recovery_done = threading.Event()
        self.recovery_done.set()
        self.on_route_recovered = on_route_recovered


class Recorder:
    """Record audio from the microphone using AVAudioEngine. Thread-safe start/stop."""

    # RMS threshold for silence detection (int16 range: 0-32768).
    # Typical quiet room noise is ~100-300, speech is ~1000+.
    DEFAULT_SILENCE_RMS = 20
    # Reference RMS for normalizing current_level to 0.0-1.0 range.
    # Normal speech (~1000-3000 RMS) maps to roughly 0.4-1.0.  The
    # reference must sit ABOVE typical speech peaks: a lower value
    # clips speech flat at 1.0, erasing the syllable modulation the
    # recording indicator's wave detector runs on.
    _LEVEL_REFERENCE_RMS = 2400.0
    # Max seconds _starting may remain True before it is considered stuck
    # and forcibly reset, allowing a new start() to proceed.
    _STARTING_STALE_SECS = 10.0
    # A live Bluetooth microphone has a small noise floor even when the
    # user is silent.  Sustained bit-exact zeros instead indicate the
    # observed AirPods HFP failure where AVAudioEngine keeps
    # delivering buffers after CoreAudio has stopped (or failed to start)
    # the input stream.  Startup gets a longer grace period.
    _BLUETOOTH_ZERO_RECOVERY_SECS = 1.0
    _BLUETOOTH_STARTUP_ZERO_RECOVERY_SECS = 1.5
    _BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS = 2.0
    _BLUETOOTH_RECOVERY_MAX_PER_SESSION = 2
    _BLUETOOTH_RECOVERY_BUILD_ATTEMPTS = 3
    _BLUETOOTH_RECOVERY_RETRY_DELAY_SECS = 0.15

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
        # Explicit input UIDs are legacy configuration. AVAudioEngine must
        # follow the current macOS default without rebinding CoreAudio.
        self._device = None
        if device:
            logger.info(
                "Ignoring legacy input device UID; using macOS default"
            )
        self.max_session_bytes = max_session_bytes
        self.silence_rms = silence_rms
        self._route_cache = None
        # Sound feedback and the next start share one non-activating route
        # observation, avoiding two independent default-device queries.
        self._preflight_route: _InputRoute | None = None
        self._preflight_chime_allowed = False

        self._queue: queue.Queue[bytes] = queue.Queue()
        self._engine: AVAudioEngine | None = None
        self._hw_sample_rate: float = 0.0
        self._resample_ratio: float = 0.0
        self._lock = threading.Lock()
        # Cleared while a recovery owns an uncommitted native engine.  A new
        # start waits here so it cannot race that engine for the HFP route.
        self._recovery_done = threading.Event()
        self._recovery_done.set()
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
    def device(self) -> str | None:
        """Always return None because capture follows the system default."""
        return None

    @device.setter
    def device(self, value: str | None) -> None:
        if value:
            logger.info(
                "Ignoring explicit input device UID; using macOS default"
            )
        self._device = None
        self._invalidate_route_cache()

    def preflight_input_route(self) -> _InputRoute:
        """Inspect and retain the default route used by the next start."""

        route = _select_input_route(None)
        with self._lock:
            if not self._recording and self._starting_since is None:
                self._preflight_route = route
                self._preflight_chime_allowed = False
        return route

    def mark_preflight_chime_allowed(self) -> None:
        """Mark that audible feedback was approved for the cached route."""

        with self._lock:
            if self._preflight_route is not None:
                self._preflight_chime_allowed = True

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

        Uses ``_LEVEL_REFERENCE_RMS`` (2400) as reference so normal
        speech (~1000-3000 RMS) maps to roughly 0.4-1.0 without
        clipping its syllable modulation flat.
        """
        session = self._session
        if session is None:
            return 0.0
        return min(1.0, session.rms / self._LEVEL_REFERENCE_RMS)

    def start(
        self,
        *,
        armed: bool = True,
        on_route_recovered: Callable[[], bool] | None = None,
    ) -> str | None:
        """Start recording. Returns the input device name, or None.

        With ``armed=False`` the engine runs but the tap discards every
        frame until :meth:`arm` is called — used to warm the microphone
        up during the start-sound guard window without capturing the
        sound itself.

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

        if not self._wait_for_recovery(gen):
            return None

        # --- Phase 2: create AVAudioEngine and audio graph (lock free) --
        built = None
        try:
            with self._lock:
                preflight_route = self._preflight_route
                chime_allowed = self._preflight_chime_allowed
                self._preflight_route = None
                self._preflight_chime_allowed = False
            route = _select_input_route(None)
            if (
                preflight_route is not None
                and chime_allowed
                and _input_route_has_bluetooth_risk(route)
            ):
                raise RuntimeError(
                    "Default input became Bluetooth or unknown after "
                    "the start sound was approved"
                )

            # Each start() gets its own session state, captured by the
            # tap closure: audio, rms, byte counts and the chunk callback
            # of a zombie engine can only ever land in its own session —
            # never a newer one's.
            session = _TapSession(
                armed=armed,
                configured_device=None,
                is_bluetooth=_input_route_has_bluetooth_risk(route),
                on_route_recovered=on_route_recovered,
            )
            built = self._build_engine(
                route,
                None,
                gen,
                session,
                engine_epoch=1,
            )
            device_name = route.name if self._query_device_name_enabled else None

            # Phase 3: commit (lock held briefly)
            with self._lock:
                if gen == self._start_gen and gen > self._abandoned_gen:
                    self._engine = built.engine
                    self._session = session
                    self._queue = session.queue
                    self._recording = True
                    self._active_gen = gen
                    self._hw_sample_rate = built.hardware_sample_rate
                    self._resample_ratio = built.resample_ratio
                    self._starting_since = None
                    self._last_device_name = device_name
                    self._config_observer = built.observer
                    logger.info(
                        "Recording started (sr=%d, hw=%.0f Hz, device=%s)",
                        self.sample_rate,
                        built.hardware_sample_rate,
                        device_name or "unknown",
                    )
                    return device_name
                if gen == self._start_gen:
                    self._starting_since = None
        except Exception:
            logger.error("Failed to create audio engine", exc_info=True)
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
        self._invalidate_route_cache()
        if built is not None:
            self._teardown_engine(built.engine, built.observer)
        return None

    def _wait_for_recovery(self, gen: int) -> bool:
        """Keep a new start behind an older session's recovery engine."""
        while not self._recovery_done.wait(timeout=0.05):
            with self._lock:
                if gen != self._start_gen or gen <= self._abandoned_gen:
                    if gen == self._start_gen:
                        self._starting_since = None
                    return False
        with self._lock:
            if gen != self._start_gen or gen <= self._abandoned_gen:
                if gen == self._start_gen:
                    self._starting_since = None
                return False
        return True

    def arm(self) -> None:
        """Open the sound-feedback gate on the committed session.

        Contract: only call after start() has returned with
        ``is_recording`` True — the committed session is then the one
        the caller warmed up.  Arming earlier would silently record
        nothing (there is no session yet to arm).
        """
        session = self._session
        if session is not None:
            session.armed = True

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
            if session is not None and session.recovery_pending:
                # The replacement has not committed, so audio between the
                # detected stall and this stop is unknowable.  Never return a
                # plausible-looking prefix as if it represented the full
                # held-key session.
                session.capture_failed = True
            # Break the transcriber reference on OUR session only — a
            # stop racing a new session can never clear the new session's
            # callback, because that one lives on a different object.
            if session is not None:
                session.on_chunk = None

        # Stop engine, remove tap, detach notification observer
        self._teardown_engine(engine, observer)

        if session is None:
            return None

        # A recovery may own an engine that has started but not committed,
        # so self._engine can already be None while the microphone is still
        # active.  Preserve stop()'s contract: return only after every native
        # engine belonging to this session has been torn down.
        session.recovery_done.wait()

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

        if session.capture_failed:
            logger.error(
                "Discarding recording because Bluetooth input recovery failed"
            )
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

    def _tap_callback(
        self,
        buffer,
        gen: int,
        ratio: float,
        session,
        *,
        engine_epoch: int | None = None,
    ) -> None:
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
            if (
                engine_epoch is not None
                and engine_epoch != session.engine_epoch
            ):
                return
            if not session.armed:
                # Sound-guard window: these frames contain the start
                # sound and must not reach rms, the buffers or on_chunk.
                # Plain attribute read — GIL-atomic against arm()'s
                # write; one boundary frame either way is fine.
                return

            in_frames = buffer.frameLength()
            if in_frames == 0:
                return

            # Read float32 samples from the native-rate input buffer
            channel0 = buffer.floatChannelData()[0]
            raw = bytes(channel0.as_buffer(in_frames))
            self._track_bluetooth_zero_audio(
                raw,
                in_frames,
                ratio,
                gen,
                session,
            )
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

    def _track_bluetooth_zero_audio(
        self,
        raw: bytes,
        in_frames: int,
        ratio: float,
        gen: int,
        session: _TapSession,
    ) -> None:
        """Schedule recovery for a live Bluetooth stream stuck at zeros.

        This method runs on the audio tap thread.  It only updates small
        counters and starts at most one helper thread; AVFoundation work is
        always performed by that helper.
        """
        if not session.is_bluetooth:
            return

        # bytes.count() stays in C and avoids a Python loop on every audio
        # block.  A real Bluetooth microphone has a non-zero noise floor;
        # bit-exact zero is deliberately stricter than ordinary silence.
        if raw.count(0) != len(raw):
            session.saw_nonzero = True
            session.exact_zero_frames = 0
            return
        session.exact_zero_frames += in_frames
        hardware_rate = ratio * self.sample_rate
        zero_seconds = (
            self._BLUETOOTH_ZERO_RECOVERY_SECS
            if session.saw_nonzero
            else self._BLUETOOTH_STARTUP_ZERO_RECOVERY_SECS
        )
        threshold = zero_seconds * hardware_rate
        if session.exact_zero_frames < threshold:
            return

        # Reset before scheduling so callbacks arriving while the helper is
        # being launched do not repeatedly cross the threshold.
        session.exact_zero_frames = 0
        self._schedule_bluetooth_recovery(gen, session)

    def _schedule_bluetooth_recovery(
        self,
        gen: int,
        session: _TapSession,
    ) -> None:
        """Claim and dispatch one bounded recovery attempt."""
        now = time.monotonic()
        with self._lock:
            if (
                not self._recording
                or self._active_gen != gen
                or self._session is not session
                or session.recovery_pending
                or not self._recovery_done.is_set()
            ):
                return
            if (
                session.recovery_count
                >= self._BLUETOOTH_RECOVERY_MAX_PER_SESSION
            ):
                session.capture_failed = True
                return
            if (
                now - session.last_recovery_at
                < self._BLUETOOTH_RECOVERY_MIN_INTERVAL_SECS
            ):
                return
            session.recovery_pending = True
            session.recovery_count += 1
            session.last_recovery_at = now
            session.recovery_done.clear()
            self._recovery_done.clear()

        try:
            worker = threading.Thread(
                target=self._run_bluetooth_recovery,
                args=(gen, session),
                name="recorder-bluetooth-recovery",
                daemon=True,
            )
            worker.start()
        except Exception:
            with self._lock:
                if self._session is session and self._active_gen == gen:
                    session.recovery_pending = False
                    session.recovery_count -= 1
                    session.last_recovery_at = 0.0
            self._recovery_done.set()
            session.recovery_done.set()
            logger.error(
                "Failed to start Bluetooth input recovery worker",
                exc_info=True,
            )

    def _run_bluetooth_recovery(
        self,
        gen: int,
        session: _TapSession,
    ) -> None:
        try:
            self._recover_bluetooth_input(gen, session)
        finally:
            session.recovery_done.set()
            self._recovery_done.set()

    def _recover_bluetooth_input(
        self,
        gen: int,
        session: _TapSession,
    ) -> None:
        """Replace a stalled AVAudioEngine while retaining session audio."""
        with self._lock:
            if (
                not self._recording
                or self._active_gen != gen
                or self._session is not session
                or not session.recovery_pending
            ):
                return
            old_engine = self._engine
            old_observer = self._config_observer
            self._engine = None
            self._config_observer = None
            next_epoch = session.engine_epoch + 1

        logger.warning(
            "Bluetooth input produced sustained exact-zero audio; "
            "restarting capture engine (recovery %d/%d)",
            session.recovery_count,
            self._BLUETOOTH_RECOVERY_MAX_PER_SESSION,
        )
        replacement = None
        self._teardown_engine(old_engine, old_observer)
        try:
            for attempt in range(
                1,
                self._BLUETOOTH_RECOVERY_BUILD_ATTEMPTS + 1,
            ):
                if not self._recovery_is_live(gen, session):
                    return
                try:
                    route = _select_input_route(None)
                    replacement = self._build_engine(
                        route,
                        None,
                        gen,
                        session,
                        engine_epoch=next_epoch,
                    )
                    break
                except Exception:
                    logger.warning(
                        "Bluetooth input recovery build failed "
                        "(attempt %d/%d)",
                        attempt,
                        self._BLUETOOTH_RECOVERY_BUILD_ATTEMPTS,
                        exc_info=True,
                    )
                    if attempt < self._BLUETOOTH_RECOVERY_BUILD_ATTEMPTS:
                        time.sleep(
                            self._BLUETOOTH_RECOVERY_RETRY_DELAY_SECS
                        )

            if replacement is None:
                logger.error(
                    "Bluetooth input recovery exhausted all build attempts"
                )
                with self._lock:
                    if self._session is session and self._active_gen == gen:
                        session.capture_failed = True
                return

            committed = False
            route_recovered = None
            replacement_name = "unknown"
            replacement_rate = 0.0
            with self._lock:
                if (
                    self._recording
                    and self._active_gen == gen
                    and self._session is session
                    and session.recovery_pending
                    and self._engine is None
                ):
                    self._engine = replacement.engine
                    self._config_observer = replacement.observer
                    self._hw_sample_rate = replacement.hardware_sample_rate
                    self._resample_ratio = replacement.resample_ratio
                    session.engine_epoch = next_epoch
                    replacement_transport = (
                        replacement.route.transport_type
                    )
                    if replacement_transport is not None:
                        session.is_bluetooth = (
                            replacement_transport in _BLUETOOTH_TRANSPORTS
                        )
                    # This session already proved that it had live input.
                    # Keep the detector armed so an all-zero replacement
                    # consumes the next bounded recovery rather than being
                    # mistaken for normal startup silence.
                    session.saw_nonzero = True
                    session.exact_zero_frames = 0
                    # The replacement is now committed. Output-route refresh
                    # is a separate fence: stop() still waits for the worker,
                    # but must not treat this live engine as an unknown gap.
                    session.recovery_pending = False
                    route_recovered = session.on_route_recovered
                    replacement_name = replacement.route.name or "unknown"
                    replacement_rate = replacement.hardware_sample_rate
                    replacement = None
                    committed = True

            if committed:
                refresh_succeeded = True
                if route_recovered is not None:
                    try:
                        refresh_succeeded = route_recovered() is not False
                    except Exception:
                        refresh_succeeded = False
                        logger.exception(
                            "Failed to refresh output duck after input recovery"
                        )
                if not refresh_succeeded:
                    # The route monitor retains a second chance. Do not discard
                    # already captured speech merely because CoreAudio had not
                    # published the replacement output route yet.
                    logger.warning(
                        "Output duck was not confirmed after input recovery"
                    )
                logger.info(
                    "Bluetooth input recovery committed "
                    "(device=%s, hw=%.0f Hz)",
                    replacement_name,
                    replacement_rate,
                )
                return
        finally:
            if replacement is not None:
                self._teardown_engine(
                    replacement.engine,
                    replacement.observer,
                )
            with self._lock:
                if self._session is session and self._active_gen == gen:
                    session.recovery_pending = False

    def _recovery_is_live(
        self,
        gen: int,
        session: _TapSession,
    ) -> bool:
        with self._lock:
            return (
                self._recording
                and self._active_gen == gen
                and self._session is session
                and session.recovery_pending
                and self._engine is None
            )

    def _build_engine(
        self,
        route: _InputRoute,
        cached_device_id: int | None,
        gen: int,
        session: _TapSession,
        *,
        engine_epoch: int,
    ) -> _BuiltEngine:
        """Create, start and observe an uncommitted capture engine."""
        engine = None
        observer = None
        try:
            engine = AVAudioEngine.alloc().init()
            input_node = engine.inputNode()

            # Never call setDeviceID_error_. Leaving the input node untouched
            # makes AVAudioEngine consume the current macOS default route.
            dev_id = None

            logger.info(
                "Input route configured=%s effective_uid=%s name=%s "
                "transport=%s coreaudio_id=%s",
                session.configured_device or "automatic",
                route.uid or "system-default",
                route.name or "unknown",
                route.transport_type
                if route.transport_type is not None
                else "unknown",
                dev_id if dev_id is not None else "system-default",
            )

            hw_fmt = input_node.outputFormatForBus_(0)
            hw_sample_rate = hw_fmt.sampleRate()
            ratio = hw_sample_rate / self.sample_rate

            def tap_block(buf, when):
                self._tap_callback(
                    buf,
                    gen,
                    ratio,
                    session,
                    engine_epoch=engine_epoch,
                )

            input_node.installTapOnBus_bufferSize_format_block_(
                0,
                int(hw_sample_rate * self.block_ms / 1000),
                hw_fmt,
                tap_block,
            )
            engine.prepare()
            ok, err = engine.startAndReturnError_(None)
            if not ok:
                raise RuntimeError(f"AVAudioEngine start failed: {err}")

            observer = (
                NSNotificationCenter.defaultCenter()
                .addObserverForName_object_queue_usingBlock_(
                    _ENGINE_CONFIG_CHANGE,
                    engine,
                    None,
                    lambda note: self._on_config_change(
                        gen,
                        session,
                        engine_epoch,
                    ),
                )
            )
            return _BuiltEngine(
                engine=engine,
                observer=observer,
                route=route,
                device_id=dev_id,
                hardware_sample_rate=hw_sample_rate,
                resample_ratio=ratio,
            )
        except Exception:
            self._invalidate_route_cache()
            self._teardown_engine(engine, observer)
            raise

    def mark_tainted(self, *, stop_async: bool = True) -> None:
        """Abandon an in-flight start() whose caller gave up waiting.

        An abandoned start() tears its engine down when it eventually
        finishes instead of committing — otherwise the microphone would
        stay open with no recording session owning it.  If start()
        committed just before the taint arrived, stop the engine on a
        helper thread by default. A caller that owns the executor future can
        pass ``stop_async=False`` and synchronously settle ``stop()`` only
        after that future returns, forming one route-teardown barrier.
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
        if not stop_async:
            logger.warning(
                "start() committed before taint; caller owns engine teardown"
            )
            return
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

    def _invalidate_route_cache(self) -> None:
        self._route_cache = None
        self._preflight_route = None
        self._preflight_chime_allowed = False

    def _on_config_change(
        self,
        gen: int | None,
        observed_session: _TapSession | None,
        engine_epoch: int | None,
    ) -> None:
        """Reject a route change only when it belongs to the live engine."""
        logger.info("Audio engine configuration changed")
        self._invalidate_route_cache()
        with self._lock:
            session = self._session
            recording = (
                self._recording
                and self._engine is not None
                and gen == self._active_gen
                and observed_session is session
                and session is not None
                and engine_epoch == session.engine_epoch
            )
            if recording:
                # The untouched AVAudioEngine follows the system default, but
                # a mid-session route switch makes the captured interval and
                # Bluetooth recovery classification ambiguous. Never return a
                # plausible prefix as if it represented the held-key session.
                session.capture_failed = True
        if recording:
            logger.warning(
                "Audio configuration changed during recording; "
                "discarding the current session"
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
