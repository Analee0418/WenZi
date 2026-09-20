"""Temporarily lower the macOS default output volume during recording."""

from __future__ import annotations

import ctypes
import ctypes.util
import fcntl
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


def _fourcc(value: str) -> int:
    return int.from_bytes(value.encode("ascii"), "big")


_SYSTEM_OBJECT = 1
_DEFAULT_OUTPUT = _fourcc("dOut")
_DEVICE_UID = _fourcc("uid ")
_TRANSLATE_UID_TO_DEVICE = _fourcc("uidd")
_VOLUME_SCALAR = _fourcc("volm")
_VIRTUAL_MAIN_VOLUME = _fourcc("vmvc")
_MUTE = _fourcc("mute")
_TRANSPORT_TYPE = _fourcc("tran")
_PREFERRED_STEREO_CHANNELS = _fourcc("dch2")
_NOMINAL_SAMPLE_RATE = _fourcc("nsrt")
_STREAM_CONFIGURATION = _fourcc("slay")
_SCOPE_GLOBAL = _fourcc("glob")
_SCOPE_OUTPUT = _fourcc("outp")
_MAIN_ELEMENT = 0
# Keep the virtual-main control distinct from raw HAL element 0. Recovery
# journals written before virtual-main support used 0 to mean ``volm/outp/0``.
_VIRTUAL_MAIN_ELEMENT = -1

_LOWER_DURATION = 0.06
# A few extra HAL writes make the short gain ramp less audible while keeping
# the work negligible compared with microphone startup and transcription.
_LOWER_STEPS = 8
_RESTORE_DURATION = 0.12
_RESTORE_STEPS = 10
_RESTORE_RETRY_DELAY = 0.05
_RESTORE_ROUTE_ATTEMPTS = 40
# AirPods expose roughly 1/127 scalar steps. This covers one half-step of
# quantization without mistaking a deliberate user volume nudge for our cap.
_OWNERSHIP_TOLERANCE = 0.005
_ROUTE_SETTLE_DURATION = 0.10
# AirPods can publish the restored A2DP route after the visible HAL scalar has
# already returned to its original value. Repeat that same value after the
# media route has had two chances to settle; the first restore remains fast.
_POST_RESTORE_SYNC_DELAYS = (0.25, 0.25)
# A real scalar transition is required because Bluetooth drivers may coalesce
# a write equal to the visible vmvc value while their remote media gain is stale.
_POST_RESTORE_TICKLE_STEP = 1.0 / 127.0
# Keep the non-equal value observable long enough to cross the Bluetooth
# driver's command-coalescing window. This runs while output mute is owned.
_POST_RESTORE_TICKLE_DWELL = 0.02
_DEFERRED_SYNC_INTERVAL = 0.10
_DEFERRED_SYNC_FAST_DURATION = 8.0
_DEFERRED_SYNC_SLOW_INTERVAL = 5.0
_DEFERRED_SYNC_JOIN_TIMEOUT = 1.0
_LIFECYCLE_LOCK_TIMEOUT = 0.10
_MONITOR_FAST_DURATION = 2.0
_MONITOR_FAST_INTERVAL = 0.02
_MONITOR_SLOW_INTERVAL = 0.25
_MONITOR_JOIN_TIMEOUT = 1.0
_RECOVERY_VERSION = 4
_COMPATIBLE_RECOVERY_VERSIONS = {2, 3, _RECOVERY_VERSION}
_MUTE_RECOVERY_VERSIONS = {3, _RECOVERY_VERSION}
_RECOVERY_RAMP_GRACE = 1.0
_MUTE_PRE_VOLUME = 0.05
_CF_STRING_ENCODING_UTF8 = 0x08000100
_BLUETOOTH_TRANSPORT_TYPES = {
    _fourcc("blue"),
    _fourcc("blea"),
}
_MIN_ROUTE_SAMPLE_RATE = 8_000
_MAX_ROUTE_SAMPLE_RATE = 768_000
_MIN_MEDIA_SAMPLE_RATE = 32_000

_PHASE_DUCKING = "ducking"
_PHASE_DUCKED = "ducked"
_PHASE_RESTORING = "restoring"
_RECOVERY_PHASES = {
    _PHASE_DUCKING,
    _PHASE_DUCKED,
    _PHASE_RESTORING,
}

_LEGACY_MIGRATION_NOT_APPLICABLE = 0
_LEGACY_MIGRATION_DEFERRED = 1
_LEGACY_MIGRATION_MIGRATED = 2
_LEGACY_MIGRATION_ABANDONED = 3


def _valid_route_signature(signature: tuple[int, int]) -> bool:
    sample_rate, output_channels = signature
    return (
        _MIN_ROUTE_SAMPLE_RATE <= sample_rate <= _MAX_ROUTE_SAMPLE_RATE
        and output_channels > 0
    )


def _is_media_route_signature(signature: tuple[int, int]) -> bool:
    return (
        _valid_route_signature(signature)
        and signature[0] >= _MIN_MEDIA_SAMPLE_RATE
        and signature[1] >= 2
    )


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


class _AudioBuffer(ctypes.Structure):
    _fields_ = [
        ("mNumberChannels", ctypes.c_uint32),
        ("mDataByteSize", ctypes.c_uint32),
        ("mData", ctypes.c_void_p),
    ]


class _AudioBufferList(ctypes.Structure):
    _fields_ = [
        ("mNumberBuffers", ctypes.c_uint32),
        ("mBuffers", _AudioBuffer * 1),
    ]


class _CoreAudioError(RuntimeError):
    def __init__(self, operation: str, status: int) -> None:
        super().__init__(f"{operation} failed with OSStatus {status}")
        self.status = status


class _DeferredRestoreAborted(RuntimeError):
    """A new recording took ownership while deferred restoration was running."""


class _DeferredRestoreMuteLost(RuntimeError):
    """The user unmuted an output during a muted deferred restoration."""


class SystemVolumeBusyError(RuntimeError):
    """A previous CoreAudio restore did not reach a cancellation point."""


class _VolumeBackend(Protocol):
    def default_output_device(self) -> int | None: ...

    def device_uid(self, device_id: int) -> str | None: ...

    def device_id_for_uid(self, device_uid: str) -> int | None: ...

    def transport_type(self, device_id: int) -> int | None: ...

    def volume_elements(self, device_id: int) -> tuple[int, ...]: ...

    def volume_profile(self, device_id: int) -> tuple[int, ...]: ...

    def output_route_signature(
        self,
        device_id: int,
    ) -> tuple[int, int] | None: ...

    def get_volume(self, device_id: int, element: int) -> float: ...

    def set_volume(self, device_id: int, element: int, value: float) -> None: ...

    def get_mute(self, device_id: int) -> bool | None: ...

    def set_mute(self, device_id: int, muted: bool) -> None: ...


class _CoreAudioBackend:
    """Small ctypes wrapper around the public CoreAudio HAL volume API."""

    def __init__(self) -> None:
        self._ca = None
        self._cf = None

    def _library(self):
        if self._ca is not None:
            return self._ca

        path = ctypes.util.find_library("CoreAudio")
        if not path:
            raise RuntimeError("CoreAudio framework was not found")
        ca = ctypes.cdll.LoadLibrary(path)
        ca.AudioObjectHasProperty.restype = ctypes.c_ubyte
        ca.AudioObjectHasProperty.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_PropertyAddress),
        ]
        ca.AudioObjectIsPropertySettable.restype = ctypes.c_int32
        ca.AudioObjectIsPropertySettable.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_PropertyAddress),
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        ca.AudioObjectGetPropertyData.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_PropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        ca.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyDataSize.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_PropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        ca.AudioObjectSetPropertyData.restype = ctypes.c_int32
        ca.AudioObjectSetPropertyData.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_PropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._ca = ca
        return ca

    def _core_foundation(self):
        if self._cf is not None:
            return self._cf

        path = ctypes.util.find_library("CoreFoundation")
        if not path:
            raise RuntimeError("CoreFoundation framework was not found")
        cf = ctypes.cdll.LoadLibrary(path)
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        cf.CFStringGetLength.restype = ctypes.c_long
        cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
        cf.CFStringGetMaximumSizeForEncoding.restype = ctypes.c_long
        cf.CFStringGetMaximumSizeForEncoding.argtypes = [
            ctypes.c_long,
            ctypes.c_uint32,
        ]
        cf.CFStringGetCString.restype = ctypes.c_ubyte
        cf.CFStringGetCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_uint32,
        ]
        cf.CFRelease.restype = None
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        self._cf = cf
        return cf

    @staticmethod
    def _address(selector: int, scope: int, element: int) -> _PropertyAddress:
        return _PropertyAddress(selector, scope, element)

    def _get_value(self, object_id: int, address: _PropertyAddress, value_type):
        ca = self._library()
        value = value_type()
        size = ctypes.c_uint32(ctypes.sizeof(value))
        status = ca.AudioObjectGetPropertyData(
            object_id,
            ctypes.byref(address),
            0,
            None,
            ctypes.byref(size),
            ctypes.byref(value),
        )
        if status != 0:
            raise _CoreAudioError("AudioObjectGetPropertyData", status)
        return value

    def _is_writable(self, object_id: int, address: _PropertyAddress) -> bool:
        ca = self._library()
        if not ca.AudioObjectHasProperty(object_id, ctypes.byref(address)):
            return False
        writable = ctypes.c_ubyte(0)
        status = ca.AudioObjectIsPropertySettable(
            object_id,
            ctypes.byref(address),
            ctypes.byref(writable),
        )
        return status == 0 and bool(writable.value)

    def default_output_device(self) -> int | None:
        address = self._address(
            _DEFAULT_OUTPUT,
            _SCOPE_GLOBAL,
            _MAIN_ELEMENT,
        )
        device_id = self._get_value(
            _SYSTEM_OBJECT,
            address,
            ctypes.c_uint32,
        ).value
        return int(device_id) if device_id else None

    def device_uid(self, device_id: int) -> str | None:
        # Load CFRelease before acquiring the caller-owned UID object so every
        # successful HAL return is covered by the finally block below.
        cf = self._core_foundation()
        address = self._address(
            _DEVICE_UID,
            _SCOPE_GLOBAL,
            _MAIN_ELEMENT,
        )
        uid_ref = self._get_value(device_id, address, ctypes.c_void_p)
        if not uid_ref.value:
            return None
        try:
            length = cf.CFStringGetLength(uid_ref)
            capacity = cf.CFStringGetMaximumSizeForEncoding(
                length,
                _CF_STRING_ENCODING_UTF8,
            )
            if capacity < 0:
                raise RuntimeError("CoreAudio returned an invalid device UID")
            buffer = ctypes.create_string_buffer(capacity + 1)
            if not cf.CFStringGetCString(
                uid_ref,
                buffer,
                len(buffer),
                _CF_STRING_ENCODING_UTF8,
            ):
                raise RuntimeError("CoreAudio device UID is not valid UTF-8")
            result = buffer.value.decode("utf-8")
            return result or None
        finally:
            cf.CFRelease(uid_ref)

    def device_id_for_uid(self, device_uid: str) -> int | None:
        if not device_uid:
            return None
        cf = self._core_foundation()
        uid_ref = cf.CFStringCreateWithCString(
            None,
            device_uid.encode("utf-8"),
            _CF_STRING_ENCODING_UTF8,
        )
        if not uid_ref:
            raise RuntimeError("Failed to create CoreAudio device UID")
        try:
            address = self._address(
                _TRANSLATE_UID_TO_DEVICE,
                _SCOPE_GLOBAL,
                _MAIN_ELEMENT,
            )
            qualifier = ctypes.c_void_p(uid_ref)
            device_id = ctypes.c_uint32(0)
            size = ctypes.c_uint32(ctypes.sizeof(device_id))
            status = self._library().AudioObjectGetPropertyData(
                _SYSTEM_OBJECT,
                ctypes.byref(address),
                ctypes.sizeof(qualifier),
                ctypes.byref(qualifier),
                ctypes.byref(size),
                ctypes.byref(device_id),
            )
            if status != 0:
                raise _CoreAudioError("AudioObjectGetPropertyData", status)
            return int(device_id.value) if device_id.value else None
        finally:
            cf.CFRelease(uid_ref)

    def transport_type(self, device_id: int) -> int | None:
        address = self._address(
            _TRANSPORT_TYPE,
            _SCOPE_GLOBAL,
            _MAIN_ELEMENT,
        )
        ca = self._library()
        if not ca.AudioObjectHasProperty(device_id, ctypes.byref(address)):
            return None
        return int(
            self._get_value(
                device_id,
                address,
                ctypes.c_uint32,
            ).value
        )

    def volume_elements(self, device_id: int) -> tuple[int, ...]:
        virtual_main = self._address(
            _VIRTUAL_MAIN_VOLUME,
            _SCOPE_OUTPUT,
            _MAIN_ELEMENT,
        )
        if self._is_writable(device_id, virtual_main):
            return (_VIRTUAL_MAIN_ELEMENT,)

        return self.volume_profile(device_id)

    def volume_profile(self, device_id: int) -> tuple[int, ...]:
        """Return raw HAL controls that distinguish media and call profiles."""

        raw_main = self._address(
            _VOLUME_SCALAR,
            _SCOPE_OUTPUT,
            _MAIN_ELEMENT,
        )
        if self._is_writable(device_id, raw_main):
            return (_MAIN_ELEMENT,)

        candidates = self._preferred_stereo_channels(device_id)
        if not candidates:
            candidates = (1, 2)
        result = []
        for element in dict.fromkeys(candidates):
            if element == _MAIN_ELEMENT:
                continue
            address = self._address(
                _VOLUME_SCALAR,
                _SCOPE_OUTPUT,
                element,
            )
            if self._is_writable(device_id, address):
                result.append(element)
        return tuple(result)

    def output_route_signature(
        self,
        device_id: int,
    ) -> tuple[int, int] | None:
        """Return the public HAL media-shape fingerprint for an output route."""

        try:
            ca = self._library()
            rate_address = self._address(
                _NOMINAL_SAMPLE_RATE,
                _SCOPE_OUTPUT,
                _MAIN_ELEMENT,
            )
            layout_address = self._address(
                _STREAM_CONFIGURATION,
                _SCOPE_OUTPUT,
                _MAIN_ELEMENT,
            )
            if not ca.AudioObjectHasProperty(
                device_id,
                ctypes.byref(rate_address),
            ) or not ca.AudioObjectHasProperty(
                device_id,
                ctypes.byref(layout_address),
            ):
                return None

            sample_rate = float(
                self._get_value(
                    device_id,
                    rate_address,
                    ctypes.c_double,
                ).value
            )
            if not math.isfinite(sample_rate) or sample_rate <= 0.0:
                return None

            data_size = ctypes.c_uint32(0)
            status = ca.AudioObjectGetPropertyDataSize(
                device_id,
                ctypes.byref(layout_address),
                0,
                None,
                ctypes.byref(data_size),
            )
            buffer_offset = _AudioBufferList.mBuffers.offset
            if status != 0 or data_size.value < (
                buffer_offset + ctypes.sizeof(_AudioBuffer)
            ):
                return None

            raw_layout = ctypes.create_string_buffer(data_size.value)
            returned_size = ctypes.c_uint32(data_size.value)
            status = ca.AudioObjectGetPropertyData(
                device_id,
                ctypes.byref(layout_address),
                0,
                None,
                ctypes.byref(returned_size),
                raw_layout,
            )
            if status != 0 or returned_size.value > data_size.value:
                return None

            buffer_count = ctypes.cast(
                raw_layout,
                ctypes.POINTER(_AudioBufferList),
            ).contents.mNumberBuffers
            required_size = buffer_offset + (
                int(buffer_count) * ctypes.sizeof(_AudioBuffer)
            )
            if buffer_count == 0 or required_size > returned_size.value:
                return None

            first_buffer = ctypes.addressof(raw_layout) + buffer_offset
            output_channels = sum(
                _AudioBuffer.from_address(
                    first_buffer + (index * ctypes.sizeof(_AudioBuffer))
                ).mNumberChannels
                for index in range(buffer_count)
            )
            rounded_rate = int(round(sample_rate))
            signature = (rounded_rate, int(output_channels))
            return signature if _valid_route_signature(signature) else None
        except Exception:
            # Bluetooth devices can disappear between any two HAL reads. Route
            # fingerprinting is advisory, so callers retry instead of failing
            # the recording or restoration path.
            return None

    def _volume_address(
        self,
        device_id: int,
        element: int,
    ) -> _PropertyAddress:
        if element == _VIRTUAL_MAIN_ELEMENT:
            return self._address(
                _VIRTUAL_MAIN_VOLUME,
                _SCOPE_OUTPUT,
                _MAIN_ELEMENT,
            )
        return self._address(
            _VOLUME_SCALAR,
            _SCOPE_OUTPUT,
            element,
        )

    def _preferred_stereo_channels(self, device_id: int) -> tuple[int, ...]:
        address = self._address(
            _PREFERRED_STEREO_CHANNELS,
            _SCOPE_OUTPUT,
            _MAIN_ELEMENT,
        )
        ca = self._library()
        if not ca.AudioObjectHasProperty(device_id, ctypes.byref(address)):
            return ()
        channels_type = ctypes.c_uint32 * 2
        try:
            channels = self._get_value(device_id, address, channels_type)
        except _CoreAudioError:
            return ()
        return tuple(int(channel) for channel in channels)

    def get_volume(self, device_id: int, element: int) -> float:
        address = self._volume_address(device_id, element)
        value = float(self._get_value(device_id, address, ctypes.c_float).value)
        if not math.isfinite(value):
            raise RuntimeError("CoreAudio returned a non-finite volume")
        return max(0.0, min(1.0, value))

    def set_volume(self, device_id: int, element: int, value: float) -> None:
        address = self._volume_address(device_id, element)
        scalar = ctypes.c_float(max(0.0, min(1.0, value)))
        status = self._library().AudioObjectSetPropertyData(
            device_id,
            ctypes.byref(address),
            0,
            None,
            ctypes.sizeof(scalar),
            ctypes.byref(scalar),
        )
        if status != 0:
            raise _CoreAudioError("AudioObjectSetPropertyData", status)

    def get_mute(self, device_id: int) -> bool | None:
        address = self._address(
            _MUTE,
            _SCOPE_OUTPUT,
            _MAIN_ELEMENT,
        )
        if not self._is_writable(device_id, address):
            return None
        value = self._get_value(device_id, address, ctypes.c_uint32).value
        return bool(value)

    def set_mute(self, device_id: int, muted: bool) -> None:
        address = self._address(
            _MUTE,
            _SCOPE_OUTPUT,
            _MAIN_ELEMENT,
        )
        value = ctypes.c_uint32(int(muted))
        status = self._library().AudioObjectSetPropertyData(
            device_id,
            ctypes.byref(address),
            0,
            None,
            ctypes.sizeof(value),
            ctypes.byref(value),
        )
        if status != 0:
            raise _CoreAudioError("AudioObjectSetPropertyData", status)


class DuckToken:
    """Opaque identity token for one active ducking session."""

    __slots__ = ()


@dataclass
class _DeviceSnapshot:
    device_uid: str
    device_id_hint: int
    profile_elements: tuple[int, ...]
    original: dict[int, float]
    duck_target: dict[int, float] = field(default_factory=dict)
    owned_values: dict[int, tuple[float, ...]] | None = None
    legacy_owned_elements: set[int] = field(default_factory=set)
    original_mute: bool | None = None
    mute_target: bool | None = None
    mute_owned: bool = False
    phase: str = _PHASE_DUCKED
    transition_started_at: float | None = None
    # The pending flag, expected route profile, mute, and scalar values are
    # persisted. A later process can replay the refresh only after every
    # user-controlled value and the Bluetooth media profile still match.
    post_restore_sync: bool = False
    # Hard ducking was used, but the route could not yet be ruled out as
    # Bluetooth. Keep the obligation until its transport and media profile settle.
    post_restore_media_pending: bool = False
    post_restore_expected_mute: bool | None = None
    post_restore_profile: tuple[int, ...] | None = None
    post_restore_route_signature: tuple[int, int] | None = None
    post_restore_values: dict[int, float] = field(default_factory=dict)
    post_restore_ready: bool = False
    post_restore_pass: int = 0
    allow_inactive_controls: bool = False


_SnapshotKey = tuple[str, tuple[int, ...]]


class SystemOutputDucker:
    """Temporarily attenuate every default output route seen by a session.

    A short-lived monitor covers CoreAudio route changes while the session is
    active. Callers may also invoke :meth:`refresh` at known route boundaries.
    """

    def __init__(
        self,
        backend: _VolumeBackend | None = None,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        monitor_waiter: Callable[[threading.Event, float], bool] | None = None,
        recovery_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._backend = backend or _CoreAudioBackend()
        self._sleep = sleeper
        self._monitor_wait = monitor_waiter or (lambda stop, timeout: stop.wait(timeout))
        self._recovery_path = Path(recovery_path).expanduser() if recovery_path is not None else None
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._active_token: DuckToken | None = None
        self._snapshots: dict[_SnapshotKey, _DeviceSnapshot] = {}
        self._deferred_snapshots: dict[_SnapshotKey, _DeviceSnapshot] = {}
        self._overridden_devices: set[str] = set()
        self._factor = 0.25
        self._max_volume = 0.05
        self._closing = False
        self._settle_completed = False
        self._monitor_stop: threading.Event | None = None
        self._monitor_thread: threading.Thread | None = None
        self._deferred_sync_thread: threading.Thread | None = None
        self._deferred_sync_stop: threading.Event | None = None
        self._deferred_sync_pair: (
            tuple[threading.Thread, threading.Event] | None
        ) = None
        self._explicit_recovery_cancel: threading.Event | None = None
        self._deferred_restart_needed = threading.Event()
        self._begin_requested = threading.Event()
        self._explicit_recovery_requested = threading.Event()
        self._background_workers_stopped = threading.Event()
        self._recovery_created_at: float | None = None
        self._lease_fd: int | None = None
        self._initial_snapshot_key: _SnapshotKey | None = None
        self._legacy_recovery_blocked = False

    def begin(
        self,
        factor: float = 0.25,
        max_volume: float = 0.05,
    ) -> DuckToken | None:
        """Lower the current default output and return its session token."""
        factor = self._validate_ratio("factor", factor)
        max_volume = self._validate_ratio("max_volume", max_volume)

        self._begin_requested.set()
        try:
            token = self._begin_with_fence(factor, max_volume)
        finally:
            self._begin_requested.clear()
            if (
                self._deferred_restart_needed.is_set()
                and self._active_token is None
            ):
                # If the cancelled worker exited while the begin fence was
                # still raised, its finally block intentionally did not start
                # a successor. Re-check after lowering the fence; a still-live
                # old pair keeps the flag set and will perform the handoff.
                self._start_deferred_sync_if_needed()
        if token is None:
            # A transient missing default route must not permanently stop the
            # previous session's pending AirPods gain refresh. Start the new
            # worker only after lowering the begin fence so it cannot exit on
            # the request that created it.
            self._stop_deferred_sync()
            self._start_deferred_sync_if_needed()
        return token

    def _begin_with_fence(
        self,
        factor: float,
        max_volume: float,
    ) -> DuckToken | None:
        """Begin after publishing a lock-free fence to the old worker."""

        if self._active_token is not None:
            if self._closing:
                raise SystemVolumeBusyError(
                    "Previous system volume session is still restoring"
                )
            return None

        # A completed session may still be waiting for AirPods to publish its
        # A2DP profile. Stop that old writer before this session takes volume
        # ownership; otherwise it could restore full gain after we start the
        # microphone and leave this recording without a duck token.
        explicit_recovery = self._explicit_recovery_cancel
        worker = self._stop_deferred_sync()
        if not self._close_lock.acquire(timeout=_LIFECYCLE_LOCK_TIMEOUT):
            if (
                explicit_recovery is not None
                or worker is not None
            ):
                self._deferred_restart_needed.set()
            raise SystemVolumeBusyError(
                "Previous system volume recovery is still inside CoreAudio"
            )
        try:
            # Serialize with a worker or close operation that started between
            # the stop request above and acquiring the lifecycle lock.
            self._request_deferred_sync_stop()
            return self._begin_locked(factor, max_volume)
        finally:
            self._close_lock.release()

    def _begin_locked(
        self,
        factor: float,
        max_volume: float,
    ) -> DuckToken | None:
        """Begin while the close/deferred lifecycle is exclusively owned."""

        with self._lock:
            if self._background_workers_stopped.is_set():
                raise RuntimeError("System output ducker has been stopped")
            if self._active_token is not None:
                return None
            if not self._acquire_lease():
                logger.info("Skipping system volume duck because another process owns it")
                return None

            rearmed_key: _SnapshotKey | None = None
            try:
                self._factor = factor
                self._max_volume = max_volume
                self._load_deferred_recovery()
                if self._legacy_recovery_blocked:
                    logger.warning("System volume duck remains disabled until the legacy low-volume state is changed manually")
                    return None
                device_id = self._backend.default_output_device()
                if device_id is None:
                    return None
                device_uid = self._backend.device_uid(device_id)
                if not device_uid:
                    return None
                elements = self._backend.volume_elements(device_id)
                if not elements:
                    return None
                deferred_ok, snapshot = self._claim_deferred_route(
                    device_id,
                    device_uid,
                    elements,
                )
                if not deferred_ok:
                    return None
                adopted = snapshot is not None
                needs_duck = snapshot is not None
                if snapshot is None:
                    snapshot = self._capture_snapshot(
                        device_id,
                        device_uid,
                        elements,
                    )
                if snapshot is None:
                    return None
                if needs_duck and not snapshot.post_restore_ready:
                    self._prepare_hard_snapshot_for_rearm(
                        snapshot,
                        device_id,
                    )
                self._configure_post_restore_sync(
                    snapshot,
                    device_id,
                )

                token = DuckToken()
                self._active_token = token
                initial_key = self._snapshot_key(snapshot)
                self._snapshots = {initial_key: snapshot}
                self._initial_snapshot_key = initial_key
                if adopted:
                    self._deferred_snapshots.pop(initial_key, None)
                    if snapshot.post_restore_sync:
                        rearmed_key = initial_key
                self._overridden_devices.clear()
                self._settle_completed = False
                if self._recovery_created_at is None:
                    self._recovery_created_at = time.time()
                if needs_duck:
                    if snapshot.post_restore_ready:
                        self._rearm_post_restore_snapshot(snapshot)
                if not adopted or needs_duck:
                    self._duck_snapshot(
                        snapshot,
                        mute_first=(
                            needs_duck
                            or (
                                self._max_volume == 0.0
                                and snapshot.mute_target is not None
                            )
                        ),
                    )
                self._start_monitor(token)
                return token
            except Exception:
                logger.warning(
                    "Failed to lower the system output volume",
                    exc_info=True,
                )
                self._rollback_and_clear(
                    rearm_post_restore_key=rearmed_key,
                )
                # A token with pending snapshots lets this recording session
                # retry restoration in its normal shutdown path.
                return self._active_token
            finally:
                if self._active_token is None:
                    self._release_lease()

    def refresh(self, token: DuckToken) -> bool:
        """Re-apply ducking after an output route or profile change."""
        with self._lock:
            if token is not self._active_token or self._closing:
                return False
            try:
                # The caller invokes refresh at a known route boundary, so a
                # same-ID reset here is not treated as a volume-key override.
                return self._refresh_locked(
                    allow_reduck=True,
                    abandon_on_deviation=False,
                    force_mute=True,
                )
            except Exception:
                logger.warning(
                    "Failed to lower a changed output route",
                    exc_info=True,
                )
                return False

    def end(self, token: DuckToken) -> bool:
        """Restore volumes owned by *token*; stale tokens are harmless."""
        if not self._close_lock.acquire(timeout=_LIFECYCLE_LOCK_TIMEOUT):
            logger.warning("System volume restore is already in progress")
            return False
        try:
            with self._lock:
                if token is not self._active_token:
                    return False
            return self._close_and_restore(token)
        finally:
            self._close_lock.release()

    def restore_all(self) -> bool:
        """Best-effort process-shutdown fallback for an active session."""
        if not self._close_lock.acquire(timeout=_LIFECYCLE_LOCK_TIMEOUT):
            logger.warning("System volume shutdown restore timed out")
            return False
        try:
            with self._lock:
                token = self._active_token
                if token is None:
                    return False
            return self._close_and_restore(token)
        finally:
            self._close_lock.release()

    def stop_background_workers(self) -> None:
        """Stop route-monitor and deferred-recovery helper threads.

        This is a lifecycle fence, not a volume restore. Callers that may own
        live volume state must run ``restore_all()`` / ``recover_stale()``
        first; tests use it after assertions so native-worker lifetimes never
        leak into unrelated cases.
        """
        # Publish the terminal fence without taking ``_lock``: a native HAL
        # call may be stalled while a worker holds that lock. Every publisher
        # rechecks this Event after installing its thread references.
        self._background_workers_stopped.set()
        self._deferred_restart_needed.clear()
        monitor_stop = self._monitor_stop
        monitor_thread = self._monitor_thread
        if monitor_stop is not None:
            monitor_stop.set()
        deferred_thread = self._stop_deferred_sync()

        for name, thread, timeout in (
            (
                "route monitor",
                monitor_thread,
                _MONITOR_JOIN_TIMEOUT,
            ),
            (
                "deferred recovery",
                deferred_thread,
                _DEFERRED_SYNC_JOIN_TIMEOUT,
            ),
        ):
            if (
                thread is None
                or thread is threading.current_thread()
                or not thread.is_alive()
            ):
                continue
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "System volume %s worker did not stop promptly",
                    name,
                )

        if self._lock.acquire(timeout=_LIFECYCLE_LOCK_TIMEOUT):
            try:
                if (
                    self._monitor_thread is monitor_thread
                    and (
                        monitor_thread is None
                        or not monitor_thread.is_alive()
                    )
                ):
                    self._monitor_stop = None
                    self._monitor_thread = None
            finally:
                self._lock.release()

    def defer_failed_restore(self, token: DuckToken) -> bool:
        """Hand an exhausted foreground restore to the durable worker."""
        if not self._close_lock.acquire(timeout=_LIFECYCLE_LOCK_TIMEOUT):
            logger.warning("Could not defer the failed system volume restore")
            return False
        deferred = False
        try:
            with self._lock:
                if token is not self._active_token:
                    return False
                moved_snapshots = dict(self._snapshots)
                previous_deferred_snapshots = dict(self._deferred_snapshots)
                for key, snapshot in list(self._snapshots.items()):
                    self._snapshots.pop(key, None)
                    self._insert_snapshot(
                        self._deferred_snapshots,
                        snapshot,
                    )
                created_at = self._recovery_created_at or time.time()
                try:
                    self._write_recovery_snapshots(
                        self._deferred_snapshots,
                        created_at,
                    )
                except Exception:
                    # Keep the live token when the latest ownership candidates
                    # could not be made durable. A later shutdown fallback can
                    # still retry without guessing at hardware state.
                    for key, snapshot in list(
                        self._deferred_snapshots.items()
                    ):
                        self._deferred_snapshots.pop(key, None)
                        self._insert_snapshot(self._snapshots, snapshot)
                    logger.warning(
                        "Failed to persist deferred system volume restore",
                        exc_info=True,
                    )
                    return False

                previous_closing = self._closing
                previous_settle_completed = self._settle_completed
                previous_monitor_stop = self._monitor_stop
                previous_monitor_thread = self._monitor_thread
                previous_initial_key = self._initial_snapshot_key
                previous_created_at = self._recovery_created_at
                previous_overridden_devices = set(self._overridden_devices)
                self._active_token = None
                self._closing = False
                self._settle_completed = False
                self._monitor_stop = None
                self._monitor_thread = None
                self._overridden_devices.clear()
                self._recovery_created_at = None
                self._initial_snapshot_key = None
                worker_started, _terminal_thread = (
                    self._launch_deferred_sync_locked()
                )
                if not worker_started:
                    # Do not report a successful handoff unless a worker has
                    # actually accepted it. Keep the live token as the final
                    # recovery owner so RecordingFlow cannot release its guard
                    # while an app-owned hard mute has nobody to clear it.
                    self._deferred_snapshots = previous_deferred_snapshots
                    self._snapshots = moved_snapshots
                    self._active_token = token
                    self._closing = previous_closing
                    self._settle_completed = previous_settle_completed
                    self._monitor_stop = previous_monitor_stop
                    self._monitor_thread = previous_monitor_thread
                    self._initial_snapshot_key = previous_initial_key
                    self._recovery_created_at = previous_created_at
                    self._overridden_devices = previous_overridden_devices
                    return False
                self._release_lease()
                deferred = True
        finally:
            self._close_lock.release()
        return deferred

    def recover_stale(self, *, start_deferred: bool = True) -> bool:
        """Restore a safely identifiable volume left by an earlier crash."""
        # Keep explicit recovery deterministic when a previous call already
        # handed an unresolved route to the background worker.
        self._explicit_recovery_requested.set()
        self._stop_deferred_sync()
        if not self._close_lock.acquire(timeout=_LIFECYCLE_LOCK_TIMEOUT):
            self._explicit_recovery_requested.clear()
            if start_deferred:
                self._start_deferred_sync_if_needed()
            logger.warning("Deferred system volume recovery did not stop")
            return False
        cancel_event = threading.Event()
        self._explicit_recovery_cancel = cancel_event
        try:
            with self._lock:
                if self._active_token is not None:
                    return False
                if not self._acquire_lease():
                    return False
                try:
                    journal = self._read_recovery_journal()
                    if journal is None:
                        self._deferred_snapshots.clear()
                        return False
                    created_at, snapshots = journal
                    self._deferred_snapshots = snapshots
                    self._recovery_created_at = created_at

                    restored_any = False
                    for key, snapshot in list(snapshots.items()):
                        migration, migrated = (
                            self._migrate_legacy_raw_master_snapshot(
                                key,
                                snapshot,
                            )
                        )
                        if migration == _LEGACY_MIGRATION_DEFERRED:
                            continue
                        if migration == _LEGACY_MIGRATION_ABANDONED:
                            continue
                        if migration == _LEGACY_MIGRATION_MIGRATED:
                            assert migrated is not None
                            snapshot = migrated
                            key = self._snapshot_key(snapshot)
                        if snapshot.post_restore_ready:
                            (
                                completed,
                                _reachable,
                                _write_failed,
                                progressed,
                            ) = self._sync_post_restore_until_blocked(
                                snapshot,
                                abort_event=cancel_event,
                            )
                            self._deferred_snapshots.pop(key, None)
                            if not completed:
                                self._insert_snapshot(
                                    self._deferred_snapshots,
                                    snapshot,
                                )
                            restored_any = (
                                restored_any or completed or progressed
                            )
                            continue
                        restore_mute_last = False
                        if migration == _LEGACY_MIGRATION_MIGRATED:
                            restore_mute_last = (
                                self._arm_deferred_restore_mute(
                                    snapshot,
                                    cancel_event,
                                )
                            )
                        (
                            unresolved,
                            changed,
                            _reachable,
                            _metadata_dirty,
                        ) = self._restore_owned_snapshot(
                            snapshot,
                            ensure_original_write=(
                                snapshot.post_restore_sync
                                or snapshot.post_restore_media_pending
                            ),
                            restore_mute_last=restore_mute_last,
                            abort_event=cancel_event,
                        )
                        restored_any = restored_any or changed
                        self._deferred_snapshots.pop(key, None)
                        if (
                            unresolved is None
                            and (
                                snapshot.post_restore_sync
                                or snapshot.post_restore_media_pending
                            )
                            and snapshot.post_restore_values
                        ):
                            self._retain_snapshot_elements(
                                snapshot,
                                list(snapshot.post_restore_values),
                            )
                            snapshot.post_restore_ready = True
                            (
                                completed,
                                _reachable,
                                _write_failed,
                                progressed,
                            ) = self._sync_post_restore_until_blocked(
                                snapshot,
                                abort_event=cancel_event,
                            )
                            restored_any = (
                                restored_any or completed or progressed
                            )
                            if not completed:
                                self._insert_snapshot(
                                    self._deferred_snapshots,
                                    snapshot,
                                )
                        elif unresolved is not None:
                            self._insert_snapshot(
                                self._deferred_snapshots,
                                unresolved,
                            )

                    if self._deferred_snapshots:
                        self._write_recovery_journal()
                    else:
                        self._delete_recovery_journal()
                    self._recovery_created_at = None
                    return restored_any
                except _DeferredRestoreAborted:
                    # Every ramp step publishes its owned candidates before a
                    # HAL write. Keep that journal for the waiting begin to
                    # adopt instead of completing the upward restore first.
                    try:
                        self._write_recovery_journal()
                    except Exception:
                        logger.warning(
                            "Failed to preserve cancelled volume recovery",
                            exc_info=True,
                        )
                    return False
                except Exception:
                    logger.warning(
                        "Failed to recover deferred system volume",
                        exc_info=True,
                    )
                    return False
                finally:
                    self._release_lease()
        finally:
            if self._explicit_recovery_cancel is cancel_event:
                self._explicit_recovery_cancel = None
            self._close_lock.release()
            self._explicit_recovery_requested.clear()
            if start_deferred:
                self._start_deferred_sync_if_needed()

    @staticmethod
    def _validate_ratio(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number from 0 to 1")
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{name} must be a number from 0 to 1")
        return result

    def _capture_snapshot(
        self,
        device_id: int,
        device_uid: str,
        elements: tuple[int, ...] | None = None,
    ) -> _DeviceSnapshot | None:
        if self._backend.device_uid(device_id) != device_uid:
            return None
        if elements is None:
            elements = self._backend.volume_elements(device_id)
        if not elements:
            return None
        original = {element: self._backend.get_volume(device_id, element) for element in elements}
        original_mute = None
        if self._max_volume == 0.0:
            try:
                original_mute = self._backend.get_mute(device_id)
            except Exception:
                # Scalar ducking remains available on devices without a
                # usable master-mute control.
                logger.debug(
                    "Could not read the system output mute state",
                    exc_info=True,
                )
        return _DeviceSnapshot(
            device_uid=device_uid,
            device_id_hint=device_id,
            profile_elements=tuple(sorted(elements)),
            original=original,
            original_mute=original_mute,
            mute_target=True if original_mute is not None else None,
        )

    def _configure_post_restore_sync(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
    ) -> None:
        if snapshot.post_restore_sync:
            return
        if self._max_volume != 0.0 or snapshot.original_mute is not False:
            return
        # Hard ducking can race CoreAudio metadata publication. Treat an
        # unreadable transport as unresolved rather than as non-Bluetooth;
        # an explicit non-Bluetooth answer below safely clears the candidate.
        snapshot.post_restore_media_pending = True
        snapshot.post_restore_expected_mute = snapshot.original_mute
        try:
            transport_type = self._backend.transport_type(device_id)
        except Exception:
            logger.debug(
                "Could not read the system output transport type",
                exc_info=True,
            )
            return
        if (
            transport_type is not None
            and transport_type not in _BLUETOOTH_TRANSPORT_TYPES
        ):
            snapshot.post_restore_media_pending = False
            snapshot.post_restore_expected_mute = None
            return
        profile = snapshot.post_restore_profile or ()
        if not profile:
            try:
                profile = tuple(
                    sorted(self._backend.volume_profile(device_id))
                )
            except Exception:
                logger.debug(
                    "Could not read the Bluetooth output profile",
                    exc_info=True,
                )
                profile = ()
        if not profile:
            # Without a raw profile fingerprint, vmvc cannot distinguish HFP
            # from A2DP. A delayed same-value write could then falsely succeed
            # on the call route and strand the remote media gain.
            return
        try:
            route_signature = self._backend.output_route_signature(device_id)
        except Exception:
            logger.debug(
                "Could not read the Bluetooth output route signature",
                exc_info=True,
            )
            return
        if (
            route_signature is None
            or not _is_media_route_signature(route_signature)
        ):
            # Raw volume elements are not a route identity: both HFP and A2DP
            # may expose element 0. Only a valid stereo HAL layout proves that
            # the delayed write belongs to the media route.
            return
        snapshot.post_restore_sync = True
        snapshot.post_restore_media_pending = False
        snapshot.post_restore_profile = profile or None
        snapshot.post_restore_route_signature = route_signature

    @staticmethod
    def _snapshot_key_for(
        device_uid: str,
        elements: tuple[int, ...] | list[int] | dict[int, float],
    ) -> _SnapshotKey:
        return device_uid, tuple(sorted(elements))

    @classmethod
    def _snapshot_key(
        cls,
        snapshot: _DeviceSnapshot,
    ) -> _SnapshotKey:
        return cls._snapshot_key_for(
            snapshot.device_uid,
            snapshot.profile_elements,
        )

    def _refresh_locked(
        self,
        *,
        allow_reduck: bool,
        abandon_on_deviation: bool,
        force_mute: bool,
    ) -> bool:
        device_id = self._backend.default_output_device()
        if device_id is None:
            return False
        device_uid = self._backend.device_uid(device_id)
        if not device_uid:
            return False
        if device_uid in self._overridden_devices:
            if not allow_reduck:
                return False
            self._overridden_devices.remove(device_uid)
        elements = self._backend.volume_elements(device_id)
        if not elements:
            return False
        deferred_ok, deferred_snapshot = self._claim_deferred_route(
            device_id,
            device_uid,
            elements,
        )
        if not deferred_ok:
            return False
        key = self._snapshot_key_for(device_uid, elements)
        snapshot = self._snapshots.get(key)
        if deferred_snapshot is not None:
            if snapshot is not None:
                return False
            needs_duck = True
            if not deferred_snapshot.post_restore_ready:
                self._prepare_hard_snapshot_for_rearm(
                    deferred_snapshot,
                    device_id,
                )
            self._configure_post_restore_sync(
                deferred_snapshot,
                device_id,
            )
            self._snapshots[key] = deferred_snapshot
            self._deferred_snapshots.pop(key, None)
            if needs_duck:
                if deferred_snapshot.post_restore_ready:
                    self._rearm_post_restore_snapshot(deferred_snapshot)
                self._duck_snapshot(
                    deferred_snapshot,
                    mute_first=True,
                )
            return True
        if snapshot is None:
            # A Bluetooth device may expose separate master and stereo
            # controls in its call and media profiles. Each exact topology
            # owns an independent original; values are never mapped between
            # profiles.
            snapshot = self._capture_snapshot(device_id, device_uid, elements)
            if snapshot is None:
                return False
            self._configure_post_restore_sync(snapshot, device_id)
            self._snapshots[key] = snapshot
            try:
                self._duck_snapshot(
                    snapshot,
                    mute_first=(
                        force_mute
                        and self._max_volume == 0.0
                        and snapshot.mute_target is not None
                    ),
                )
            except Exception:
                rolled_back = self._rollback_failed_duck(snapshot)
                self._snapshots.pop(key, None)
                if not rolled_back:
                    self._insert_snapshot(self._snapshots, snapshot)
                self._write_recovery_journal()
                raise
            return True

        route_replaced = device_id != snapshot.device_id_hint
        if self._at_duck_target(snapshot, device_id=device_id):
            snapshot.device_id_hint = device_id
            return True
        if route_replaced:
            snapshot.device_id_hint = device_id
            self._duck_snapshot(
                snapshot,
                mute_first=(
                    self._max_volume == 0.0
                    and snapshot.mute_target is not None
                ),
            )
            return True
        if force_mute and snapshot.mute_target is not None:
            self._duck_snapshot(
                snapshot,
                mute_first=(self._max_volume == 0.0),
            )
            return True
        if not allow_reduck and abandon_on_deviation:
            self._abandon_changed_elements(snapshot, device_id)
            return False
        if not allow_reduck:
            return False
        self._duck_snapshot(snapshot)
        return True

    def _duck_snapshot(
        self,
        snapshot: _DeviceSnapshot,
        *,
        mute_first: bool = False,
    ) -> None:
        if not snapshot.duck_target:
            peak = max(snapshot.original.values(), default=0.0)
            if self._max_volume == 0.0 and snapshot.mute_target is not None:
                # HFP clamps scalar volume to roughly five percent. Reach a
                # quiet, restorable floor first, then use the actual mute
                # control for max_volume=0 instead of fighting that clamp.
                scale = 1.0
                if peak > 0.0:
                    scale = min(scale, _MUTE_PRE_VOLUME / peak)
            else:
                scale = self._factor
                if peak > 0.0:
                    scale = min(scale, self._max_volume / peak)
            snapshot.duck_target = {element: volume * scale for element, volume in snapshot.original.items()}
            if mute_first:
                # A hard handoff may already be quieter than this session's
                # configured cap. Ducking is one-way: never turn that owned
                # value up merely because settings changed between sessions.
                snapshot.duck_target = {
                    element: min(
                        target,
                        self._get_snapshot_volume(snapshot, element),
                    )
                    for element, target in snapshot.duck_target.items()
                }
        if (
            (not mute_first or snapshot.mute_target is None)
            and self._at_duck_target(snapshot)
        ):
            snapshot.phase = _PHASE_DUCKED
            snapshot.transition_started_at = time.time()
            snapshot.owned_values = {element: (value,) for element, value in snapshot.duck_target.items()}
            snapshot.legacy_owned_elements.difference_update(snapshot.duck_target)
            self._write_recovery_journal()
            return
        snapshot.phase = _PHASE_DUCKING
        snapshot.transition_started_at = time.time()
        previous_owned = snapshot.owned_values or {}
        snapshot.owned_values = {
            element: previous_owned.get(element, (value,))
            for element, value in snapshot.original.items()
        }
        snapshot.legacy_owned_elements.difference_update(snapshot.original)
        # Persist originals and this ramp's timestamp before the first
        # hardware write. A crash can then be recovered without guessing.
        self._write_recovery_journal()
        if mute_first:
            # A pending AirPods refresh means the visible scalar is original
            # while the remote gain may still be quiet. Mute before touching
            # that scalar so a newly active A2DP route cannot produce a brief
            # full-volume pulse on the next hotkey press.
            self._apply_mute_target(snapshot)
        self._ramp(
            snapshot,
            snapshot.duck_target,
            duration=_LOWER_DURATION,
            steps=_LOWER_STEPS,
        )
        if mute_first and self._max_volume != 0.0:
            mute_unresolved, _changed, _reachable = (
                self._restore_owned_mute(snapshot)
            )
            if mute_unresolved:
                raise RuntimeError(
                    "Could not release temporary output mute after ducking"
                )
        if not mute_first:
            self._apply_mute_target(snapshot)
        snapshot.phase = _PHASE_DUCKED
        snapshot.transition_started_at = time.time()
        snapshot.owned_values = {element: (value,) for element, value in snapshot.duck_target.items()}
        snapshot.legacy_owned_elements.difference_update(snapshot.duck_target)
        self._write_recovery_journal()

    def _at_duck_target(
        self,
        snapshot: _DeviceSnapshot,
        *,
        device_id: int | None = None,
    ) -> bool:
        if not snapshot.duck_target:
            return False
        if snapshot.mute_target is not None:
            try:
                actual_mute = self._get_snapshot_mute(
                    snapshot,
                    device_id=device_id,
                )
            except Exception:
                return False
            return actual_mute is snapshot.mute_target
        for element, expected in snapshot.duck_target.items():
            if device_id is None:
                actual = self._get_snapshot_volume(snapshot, element)
            else:
                # The monitor just resolved this default ID, UID, and exact
                # topology. Avoid a second CFString round-trip on its hot path.
                actual = self._backend.get_volume(device_id, element)
            if abs(actual - expected) > _OWNERSHIP_TOLERANCE:
                return False
        return True

    def _prepare_hard_snapshot_for_rearm(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
    ) -> None:
        """Apply this session's duck settings to an adopted hard snapshot."""

        old_original_mute = snapshot.original_mute
        old_mute_target = snapshot.mute_target
        old_mute_owned = snapshot.mute_owned
        snapshot.duck_target.clear()
        try:
            current_mute = self._backend.get_mute(device_id)
        except Exception:
            current_mute = None
        if current_mute is None:
            # An unreadable device-wide property during a Bluetooth transition
            # is not evidence that our durable mute claim disappeared. The
            # following duck/restore path will retry the native read.
            return

        still_owns_old_mute = (
            old_mute_owned
            and old_mute_target is True
            and current_mute is old_mute_target
            and old_original_mute is not None
        )
        if self._max_volume != 0.0 and old_mute_target is None:
            snapshot.original_mute = None
            snapshot.mute_target = None
            snapshot.mute_owned = False
            return

        # CoreAudio may reset mute while changing Bluetooth profiles, and the
        # user may also have changed it while this hard snapshot was pending.
        # Preserve the old original only while its target is still visibly
        # ours; otherwise this recording owns the current value as its new
        # original and must restore that exact value when it ends.
        snapshot.original_mute = (
            old_original_mute if still_owns_old_mute else current_mute
        )
        snapshot.mute_target = True
        snapshot.mute_owned = still_owns_old_mute

    def _apply_mute_target(self, snapshot: _DeviceSnapshot) -> None:
        target = snapshot.mute_target
        if target is None:
            return
        actual = self._get_snapshot_mute(snapshot)
        if actual is target:
            return
        if snapshot.original_mute is not target:
            # Persist ownership before the HAL write. Recovery is then safe
            # whether a write fails before changing hardware or succeeds and
            # raises afterward.
            snapshot.mute_owned = True
            self._write_recovery_journal()
        self._set_snapshot_mute(snapshot, target)

    def _abandon_changed_elements(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
    ) -> None:
        """Release only channels changed outside this ducking session."""
        if not self._snapshot_media_profile_is_current(snapshot, device_id):
            return
        mute_abandoned = False
        if snapshot.mute_target is not None:
            try:
                actual_mute = self._get_snapshot_mute(
                    snapshot,
                    device_id=device_id,
                )
            except Exception:
                actual_mute = snapshot.mute_target
            if actual_mute is not snapshot.mute_target:
                snapshot.original_mute = None
                snapshot.mute_target = None
                snapshot.mute_owned = False
                mute_abandoned = True
        retained = []
        for element, target in snapshot.duck_target.items():
            try:
                actual = self._backend.get_volume(device_id, element)
            except Exception:
                # A transient read failure is not evidence of user ownership.
                retained.append(element)
                continue
            if abs(actual - target) <= _OWNERSHIP_TOLERANCE:
                retained.append(element)
        if len(retained) == len(snapshot.original) and not mute_abandoned:
            return
        if retained:
            self._retain_snapshot_elements(snapshot, retained)
        else:
            self._snapshots.pop(self._snapshot_key(snapshot), None)
            self._overridden_devices.add(snapshot.device_uid)
        try:
            self._write_recovery_journal()
        except Exception:
            # A stale journal still contains the old target. Recovery checks
            # the current value before writing, so the user override is safe.
            logger.warning(
                "Failed to remove an overridden output from recovery",
                exc_info=True,
            )

    def _ramp(
        self,
        snapshot: _DeviceSnapshot,
        targets: dict[int, float],
        *,
        duration: float,
        steps: int,
        expected_values: dict[int, float] | None = None,
        abort_event: threading.Event | None = None,
        require_muted: bool = False,
    ) -> set[int]:
        if expected_values is None:
            starts = {element: self._get_snapshot_volume(snapshot, element) for element in targets}
        else:
            starts = dict(expected_values)
        last_values = dict(starts)
        active_elements = set(targets)
        ownership_lost: set[int] = set()
        delay = duration / steps
        for step in range(1, steps + 1):
            if abort_event is not None and abort_event.is_set():
                raise _DeferredRestoreAborted
            if require_muted and not self._get_snapshot_mute(snapshot):
                raise _DeferredRestoreMuteLost
            if expected_values is not None:
                for element in list(active_elements):
                    actual = self._get_snapshot_volume(snapshot, element)
                    if abs(actual - expected_values[element]) > _OWNERSHIP_TOLERANCE:
                        durable_candidates = (snapshot.owned_values or {}).get(
                            element,
                            (),
                        )
                        if not any(
                            abs(actual - candidate) <= _OWNERSHIP_TOLERANCE
                            for candidate in durable_candidates
                        ):
                            active_elements.remove(element)
                            ownership_lost.add(element)
                            continue
                    # A Bluetooth call profile may accept an upward write but
                    # leave the exposed scalar clamped at an earlier value. It
                    # is still our durable value, not evidence of a user edit.
                    # Carry the observed value into the next write-ahead pair so
                    # the following ramp step cannot discard that ownership.
                    last_values[element] = actual
            progress = step / steps
            progress = progress * progress * (3.0 - 2.0 * progress)
            next_values = {element: starts[element] + (targets[element] - starts[element]) * progress for element in active_elements}
            owned_values = dict(snapshot.owned_values or {})
            for element in ownership_lost:
                owned_values.pop(element, None)
                snapshot.legacy_owned_elements.discard(element)
            for element, value in next_values.items():
                previous = last_values[element]
                owned_values[element] = (previous,) if abs(previous - value) <= _OWNERSHIP_TOLERANCE else (previous, value)
                snapshot.legacy_owned_elements.discard(element)
            snapshot.owned_values = owned_values
            # Persist both possible values before each HAL write. A crash can
            # happen on either side of the write without turning a ramp value
            # into an ambiguous user override.
            self._write_recovery_journal()
            if not active_elements:
                break
            for element, value in next_values.items():
                if abort_event is not None and abort_event.is_set():
                    raise _DeferredRestoreAborted
                self._set_snapshot_volume(snapshot, element, value)
                if require_muted and not self._get_snapshot_mute(snapshot):
                    raise _DeferredRestoreMuteLost
                if abort_event is not None and abort_event.is_set():
                    raise _DeferredRestoreAborted
                if expected_values is not None:
                    expected_values[element] = value
                last_values[element] = value
            self._sleep(delay)
        if abort_event is not None and abort_event.is_set():
            raise _DeferredRestoreAborted
        if require_muted and not self._get_snapshot_mute(snapshot):
            raise _DeferredRestoreMuteLost
        return ownership_lost

    def _resolve_snapshot_device(
        self,
        snapshot: _DeviceSnapshot,
    ) -> int | None:
        """Resolve and validate a disposable AudioDeviceID for one operation."""
        try:
            device_id = self._backend.device_id_for_uid(snapshot.device_uid)
            if device_id is None:
                return None
            if self._backend.device_uid(device_id) != snapshot.device_uid:
                return None
            elements = tuple(
                sorted(self._backend.volume_elements(device_id))
            )
            if elements == snapshot.profile_elements:
                pass
            elif snapshot.allow_inactive_controls:
                # A v2/v3 raw-control snapshot may still be recoverable after
                # the route starts preferring virtual-main.
                profile = tuple(
                    sorted(self._backend.volume_profile(device_id))
                )
                if profile != snapshot.profile_elements:
                    return None
                for element in snapshot.original:
                    self._backend.get_volume(device_id, element)
            else:
                return None
        except Exception:
            logger.debug(
                "Output device UID %s is temporarily unavailable",
                snapshot.device_uid,
                exc_info=True,
            )
            return None
        snapshot.device_id_hint = device_id
        return device_id

    def _resolve_snapshot_uid_device(
        self,
        snapshot: _DeviceSnapshot,
    ) -> int | None:
        """Resolve a UID without coupling a device-wide property to volume topology."""
        try:
            device_id = self._backend.device_id_for_uid(snapshot.device_uid)
            if device_id is None:
                return None
            if self._backend.device_uid(device_id) != snapshot.device_uid:
                return None
        except Exception:
            logger.debug(
                "Output device UID %s is temporarily unavailable",
                snapshot.device_uid,
                exc_info=True,
            )
            return None
        snapshot.device_id_hint = device_id
        return device_id

    def _get_snapshot_volume(
        self,
        snapshot: _DeviceSnapshot,
        element: int,
    ) -> float:
        device_id = self._resolve_snapshot_device(snapshot)
        if device_id is None:
            raise RuntimeError(f"Output device {snapshot.device_uid!r} is unavailable")
        return self._backend.get_volume(device_id, element)

    def _set_snapshot_volume(
        self,
        snapshot: _DeviceSnapshot,
        element: int,
        value: float,
    ) -> None:
        # AudioDeviceID values are recycled during Bluetooth profile changes.
        # Resolve and validate the stable UID immediately before every write.
        device_id = self._resolve_snapshot_device(snapshot)
        if device_id is None:
            raise RuntimeError(f"Output device {snapshot.device_uid!r} is unavailable")
        self._backend.set_volume(device_id, element, value)

    def _get_snapshot_mute(
        self,
        snapshot: _DeviceSnapshot,
        *,
        device_id: int | None = None,
    ) -> bool:
        if device_id is None:
            device_id = self._resolve_snapshot_uid_device(snapshot)
        if device_id is None:
            raise RuntimeError(f"Output device {snapshot.device_uid!r} is unavailable")
        muted = self._backend.get_mute(device_id)
        if muted is None:
            raise RuntimeError(
                f"Output device {snapshot.device_uid!r} has no writable mute control"
            )
        return muted

    def _set_snapshot_mute(
        self,
        snapshot: _DeviceSnapshot,
        muted: bool,
    ) -> None:
        device_id = self._resolve_snapshot_uid_device(snapshot)
        if device_id is None:
            raise RuntimeError(f"Output device {snapshot.device_uid!r} is unavailable")
        self._backend.set_mute(device_id, muted)

    def _start_monitor(self, token: DuckToken) -> None:
        if self._background_workers_stopped.is_set():
            raise RuntimeError("System output ducker has been stopped")
        stop = threading.Event()
        thread = threading.Thread(
            target=self._monitor_routes,
            args=(token, stop),
            name="system-volume-monitor",
            daemon=True,
        )
        self._monitor_stop = stop
        self._monitor_thread = thread
        thread.start()
        if self._background_workers_stopped.is_set():
            stop.set()
            raise RuntimeError("System output ducker stopped during monitor start")

    def _monitor_routes(
        self,
        token: DuckToken,
        stop: threading.Event,
    ) -> None:
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            interval = _MONITOR_FAST_INTERVAL if elapsed < _MONITOR_FAST_DURATION else _MONITOR_SLOW_INTERVAL
            if self._monitor_wait(stop, interval):
                return
            with self._lock:
                if stop.is_set() or token is not self._active_token or self._closing:
                    return
                try:
                    self._refresh_locked(
                        allow_reduck=False,
                        abandon_on_deviation=True,
                        # CoreAudio forcibly unmutes AirPods while HFP is
                        # coming up. Reassert only during the existing fast
                        # route-settle window; later user unmute actions win.
                        force_mute=(elapsed < _MONITOR_FAST_DURATION),
                    )
                except Exception:
                    logger.debug(
                        "System output route monitor refresh failed",
                        exc_info=True,
                    )

    def _close_and_restore(self, token: DuckToken) -> bool:
        with self._lock:
            if token is not self._active_token:
                return False
            self._closing = True
            stop = self._monitor_stop
            thread = self._monitor_thread
            if stop is not None:
                stop.set()

        if thread is not None and thread.is_alive():
            if thread is not threading.current_thread():
                thread.join(timeout=_MONITOR_JOIN_TIMEOUT)
            if thread.is_alive():
                # The stop flag was published under the same lock used by the
                # monitor's pre-refresh check. A late monitor can only leave
                # its waiter and observe stop; it cannot write volume again.
                # Continue restoring so a delayed daemon thread cannot strand
                # the active token and lowered output for the process lifetime.
                logger.warning("System output route monitor did not stop before restore")

        if not self._settle_completed:
            # Recorder.stop() can return just before CoreAudio publishes the
            # recovered Bluetooth output route.
            self._sleep(_ROUTE_SETTLE_DURATION)
            with self._lock:
                if token is not self._active_token:
                    return False
                self._settle_completed = True

        write_retries = 0
        route_attempt = 0
        while route_attempt < _RESTORE_ROUTE_ATTEMPTS:
            with self._lock:
                (
                    restored,
                    unavailable,
                    write_failed,
                    post_restore_delay,
                ) = self._restore_pending_locked(
                    token,
                    defer_unavailable=False,
                )
            if restored:
                self._start_deferred_sync_if_needed()
                return True
            if post_restore_delay is not None:
                # Keep the token active, but do not hold the lifecycle lock
                # while A2DP becomes ready. A concurrent begin() can then fail
                # fast instead of waiting and starting a new session afterward.
                self._sleep(post_restore_delay)
                continue
            route_attempt += 1
            if write_failed:
                if write_retries >= 1:
                    return False
                write_retries += 1
            elif not unavailable:
                return False
            if route_attempt < _RESTORE_ROUTE_ATTEMPTS:
                self._sleep(_RESTORE_RETRY_DELAY)

        # The route may stay hidden until the next Bluetooth connection.
        # Persist its UID, release this live token, and resolve it before any
        # later session is allowed to capture a new original.
        with self._lock:
            (
                restored,
                _unavailable,
                _write_failed,
                _post_restore_delay,
            ) = self._restore_pending_locked(
                token,
                defer_unavailable=True,
            )
        if restored:
            self._start_deferred_sync_if_needed()
        return restored

    def _restore_pending_locked(
        self,
        token: DuckToken,
        *,
        defer_unavailable: bool,
    ) -> tuple[bool, bool, bool, float | None]:
        if token is not self._active_token:
            return False, False, False, None

        unavailable = False
        write_failed = False
        journal_changed = False
        post_restore_delay: float | None = None
        for snapshot in list(self._snapshots.values()):
            key = self._snapshot_key(snapshot)
            before_elements = set(snapshot.original)
            if not snapshot.post_restore_sync:
                device_id = self._resolve_snapshot_device(snapshot)
                if device_id is not None:
                    # A Bluetooth route can be unreadable while begin() is
                    # capturing it, then publish its public media layout before
                    # the first restore write. Retry here so that transient HAL
                    # metadata loss does not silently skip the A2DP refresh.
                    self._configure_post_restore_sync(snapshot, device_id)
            if snapshot.post_restore_ready:
                completed, reachable, sync_write_failed = (
                    self._sync_post_restore_snapshot(snapshot)
                )
                journal_changed = True
                self._snapshots.pop(key, None)
                if completed:
                    journal_changed = True
                    continue

                next_delay = self._next_post_restore_delay(snapshot)
                if (
                    next_delay is not None
                    and reachable
                    and not sync_write_failed
                ):
                    if set(snapshot.original) != before_elements:
                        journal_changed = True
                    self._insert_snapshot(self._snapshots, snapshot)
                    post_restore_delay = (
                        next_delay
                        if post_restore_delay is None
                        else min(post_restore_delay, next_delay)
                    )
                    continue

                if set(snapshot.original) != before_elements:
                    journal_changed = True
                should_defer = not reachable and (
                    defer_unavailable or key != self._initial_snapshot_key
                )
                if should_defer:
                    self._insert_snapshot(
                        self._deferred_snapshots,
                        snapshot,
                    )
                    continue
                self._insert_snapshot(self._snapshots, snapshot)
                if sync_write_failed or reachable:
                    write_failed = True
                else:
                    unavailable = True
                continue

            (
                unresolved,
                _changed,
                reachable,
                metadata_dirty,
            ) = self._restore_owned_snapshot(
                snapshot,
                ensure_original_write=True,
                restore_mute_last=(
                    self._live_mute_is_still_owned(snapshot)
                ),
            )
            journal_changed = journal_changed or metadata_dirty
            self._snapshots.pop(key, None)
            if unresolved is None:
                if (
                    snapshot.post_restore_sync
                    or snapshot.post_restore_media_pending
                ) and snapshot.post_restore_values:
                    self._retain_snapshot_elements(
                        snapshot,
                        list(snapshot.post_restore_values),
                    )
                    snapshot.post_restore_ready = True
                    self._insert_snapshot(self._snapshots, snapshot)
                    journal_changed = True
                    next_delay = self._next_post_restore_delay(snapshot)
                    if next_delay is not None:
                        post_restore_delay = (
                            next_delay
                            if post_restore_delay is None
                            else min(post_restore_delay, next_delay)
                        )
                    continue
                self._clear_post_restore_sync(snapshot)
                journal_changed = True
                continue
            if set(unresolved.original) != before_elements:
                journal_changed = True
            should_defer = not reachable and (defer_unavailable or key != self._initial_snapshot_key)
            if should_defer:
                self._insert_snapshot(
                    self._deferred_snapshots,
                    unresolved,
                )
                continue
            self._insert_snapshot(self._snapshots, unresolved)
            if reachable:
                write_failed = True
            else:
                unavailable = True

        if self._snapshots:
            if journal_changed:
                try:
                    self._write_recovery_journal()
                except Exception:
                    # The previous journal remains intact because updates use
                    # an atomic replace. Keep snapshots for a retry.
                    logger.warning(
                        "Failed to update system volume recovery journal",
                        exc_info=True,
                    )
                    write_failed = True
            return (
                False,
                unavailable,
                write_failed,
                post_restore_delay,
            )

        created_at = self._recovery_created_at or time.time()
        if self._deferred_snapshots:
            try:
                self._write_recovery_snapshots(
                    self._deferred_snapshots,
                    created_at,
                )
            except Exception:
                logger.warning(
                    "Failed to retain deferred volume recovery",
                    exc_info=True,
                )
                return False, False, True, None
        else:
            self._delete_recovery_journal()

        self._active_token = None
        self._closing = False
        self._settle_completed = False
        self._monitor_stop = None
        self._monitor_thread = None
        self._overridden_devices.clear()
        self._recovery_created_at = None
        self._initial_snapshot_key = None
        self._release_lease()
        return True, False, False, None

    @staticmethod
    def _next_post_restore_delay(
        snapshot: _DeviceSnapshot,
    ) -> float | None:
        if snapshot.post_restore_pass >= len(_POST_RESTORE_SYNC_DELAYS):
            return None
        return _POST_RESTORE_SYNC_DELAYS[snapshot.post_restore_pass]

    @staticmethod
    def _post_restore_tickle_value(original: float) -> float:
        if original >= _POST_RESTORE_TICKLE_STEP:
            return max(0.0, original - _POST_RESTORE_TICKLE_STEP)
        return min(1.0, original + _POST_RESTORE_TICKLE_STEP)

    @staticmethod
    def _post_restore_value_is_owned(
        snapshot: _DeviceSnapshot,
        element: int,
        actual: float,
        original: float,
    ) -> bool:
        if abs(actual - original) <= _OWNERSHIP_TOLERANCE:
            return True
        return any(
            abs(actual - candidate) <= _OWNERSHIP_TOLERANCE
            for candidate in (snapshot.owned_values or {}).get(element, ())
        )

    @staticmethod
    def _clear_post_restore_sync(snapshot: _DeviceSnapshot) -> None:
        # Mute ownership is deliberately not touched here. Only the verified
        # release paths may clear it; otherwise a future caller could discard
        # the final recovery handle while hardware is still muted.
        snapshot.post_restore_sync = False
        snapshot.post_restore_media_pending = False
        snapshot.post_restore_expected_mute = None
        snapshot.post_restore_profile = None
        snapshot.post_restore_route_signature = None
        snapshot.post_restore_values.clear()
        snapshot.post_restore_ready = False
        snapshot.post_restore_pass = 0

    def _complete_post_restore_sync(
        self,
        snapshot: _DeviceSnapshot,
        *,
        abort_event: threading.Event | None = None,
        abandoned_for_sibling_profile: bool = False,
    ) -> tuple[bool, bool, bool]:
        """Clear a gain refresh only after releasing our durable hard mute."""

        parked = (
            snapshot.original_mute is False
            and snapshot.mute_target is True
            and snapshot.mute_owned
        )
        if parked:
            # Resolve from the stable UID instead of trusting an AudioDeviceID
            # captured before a Bluetooth profile/topology transition.
            device_id = self._resolve_snapshot_uid_device(snapshot)
            if device_id is None:
                return False, False, False
            inactive_profile = False
            if abandoned_for_sibling_profile:
                try:
                    current_id = self._backend.default_output_device()
                    inactive_profile = (
                        current_id == device_id
                        and self._backend.device_uid(device_id)
                        == snapshot.device_uid
                        and tuple(
                            sorted(self._backend.volume_elements(device_id))
                        )
                        != snapshot.profile_elements
                    )
                except Exception:
                    inactive_profile = False
            if (
                not inactive_profile
                and not self._parked_post_restore_scalar_is_safe_to_unmute(
                    snapshot,
                    device_id,
                )
            ):
                return False, False, False
            restored, failed = self._restore_parked_post_restore_mute(
                snapshot,
                device_id,
                abort_event,
            )
            if not restored:
                return False, True, failed

        self._clear_post_restore_sync(snapshot)
        return True, True, False

    def _prepare_post_restore_snapshot_for_rearm(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
        device_uid: str,
        elements: tuple[int, ...],
    ) -> bool:
        """Adopt the route's latest visible user state without restoring first."""
        if not self._snapshot_media_profile_is_current(snapshot, device_id):
            return False
        try:
            if self._backend.default_output_device() != device_id:
                return False
            if self._backend.device_uid(device_id) != device_uid:
                return False
            if device_uid != snapshot.device_uid:
                return False
            if tuple(sorted(elements)) != snapshot.profile_elements:
                return False
            if (
                tuple(sorted(self._backend.volume_elements(device_id)))
                != snapshot.profile_elements
            ):
                return False
            current_mute = self._backend.get_mute(device_id)
            if current_mute is None:
                return False
            current_values = {
                element: self._backend.get_volume(device_id, element)
                for element in snapshot.profile_elements
            }
        except Exception:
            return False

        worker_owned_mute = (
            snapshot.mute_owned
            and snapshot.mute_target is current_mute
            and snapshot.original_mute is not None
        )
        adopted_original: dict[int, float] = {}
        for element, current in current_values.items():
            desired = snapshot.post_restore_values.get(
                element,
                snapshot.original[element],
            )
            if self._post_restore_value_is_owned(
                snapshot,
                element,
                current,
                desired,
            ):
                # A crash may leave the adjacent tickle value visible. Keep the
                # already-persisted desired original instead of adopting that
                # app-owned probe as a user adjustment.
                adopted_original[element] = desired
            else:
                adopted_original[element] = current
                if element in snapshot.post_restore_values:
                    snapshot.post_restore_values[element] = current
        snapshot.original = adopted_original
        snapshot.post_restore_expected_mute = (
            snapshot.original_mute if worker_owned_mute else current_mute
        )
        return True

    def _rearm_post_restore_snapshot(
        self,
        snapshot: _DeviceSnapshot,
    ) -> None:
        """Turn a soft AirPods refresh back into this session's ownership."""
        snapshot.post_restore_values.clear()
        snapshot.post_restore_ready = False
        snapshot.post_restore_pass = 0
        snapshot.duck_target.clear()
        snapshot.owned_values = None
        snapshot.legacy_owned_elements.clear()
        worker_owned_mute = (
            snapshot.mute_owned
            and snapshot.mute_target is True
            and snapshot.original_mute
            is snapshot.post_restore_expected_mute
        )
        snapshot.mute_owned = worker_owned_mute
        if snapshot.post_restore_expected_mute is not None:
            # Even a non-zero duck target needs a temporary mute while the
            # visible original scalar is lowered. Otherwise a newly active
            # A2DP route could turn the first ramp step into a loud pulse.
            snapshot.original_mute = snapshot.post_restore_expected_mute
            snapshot.mute_target = True
        else:
            snapshot.original_mute = None
            snapshot.mute_target = None

    def _request_deferred_sync_stop(self) -> threading.Thread | None:
        """Fence a deferred worker so it cannot perform another HAL write."""
        recovery_cancel = self._explicit_recovery_cancel
        if recovery_cancel is not None:
            recovery_cancel.set()
        # The worker can be blocked in a native HAL call while holding
        # ``_lock``. This atomically published pair lets begin/exit raise the
        # stop fence without waiting for that call to return.
        pair = self._deferred_sync_pair
        if pair is None:
            return None
        thread, stop = pair
        stop.set()
        return thread

    def _stop_deferred_sync(self) -> threading.Thread | None:
        """Stop the previous session's deferred writer before a new begin."""
        thread = self._request_deferred_sync_stop()
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(timeout=_DEFERRED_SYNC_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.warning(
                    "Deferred Bluetooth volume recovery did not stop promptly"
                )
        return thread

    def _arm_deferred_restore_mute(
        self,
        snapshot: _DeviceSnapshot,
        abort_event: threading.Event,
    ) -> bool:
        """Temporarily hide a deferred upward ramp behind a durable mute."""

        if abort_event.is_set():
            raise _DeferredRestoreAborted
        # Scalar ownership is valid only for the exact captured topology. Do
        # not touch device-wide mute while an A2DP snapshot is seeing HFP.
        device_id = self._resolve_snapshot_device(snapshot)
        if device_id is None:
            return False
        try:
            muted = self._get_snapshot_mute(
                snapshot,
                device_id=device_id,
            )
        except Exception:
            return False

        target = snapshot.mute_target
        original = snapshot.original_mute
        if target is not True or original is None:
            return False
        if muted is original:
            # The OS or user already restored mute. Do not reclaim it merely
            # to hide the scalar ramp; fall back to abort-aware restoration.
            snapshot.original_mute = None
            snapshot.mute_target = None
            snapshot.mute_owned = False
            self._write_recovery_journal()
            return False
        if muted is not target:
            return False
        if abort_event.is_set():
            # Leave the existing durable mute in place for begin() to adopt.
            raise _DeferredRestoreAborted
        return True

    def _launch_deferred_sync_locked(
        self,
    ) -> tuple[bool, threading.Thread | None]:
        """Start or confirm a deferred worker while ``_lock`` is held."""

        if self._background_workers_stopped.is_set():
            self._deferred_restart_needed.clear()
            return False, None
        if not self._deferred_snapshots:
            self._deferred_restart_needed.clear()
            return True, None
        self._deferred_restart_needed.set()
        if (
            self._begin_requested.is_set()
            or self._explicit_recovery_requested.is_set()
            or self._active_token is not None
        ):
            return False, None
        pair = self._deferred_sync_pair
        if pair is not None and pair[0].is_alive():
            return not pair[1].is_set(), None

        stop = threading.Event()
        thread = threading.Thread(
            target=self._run_deferred_sync,
            args=(stop,),
            name="system-volume-deferred-sync",
            daemon=True,
        )
        self._deferred_sync_thread = thread
        self._deferred_sync_stop = stop
        self._deferred_sync_pair = (thread, stop)
        try:
            thread.start()
        except Exception:
            self._deferred_sync_thread = None
            self._deferred_sync_stop = None
            self._deferred_sync_pair = None
            logger.warning(
                "Failed to start deferred system volume recovery",
                exc_info=True,
            )
            return False, None
        terminal_thread = None
        if self._background_workers_stopped.is_set():
            stop.set()
            terminal_thread = thread
        return True, terminal_thread

    def _start_deferred_sync_if_needed(self) -> bool:
        if self._recovery_path is None:
            return not self._deferred_snapshots
        if self._background_workers_stopped.is_set():
            self._deferred_restart_needed.clear()
            return False
        if self._explicit_recovery_requested.is_set():
            self._deferred_restart_needed.set()
            return False
        pair = self._deferred_sync_pair
        if (
            pair is not None
            and pair[0].is_alive()
            and not pair[1].is_set()
        ):
            return True
        if not self._lock.acquire(timeout=_LIFECYCLE_LOCK_TIMEOUT):
            logger.warning(
                "Could not schedule deferred system volume recovery promptly"
            )
            return False
        try:
            started, terminal_thread = self._launch_deferred_sync_locked()
        finally:
            self._lock.release()
        if (
            terminal_thread is not None
            and terminal_thread is not threading.current_thread()
        ):
            terminal_thread.join(timeout=_DEFERRED_SYNC_JOIN_TIMEOUT)
        return started

    def _run_deferred_sync(self, stop: threading.Event) -> None:
        current = threading.current_thread()
        started_at = time.monotonic()
        try:
            while (
                not stop.is_set()
                and not self._background_workers_stopped.is_set()
                and not self._explicit_recovery_requested.is_set()
            ):
                still_pending = False
                with self._close_lock:
                    with self._lock:
                        if (
                            stop.is_set()
                            or self._background_workers_stopped.is_set()
                            or self._explicit_recovery_requested.is_set()
                            or self._begin_requested.is_set()
                            or self._active_token is not None
                        ):
                            return
                        if not self._acquire_lease():
                            still_pending = True
                        else:
                            try:
                                # The lease is released between polls so another
                                # WenZi process can record. Reload after every
                                # acquisition; an older worker must never write
                                # its stale in-memory copy over that process's
                                # newer journal.
                                journal = self._read_recovery_journal()
                                if journal is None:
                                    self._deferred_snapshots.clear()
                                    return
                                created_at, snapshots = journal
                                self._deferred_snapshots = snapshots
                                self._recovery_created_at = created_at
                                pending = list(snapshots.items())
                                if not pending:
                                    return
                                journal_changed = False
                                for key, snapshot in pending:
                                    migration, migrated = (
                                        self._migrate_legacy_raw_master_snapshot(
                                            key,
                                            snapshot,
                                        )
                                    )
                                    if migration == _LEGACY_MIGRATION_DEFERRED:
                                        still_pending = True
                                        continue
                                    if migration == _LEGACY_MIGRATION_ABANDONED:
                                        journal_changed = True
                                        continue
                                    if migration == _LEGACY_MIGRATION_MIGRATED:
                                        assert migrated is not None
                                        snapshot = migrated
                                        key = self._snapshot_key(snapshot)
                                        journal_changed = True
                                    if not snapshot.post_restore_ready:
                                        before_elements = set(snapshot.original)
                                        try:
                                            restore_mute_last = (
                                                self._arm_deferred_restore_mute(
                                                    snapshot,
                                                    stop,
                                                )
                                            )
                                            (
                                                unresolved,
                                                changed,
                                                _reachable,
                                                metadata_dirty,
                                            ) = self._restore_owned_snapshot(
                                                snapshot,
                                                ensure_original_write=(
                                                snapshot.post_restore_sync
                                                or snapshot.post_restore_media_pending
                                                ),
                                                restore_mute_last=(
                                                    restore_mute_last
                                                ),
                                                abort_event=(
                                                    stop
                                                ),
                                            )
                                        except _DeferredRestoreAborted:
                                            return
                                        except _DeferredRestoreMuteLost:
                                            # A user unmute wins. Do not keep a
                                            # stale mute claim that could turn
                                            # their output off again later.
                                            snapshot.original_mute = None
                                            snapshot.mute_target = None
                                            snapshot.mute_owned = False
                                            self._write_recovery_journal()
                                            if stop.is_set():
                                                return
                                            continue
                                        self._deferred_snapshots.pop(key, None)
                                        journal_changed = (
                                            journal_changed
                                            or changed
                                            or metadata_dirty
                                            or unresolved is None
                                            or (
                                                unresolved is not None
                                                and set(unresolved.original)
                                                != before_elements
                                            )
                                        )
                                        if unresolved is not None:
                                            self._insert_snapshot(
                                                self._deferred_snapshots,
                                                unresolved,
                                            )
                                            continue
                                        if not (
                                            (
                                                snapshot.post_restore_sync
                                                or snapshot.post_restore_media_pending
                                            )
                                            and snapshot.post_restore_values
                                        ):
                                            self._clear_post_restore_sync(
                                                snapshot
                                            )
                                            continue

                                        # Keep the snapshot in the durable map
                                        # while the same-value A2DP refresh
                                        # arms its temporary-mute write-ahead
                                        # state.
                                        self._retain_snapshot_elements(
                                            snapshot,
                                            list(
                                                snapshot.post_restore_values
                                            ),
                                        )
                                        snapshot.post_restore_ready = True
                                        self._insert_snapshot(
                                            self._deferred_snapshots,
                                            snapshot,
                                        )

                                    before_values = dict(
                                        snapshot.post_restore_values
                                    )
                                    before_sync = snapshot.post_restore_sync
                                    before_pass = snapshot.post_restore_pass
                                    completed, _reachable, _failed = (
                                        self._sync_post_restore_snapshot(
                                            snapshot,
                                            abort_event=stop,
                                        )
                                    )
                                    self._deferred_snapshots.pop(key, None)
                                    if not completed:
                                        self._insert_snapshot(
                                            self._deferred_snapshots,
                                            snapshot,
                                        )
                                    journal_changed = (
                                        journal_changed
                                        or completed
                                        or snapshot.post_restore_sync
                                        is not before_sync
                                        or snapshot.post_restore_values
                                        != before_values
                                        or snapshot.post_restore_pass
                                        != before_pass
                                    )
                                if journal_changed:
                                    self._write_recovery_journal()
                                still_pending = bool(
                                    self._deferred_snapshots
                                )
                            except Exception:
                                still_pending = True
                                logger.debug(
                                    "Deferred system volume recovery failed",
                                    exc_info=True,
                                )
                            finally:
                                self._release_lease()
                if not still_pending:
                    return
                elapsed = time.monotonic() - started_at
                interval = (
                    _DEFERRED_SYNC_INTERVAL
                    if elapsed < _DEFERRED_SYNC_FAST_DURATION
                    else _DEFERRED_SYNC_SLOW_INTERVAL
                )
                if stop.wait(interval):
                    return
        finally:
            restart_needed = False
            with self._lock:
                if self._deferred_sync_thread is current:
                    self._deferred_sync_thread = None
                    self._deferred_sync_stop = None
                    if (
                        self._deferred_sync_pair is not None
                        and self._deferred_sync_pair[0] is current
                    ):
                        self._deferred_sync_pair = None
                    restart_needed = (
                        not self._background_workers_stopped.is_set()
                        and self._deferred_restart_needed.is_set()
                        and self._active_token is None
                        and not self._begin_requested.is_set()
                        and not self._explicit_recovery_requested.is_set()
                    )
            if restart_needed:
                self._start_deferred_sync_if_needed()

    def _resolve_post_restore_default(
        self,
        snapshot: _DeviceSnapshot,
    ) -> tuple[int | None, bool]:
        """Return the exact current default, plus whether to abandon the route."""
        try:
            device_id = self._backend.default_output_device()
            if device_id is None:
                snapshot.post_restore_pass = 0
                return None, False
            device_uid = self._backend.device_uid(device_id)
            if not device_uid:
                snapshot.post_restore_pass = 0
                return None, False
            if device_uid != snapshot.device_uid:
                # Keep this soft pending state by UID. A temporary fallback or
                # deliberate output switch must not erase the AirPods remote
                # gain refresh needed when that output becomes default again.
                snapshot.post_restore_pass = 0
                return None, False
            elements = tuple(sorted(self._backend.volume_elements(device_id)))
            expected_profile = snapshot.post_restore_profile
            media_pending = snapshot.post_restore_media_pending
            profile = None
            route_signature = None
            if expected_profile is not None or media_pending:
                profile = tuple(
                    sorted(self._backend.volume_profile(device_id))
                )
            expected_route_signature = (
                snapshot.post_restore_route_signature
            )
            if expected_route_signature is not None or media_pending:
                route_signature = self._backend.output_route_signature(
                    device_id
                )
            if media_pending:
                transport_type = self._backend.transport_type(device_id)
        except Exception:
            snapshot.post_restore_pass = 0
            logger.debug(
                "Could not resolve the default output for post-restore sync",
                exc_info=True,
            )
            return None, False

        if elements != snapshot.profile_elements:
            snapshot.post_restore_pass = 0
            current_key = self._snapshot_key_for(device_uid, elements)
            abandon = (
                current_key in self._snapshots
                and current_key != self._snapshot_key(snapshot)
            )
            return None, abandon
        if media_pending:
            if (
                transport_type is not None
                and transport_type not in _BLUETOOTH_TRANSPORT_TYPES
            ):
                # A trustworthy non-Bluetooth answer resolves the conservative
                # candidate without ever writing a Bluetooth-only probe.
                return None, True
            if (
                transport_type is None
                or not profile
                or route_signature is None
                or not _is_media_route_signature(route_signature)
            ):
                snapshot.post_restore_pass = 0
                return None, False
            snapshot.post_restore_profile = profile
            snapshot.post_restore_route_signature = route_signature
            snapshot.post_restore_sync = True
            snapshot.post_restore_media_pending = False
            expected_profile = profile
            expected_route_signature = route_signature
        if (
            expected_profile is not None
            and profile != expected_profile
        ):
            # vmvc is present in both HFP and A2DP. The raw controls are the
            # stable signal that the media route is actually back.
            snapshot.post_restore_pass = 0
            return None, False
        if (
            expected_route_signature is not None
            and (
                route_signature is None
                or not _is_media_route_signature(route_signature)
                or route_signature[1] != expected_route_signature[1]
            )
        ):
            # A2DP can return at a different media sample rate after microphone
            # use. Keep the channel layout and media-route checks without
            # stranding recovery solely on a 44.1/48 kHz renegotiation.
            snapshot.post_restore_pass = 0
            return None, False
        snapshot.device_id_hint = device_id
        return device_id, False

    def _post_restore_mute_matches(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
    ) -> bool | None:
        expected_mute = snapshot.post_restore_expected_mute
        if expected_mute is None:
            return True
        try:
            actual_mute = self._backend.get_mute(device_id)
        except Exception:
            logger.debug(
                "Could not read output mute during post-restore sync",
                exc_info=True,
            )
            return None
        if actual_mute is None:
            return None
        return actual_mute is expected_mute

    @staticmethod
    def _post_restore_aborted(
        abort_event: threading.Event | None,
    ) -> bool:
        return abort_event is not None and abort_event.is_set()

    def _park_post_restore_for_begin(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
    ) -> bool:
        """Silence a just-restored route until the waiting begin takes over."""
        expected_mute = snapshot.post_restore_expected_mute
        if expected_mute is not False:
            return expected_mute is True
        try:
            already_owned = (
                snapshot.original_mute is False
                and snapshot.mute_target is True
                and snapshot.mute_owned
            )
            if not already_owned:
                snapshot.original_mute = False
                snapshot.mute_target = True
                snapshot.mute_owned = True
                # Publish ownership first so a hard exit cannot strand mute.
                self._write_recovery_journal()
            self._backend.set_mute(device_id, True)
            return self._backend.get_mute(device_id) is True
        except Exception:
            logger.debug(
                "Could not park Bluetooth output for a waiting recording",
                exc_info=True,
            )
            return False

    def _restore_parked_post_restore_mute(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
        abort_event: threading.Event | None,
    ) -> tuple[bool, bool]:
        """Undo a prior begin fence before retrying the remote refresh."""
        parked = (
            snapshot.original_mute is False
            and snapshot.mute_target is True
            and snapshot.mute_owned
        )
        if not parked:
            return True, False
        if self._post_restore_aborted(abort_event):
            return False, False
        try:
            actual = self._backend.get_mute(device_id)
            if actual is True:
                self._backend.set_mute(device_id, False)
                if self._backend.get_mute(device_id) is not False:
                    return False, True
            elif actual is not False:
                return False, True
            if self._post_restore_aborted(abort_event):
                self._backend.set_mute(device_id, True)
                return False, False
            snapshot.original_mute = None
            snapshot.mute_target = None
            snapshot.mute_owned = False
            self._write_recovery_journal()
            if self._post_restore_aborted(abort_event):
                self._park_post_restore_for_begin(
                    snapshot,
                    device_id,
                )
                return False, False
            return True, False
        except Exception:
            logger.debug(
                "Could not restore parked Bluetooth output mute",
                exc_info=True,
            )
            return False, True

    def _parked_post_restore_scalar_is_safe_to_unmute(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
    ) -> bool:
        """Keep mute when a crash may have left an app-owned probe audible."""

        for element, original in snapshot.post_restore_values.items():
            try:
                actual = self._backend.get_volume(device_id, element)
            except Exception:
                return False
            if abs(actual - original) <= _OWNERSHIP_TOLERANCE:
                continue
            candidates = (snapshot.owned_values or {}).get(element, ())
            if any(
                abs(actual - candidate) <= _OWNERSHIP_TOLERANCE
                for candidate in candidates
            ):
                return False
            # A non-candidate value is a user override. It is safe to unmute,
            # and the later ownership check will remove this element.
        return True

    def _live_post_restore_mute_is_owned(
        self,
        snapshot: _DeviceSnapshot,
    ) -> bool:
        return (
            self._active_token is not None
            and self._snapshots.get(self._snapshot_key(snapshot)) is snapshot
        )

    def _sync_post_restore_until_blocked(
        self,
        snapshot: _DeviceSnapshot,
        *,
        abort_event: threading.Event | None = None,
    ) -> tuple[bool, bool, bool, bool]:
        """Finish durable gain-refresh passes while one media route is stable."""

        progressed = False
        while True:
            before_pass = snapshot.post_restore_pass
            completed, reachable, write_failed = (
                self._sync_post_restore_snapshot(
                    snapshot,
                    abort_event=abort_event,
                )
            )
            progressed = progressed or snapshot.post_restore_pass > before_pass
            if (
                completed
                or write_failed
                or not reachable
                or self._post_restore_aborted(abort_event)
            ):
                return completed, reachable, write_failed, progressed
            if snapshot.post_restore_pass <= before_pass:
                return completed, reachable, write_failed, progressed
            delay = self._next_post_restore_delay(snapshot)
            if delay is None:
                return completed, reachable, write_failed, progressed
            self._sleep(delay)

    def _sync_post_restore_snapshot(
        self,
        snapshot: _DeviceSnapshot,
        *,
        abort_event: threading.Event | None = None,
    ) -> tuple[bool, bool, bool]:
        """Refresh AirPods remote gain after the visible scalar is restored.

        Returns ``(completed, reachable, write_failed)``. Scheduled passes
        retain every successfully written element because the first pass may
        still land on the HFP transition; the final pass retains only failures.
        """
        if self._post_restore_aborted(abort_event):
            return False, False, False

        values = dict(snapshot.post_restore_values)
        if not values or snapshot.post_restore_pass >= len(
            _POST_RESTORE_SYNC_DELAYS
        ):
            return self._complete_post_restore_sync(
                snapshot,
                abort_event=abort_event,
            )

        parked = (
            snapshot.original_mute is False
            and snapshot.mute_target is True
            and snapshot.mute_owned
        )
        device_id, abandon_route = self._resolve_post_restore_default(
            snapshot
        )
        if abandon_route:
            return self._complete_post_restore_sync(
                snapshot,
                abort_event=abort_event,
                abandoned_for_sibling_profile=True,
            )
        if device_id is None:
            # A begin fence or failed transaction may have left our temporary
            # mute durable. Do not keep the user muted while HFP or another
            # default route is active; the scalar obligation remains pending.
            if parked:
                parked_device_id = self._resolve_snapshot_uid_device(snapshot)
                if parked_device_id is not None:
                    if not self._parked_post_restore_scalar_is_safe_to_unmute(
                        snapshot,
                        parked_device_id,
                    ):
                        # The UID is present, but the exact media route needed
                        # to remove a persisted probe is not. Report it as
                        # unavailable so the foreground restore consumes its
                        # bounded route budget and hands the mute-safe WAL to
                        # the deferred worker instead of polling forever.
                        return False, False, False
                    restored, failed = self._restore_parked_post_restore_mute(
                        snapshot,
                        parked_device_id,
                        abort_event,
                    )
                    if not restored:
                        return False, False, failed
            return False, False, False

        if parked:
            try:
                actual_mute = self._backend.get_mute(device_id)
            except Exception:
                return False, True, True
            if actual_mute is False:
                if self._live_post_restore_mute_is_owned(snapshot):
                    # CoreAudio briefly clears AirPods mute while leaving the
                    # call profile. During the bounded foreground close this is
                    # route noise, not a durable user override. Reassert the
                    # already-journaled claim so a following route re-mute can
                    # never outlive the snapshot that owns it.
                    if not self._park_post_restore_for_begin(
                        snapshot,
                        device_id,
                    ):
                        return False, True, True
                else:
                    # Deferred recovery has no live recording owner. Here a
                    # user unmute wins and must not be reclaimed later.
                    snapshot.original_mute = None
                    snapshot.mute_target = None
                    snapshot.mute_owned = False
                    parked = False
                    self._write_recovery_journal()
            elif actual_mute is not True:
                return False, True, True
        else:
            mute_matches = self._post_restore_mute_matches(
                snapshot,
                device_id,
            )
            if mute_matches is None:
                return False, True, True
            if not mute_matches:
                # A user mute change wins over an old remote-gain refresh.
                return self._complete_post_restore_sync(
                    snapshot,
                    abort_event=abort_event,
                )

        eligible: dict[int, float] = {}
        for element, original in values.items():
            try:
                actual = self._backend.get_volume(device_id, element)
            except Exception:
                return False, True, True
            if self._post_restore_value_is_owned(
                snapshot,
                element,
                actual,
                original,
            ):
                eligible[element] = original
        if not eligible:
            return self._complete_post_restore_sync(
                snapshot,
                abort_event=abort_event,
            )

        snapshot.post_restore_values = dict(eligible)
        self._retain_snapshot_elements(snapshot, list(eligible))

        # Resolve the disposable AudioDeviceID again immediately before taking
        # temporary mute ownership. This validates stable UID, topology, raw
        # profile, sample rate, and channel layout as one media-route fence.
        write_device_id, abandon_route = self._resolve_post_restore_default(
            snapshot
        )
        if abandon_route:
            return self._complete_post_restore_sync(
                snapshot,
                abort_event=abort_event,
                abandoned_for_sibling_profile=True,
            )
        if write_device_id is None:
            if parked:
                parked_device_id = self._resolve_snapshot_uid_device(snapshot)
                if (
                    parked_device_id is not None
                    and self._parked_post_restore_scalar_is_safe_to_unmute(
                        snapshot,
                        parked_device_id,
                    )
                ):
                    self._restore_parked_post_restore_mute(
                        snapshot,
                        parked_device_id,
                        abort_event,
                    )
            return False, False, False

        # Re-read after the route fence and before claiming mute. A user can
        # change scalar or mute while CoreAudio is publishing the media route;
        # that newer state must win over this old refresh obligation.
        rechecked: dict[int, float] = {}
        for element, original in eligible.items():
            try:
                actual = self._backend.get_volume(write_device_id, element)
            except Exception:
                return False, True, True
            if self._post_restore_value_is_owned(
                snapshot,
                element,
                actual,
                original,
            ):
                rechecked[element] = original
        if not rechecked:
            return self._complete_post_restore_sync(
                snapshot,
                abort_event=abort_event,
            )
        eligible = rechecked
        snapshot.post_restore_values = dict(eligible)
        self._retain_snapshot_elements(snapshot, list(eligible))

        expected_mute = snapshot.post_restore_expected_mute
        temporary_mute = expected_mute is False
        if temporary_mute and not parked:
            mute_matches = self._post_restore_mute_matches(
                snapshot,
                write_device_id,
            )
            if mute_matches is None:
                return False, True, True
            if not mute_matches:
                return self._complete_post_restore_sync(
                    snapshot,
                    abort_event=abort_event,
                )
            if not self._park_post_restore_for_begin(
                snapshot,
                write_device_id,
            ):
                return False, True, True
            parked = True
        elif expected_mute is True:
            try:
                if self._backend.get_mute(write_device_id) is not True:
                    return self._complete_post_restore_sync(
                        snapshot,
                        abort_event=abort_event,
                    )
            except Exception:
                return False, True, True

        if self._post_restore_aborted(abort_event):
            return False, True, False

        final_device_id, abandon_route = self._resolve_post_restore_default(
            snapshot
        )
        if abandon_route:
            return self._complete_post_restore_sync(
                snapshot,
                abort_event=abort_event,
                abandoned_for_sibling_profile=True,
            )
        if final_device_id != write_device_id:
            if parked:
                parked_device_id = self._resolve_snapshot_uid_device(snapshot)
                if (
                    parked_device_id is not None
                    and self._parked_post_restore_scalar_is_safe_to_unmute(
                        snapshot,
                        parked_device_id,
                    )
                ):
                    self._restore_parked_post_restore_mute(
                        snapshot,
                        parked_device_id,
                        abort_event,
                    )
            return False, False, False
        if expected_mute is not None:
            try:
                required_mute = True if temporary_mute else expected_mute
                actual_mute = self._backend.get_mute(final_device_id)
                if actual_mute is None:
                    return False, True, True
                if actual_mute is not required_mute:
                    if temporary_mute:
                        if (
                            parked
                            and self._live_post_restore_mute_is_owned(snapshot)
                        ):
                            if not self._park_post_restore_for_begin(
                                snapshot,
                                final_device_id,
                            ):
                                return False, True, True
                            actual_mute = True
                        if actual_mute is not required_mute:
                            snapshot.original_mute = None
                            snapshot.mute_target = None
                            snapshot.mute_owned = False
                            return False, True, False
                    else:
                        return False, True, False
            except Exception:
                return False, True, True

        probes = {
            element: self._post_restore_tickle_value(original)
            for element, original in eligible.items()
        }
        owned_values = dict(snapshot.owned_values or {})
        for element, original in eligible.items():
            candidates: list[float] = []
            for candidate in (
                *owned_values.get(element, ()),
                original,
                probes[element],
            ):
                if not any(
                    abs(candidate - retained) <= _OWNERSHIP_TOLERANCE
                    for retained in candidates
                ):
                    candidates.append(candidate)
            # Keep a call-profile clamp (usually 5%) recoverable until the
            # first A2DP write is observed. Replacing it with only the
            # original/probe pair would misclassify the clamp as a user edit.
            owned_values[element] = tuple(candidates)
        snapshot.owned_values = owned_values
        # Persist both sides of every non-equal write before touching HAL. A
        # crash after the probe is then recoverable without mistaking it for a
        # user volume adjustment.
        try:
            self._write_recovery_journal()
        except Exception:
            if parked:
                self._restore_parked_post_restore_mute(
                    snapshot,
                    final_device_id,
                    abort_event,
                )
            return False, True, True

        write_failed = False
        for element, original in eligible.items():
            probe = probes[element]
            probe_written = False
            try:
                actual = self._backend.get_volume(final_device_id, element)
                if not self._post_restore_value_is_owned(
                    snapshot,
                    element,
                    actual,
                    original,
                ):
                    write_failed = True
                    continue
                if abs(actual - original) > _OWNERSHIP_TOLERANCE:
                    self._backend.set_volume(
                        final_device_id,
                        element,
                        original,
                    )
                self._backend.set_volume(
                    final_device_id,
                    element,
                    probe,
                )
                probe_written = True
                # Do not observe cancellation between these writes. The probe
                # must persist briefly, but output remains muted for the whole
                # transaction so this driver-level refresh is inaudible.
                self._sleep(_POST_RESTORE_TICKLE_DWELL)
                # Do not observe the abort fence between probe and original.
                # The pair is one tiny transaction under the lifecycle lock;
                # exposing the probe to begin() would recreate the same bug.
                self._backend.set_volume(
                    final_device_id,
                    element,
                    original,
                )
                probe_written = False
            except Exception:
                write_failed = True
                logger.debug(
                    "Could not tickle Bluetooth output gain for UID %s element %s",
                    snapshot.device_uid,
                    element,
                    exc_info=True,
                )
            finally:
                if probe_written:
                    try:
                        self._backend.set_volume(
                            final_device_id,
                            element,
                            original,
                        )
                    except Exception:
                        write_failed = True

        visible_restored = True
        for element, original in eligible.items():
            try:
                if (
                    abs(
                        self._backend.get_volume(final_device_id, element)
                        - original
                    )
                    > _OWNERSHIP_TOLERANCE
                ):
                    visible_restored = False
            except Exception:
                visible_restored = False

        stable_journal_failed = False
        if visible_restored:
            stable_owned = dict(snapshot.owned_values or {})
            for element, original in eligible.items():
                stable_owned[element] = (original,)
            snapshot.owned_values = stable_owned
            try:
                # The probe is no longer a possible physical value. Publish the
                # stable scalar before unmuting so a crash cannot expose it.
                self._write_recovery_journal()
            except Exception:
                stable_journal_failed = True

        if parked and visible_restored:
            restored, mute_failed = self._restore_parked_post_restore_mute(
                snapshot,
                final_device_id,
                abort_event,
            )
            if not restored:
                return False, True, mute_failed
            parked = False
        if self._post_restore_aborted(abort_event):
            return False, True, False
        if write_failed or not visible_restored or stable_journal_failed:
            return False, True, True

        previous_pass = snapshot.post_restore_pass
        snapshot.post_restore_pass = previous_pass + 1
        try:
            self._write_recovery_journal()
        except Exception:
            # Disk still contains the previous pass. Repeat this harmless
            # transaction rather than claiming success that was not durable.
            snapshot.post_restore_pass = previous_pass
            return False, True, True

        logger.info(
            "Refreshed Bluetooth output gain UID %s route=%s pass=%s/%s",
            snapshot.device_uid,
            snapshot.post_restore_route_signature,
            snapshot.post_restore_pass,
            len(_POST_RESTORE_SYNC_DELAYS),
        )
        if snapshot.post_restore_pass < len(_POST_RESTORE_SYNC_DELAYS):
            return False, True, False

        return self._complete_post_restore_sync(
            snapshot,
            abort_event=abort_event,
        )

    def _restore_targets(
        self,
        snapshot: _DeviceSnapshot,
        elements: list[int],
        *,
        expected_values: dict[int, float] | None = None,
        abort_event: threading.Event | None = None,
        require_muted: bool = False,
    ) -> tuple[set[int], set[int]]:
        targets = {element: snapshot.original[element] for element in elements}
        try:
            ownership_lost = self._ramp(
                snapshot,
                targets,
                duration=_RESTORE_DURATION,
                steps=_RESTORE_STEPS,
                expected_values=expected_values,
                abort_event=abort_event,
                require_muted=require_muted,
            )
            return set(), ownership_lost
        except (_DeferredRestoreAborted, _DeferredRestoreMuteLost):
            raise
        except Exception:
            logger.warning(
                "Failed to ramp-restore output device UID %s (last ID %s)",
                snapshot.device_uid,
                snapshot.device_id_hint,
                exc_info=True,
            )
            return self._set_all_original(
                snapshot,
                elements,
                expected_values=expected_values,
                abort_event=abort_event,
                require_muted=require_muted,
            )

    def _set_all_original(
        self,
        snapshot: _DeviceSnapshot,
        elements: list[int] | None = None,
        *,
        expected_values: dict[int, float] | None = None,
        abort_event: threading.Event | None = None,
        require_muted: bool = False,
    ) -> tuple[set[int], set[int]]:
        unresolved: set[int] = set()
        ownership_lost: set[int] = set()
        if elements is None:
            elements = list(snapshot.original)
        for element in elements:
            try:
                if abort_event is not None and abort_event.is_set():
                    raise _DeferredRestoreAborted
                if require_muted and not self._get_snapshot_mute(snapshot):
                    raise _DeferredRestoreMuteLost
                actual = self._get_snapshot_volume(snapshot, element)
                if expected_values is not None:
                    if abs(actual - expected_values[element]) > _OWNERSHIP_TOLERANCE:
                        durable_candidates = (snapshot.owned_values or {}).get(element, ())
                        if not any(abs(actual - candidate) <= _OWNERSHIP_TOLERANCE for candidate in durable_candidates):
                            ownership_lost.add(element)
                            if snapshot.owned_values is not None:
                                snapshot.owned_values.pop(element, None)
                            snapshot.legacy_owned_elements.discard(element)
                            continue
                owned_values = dict(snapshot.owned_values or {})
                original = snapshot.original[element]
                owned_values[element] = (actual,) if abs(actual - original) <= _OWNERSHIP_TOLERANCE else (actual, original)
                snapshot.owned_values = owned_values
                snapshot.legacy_owned_elements.discard(element)
                self._write_recovery_journal()
                self._set_snapshot_volume(
                    snapshot,
                    element,
                    original,
                )
                if require_muted and not self._get_snapshot_mute(snapshot):
                    raise _DeferredRestoreMuteLost
                if abort_event is not None and abort_event.is_set():
                    raise _DeferredRestoreAborted
            except (_DeferredRestoreAborted, _DeferredRestoreMuteLost):
                raise
            except Exception:
                unresolved.add(element)
                logger.warning(
                    "Failed to restore output device UID %s element %s",
                    snapshot.device_uid,
                    element,
                    exc_info=True,
                )
        return unresolved, ownership_lost

    def _restore_owned_snapshot(
        self,
        snapshot: _DeviceSnapshot,
        *,
        ensure_original_write: bool = False,
        restore_mute_last: bool = False,
        abort_event: threading.Event | None = None,
    ) -> tuple[_DeviceSnapshot | None, bool, bool, bool]:
        mute_metadata_before = (
            snapshot.original_mute,
            snapshot.mute_target,
            snapshot.mute_owned,
        )
        mute_changed = False
        mute_reachable = True
        mute_metadata_dirty = False
        if not restore_mute_last:
            mute_unresolved, mute_changed, mute_reachable = (
                self._restore_owned_mute(
                    snapshot,
                    abort_event=abort_event,
                )
            )
            mute_metadata_dirty = mute_metadata_before != (
                snapshot.original_mute,
                snapshot.mute_target,
                snapshot.mute_owned,
            )
            if mute_unresolved:
                return snapshot, False, mute_reachable, mute_metadata_dirty

        (
            restore_elements,
            unresolved_elements,
            already_original_elements,
            reachable,
            observed_values,
        ) = self._classify_owned_elements(snapshot)
        if ensure_original_write:
            restore_elements.extend(already_original_elements)

        restored = False
        post_restore_elements: set[int] = set()
        if restore_elements:
            if ensure_original_write and (
                snapshot.post_restore_sync
                or snapshot.post_restore_media_pending
            ):
                # Arm the remote-gain refresh before the first local restore
                # write. A hard process exit after the final vmvc write must
                # not lose the fact that A2DP still needs a same-value refresh.
                snapshot.post_restore_values = {
                    element: snapshot.original[element]
                    for element in restore_elements
                }
                snapshot.post_restore_ready = False
            # Drop user-owned and already-restored elements from the durable
            # record before publishing the restoring phase. Otherwise a crash
            # during another channel's ramp could reclaim them on restart.
            self._retain_snapshot_elements(
                snapshot,
                [*restore_elements, *unresolved_elements],
            )
            snapshot.phase = _PHASE_RESTORING
            snapshot.transition_started_at = time.time()
            current_owned = dict(snapshot.owned_values or {})
            for element in restore_elements:
                current_owned[element] = (observed_values[element],)
                snapshot.legacy_owned_elements.discard(element)
            snapshot.owned_values = current_owned
            try:
                # A crash during the upward ramp must remain distinguishable
                # from a user volume change on the next process start.
                self._write_recovery_journal()
            except Exception:
                logger.warning(
                    "Failed to journal output restoration transition",
                    exc_info=True,
                )
                unresolved_elements.extend(restore_elements)
            else:
                failed_elements, lost_elements = self._restore_targets(
                    snapshot,
                    restore_elements,
                    expected_values={element: observed_values[element] for element in restore_elements},
                    abort_event=(
                        None if restore_mute_last else abort_event
                    ),
                    require_muted=restore_mute_last,
                )
                unresolved_elements.extend(failed_elements)
                post_restore_elements = (
                    set(restore_elements) - failed_elements - lost_elements
                )
                restored = bool(post_restore_elements)

        if ensure_original_write and (
            snapshot.post_restore_sync
            or snapshot.post_restore_media_pending
        ):
            snapshot.post_restore_values = {
                element: snapshot.original[element]
                for element in post_restore_elements
            }
            if unresolved_elements:
                # Keep locally restored channels alongside failures until the
                # complete route is ready for its delayed remote-gain refresh.
                unresolved_elements.extend(
                    element
                    for element in post_restore_elements
                    if element not in unresolved_elements
                )

        hold_mute_for_post_restore = (
            restore_mute_last
            and snapshot.post_restore_sync
            and bool(snapshot.post_restore_values)
            and not unresolved_elements
        )
        if restore_mute_last and not hold_mute_for_post_restore:
            if abort_event is not None and abort_event.is_set():
                raise _DeferredRestoreAborted
            mute_unresolved, mute_changed, mute_reachable = (
                self._restore_owned_mute(
                    snapshot,
                    abort_event=abort_event,
                )
            )
            mute_metadata_dirty = mute_metadata_before != (
                snapshot.original_mute,
                snapshot.mute_target,
                snapshot.mute_owned,
            )
            if mute_unresolved:
                # Retain the scalar evidence as an anchor for the unresolved
                # mute claim. owned_values still prevents user-owned scalar
                # changes from being restored on a later retry.
                return (
                    snapshot,
                    mute_changed or restored or bool(already_original_elements),
                    reachable or mute_reachable,
                    mute_metadata_dirty,
                )

        if abort_event is not None and abort_event.is_set():
            raise _DeferredRestoreAborted

        return (
            self._snapshot_part(snapshot, unresolved_elements),
            mute_changed or restored or bool(already_original_elements),
            reachable,
            mute_metadata_dirty,
        )

    def _live_mute_is_still_owned(
        self,
        snapshot: _DeviceSnapshot,
    ) -> bool:
        """Keep an app-owned hard mute until live scalar restore completes."""
        if not (
            snapshot.original_mute is False
            and snapshot.mute_target is True
            and snapshot.mute_owned
        ):
            return False
        try:
            return self._get_snapshot_mute(snapshot) is True
        except Exception:
            # The normal mute-first restore path retains an unreachable claim;
            # it is safer than assuming ownership while the route is missing.
            return False

    def _restore_owned_mute(
        self,
        snapshot: _DeviceSnapshot,
        *,
        abort_event: threading.Event | None = None,
    ) -> tuple[bool, bool, bool]:
        target = snapshot.mute_target
        original = snapshot.original_mute
        if target is None or original is None:
            return False, False, True

        device_id = self._resolve_snapshot_uid_device(snapshot)
        if device_id is None:
            return True, False, False
        try:
            actual = self._get_snapshot_mute(
                snapshot,
                device_id=device_id,
            )
        except Exception:
            logger.warning(
                "Failed to read output mute for device UID %s",
                snapshot.device_uid,
                exc_info=True,
            )
            return True, False, True

        if actual is original:
            snapshot.original_mute = None
            snapshot.mute_target = None
            snapshot.mute_owned = False
            return False, True, True

        if not snapshot.mute_owned or actual is not target:
            # A value outside our original/target pair belongs to the user.
            snapshot.original_mute = None
            snapshot.mute_target = None
            snapshot.mute_owned = False
            return False, False, True

        snapshot.phase = _PHASE_RESTORING
        snapshot.transition_started_at = time.time()
        try:
            # Unmute while volume is still at the quiet floor, then let the
            # regular scalar ramp bring playback back smoothly.
            self._write_recovery_journal()
            if abort_event is not None and abort_event.is_set():
                raise _DeferredRestoreAborted
            self._set_snapshot_mute(snapshot, original)
            if abort_event is not None and abort_event.is_set():
                # A new recording arrived during the native unmute. Put the
                # output back under the existing durable mute claim before
                # handing the snapshot to that session.
                self._set_snapshot_mute(snapshot, target)
                raise _DeferredRestoreAborted
            if self._get_snapshot_mute(snapshot) is not original:
                return True, False, True
        except _DeferredRestoreAborted:
            raise
        except Exception:
            logger.warning(
                "Failed to restore output mute for device UID %s",
                snapshot.device_uid,
                exc_info=True,
            )
            return True, False, True

        snapshot.original_mute = None
        snapshot.mute_target = None
        snapshot.mute_owned = False
        return False, True, True

    @staticmethod
    def _retain_snapshot_elements(
        snapshot: _DeviceSnapshot,
        elements: list[int],
    ) -> None:
        retained = set(elements)
        snapshot.original = {element: value for element, value in snapshot.original.items() if element in retained}
        snapshot.duck_target = {element: value for element, value in snapshot.duck_target.items() if element in retained}
        if snapshot.owned_values is not None:
            snapshot.owned_values = {element: value for element, value in snapshot.owned_values.items() if element in retained}
        snapshot.legacy_owned_elements.intersection_update(retained)

    def _snapshot_media_profile_is_current(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
    ) -> bool:
        expected_profile = snapshot.post_restore_profile
        if expected_profile is None:
            return True
        try:
            if self._backend.device_uid(device_id) != snapshot.device_uid:
                return False
            profile = tuple(sorted(self._backend.volume_profile(device_id)))
            route_signature = self._backend.output_route_signature(device_id)
        except Exception:
            return False
        expected_signature = snapshot.post_restore_route_signature
        # AirPods can keep the same UID, vmvc control, and sample rate while
        # HFP exposes a different scalar. That value cannot prove a user edit
        # to the media volume saved by this snapshot.
        return (
            profile == expected_profile
            and route_signature is not None
            and _is_media_route_signature(route_signature)
            and (
                expected_signature is None
                or route_signature[1] == expected_signature[1]
            )
        )

    def _classify_owned_elements(
        self,
        snapshot: _DeviceSnapshot,
    ) -> tuple[
        list[int],
        list[int],
        list[int],
        bool,
        dict[int, float],
    ]:
        restore_elements: list[int] = []
        unresolved_elements: list[int] = []
        already_original_elements: list[int] = []
        observed_values: dict[int, float] = {}
        started_at = snapshot.transition_started_at
        age = max(0.0, time.time() - started_at) if started_at is not None else float("inf")
        transition_in_progress = snapshot.phase in {
            _PHASE_DUCKING,
            _PHASE_RESTORING,
        }

        device_id = self._resolve_snapshot_device(snapshot)
        if device_id is None or not self._snapshot_media_profile_is_current(
            snapshot, device_id
        ):
            return (
                restore_elements,
                list(snapshot.original),
                already_original_elements,
                False,
                observed_values,
            )

        read_any = False
        for element, original in snapshot.original.items():
            target = snapshot.duck_target[element]
            try:
                actual = self._backend.get_volume(device_id, element)
            except Exception:
                unresolved_elements.append(element)
                continue
            read_any = True
            observed_values[element] = actual
            if abs(actual - original) <= _OWNERSHIP_TOLERANCE:
                already_original_elements.append(element)
                continue
            if snapshot.owned_values is not None and element not in snapshot.legacy_owned_elements:
                candidates = snapshot.owned_values.get(element, ())
                if any(abs(actual - candidate) <= _OWNERSHIP_TOLERANCE for candidate in candidates):
                    restore_elements.append(element)
                # No candidate match means this element belongs to the user.
                continue
            if abs(actual - target) <= _OWNERSHIP_TOLERANCE:
                restore_elements.append(element)
                continue
            if (
                transition_in_progress
                and age <= _RECOVERY_RAMP_GRACE
                and min(original, target) - _OWNERSHIP_TOLERANCE <= actual <= max(original, target) + _OWNERSHIP_TOLERANCE
            ):
                restore_elements.append(element)
            # Any other value is a user change. Drop only this element.

        return (
            restore_elements,
            unresolved_elements,
            already_original_elements,
            read_any,
            observed_values,
        )

    def _rollback_failed_duck(self, snapshot: _DeviceSnapshot) -> bool:
        """Undo this process's own incomplete ramp without ownership guessing."""
        snapshot.phase = _PHASE_RESTORING
        snapshot.transition_started_at = time.time()
        try:
            self._write_recovery_journal()
        except Exception:
            logger.warning(
                "Failed to journal failed-duck rollback",
                exc_info=True,
            )
            return False
        mute_unresolved, _mute_changed, _mute_reachable = (
            self._restore_owned_mute(snapshot)
        )
        if mute_unresolved:
            return False
        failed_elements, _lost_elements = self._restore_targets(
            snapshot,
            list(snapshot.original),
        )
        return not failed_elements

    @staticmethod
    def _snapshot_part(
        snapshot: _DeviceSnapshot,
        elements: list[int],
    ) -> _DeviceSnapshot | None:
        if not elements:
            return None
        return _DeviceSnapshot(
            device_uid=snapshot.device_uid,
            device_id_hint=snapshot.device_id_hint,
            profile_elements=snapshot.profile_elements,
            original={element: snapshot.original[element] for element in elements},
            duck_target={element: snapshot.duck_target[element] for element in elements},
            owned_values=(
                {element: snapshot.owned_values[element] for element in elements if element in snapshot.owned_values}
                if snapshot.owned_values is not None
                else None
            ),
            legacy_owned_elements=(snapshot.legacy_owned_elements.intersection(elements)),
            original_mute=snapshot.original_mute,
            mute_target=snapshot.mute_target,
            mute_owned=snapshot.mute_owned,
            phase=snapshot.phase,
            transition_started_at=snapshot.transition_started_at,
            post_restore_sync=snapshot.post_restore_sync,
            post_restore_media_pending=(
                snapshot.post_restore_media_pending
            ),
            post_restore_expected_mute=(
                snapshot.post_restore_expected_mute
            ),
            post_restore_profile=snapshot.post_restore_profile,
            post_restore_route_signature=(
                snapshot.post_restore_route_signature
            ),
            post_restore_values={
                element: value
                for element, value in snapshot.post_restore_values.items()
                if element in elements
            },
            post_restore_ready=snapshot.post_restore_ready,
            post_restore_pass=snapshot.post_restore_pass,
            allow_inactive_controls=snapshot.allow_inactive_controls,
        )

    def _read_recovery_journal(
        self,
    ) -> (
        tuple[
            float,
            dict[_SnapshotKey, _DeviceSnapshot],
        ]
        | None
    ):
        path = self._recovery_path
        if path is None or not path.exists():
            self._legacy_recovery_blocked = False
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("invalid recovery journal")
            version = data.get("version")
            if version == 1:
                # AudioDeviceID is recycled by CoreAudio. A v1 entry cannot
                # prove that its numeric ID still identifies the same output,
                # and known v1 data may already contain compounded targets.
                if self._legacy_v1_may_still_be_ducked(data):
                    self._legacy_recovery_blocked = True
                    logger.error("Legacy system volume recovery is ambiguous; raise the playback volume once before ducking is re-enabled")
                    return None
                self._legacy_recovery_blocked = False
                logger.warning("Discarding inactive v1 volume recovery journal")
                self._delete_recovery_journal()
                return None
            if version not in _COMPATIBLE_RECOVERY_VERSIONS:
                raise ValueError("unsupported recovery journal version")
            allow_virtual = version == _RECOVERY_VERSION
            created_at = float(data["created_at"])
            if not math.isfinite(created_at):
                raise ValueError("invalid recovery timestamp")
            snapshots: dict[_SnapshotKey, _DeviceSnapshot] = {}
            for item in data["devices"]:
                device_uid = item["device_uid"]
                if not isinstance(device_uid, str) or not device_uid:
                    raise ValueError("invalid recovery device UID")
                device_id_hint = self._journal_integer(item["device_id_hint"])
                original = self._journal_volumes(
                    item["original"],
                    allow_virtual=allow_virtual,
                )
                target = self._journal_volumes(
                    item["target"],
                    allow_virtual=allow_virtual,
                )
                if not original or original.keys() != target.keys():
                    raise ValueError("invalid recovery volume elements")
                owned_values = None
                legacy_owned_elements: set[int]
                if "owned" in item:
                    owned_values = self._journal_owned_values(
                        item["owned"],
                        allow_virtual=allow_virtual,
                    )
                    if not set(owned_values).issubset(original):
                        raise ValueError("invalid recovery owned elements")
                    legacy_data = item.get("legacy", [])
                    if not isinstance(legacy_data, list):
                        raise ValueError("invalid legacy owned elements")
                    legacy_owned_elements = {
                        self._journal_integer(
                            element,
                            allow_zero=True,
                            allow_virtual=allow_virtual,
                        )
                        for element in legacy_data
                    }
                    if not legacy_owned_elements.issubset(original) or legacy_owned_elements.intersection(owned_values):
                        raise ValueError("invalid legacy owned elements")
                else:
                    legacy_owned_elements = set(original)
                profile_elements = tuple(
                    self._journal_integer(
                        element,
                        allow_zero=True,
                        allow_virtual=allow_virtual,
                    )
                    for element in item["elements"]
                )
                if (
                    not profile_elements
                    or len(set(profile_elements)) != len(profile_elements)
                    or not set(original).issubset(profile_elements)
                ):
                    raise ValueError("invalid recovery device profile")
                phase = item["phase"]
                if phase not in _RECOVERY_PHASES:
                    raise ValueError("invalid recovery phase")
                transition_started_at = float(item.get("transition_started_at", created_at))
                if not math.isfinite(transition_started_at):
                    raise ValueError("invalid transition timestamp")
                original_mute = None
                mute_target = None
                mute_owned = False
                mute_data = item.get("mute")
                if mute_data is not None:
                    if (
                        version not in _MUTE_RECOVERY_VERSIONS
                        or not isinstance(mute_data, dict)
                    ):
                        raise ValueError("invalid recovery mute state")
                    original_mute = mute_data.get("original")
                    mute_target = mute_data.get("target")
                    mute_owned = mute_data.get("owned")
                    if not all(
                        isinstance(value, bool)
                        for value in (original_mute, mute_target, mute_owned)
                    ):
                        raise ValueError("invalid recovery mute state")
                    if mute_owned and original_mute is mute_target:
                        raise ValueError("invalid recovery mute ownership")
                post_restore_sync = False
                post_restore_media_pending = False
                post_restore_expected_mute = None
                post_restore_profile = None
                post_restore_route_signature = None
                post_restore_values: dict[int, float] = {}
                post_restore_ready = False
                post_restore_pass = 0
                stored_profile = item.get("volume_profile")
                if stored_profile is not None:
                    if (
                        version != _RECOVERY_VERSION
                        or not isinstance(stored_profile, list)
                        or not stored_profile
                    ):
                        raise ValueError("invalid recovery volume profile")
                    post_restore_profile = tuple(
                        sorted(
                            self._journal_integer(
                                element,
                                allow_zero=True,
                            )
                            for element in stored_profile
                        )
                    )
                    if len(set(post_restore_profile)) != len(
                        post_restore_profile
                    ):
                        raise ValueError("invalid recovery volume profile")
                stored_route_signature = item.get("route_signature")
                if stored_route_signature is not None:
                    if version != _RECOVERY_VERSION:
                        raise ValueError(
                            "invalid recovery route signature"
                        )
                    post_restore_route_signature = (
                        self._journal_route_signature(
                            stored_route_signature
                        )
                    )
                stored_media_pending = item.get(
                    "post_restore_media_pending",
                    False,
                )
                if not isinstance(stored_media_pending, bool):
                    raise ValueError("invalid post-restore media-pending state")
                if stored_media_pending:
                    if version != _RECOVERY_VERSION:
                        raise ValueError("invalid post-restore media-pending state")
                    post_restore_sync = False
                    post_restore_media_pending = True
                    post_restore_expected_mute = False
                post_data = item.get("post_restore")
                if post_data is not None:
                    if (
                        version != _RECOVERY_VERSION
                        or not isinstance(post_data, dict)
                    ):
                        raise ValueError("invalid post-restore recovery state")
                    post_restore_expected_mute = post_data.get("expected_mute")
                    if not isinstance(post_restore_expected_mute, bool):
                        raise ValueError("invalid post-restore mute state")
                    if (
                        post_restore_media_pending
                        and post_restore_expected_mute is not False
                    ):
                        raise ValueError("invalid pending post-restore mute state")
                    profile_data = post_data.get("profile")
                    if profile_data is None and post_restore_media_pending:
                        pending_profile = None
                    else:
                        if not isinstance(profile_data, list) or not profile_data:
                            raise ValueError("invalid post-restore profile")
                        pending_profile = tuple(
                            sorted(
                                self._journal_integer(
                                    element,
                                    allow_zero=True,
                                )
                                for element in profile_data
                            )
                        )
                        if len(set(pending_profile)) != len(pending_profile):
                            raise ValueError("invalid post-restore profile")
                        if (
                            post_restore_profile is not None
                            and pending_profile != post_restore_profile
                        ):
                            raise ValueError("conflicting post-restore profile")
                        post_restore_profile = pending_profile
                    pending_route_signature_data = post_data.get(
                        "route_signature"
                    )
                    if pending_route_signature_data is not None:
                        pending_route_signature = (
                            self._journal_route_signature(
                                pending_route_signature_data
                            )
                        )
                        if (
                            post_restore_route_signature is not None
                            and pending_route_signature
                            != post_restore_route_signature
                        ):
                            raise ValueError(
                                "conflicting post-restore route signature"
                            )
                        post_restore_route_signature = (
                            pending_route_signature
                        )
                    post_restore_values = self._journal_volumes(
                        post_data.get("values"),
                        allow_virtual=True,
                    )
                    if (
                        not post_restore_values
                        or not set(post_restore_values).issubset(original)
                    ):
                        raise ValueError("invalid post-restore values")
                    post_restore_ready = post_data.get("ready")
                    if not isinstance(post_restore_ready, bool):
                        raise ValueError("invalid post-restore readiness")
                    stored_pass = post_data.get("pass", 0)
                    if (
                        isinstance(stored_pass, bool)
                        or not isinstance(stored_pass, int)
                        or not 0
                        <= stored_pass
                        <= len(_POST_RESTORE_SYNC_DELAYS)
                    ):
                        raise ValueError("invalid post-restore pass")
                    post_restore_pass = stored_pass
                    post_restore_sync = not post_restore_media_pending
                elif (
                    version == _RECOVERY_VERSION
                    and post_restore_profile
                    and _VIRTUAL_MAIN_ELEMENT in profile_elements
                ):
                    # A v4 duck journal already carries enough durable intent
                    # to finish the Bluetooth refresh after a crash, even if
                    # the local restore had not yet armed concrete values. The
                    # mute ownership may already have been released after HFP
                    # reset it, but this profile was captured only after proving
                    # the original Bluetooth output was unmuted.
                    post_restore_sync = True
                    post_restore_expected_mute = False
                allow_inactive_controls = version in {2, 3}
                stored_allow_inactive = item.get(
                    "allow_inactive_controls"
                )
                if stored_allow_inactive is not None:
                    if (
                        version != _RECOVERY_VERSION
                        or not isinstance(stored_allow_inactive, bool)
                    ):
                        raise ValueError(
                            "invalid inactive-control recovery state"
                        )
                    allow_inactive_controls = stored_allow_inactive
                snapshot = _DeviceSnapshot(
                    device_uid=device_uid,
                    device_id_hint=device_id_hint,
                    profile_elements=tuple(sorted(profile_elements)),
                    original=original,
                    duck_target=target,
                    owned_values=owned_values,
                    legacy_owned_elements=legacy_owned_elements,
                    original_mute=original_mute,
                    mute_target=mute_target,
                    mute_owned=mute_owned,
                    phase=phase,
                    transition_started_at=transition_started_at,
                    post_restore_sync=post_restore_sync,
                    post_restore_media_pending=(
                        post_restore_media_pending
                    ),
                    post_restore_expected_mute=(
                        post_restore_expected_mute
                    ),
                    post_restore_profile=post_restore_profile,
                    post_restore_route_signature=(
                        post_restore_route_signature
                    ),
                    post_restore_values=post_restore_values,
                    post_restore_ready=post_restore_ready,
                    post_restore_pass=post_restore_pass,
                    allow_inactive_controls=allow_inactive_controls,
                )
                key = self._snapshot_key(snapshot)
                if key in snapshots:
                    raise ValueError("duplicate recovery device profile")
                snapshots[key] = snapshot
            if not snapshots:
                raise ValueError("empty recovery journal")
            return created_at, snapshots
        except Exception:
            logger.warning("Invalid system volume recovery journal", exc_info=True)
            self._delete_recovery_journal()
            return None

    def _legacy_v1_may_still_be_ducked(self, data: dict) -> bool:
        """Conservatively detect a v1 cap without trusting recycled IDs."""
        try:
            target_values = [volume for item in data["devices"] for volume in self._journal_volumes(item["target"]).values()]
            if not target_values:
                return True
            device_id = self._backend.default_output_device()
            if device_id is None:
                return True
            elements = self._backend.volume_elements(device_id)
            if not elements:
                return True
            current_values = [self._backend.get_volume(device_id, element) for element in elements]
            if not current_values:
                return True
        except Exception:
            logger.debug(
                "Could not classify legacy volume recovery",
                exc_info=True,
            )
            return True
        return max(current_values) <= (max(target_values) + _OWNERSHIP_TOLERANCE)

    def _load_deferred_recovery(self) -> None:
        if self._recovery_path is None:
            return
        journal = self._read_recovery_journal()
        if journal is None:
            self._deferred_snapshots = {}
            return
        created_at, self._deferred_snapshots = journal
        self._recovery_created_at = created_at

    @staticmethod
    def _legacy_raw_master_candidates(
        snapshot: _DeviceSnapshot,
    ) -> tuple[float, ...]:
        candidates = list(
            (snapshot.owned_values or {}).get(_MAIN_ELEMENT, ())
        )
        target = snapshot.duck_target[_MAIN_ELEMENT]
        if not any(
            abs(target - candidate) <= _OWNERSHIP_TOLERANCE
            for candidate in candidates
        ):
            candidates.append(target)
        return tuple(candidates)

    @staticmethod
    def _matches_any_owned_value(
        value: float,
        candidates: tuple[float, ...],
    ) -> bool:
        return any(
            abs(value - candidate) <= _OWNERSHIP_TOLERANCE
            for candidate in candidates
        )

    def _persist_legacy_snapshot_replacement(
        self,
        old_key: _SnapshotKey,
        old_snapshot: _DeviceSnapshot,
        replacement: _DeviceSnapshot | None,
    ) -> bool:
        """Publish a legacy migration before its new ownership can touch HAL."""

        replacement_key = (
            self._snapshot_key(replacement)
            if replacement is not None
            else None
        )
        if (
            replacement_key is not None
            and replacement_key != old_key
            and replacement_key in self._deferred_snapshots
        ):
            return False

        self._deferred_snapshots.pop(old_key, None)
        if replacement is not None:
            self._deferred_snapshots[replacement_key] = replacement
        try:
            self._write_recovery_journal()
        except Exception:
            if replacement_key is not None:
                self._deferred_snapshots.pop(replacement_key, None)
            self._deferred_snapshots[old_key] = old_snapshot
            logger.warning(
                "Failed to persist legacy output recovery migration",
                exc_info=True,
            )
            return False
        return True

    def _migrate_legacy_raw_master_snapshot(
        self,
        key: _SnapshotKey,
        snapshot: _DeviceSnapshot,
    ) -> tuple[int, _DeviceSnapshot | None]:
        """Conservatively translate a v3 raw-master claim to A2DP vmvc."""

        if (
            snapshot.profile_elements != (_MAIN_ELEMENT,)
            or not snapshot.allow_inactive_controls
        ):
            return _LEGACY_MIGRATION_NOT_APPLICABLE, None

        try:
            device_id = self._backend.device_id_for_uid(snapshot.device_uid)
            if device_id is None:
                return _LEGACY_MIGRATION_DEFERRED, None
            if self._backend.device_uid(device_id) != snapshot.device_uid:
                return _LEGACY_MIGRATION_DEFERRED, None
            transport = self._backend.transport_type(device_id)
        except Exception:
            logger.debug(
                "Could not classify legacy raw-master output recovery",
                exc_info=True,
            )
            return _LEGACY_MIGRATION_DEFERRED, None

        if transport not in _BLUETOOTH_TRANSPORT_TYPES:
            # Built-in and other non-Bluetooth outputs keep the existing raw
            # control recovery semantics.
            return _LEGACY_MIGRATION_NOT_APPLICABLE, None

        try:
            default_device = self._backend.default_output_device()
            if default_device != device_id:
                return _LEGACY_MIGRATION_DEFERRED, None
            if self._backend.device_uid(default_device) != snapshot.device_uid:
                return _LEGACY_MIGRATION_DEFERRED, None
            elements = tuple(
                sorted(self._backend.volume_elements(device_id))
            )
            if elements != (_VIRTUAL_MAIN_ELEMENT,):
                return _LEGACY_MIGRATION_DEFERRED, None
            profile = tuple(
                sorted(self._backend.volume_profile(device_id))
            )
            route_signature = self._backend.output_route_signature(device_id)
            if (
                not profile
                or route_signature is None
                or not _is_media_route_signature(route_signature)
            ):
                return _LEGACY_MIGRATION_DEFERRED, None
            virtual_value = self._backend.get_volume(
                device_id,
                _VIRTUAL_MAIN_ELEMENT,
            )
            raw_value = self._backend.get_volume(device_id, _MAIN_ELEMENT)
            current_mute = self._backend.get_mute(device_id)
            if current_mute is None:
                return _LEGACY_MIGRATION_DEFERRED, None
        except Exception:
            logger.debug(
                "Legacy raw-master output is not ready for migration",
                exc_info=True,
            )
            return _LEGACY_MIGRATION_DEFERRED, None

        original = snapshot.original[_MAIN_ELEMENT]
        owned_candidates = self._legacy_raw_master_candidates(snapshot)
        soft_pending = (
            abs(virtual_value - original) <= _OWNERSHIP_TOLERANCE
        )
        hard_pending = (
            self._matches_any_owned_value(
                virtual_value,
                owned_candidates,
            )
            and self._matches_any_owned_value(
                raw_value,
                owned_candidates,
            )
        )
        owns_current_mute = (
            snapshot.original_mute is not None
            and snapshot.mute_target is not None
            and snapshot.mute_owned
            and current_mute is snapshot.mute_target
        )
        if not soft_pending and not hard_pending:
            # vmvc is the user-visible control. A value outside the old
            # original/owned set proves that the old process no longer owns it.
            if owns_current_mute:
                mute_only = _DeviceSnapshot(
                    device_uid=snapshot.device_uid,
                    device_id_hint=device_id,
                    profile_elements=(_VIRTUAL_MAIN_ELEMENT,),
                    original={_VIRTUAL_MAIN_ELEMENT: virtual_value},
                    duck_target={_VIRTUAL_MAIN_ELEMENT: virtual_value},
                    owned_values={
                        _VIRTUAL_MAIN_ELEMENT: (virtual_value,),
                    },
                    original_mute=snapshot.original_mute,
                    mute_target=snapshot.mute_target,
                    mute_owned=True,
                    phase=_PHASE_RESTORING,
                    transition_started_at=snapshot.transition_started_at,
                    allow_inactive_controls=False,
                )
                if self._persist_legacy_snapshot_replacement(
                    key,
                    snapshot,
                    mute_only,
                ):
                    return _LEGACY_MIGRATION_MIGRATED, mute_only
                return _LEGACY_MIGRATION_DEFERRED, None
            if self._persist_legacy_snapshot_replacement(
                key,
                snapshot,
                None,
            ):
                return _LEGACY_MIGRATION_ABANDONED, None
            return _LEGACY_MIGRATION_DEFERRED, None

        original_mute = None
        mute_target = None
        mute_owned = False
        expected_mute = current_mute
        if owns_current_mute:
            original_mute = snapshot.original_mute
            mute_target = snapshot.mute_target
            mute_owned = True
            expected_mute = snapshot.original_mute

        migrated = _DeviceSnapshot(
            device_uid=snapshot.device_uid,
            device_id_hint=device_id,
            profile_elements=(_VIRTUAL_MAIN_ELEMENT,),
            original={_VIRTUAL_MAIN_ELEMENT: original},
            duck_target={_VIRTUAL_MAIN_ELEMENT: virtual_value},
            owned_values={
                _VIRTUAL_MAIN_ELEMENT: (virtual_value,),
            },
            original_mute=original_mute,
            mute_target=mute_target,
            mute_owned=mute_owned,
            phase=(
                _PHASE_RESTORING if soft_pending else _PHASE_DUCKED
            ),
            transition_started_at=snapshot.transition_started_at,
            post_restore_sync=True,
            post_restore_expected_mute=expected_mute,
            post_restore_profile=profile,
            post_restore_route_signature=route_signature,
            post_restore_values={_VIRTUAL_MAIN_ELEMENT: original},
            post_restore_ready=soft_pending,
            post_restore_pass=0,
            allow_inactive_controls=False,
        )
        if not self._persist_legacy_snapshot_replacement(
            key,
            snapshot,
            migrated,
        ):
            return _LEGACY_MIGRATION_DEFERRED, None
        return _LEGACY_MIGRATION_MIGRATED, migrated

    def _claim_deferred_route(
        self,
        device_id: int,
        device_uid: str,
        elements: tuple[int, ...],
    ) -> tuple[bool, _DeviceSnapshot | None]:
        """Adopt an unchanged old cap without an audible restore/re-duck pulse."""
        key = self._snapshot_key_for(device_uid, elements)
        snapshot = self._deferred_snapshots.get(key)
        if snapshot is None:
            if not self._resolve_legacy_controls_before_virtual_main(
                device_id,
                device_uid,
                elements,
            ):
                return False, None
            snapshot = self._deferred_snapshots.get(key)
            if snapshot is None:
                return True, None

        if snapshot.post_restore_ready:
            if self._prepare_post_restore_snapshot_for_rearm(
                snapshot,
                device_id,
                device_uid,
                elements,
            ):
                snapshot.device_id_hint = device_id
                return True, snapshot
            # Never replay a full-gain post restore from the hotkey path. The
            # fenced background worker will retry after this failed begin.
            return False, None

        complete_profile = set(snapshot.original) == set(elements)
        if (
            complete_profile
            and self._rebase_muted_deferred_scalar_overrides(
                snapshot,
                device_id,
                device_uid,
                elements,
            )
        ):
            snapshot.device_id_hint = device_id
            return True, snapshot
        target_is_owned = all(
            element in snapshot.legacy_owned_elements
            or snapshot.owned_values is None
            or any(
                abs(snapshot.duck_target[element] - candidate) <= _OWNERSHIP_TOLERANCE
                for candidate in snapshot.owned_values.get(element, ())
            )
            for element in elements
        )
        if complete_profile and target_is_owned:
            try:
                exact_route = (
                    self._backend.device_uid(device_id) == device_uid
                    and tuple(sorted(self._backend.volume_elements(device_id))) == snapshot.profile_elements
                )
                still_at_target = exact_route and self._at_duck_target(
                    snapshot,
                    device_id=device_id,
                )
            except Exception:
                still_at_target = False
            if still_at_target:
                snapshot.device_id_hint = device_id
                return True, snapshot

        if (
            complete_profile
            and self._deferred_snapshot_is_owned_for_handoff(
                snapshot,
                device_id,
                elements,
            )
        ):
            # A fenced background restore may have stopped at any WAL-owned
            # scalar candidate. The new recording must lower that candidate
            # directly, never finish the old upward restore first.
            snapshot.device_id_hint = device_id
            return True, snapshot

        # A restored or user-modified route must finish the old ownership
        # classification before this session captures a fresh original.
        return self._resolve_deferred_route(device_uid, elements), None

    def _rebase_muted_deferred_scalar_overrides(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
        device_uid: str,
        elements: tuple[int, ...],
    ) -> bool:
        """Adopt scalar changes made while a hard snapshot remains muted."""

        if not self._snapshot_media_profile_is_current(snapshot, device_id):
            return False
        owned_values = snapshot.owned_values
        if (
            snapshot.mute_target is not True
            or owned_values is None
            or snapshot.legacy_owned_elements
        ):
            return False
        try:
            if self._backend.device_uid(device_id) != device_uid:
                return False
            if (
                tuple(sorted(self._backend.volume_elements(device_id)))
                != snapshot.profile_elements
            ):
                return False
            if self._backend.get_mute(device_id) is not True:
                return False

            current_values: dict[int, float] = {}
            changed_elements: list[int] = []
            for element in elements:
                candidates = owned_values.get(element, ())
                if not candidates:
                    return False
                actual = self._backend.get_volume(device_id, element)
                current_values[element] = actual
                if not any(
                    abs(actual - candidate) <= _OWNERSHIP_TOLERANCE
                    for candidate in candidates
                ):
                    changed_elements.append(element)
        except Exception:
            return False
        if not changed_elements:
            return False

        for element in changed_elements:
            actual = current_values[element]
            snapshot.original[element] = actual
            owned_values[element] = (actual,)
            if element in snapshot.post_restore_values:
                snapshot.post_restore_values[element] = actual
        return True

    def _deferred_snapshot_is_owned_for_handoff(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
        elements: tuple[int, ...],
    ) -> bool:
        try:
            if self._backend.device_uid(device_id) != snapshot.device_uid:
                return False
            if (
                tuple(sorted(self._backend.volume_elements(device_id)))
                != snapshot.profile_elements
            ):
                return False
            owned_values = snapshot.owned_values
            if owned_values is None:
                return False
            for element in elements:
                if element in snapshot.legacy_owned_elements:
                    return False
                actual = self._backend.get_volume(device_id, element)
                candidates = owned_values.get(element, ())
                if not any(
                    abs(actual - candidate) <= _OWNERSHIP_TOLERANCE
                    for candidate in candidates
                ):
                    return False
        except Exception:
            return False
        return True

    def _resolve_legacy_controls_before_virtual_main(
        self,
        device_id: int,
        device_uid: str,
        elements: tuple[int, ...],
    ) -> bool:
        """Finish old raw-control ownership before capturing virtual-main."""
        if _VIRTUAL_MAIN_ELEMENT not in elements:
            return True

        legacy = [
            (key, snapshot)
            for key, snapshot in self._deferred_snapshots.items()
            if snapshot.device_uid == device_uid
            and _VIRTUAL_MAIN_ELEMENT not in snapshot.profile_elements
        ]
        if not legacy:
            return True

        unresolved_any = False
        for key, snapshot in legacy:
            migration, _migrated = (
                self._migrate_legacy_raw_master_snapshot(
                    key,
                    snapshot,
                )
            )
            if migration == _LEGACY_MIGRATION_DEFERRED:
                unresolved_any = True
                continue
            if migration in {
                _LEGACY_MIGRATION_MIGRATED,
                _LEGACY_MIGRATION_ABANDONED,
            }:
                continue
            snapshot.allow_inactive_controls = True
            unresolved, _changed, _reachable, _metadata_dirty = (
                self._restore_owned_snapshot(snapshot)
            )
            self._deferred_snapshots.pop(key, None)
            if (
                unresolved is not None
                and self._virtual_main_proves_legacy_master_is_unowned(
                    unresolved,
                    device_id,
                )
            ):
                unresolved = None
            if unresolved is not None:
                unresolved_any = True
                self._insert_snapshot(self._deferred_snapshots, unresolved)

        try:
            self._write_recovery_journal()
        except Exception:
            logger.warning(
                "Failed to update legacy output recovery before virtual-main capture",
                exc_info=True,
            )
            return False
        return not unresolved_any

    def _virtual_main_proves_legacy_master_is_unowned(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
    ) -> bool:
        """Use vmvc only to abandon an old raw-master claim, never to restore it."""
        if snapshot.profile_elements != (_MAIN_ELEMENT,):
            return False
        if snapshot.original_mute is not None or snapshot.mute_target is not None:
            return False
        try:
            current = self._backend.get_volume(
                device_id,
                _VIRTUAL_MAIN_ELEMENT,
            )
        except Exception:
            return False

        original = snapshot.original[_MAIN_ELEMENT]
        if abs(current - original) <= _OWNERSHIP_TOLERANCE:
            return True
        candidates = (snapshot.owned_values or {}).get(
            _MAIN_ELEMENT,
            (),
        )
        if any(
            abs(current - candidate) <= _OWNERSHIP_TOLERANCE
            for candidate in candidates
        ):
            return False
        if (
            snapshot.owned_values is None
            or _MAIN_ELEMENT in snapshot.legacy_owned_elements
        ) and abs(
            current - snapshot.duck_target[_MAIN_ELEMENT]
        ) <= _OWNERSHIP_TOLERANCE:
            return False

        started_at = snapshot.transition_started_at
        age = (
            max(0.0, time.time() - started_at)
            if started_at is not None
            else float("inf")
        )
        if (
            snapshot.phase in {_PHASE_DUCKING, _PHASE_RESTORING}
            and age <= _RECOVERY_RAMP_GRACE
            and min(original, snapshot.duck_target[_MAIN_ELEMENT])
            - _OWNERSHIP_TOLERANCE
            <= current
            <= max(original, snapshot.duck_target[_MAIN_ELEMENT])
            + _OWNERSHIP_TOLERANCE
        ):
            return False
        return True

    def _resolve_deferred_route(
        self,
        device_uid: str,
        elements: tuple[int, ...],
    ) -> bool:
        """Resolve old ownership before capturing a reconnected route."""
        key = self._snapshot_key_for(device_uid, elements)
        snapshot = self._deferred_snapshots.get(key)
        if snapshot is None:
            return True

        (
            unresolved,
            _changed,
            _reachable,
            _metadata_dirty,
        ) = self._restore_owned_snapshot(snapshot)
        self._deferred_snapshots.pop(key, None)
        if unresolved is not None:
            self._insert_snapshot(
                self._deferred_snapshots,
                unresolved,
            )

        try:
            self._write_recovery_journal()
        except Exception:
            logger.warning(
                "Failed to update reconnected route recovery",
                exc_info=True,
            )
            return False
        return unresolved is None

    @staticmethod
    def _journal_integer(
        value,
        *,
        allow_zero: bool = False,
        allow_virtual: bool = False,
    ) -> int:
        if isinstance(value, bool):
            raise ValueError("invalid recovery integer")
        if isinstance(value, str):
            if allow_virtual and value == str(_VIRTUAL_MAIN_ELEMENT):
                return _VIRTUAL_MAIN_ELEMENT
            if not value.isascii() or not value.isdecimal():
                raise ValueError("invalid recovery integer")
            result = int(value)
        elif isinstance(value, int):
            result = value
        else:
            raise ValueError("invalid recovery integer")
        if allow_virtual and result == _VIRTUAL_MAIN_ELEMENT:
            return result
        minimum = 0 if allow_zero else 1
        if result < minimum:
            raise ValueError("invalid recovery integer")
        return result

    @classmethod
    def _journal_route_signature(cls, value) -> tuple[int, int]:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("invalid recovery route signature")
        signature = (
            cls._journal_integer(value[0]),
            cls._journal_integer(value[1]),
        )
        if not _valid_route_signature(signature):
            raise ValueError("invalid recovery route signature")
        return signature

    @classmethod
    def _journal_volumes(
        cls,
        values,
        *,
        allow_virtual: bool = False,
    ) -> dict[int, float]:
        if not isinstance(values, dict):
            raise ValueError("invalid recovery volumes")
        result: dict[int, float] = {}
        for element, volume in values.items():
            element_id = cls._journal_integer(
                element,
                allow_zero=True,
                allow_virtual=allow_virtual,
            )
            scalar = float(volume)
            if isinstance(volume, bool) or not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
                raise ValueError("invalid recovery volume")
            result[element_id] = scalar
        return result

    @classmethod
    def _journal_owned_values(
        cls,
        values,
        *,
        allow_virtual: bool = False,
    ) -> dict[int, tuple[float, ...]]:
        if not isinstance(values, dict):
            raise ValueError("invalid recovery owned values")
        result: dict[int, tuple[float, ...]] = {}
        for element, candidates in values.items():
            element_id = cls._journal_integer(
                element,
                allow_zero=True,
                allow_virtual=allow_virtual,
            )
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
                raise ValueError("invalid recovery owned candidates")
            parsed = tuple(float(candidate) for candidate in candidates)
            if any(
                isinstance(candidate, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0
                for candidate, value in zip(candidates, parsed, strict=True)
            ):
                raise ValueError("invalid recovery owned candidate")
            result[element_id] = parsed
        return result

    def _write_recovery_journal(self) -> None:
        created_at = self._recovery_created_at
        if created_at is None:
            created_at = time.time()
            self._recovery_created_at = created_at
        snapshots: dict[_SnapshotKey, _DeviceSnapshot] = {}
        for snapshot in self._snapshots.values():
            self._insert_snapshot(snapshots, snapshot)
        for snapshot in self._deferred_snapshots.values():
            self._insert_snapshot(snapshots, snapshot)
        self._write_recovery_snapshots(snapshots, created_at)

    def _write_recovery_snapshots(
        self,
        snapshots: dict[_SnapshotKey, _DeviceSnapshot],
        created_at: float,
    ) -> None:
        path = self._recovery_path
        if path is None:
            return
        if not snapshots:
            self._delete_recovery_journal()
            return
        devices = []
        for _key, snapshot in sorted(snapshots.items()):
            item = {
                "device_uid": snapshot.device_uid,
                "device_id_hint": snapshot.device_id_hint,
                "elements": list(snapshot.profile_elements),
                "original": {str(element): volume for element, volume in snapshot.original.items()},
                "target": {str(element): volume for element, volume in snapshot.duck_target.items()},
                "phase": snapshot.phase,
                "transition_started_at": (snapshot.transition_started_at or created_at),
            }
            if snapshot.owned_values is not None:
                item["owned"] = {str(element): list(candidates) for element, candidates in snapshot.owned_values.items()}
            if snapshot.post_restore_profile:
                item["volume_profile"] = list(
                    snapshot.post_restore_profile
                )
            if snapshot.post_restore_media_pending:
                item["post_restore_media_pending"] = True
            if snapshot.post_restore_route_signature is not None:
                item["route_signature"] = list(
                    snapshot.post_restore_route_signature
                )
            if snapshot.allow_inactive_controls:
                item["allow_inactive_controls"] = True
            if snapshot.legacy_owned_elements:
                item["legacy"] = sorted(snapshot.legacy_owned_elements)
            if (
                snapshot.original_mute is not None
                and snapshot.mute_target is not None
            ):
                item["mute"] = {
                    "original": snapshot.original_mute,
                    "target": snapshot.mute_target,
                    "owned": snapshot.mute_owned,
                }
            if (
                (
                    snapshot.post_restore_sync
                    or snapshot.post_restore_media_pending
                )
                and snapshot.post_restore_expected_mute is not None
                and snapshot.post_restore_values
            ):
                item["post_restore"] = {
                    "expected_mute": snapshot.post_restore_expected_mute,
                    "ready": snapshot.post_restore_ready,
                    "pass": snapshot.post_restore_pass,
                    "values": {
                        str(element): value
                        for element, value in snapshot.post_restore_values.items()
                    },
                }
                if snapshot.post_restore_profile:
                    item["post_restore"]["profile"] = list(
                        snapshot.post_restore_profile
                    )
                if snapshot.post_restore_route_signature is not None:
                    item["post_restore"]["route_signature"] = list(
                        snapshot.post_restore_route_signature
                    )
            devices.append(item)
        data = {
            "version": _RECOVERY_VERSION,
            "created_at": created_at,
            "devices": devices,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        fd = os.open(
            temp_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

    @classmethod
    def _insert_snapshot(
        cls,
        snapshots: dict[_SnapshotKey, _DeviceSnapshot],
        snapshot: _DeviceSnapshot,
    ) -> None:
        key = cls._snapshot_key(snapshot)
        existing = snapshots.get(key)
        if existing is None:
            snapshots[key] = snapshot
            return
        if existing != snapshot:
            raise RuntimeError(f"Conflicting volume ownership for device UID {snapshot.device_uid!r}")

    def _acquire_lease(self) -> bool:
        if self._lease_fd is not None:
            return True
        path = self._recovery_path
        if path is None:
            return True
        lock_path = path.with_name(f"{path.name}.lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception:
                os.close(fd)
                raise
        except BlockingIOError:
            return False
        except OSError:
            logger.warning(
                "Failed to acquire system volume ownership lease",
                exc_info=True,
            )
            return False
        self._lease_fd = fd
        return True

    def _release_lease(self) -> None:
        fd = self._lease_fd
        if fd is None:
            return
        self._lease_fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            logger.debug("Failed to unlock system volume lease", exc_info=True)
        try:
            os.close(fd)
        except OSError:
            logger.debug("Failed to close system volume lease", exc_info=True)

    def _delete_recovery_journal(self) -> None:
        path = self._recovery_path
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to delete system volume recovery journal", exc_info=True)

    def _rollback_and_clear(
        self,
        *,
        rearm_post_restore_key: _SnapshotKey | None = None,
    ) -> None:
        if self._monitor_stop is not None:
            self._monitor_stop.set()
        for snapshot in list(self._snapshots.values()):
            key = self._snapshot_key(snapshot)
            rolled_back = self._rollback_failed_duck(snapshot)
            self._snapshots.pop(key, None)
            if not rolled_back:
                self._insert_snapshot(self._snapshots, snapshot)
            elif key == rearm_post_restore_key:
                # This session adopted an AirPods snapshot whose hidden gain
                # still needed an A2DP refresh. A failed re-duck may restore
                # the local scalar successfully, but it must not erase that
                # remote refresh obligation.
                snapshot.post_restore_values = dict(snapshot.original)
                snapshot.post_restore_ready = True
                snapshot.post_restore_pass = 0
                self._insert_snapshot(self._deferred_snapshots, snapshot)
        if self._snapshots:
            self._closing = True
            try:
                self._write_recovery_journal()
            except Exception:
                logger.warning(
                    "Failed to preserve system volume recovery journal",
                    exc_info=True,
                )
        else:
            self._active_token = None
            self._closing = False
            self._settle_completed = False
            self._monitor_stop = None
            self._monitor_thread = None
            self._overridden_devices.clear()
            self._initial_snapshot_key = None
            created_at = self._recovery_created_at or time.time()
            if self._deferred_snapshots:
                try:
                    self._write_recovery_snapshots(
                        self._deferred_snapshots,
                        created_at,
                    )
                except Exception:
                    logger.warning(
                        "Failed to retain deferred volume recovery",
                        exc_info=True,
                    )
                    self._closing = True
                    return
            else:
                self._delete_recovery_journal()
            self._recovery_created_at = None
            self._release_lease()
