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
_PREFERRED_STEREO_CHANNELS = _fourcc("dch2")
_SCOPE_GLOBAL = _fourcc("glob")
_SCOPE_OUTPUT = _fourcc("outp")
_MAIN_ELEMENT = 0

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
_MONITOR_FAST_DURATION = 2.0
_MONITOR_FAST_INTERVAL = 0.02
_MONITOR_SLOW_INTERVAL = 0.25
_MONITOR_JOIN_TIMEOUT = 1.0
_RECOVERY_VERSION = 2
_RECOVERY_RAMP_GRACE = 1.0
_CF_STRING_ENCODING_UTF8 = 0x08000100

_PHASE_DUCKING = "ducking"
_PHASE_DUCKED = "ducked"
_PHASE_RESTORING = "restoring"
_RECOVERY_PHASES = {
    _PHASE_DUCKING,
    _PHASE_DUCKED,
    _PHASE_RESTORING,
}


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


class _CoreAudioError(RuntimeError):
    def __init__(self, operation: str, status: int) -> None:
        super().__init__(f"{operation} failed with OSStatus {status}")
        self.status = status


class _VolumeBackend(Protocol):
    def default_output_device(self) -> int | None: ...

    def device_uid(self, device_id: int) -> str | None: ...

    def device_id_for_uid(self, device_uid: str) -> int | None: ...

    def volume_elements(self, device_id: int) -> tuple[int, ...]: ...

    def get_volume(self, device_id: int, element: int) -> float: ...

    def set_volume(self, device_id: int, element: int, value: float) -> None: ...


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

    def volume_elements(self, device_id: int) -> tuple[int, ...]:
        master = self._address(
            _VOLUME_SCALAR,
            _SCOPE_OUTPUT,
            _MAIN_ELEMENT,
        )
        if self._is_writable(device_id, master):
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
        address = self._address(
            _VOLUME_SCALAR,
            _SCOPE_OUTPUT,
            element,
        )
        value = float(self._get_value(device_id, address, ctypes.c_float).value)
        if not math.isfinite(value):
            raise RuntimeError("CoreAudio returned a non-finite volume")
        return max(0.0, min(1.0, value))

    def set_volume(self, device_id: int, element: int, value: float) -> None:
        address = self._address(
            _VOLUME_SCALAR,
            _SCOPE_OUTPUT,
            element,
        )
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
    phase: str = _PHASE_DUCKED
    transition_started_at: float | None = None


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

        with self._lock:
            if self._active_token is not None:
                return None
            if not self._acquire_lease():
                logger.info("Skipping system volume duck because another process owns it")
                return None

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
                if snapshot is None:
                    snapshot = self._capture_snapshot(
                        device_id,
                        device_uid,
                        elements,
                    )
                if snapshot is None:
                    return None

                token = DuckToken()
                self._active_token = token
                initial_key = self._snapshot_key(snapshot)
                self._snapshots = {initial_key: snapshot}
                self._initial_snapshot_key = initial_key
                if adopted:
                    self._deferred_snapshots.pop(initial_key, None)
                self._overridden_devices.clear()
                self._settle_completed = False
                if self._recovery_created_at is None:
                    self._recovery_created_at = time.time()
                if not adopted:
                    self._duck_snapshot(snapshot)
                self._start_monitor(token)
                return token
            except Exception:
                logger.warning(
                    "Failed to lower the system output volume",
                    exc_info=True,
                )
                self._rollback_and_clear()
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
                )
            except Exception:
                logger.warning(
                    "Failed to lower a changed output route",
                    exc_info=True,
                )
                return False

    def end(self, token: DuckToken) -> bool:
        """Restore volumes owned by *token*; stale tokens are harmless."""
        with self._close_lock:
            with self._lock:
                if token is not self._active_token:
                    return False
            return self._close_and_restore(token)

    def restore_all(self) -> bool:
        """Best-effort process-shutdown fallback for an active session."""
        with self._close_lock:
            with self._lock:
                token = self._active_token
                if token is None:
                    return False
            return self._close_and_restore(token)

    def recover_stale(self) -> bool:
        """Restore a safely identifiable volume left by an earlier crash."""
        with self._close_lock:
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
                        unresolved, changed, _reachable = self._restore_owned_snapshot(snapshot)
                        restored_any = restored_any or changed
                        self._deferred_snapshots.pop(key, None)
                        if unresolved is not None:
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
                except Exception:
                    logger.warning(
                        "Failed to recover deferred system volume",
                        exc_info=True,
                    )
                    return False
                finally:
                    self._release_lease()

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
        return _DeviceSnapshot(
            device_uid=device_uid,
            device_id_hint=device_id,
            profile_elements=tuple(sorted(elements)),
            original=original,
        )

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
            self._snapshots[key] = deferred_snapshot
            self._deferred_snapshots.pop(key, None)
            return True
        if snapshot is None:
            # A Bluetooth device may expose separate master and stereo
            # controls in its call and media profiles. Each exact topology
            # owns an independent original; values are never mapped between
            # profiles.
            snapshot = self._capture_snapshot(device_id, device_uid, elements)
            if snapshot is None:
                return False
            self._snapshots[key] = snapshot
            try:
                self._duck_snapshot(snapshot)
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
            self._duck_snapshot(snapshot)
            return True
        if not allow_reduck and abandon_on_deviation:
            self._abandon_changed_elements(snapshot, device_id)
            return False
        if not allow_reduck:
            return False
        self._duck_snapshot(snapshot)
        return True

    def _duck_snapshot(self, snapshot: _DeviceSnapshot) -> None:
        if not snapshot.duck_target:
            peak = max(snapshot.original.values(), default=0.0)
            scale = self._factor
            if peak > 0.0:
                scale = min(scale, self._max_volume / peak)
            snapshot.duck_target = {element: volume * scale for element, volume in snapshot.original.items()}
        if self._at_duck_target(snapshot):
            snapshot.phase = _PHASE_DUCKED
            snapshot.transition_started_at = time.time()
            snapshot.owned_values = {element: (value,) for element, value in snapshot.duck_target.items()}
            snapshot.legacy_owned_elements.difference_update(snapshot.duck_target)
            self._write_recovery_journal()
            return
        snapshot.phase = _PHASE_DUCKING
        snapshot.transition_started_at = time.time()
        snapshot.owned_values = {element: (value,) for element, value in snapshot.original.items()}
        snapshot.legacy_owned_elements.difference_update(snapshot.original)
        # Persist originals and this ramp's timestamp before the first
        # hardware write. A crash can then be recovered without guessing.
        self._write_recovery_journal()
        self._ramp(
            snapshot,
            snapshot.duck_target,
            duration=_LOWER_DURATION,
            steps=_LOWER_STEPS,
        )
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

    def _abandon_changed_elements(
        self,
        snapshot: _DeviceSnapshot,
        device_id: int,
    ) -> None:
        """Release only channels changed outside this ducking session."""
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
        if len(retained) == len(snapshot.original):
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
            if expected_values is not None:
                for element in list(active_elements):
                    actual = self._get_snapshot_volume(snapshot, element)
                    if abs(actual - expected_values[element]) > _OWNERSHIP_TOLERANCE:
                        active_elements.remove(element)
                        ownership_lost.add(element)
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
                self._set_snapshot_volume(snapshot, element, value)
                if expected_values is not None:
                    expected_values[element] = value
                last_values[element] = value
            self._sleep(delay)
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
            elements = self._backend.volume_elements(device_id)
        except Exception:
            logger.debug(
                "Output device UID %s is temporarily unavailable",
                snapshot.device_uid,
                exc_info=True,
            )
            return None
        if tuple(sorted(elements)) != snapshot.profile_elements:
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

    def _start_monitor(self, token: DuckToken) -> None:
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
        for attempt in range(_RESTORE_ROUTE_ATTEMPTS):
            with self._lock:
                restored, unavailable, write_failed = self._restore_pending_locked(
                    token,
                    defer_unavailable=False,
                )
            if restored:
                return True
            if write_failed:
                if write_retries >= 1:
                    return False
                write_retries += 1
            elif not unavailable:
                return False
            if attempt + 1 < _RESTORE_ROUTE_ATTEMPTS:
                self._sleep(_RESTORE_RETRY_DELAY)

        # The route may stay hidden until the next Bluetooth connection.
        # Persist its UID, release this live token, and resolve it before any
        # later session is allowed to capture a new original.
        with self._lock:
            restored, _unavailable, _write_failed = self._restore_pending_locked(
                token,
                defer_unavailable=True,
            )
            return restored

    def _restore_pending_locked(
        self,
        token: DuckToken,
        *,
        defer_unavailable: bool,
    ) -> tuple[bool, bool, bool]:
        if token is not self._active_token:
            return False, False, False

        unavailable = False
        write_failed = False
        journal_changed = False
        for snapshot in list(self._snapshots.values()):
            key = self._snapshot_key(snapshot)
            before_elements = set(snapshot.original)
            unresolved, _changed, reachable = self._restore_owned_snapshot(
                snapshot,
                ensure_original_write=True,
            )
            self._snapshots.pop(key, None)
            if unresolved is None:
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
            return False, unavailable, write_failed

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
                return False, False, True
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
        return True, False, False

    def _restore_targets(
        self,
        snapshot: _DeviceSnapshot,
        elements: list[int],
        *,
        expected_values: dict[int, float] | None = None,
    ) -> tuple[set[int], set[int]]:
        targets = {element: snapshot.original[element] for element in elements}
        try:
            ownership_lost = self._ramp(
                snapshot,
                targets,
                duration=_RESTORE_DURATION,
                steps=_RESTORE_STEPS,
                expected_values=expected_values,
            )
            return set(), ownership_lost
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
            )

    def _set_all_original(
        self,
        snapshot: _DeviceSnapshot,
        elements: list[int] | None = None,
        *,
        expected_values: dict[int, float] | None = None,
    ) -> tuple[set[int], set[int]]:
        unresolved: set[int] = set()
        ownership_lost: set[int] = set()
        if elements is None:
            elements = list(snapshot.original)
        for element in elements:
            try:
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
    ) -> tuple[_DeviceSnapshot | None, bool, bool]:
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
        if restore_elements:
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
                )
                unresolved_elements.extend(failed_elements)
                restored = bool(set(restore_elements) - failed_elements - lost_elements)

        return (
            self._snapshot_part(snapshot, unresolved_elements),
            restored or bool(already_original_elements),
            reachable,
        )

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
        if device_id is None:
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
            phase=snapshot.phase,
            transition_started_at=snapshot.transition_started_at,
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
            if data.get("version") == 1:
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
            if data.get("version") != _RECOVERY_VERSION:
                raise ValueError("unsupported recovery journal version")
            created_at = float(data["created_at"])
            if not math.isfinite(created_at):
                raise ValueError("invalid recovery timestamp")
            snapshots: dict[_SnapshotKey, _DeviceSnapshot] = {}
            for item in data["devices"]:
                device_uid = item["device_uid"]
                if not isinstance(device_uid, str) or not device_uid:
                    raise ValueError("invalid recovery device UID")
                device_id_hint = self._journal_integer(item["device_id_hint"])
                original = self._journal_volumes(item["original"])
                target = self._journal_volumes(item["target"])
                if not original or original.keys() != target.keys():
                    raise ValueError("invalid recovery volume elements")
                owned_values = None
                legacy_owned_elements: set[int]
                if "owned" in item:
                    owned_values = self._journal_owned_values(item["owned"])
                    if not set(owned_values).issubset(original):
                        raise ValueError("invalid recovery owned elements")
                    legacy_data = item.get("legacy", [])
                    if not isinstance(legacy_data, list):
                        raise ValueError("invalid legacy owned elements")
                    legacy_owned_elements = {self._journal_integer(element, allow_zero=True) for element in legacy_data}
                    if not legacy_owned_elements.issubset(original) or legacy_owned_elements.intersection(owned_values):
                        raise ValueError("invalid legacy owned elements")
                else:
                    legacy_owned_elements = set(original)
                profile_elements = tuple(self._journal_integer(element, allow_zero=True) for element in item["elements"])
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
                snapshot = _DeviceSnapshot(
                    device_uid=device_uid,
                    device_id_hint=device_id_hint,
                    profile_elements=tuple(sorted(profile_elements)),
                    original=original,
                    duck_target=target,
                    owned_values=owned_values,
                    legacy_owned_elements=legacy_owned_elements,
                    phase=phase,
                    transition_started_at=transition_started_at,
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
            return True, None

        complete_profile = set(snapshot.original) == set(elements)
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
                still_at_target = exact_route and all(
                    abs(self._backend.get_volume(device_id, element) - snapshot.duck_target[element]) <= _OWNERSHIP_TOLERANCE
                    for element in elements
                )
            except Exception:
                still_at_target = False
            if still_at_target:
                snapshot.device_id_hint = device_id
                return True, snapshot

        # A restored or user-modified route must finish the old ownership
        # classification before this session captures a fresh original.
        return self._resolve_deferred_route(device_uid, elements), None

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

        unresolved, _changed, _reachable = self._restore_owned_snapshot(snapshot)
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
    def _journal_integer(value, *, allow_zero: bool = False) -> int:
        if isinstance(value, bool):
            raise ValueError("invalid recovery integer")
        if isinstance(value, str):
            if not value.isascii() or not value.isdecimal():
                raise ValueError("invalid recovery integer")
            result = int(value)
        elif isinstance(value, int):
            result = value
        else:
            raise ValueError("invalid recovery integer")
        minimum = 0 if allow_zero else 1
        if result < minimum:
            raise ValueError("invalid recovery integer")
        return result

    @classmethod
    def _journal_volumes(cls, values) -> dict[int, float]:
        if not isinstance(values, dict):
            raise ValueError("invalid recovery volumes")
        result: dict[int, float] = {}
        for element, volume in values.items():
            element_id = cls._journal_integer(element, allow_zero=True)
            scalar = float(volume)
            if isinstance(volume, bool) or not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
                raise ValueError("invalid recovery volume")
            result[element_id] = scalar
        return result

    @classmethod
    def _journal_owned_values(
        cls,
        values,
    ) -> dict[int, tuple[float, ...]]:
        if not isinstance(values, dict):
            raise ValueError("invalid recovery owned values")
        result: dict[int, tuple[float, ...]] = {}
        for element, candidates in values.items():
            element_id = cls._journal_integer(element, allow_zero=True)
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
            if snapshot.legacy_owned_elements:
                item["legacy"] = sorted(snapshot.legacy_owned_elements)
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

    def _rollback_and_clear(self) -> None:
        if self._monitor_stop is not None:
            self._monitor_stop.set()
        for snapshot in list(self._snapshots.values()):
            key = self._snapshot_key(snapshot)
            rolled_back = self._rollback_failed_duck(snapshot)
            self._snapshots.pop(key, None)
            if not rolled_back:
                self._insert_snapshot(self._snapshots, snapshot)
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
