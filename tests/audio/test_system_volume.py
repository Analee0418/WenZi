"""Tests for system output volume ducking without real CoreAudio access."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import pytest

import wenzi.audio.system_volume as system_volume
from wenzi.audio.system_volume import (
    DuckToken,
    SystemOutputDucker,
    _CoreAudioBackend,
)


def _void_pointer_value(value) -> int | None:
    if isinstance(value, ctypes.c_void_p):
        return value.value
    return value


class _FakeCreatedUidCoreFoundation:
    def __init__(self, uid_ref: int = 0x12345678) -> None:
        self.uid_ref = uid_ref
        self.created: list[tuple[bytes, int]] = []
        self.released: list[int | None] = []

    def CFStringCreateWithCString(
        self,
        _allocator,
        value: bytes,
        encoding: int,
    ) -> int:
        self.created.append((value, encoding))
        return self.uid_ref

    def CFRelease(self, value) -> None:
        self.released.append(_void_pointer_value(value))


class _FakeUidTranslationCoreAudio:
    def __init__(self, *, status: int = 0, device_id: int = 106) -> None:
        self.status = status
        self.device_id = device_id
        self.call: dict[str, int] | None = None

    def AudioObjectGetPropertyData(
        self,
        object_id: int,
        address_pointer,
        qualifier_size: int,
        qualifier_pointer,
        size_pointer,
        output_pointer,
    ) -> int:
        address = ctypes.cast(
            address_pointer,
            ctypes.POINTER(system_volume._PropertyAddress),
        ).contents
        qualifier_ref = ctypes.cast(
            qualifier_pointer,
            ctypes.POINTER(ctypes.c_void_p),
        ).contents.value
        output_size = ctypes.cast(
            size_pointer,
            ctypes.POINTER(ctypes.c_uint32),
        ).contents.value
        self.call = {
            "object_id": object_id,
            "selector": address.mSelector,
            "scope": address.mScope,
            "element": address.mElement,
            "qualifier_size": qualifier_size,
            "qualifier_ref": qualifier_ref,
            "output_size": output_size,
        }
        ctypes.cast(
            output_pointer,
            ctypes.POINTER(ctypes.c_uint32),
        ).contents.value = self.device_id
        return self.status


class FakeVolumeBackend:
    def __init__(self) -> None:
        self.default_device: int | None = 1
        self.elements: dict[int, tuple[int, ...]] = {1: (0,)}
        self.values: dict[tuple[int, int], float] = {(1, 0): 0.8}
        self.device_uids: dict[int, str] = {1: "uid-1"}
        self.reads: list[tuple] = []
        self.writes: list[tuple[int, int, float]] = []
        self.fail_writes: dict[tuple[int, int], int] = defaultdict(int)

    def default_output_device(self) -> int | None:
        self.reads.append(("default_output_device",))
        return self.default_device

    def device_uid(self, device_id: int) -> str | None:
        self.reads.append(("device_uid", device_id))
        if device_id in self.device_uids:
            return self.device_uids[device_id]
        if device_id in self.elements:
            return f"uid-{device_id}"
        return None

    def device_id_for_uid(self, device_uid: str) -> int | None:
        self.reads.append(("device_id_for_uid", device_uid))
        for device_id, known_uid in self.device_uids.items():
            if known_uid == device_uid and device_id in self.elements:
                return device_id
        if device_uid.startswith("uid-"):
            suffix = device_uid.removeprefix("uid-")
            if suffix.isdecimal() and int(suffix) in self.elements:
                return int(suffix)
        return None

    def volume_elements(self, device_id: int) -> tuple[int, ...]:
        self.reads.append(("volume_elements", device_id))
        return self.elements.get(device_id, ())

    def get_volume(self, device_id: int, element: int) -> float:
        self.reads.append(("get_volume", device_id, element))
        return self.values[(device_id, element)]

    def set_volume(self, device_id: int, element: int, value: float) -> None:
        key = (device_id, element)
        if self.fail_writes[key]:
            self.fail_writes[key] -= 1
            raise RuntimeError("injected write failure")
        self.values[key] = value
        self.writes.append((device_id, element, value))


@pytest.mark.parametrize("conversion_succeeds", [True, False])
def test_coreaudio_device_uid_always_releases_returned_cfstring(
    conversion_succeeds: bool,
) -> None:
    uid_ref = 0x10203040

    class FakeCoreAudio:
        def AudioObjectGetPropertyData(
            self,
            _object_id,
            address_pointer,
            _qualifier_size,
            _qualifier_pointer,
            _size_pointer,
            output_pointer,
        ) -> int:
            address = ctypes.cast(
                address_pointer,
                ctypes.POINTER(system_volume._PropertyAddress),
            ).contents
            assert address.mSelector == system_volume._DEVICE_UID
            ctypes.cast(
                output_pointer,
                ctypes.POINTER(ctypes.c_void_p),
            ).contents.value = uid_ref
            return 0

    class FakeCoreFoundation:
        def __init__(self) -> None:
            self.released: list[int | None] = []

        def CFStringGetLength(self, value) -> int:
            assert _void_pointer_value(value) == uid_ref
            return len("airpods-output")

        def CFStringGetMaximumSizeForEncoding(
            self,
            _length: int,
            _encoding: int,
        ) -> int:
            return 64

        def CFStringGetCString(
            self,
            value,
            buffer,
            buffer_size: int,
            _encoding: int,
        ) -> int:
            assert _void_pointer_value(value) == uid_ref
            assert buffer_size >= len(b"airpods-output") + 1
            if not conversion_succeeds:
                return 0
            buffer.value = b"airpods-output"
            return 1

        def CFRelease(self, value) -> None:
            self.released.append(_void_pointer_value(value))

    backend = _CoreAudioBackend()
    backend._ca = FakeCoreAudio()
    fake_cf = FakeCoreFoundation()
    backend._cf = fake_cf

    if conversion_succeeds:
        assert backend.device_uid(96) == "airpods-output"
    else:
        with pytest.raises(RuntimeError, match="not valid UTF-8"):
            backend.device_uid(96)

    assert fake_cf.released == [uid_ref]


def test_coreaudio_uid_translation_uses_cfstring_qualifier() -> None:
    fake_ca = _FakeUidTranslationCoreAudio(device_id=106)
    fake_cf = _FakeCreatedUidCoreFoundation()
    backend = _CoreAudioBackend()
    backend._ca = fake_ca
    backend._cf = fake_cf

    assert backend.device_id_for_uid("airpods-output") == 106

    assert fake_cf.created == [(b"airpods-output", system_volume._CF_STRING_ENCODING_UTF8)]
    assert fake_ca.call == {
        "object_id": system_volume._SYSTEM_OBJECT,
        "selector": system_volume._TRANSLATE_UID_TO_DEVICE,
        "scope": system_volume._SCOPE_GLOBAL,
        "element": system_volume._MAIN_ELEMENT,
        "qualifier_size": ctypes.sizeof(ctypes.c_void_p),
        "qualifier_ref": fake_cf.uid_ref,
        "output_size": ctypes.sizeof(ctypes.c_uint32),
    }
    assert fake_cf.released == [fake_cf.uid_ref]


@pytest.mark.parametrize(
    ("status", "translated_id", "raises"),
    [
        (0, 0, False),
        (-50, 106, True),
    ],
)
def test_coreaudio_uid_translation_handles_unknown_and_error_with_release(
    status: int,
    translated_id: int,
    raises: bool,
) -> None:
    fake_ca = _FakeUidTranslationCoreAudio(
        status=status,
        device_id=translated_id,
    )
    fake_cf = _FakeCreatedUidCoreFoundation()
    backend = _CoreAudioBackend()
    backend._ca = fake_ca
    backend._cf = fake_cf

    if raises:
        with pytest.raises(system_volume._CoreAudioError) as error:
            backend.device_id_for_uid("airpods-output")
        assert error.value.status == status
    else:
        assert backend.device_id_for_uid("airpods-output") is None

    assert fake_cf.released == [fake_cf.uid_ref]


def _write_recovery_journal(
    path: Path,
    *,
    current_time: float | None = None,
    transition_started_at: float | None = None,
    device_uid: str = "uid-1",
    device_id_hint: int = 1,
    phase: str = "ducked",
    original: float = 0.8,
    target: float = 0.05,
) -> None:
    created_at = time.time() if current_time is None else current_time
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": device_uid,
                        "device_id_hint": device_id_hint,
                        "elements": [0],
                        "original": {"0": original},
                        "target": {"0": target},
                        "phase": phase,
                        "transition_started_at": (created_at if transition_started_at is None else transition_started_at),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def backend() -> FakeVolumeBackend:
    return FakeVolumeBackend()


@pytest.fixture
def ducker(backend: FakeVolumeBackend) -> SystemOutputDucker:
    return SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )


def test_master_volume_is_lowered_and_restored(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin()

    assert isinstance(token, DuckToken)
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_low_volume_is_attenuated_without_being_raised(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.values[(1, 0)] = 0.04

    token = ducker.begin()

    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.01)
    assert ducker.end(token)


def test_channel_fallback_preserves_balance_and_restores(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.8, (1, 2): 0.4}

    token = ducker.begin()

    assert token is not None
    assert backend.values[(1, 1)] == pytest.approx(0.05)
    assert backend.values[(1, 2)] == pytest.approx(0.025)
    assert ducker.end(token)
    assert backend.values[(1, 1)] == pytest.approx(0.8)
    assert backend.values[(1, 2)] == pytest.approx(0.4)


def test_unsupported_output_is_ignored(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.elements[1] = ()

    assert ducker.begin() is None
    assert backend.writes == []
    assert not ducker.restore_all()


def test_refresh_ducks_each_new_default_route(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    token = ducker.begin()
    assert token is not None

    backend.default_device = 2
    assert ducker.refresh(token)

    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert backend.values[(2, 0)] == pytest.approx(0.05)
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.values[(2, 0)] == pytest.approx(0.6)


def test_refresh_is_idempotent_for_an_existing_route(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin()
    assert token is not None
    write_count = len(backend.writes)

    assert ducker.refresh(token)
    assert len(backend.writes) == write_count
    assert ducker.end(token)


def test_same_device_profile_change_is_reducked(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin()
    assert token is not None
    backend.values[(1, 0)] = 0.65

    assert ducker.refresh(token)
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_same_device_channel_topologies_keep_independent_originals(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin()
    assert token is not None

    backend.elements[1] = (1, 2)
    backend.values[(1, 1)] = 0.7
    backend.values[(1, 2)] = 0.5
    assert ducker.refresh(token)
    assert backend.values[(1, 1)] == pytest.approx(0.05)
    assert backend.values[(1, 2)] == pytest.approx(0.05 * 0.5 / 0.7)
    assert ducker.end(token)
    assert backend.values[(1, 1)] == pytest.approx(0.7)
    assert backend.values[(1, 2)] == pytest.approx(0.5)
    assert backend.values[(1, 0)] == pytest.approx(0.05)

    backend.elements[1] = (0,)
    second = ducker.begin()
    assert second is not None
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert ducker.end(second)
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_deferred_profile_does_not_block_an_independent_topology(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [1, 2],
                        "original": {"1": 0.8, "2": 0.4},
                        "target": {"1": 0.05, "2": 0.025},
                        "phase": "ducked",
                        "transition_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    token = ducker.begin()

    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert {tuple(item["elements"]) for item in journal["devices"]} == {
        (0,),
        (1, 2),
    }
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_stale_and_duplicate_tokens_are_harmless(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    first = ducker.begin()
    assert first is not None
    assert ducker.end(first)

    second = ducker.begin()
    assert second is not None
    write_count = len(backend.writes)

    assert not ducker.refresh(first)
    assert not ducker.end(first)
    assert len(backend.writes) == write_count
    assert ducker.end(second)


def test_begin_rejects_a_second_active_session(
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin()

    assert token is not None
    assert ducker.begin() is None
    assert ducker.end(token)


def test_partial_channel_failure_rolls_back_before_returning_none(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.8, (1, 2): 0.4}
    backend.fail_writes[(1, 2)] = 1

    assert ducker.begin() is None
    assert backend.values[(1, 1)] == pytest.approx(0.8)
    assert backend.values[(1, 2)] == pytest.approx(0.4)
    assert not ducker.restore_all()


def test_begin_clears_token_when_failed_channel_was_already_original(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.8, (1, 2): 0.4}
    backend.fail_writes[(1, 2)] = 2

    token = ducker.begin()

    assert token is None
    assert backend.values[(1, 1)] == pytest.approx(0.8)
    assert backend.values[(1, 2)] == pytest.approx(0.4)
    assert not ducker.restore_all()


def test_failed_refresh_keeps_the_original_session_active() -> None:
    class LaterFailureBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.second_channel_writes = 0

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            if (device_id, element) == (2, 2):
                self.second_channel_writes += 1
                if self.second_channel_writes == 4:
                    raise RuntimeError("injected later write failure")
            super().set_volume(device_id, element, value)

    backend = LaterFailureBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin()
    assert token is not None
    backend.default_device = 2
    backend.elements[2] = (1, 2)
    backend.values[(2, 1)] = 0.7
    backend.values[(2, 2)] = 0.6

    assert not ducker.refresh(token)
    assert backend.values[(2, 1)] == pytest.approx(0.7)
    assert backend.values[(2, 2)] == pytest.approx(0.6)
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_restore_all_is_an_idempotent_shutdown_fallback(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    assert ducker.begin() is not None

    assert ducker.restore_all()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not ducker.restore_all()


def test_concurrent_end_restores_exactly_once(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin()
    assert token is not None
    before_restore = len(backend.writes)
    results: list[bool] = []

    threads = [threading.Thread(target=lambda: results.append(ducker.end(token))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [False, False, False, True]
    restored_values = [value for _device, _element, value in backend.writes[before_restore:]]
    assert all(current > previous for previous, current in zip(restored_values, restored_values[1:]))
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_delayed_coreaudio_write_cannot_strand_ducked_volume() -> None:
    class DelayedWriteBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.pending: list[tuple[int, int, float]] = []

        def set_volume(self, device_id: int, element: int, value: float) -> None:
            self.pending.append((device_id, element, value))
            self.writes.append((device_id, element, value))

        def settle(self) -> None:
            for device_id, element, value in self.pending:
                self.values[(device_id, element)] = value
            self.pending.clear()

    backend = DelayedWriteBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )

    token = ducker.begin()
    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.8)

    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    backend.settle()
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_restore_failure_keeps_snapshot_for_restore_all_retry(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.8, (1, 2): 0.4}
    token = ducker.begin()
    assert token is not None
    backend.fail_writes[(1, 2)] = 4

    assert not ducker.end(token)
    assert backend.values[(1, 1)] == pytest.approx(0.8)
    assert backend.values[(1, 2)] == pytest.approx(0.025)

    assert ducker.restore_all()
    assert backend.values[(1, 1)] == pytest.approx(0.8)
    assert backend.values[(1, 2)] == pytest.approx(0.4)


def test_end_retries_a_transient_restore_failure(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin()
    assert token is not None
    backend.fail_writes[(1, 0)] = 2

    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not ducker.restore_all()


def test_monitor_ducks_a_route_change_without_explicit_refresh() -> None:
    backend = FakeVolumeBackend()
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    ducker = SystemOutputDucker(backend, sleeper=lambda _delay: None)
    token = ducker.begin()
    assert token is not None

    backend.default_device = 2
    deadline = time.monotonic() + 1.0
    while backend.values[(2, 0)] != pytest.approx(0.05):
        if time.monotonic() >= deadline:
            raise AssertionError("monitor did not duck the changed route")
        time.sleep(0.005)

    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.values[(2, 0)] == pytest.approx(0.6)
    assert not any(thread.name == "system-volume-monitor" and thread.is_alive() for thread in threading.enumerate())


def test_monitor_reducks_a_recreated_device_with_the_same_uid_and_topology() -> None:
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(backend, sleeper=lambda _delay: None)
    token = ducker.begin()
    assert token is not None

    backend.default_device = 2
    backend.elements = {2: (0,)}
    backend.values = {(2, 0): 0.65}
    backend.device_uids = {2: "uid-1"}
    deadline = time.monotonic() + 1.0
    while backend.values[(2, 0)] != pytest.approx(0.05):
        if time.monotonic() >= deadline:
            raise AssertionError("monitor did not re-duck the recreated route")
        time.sleep(0.005)

    assert ducker.end(token)
    assert backend.values[(2, 0)] == pytest.approx(0.8)


def test_immediate_manual_volume_change_is_not_reducked_or_restored() -> None:
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(backend, sleeper=lambda _delay: None)
    token = ducker.begin()
    assert token is not None

    backend.values[(1, 0)] = 0.55
    deadline = time.monotonic() + 1.0
    while "uid-1" not in ducker._overridden_devices:
        if time.monotonic() >= deadline:
            raise AssertionError("monitor did not recognize the user override")
        time.sleep(0.005)
    write_count = len(backend.writes)

    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.55)
    assert len(backend.writes) == write_count


def test_monitor_releases_only_the_user_changed_channel() -> None:
    backend = FakeVolumeBackend()
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.8, (1, 2): 0.4}
    ducker = SystemOutputDucker(backend, sleeper=lambda _delay: None)
    token = ducker.begin()
    assert token is not None

    backend.values[(1, 1)] = 0.3
    deadline = time.monotonic() + 1.0
    while set(ducker._snapshots[("uid-1", (1, 2))].original) != {2}:
        if time.monotonic() >= deadline:
            raise AssertionError("monitor did not release the changed channel")
        time.sleep(0.005)

    assert ducker.end(token)
    assert backend.values[(1, 1)] == pytest.approx(0.3)
    assert backend.values[(1, 2)] == pytest.approx(0.4)


def test_restore_continues_other_channels_after_a_mid_ramp_user_change(
    tmp_path: Path,
) -> None:
    class MidRestoreChangeBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.inject_change = False
            self.changed = False

        def set_volume(self, device_id: int, element: int, value: float) -> None:
            super().set_volume(device_id, element, value)
            if self.inject_change and not self.changed and element == 1:
                self.values[(device_id, element)] = 0.3
                self.changed = True

    journal_path = tmp_path / "system-volume.json"
    backend = MidRestoreChangeBackend()
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.8, (1, 2): 0.4}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin()
    assert token is not None
    backend.inject_change = True

    assert ducker.end(token)
    assert backend.values[(1, 1)] == pytest.approx(0.3)
    assert backend.values[(1, 2)] == pytest.approx(0.4)
    assert not journal_path.exists()


def test_end_waits_for_inflight_monitor_before_restoring() -> None:
    class BlockingBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.block_monitor = False
            self.monitor_entered = threading.Event()
            self.monitor_gate = threading.Event()

        def default_output_device(self) -> int | None:
            if self.block_monitor and threading.current_thread().name == "system-volume-monitor":
                self.monitor_entered.set()
                self.monitor_gate.wait(timeout=2.0)
            return super().default_output_device()

    backend = BlockingBackend()
    ducker = SystemOutputDucker(backend, sleeper=lambda _delay: None)
    token = ducker.begin()
    assert token is not None
    backend.block_monitor = True
    assert backend.monitor_entered.wait(timeout=1.0)

    result: list[bool] = []
    finished = threading.Event()

    def _end() -> None:
        result.append(ducker.end(token))
        finished.set()

    thread = threading.Thread(target=_end)
    thread.start()
    assert not finished.wait(timeout=0.05)
    backend.monitor_gate.set()
    thread.join(timeout=2.0)

    assert result == [True]
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not any(item.name == "system-volume-monitor" and item.is_alive() for item in threading.enumerate())


def test_end_does_not_touch_a_new_uid_that_appears_during_settle() -> None:
    backend = FakeVolumeBackend()
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    switched = False

    def _sleep(delay: float) -> None:
        nonlocal switched
        if delay == pytest.approx(0.10) and not switched:
            switched = True
            backend.default_device = 2

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin()
    assert token is not None

    assert ducker.end(token)
    assert switched
    assert not any(device_id == 2 for device_id, _element, _value in backend.writes)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.values[(2, 0)] == pytest.approx(0.6)


def test_monitor_join_timeout_still_restores_and_allows_next_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(system_volume, "_MONITOR_JOIN_TIMEOUT", 0.01)
    backend = FakeVolumeBackend()
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    monitor_entered = threading.Event()
    monitor_gate = threading.Event()

    def _waiter(stop: threading.Event, _timeout: float) -> bool:
        monitor_entered.set()
        monitor_gate.wait(timeout=2.0)
        return stop.is_set()

    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=_waiter,
    )
    token = ducker.begin()
    assert token is not None
    assert monitor_entered.wait(timeout=1.0)

    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)

    monitor_gate.set()
    backend.default_device = 2
    second = ducker.begin()
    assert second is not None
    assert ducker.end(second)
    assert backend.values[(2, 0)] == pytest.approx(0.6)


def test_recovery_journal_is_written_before_duck_and_deleted_after_restore(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    phases_seen_during_writes: list[str] = []

    class JournalCheckingBackend(FakeVolumeBackend):
        def set_volume(self, device_id: int, element: int, value: float) -> None:
            assert journal_path.exists()
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            matching = [item for item in journal["devices"] if item["device_uid"] == self.device_uid(device_id)]
            assert len(matching) == 1
            candidates = matching[0]["owned"][str(element)]
            current = self.values[(device_id, element)]
            assert any(current == pytest.approx(candidate) for candidate in candidates)
            assert any(value == pytest.approx(candidate) for candidate in candidates)
            phases_seen_during_writes.append(matching[0]["phase"])
            super().set_volume(device_id, element, value)

    backend = JournalCheckingBackend()
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    token = ducker.begin()

    assert token is not None
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["version"] == 2
    assert len(journal["devices"]) == 1
    first = journal["devices"][0]
    assert first["device_uid"] == "uid-1"
    assert first["device_id_hint"] == 1
    assert first["elements"] == [0]
    assert first["original"] == {"0": 0.8}
    assert first["target"] == {"0": pytest.approx(0.05)}
    assert first["phase"] == "ducked"
    assert isinstance(first["transition_started_at"], float)
    assert "ducking" in phases_seen_during_writes
    backend.default_device = 2
    assert ducker.refresh(token)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert {item["device_uid"] for item in journal["devices"]} == {
        "uid-1",
        "uid-2",
    }
    assert ducker.end(token)
    assert "restoring" in phases_seen_during_writes
    assert not journal_path.exists()


def test_recover_stale_restores_owned_target(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 24 * 60 * 60,
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_recover_stale_resolves_same_uid_after_device_id_changes(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
        device_uid="airpods-uid",
        device_id_hint=96,
    )
    backend = FakeVolumeBackend()
    backend.default_device = 106
    backend.elements = {106: (0,)}
    backend.values = {(106, 0): 0.05}
    backend.device_uids = {106: "airpods-uid"}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.values[(106, 0)] == pytest.approx(0.8)
    assert backend.writes
    assert {device_id for device_id, _element, _value in backend.writes} == {106}
    assert not journal_path.exists()


def test_live_restore_waits_for_same_uid_to_reappear_with_a_new_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(system_volume, "_RESTORE_ROUTE_ATTEMPTS", 8)
    backend = FakeVolumeBackend()
    backend.default_device = 96
    backend.elements = {96: (0,)}
    backend.values = {(96, 0): 0.8}
    backend.device_uids = {96: "airpods-output"}
    retry_count = 0

    def _sleep(delay: float) -> None:
        nonlocal retry_count
        if delay != pytest.approx(system_volume._RESTORE_RETRY_DELAY):
            return
        retry_count += 1
        if retry_count == 3:
            backend.elements = {106: (0,)}
            backend.values[(106, 0)] = backend.values[(96, 0)]
            backend.device_uids = {106: "airpods-output"}
            backend.default_device = 106

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin()
    assert token is not None
    write_count = len(backend.writes)
    backend.elements = {}
    backend.device_uids = {}
    backend.default_device = None

    assert ducker.end(token)
    restore_writes = backend.writes[write_count:]
    assert retry_count == 3
    assert restore_writes
    assert {device_id for device_id, _element, _value in restore_writes} == {106}
    assert backend.values[(106, 0)] == pytest.approx(0.8)


def test_unavailable_route_polling_persists_only_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(system_volume, "_RESTORE_ROUTE_ATTEMPTS", 6)
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=tmp_path / "system-volume.json",
    )
    write_count = 0
    original_write = ducker._write_recovery_snapshots

    def _counted_write(*args, **kwargs) -> None:
        nonlocal write_count
        write_count += 1
        original_write(*args, **kwargs)

    monkeypatch.setattr(ducker, "_write_recovery_snapshots", _counted_write)
    token = ducker.begin()
    assert token is not None
    writes_after_begin = write_count
    backend.elements = {}
    backend.device_uids = {}
    backend.default_device = None

    assert ducker.end(token)
    assert write_count - writes_after_begin == 1


def test_live_restore_never_writes_to_a_reused_device_id() -> None:
    backend = FakeVolumeBackend()
    backend.default_device = 96
    backend.elements = {96: (0,)}
    backend.values = {(96, 0): 0.8}
    backend.device_uids = {96: "airpods-output"}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin()
    assert token is not None
    write_count = len(backend.writes)

    backend.elements = {96: (0,), 106: (0,)}
    backend.values[(96, 0)] = 0.3
    backend.values[(106, 0)] = 0.05
    backend.device_uids = {
        96: "unrelated-output",
        106: "airpods-output",
    }
    backend.default_device = 96

    assert ducker.end(token)
    restore_writes = backend.writes[write_count:]
    assert restore_writes
    assert {device_id for device_id, _element, _value in restore_writes} == {106}
    assert backend.values[(96, 0)] == pytest.approx(0.3)
    assert backend.values[(106, 0)] == pytest.approx(0.8)


def test_recover_stale_never_writes_to_reused_id_with_a_different_uid(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
        device_uid="old-device-uid",
        device_id_hint=96,
    )
    backend = FakeVolumeBackend()
    backend.default_device = 96
    backend.elements = {96: (0,)}
    backend.values = {(96, 0): 0.05}
    backend.device_uids = {96: "replacement-device-uid"}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale()
    assert backend.values[(96, 0)] == pytest.approx(0.05)
    assert backend.writes == []
    assert journal_path.exists()


def test_recover_stale_does_not_restore_old_original_to_unrelated_default(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
        device_uid="old-device-uid",
        device_id_hint=96,
    )
    backend = FakeVolumeBackend()
    backend.default_device = 200
    backend.elements = {106: (0,), 200: (0,)}
    backend.values = {(106, 0): 0.05, (200, 0): 0.3}
    backend.device_uids = {
        106: "old-device-uid",
        200: "unrelated-default-uid",
    }
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.values[(106, 0)] == pytest.approx(0.8)
    assert backend.values[(200, 0)] == pytest.approx(0.3)
    assert all(device_id == 106 for device_id, _element, _value in backend.writes)


def test_same_uid_deferred_recovery_is_not_compound_ducked_next_session(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    token = ducker.begin()

    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert backend.writes == []
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    active = next(item for item in journal["devices"] if item["device_uid"] == "uid-1")
    assert active["original"] == {"0": pytest.approx(0.8)}
    assert active["target"] == {"0": pytest.approx(0.05)}
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_deferred_target_is_adopted_after_coreaudio_reuses_the_device_id(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
        device_uid="airpods-output",
        device_id_hint=96,
    )
    backend = FakeVolumeBackend()
    backend.default_device = 106
    backend.elements = {106: (0,)}
    backend.values = {(106, 0): 0.05}
    backend.device_uids = {106: "airpods-output"}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    token = ducker.begin(factor=0.5, max_volume=0.2)

    assert token is not None
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert backend.writes == []
    assert ducker.end(token)
    assert backend.values[(106, 0)] == pytest.approx(0.8)
    assert {device_id for device_id, _element, _value in backend.writes} == {106}


def test_deferred_target_without_durable_ownership_is_not_reclaimed(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [0],
                        "original": {"0": 0.8},
                        "target": {"0": 0.05},
                        "owned": {},
                        "phase": "ducked",
                        "transition_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    token = ducker.begin()

    assert token is not None
    assert max(value for _device, _element, value in backend.writes) <= 0.05
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.05)


def test_refresh_adopts_an_unchanged_deferred_route_without_a_volume_pulse(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
        device_uid="uid-2",
        device_id_hint=2,
        original=0.6,
        target=0.04,
    )
    backend = FakeVolumeBackend()
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.04
    backend.device_uids[2] = "uid-2"
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin()
    assert token is not None
    backend.writes.clear()

    backend.default_device = 2
    assert ducker.refresh(token)

    assert backend.values[(2, 0)] == pytest.approx(0.04)
    assert backend.writes == []
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.values[(2, 0)] == pytest.approx(0.6)


def test_unreadable_deferred_target_is_not_adopted_or_reducked(
    tmp_path: Path,
) -> None:
    class UnreadableBackend(FakeVolumeBackend):
        def get_volume(self, device_id: int, element: int) -> float:
            raise RuntimeError("injected read failure")

    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
    )
    backend = UnreadableBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    assert ducker.begin() is None
    assert backend.writes == []
    assert journal_path.exists()


def test_monitor_start_failure_rolls_back_an_adopted_deferred_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    def fail_monitor(_token: DuckToken) -> None:
        raise RuntimeError("injected monitor start failure")

    monkeypatch.setattr(ducker, "_start_monitor", fail_monitor)

    assert ducker.begin() is None
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()
    monkeypatch.setattr(ducker, "_start_monitor", lambda _token: None)
    next_token = ducker.begin()
    assert next_token is not None
    assert ducker.end(next_token)


def test_recover_stale_defers_same_uid_with_mismatched_channel_topology(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [1, 2],
                        "original": {"1": 0.8, "2": 0.4},
                        "target": {"1": 0.05, "2": 0.025},
                        "phase": "ducked",
                        "transition_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    backend.elements[1] = (0,)
    backend.values = {(1, 0): 0.05}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale()
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert backend.writes == []
    assert journal_path.exists()
    deferred = json.loads(journal_path.read_text(encoding="utf-8"))
    assert deferred["devices"][0]["original"] == {"1": 0.8, "2": 0.4}


def test_v1_recovery_is_discarded_when_current_volume_is_not_capped(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_id": 1,
                        "original": {"0": 0.8},
                        "target": {"0": 0.05},
                        "duck_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale()
    assert backend.writes == []
    assert not journal_path.exists()


def test_v1_target_blocks_reduck_until_the_user_changes_volume(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_id": 96,
                        "original": {"0": 0.8},
                        "target": {"0": 0.05},
                        "duck_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale()
    assert ducker.begin() is None
    assert backend.writes == []
    assert journal_path.exists()

    backend.values[(1, 0)] = 0.2
    token = ducker.begin()
    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.2)
    assert not journal_path.exists()


def test_two_duckers_with_same_journal_hold_an_exclusive_lease(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    first = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    second = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    first_token = first.begin()
    assert first_token is not None
    write_count = len(backend.writes)

    assert second.begin() is None
    assert len(backend.writes) == write_count
    assert backend.values[(1, 0)] == pytest.approx(0.05)

    assert first.end(first_token)
    second_token = second.begin()
    assert second_token is not None
    assert second.end(second_token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_ducker_lease_is_exclusive_across_processes(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    lock_path = journal_path.with_name(f"{journal_path.name}.lock")
    child_code = """
import fcntl
import os
import sys

fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("locked", flush=True)
sys.stdin.read()
os.close(fd)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        backend = FakeVolumeBackend()
        ducker = SystemOutputDucker(
            backend,
            sleeper=lambda _delay: None,
            recovery_path=journal_path,
        )

        assert ducker.begin() is None
        assert backend.writes == []
    finally:
        if child.stdin is not None:
            child.stdin.close()
        child.wait(timeout=2.0)


def test_coreaudio_uid_round_trip_when_device_is_available() -> None:
    backend = _CoreAudioBackend()
    try:
        device_id = backend.default_output_device()
    except Exception as exc:
        pytest.skip(f"CoreAudio is unavailable: {exc}")
    if device_id is None:
        pytest.skip("No CoreAudio default output is visible")

    device_uid = backend.device_uid(device_id)
    assert device_uid
    resolved_id = backend.device_id_for_uid(device_uid)
    assert resolved_id is not None
    assert backend.device_uid(resolved_id) == device_uid


@pytest.mark.parametrize("user_volume", [0.06, 0.45])
def test_recover_stale_does_not_overwrite_user_volume(
    tmp_path: Path,
    user_volume: float,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = user_volume
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale()
    assert backend.values[(1, 0)] == pytest.approx(user_volume)
    assert backend.writes == []
    assert not journal_path.exists()


@pytest.mark.parametrize("phase", ["ducking", "restoring"])
def test_recover_stale_accepts_a_fresh_incomplete_volume_transition(
    tmp_path: Path,
    phase: str,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(journal_path, phase=phase)
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.45
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_recover_stale_uses_each_route_duck_timestamp(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    now = time.time()
    _write_recovery_journal(
        journal_path,
        current_time=now - 60.0,
        transition_started_at=now,
        phase="ducking",
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.45
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_stale_ramp_recovers_only_an_exact_durable_owned_value(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [0],
                        "original": {"0": 0.8},
                        "target": {"0": 0.05},
                        "owned": {"0": [0.065, 0.071]},
                        "phase": "restoring",
                        "transition_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.071
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_stale_ramp_does_not_reclaim_a_non_owned_user_value(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [0],
                        "original": {"0": 0.8},
                        "target": {"0": 0.05},
                        "owned": {"0": [0.065, 0.071]},
                        "phase": "restoring",
                        "transition_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.09
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale()
    assert backend.values[(1, 0)] == pytest.approx(0.09)
    assert backend.writes == []
    assert not journal_path.exists()


def test_legacy_v2_partial_read_keeps_unreadable_channel_recoverable(
    tmp_path: Path,
) -> None:
    class OneReadFailureBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.fail_second_channel_once = True

        def get_volume(self, device_id: int, element: int) -> float:
            if element == 2 and self.fail_second_channel_once:
                self.fail_second_channel_once = False
                raise RuntimeError("injected transient read failure")
            return super().get_volume(device_id, element)

    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [1, 2],
                        "original": {"1": 0.8, "2": 0.4},
                        "target": {"1": 0.05, "2": 0.025},
                        "phase": "ducked",
                        "transition_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = OneReadFailureBackend()
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.05, (1, 2): 0.025}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.values[(1, 1)] == pytest.approx(0.8)
    assert backend.values[(1, 2)] == pytest.approx(0.025)
    pending = json.loads(journal_path.read_text(encoding="utf-8"))
    assert pending["devices"][0]["owned"] == {}
    assert pending["devices"][0]["legacy"] == [2]

    assert ducker.recover_stale()
    assert backend.values[(1, 2)] == pytest.approx(0.4)
    assert not journal_path.exists()


def test_restore_accepts_a_durable_next_value_when_hal_writes_then_raises(
    tmp_path: Path,
) -> None:
    class WriteThenRaiseBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.raise_after_write = False
            self.raised = False

        def set_volume(self, device_id: int, element: int, value: float) -> None:
            super().set_volume(device_id, element, value)
            if self.raise_after_write and not self.raised:
                self.raised = True
                raise RuntimeError("injected post-write HAL failure")

    journal_path = tmp_path / "system-volume.json"
    backend = WriteThenRaiseBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin()
    assert token is not None
    backend.raise_after_write = True

    assert ducker.end(token)
    assert backend.raised
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_recover_stale_handles_channels_independently(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [1, 2],
                        "original": {"1": 0.8, "2": 0.4},
                        "target": {"1": 0.12, "2": 0.06},
                        "phase": "ducked",
                        "transition_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.8, (1, 2): 0.06}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.values[(1, 1)] == pytest.approx(0.8)
    assert backend.values[(1, 2)] == pytest.approx(0.4)
    assert {element for _device, element, _value in backend.writes} == {2}
    assert not journal_path.exists()


def test_restore_journal_drops_user_owned_channel_before_first_write(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": time.time() - 30.0,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [1, 2],
                        "original": {"1": 0.8, "2": 0.4},
                        "target": {"1": 0.05, "2": 0.025},
                        "phase": "ducked",
                        "transition_started_at": time.time() - 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class InspectingBackend(FakeVolumeBackend):
        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            assert journal["devices"][0]["phase"] == "restoring"
            assert journal["devices"][0]["original"] == {"1": 0.8}
            super().set_volume(device_id, element, value)

    backend = InspectingBackend()
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.05, (1, 2): 0.2}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.values[(1, 1)] == pytest.approx(0.8)
    assert backend.values[(1, 2)] == pytest.approx(0.2)
    assert {element for _device, element, _value in backend.writes} == {1}
    assert not journal_path.exists()


def test_user_change_after_restore_classification_abandons_the_ramp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(
        journal_path,
        current_time=time.time() - 30.0,
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )
    restoring_entered = threading.Event()
    restoring_gate = threading.Event()
    original_write = ducker._write_recovery_journal

    def _blocking_write() -> None:
        if any(snapshot.phase == "restoring" for snapshot in ducker._deferred_snapshots.values()) and not restoring_entered.is_set():
            restoring_entered.set()
            restoring_gate.wait(timeout=2.0)
        original_write()

    monkeypatch.setattr(ducker, "_write_recovery_journal", _blocking_write)
    result: list[bool] = []
    thread = threading.Thread(target=lambda: result.append(ducker.recover_stale()))
    thread.start()
    assert restoring_entered.wait(timeout=1.0)
    backend.values[(1, 0)] = 0.3
    restoring_gate.set()
    thread.join(timeout=2.0)

    assert result == [False]
    assert backend.values[(1, 0)] == pytest.approx(0.3)
    assert backend.writes == []
    assert not journal_path.exists()


def test_recover_stale_keeps_failed_snapshot_for_retry(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_recovery_journal(journal_path)
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    backend.fail_writes[(1, 0)] = 2
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale()
    assert journal_path.exists()

    assert ducker.recover_stale()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_failed_end_keeps_journal_until_restore_all_retry(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin()
    assert token is not None
    backend.fail_writes[(1, 0)] = 4

    assert not ducker.end(token)
    assert journal_path.exists()

    assert ducker.restore_all()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_unavailable_recovery_survives_a_new_duck_session(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    journal_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": time.time(),
                "devices": [
                    {
                        "device_uid": "uid-2",
                        "device_id_hint": 2,
                        "elements": [0],
                        "original": {"0": 0.6},
                        "target": {"0": 0.12},
                        "phase": "ducked",
                        "transition_started_at": time.time(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale()
    token = ducker.begin()
    assert token is not None
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert {item["device_uid"] for item in journal["devices"]} == {
        "uid-1",
        "uid-2",
    }

    assert ducker.end(token)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert {item["device_uid"] for item in journal["devices"]} == {"uid-2"}


def test_disconnected_route_is_deferred_without_blocking_later_sessions(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class DisconnectingBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.unavailable: set[int] = set()

        def get_volume(self, device_id: int, element: int) -> float:
            if device_id in self.unavailable:
                raise RuntimeError("device disconnected")
            return super().get_volume(device_id, element)

        def device_id_for_uid(self, device_uid: str) -> int | None:
            device_id = super().device_id_for_uid(device_uid)
            if device_id in self.unavailable:
                return None
            return device_id

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            if device_id in self.unavailable:
                raise RuntimeError("device disconnected")
            super().set_volume(device_id, element, value)

    backend = DisconnectingBackend()
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    first = ducker.begin()
    assert first is not None
    backend.default_device = 2
    assert ducker.refresh(first)
    backend.unavailable.add(1)

    assert ducker.end(first)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert {item["device_uid"] for item in journal["devices"]} == {"uid-1"}

    second = ducker.begin()
    assert second is not None
    assert ducker.end(second)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert {item["device_uid"] for item in journal["devices"]} == {"uid-1"}

    backend.default_device = 1
    assert ducker.begin() is None
    assert journal_path.exists()

    backend.unavailable.remove(1)
    third = ducker.begin()
    assert third is not None
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    active = next(item for item in journal["devices"] if item["device_uid"] == "uid-1")
    assert active["original"] == {"0": pytest.approx(0.8)}
    assert ducker.end(third)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


@pytest.mark.parametrize(
    ("factor", "max_volume"),
    [
        (-0.1, 0.12),
        (1.1, 0.12),
        (0.25, -0.1),
        (0.25, 1.1),
        (True, 0.12),
        (0.25, float("nan")),
    ],
)
def test_invalid_ratios_are_rejected(
    ducker: SystemOutputDucker,
    factor: float,
    max_volume: float,
) -> None:
    with pytest.raises(ValueError):
        ducker.begin(factor=factor, max_volume=max_volume)
