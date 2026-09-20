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
        self.transports: dict[int, int | None] = {
            1: system_volume._fourcc("bltn")
        }
        self.route_signatures: dict[int, tuple[int, int] | None] = {
            1: (48_000, 2)
        }
        self.reads: list[tuple] = []
        self.writes: list[tuple[int, int, float]] = []
        self.events: list[tuple] = []
        self.fail_writes: dict[tuple[int, int], int] = defaultdict(int)
        self.mutes: dict[int, bool | None] = {1: False}
        self.mute_writes: list[tuple[int, bool]] = []
        self.fail_mute_writes: dict[int, int] = defaultdict(int)

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

    def transport_type(self, device_id: int) -> int | None:
        self.reads.append(("transport_type", device_id))
        return self.transports.get(device_id)

    def volume_elements(self, device_id: int) -> tuple[int, ...]:
        self.reads.append(("volume_elements", device_id))
        return self.elements.get(device_id, ())

    def volume_profile(self, device_id: int) -> tuple[int, ...]:
        self.reads.append(("volume_profile", device_id))
        return self.elements.get(device_id, ())

    def output_route_signature(
        self,
        device_id: int,
    ) -> tuple[int, int] | None:
        self.reads.append(("output_route_signature", device_id))
        if device_id in self.route_signatures:
            return self.route_signatures[device_id]
        # Any other live device publishes a readable media-shaped route,
        # mirroring real CoreAudio devices after an ID recycle.
        if device_id in self.elements:
            return (48_000, 2)
        return None

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
        self.events.append(("volume", device_id, element, value))

    def get_mute(self, device_id: int) -> bool | None:
        self.reads.append(("get_mute", device_id))
        return self.mutes.get(device_id)

    def set_mute(self, device_id: int, muted: bool) -> None:
        if self.fail_mute_writes[device_id]:
            self.fail_mute_writes[device_id] -= 1
            raise RuntimeError("injected mute write failure")
        self.mutes[device_id] = muted
        self.mute_writes.append((device_id, muted))
        self.events.append(("mute", device_id, muted))


class BluetoothGainBackend(FakeVolumeBackend):
    """Model visible HAL scalar separately from AirPods remote media gain."""

    def __init__(self) -> None:
        super().__init__()
        self.transports[1] = system_volume._fourcc("blue")
        self.a2dp_active = True
        self.tracked_device = 1
        self.tracked_original = 0.8
        self.effective_gain = self.values[(1, 0)]
        self.restore_started = False
        self.local_original_written = False
        self.original_writes_while_a2dp_active = 0
        self.post_a2dp_unmute_seen = False

    def volume_profile(self, device_id: int) -> tuple[int, ...]:
        self.reads.append(("volume_profile", device_id))
        if device_id == self.tracked_device:
            return (1, 2) if self.a2dp_active else (0,)
        return self.elements.get(device_id, ())

    def output_route_signature(
        self,
        device_id: int,
    ) -> tuple[int, int] | None:
        if device_id == self.tracked_device:
            self.reads.append(("output_route_signature", device_id))
            return (48_000, 2) if self.a2dp_active else (24_000, 1)
        return super().output_route_signature(device_id)

    def set_mute(self, device_id: int, muted: bool) -> None:
        super().set_mute(device_id, muted)
        if (
            self.restore_started
            and device_id == self.tracked_device
            and self.a2dp_active
            and not muted
        ):
            self.post_a2dp_unmute_seen = True

    def set_volume(
        self,
        device_id: int,
        element: int,
        value: float,
    ) -> None:
        super().set_volume(device_id, element, value)
        if (
            device_id == self.tracked_device
            and self.a2dp_active
        ):
            self.effective_gain = value
        if (
            self.restore_started
            and device_id == self.tracked_device
            and abs(value - self.tracked_original) <= 1e-9
        ):
            self.local_original_written = True
            if self.a2dp_active:
                self.original_writes_while_a2dp_active += 1


class CoalescingBluetoothGainBackend(BluetoothGainBackend):
    """Model a driver that ignores writes equal to the visible HAL scalar."""

    def set_volume(
        self,
        device_id: int,
        element: int,
        value: float,
    ) -> None:
        previous_visible = self.values[(device_id, element)]
        previous_effective = self.effective_gain
        super().set_volume(device_id, element, value)
        if device_id != self.tracked_device or not self.a2dp_active:
            return
        if abs(value - previous_visible) <= 1e-9:
            self.effective_gain = previous_effective
        else:
            self.effective_gain = value


class TemporalCoalescingBluetoothGainBackend(BluetoothGainBackend):
    """Model a driver that merges a quick scalar round trip into a no-op."""

    COALESCE_WINDOW = 0.015

    def __init__(self) -> None:
        super().__init__()
        self.clock = 0.0
        self._last_write_at: float | None = None
        self._burst_visible = 0.0
        self._burst_effective = 0.0
        self.muted_during_tickle: list[bool | None] = []

    def advance(self, delay: float) -> None:
        self.clock += delay

    def set_volume(
        self,
        device_id: int,
        element: int,
        value: float,
    ) -> None:
        previous_visible = self.values[(device_id, element)]
        previous_effective = self.effective_gain
        previous_write_at = self._last_write_at
        super().set_volume(device_id, element, value)
        if (
            device_id != self.tracked_device
            or not self.a2dp_active
            or not self.restore_started
        ):
            return

        self.muted_during_tickle.append(self.mutes.get(device_id))
        if (
            previous_write_at is not None
            and self.clock - previous_write_at <= self.COALESCE_WINDOW
            and abs(value - self._burst_visible) <= 1e-9
        ):
            # The driver saw only the final value, which equals the already
            # visible slider, so its stale remote media gain did not move.
            self.effective_gain = self._burst_effective
        else:
            self._burst_visible = previous_visible
            self._burst_effective = previous_effective
        self._last_write_at = self.clock


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


def test_coreaudio_transport_type_reads_global_device_property() -> None:
    class FakeCoreAudio:
        def __init__(self) -> None:
            self.addresses: list[tuple[int, int, int]] = []

        @staticmethod
        def _address(address_pointer):
            return ctypes.cast(
                address_pointer,
                ctypes.POINTER(system_volume._PropertyAddress),
            ).contents

        def AudioObjectHasProperty(self, object_id, address_pointer) -> int:
            assert object_id == 42
            address = self._address(address_pointer)
            self.addresses.append(
                (
                    address.mSelector,
                    address.mScope,
                    address.mElement,
                )
            )
            return 1

        def AudioObjectGetPropertyData(
            self,
            object_id,
            address_pointer,
            _qualifier_size,
            _qualifier_pointer,
            size_pointer,
            output_pointer,
        ) -> int:
            assert object_id == 42
            address = self._address(address_pointer)
            self.addresses.append(
                (
                    address.mSelector,
                    address.mScope,
                    address.mElement,
                )
            )
            assert (
                ctypes.cast(
                    size_pointer,
                    ctypes.POINTER(ctypes.c_uint32),
                ).contents.value
                == ctypes.sizeof(ctypes.c_uint32)
            )
            ctypes.cast(
                output_pointer,
                ctypes.POINTER(ctypes.c_uint32),
            ).contents.value = system_volume._fourcc("blue")
            return 0

    fake_ca = FakeCoreAudio()
    backend = _CoreAudioBackend()
    backend._ca = fake_ca

    assert backend.transport_type(42) == system_volume._fourcc("blue")
    assert fake_ca.addresses == [
        (
            system_volume._TRANSPORT_TYPE,
            system_volume._SCOPE_GLOBAL,
            system_volume._MAIN_ELEMENT,
        ),
        (
            system_volume._TRANSPORT_TYPE,
            system_volume._SCOPE_GLOBAL,
            system_volume._MAIN_ELEMENT,
        ),
    ]


def test_coreaudio_prefers_virtual_main_volume_for_system_slider_semantics() -> None:
    class FakeCoreAudio:
        def __init__(self) -> None:
            self.writes: list[tuple[int, int, int, int, float]] = []

        @staticmethod
        def _address(address_pointer):
            return ctypes.cast(
                address_pointer,
                ctypes.POINTER(system_volume._PropertyAddress),
            ).contents

        def AudioObjectHasProperty(self, object_id, address_pointer) -> int:
            assert object_id == 42
            address = self._address(address_pointer)
            return int(
                address.mSelector == system_volume._VIRTUAL_MAIN_VOLUME
                and address.mScope == system_volume._SCOPE_OUTPUT
                and address.mElement == system_volume._MAIN_ELEMENT
            )

        def AudioObjectIsPropertySettable(
            self,
            object_id,
            address_pointer,
            writable_pointer,
        ) -> int:
            assert object_id == 42
            address = self._address(address_pointer)
            assert address.mSelector == system_volume._VIRTUAL_MAIN_VOLUME
            ctypes.cast(
                writable_pointer,
                ctypes.POINTER(ctypes.c_ubyte),
            ).contents.value = 1
            return 0

        def AudioObjectGetPropertyData(
            self,
            object_id,
            address_pointer,
            _qualifier_size,
            _qualifier_pointer,
            size_pointer,
            output_pointer,
        ) -> int:
            assert object_id == 42
            address = self._address(address_pointer)
            assert address.mSelector == system_volume._VIRTUAL_MAIN_VOLUME
            assert address.mScope == system_volume._SCOPE_OUTPUT
            assert address.mElement == system_volume._MAIN_ELEMENT
            assert (
                ctypes.cast(
                    size_pointer,
                    ctypes.POINTER(ctypes.c_uint32),
                ).contents.value
                == ctypes.sizeof(ctypes.c_float)
            )
            ctypes.cast(
                output_pointer,
                ctypes.POINTER(ctypes.c_float),
            ).contents.value = 0.38
            return 0

        def AudioObjectSetPropertyData(
            self,
            object_id,
            address_pointer,
            _qualifier_size,
            _qualifier_pointer,
            data_size,
            data_pointer,
        ) -> int:
            address = self._address(address_pointer)
            value = ctypes.cast(
                data_pointer,
                ctypes.POINTER(ctypes.c_float),
            ).contents.value
            self.writes.append(
                (
                    object_id,
                    address.mSelector,
                    address.mScope,
                    address.mElement,
                    value,
                )
            )
            assert data_size == ctypes.sizeof(ctypes.c_float)
            return 0

    fake_ca = FakeCoreAudio()
    backend = _CoreAudioBackend()
    backend._ca = fake_ca

    assert backend.volume_elements(42) == (
        system_volume._VIRTUAL_MAIN_ELEMENT,
    )
    assert backend.get_volume(
        42,
        system_volume._VIRTUAL_MAIN_ELEMENT,
    ) == pytest.approx(0.38)
    backend.set_volume(42, system_volume._VIRTUAL_MAIN_ELEMENT, 0.41)

    assert fake_ca.writes == [
        (
            42,
            system_volume._VIRTUAL_MAIN_VOLUME,
            system_volume._SCOPE_OUTPUT,
            system_volume._MAIN_ELEMENT,
            pytest.approx(0.41),
        )
    ]


def test_coreaudio_keeps_raw_profile_fingerprint_beside_virtual_main() -> None:
    class FakeCoreAudio:
        @staticmethod
        def _address(address_pointer):
            return ctypes.cast(
                address_pointer,
                ctypes.POINTER(system_volume._PropertyAddress),
            ).contents

        def AudioObjectHasProperty(self, object_id, address_pointer) -> int:
            assert object_id == 42
            address = self._address(address_pointer)
            return int(
                (
                    address.mSelector
                    == system_volume._VIRTUAL_MAIN_VOLUME
                    and address.mElement == system_volume._MAIN_ELEMENT
                )
                or (
                    address.mSelector
                    == system_volume._PREFERRED_STEREO_CHANNELS
                    and address.mElement == system_volume._MAIN_ELEMENT
                )
                or (
                    address.mSelector == system_volume._VOLUME_SCALAR
                    and address.mElement in (1, 2)
                )
            )

        def AudioObjectIsPropertySettable(
            self,
            object_id,
            _address_pointer,
            writable_pointer,
        ) -> int:
            assert object_id == 42
            ctypes.cast(
                writable_pointer,
                ctypes.POINTER(ctypes.c_ubyte),
            ).contents.value = 1
            return 0

        def AudioObjectGetPropertyData(
            self,
            object_id,
            address_pointer,
            _qualifier_size,
            _qualifier_pointer,
            size_pointer,
            output_pointer,
        ) -> int:
            assert object_id == 42
            address = self._address(address_pointer)
            assert (
                address.mSelector
                == system_volume._PREFERRED_STEREO_CHANNELS
            )
            channels = ctypes.cast(
                output_pointer,
                ctypes.POINTER(ctypes.c_uint32 * 2),
            ).contents
            channels[0] = 1
            channels[1] = 2
            ctypes.cast(
                size_pointer,
                ctypes.POINTER(ctypes.c_uint32),
            ).contents.value = ctypes.sizeof(ctypes.c_uint32 * 2)
            return 0

    backend = _CoreAudioBackend()
    backend._ca = FakeCoreAudio()

    assert backend.volume_elements(42) == (
        system_volume._VIRTUAL_MAIN_ELEMENT,
    )
    assert backend.volume_profile(42) == (1, 2)


class _FakeRouteSignatureCoreAudio:
    def __init__(self, failure: str | None = None) -> None:
        self.failure = failure

    @staticmethod
    def _address(address_pointer):
        return ctypes.cast(
            address_pointer,
            ctypes.POINTER(system_volume._PropertyAddress),
        ).contents

    def AudioObjectHasProperty(self, object_id, address_pointer) -> int:
        assert object_id == 42
        address = self._address(address_pointer)
        assert address.mScope == system_volume._SCOPE_OUTPUT
        assert address.mElement == system_volume._MAIN_ELEMENT
        if (
            self.failure == "missing"
            and address.mSelector == system_volume._STREAM_CONFIGURATION
        ):
            return 0
        return int(
            address.mSelector
            in {
                system_volume._NOMINAL_SAMPLE_RATE,
                system_volume._STREAM_CONFIGURATION,
            }
        )

    def AudioObjectGetPropertyDataSize(
        self,
        object_id,
        address_pointer,
        _qualifier_size,
        _qualifier_pointer,
        size_pointer,
    ) -> int:
        assert object_id == 42
        assert (
            self._address(address_pointer).mSelector
            == system_volume._STREAM_CONFIGURATION
        )
        if self.failure == "size":
            return -1
        layout_size = (
            system_volume._AudioBufferList.mBuffers.offset
            + 2 * ctypes.sizeof(system_volume._AudioBuffer)
        )
        ctypes.cast(
            size_pointer,
            ctypes.POINTER(ctypes.c_uint32),
        ).contents.value = layout_size
        return 0

    def AudioObjectGetPropertyData(
        self,
        object_id,
        address_pointer,
        _qualifier_size,
        _qualifier_pointer,
        _size_pointer,
        output_pointer,
    ) -> int:
        assert object_id == 42
        selector = self._address(address_pointer).mSelector
        if selector == system_volume._NOMINAL_SAMPLE_RATE:
            if self.failure == "rate":
                return -1
            ctypes.cast(
                output_pointer,
                ctypes.POINTER(ctypes.c_double),
            ).contents.value = 48_000.0
            return 0

        assert selector == system_volume._STREAM_CONFIGURATION
        if self.failure == "layout":
            return -1
        base_address = ctypes.cast(
            output_pointer,
            ctypes.c_void_p,
        ).value
        assert base_address is not None
        ctypes.c_uint32.from_address(base_address).value = 2
        first_buffer = (
            base_address + system_volume._AudioBufferList.mBuffers.offset
        )
        for index in range(2):
            audio_buffer = system_volume._AudioBuffer.from_address(
                first_buffer
                + index * ctypes.sizeof(system_volume._AudioBuffer)
            )
            audio_buffer.mNumberChannels = 1
        return 0


def test_coreaudio_output_route_signature_reads_public_hal_shape() -> None:
    backend = _CoreAudioBackend()
    backend._ca = _FakeRouteSignatureCoreAudio()

    assert backend.output_route_signature(42) == (48_000, 2)


@pytest.mark.parametrize("failure", ["missing", "rate", "size", "layout"])
def test_coreaudio_output_route_signature_is_optional(failure: str) -> None:
    backend = _CoreAudioBackend()
    backend._ca = _FakeRouteSignatureCoreAudio(failure)

    assert backend.output_route_signature(42) is None


def test_coreaudio_mute_uses_writable_master_output_property() -> None:
    class FakeCoreAudio:
        def __init__(self) -> None:
            self.writes: list[tuple[int, int, int, int, bool]] = []

        @staticmethod
        def _address(address_pointer):
            return ctypes.cast(
                address_pointer,
                ctypes.POINTER(system_volume._PropertyAddress),
            ).contents

        def AudioObjectHasProperty(self, object_id, address_pointer) -> int:
            address = self._address(address_pointer)
            assert object_id == 42
            assert address.mSelector == system_volume._MUTE
            assert address.mScope == system_volume._SCOPE_OUTPUT
            assert address.mElement == system_volume._MAIN_ELEMENT
            return 1

        def AudioObjectIsPropertySettable(
            self,
            object_id,
            address_pointer,
            writable_pointer,
        ) -> int:
            assert object_id == 42
            assert self._address(address_pointer).mSelector == system_volume._MUTE
            ctypes.cast(
                writable_pointer,
                ctypes.POINTER(ctypes.c_ubyte),
            ).contents.value = 1
            return 0

        def AudioObjectGetPropertyData(
            self,
            object_id,
            address_pointer,
            _qualifier_size,
            _qualifier_pointer,
            size_pointer,
            output_pointer,
        ) -> int:
            address = self._address(address_pointer)
            assert object_id == 42
            assert address.mSelector == system_volume._MUTE
            assert address.mScope == system_volume._SCOPE_OUTPUT
            assert address.mElement == system_volume._MAIN_ELEMENT
            assert (
                ctypes.cast(
                    size_pointer,
                    ctypes.POINTER(ctypes.c_uint32),
                ).contents.value
                == ctypes.sizeof(ctypes.c_uint32)
            )
            ctypes.cast(
                output_pointer,
                ctypes.POINTER(ctypes.c_uint32),
            ).contents.value = 1
            return 0

        def AudioObjectSetPropertyData(
            self,
            object_id,
            address_pointer,
            _qualifier_size,
            _qualifier_pointer,
            data_size,
            data_pointer,
        ) -> int:
            address = self._address(address_pointer)
            muted = bool(
                ctypes.cast(
                    data_pointer,
                    ctypes.POINTER(ctypes.c_uint32),
                ).contents.value
            )
            self.writes.append(
                (
                    object_id,
                    address.mSelector,
                    address.mScope,
                    address.mElement,
                    muted,
                )
            )
            assert data_size == ctypes.sizeof(ctypes.c_uint32)
            return 0

    fake_ca = FakeCoreAudio()
    backend = _CoreAudioBackend()
    backend._ca = fake_ca

    assert backend.get_mute(42) is True
    backend.set_mute(42, False)

    assert fake_ca.writes == [
        (
            42,
            system_volume._MUTE,
            system_volume._SCOPE_OUTPUT,
            system_volume._MAIN_ELEMENT,
            False,
        )
    ]


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


@pytest.fixture(autouse=True)
def _stop_system_volume_workers(monkeypatch):
    instances: list[SystemOutputDucker] = []
    original_init = SystemOutputDucker.__init__

    def _tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)

    monkeypatch.setattr(SystemOutputDucker, "__init__", _tracked_init)
    yield

    for instance in instances:
        instance.stop_background_workers()
    leaked = [
        thread.name
        for thread in threading.enumerate()
        if thread.name
        in {
            "system-volume-monitor",
            "system-volume-deferred-sync",
        }
    ]
    assert leaked == []


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


def test_zero_max_volume_uses_mute_and_restores_in_safe_order() -> None:
    operations: list[tuple[str, float | bool]] = []

    class OrderedBackend(FakeVolumeBackend):
        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            operations.append(("volume", value))
            super().set_volume(device_id, element, value)

        def set_mute(self, device_id: int, muted: bool) -> None:
            operations.append(("mute", muted))
            super().set_mute(device_id, muted)

    backend = OrderedBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert backend.mutes[1] is True
    mute_on = operations.index(("mute", True))
    assert mute_on == 0
    assert any(name == "volume" for name, _value in operations[1:])

    operations.clear()
    assert ducker.end(token)
    assert backend.mutes[1] is False
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    mute_off = operations.index(("mute", False))
    assert mute_off == len(operations) - 1
    assert all(name == "volume" for name, _value in operations[:mute_off])


def test_live_zero_mute_rewrites_original_after_a2dp_activation() -> None:
    backend = BluetoothGainBackend()
    elapsed_after_local_restore = 0.0

    def _sleep(delay: float) -> None:
        nonlocal elapsed_after_local_restore
        if not backend.local_original_written:
            return
        elapsed_after_local_restore += delay
        if elapsed_after_local_restore >= 0.20:
            backend.a2dp_active = True

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    assert backend.effective_gain == pytest.approx(0.05)

    # HFP teardown accepts visible HAL writes while its hidden remote media
    # gain remains at the quiet floor. A2DP becomes active only after the
    # regular restore has already written the original scalar.
    backend.a2dp_active = False
    backend.restore_started = True

    assert ducker.end(token)
    assert backend.local_original_written
    assert backend.original_writes_while_a2dp_active >= 1
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.effective_gain == pytest.approx(0.8)
    final_pass = backend.events[-4:]
    assert final_pass[0] == ("mute", 1, True)
    assert final_pass[1][:3] == ("volume", 1, 0)
    assert final_pass[1][3] != pytest.approx(0.8)
    assert final_pass[2] == ("volume", 1, 0, pytest.approx(0.8))
    assert final_pass[3] == ("mute", 1, False)


def test_post_restore_tickle_updates_coalesced_airpods_gain() -> None:
    class MuteTrackingBackend(CoalescingBluetoothGainBackend):
        def __init__(self) -> None:
            super().__init__()
            self.media_write_mutes: list[bool | None] = []

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            if self.a2dp_active:
                self.media_write_mutes.append(self.mutes[device_id])
            super().set_volume(device_id, element, value)

    backend = MuteTrackingBackend()
    elapsed_after_local_restore = 0.0
    media_events: list[tuple] = []

    def _sleep(delay: float) -> None:
        nonlocal elapsed_after_local_restore
        if not backend.local_original_written:
            return
        elapsed_after_local_restore += delay
        if elapsed_after_local_restore >= 0.20 and not backend.a2dp_active:
            backend.a2dp_active = True
            backend.events.clear()

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    assert backend.effective_gain == pytest.approx(0.05)
    backend.a2dp_active = False
    backend.restore_started = True

    assert ducker.end(token)
    media_events.extend(backend.events)

    assert backend.effective_gain == pytest.approx(0.8)
    assert backend.media_write_mutes
    assert all(muted is True for muted in backend.media_write_mutes)
    volume_values = [
        event[3] for event in media_events if event[0] == "volume"
    ]
    assert any(abs(value - 0.8) > 0.005 for value in volume_values)
    assert volume_values[-1] == pytest.approx(0.8)
    assert media_events[-1] == ("mute", 1, False)


def test_post_restore_tickle_survives_temporal_driver_coalescing() -> None:
    backend = TemporalCoalescingBluetoothGainBackend()
    backend.values[(1, 0)] = 0.8
    backend.effective_gain = 0.05
    backend.restore_started = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.0},
        owned_values={0: (0.8,)},
        original_mute=False,
        mute_target=True,
        mute_owned=False,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_route_signature=(48_000, 2),
        post_restore_values={0: 0.8},
        post_restore_ready=True,
        post_restore_pass=len(system_volume._POST_RESTORE_SYNC_DELAYS) - 1,
    )
    sleep_calls: list[float] = []

    def _sleep(delay: float) -> None:
        assert backend.mutes[1] is True
        sleep_calls.append(delay)
        backend.advance(delay)

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )

    completed, reachable, failed = ducker._sync_post_restore_snapshot(snapshot)

    assert (completed, reachable, failed) == (True, True, False)
    assert sleep_calls == [system_volume._POST_RESTORE_TICKLE_DWELL]
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.effective_gain == pytest.approx(0.8)
    assert backend.muted_during_tickle == [True, True]
    assert backend.events == [
        ("mute", 1, True),
        ("volume", 1, 0, pytest.approx(0.8 - 1.0 / 127.0)),
        ("volume", 1, 0, pytest.approx(0.8)),
        ("mute", 1, False),
    ]


def test_post_restore_pass_survives_process_restart(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = CoalescingBluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.05
    backend.restore_started = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.8,)},
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_route_signature=(48_000, 2),
        post_restore_values={virtual: 0.8},
        post_restore_ready=True,
    )
    first = SystemOutputDucker(backend, recovery_path=journal_path)
    first._deferred_snapshots[first._snapshot_key(snapshot)] = snapshot
    first._write_recovery_journal()

    completed, reachable, failed = first._sync_post_restore_snapshot(snapshot)

    assert not completed
    assert reachable
    assert not failed
    assert snapshot.post_restore_pass == 1
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["devices"][0]["post_restore"]["pass"] == 1
    assert backend.effective_gain == pytest.approx(0.8)

    second = SystemOutputDucker(backend, recovery_path=journal_path)
    assert second.recover_stale(start_deferred=False)
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_post_restore_waits_for_media_profile_instead_of_fixed_delay() -> None:
    backend = BluetoothGainBackend()
    elapsed_after_local_restore = 0.0

    def _sleep(delay: float) -> None:
        nonlocal elapsed_after_local_restore
        if not backend.local_original_written:
            return
        elapsed_after_local_restore += delay
        if elapsed_after_local_restore >= 1.10:
            backend.a2dp_active = True

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.a2dp_active = False
    backend.restore_started = True

    assert ducker.end(token)
    assert elapsed_after_local_restore >= 1.10
    assert backend.original_writes_while_a2dp_active >= 1
    assert backend.effective_gain == pytest.approx(0.8)


def test_post_restore_waits_for_media_signature_when_raw_profile_is_same() -> None:
    class SameRawProfileBackend(BluetoothGainBackend):
        def volume_profile(self, device_id: int) -> tuple[int, ...]:
            self.reads.append(("volume_profile", device_id))
            return (0,) if device_id == self.tracked_device else ()

    backend = SameRawProfileBackend()
    elapsed_after_local_restore = 0.0

    def _sleep(delay: float) -> None:
        nonlocal elapsed_after_local_restore
        if not backend.local_original_written:
            return
        elapsed_after_local_restore += delay
        if elapsed_after_local_restore >= 0.65:
            backend.a2dp_active = True

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    snapshot = ducker._snapshots[("uid-1", (0,))]
    assert snapshot.post_restore_profile == (0,)
    assert snapshot.post_restore_route_signature == (48_000, 2)

    backend.a2dp_active = False
    backend.restore_started = True

    assert ducker.end(token)
    assert elapsed_after_local_restore >= 0.65
    assert backend.original_writes_while_a2dp_active >= 1
    assert backend.effective_gain == pytest.approx(0.8)


def test_restore_retries_route_signature_that_was_unreadable_at_begin() -> None:
    class LateSignatureBackend(BluetoothGainBackend):
        signature_available = False

        def output_route_signature(
            self,
            device_id: int,
        ) -> tuple[int, int] | None:
            if not self.signature_available:
                self.reads.append(("output_route_signature", device_id))
                return None
            return super().output_route_signature(device_id)

    backend = LateSignatureBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    snapshot = ducker._snapshots[("uid-1", (0,))]
    assert not snapshot.post_restore_sync
    assert snapshot.post_restore_media_pending

    backend.signature_available = True
    backend.restore_started = True

    assert ducker.end(token)
    assert backend.original_writes_while_a2dp_active >= 1
    assert backend.effective_gain == pytest.approx(0.8)


@pytest.mark.parametrize("failure", ["empty", "error"])
def test_post_restore_sync_requires_a_trustworthy_media_profile(
    failure: str,
) -> None:
    class UnknownProfileBackend(BluetoothGainBackend):
        def volume_profile(self, device_id: int) -> tuple[int, ...]:
            if failure == "error":
                raise RuntimeError("injected profile read failure")
            return ()

    backend = UnknownProfileBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    snapshot = ducker._snapshots[("uid-1", (0,))]
    assert not snapshot.post_restore_sync
    assert snapshot.post_restore_media_pending
    backend.a2dp_active = False
    backend.restore_started = True
    assert ducker.end(token)
    assert not backend.post_a2dp_unmute_seen
    assert backend.original_writes_while_a2dp_active == 0


@pytest.mark.parametrize(
    "signature",
    [None, (24_000, 1), (24_000, 2), (48_000, 1)],
)
def test_post_restore_sync_requires_a_trustworthy_media_signature(
    signature: tuple[int, int] | None,
) -> None:
    class UntrustworthySignatureBackend(BluetoothGainBackend):
        def output_route_signature(
            self,
            device_id: int,
        ) -> tuple[int, int] | None:
            self.reads.append(("output_route_signature", device_id))
            return signature

    backend = UntrustworthySignatureBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    snapshot = ducker._snapshots[("uid-1", (0,))]
    assert not snapshot.post_restore_sync
    assert snapshot.post_restore_media_pending
    assert ducker.end(token)


def test_pending_media_candidate_waits_for_a2dp_before_tickle() -> None:
    class LateMetadataBackend(BluetoothGainBackend):
        metadata_available = False

        def transport_type(self, device_id: int) -> int | None:
            if not self.metadata_available:
                self.reads.append(("transport_type", device_id))
                return None
            return super().transport_type(device_id)

        def volume_profile(self, device_id: int) -> tuple[int, ...]:
            if not self.metadata_available:
                self.reads.append(("volume_profile", device_id))
                return ()
            return super().volume_profile(device_id)

        def output_route_signature(
            self,
            device_id: int,
        ) -> tuple[int, int] | None:
            if not self.metadata_available:
                self.reads.append(("output_route_signature", device_id))
                return None
            return super().output_route_signature(device_id)

    backend = LateMetadataBackend()
    elapsed_after_local_restore = 0.0

    def _sleep(delay: float) -> None:
        nonlocal elapsed_after_local_restore
        if not backend.local_original_written:
            return
        elapsed_after_local_restore += delay
        if elapsed_after_local_restore >= 0.35:
            backend.metadata_available = True
            backend.a2dp_active = True

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    snapshot = ducker._snapshots[("uid-1", (0,))]
    assert snapshot.post_restore_media_pending
    assert not snapshot.post_restore_sync

    backend.a2dp_active = False
    backend.restore_started = True

    assert ducker.end(token)
    assert elapsed_after_local_restore >= 0.35
    assert backend.original_writes_while_a2dp_active >= 1
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.effective_gain == pytest.approx(0.8)
    assert backend.post_a2dp_unmute_seen
    assert not ducker._deferred_snapshots


def test_pending_media_candidate_survives_cleared_mute_metadata(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.0},
        owned_values={0: (0.8,)},
        original_mute=None,
        mute_target=None,
        mute_owned=False,
        post_restore_media_pending=True,
        post_restore_expected_mute=False,
        post_restore_values={0: 0.8},
        post_restore_ready=True,
    )
    writer = SystemOutputDucker(
        FakeVolumeBackend(),
        recovery_path=journal_path,
    )
    writer._deferred_snapshots[writer._snapshot_key(snapshot)] = snapshot
    writer._write_recovery_journal()

    backend = BluetoothGainBackend()
    backend.values[(1, 0)] = 0.8
    backend.effective_gain = 0.05
    backend.restore_started = True
    reader = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )
    journal = reader._read_recovery_journal()

    assert journal is not None
    _, snapshots = journal
    restored = snapshots[("uid-1", (0,))]
    assert restored.post_restore_media_pending
    assert not restored.post_restore_sync
    assert restored.post_restore_expected_mute is False
    assert restored.post_restore_values == {0: pytest.approx(0.8)}
    assert restored.post_restore_ready
    assert reader.recover_stale(start_deferred=False)
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_pending_media_candidate_round_trips_before_local_restore(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.0},
        owned_values={0: (0.0,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        post_restore_media_pending=True,
        post_restore_expected_mute=False,
    )
    writer = SystemOutputDucker(
        FakeVolumeBackend(),
        recovery_path=journal_path,
    )
    writer._deferred_snapshots[writer._snapshot_key(snapshot)] = snapshot
    writer._write_recovery_journal()

    reader = SystemOutputDucker(
        FakeVolumeBackend(),
        recovery_path=journal_path,
    )
    journal = reader._read_recovery_journal()

    assert journal is not None
    _, snapshots = journal
    restored = snapshots[("uid-1", (0,))]
    assert restored.post_restore_media_pending
    assert not restored.post_restore_sync
    assert not restored.post_restore_ready
    assert restored.post_restore_values == {}


@pytest.mark.parametrize("override", ["scalar", "mute"])
def test_pending_media_candidate_preserves_user_override(override: str) -> None:
    backend = BluetoothGainBackend()
    backend.values[(1, 0)] = 0.8
    backend.effective_gain = 0.05
    if override == "scalar":
        backend.values[(1, 0)] = 0.6
    else:
        backend.mutes[1] = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.0},
        owned_values={0: (0.8,)},
        post_restore_media_pending=True,
        post_restore_expected_mute=False,
        post_restore_values={0: 0.8},
        post_restore_ready=True,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )

    completed, reachable, failed = ducker._sync_post_restore_snapshot(snapshot)

    assert (completed, reachable, failed) == (True, True, False)
    assert backend.events == []
    if override == "scalar":
        assert backend.values[(1, 0)] == pytest.approx(0.6)
        assert backend.mutes[1] is False
    else:
        assert backend.values[(1, 0)] == pytest.approx(0.8)
        assert backend.mutes[1] is True


def test_pending_media_candidate_stops_on_confirmed_non_bluetooth() -> None:
    backend = FakeVolumeBackend()
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.0},
        owned_values={0: (0.8,)},
        post_restore_media_pending=True,
        post_restore_expected_mute=False,
        post_restore_values={0: 0.8},
        post_restore_ready=True,
    )
    ducker = SystemOutputDucker(backend)

    completed, reachable, failed = ducker._sync_post_restore_snapshot(snapshot)

    assert (completed, reachable, failed) == (True, True, False)
    assert not snapshot.post_restore_media_pending
    assert not snapshot.post_restore_sync
    assert backend.events == []


def test_pending_bluetooth_gain_refresh_survives_process_restart(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = BluetoothGainBackend()
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    first = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = first.begin(max_volume=0.0)
    assert token is not None
    assert first._monitor_stop is not None
    first._monitor_stop.set()
    assert first._monitor_thread is not None
    first._monitor_thread.join(timeout=1.0)

    backend.a2dp_active = False
    backend.restore_started = True
    with first._lock:
        restored, _unavailable, _failed, delay = (
            first._restore_pending_locked(
                token,
                defer_unavailable=False,
            )
        )
    assert not restored
    assert delay == pytest.approx(
        system_volume._POST_RESTORE_SYNC_DELAYS[0]
    )
    assert backend.effective_gain == pytest.approx(0.05)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    pending = journal["devices"][0]["post_restore"]
    assert journal["devices"][0]["route_signature"] == [48_000, 2]
    assert pending["expected_mute"] is False
    assert pending["profile"] == [1, 2]
    assert pending["route_signature"] == [48_000, 2]
    assert pending["values"] == {str(virtual): pytest.approx(0.8)}

    # Simulate a process exit after the local scalar restore but before A2DP
    # became active. The next process must retain and finish the remote refresh.
    with first._lock:
        first._active_token = None
        first._snapshots.clear()
        first._release_lease()
    backend.a2dp_active = True
    recovery = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert recovery.recover_stale()
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_hard_exit_after_local_restore_keeps_remote_refresh_durable(
    tmp_path: Path,
) -> None:
    class CrashAfterLocalRestore(BluetoothGainBackend):
        crash_on_original = False

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            super().set_volume(device_id, element, value)
            if (
                self.crash_on_original
                and device_id == self.tracked_device
                and abs(value - self.tracked_original) <= 1e-9
            ):
                self.crash_on_original = False
                raise SystemExit("injected hard process exit")

    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = CrashAfterLocalRestore()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    first = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = first.begin(max_volume=0.0)
    assert token is not None
    assert first._monitor_stop is not None
    first._monitor_stop.set()
    assert first._monitor_thread is not None
    first._monitor_thread.join(timeout=1.0)
    backend.a2dp_active = False
    backend.restore_started = True
    backend.crash_on_original = True

    with pytest.raises(SystemExit, match="hard process exit"):
        with first._lock:
            first._restore_pending_locked(
                token,
                defer_unavailable=False,
            )

    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    pending = journal["devices"][0]["post_restore"]
    assert pending["ready"] is False
    assert pending["values"] == {str(virtual): pytest.approx(0.8)}
    assert backend.effective_gain == pytest.approx(0.05)

    with first._lock:
        first._active_token = None
        first._snapshots.clear()
        first._release_lease()
    backend.a2dp_active = True
    recovery = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert recovery.recover_stale()
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_crash_while_ducked_preserves_post_restore_intent(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    first = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = first.begin(max_volume=0.0)
    assert token is not None
    assert backend.effective_gain == pytest.approx(0.05)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["devices"][0]["volume_profile"] == [1, 2]
    assert "post_restore" not in journal["devices"][0]

    assert first._monitor_stop is not None
    first._monitor_stop.set()
    assert first._monitor_thread is not None
    first._monitor_thread.join(timeout=1.0)
    with first._lock:
        first._active_token = None
        first._snapshots.clear()
        first._release_lease()
    backend.restore_started = True
    recovery = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert recovery.recover_stale()
    assert backend.mutes[1] is False
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_crash_after_mute_ownership_release_preserves_post_restore_intent(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    first = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = first.begin(max_volume=0.0)
    assert token is not None
    assert first._monitor_stop is not None
    first._monitor_stop.set()
    assert first._monitor_thread is not None
    first._monitor_thread.join(timeout=1.0)

    # HFP may clear mute after the monitor's force-mute window. Releasing that
    # ownership must not also erase the independently captured A2DP intent.
    backend.mutes[1] = False
    with first._lock:
        first._refresh_locked(
            allow_reduck=False,
            abandon_on_deviation=True,
            force_mute=False,
        )
        snapshot = first._snapshots[("uid-1", (virtual,))]
        assert snapshot.original_mute is None
        assert snapshot.post_restore_sync

    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["devices"][0]["volume_profile"] == [1, 2]
    assert "mute" not in journal["devices"][0]
    assert "post_restore" not in journal["devices"][0]

    # Simulate a hard exit before release. The next process must reconstruct
    # the post-restore intent from the durable media-profile fingerprint.
    with first._lock:
        first._active_token = None
        first._snapshots.clear()
        first._release_lease()
    backend.restore_started = True
    backend.a2dp_active = True
    recovery = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert recovery.recover_stale()
    assert backend.values[(1, virtual)] == pytest.approx(0.8)
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_deferred_refresh_continues_when_a2dp_returns_late(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.a2dp_active = False
    backend.restore_started = True

    assert ducker.end(token)
    assert backend.effective_gain == pytest.approx(0.05)
    thread = ducker._deferred_sync_thread
    assert thread is not None
    assert thread.is_alive()

    backend.a2dp_active = True
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_next_begin_rearms_pending_refresh_without_old_worker_raise(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    first = ducker.begin(max_volume=0.0)
    assert first is not None
    backend.a2dp_active = False
    backend.restore_started = True
    assert ducker.end(first)
    old_worker = ducker._deferred_sync_thread
    assert old_worker is not None
    assert old_worker.is_alive()

    second = ducker.begin(max_volume=0.0)

    assert second is not None
    assert ducker._active_token is second
    old_worker.join(timeout=1.0)
    assert not old_worker.is_alive()
    assert backend.mutes[1] is True
    assert backend.effective_gain == pytest.approx(0.05)

    # The old worker's finalizer must not release the new session's lease.
    contender = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )
    assert contender.begin(max_volume=0.0) is None

    backend.a2dp_active = True
    time.sleep(0.15)
    assert backend.effective_gain == pytest.approx(0.05)
    assert ducker.end(second)
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


@pytest.mark.parametrize("second_max_volume", [0.0, 0.05])
def test_next_begin_never_restores_full_gain_before_rearming(
    tmp_path: Path,
    second_max_volume: float,
) -> None:
    class AudibleGainBackend(BluetoothGainBackend):
        def __init__(self) -> None:
            super().__init__()
            self.audible_gains: list[float] = []

        def _record_audible_gain(self, device_id: int) -> None:
            if device_id != self.tracked_device:
                return
            self.audible_gains.append(
                0.0 if self.mutes[device_id] else self.effective_gain
            )

        def set_mute(self, device_id: int, muted: bool) -> None:
            super().set_mute(device_id, muted)
            self._record_audible_gain(device_id)

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            super().set_volume(device_id, element, value)
            self._record_audible_gain(device_id)

    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = AudibleGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    first = ducker.begin(max_volume=0.0)
    assert first is not None
    backend.a2dp_active = False
    backend.restore_started = True
    assert ducker.end(first)
    old_worker = ducker._deferred_sync_thread
    assert old_worker is not None

    # A2DP returns after the synchronous close but before the worker refreshes
    # its hidden gain. The next begin must mute/re-duck directly, not replay the
    # pending full-volume restore and then lower it again.
    backend.a2dp_active = True
    backend.audible_gains.clear()
    backend.events.clear()
    second = ducker.begin(max_volume=second_max_volume)

    assert second is not None
    assert backend.events[0] == ("mute", 1, True)
    assert backend.audible_gains
    assert max(backend.audible_gains) <= 0.05 + 1e-9
    assert backend.mutes[1] is (second_max_volume == 0.0)
    assert ducker.end(second)
    assert backend.effective_gain == pytest.approx(0.8)


def test_failed_begin_restarts_pending_refresh_worker(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    first = ducker.begin(max_volume=0.0)
    assert first is not None
    backend.a2dp_active = False
    backend.restore_started = True
    assert ducker.end(first)
    old_worker = ducker._deferred_sync_thread
    assert old_worker is not None

    backend.default_device = None
    assert ducker.begin(max_volume=0.0) is None
    old_worker.join(timeout=1.0)
    assert not old_worker.is_alive()
    replacement = ducker._deferred_sync_thread
    assert replacement is not None
    assert replacement is not old_worker
    assert replacement.is_alive()

    backend.default_device = 1
    backend.a2dp_active = True
    replacement.join(timeout=2.0)
    assert not replacement.is_alive()
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


@pytest.mark.parametrize("failure", ["mute", "volume"])
def test_failed_rearm_preserves_pending_remote_gain_refresh(
    tmp_path: Path,
    failure: str,
) -> None:
    class FailingRearmBackend(BluetoothGainBackend):
        fail_next_mute = False

        def set_mute(self, device_id: int, muted: bool) -> None:
            if self.fail_next_mute and muted:
                self.fail_next_mute = False
                raise RuntimeError("injected rearm mute failure")
            super().set_mute(device_id, muted)

    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = FailingRearmBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    first = ducker.begin(max_volume=0.0)
    assert first is not None
    backend.a2dp_active = False
    backend.restore_started = True
    assert ducker.end(first)

    if failure == "mute":
        backend.fail_next_mute = True
    else:
        backend.fail_writes[(1, virtual)] = 1
    assert ducker.begin(max_volume=0.0) is None
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["devices"][0]["post_restore"]["ready"] is True
    assert backend.values[(1, virtual)] == pytest.approx(0.8)
    assert backend.mutes[1] is False

    worker = ducker._deferred_sync_thread
    assert worker is not None
    backend.a2dp_active = True
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_failed_hard_rearm_preserves_pending_remote_gain_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.05
    backend.effective_gain = 0.05
    backend.a2dp_active = False
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.05,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        phase=system_volume._PHASE_DUCKED,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()
    monkeypatch.setattr(
        ducker,
        "_start_monitor",
        lambda _token: (_ for _ in ()).throw(
            RuntimeError("injected monitor start failure")
        ),
    )

    assert ducker.begin(max_volume=0.0) is None
    assert backend.values[(1, virtual)] == pytest.approx(0.8)
    assert backend.mutes[1] is False
    assert backend.effective_gain == pytest.approx(0.05)
    pending = json.loads(journal_path.read_text(encoding="utf-8"))
    assert pending["devices"][0]["post_restore"]["ready"] is True

    worker = ducker._deferred_sync_thread
    assert worker is not None
    backend.a2dp_active = True
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


@pytest.mark.parametrize("override", ["volume", "mute"])
def test_pending_refresh_adopts_user_override_for_next_session(
    tmp_path: Path,
    override: str,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.8
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    first = ducker.begin(max_volume=0.0)
    assert first is not None
    backend.a2dp_active = False
    backend.restore_started = True
    assert ducker.end(first)

    if override == "volume":
        backend.values[(1, virtual)] = 0.6
    else:
        backend.mutes[1] = True
    second = ducker.begin(max_volume=0.0)

    assert second is not None
    assert ducker._active_token is second
    assert backend.mutes[1] is True
    backend.a2dp_active = True
    assert ducker.end(second)
    if override == "volume":
        assert backend.values[(1, virtual)] == pytest.approx(0.6)
        assert backend.mutes[1] is False
        assert backend.effective_gain == pytest.approx(0.6)
    else:
        assert backend.values[(1, virtual)] == pytest.approx(0.8)
        assert backend.mutes[1] is True


def test_deferred_worker_reloads_journal_after_reacquiring_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.05
    backend.a2dp_active = False
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.8,)},
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_values={virtual: 0.8},
        post_restore_ready=True,
    )
    key = ducker._snapshot_key(snapshot)
    ducker._deferred_snapshots[key] = snapshot
    ducker._write_recovery_journal()

    first_attempt = threading.Event()
    real_acquire = ducker._acquire_lease
    attempts = 0

    def _acquire_after_first_poll() -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_attempt.set()
            return False
        return real_acquire()

    monkeypatch.setattr(ducker, "_acquire_lease", _acquire_after_first_poll)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_INTERVAL", 0.02)
    ducker._start_deferred_sync_if_needed()
    assert first_attempt.wait(timeout=1.0)

    # Model a newer process replacing the journal while this stale worker does
    # not own the lease. The next poll must treat disk as authoritative.
    replacement = {
        "version": system_volume._RECOVERY_VERSION,
        "created_at": time.time(),
        "devices": [
            {
                "device_uid": "new-process-output",
                "device_id_hint": 2,
                "elements": [0],
                "original": {"0": 0.6},
                "target": {"0": 0.05},
                "phase": "ducked",
                "transition_started_at": time.time(),
                "owned": {"0": [0.05]},
            }
        ],
    }
    journal_path.write_text(json.dumps(replacement), encoding="utf-8")
    backend.a2dp_active = True

    worker = ducker._deferred_sync_thread
    assert worker is not None
    time.sleep(0.1)
    assert worker.is_alive()
    assert json.loads(journal_path.read_text(encoding="utf-8")) == replacement
    assert backend.effective_gain == pytest.approx(0.05)
    ducker._stop_deferred_sync()
    assert not worker.is_alive()


def test_parked_post_restore_mute_recovers_after_process_exit(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.05
    backend.mutes[1] = True
    backend.restore_started = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.8,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_values={virtual: 0.8},
        post_restore_ready=True,
    )
    first = SystemOutputDucker(backend, recovery_path=journal_path)
    first._deferred_snapshots[first._snapshot_key(snapshot)] = snapshot
    first._write_recovery_journal()

    recovery = SystemOutputDucker(backend, recovery_path=journal_path)
    assert recovery.recover_stale()

    assert backend.mutes[1] is False
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_parked_mute_is_restored_before_a2dp_profile_returns(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.05
    backend.mutes[1] = True
    backend.a2dp_active = False
    backend.restore_started = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.8,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_values={virtual: 0.8},
        post_restore_ready=True,
    )
    first = SystemOutputDucker(backend, recovery_path=journal_path)
    first._deferred_snapshots[first._snapshot_key(snapshot)] = snapshot
    first._write_recovery_journal()

    recovery = SystemOutputDucker(backend, recovery_path=journal_path)
    assert not recovery.recover_stale()

    assert backend.mutes[1] is False
    pending = json.loads(journal_path.read_text(encoding="utf-8"))
    assert pending["devices"][0]["post_restore"]["ready"] is True
    worker = recovery._deferred_sync_thread
    assert worker is not None
    assert worker.is_alive()

    backend.a2dp_active = True
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_parked_probe_on_hfp_does_not_schedule_foreground_post_delay() -> None:
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    original = 0.8
    probe = original - system_volume._POST_RESTORE_TICKLE_STEP
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = probe
    backend.mutes[1] = True
    backend.a2dp_active = False
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: original},
        duck_target={virtual: 0.05},
        owned_values={virtual: (original, probe)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_route_signature=(48_000, 2),
        post_restore_values={virtual: original},
        post_restore_ready=True,
        post_restore_pass=1,
    )
    ducker = SystemOutputDucker(backend)
    token = DuckToken()
    key = ducker._snapshot_key(snapshot)
    ducker._active_token = token
    ducker._snapshots[key] = snapshot
    ducker._initial_snapshot_key = key

    with ducker._lock:
        restored, unavailable, write_failed, post_restore_delay = (
            ducker._restore_pending_locked(
                token,
                defer_unavailable=False,
            )
        )

    assert not restored
    assert unavailable
    assert not write_failed
    assert post_restore_delay is None
    assert backend.values[(1, virtual)] == pytest.approx(probe)
    assert backend.mutes[1] is True


def test_live_post_restore_survives_transient_unmute_then_route_remute(
    tmp_path: Path,
) -> None:
    class RouteRemuteBackend(BluetoothGainBackend):
        def __init__(self) -> None:
            super().__init__()
            self.return_stale_unmute = True

        def get_mute(self, device_id: int) -> bool | None:
            if self.return_stale_unmute:
                self.return_stale_unmute = False
                # Model CoreAudio reporting the HFP transition's temporary
                # unmute while the route has already re-applied mute.
                self.mutes[device_id] = True
                return False
            return super().get_mute(device_id)

    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = RouteRemuteBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.mutes[1] = True
    backend.restore_started = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.8,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        phase=system_volume._PHASE_RESTORING,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_route_signature=(48_000, 2),
        post_restore_values={virtual: 0.8},
        post_restore_ready=True,
        post_restore_pass=len(system_volume._POST_RESTORE_SYNC_DELAYS) - 1,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=tmp_path / "system-volume.json",
    )
    token = DuckToken()
    ducker._active_token = token
    ducker._snapshots[ducker._snapshot_key(snapshot)] = snapshot

    completed, reachable, failed = ducker._sync_post_restore_snapshot(snapshot)

    assert (completed, reachable, failed) == (True, True, False)
    assert backend.mutes[1] is False
    assert snapshot.mute_target is None
    assert not snapshot.mute_owned


def test_abandoning_inactive_sibling_profile_releases_owned_mute(
    tmp_path: Path,
) -> None:
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.8, (1, 2): 0.8}
    backend.mutes[1] = True
    old = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.8,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        phase=system_volume._PHASE_RESTORING,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_route_signature=(48_000, 2),
        post_restore_values={virtual: 0.8},
        post_restore_ready=True,
    )
    sibling = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(1, 2),
        original={1: 0.8, 2: 0.8},
        duck_target={1: 0.05, 2: 0.05},
        owned_values={1: (0.8,), 2: (0.8,)},
        original_mute=True,
        mute_target=True,
        mute_owned=False,
        phase=system_volume._PHASE_RESTORING,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=tmp_path / "system-volume.json",
    )
    token = DuckToken()
    old_key = ducker._snapshot_key(old)
    ducker._active_token = token
    ducker._snapshots[old_key] = old
    ducker._snapshots[ducker._snapshot_key(sibling)] = sibling
    ducker._initial_snapshot_key = old_key

    assert ducker.end(token)

    assert backend.mutes[1] is False
    assert old.mute_target is None
    assert not old.mute_owned
    assert ducker._active_token is None
    assert not ducker._snapshots
    assert not ducker._deferred_snapshots


def test_hfp_clamped_during_restore_keeps_a2dp_refresh_owned(
    tmp_path: Path,
) -> None:
    class HfpClampingBackend(BluetoothGainBackend):
        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            if self.restore_started and not self.a2dp_active and value > 0.05:
                value = 0.05
            super().set_volume(device_id, element, value)

    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = HfpClampingBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.05
    backend.effective_gain = 0.05
    backend.mutes[1] = True
    backend.a2dp_active = False
    backend.restore_started = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.05,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        phase=system_volume._PHASE_DUCKED,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_route_signature=(48_000, 2),
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=tmp_path / "system-volume.json",
    )
    ducker._snapshots[ducker._snapshot_key(snapshot)] = snapshot

    unresolved, _changed, reachable, _metadata_dirty = (
        ducker._restore_owned_snapshot(
            snapshot,
            ensure_original_write=True,
            restore_mute_last=True,
        )
    )

    assert unresolved is None
    assert reachable
    assert snapshot.post_restore_values == {virtual: pytest.approx(0.8)}
    assert snapshot.mute_owned
    assert backend.values[(1, virtual)] == pytest.approx(0.05)

    backend.a2dp_active = True
    snapshot.post_restore_ready = True
    snapshot.post_restore_pass = len(system_volume._POST_RESTORE_SYNC_DELAYS) - 1
    completed, reachable, failed = ducker._sync_post_restore_snapshot(snapshot)

    assert (completed, reachable, failed) == (True, True, False)
    assert backend.values[(1, virtual)] == pytest.approx(0.8)
    assert backend.effective_gain == pytest.approx(0.8)
    assert backend.mutes[1] is False


def test_silent_unmute_write_is_retried_before_normal_restore_completes(
    tmp_path: Path,
) -> None:
    class IgnoreFirstUnmuteBackend(FakeVolumeBackend):
        ignored_unmutes = 0

        def set_mute(self, device_id: int, muted: bool) -> None:
            if not muted and self.ignored_unmutes == 0:
                self.ignored_unmutes += 1
                self.mute_writes.append((device_id, muted))
                self.events.append(("mute", device_id, muted))
                return
            super().set_mute(device_id, muted)

    backend = IgnoreFirstUnmuteBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=tmp_path / "system-volume.json",
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None

    assert ducker.end(token)

    assert backend.ignored_unmutes == 1
    assert backend.mutes[1] is False
    assert ducker._active_token is None
    assert not ducker._snapshots


@pytest.mark.parametrize("ignored_unmute", [1, 2])
def test_silent_unmute_write_is_retried_before_post_restore_completes(
    tmp_path: Path,
    ignored_unmute: int,
) -> None:
    class IgnoreFirstUnmuteBackend(BluetoothGainBackend):
        unmute_calls = 0

        def set_mute(self, device_id: int, muted: bool) -> None:
            if not muted:
                self.unmute_calls += 1
            if not muted and self.unmute_calls == ignored_unmute:
                self.mute_writes.append((device_id, muted))
                self.events.append(("mute", device_id, muted))
                return
            super().set_mute(device_id, muted)

    backend = IgnoreFirstUnmuteBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=tmp_path / "system-volume.json",
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.restore_started = True

    assert ducker.end(token)

    assert backend.unmute_calls > ignored_unmute
    assert backend.mutes[1] is False
    assert ducker._active_token is None
    assert not ducker._snapshots


def test_unknown_final_mute_read_keeps_post_restore_ownership(
    tmp_path: Path,
) -> None:
    class UnknownFinalMuteBackend(BluetoothGainBackend):
        mute_reads = 0

        def get_mute(self, device_id: int) -> bool | None:
            self.mute_reads += 1
            if self.mute_reads == 2:
                return None
            return super().get_mute(device_id)

    backend = UnknownFinalMuteBackend()
    backend.mutes[1] = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.8,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_route_signature=(48_000, 2),
        post_restore_values={0: 0.8},
        post_restore_ready=True,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=tmp_path / "system-volume.json",
    )

    completed, reachable, failed = ducker._sync_post_restore_snapshot(snapshot)

    assert (completed, reachable, failed) == (False, True, True)
    assert backend.mutes[1] is True
    assert snapshot.original_mute is False
    assert snapshot.mute_target is True
    assert snapshot.mute_owned
    assert snapshot.post_restore_sync


def test_user_mute_after_parked_restore_is_not_cleared_on_next_poll(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.05
    backend.mutes[1] = True
    backend.a2dp_active = False
    backend.restore_started = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.8,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_values={virtual: 0.8},
        post_restore_ready=True,
    )
    ducker = SystemOutputDucker(backend, recovery_path=journal_path)
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()

    completed, reachable, failed = ducker._sync_post_restore_snapshot(
        snapshot
    )

    assert not completed
    assert not reachable
    assert not failed
    assert backend.mutes[1] is False
    assert snapshot.mute_target is None
    assert snapshot.mute_owned is False

    backend.mutes[1] = True
    mute_write_count = len(backend.mute_writes)
    backend.a2dp_active = True
    completed, reachable, failed = ducker._sync_post_restore_snapshot(
        snapshot
    )

    assert completed
    assert reachable
    assert not failed
    assert backend.mutes[1] is True
    assert len(backend.mute_writes) == mute_write_count


def test_deferred_worker_polls_indefinitely_without_rewriting_unchanged_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.effective_gain = 0.05
    backend.a2dp_active = False
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.8,)},
        post_restore_sync=True,
        post_restore_expected_mute=False,
        post_restore_profile=(1, 2),
        post_restore_values={virtual: 0.8},
        post_restore_ready=True,
    )
    ducker = SystemOutputDucker(backend, recovery_path=journal_path)
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()

    writes = 0
    real_write = ducker._write_recovery_journal

    def _count_write() -> None:
        nonlocal writes
        writes += 1
        real_write()

    monkeypatch.setattr(ducker, "_write_recovery_journal", _count_write)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_FAST_DURATION", 0.0)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_SLOW_INTERVAL", 0.02)
    ducker._start_deferred_sync_if_needed()
    worker = ducker._deferred_sync_thread
    assert worker is not None

    time.sleep(0.15)
    assert worker.is_alive()
    assert writes == 0

    backend.a2dp_active = True
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


def test_output_switch_defers_airpods_refresh_until_they_are_default() -> None:
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.8
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    backend.device_uids[2] = "built-in-output"
    backend.transports[2] = system_volume._fourcc("bltn")
    backend.mutes[2] = False
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.a2dp_active = False
    backend.restore_started = True
    backend.default_device = 2

    assert ducker.end(token)
    key = ("uid-1", (virtual,))
    assert key in ducker._deferred_snapshots
    pending = ducker._deferred_snapshots[key]
    assert pending.post_restore_ready
    assert backend.effective_gain == pytest.approx(0.05)

    backend.default_device = 1
    backend.a2dp_active = True
    completed, reachable, failed = ducker._sync_post_restore_snapshot(
        pending
    )
    assert not completed
    assert reachable
    assert not failed
    assert pending.post_restore_pass == 1
    completed, reachable, failed = ducker._sync_post_restore_snapshot(
        pending
    )
    assert completed
    assert reachable
    assert not failed
    assert not pending.post_restore_sync
    assert pending.post_restore_pass == 0
    assert backend.effective_gain == pytest.approx(0.8)


def test_post_a2dp_sync_tolerates_one_transient_default_uid() -> None:
    backend = BluetoothGainBackend()
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    backend.device_uids[2] = "built-in-output"
    backend.transports[2] = system_volume._fourcc("bltn")
    backend.mutes[2] = False
    post_waits = 0
    route_retries = 0

    def _sleep(delay: float) -> None:
        nonlocal post_waits, route_retries
        if (
            backend.default_device == 2
            and delay == pytest.approx(
                system_volume._RESTORE_RETRY_DELAY
            )
        ):
            route_retries += 1
            backend.default_device = 1
            backend.a2dp_active = True
            return
        if (
            not backend.local_original_written
            or backend.a2dp_active
            or delay != pytest.approx(
                system_volume._POST_RESTORE_SYNC_DELAYS[0]
            )
        ):
            return
        post_waits += 1
        assert post_waits == 1
        backend.default_device = 2

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.a2dp_active = False
    backend.restore_started = True

    assert ducker.end(token)
    assert post_waits == 1
    assert route_retries >= 1
    assert backend.original_writes_while_a2dp_active >= 1
    assert backend.effective_gain == pytest.approx(0.8)


def test_post_a2dp_sync_abandons_a_persistent_default_uid_change() -> None:
    backend = BluetoothGainBackend()
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    backend.device_uids[2] = "user-selected-output"
    backend.transports[2] = system_volume._fourcc("bltn")
    backend.mutes[2] = False
    writes_to_old_at_switch = 0
    switched = False

    def _sleep(delay: float) -> None:
        nonlocal switched, writes_to_old_at_switch
        if (
            switched
            or not backend.local_original_written
            or delay != pytest.approx(
                system_volume._POST_RESTORE_SYNC_DELAYS[0]
            )
        ):
            return
        switched = True
        writes_to_old_at_switch = sum(
            device_id == 1
            for device_id, _element, _value in backend.writes
        )
        backend.default_device = 2

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.a2dp_active = False
    backend.restore_started = True

    assert ducker.end(token)
    assert switched
    assert (
        sum(
            device_id == 1
            for device_id, _element, _value in backend.writes
        )
        == writes_to_old_at_switch
    )
    assert backend.default_device == 2
    assert backend.values[(2, 0)] == pytest.approx(0.6)


def test_post_a2dp_sync_revalidates_device_identity_before_write() -> None:
    class RecycledIdBackend(BluetoothGainBackend):
        arm_recycle = False
        recycled = False
        writes_before_recycle = 0

        def get_volume(self, device_id: int, element: int) -> float:
            value = super().get_volume(device_id, element)
            if (
                self.arm_recycle
                and not self.recycled
                and device_id == 1
                and abs(value - self.tracked_original) <= 1e-9
            ):
                self.recycled = True
                self.writes_before_recycle = len(self.writes)
                # CoreAudio reused the numeric ID for another default output
                # after the ownership read but before the pending write.
                self.device_uids[1] = "replacement-output"
            return value

    backend = RecycledIdBackend()

    def _sleep(delay: float) -> None:
        if (
            backend.local_original_written
            and delay
            == pytest.approx(system_volume._POST_RESTORE_SYNC_DELAYS[0])
        ):
            backend.arm_recycle = True

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.restore_started = True

    assert ducker.end(token)
    assert backend.recycled
    assert len(backend.writes) == backend.writes_before_recycle
    assert backend.device_uids[1] == "replacement-output"


def test_new_bluetooth_route_captured_during_session_gets_post_sync() -> None:
    backend = BluetoothGainBackend()
    backend.transports[1] = system_volume._fourcc("bltn")
    elapsed_after_local_restore = 0.0

    def _sleep(delay: float) -> None:
        nonlocal elapsed_after_local_restore
        if not backend.local_original_written:
            return
        elapsed_after_local_restore += delay
        if elapsed_after_local_restore >= 0.20:
            backend.a2dp_active = True

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None

    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    backend.device_uids[2] = "airpods-output"
    backend.transports[2] = system_volume._fourcc("blue")
    backend.mutes[2] = False
    backend.default_device = 2
    backend.tracked_device = 2
    backend.tracked_original = 0.6
    backend.effective_gain = 0.6
    assert ducker.refresh(token)
    assert backend.effective_gain == pytest.approx(0.05)

    backend.a2dp_active = False
    backend.restore_started = True
    assert ducker.end(token)
    assert backend.original_writes_while_a2dp_active >= 1
    assert backend.effective_gain == pytest.approx(0.6)


def test_adopted_bluetooth_recovery_gets_post_sync(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    created_at = time.time() - 30.0
    journal_path.write_text(
        json.dumps(
            {
                "version": system_volume._RECOVERY_VERSION,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [
                            system_volume._VIRTUAL_MAIN_ELEMENT
                        ],
                        "original": {
                            str(system_volume._VIRTUAL_MAIN_ELEMENT): 0.8
                        },
                        "target": {
                            str(system_volume._VIRTUAL_MAIN_ELEMENT): 0.05
                        },
                        "owned": {
                            str(system_volume._VIRTUAL_MAIN_ELEMENT): [0.05]
                        },
                        "legacy": [],
                        "volume_profile": [1, 2],
                        "mute": {
                            "original": False,
                            "target": True,
                            "owned": True,
                        },
                        "phase": "ducked",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = BluetoothGainBackend()
    backend.elements[1] = (system_volume._VIRTUAL_MAIN_ELEMENT,)
    backend.values[(1, system_volume._VIRTUAL_MAIN_ELEMENT)] = 0.05
    backend.mutes[1] = True
    backend.effective_gain = 0.05
    backend.a2dp_active = False
    elapsed_after_local_restore = 0.0

    def _sleep(delay: float) -> None:
        nonlocal elapsed_after_local_restore
        if not backend.local_original_written:
            return
        elapsed_after_local_restore += delay
        if elapsed_after_local_restore >= 0.20:
            backend.a2dp_active = True

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    recovered = ducker._read_recovery_journal()
    assert recovered is not None
    _created_at, recovered_snapshots = recovered
    assert next(
        iter(recovered_snapshots.values())
    ).post_restore_route_signature is None
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.restore_started = True

    assert ducker.end(token)
    assert backend.original_writes_while_a2dp_active >= 1
    assert backend.effective_gain == pytest.approx(0.8)
    assert not journal_path.exists()


class _LegacyAirPodsVmvcBackend(FakeVolumeBackend):
    def __init__(self) -> None:
        super().__init__()
        virtual = system_volume._VIRTUAL_MAIN_ELEMENT
        self.default_device = 106
        self.elements = {106: (virtual,)}
        self.values = {(106, virtual): 0.05, (106, 0): 0.05}
        self.device_uids = {106: "airpods-max"}
        self.transports = {106: system_volume._fourcc("blue")}
        self.mutes = {106: False}
        self.raw_profile = (0,)
        self.route_signature = (24_000, 1)
        self.before_volume_write = None
        self.before_mute_write = None

    def volume_profile(self, device_id: int) -> tuple[int, ...]:
        self.reads.append(("volume_profile", device_id))
        return self.raw_profile

    def output_route_signature(
        self,
        device_id: int,
    ) -> tuple[int, int] | None:
        self.reads.append(("output_route_signature", device_id))
        return self.route_signature

    def set_volume(
        self,
        device_id: int,
        element: int,
        value: float,
    ) -> None:
        if self.before_volume_write is not None:
            self.before_volume_write(device_id, element, value)
        super().set_volume(device_id, element, value)

    def set_mute(self, device_id: int, muted: bool) -> None:
        if self.before_mute_write is not None:
            self.before_mute_write(device_id, muted)
        super().set_mute(device_id, muted)


def _write_v3_raw_master_journal(
    path: Path,
    *,
    mute_owned: bool = False,
) -> None:
    created_at = time.time() - 30.0
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "airpods-max",
                        "device_id_hint": 111,
                        "elements": [0],
                        "original": {"0": 0.8},
                        "target": {"0": 0.05},
                        "owned": {"0": [0.05]},
                        "legacy": [],
                        **(
                            {
                                "mute": {
                                    "original": False,
                                    "target": True,
                                    "owned": True,
                                }
                            }
                            if mute_owned
                            else {}
                        ),
                        "phase": "ducked",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("blocked_route", ["hfp", "mono", "non-default"])
def test_v3_raw_master_waits_for_current_default_a2dp_route(
    tmp_path: Path,
    blocked_route: str,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_v3_raw_master_journal(journal_path)
    backend = _LegacyAirPodsVmvcBackend()
    if blocked_route == "mono":
        backend.raw_profile = (1, 2)
        backend.route_signature = (48_000, 1)
    elif blocked_route == "non-default":
        backend.raw_profile = (1, 2)
        backend.route_signature = (48_000, 2)
        backend.default_device = 1
        backend.elements[1] = (system_volume._VIRTUAL_MAIN_ELEMENT,)
        backend.values[(1, system_volume._VIRTUAL_MAIN_ELEMENT)] = 0.4
        backend.device_uids[1] = "built-in"
        backend.transports[1] = system_volume._fourcc("bltn")
        backend.mutes[1] = False
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale(start_deferred=False)

    assert backend.writes == []
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert journal_path.exists()
    pending = json.loads(journal_path.read_text(encoding="utf-8"))
    assert pending["devices"][0]["elements"] == [0]
    assert pending["devices"][0]["allow_inactive_controls"] is True


def test_v3_raw_master_migrates_after_hfp_returns_to_a2dp(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_v3_raw_master_journal(journal_path)
    backend = _LegacyAirPodsVmvcBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale(start_deferred=False)
    assert backend.writes == []
    assert journal_path.exists()

    backend.raw_profile = (1, 2)
    backend.route_signature = (48_000, 2)
    journal_before_first_hal_write: list[dict] = []

    def _capture_journal_before_write(
        _device_id: int,
        _element: int,
        _value: float,
    ) -> None:
        if not journal_before_first_hal_write:
            journal_before_first_hal_write.append(
                json.loads(journal_path.read_text(encoding="utf-8"))
            )

    backend.before_volume_write = _capture_journal_before_write

    assert ducker.recover_stale(start_deferred=False)

    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    assert backend.values[(106, virtual)] == pytest.approx(0.8)
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert all(element == virtual for _, element, _ in backend.writes)
    assert not journal_path.exists()
    persisted = journal_before_first_hal_write[0]
    assert persisted["version"] == system_volume._RECOVERY_VERSION
    migrated = persisted["devices"][0]
    assert migrated["device_id_hint"] == 106
    assert migrated["elements"] == [virtual]
    assert migrated["original"] == {str(virtual): 0.8}
    assert migrated["target"] == {str(virtual): 0.05}
    assert migrated["owned"][str(virtual)][0] == pytest.approx(0.05)
    assert migrated["volume_profile"] == [1, 2]
    assert migrated["route_signature"] == [48_000, 2]
    assert migrated["post_restore"] == {
        "expected_mute": False,
        "profile": [1, 2],
        "ready": False,
        "pass": 0,
        "values": {str(virtual): 0.8},
        "route_signature": [48_000, 2],
    }


def test_v3_raw_master_keeps_owned_mute_until_vmvc_is_restored(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_v3_raw_master_journal(journal_path, mute_owned=True)
    backend = _LegacyAirPodsVmvcBackend()
    backend.raw_profile = (1, 2)
    backend.route_signature = (48_000, 2)
    backend.mutes[106] = True
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale(start_deferred=False)

    first_volume = next(
        index
        for index, event in enumerate(backend.events)
        if event[0] == "volume"
    )
    first_unmute = next(
        index
        for index, event in enumerate(backend.events)
        if event == ("mute", 106, False)
    )
    assert first_volume < first_unmute
    assert backend.mutes[106] is False
    assert backend.values[
        (106, system_volume._VIRTUAL_MAIN_ELEMENT)
    ] == pytest.approx(0.8)
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert not journal_path.exists()


def test_begin_immediately_claims_migrated_v3_raw_master(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_v3_raw_master_journal(journal_path)
    backend = _LegacyAirPodsVmvcBackend()
    backend.raw_profile = (1, 2)
    backend.route_signature = (48_000, 2)
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    active = ducker._snapshots[("airpods-max", (virtual,))]
    assert active.original[virtual] == pytest.approx(0.8)
    assert backend.values[(106, virtual)] == pytest.approx(0.05)
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert all(
        element == virtual and value <= 0.05 + system_volume._OWNERSHIP_TOLERANCE
        for _, element, value in backend.writes
    )

    assert ducker.end(token)
    assert backend.values[(106, virtual)] == pytest.approx(0.8)
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert not journal_path.exists()


def test_v3_raw_master_migrates_restored_vmvc_as_soft_pending(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_v3_raw_master_journal(journal_path)
    backend = _LegacyAirPodsVmvcBackend()
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend.raw_profile = (1, 2)
    backend.route_signature = (48_000, 2)
    backend.values[(106, virtual)] = 0.8
    journal_before_first_hal_write: list[dict] = []

    def _capture_journal_before_write(
        _device_id: int,
        _element: int,
        _value: float,
    ) -> None:
        if not journal_before_first_hal_write:
            journal_before_first_hal_write.append(
                json.loads(journal_path.read_text(encoding="utf-8"))
            )

    backend.before_volume_write = _capture_journal_before_write
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale(start_deferred=False)

    assert backend.values[(106, virtual)] == pytest.approx(0.8)
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert all(element == virtual for _, element, _ in backend.writes)
    assert journal_before_first_hal_write[0]["devices"][0]["post_restore"][
        "ready"
    ] is True
    assert not journal_path.exists()


def test_v3_raw_master_vmvc_user_override_abandons_old_ownership(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_v3_raw_master_journal(journal_path)
    backend = _LegacyAirPodsVmvcBackend()
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend.raw_profile = (1, 2)
    backend.route_signature = (48_000, 2)
    backend.values[(106, virtual)] = 0.33
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert not ducker.recover_stale(start_deferred=False)

    assert backend.writes == []
    assert backend.values[(106, virtual)] == pytest.approx(0.33)
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert not journal_path.exists()


def test_v3_raw_master_vmvc_override_restores_only_app_owned_mute(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_v3_raw_master_journal(journal_path, mute_owned=True)
    backend = _LegacyAirPodsVmvcBackend()
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend.raw_profile = (1, 2)
    backend.route_signature = (48_000, 2)
    backend.values[(106, virtual)] = 0.33
    backend.mutes[106] = True
    journal_before_mute_write: list[dict] = []

    def _capture_journal_before_mute_write(
        _device_id: int,
        _muted: bool,
    ) -> None:
        if not journal_before_mute_write:
            journal_before_mute_write.append(
                json.loads(journal_path.read_text(encoding="utf-8"))
            )

    backend.before_mute_write = _capture_journal_before_mute_write
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale(start_deferred=False)

    assert backend.writes == []
    assert backend.values[(106, virtual)] == pytest.approx(0.33)
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert backend.mute_writes == [(106, False)]
    persisted = journal_before_mute_write[0]["devices"][0]
    assert persisted["elements"] == [virtual]
    assert persisted["original"] == {str(virtual): 0.33}
    assert persisted["target"] == {str(virtual): 0.33}
    assert persisted["mute"] == {
        "original": False,
        "target": True,
        "owned": True,
    }
    assert "post_restore" not in persisted
    assert not journal_path.exists()


def test_migrated_vmvc_override_mute_only_survives_restart(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_v3_raw_master_journal(journal_path, mute_owned=True)
    backend = _LegacyAirPodsVmvcBackend()
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend.raw_profile = (1, 2)
    backend.route_signature = (48_000, 2)
    backend.values[(106, virtual)] = 0.33
    backend.mutes[106] = True
    backend.fail_mute_writes[106] = 1
    first = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert first.recover_stale(start_deferred=False)
    assert backend.writes == []
    assert backend.mutes[106] is True
    assert journal_path.exists()
    pending = json.loads(journal_path.read_text(encoding="utf-8"))
    assert pending["version"] == system_volume._RECOVERY_VERSION
    assert pending["devices"][0]["elements"] == [virtual]
    assert "post_restore" not in pending["devices"][0]

    second = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )
    assert second.recover_stale(start_deferred=False)

    assert backend.writes == []
    assert backend.values[(106, virtual)] == pytest.approx(0.33)
    assert backend.values[(106, 0)] == pytest.approx(0.05)
    assert backend.mutes[106] is False
    assert not journal_path.exists()


def test_v3_raw_master_built_in_output_keeps_raw_recovery(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    _write_v3_raw_master_journal(journal_path)
    backend = _LegacyAirPodsVmvcBackend()
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend.transports[106] = system_volume._fourcc("bltn")
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale(start_deferred=False)

    assert backend.values[(106, 0)] == pytest.approx(0.8)
    assert backend.values[(106, virtual)] == pytest.approx(0.05)
    assert backend.writes
    assert all(element == 0 for _, element, _ in backend.writes)
    assert not journal_path.exists()


def test_v3_raw_master_recovery_is_not_reinterpreted_as_virtual_main(
    tmp_path: Path,
) -> None:
    class RawMasterProfileBackend(FakeVolumeBackend):
        def volume_profile(self, device_id: int) -> tuple[int, ...]:
            return (0,)

    journal_path = tmp_path / "system-volume.json"
    created_at = time.time() - 30.0
    journal_path.write_text(
        json.dumps(
            {
                "version": 3,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [0],
                        "original": {"0": 0.26},
                        "target": {"0": 0.05},
                        "owned": {"0": [0.05]},
                        "legacy": [],
                        "phase": "restoring",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = RawMasterProfileBackend()
    backend.elements[1] = (system_volume._VIRTUAL_MAIN_ELEMENT,)
    backend.values = {
        (1, system_volume._VIRTUAL_MAIN_ELEMENT): 0.38,
        (1, 0): 0.05,
    }
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    token = ducker.begin()

    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.26)
    snapshot = ducker._snapshots[
        ("uid-1", (system_volume._VIRTUAL_MAIN_ELEMENT,))
    ]
    assert snapshot.original[system_volume._VIRTUAL_MAIN_ELEMENT] == pytest.approx(
        0.38
    )
    assert ducker.end(token)
    assert backend.values[
        (1, system_volume._VIRTUAL_MAIN_ELEMENT)
    ] == pytest.approx(0.38)


def test_startup_recovers_v3_raw_master_behind_virtual_main(
    tmp_path: Path,
) -> None:
    class RawMasterProfileBackend(FakeVolumeBackend):
        def volume_profile(self, device_id: int) -> tuple[int, ...]:
            return (0,)

    journal_path = tmp_path / "system-volume.json"
    created_at = time.time() - 30.0
    journal_path.write_text(
        json.dumps(
            {
                "version": 3,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [0],
                        "original": {"0": 0.26},
                        "target": {"0": 0.05},
                        "owned": {"0": [0.05]},
                        "legacy": [],
                        "phase": "restoring",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = RawMasterProfileBackend()
    backend.elements[1] = (virtual,)
    backend.values = {(1, virtual): 0.38, (1, 0): 0.05}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale(start_deferred=False)

    assert backend.values[(1, 0)] == pytest.approx(0.26)
    assert backend.values[(1, virtual)] == pytest.approx(0.38)
    assert not journal_path.exists()


def test_partial_v3_stereo_recovery_keeps_raw_profile_marker(
    tmp_path: Path,
) -> None:
    class RawStereoProfileBackend(FakeVolumeBackend):
        def volume_profile(self, device_id: int) -> tuple[int, ...]:
            return (1, 2)

    journal_path = tmp_path / "system-volume.json"
    created_at = time.time() - 30.0
    journal_path.write_text(
        json.dumps(
            {
                "version": 3,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [1, 2],
                        "original": {"1": 0.26, "2": 0.28},
                        "target": {"1": 0.05, "2": 0.05},
                        "owned": {"1": [0.05], "2": [0.05]},
                        "legacy": [],
                        "phase": "restoring",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = RawStereoProfileBackend()
    backend.elements[1] = (virtual,)
    backend.values = {
        (1, virtual): 0.38,
        (1, 1): 0.05,
        (1, 2): 0.05,
    }
    backend.fail_writes[(1, 2)] = 2
    first = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert first.recover_stale(start_deferred=False)
    pending = json.loads(journal_path.read_text(encoding="utf-8"))
    assert pending["version"] == system_volume._RECOVERY_VERSION
    assert pending["devices"][0]["allow_inactive_controls"] is True
    assert backend.values[(1, 1)] == pytest.approx(0.26)
    assert backend.values[(1, 2)] == pytest.approx(0.05)

    second = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )
    assert second.recover_stale(start_deferred=False)
    assert backend.values[(1, 2)] == pytest.approx(0.28)
    assert backend.values[(1, virtual)] == pytest.approx(0.38)
    assert not journal_path.exists()


def test_unavailable_v3_raw_master_does_not_block_changed_virtual_main(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    created_at = time.time() - 30.0
    journal_path.write_text(
        json.dumps(
            {
                "version": 3,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [0],
                        "original": {"0": 0.26},
                        "target": {"0": 0.05},
                        "owned": {"0": [0.05]},
                        "legacy": [],
                        "phase": "restoring",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend.elements[1] = (virtual,)
    backend.values = {(1, virtual): 0.38}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    token = ducker.begin()

    assert token is not None
    snapshot = ducker._snapshots[("uid-1", (virtual,))]
    assert snapshot.original[virtual] == pytest.approx(0.38)
    assert ducker.end(token)
    assert backend.values[(1, virtual)] == pytest.approx(0.38)
    assert not journal_path.exists()


def test_v3_stereo_recovery_finishes_before_virtual_main_capture(
    tmp_path: Path,
) -> None:
    class MigratingBackend(FakeVolumeBackend):
        def volume_profile(self, device_id: int) -> tuple[int, ...]:
            return (1, 2)

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            super().set_volume(device_id, element, value)
            if element in (1, 2):
                self.values[(
                    device_id,
                    system_volume._VIRTUAL_MAIN_ELEMENT,
                )] = max(
                    self.values[(device_id, 1)],
                    self.values[(device_id, 2)],
                )

    journal_path = tmp_path / "system-volume.json"
    created_at = time.time() - 30.0
    journal_path.write_text(
        json.dumps(
            {
                "version": 3,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [1, 2],
                        "original": {"1": 0.8, "2": 0.6},
                        "target": {"1": 0.05, "2": 0.05},
                        "owned": {"1": [0.05], "2": [0.05]},
                        "legacy": [],
                        "mute": {
                            "original": False,
                            "target": True,
                            "owned": True,
                        },
                        "phase": "ducked",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = MigratingBackend()
    backend.elements[1] = (system_volume._VIRTUAL_MAIN_ELEMENT,)
    backend.values = {
        (1, system_volume._VIRTUAL_MAIN_ELEMENT): 0.05,
        (1, 1): 0.05,
        (1, 2): 0.05,
    }
    backend.mutes[1] = True
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    snapshot = ducker._snapshots[
        ("uid-1", (system_volume._VIRTUAL_MAIN_ELEMENT,))
    ]
    assert snapshot.original[system_volume._VIRTUAL_MAIN_ELEMENT] == pytest.approx(
        0.8
    )
    assert ducker.end(token)
    assert backend.values[
        (1, system_volume._VIRTUAL_MAIN_ELEMENT)
    ] == pytest.approx(0.8)
    assert backend.mutes[1] is False


def test_unreadable_v3_stereo_blocks_virtual_main_recapture(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    created_at = time.time() - 30.0
    journal_path.write_text(
        json.dumps(
            {
                "version": 3,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [1, 2],
                        "original": {"1": 0.8, "2": 0.6},
                        "target": {"1": 0.05, "2": 0.05},
                        "owned": {"1": [0.05], "2": [0.05]},
                        "legacy": [],
                        "mute": {
                            "original": False,
                            "target": True,
                            "owned": True,
                        },
                        "phase": "ducked",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    backend.elements[1] = (system_volume._VIRTUAL_MAIN_ELEMENT,)
    backend.values = {
        (1, system_volume._VIRTUAL_MAIN_ELEMENT): 0.05,
    }
    backend.mutes[1] = True
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    assert ducker.begin(max_volume=0.0) is None
    assert ducker._active_token is None
    assert backend.values[
        (1, system_volume._VIRTUAL_MAIN_ELEMENT)
    ] == pytest.approx(0.05)
    assert backend.mutes[1] is False
    assert journal_path.exists()


def test_readable_v3_stereo_is_not_written_while_hfp_is_active(
    tmp_path: Path,
) -> None:
    class HfpWithStaleStereoBackend(FakeVolumeBackend):
        def volume_profile(self, device_id: int) -> tuple[int, ...]:
            return (0,)

    journal_path = tmp_path / "system-volume.json"
    created_at = time.time() - 30.0
    journal_path.write_text(
        json.dumps(
            {
                "version": 3,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [1, 2],
                        "original": {"1": 0.8, "2": 0.6},
                        "target": {"1": 0.05, "2": 0.05},
                        "owned": {"1": [0.05], "2": [0.05]},
                        "legacy": [],
                        "phase": "ducked",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = HfpWithStaleStereoBackend()
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend.elements[1] = (virtual,)
    backend.values = {
        (1, virtual): 0.05,
        (1, 1): 0.05,
        (1, 2): 0.05,
    }
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    assert ducker.begin() is None
    assert backend.values[(1, 1)] == pytest.approx(0.05)
    assert backend.values[(1, 2)] == pytest.approx(0.05)
    assert backend.writes == []
    assert journal_path.exists()


@pytest.mark.parametrize("override", ["scalar", "unmute"])
def test_post_a2dp_sync_preserves_manual_override(override: str) -> None:
    backend = BluetoothGainBackend()
    injected = False
    writes_at_override = 0
    events_at_override = 0

    def _sleep(delay: float) -> None:
        nonlocal injected, writes_at_override, events_at_override
        if (
            injected
            or not backend.local_original_written
            or delay != pytest.approx(
                system_volume._POST_RESTORE_SYNC_DELAYS[0]
            )
        ):
            return
        injected = True
        if override == "scalar":
            backend.values[(1, 0)] = 0.3
        else:
            backend.mutes[1] = False
        writes_at_override = len(backend.writes)
        events_at_override = len(backend.events)

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.restore_started = True

    assert ducker.end(token)
    assert injected
    if override == "scalar":
        assert len(backend.writes) == writes_at_override
        assert all(
            event[0] != "volume"
            for event in backend.events[events_at_override:]
        )
        assert backend.values[(1, 0)] == pytest.approx(0.3)
        assert backend.mutes[1] is False
    else:
        assert backend.values[(1, 0)] == pytest.approx(0.8)
        assert backend.mutes[1] is False
        assert len(backend.writes) > writes_at_override


def test_post_a2dp_sync_recovers_route_unmute_after_scalar_recheck() -> None:
    class LateMuteBackend(BluetoothGainBackend):
        arm_late_mute = False
        post_sync_reads = 0

        def get_volume(self, device_id: int, element: int) -> float:
            value = super().get_volume(device_id, element)
            if self.arm_late_mute and device_id == self.tracked_device:
                self.post_sync_reads += 1
                if self.post_sync_reads == 2:
                    # This lands after the earlier mute check and immediately
                    # before the final route validation.
                    self.mutes[device_id] = False
            return value

    backend = LateMuteBackend()
    events_before_override = 0

    def _sleep(delay: float) -> None:
        nonlocal events_before_override
        if (
            backend.local_original_written
            and delay
            == pytest.approx(system_volume._POST_RESTORE_SYNC_DELAYS[0])
        ):
            backend.arm_late_mute = True
            events_before_override = len(backend.events)

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.restore_started = True

    assert ducker.end(token)
    assert backend.post_sync_reads >= 2
    assert backend.mutes[1] is False
    events_after_override = backend.events[events_before_override:]
    assert ("mute", 1, True) in events_after_override
    assert events_after_override[-1] == ("mute", 1, False)


def test_post_a2dp_sync_retries_a_temporarily_unavailable_route(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = BluetoothGainBackend()
    post_waits = 0
    route_retries = 0
    unavailable_seen = False
    ducker: SystemOutputDucker
    token: DuckToken | None = None

    def _sleep(delay: float) -> None:
        nonlocal post_waits, route_retries, unavailable_seen
        if (
            backend.default_device is None
            and post_waits >= 1
            and delay == pytest.approx(system_volume._RESTORE_RETRY_DELAY)
        ):
            route_retries += 1
            backend.default_device = 1
            backend.a2dp_active = True
            return
        if (
            backend.local_original_written
            and not backend.a2dp_active
            and delay
            == pytest.approx(system_volume._POST_RESTORE_SYNC_DELAYS[0])
        ):
            post_waits += 1
            assert post_waits == 1
            backend.default_device = None
            unavailable_seen = True
            assert ducker._active_token is token
            assert journal_path.exists()

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.a2dp_active = False
    backend.restore_started = True

    assert ducker.end(token)
    assert unavailable_seen
    assert post_waits == 1
    assert route_retries >= 1
    assert backend.original_writes_while_a2dp_active >= 1
    assert backend.effective_gain == pytest.approx(0.8)
    assert ducker._active_token is None
    assert not journal_path.exists()


def test_post_a2dp_sync_write_failure_keeps_ownership_for_retry(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = BluetoothGainBackend()
    failures_injected = False

    def _sleep(delay: float) -> None:
        nonlocal failures_injected
        if (
            failures_injected
            or not backend.local_original_written
            or delay != pytest.approx(
                system_volume._POST_RESTORE_SYNC_DELAYS[0]
            )
        ):
            return
        failures_injected = True
        backend.a2dp_active = True
        backend.fail_writes[(1, 0)] = 3

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.a2dp_active = False
    backend.restore_started = True

    assert not ducker.end(token)
    assert failures_injected
    assert ducker._active_token is token
    assert journal_path.exists()
    assert backend.effective_gain == pytest.approx(0.05)

    assert ducker.restore_all()
    assert backend.effective_gain == pytest.approx(0.8)
    assert ducker._active_token is None
    assert not journal_path.exists()


def test_post_a2dp_sync_unmute_failure_keeps_ownership_for_retry(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = BluetoothGainBackend()
    failures_injected = False

    def _sleep(delay: float) -> None:
        nonlocal failures_injected
        if (
            failures_injected
            or not backend.local_original_written
            or delay != pytest.approx(
                system_volume._POST_RESTORE_SYNC_DELAYS[0]
            )
        ):
            return
        failures_injected = True
        backend.a2dp_active = True
        backend.fail_mute_writes[1] = 3

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.a2dp_active = False
    backend.restore_started = True

    assert not ducker.end(token)
    assert failures_injected
    assert ducker._active_token is token
    assert journal_path.exists()
    # The visible and remote scalar are already repaired, but the failed
    # final unmute keeps the route silent and the ownership journal live.
    assert backend.effective_gain == pytest.approx(0.8)
    assert backend.mutes[1] is True

    assert ducker.restore_all()
    assert backend.effective_gain == pytest.approx(0.8)
    assert ducker._active_token is None
    assert not journal_path.exists()


def test_post_restore_wait_keeps_token_without_holding_lifecycle_lock() -> None:
    backend = BluetoothGainBackend()
    wait_started = threading.Event()
    release_wait = threading.Event()
    blocked_once = False

    def _sleep(delay: float) -> None:
        nonlocal blocked_once
        if (
            blocked_once
            or not backend.local_original_written
            or delay != pytest.approx(
                system_volume._POST_RESTORE_SYNC_DELAYS[0]
            )
        ):
            return
        blocked_once = True
        wait_started.set()
        assert release_wait.wait(timeout=2.0)

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.restore_started = True
    result: list[bool] = []
    thread = threading.Thread(target=lambda: result.append(ducker.end(token)))
    thread.start()
    assert wait_started.wait(timeout=1.0)

    started_at = time.monotonic()
    with pytest.raises(system_volume.SystemVolumeBusyError):
        ducker.begin(max_volume=0.0)
    assert time.monotonic() - started_at < 0.1

    release_wait.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert result == [True]


def test_zero_max_volume_reasserts_mute_after_hfp_resets_it(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.elements[1] = (1, 2)
    backend.values = {(1, 1): 0.23, (1, 2): 0.23}

    token = ducker.begin(max_volume=0.0)
    assert token is not None
    assert backend.mutes[1] is True
    assert backend.values[(1, 1)] == pytest.approx(0.05)

    # AirPods keeps the exposed A2DP topology at its quiet target while HFP
    # separately clears master mute and clamps hidden call volume to 5%.
    backend.mutes[1] = False
    backend.values[(1, 1)] = 0.18
    backend.values[(1, 2)] = 0.18
    write_count = len(backend.writes)
    backend.events.clear()

    assert ducker.refresh(token)
    assert backend.events[0] == ("mute", 1, True)
    assert backend.mutes[1] is True
    assert len(backend.writes) > write_count
    assert ducker.end(token)


def test_zero_max_volume_mutes_new_route_before_lowering_scalar(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.default_device = 2
    backend.elements[2] = (0,)
    backend.values[(2, 0)] = 0.6
    backend.device_uids[2] = "uid-2"
    backend.mutes[2] = False
    backend.events.clear()

    assert ducker.refresh(token)

    assert backend.events[0] == ("mute", 2, True)
    assert any(event[0] == "volume" for event in backend.events[1:])
    assert ducker.end(token)
    assert backend.values[(2, 0)] == pytest.approx(0.6)
    assert backend.mutes[2] is False


def test_zero_max_volume_mutes_recreated_route_before_lowering_scalar(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.default_device = 2
    backend.elements = {2: (0,)}
    backend.values = {(2, 0): 0.65}
    backend.device_uids = {2: "uid-1"}
    backend.mutes = {2: False}
    backend.events.clear()

    assert ducker.refresh(token)

    assert backend.events[0] == ("mute", 2, True)
    assert any(event[0] == "volume" for event in backend.events[1:])
    assert ducker.end(token)
    assert backend.values[(2, 0)] == pytest.approx(0.8)
    assert backend.mutes[2] is False


def test_monitor_reasserts_mute_during_hfp_route_settle() -> None:
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(backend, sleeper=lambda _delay: None)

    token = ducker.begin(max_volume=0.0)
    assert token is not None
    assert backend.mutes[1] is True
    initial_mute_writes = len(backend.mute_writes)

    # HFP setup resets this independently of the A2DP scalar controls. The
    # existing short-lived route monitor must close that race without relying
    # on a caller knowing exactly when CoreAudio switched profiles.
    backend.mutes[1] = False
    deadline = time.monotonic() + 1.0
    while backend.mutes[1] is not True and time.monotonic() < deadline:
        time.sleep(0.005)

    assert backend.mutes[1] is True
    assert len(backend.mute_writes) > initial_mute_writes
    assert ducker.end(token)


def test_zero_max_volume_preserves_an_existing_user_mute(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.mutes[1] = True

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    assert backend.mutes[1] is True
    assert backend.mute_writes == []
    assert ducker.end(token)
    assert backend.mutes[1] is True
    assert backend.mute_writes == []


def test_zero_max_volume_falls_back_to_scalar_when_mute_is_unsupported(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.mutes[1] = None

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.0)
    assert backend.mute_writes == []
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_user_unmute_after_route_settle_is_not_overridden(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.mutes[1] = False

    with ducker._lock:
        ducker._refresh_locked(
            allow_reduck=False,
            abandon_on_deviation=True,
            force_mute=False,
        )

    snapshot = ducker._snapshots[("uid-1", (0,))]
    assert snapshot.mute_target is None
    assert backend.mutes[1] is False
    mute_write_count = len(backend.mute_writes)
    assert ducker.end(token)
    assert len(backend.mute_writes) == mute_write_count


def test_mute_write_failure_rolls_volume_back(
    backend: FakeVolumeBackend,
    ducker: SystemOutputDucker,
) -> None:
    backend.fail_mute_writes[1] = 1

    assert ducker.begin(max_volume=0.0) is None
    assert backend.mutes[1] is False
    assert backend.values[(1, 0)] == pytest.approx(0.8)


def test_mute_restore_failure_keeps_journal_for_retry(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    assert backend.mutes[1] is True

    # end() performs one bounded retry. Both writes fail before changing the
    # hardware, so ownership must remain durable for restore_all().
    backend.fail_mute_writes[1] = 2
    assert not ducker.end(token)
    assert backend.mutes[1] is True
    assert journal_path.exists()

    assert ducker.restore_all()
    assert backend.mutes[1] is False
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_mute_restore_accepts_a_write_that_succeeds_then_raises(
    tmp_path: Path,
) -> None:
    class WriteThenRaiseBackend(FakeVolumeBackend):
        raise_after_unmute = True

        def set_mute(self, device_id: int, muted: bool) -> None:
            super().set_mute(device_id, muted)
            if not muted and self.raise_after_unmute:
                self.raise_after_unmute = False
                raise RuntimeError("injected post-write mute failure")

    journal_path = tmp_path / "system-volume.json"
    backend = WriteThenRaiseBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None

    assert ducker.end(token)
    assert backend.raise_after_unmute is False
    assert backend.mutes[1] is False
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_unmute_route_disappearance_retains_ownership_until_readback(
    tmp_path: Path,
) -> None:
    class DisappearAfterUnmuteBackend(FakeVolumeBackend):
        route_available = True

        def device_id_for_uid(self, device_uid: str) -> int | None:
            if not self.route_available:
                return None
            return super().device_id_for_uid(device_uid)

        def set_mute(self, device_id: int, muted: bool) -> None:
            super().set_mute(device_id, muted)
            if not muted:
                self.route_available = False

    journal_path = tmp_path / "system-volume.json"
    backend = DisappearAfterUnmuteBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    assert ducker._monitor_stop is not None
    ducker._monitor_stop.set()
    assert ducker._monitor_thread is not None
    ducker._monitor_thread.join(timeout=1.0)

    with ducker._lock:
        restored, unavailable, write_failed, delay = (
            ducker._restore_pending_locked(
                token,
                defer_unavailable=False,
            )
        )
    assert not restored
    assert not unavailable
    assert write_failed
    assert delay is None
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.mutes[1] is False
    assert journal_path.exists()
    assert ducker._active_token is token

    backend.route_available = True
    with ducker._lock:
        restored, unavailable, write_failed, delay = (
            ducker._restore_pending_locked(
                token,
                defer_unavailable=False,
            )
        )
    assert restored
    assert not unavailable
    assert not write_failed
    assert delay is None
    assert not journal_path.exists()


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


def test_monitor_freezes_media_snapshot_during_call_profile() -> None:
    """An A2DP→HFP flip must not be misread as a user volume change.

    HFP exposes an independent volume scale under the same element layout,
    so the ducked media value is invisible there. The monitor must freeze
    instead of abandoning the media original (which orphaned the duck and
    left music stuck quiet after recording).
    """
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = FakeVolumeBackend()
    backend.elements = {1: (virtual,)}
    backend.values = {(1, virtual): 0.8}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin()
    assert token is not None
    assert backend.values[(1, virtual)] == pytest.approx(0.05)

    # Microphone opened: the headset flips to its call profile, whose scale
    # reads a foreign value through the same virtual-main element.
    backend.route_signatures[1] = (24_000, 1)
    backend.values[(1, virtual)] = 0.7
    with ducker._lock:
        assert ducker._refresh_locked(
            allow_reduck=False,
            abandon_on_deviation=True,
            force_mute=False,
        )
    assert ("uid-1", (virtual,)) in ducker._snapshots
    assert "uid-1" not in ducker._overridden_devices
    assert backend.values[(1, virtual)] == pytest.approx(0.7)

    # Media profile returns (still ducked); the original must be restored.
    backend.route_signatures[1] = (48_000, 2)
    backend.values[(1, virtual)] = 0.05
    assert ducker.end(token)
    assert backend.values[(1, virtual)] == pytest.approx(0.8)


def test_frozen_call_profile_still_reasserts_hard_mute() -> None:
    """Hard-duck (max_volume=0) must keep silencing audio during HFP.

    CoreAudio clears the device mute as HFP comes up. The route freeze must
    still reassert the mute we own (so background music stays at 0 while
    recording) without touching the frozen A2DP scalar snapshot.
    """
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = FakeVolumeBackend()
    backend.elements = {1: (virtual,)}
    backend.values = {(1, virtual): 0.8}
    backend.mutes = {1: False}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    assert backend.mutes[1] is True  # begin silenced A2DP

    # Microphone opens: HFP comes up and CoreAudio clears the mute.
    backend.route_signatures[1] = (24_000, 1)
    backend.mutes[1] = False
    writes_before = len(backend.writes)
    with ducker._lock:
        assert ducker._refresh_locked(
            allow_reduck=False,
            abandon_on_deviation=True,
            force_mute=True,
        )
    # Re-silenced, snapshot preserved, and no scalar write on the HFP scale.
    assert backend.mutes[1] is True
    assert ("uid-1", (virtual,)) in ducker._snapshots
    assert "uid-1" not in ducker._overridden_devices
    assert len(backend.writes) == writes_before

    # Media profile returns; end() restores both volume and mute.
    backend.route_signatures[1] = (48_000, 2)
    assert ducker.end(token)
    assert backend.values[(1, virtual)] == pytest.approx(0.8)
    assert backend.mutes[1] is False


def test_frozen_call_profile_does_not_fight_user_after_fast_window() -> None:
    """Past the fast window (force_mute=False) the freeze leaves mute alone."""
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = FakeVolumeBackend()
    backend.elements = {1: (virtual,)}
    backend.values = {(1, virtual): 0.8}
    backend.mutes = {1: False}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None

    backend.route_signatures[1] = (24_000, 1)
    backend.mutes[1] = False  # user unmuted to hear music mid-recording
    with ducker._lock:
        assert ducker._refresh_locked(
            allow_reduck=False,
            abandon_on_deviation=True,
            force_mute=False,
        )
    assert backend.mutes[1] is False  # not fought

    backend.route_signatures[1] = (48_000, 2)
    assert ducker.end(token)


def test_monitor_does_not_capture_transient_call_profile_topology() -> None:
    """The brief HFP-only element layout must not become an owned snapshot.

    A snapshot captured on the call topology can never be resolved once the
    media profile returns; it used to strand the recovery journal and left
    every later session re-adopting an unrestorable duck.
    """
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = FakeVolumeBackend()
    backend.elements = {1: (virtual,)}
    backend.values = {(1, virtual): 0.8}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin()
    assert token is not None

    # Transition window: the call profile briefly exposes only the raw
    # main element with its own scale.
    backend.route_signatures[1] = (24_000, 1)
    backend.elements = {1: (0,)}
    backend.values[(1, 0)] = 0.0625
    with ducker._lock:
        assert ducker._refresh_locked(
            allow_reduck=False,
            abandon_on_deviation=True,
            force_mute=False,
        )
    assert ("uid-1", (0,)) not in ducker._snapshots
    assert backend.values[(1, 0)] == pytest.approx(0.0625)

    backend.route_signatures[1] = (48_000, 2)
    backend.elements = {1: (virtual,)}
    assert ducker.end(token)
    assert backend.values[(1, virtual)] == pytest.approx(0.8)


def test_begin_leaves_call_profile_snapshot_deferred_on_media_route() -> None:
    """A call-scale original must never be adopted into a media session."""
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = FakeVolumeBackend()
    backend.elements = {1: (virtual,)}
    backend.values = {(1, virtual): 0.0625}
    backend.route_signatures[1] = (24_000, 1)
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    first = ducker.begin()
    assert first is not None
    snapshot = ducker._snapshots[("uid-1", (virtual,))]
    assert snapshot.capture_route == (24_000, 1)

    # Restore fails while the call profile is still up; the snapshot is
    # parked for background recovery.
    backend.fail_writes[(1, virtual)] = 100
    assert not ducker.end(first)
    assert ducker.defer_failed_restore(first)
    assert ("uid-1", (virtual,)) in ducker._deferred_snapshots
    backend.fail_writes.clear()

    # Media profile is back with its own scale; the parked call-scale
    # values must stay parked instead of becoming this session's original.
    backend.route_signatures[1] = (48_000, 2)
    backend.values[(1, virtual)] = 0.8
    second = ducker.begin()
    assert second is None
    assert ("uid-1", (virtual,)) in ducker._deferred_snapshots
    assert ("uid-1", (virtual,)) not in ducker._snapshots
    assert backend.values[(1, virtual)] == pytest.approx(0.8)
    ducker.stop_background_workers()


def test_media_snapshot_survives_sample_rate_renegotiation() -> None:
    """44.1k/48k renegotiation stays within the media class: no freeze."""
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = FakeVolumeBackend()
    backend.elements = {1: (virtual,)}
    backend.values = {(1, virtual): 0.8}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
    )
    token = ducker.begin()
    assert token is not None

    backend.route_signatures[1] = (44_100, 2)
    backend.values[(1, virtual)] = 0.6
    with ducker._lock:
        # Same media class: an off-target value is still a user change and
        # keeps the established abandonment contract.
        assert not ducker._refresh_locked(
            allow_reduck=False,
            abandon_on_deviation=True,
            force_mute=False,
        )
    assert "uid-1" in ducker._overridden_devices

    assert ducker.end(token)
    assert backend.values[(1, virtual)] == pytest.approx(0.6)


def test_recovery_journal_round_trips_capture_route(tmp_path: Path) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = FakeVolumeBackend()
    backend.elements = {1: (virtual,)}
    backend.values = {(1, virtual): 0.8}
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin()
    assert token is not None
    data = json.loads(journal_path.read_text(encoding="utf-8"))
    assert data["devices"][0]["capture_route"] == [48_000, 2]

    loaded = ducker._read_recovery_journal()
    assert loaded is not None
    _created_at, snapshots = loaded
    assert snapshots[("uid-1", (virtual,))].capture_route == (48_000, 2)

    # A journal written before capture routes existed keeps loading and
    # falls back to route-agnostic behavior.
    del data["devices"][0]["capture_route"]
    journal_path.write_text(json.dumps(data), encoding="utf-8")
    loaded = ducker._read_recovery_journal()
    assert loaded is not None
    _created_at, snapshots = loaded
    assert snapshots[("uid-1", (virtual,))].capture_route is None

    assert ducker.end(token)


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
    assert journal["version"] == system_volume._RECOVERY_VERSION
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


def test_zero_max_volume_journals_and_recovers_mute_ownership(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    created_at = time.time() - 60.0
    journal_path.write_text(
        json.dumps(
            {
                "version": system_volume._RECOVERY_VERSION,
                "created_at": created_at,
                "devices": [
                    {
                        "device_uid": "uid-1",
                        "device_id_hint": 1,
                        "elements": [0],
                        "original": {"0": 0.8},
                        "target": {"0": 0.05},
                        "owned": {"0": [0.05]},
                        "legacy": [],
                        "mute": {
                            "original": False,
                            "target": True,
                            "owned": True,
                        },
                        "phase": "ducked",
                        "transition_started_at": created_at,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    backend.mutes[1] = True
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )

    assert ducker.recover_stale()
    assert backend.mute_writes[0] == (1, False)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.mutes[1] is False
    assert not journal_path.exists()


def test_mute_write_that_succeeds_then_raises_is_rolled_back(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class WriteThenRaiseBackend(FakeVolumeBackend):
        should_raise = True

        def set_mute(self, device_id: int, muted: bool) -> None:
            super().set_mute(device_id, muted)
            if muted and self.should_raise:
                self.should_raise = False
                raise RuntimeError("injected post-write failure")

    backend = WriteThenRaiseBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )

    assert ducker.begin(max_volume=0.0) is None
    assert backend.mutes[1] is False
    assert backend.values[(1, 0)] == pytest.approx(0.8)
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

    worker = ducker._deferred_sync_thread
    assert worker is not None
    worker.join(timeout=1.0)
    assert not worker.is_alive()
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

    worker = ducker._deferred_sync_thread
    assert worker is not None
    worker.join(timeout=1.0)
    assert not worker.is_alive()
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


def test_deferred_worker_restores_disconnected_route_after_reconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class DisconnectingBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.unavailable = False

        def device_id_for_uid(self, device_uid: str) -> int | None:
            if self.unavailable:
                return None
            return super().device_id_for_uid(device_uid)

    backend = DisconnectingBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_INTERVAL", 0.02)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_FAST_DURATION", 0.0)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_SLOW_INTERVAL", 0.02)

    token = ducker.begin(max_volume=0.0)
    assert token is not None
    assert backend.mutes[1] is True
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    backend.unavailable = True

    assert ducker.end(token)
    pending = json.loads(journal_path.read_text(encoding="utf-8"))
    post_restore = pending["devices"][0].get("post_restore")
    assert post_restore is None or post_restore["ready"] is False
    worker = ducker._deferred_sync_thread
    assert worker is not None
    assert worker.is_alive()

    backend.unavailable = False
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert backend.mutes[1] is False
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


@pytest.mark.parametrize("user_override", [None, 0.3])
def test_deferred_worker_restores_scalar_only_route_after_reconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    user_override: float | None,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class DisconnectingBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.unavailable = False
            self.mutes[1] = None

        def device_id_for_uid(self, device_uid: str) -> int | None:
            if self.unavailable:
                return None
            return super().device_id_for_uid(device_uid)

    backend = DisconnectingBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_INTERVAL", 0.02)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_FAST_DURATION", 0.0)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_SLOW_INTERVAL", 0.02)

    token = ducker.begin(max_volume=0.05)
    assert token is not None
    backend.unavailable = True
    assert ducker.end(token)
    worker = ducker._deferred_sync_thread
    assert worker is not None
    assert worker.is_alive()

    writes_before_reconnect = len(backend.writes)
    if user_override is not None:
        backend.values[(1, 0)] = user_override
    backend.unavailable = False
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert backend.mute_writes == []
    if user_override is None:
        assert backend.values[(1, 0)] == pytest.approx(0.8)
        assert len(backend.writes) > writes_before_reconnect
    else:
        assert backend.values[(1, 0)] == pytest.approx(user_override)
        assert len(backend.writes) == writes_before_reconnect
    assert not journal_path.exists()


@pytest.mark.parametrize("second_max_volume", [0.0, 0.05])
def test_begin_adopts_muted_hard_restore_without_audible_full_gain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_max_volume: float,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class BlockingReconnectBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.unavailable = False
            self.block_restore = False
            self.restore_write_entered = threading.Event()
            self.allow_restore_write = threading.Event()
            self.volume_events: list[tuple[str, float, bool | None]] = []

        def device_id_for_uid(self, device_uid: str) -> int | None:
            if self.unavailable:
                return None
            return super().device_id_for_uid(device_uid)

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            current = self.values[(device_id, element)]
            if self.block_restore and value > current and not self.restore_write_entered.is_set():
                self.restore_write_entered.set()
                assert self.allow_restore_write.wait(timeout=1.0)
            self.volume_events.append(
                (threading.current_thread().name, value, self.mutes[device_id])
            )
            super().set_volume(device_id, element, value)

    backend = BlockingReconnectBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_INTERVAL", 0.01)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_FAST_DURATION", 0.0)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_SLOW_INTERVAL", 0.01)

    first = ducker.begin(max_volume=0.0)
    assert first is not None
    backend.unavailable = True
    assert ducker.end(first)
    worker = ducker._deferred_sync_thread
    assert worker is not None

    backend.block_restore = True
    backend.unavailable = False
    assert backend.restore_write_entered.wait(timeout=1.0)
    begin_entered = threading.Event()
    real_begin_with_fence = ducker._begin_with_fence

    def _begin_with_fence(factor: float, max_volume: float):
        begin_entered.set()
        return real_begin_with_fence(factor, max_volume)

    monkeypatch.setattr(ducker, "_begin_with_fence", _begin_with_fence)
    result: list[DuckToken | None] = []
    begin_thread = threading.Thread(
        target=lambda: result.append(
            ducker.begin(max_volume=second_max_volume)
        ),
        name="new-recording-begin",
    )
    begin_thread.start()
    assert begin_entered.wait(timeout=1.0)
    backend.allow_restore_write.set()
    begin_thread.join(timeout=2.0)

    assert not begin_thread.is_alive()
    assert result and result[0] is not None
    token = result[0]
    assert token is not None
    upward_events = [
        event
        for event in backend.volume_events
        if event[0] == "system-volume-deferred-sync" and event[1] > 0.05
    ]
    assert upward_events
    assert all(muted is True for _thread, _value, muted in upward_events)
    assert backend.values[(1, 0)] == pytest.approx(second_max_volume or 0.05)
    assert backend.mutes[1] is (second_max_volume == 0.0)

    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.mutes[1] is False
    assert not journal_path.exists()


def test_begin_fences_scalar_only_hard_restore_after_one_inflight_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class BlockingReconnectBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.unavailable = False
            self.mutes[1] = None
            self.block_restore = False
            self.restore_write_entered = threading.Event()
            self.allow_restore_write = threading.Event()
            self.worker_writes: list[float] = []

        def device_id_for_uid(self, device_uid: str) -> int | None:
            if self.unavailable:
                return None
            return super().device_id_for_uid(device_uid)

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            current = self.values[(device_id, element)]
            is_worker = threading.current_thread().name == "system-volume-deferred-sync"
            if self.block_restore and is_worker and value > current:
                self.worker_writes.append(value)
                if not self.restore_write_entered.is_set():
                    self.restore_write_entered.set()
                    assert self.allow_restore_write.wait(timeout=1.0)
            super().set_volume(device_id, element, value)

    backend = BlockingReconnectBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_INTERVAL", 0.01)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_FAST_DURATION", 0.0)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_SLOW_INTERVAL", 0.01)

    first = ducker.begin(max_volume=0.05)
    assert first is not None
    backend.unavailable = True
    assert ducker.end(first)
    backend.block_restore = True
    backend.unavailable = False
    assert backend.restore_write_entered.wait(timeout=1.0)
    begin_entered = threading.Event()
    real_begin_with_fence = ducker._begin_with_fence

    def _begin_with_fence(factor: float, max_volume: float):
        begin_entered.set()
        return real_begin_with_fence(factor, max_volume)

    monkeypatch.setattr(ducker, "_begin_with_fence", _begin_with_fence)
    result: list[DuckToken | None] = []
    begin_thread = threading.Thread(
        target=lambda: result.append(ducker.begin(max_volume=0.05)),
        name="new-recording-begin",
    )
    begin_thread.start()
    assert begin_entered.wait(timeout=1.0)
    backend.allow_restore_write.set()
    begin_thread.join(timeout=2.0)

    assert not begin_thread.is_alive()
    assert len(backend.worker_writes) == 1
    assert backend.worker_writes[0] < 0.8
    assert result and result[0] is not None
    token = result[0]
    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.05)

    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_timed_out_begin_aborts_recording_and_hands_worker_to_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class BlockingRestoreBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.mutes[1] = None
            self.restore_entered = threading.Event()
            self.release_restore = threading.Event()
            self.active_writers = 0
            self.max_active_writers = 0

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            current = self.values[(device_id, element)]
            if value > current and not self.restore_entered.is_set():
                self.active_writers += 1
                self.max_active_writers = max(
                    self.max_active_writers,
                    self.active_writers,
                )
                self.restore_entered.set()
                try:
                    assert self.release_restore.wait(timeout=2.0)
                finally:
                    self.active_writers -= 1
            super().set_volume(device_id, element, value)

    backend = BlockingRestoreBackend()
    backend.values[(1, 0)] = 0.05
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
        phase=system_volume._PHASE_DUCKED,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_JOIN_TIMEOUT", 0.05)
    monkeypatch.setattr(system_volume, "_LIFECYCLE_LOCK_TIMEOUT", 0.02)
    ducker._start_deferred_sync_if_needed()
    old_worker = ducker._deferred_sync_thread
    assert old_worker is not None
    assert backend.restore_entered.wait(timeout=1.0)

    started_at = time.monotonic()
    with pytest.raises(system_volume.SystemVolumeBusyError):
        ducker.begin(max_volume=0.05)
    assert time.monotonic() - started_at < 0.2
    assert backend.max_active_writers == 1
    assert old_worker.is_alive()

    backend.release_restore.set()
    old_worker.join(timeout=1.0)
    assert not old_worker.is_alive()
    deadline = time.monotonic() + 1.0
    replacement = ducker._deferred_sync_thread
    while (
        (replacement is None or replacement is old_worker)
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)
        replacement = ducker._deferred_sync_thread
    assert replacement is not None
    assert replacement is not old_worker
    replacement.join(timeout=1.0)

    assert not replacement.is_alive()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.max_active_writers == 1
    assert not journal_path.exists()


def test_timed_out_begin_persistently_cancels_explicit_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class BlockingRestoreBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.mutes[1] = None
            self.restore_entered = threading.Event()
            self.release_restore = threading.Event()
            self.recover_writes: list[float] = []

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            if threading.current_thread().name == "explicit-recovery":
                self.recover_writes.append(value)
                if not self.restore_entered.is_set():
                    self.restore_entered.set()
                    assert self.release_restore.wait(timeout=2.0)
            super().set_volume(device_id, element, value)

    backend = BlockingRestoreBackend()
    backend.values[(1, 0)] = 0.05
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
        phase=system_volume._PHASE_DUCKED,
    )
    first = SystemOutputDucker(backend, recovery_path=journal_path)
    first._deferred_snapshots[first._snapshot_key(snapshot)] = snapshot
    first._write_recovery_journal()
    recovery = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )
    restart_set_entered = threading.Event()
    allow_restart_set = threading.Event()
    worker_started = threading.Event()
    allow_worker = threading.Event()

    class GateEvent(threading.Event):
        def set(self) -> None:
            super().set()
            if threading.current_thread().name == "timed-out-begin":
                restart_set_entered.set()
                assert allow_restart_set.wait(timeout=2.0)

    recovery._deferred_restart_needed = GateEvent()
    original_run_deferred = recovery._run_deferred_sync

    def _gated_run_deferred(stop: threading.Event) -> None:
        worker_started.set()
        assert allow_worker.wait(timeout=2.0)
        original_run_deferred(stop)

    monkeypatch.setattr(recovery, "_run_deferred_sync", _gated_run_deferred)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_JOIN_TIMEOUT", 0.05)
    monkeypatch.setattr(system_volume, "_LIFECYCLE_LOCK_TIMEOUT", 0.02)
    results: list[bool] = []
    recovery_thread = threading.Thread(
        target=lambda: results.append(recovery.recover_stale()),
        name="explicit-recovery",
    )
    recovery_thread.start()
    assert backend.restore_entered.wait(timeout=1.0)

    begin_errors: list[BaseException] = []

    def _timed_out_begin() -> None:
        try:
            recovery.begin(max_volume=0.05)
        except BaseException as exc:
            begin_errors.append(exc)

    begin_thread = threading.Thread(
        target=_timed_out_begin,
        name="timed-out-begin",
    )
    begin_thread.start()
    try:
        assert restart_set_entered.wait(timeout=1.0)
        backend.release_restore.set()
        recovery_thread.join(timeout=1.0)

        assert not recovery_thread.is_alive()
        assert results == [False]
        assert len(backend.recover_writes) == 1
        assert recovery._deferred_restart_needed.is_set()
        assert not worker_started.is_set()

        # Only lowering the begin fence may hand the still-pending journal to
        # a successor. The recovery thread must not consume that request.
        allow_restart_set.set()
        begin_thread.join(timeout=1.0)
        assert not begin_thread.is_alive()
        assert len(begin_errors) == 1
        assert isinstance(begin_errors[0], system_volume.SystemVolumeBusyError)
        assert worker_started.wait(timeout=1.0)
        worker = recovery._deferred_sync_thread
        assert worker is not None
        assert worker.is_alive()
    finally:
        backend.release_restore.set()
        allow_restart_set.set()
        allow_worker.set()
        recovery_thread.join(timeout=1.0)
        begin_thread.join(timeout=1.0)

    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_terminal_stop_blocks_cached_deferred_worker_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    snapshot = system_volume._DeviceSnapshot(
        device_uid="disconnected-output",
        device_id_hint=99,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
    )
    ducker = SystemOutputDucker(backend, recovery_path=journal_path)
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()
    stop = threading.Event()
    stop.set()
    restart_entered = threading.Event()
    allow_restart = threading.Event()
    original_start = ducker._start_deferred_sync_if_needed

    def _blocked_restart() -> None:
        restart_entered.set()
        assert allow_restart.wait(timeout=2.0)
        original_start()

    monkeypatch.setattr(ducker, "_start_deferred_sync_if_needed", _blocked_restart)
    worker = threading.Thread(
        target=ducker._run_deferred_sync,
        args=(stop,),
        name="system-volume-deferred-sync",
        daemon=True,
    )
    ducker._deferred_sync_thread = worker
    ducker._deferred_sync_stop = stop
    ducker._deferred_sync_pair = (worker, stop)
    ducker._deferred_restart_needed.set()
    worker.start()
    try:
        assert restart_entered.wait(timeout=1.0)
        assert ducker._deferred_sync_pair is None
        ducker.stop_background_workers()
        allow_restart.set()
        worker.join(timeout=1.0)
    finally:
        allow_restart.set()
        worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert ducker._background_workers_stopped.is_set()
    assert ducker._deferred_sync_thread is None
    assert ducker._deferred_sync_pair is None
    assert not any(
        thread.name == "system-volume-deferred-sync"
        for thread in threading.enumerate()
    )


def test_explicit_recovery_without_deferred_restart_fences_old_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    snapshot = system_volume._DeviceSnapshot(
        device_uid="disconnected-output",
        device_id_hint=99,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
    )
    ducker = SystemOutputDucker(backend, recovery_path=journal_path)
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_INTERVAL", 0.01)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_FAST_DURATION", 0.0)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_SLOW_INTERVAL", 0.01)

    ducker._start_deferred_sync_if_needed()
    old_worker = ducker._deferred_sync_thread
    assert old_worker is not None
    assert old_worker.is_alive()

    assert not ducker.recover_stale(start_deferred=False)
    old_worker.join(timeout=1.0)

    assert not old_worker.is_alive()
    assert not ducker._explicit_recovery_requested.is_set()
    assert ducker._deferred_sync_thread is None
    assert ducker._deferred_sync_pair is None
    assert journal_path.exists()
    assert ducker._deferred_snapshots


def test_worker_start_cannot_consume_concurrent_restart_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    snapshot = system_volume._DeviceSnapshot(
        device_uid="disconnected-output",
        device_id_hint=99,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
    )
    ducker = SystemOutputDucker(backend, recovery_path=journal_path)
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()
    start_entered = threading.Event()
    allow_start_return = threading.Event()
    restart_set_entered = threading.Event()
    allow_restart_set = threading.Event()
    real_thread = threading.Thread

    class GateEvent(threading.Event):
        def set(self) -> None:
            super().set()
            if threading.current_thread().name == "racing-begin":
                restart_set_entered.set()
                assert allow_restart_set.wait(timeout=2.0)

    class GateStartThread(real_thread):
        def start(self) -> None:
            super().start()
            if self.name == "system-volume-deferred-sync":
                start_entered.set()
                assert allow_start_return.wait(timeout=2.0)

    ducker._deferred_restart_needed = GateEvent()
    monkeypatch.setattr(system_volume.threading, "Thread", GateStartThread)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_JOIN_TIMEOUT", 0.02)
    monkeypatch.setattr(system_volume, "_LIFECYCLE_LOCK_TIMEOUT", 0.02)

    def _start_with_close_lock() -> None:
        with ducker._close_lock:
            ducker._start_deferred_sync_if_needed()

    start_thread = real_thread(
        target=_start_with_close_lock,
        name="deferred-starter",
    )
    begin_errors: list[BaseException] = []

    def _begin() -> None:
        try:
            ducker.begin(max_volume=0.05)
        except BaseException as exc:
            begin_errors.append(exc)

    begin_thread = real_thread(target=_begin, name="racing-begin")
    replacement: threading.Thread | None = None
    start_thread.start()
    assert start_entered.wait(timeout=1.0)
    original_worker = ducker._deferred_sync_thread
    assert original_worker is not None
    begin_thread.start()
    try:
        assert restart_set_entered.wait(timeout=1.0)
        allow_start_return.set()
        start_thread.join(timeout=1.0)
        assert not start_thread.is_alive()
        allow_restart_set.set()
        begin_thread.join(timeout=1.0)
        assert not begin_thread.is_alive()
        assert len(begin_errors) == 1
        assert isinstance(begin_errors[0], system_volume.SystemVolumeBusyError)

        deadline = time.monotonic() + 1.0
        replacement = ducker._deferred_sync_thread
        while (
            (replacement is None or replacement is original_worker)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
            replacement = ducker._deferred_sync_thread
        assert replacement is not None
        assert replacement is not original_worker
        assert replacement.is_alive()
    finally:
        allow_start_return.set()
        allow_restart_set.set()
        start_thread.join(timeout=1.0)
        begin_thread.join(timeout=1.0)
        ducker.stop_background_workers()

    original_worker.join(timeout=1.0)
    assert not original_worker.is_alive()
    assert not replacement.is_alive()


def test_terminal_stop_does_not_block_on_worker_holding_hal_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ducker = SystemOutputDucker(
        FakeVolumeBackend(),
        recovery_path=tmp_path / "system-volume.json",
    )
    lock_held = threading.Event()
    release_lock = threading.Event()
    stop = threading.Event()

    def _blocked_hal_call() -> None:
        with ducker._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2.0)

    worker = threading.Thread(
        target=_blocked_hal_call,
        name="system-volume-deferred-sync",
        daemon=True,
    )
    ducker._deferred_sync_thread = worker
    ducker._deferred_sync_stop = stop
    ducker._deferred_sync_pair = (worker, stop)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_JOIN_TIMEOUT", 0.02)
    monkeypatch.setattr(system_volume, "_LIFECYCLE_LOCK_TIMEOUT", 0.02)
    worker.start()
    assert lock_held.wait(timeout=1.0)

    started_at = time.monotonic()
    try:
        ducker.stop_background_workers()
        assert time.monotonic() - started_at < 0.3
        assert ducker._background_workers_stopped.is_set()
        assert stop.is_set()
    finally:
        release_lock.set()
        worker.join(timeout=1.0)

    assert not worker.is_alive()


def test_exhausted_foreground_restore_hands_off_to_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class RecoveringBackend(FakeVolumeBackend):
        fail_restore = True

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            if self.fail_restore and value > self.values[(device_id, element)]:
                raise RuntimeError("injected persistent restore failure")
            super().set_volume(device_id, element, value)

    backend = RecoveringBackend()
    backend.mutes[1] = None
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_INTERVAL", 0.01)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_FAST_DURATION", 0.0)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_SLOW_INTERVAL", 0.01)
    token = ducker.begin(max_volume=0.05)
    assert token is not None

    assert not ducker.end(token)
    assert not ducker.end(token)
    assert not ducker.end(token)
    assert ducker._active_token is token
    assert ducker.defer_failed_restore(token)
    assert ducker._active_token is None
    assert journal_path.exists()

    backend.fail_restore = False
    worker = ducker._deferred_sync_thread
    assert worker is not None
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()
    next_token = ducker.begin(max_volume=0.05)
    assert next_token is not None
    assert ducker.end(next_token)


def test_failed_deferred_worker_start_keeps_live_restore_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    token = ducker.begin(max_volume=0.0)
    assert token is not None
    real_start = threading.Thread.start

    def _fail_deferred_start(thread: threading.Thread) -> None:
        if thread.name == "system-volume-deferred-sync":
            raise RuntimeError("injected deferred worker start failure")
        real_start(thread)

    with monkeypatch.context() as scoped:
        scoped.setattr(threading.Thread, "start", _fail_deferred_start)
        assert not ducker.defer_failed_restore(token)

    assert ducker._active_token is token
    assert ducker._snapshots
    assert not ducker._deferred_snapshots
    assert ducker._deferred_sync_thread is None
    assert backend.mutes[1] is True
    assert journal_path.exists()

    assert ducker.end(token)
    assert backend.mutes[1] is False
    assert not journal_path.exists()


def test_hard_handoff_wal_keeps_partial_scalar_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    backend.mutes[1] = None
    backend.values[(1, 0)] = 0.3
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.2, 0.3)},
        phase=system_volume._PHASE_RESTORING,
        transition_started_at=time.time(),
    )
    crashed = SystemOutputDucker(backend, recovery_path=journal_path)
    crashed._deferred_snapshots[crashed._snapshot_key(snapshot)] = snapshot
    crashed._write_recovery_journal()
    real_set_volume = backend.set_volume

    def _crash_before_first_redduck_write(
        device_id: int,
        element: int,
        value: float,
    ) -> None:
        if value < backend.values[(device_id, element)]:
            raise SystemExit("injected hard exit")
        real_set_volume(device_id, element, value)

    monkeypatch.setattr(backend, "set_volume", _crash_before_first_redduck_write)
    with pytest.raises(SystemExit, match="injected hard exit"):
        crashed.begin(max_volume=0.05)
    assert backend.values[(1, 0)] == pytest.approx(0.3)
    crashed._release_lease()

    durable = json.loads(journal_path.read_text(encoding="utf-8"))
    item = durable["devices"][0]
    assert item["original"] == {"0": pytest.approx(0.8)}
    assert any(
        value == pytest.approx(0.3)
        for value in item["owned"]["0"]
    )
    item["transition_started_at"] = (
        time.time() - system_volume._RECOVERY_RAMP_GRACE - 1.0
    )
    journal_path.write_text(json.dumps(durable), encoding="utf-8")

    monkeypatch.setattr(backend, "set_volume", real_set_volume)
    recovery = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        recovery_path=journal_path,
    )
    assert recovery.recover_stale()
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


@pytest.mark.parametrize(
    (
        "old_max_volume",
        "new_max_volume",
        "old_muted",
        "expected_muted",
        "expected_volume",
    ),
    [
        (0.05, 0.0, False, True, 0.05),
        (0.0, 0.01, True, False, 0.01),
    ],
)
def test_hard_snapshot_at_old_target_applies_new_duck_settings(
    tmp_path: Path,
    old_max_volume: float,
    new_max_volume: float,
    old_muted: bool,
    expected_muted: bool,
    expected_volume: float,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    backend.mutes[1] = old_muted
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
        original_mute=False if old_max_volume == 0.0 else None,
        mute_target=True if old_max_volume == 0.0 else None,
        mute_owned=old_max_volume == 0.0,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()

    token = ducker.begin(max_volume=new_max_volume)

    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(expected_volume)
    assert backend.mutes[1] is expected_muted
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.mutes[1] is False
    assert not journal_path.exists()


def test_begin_reclaims_hard_snapshot_after_coreaudio_resets_mute(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    backend.mutes[1] = False
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        phase=system_volume._PHASE_DUCKED,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()
    backend.events.clear()

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    assert backend.events[0] == ("mute", 1, True)
    assert not any(
        event[0] == "volume" and event[3] > 0.05
        for event in backend.events
    )
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert backend.mutes[1] is True
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.mutes[1] is False
    assert not journal_path.exists()


def test_hard_rearm_uses_current_user_mute_as_new_original(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    backend = FakeVolumeBackend()
    backend.values[(1, 0)] = 0.05
    backend.mutes[1] = False
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
        original_mute=True,
        mute_target=True,
        mute_owned=False,
        phase=system_volume._PHASE_DUCKED,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()
    backend.events.clear()

    token = ducker.begin(max_volume=0.05)

    assert token is not None
    mute_events = [event for event in backend.events if event[0] == "mute"]
    assert mute_events == [("mute", 1, True), ("mute", 1, False)]
    assert backend.mutes[1] is False
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert backend.mutes[1] is False
    assert not journal_path.exists()


@pytest.mark.parametrize(
    ("original_mute", "mute_owned"),
    [(False, True), (True, False)],
)
def test_hard_rearm_rebases_scalar_changed_while_muted(
    tmp_path: Path,
    original_mute: bool,
    mute_owned: bool,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class MuteTrackingBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.muted_during_volume_writes: list[bool | None] = []

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            self.muted_during_volume_writes.append(self.mutes[device_id])
            super().set_volume(device_id, element, value)

    backend = MuteTrackingBackend()
    backend.values[(1, 0)] = 0.2
    backend.mutes[1] = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
        original_mute=original_mute,
        mute_target=True,
        mute_owned=mute_owned,
        phase=system_volume._PHASE_DUCKED,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    assert backend.values[(1, 0)] == pytest.approx(0.05)
    assert backend.muted_during_volume_writes
    assert all(backend.muted_during_volume_writes)
    assert ducker.end(token)
    assert backend.values[(1, 0)] == pytest.approx(0.2)
    assert backend.mutes[1] is original_mute
    assert not journal_path.exists()


def test_hard_rearm_enables_airpods_remote_gain_refresh(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "system-volume.json"
    virtual = system_volume._VIRTUAL_MAIN_ELEMENT
    backend = BluetoothGainBackend()
    backend.elements[1] = (virtual,)
    backend.values[(1, virtual)] = 0.05
    backend.effective_gain = 0.05
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(virtual,),
        original={virtual: 0.8},
        duck_target={virtual: 0.05},
        owned_values={virtual: (0.05,)},
        phase=system_volume._PHASE_DUCKED,
    )

    def _sleep(delay: float) -> None:
        if (
            backend.local_original_written
            and delay in system_volume._POST_RESTORE_SYNC_DELAYS
        ):
            backend.a2dp_active = True

    ducker = SystemOutputDucker(
        backend,
        sleeper=_sleep,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    active_snapshot = ducker._snapshots[("uid-1", (virtual,))]
    assert active_snapshot.post_restore_sync
    assert active_snapshot.post_restore_profile == (1, 2)

    backend.a2dp_active = False
    backend.restore_started = True
    assert ducker.end(token)
    assert backend.a2dp_active
    assert backend.original_writes_while_a2dp_active > 0
    assert backend.effective_gain == pytest.approx(0.8)
    assert backend.mutes[1] is False
    assert not journal_path.exists()


def test_hard_rearm_retains_mute_owner_across_transient_unknown_read(
    tmp_path: Path,
) -> None:
    class TransientUnknownMuteBackend(FakeVolumeBackend):
        mute_reads = 0

        def get_mute(self, device_id: int) -> bool | None:
            self.mute_reads += 1
            if self.mute_reads == 2:
                return None
            return super().get_mute(device_id)

    journal_path = tmp_path / "system-volume.json"
    backend = TransientUnknownMuteBackend()
    backend.values[(1, 0)] = 0.05
    backend.mutes[1] = True
    snapshot = system_volume._DeviceSnapshot(
        device_uid="uid-1",
        device_id_hint=1,
        profile_elements=(0,),
        original={0: 0.8},
        duck_target={0: 0.05},
        owned_values={0: (0.05,)},
        original_mute=False,
        mute_target=True,
        mute_owned=True,
        phase=system_volume._PHASE_DUCKED,
    )
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    ducker._deferred_snapshots[ducker._snapshot_key(snapshot)] = snapshot
    ducker._write_recovery_journal()

    token = ducker.begin(max_volume=0.0)

    assert token is not None
    active = ducker._snapshots[("uid-1", (0,))]
    assert active.original_mute is False
    assert active.mute_target is True
    assert active.mute_owned
    assert ducker.end(token)
    assert backend.mutes[1] is False
    assert not journal_path.exists()


def test_user_unmute_during_deferred_restore_is_not_reclaimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class UserUnmuteBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.unavailable = False
            self.user_unmuted = threading.Event()

        def device_id_for_uid(self, device_uid: str) -> int | None:
            if self.unavailable:
                return None
            return super().device_id_for_uid(device_uid)

        def set_volume(
            self,
            device_id: int,
            element: int,
            value: float,
        ) -> None:
            current = self.values[(device_id, element)]
            super().set_volume(device_id, element, value)
            if (
                threading.current_thread().name
                == "system-volume-deferred-sync"
                and value > current
                and not self.user_unmuted.is_set()
            ):
                self.mutes[device_id] = False
                self.user_unmuted.set()

    backend = UserUnmuteBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_INTERVAL", 0.01)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_FAST_DURATION", 0.0)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_SLOW_INTERVAL", 0.01)

    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.unavailable = True
    assert ducker.end(token)
    worker = ducker._deferred_sync_thread
    assert worker is not None

    backend.unavailable = False
    worker.join(timeout=1.0)

    assert backend.user_unmuted.is_set()
    assert not worker.is_alive()
    assert backend.mute_writes == [(1, True)]
    assert backend.mutes[1] is False
    assert backend.values[(1, 0)] == pytest.approx(0.8)
    assert not journal_path.exists()


def test_deferred_restore_preserves_user_original_mute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "system-volume.json"

    class DisconnectingBackend(FakeVolumeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.unavailable = False
            self.mutes[1] = True

        def device_id_for_uid(self, device_uid: str) -> int | None:
            if self.unavailable:
                return None
            return super().device_id_for_uid(device_uid)

    backend = DisconnectingBackend()
    ducker = SystemOutputDucker(
        backend,
        sleeper=lambda _delay: None,
        monitor_waiter=lambda stop, _timeout: stop.wait(),
        recovery_path=journal_path,
    )
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_INTERVAL", 0.01)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_FAST_DURATION", 0.0)
    monkeypatch.setattr(system_volume, "_DEFERRED_SYNC_SLOW_INTERVAL", 0.01)

    token = ducker.begin(max_volume=0.0)
    assert token is not None
    backend.unavailable = True
    assert ducker.end(token)
    worker = ducker._deferred_sync_thread
    assert worker is not None

    backend.unavailable = False
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert backend.mute_writes == []
    assert backend.mutes[1] is True
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
