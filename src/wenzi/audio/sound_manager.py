"""Sound feedback manager for WenZi recording events."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from wenzi.config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)

# Bundled default sound shipped with the package
BUNDLED_SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")
DEFAULT_START_SOUND = "start_default.wav"

# User-custom sound in ~/.config/WenZi/sounds/
USER_SOUNDS_DIR = os.path.join(DEFAULT_CONFIG_DIR, "sounds")
CUSTOM_START_SOUND = "start_custom.wav"

# AVCaptureDevice.transportType uses the CoreAudio transport fourcc values.
_BLUETOOTH_TRANSPORTS = {
    int.from_bytes(b"blue", "big"),
    int.from_bytes(b"blea", "big"),
}


def _input_device_uses_bluetooth(
    configured_uid: str | None,
) -> bool | None:
    """Return Bluetooth state for the current default input.

    ``None`` means the route could not be classified. The legacy UID argument
    is ignored because WenZi never overrides the macOS default input.
    """
    del configured_uid
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

        device = AVCaptureDevice.defaultDeviceWithMediaType_(AVMediaTypeAudio)
        if device is None:
            return None
        return int(device.transportType()) in _BLUETOOTH_TRANSPORTS
    except Exception:
        logger.debug("Failed to inspect input transport", exc_info=True)
        return None


def _resolve_start_sound(config_dir: str | None = None) -> str:
    """Return the path to the start sound file.

    Priority:
    1. User-custom sound: ~/.config/WenZi/sounds/start_custom.wav
    2. Bundled default:   <package>/sounds/start_default.wav
    """
    if config_dir:
        user_dir = os.path.join(config_dir, "sounds")
    else:
        user_dir = USER_SOUNDS_DIR
    user_dir = os.path.expanduser(user_dir)

    custom_path = os.path.join(user_dir, CUSTOM_START_SOUND)
    if os.path.exists(custom_path):
        logger.debug("Using custom start sound: %s", custom_path)
        return custom_path

    bundled_path = os.path.join(BUNDLED_SOUNDS_DIR, DEFAULT_START_SOUND)
    logger.debug("Using bundled start sound: %s", bundled_path)
    return bundled_path


class SoundManager:
    """Play sound feedback for recording events."""

    def __init__(
        self,
        enabled: bool = True,
        volume: float = 0.1,
        config_dir: str | None = None,
        input_route_provider: Callable[[], object | None] | None = None,
        input_route_chime_notifier: Callable[[], None] | None = None,
    ) -> None:
        self._enabled = enabled
        self._volume = volume
        self._start_sound_path = _resolve_start_sound(config_dir)
        self._cached_sound: object = None  # Cached NSSound instance
        self._input_route_provider = input_route_provider
        self._input_route_chime_notifier = input_route_chime_notifier

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def should_play_start(self, input_device_uid: str | None) -> bool:
        """Return whether audible start feedback is safe for this input.

        ``NSSound`` opens the playback route.  Starting a Bluetooth microphone
        while the 300 ms chime still owns A2DP can race the A2DP-to-HFP switch
        and leave the capture stream delivering silence.  The visual indicator
        remains the start feedback for Bluetooth sessions.
        """
        bluetooth: bool | None
        provider = self._input_route_provider
        if provider is not None:
            try:
                route = provider()
                transport_type = getattr(route, "transport_type", None)
                bluetooth = (
                    int(transport_type) in _BLUETOOTH_TRANSPORTS
                    if transport_type is not None
                    else None
                )
            except Exception:
                logger.debug(
                    "Failed to preflight the input route",
                    exc_info=True,
                )
                bluetooth = None
        else:
            if not self._enabled:
                return False
            bluetooth = _input_device_uses_bluetooth(input_device_uid)
        if not self._enabled:
            # Calling the provider above refreshes Recorder's attempt-scoped
            # preflight even when feedback was disabled after a quick tap.
            return False
        if bluetooth is True:
            logger.info("Skipping start sound for Bluetooth input")
            return False
        if bluetooth is None:
            logger.info("Skipping start sound because input route is unknown")
            return False
        notifier = self._input_route_chime_notifier
        if notifier is not None:
            try:
                notifier()
            except Exception:
                logger.debug(
                    "Failed to mark the preflight route for start sound",
                    exc_info=True,
                )
                return False
        return True

    def warmup(self) -> None:
        """Pre-load the NSSound object on the main thread.

        Call via AppHelper.callAfter() after the event loop starts to
        eliminate first-play latency.
        """
        if self._cached_sound is not None:
            return
        try:
            from AppKit import NSSound

            if not os.path.exists(self._start_sound_path):
                return
            sound = NSSound.alloc().initWithContentsOfFile_byReference_(
                self._start_sound_path, True
            )
            if sound is not None:
                sound.setVolume_(self._volume)
                self._cached_sound = sound
                logger.debug("NSSound pre-loaded: %s", self._start_sound_path)
        except Exception as e:
            logger.debug("NSSound warmup failed: %s", e)

    def play(self, event: str) -> None:
        """Play the sound for the given event. Only 'start' is supported."""
        if not self._enabled:
            return

        if event != "start":
            return

        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(self._play_on_main_thread)
        except Exception as e:
            logger.warning("Failed to schedule sound playback: %s", e)

    def _play_on_main_thread(self) -> None:
        """Actually play the sound file. Must be called on the main thread."""
        try:
            if self._cached_sound is not None:
                # Stop any ongoing playback and replay from the beginning
                self._cached_sound.stop()
                self._cached_sound.play()
                return

            # Fallback: load on demand if warmup was not called
            from AppKit import NSSound

            if not os.path.exists(self._start_sound_path):
                logger.warning("Sound file not found: %s", self._start_sound_path)
                return

            sound = NSSound.alloc().initWithContentsOfFile_byReference_(
                self._start_sound_path, True
            )
            if sound is None:
                logger.warning("Failed to load sound: %s", self._start_sound_path)
                return
            sound.setVolume_(self._volume)
            sound.play()
            self._cached_sound = sound
        except Exception as e:
            logger.warning("Failed to play sound: %s", e)
