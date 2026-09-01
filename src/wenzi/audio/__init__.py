"""Audio subpackage — recording, sound feedback, and recording indicator."""

from .recorder import Recorder, default_input_device_name, list_input_devices
from .recording_indicator import RecordingIndicatorPanel
from .sound_manager import SoundManager
from .system_volume import SystemOutputDucker

__all__ = [
    "Recorder",
    "RecordingIndicatorPanel",
    "SoundManager",
    "SystemOutputDucker",
    "default_input_device_name",
    "list_input_devices",
]
