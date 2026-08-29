"""Tests for Recorder audio chunk callback (streaming support)."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock

from wenzi.audio.recorder import Recorder, _TapSession


def _session_recorder(max_session_bytes: int | None = None) -> Recorder:
    """A recorder with a committed fake session (no real engine)."""
    kwargs = {"sample_rate": 16000, "block_ms": 20}
    if max_session_bytes is not None:
        kwargs["max_session_bytes"] = max_session_bytes
    r = Recorder(**kwargs)
    r._session = _TapSession()
    r._queue = r._session.queue
    r._recording = True
    r._active_gen = 1
    return r


def _voice_buffer(frames: int = 3):
    buf = MagicMock()
    buf.frameLength.return_value = frames
    ch = MagicMock()
    ch.as_buffer.return_value = struct.pack(f"<{frames}f", *([0.5] * frames))
    buf.floatChannelData.return_value = [ch]
    return buf


class TestAudioChunkCallback:
    def test_default_no_callback(self):
        r = Recorder()
        assert r._session is None
        # Setting/clearing without a session must be a safe no-op
        r.set_on_audio_chunk(lambda data: None)
        r.clear_on_audio_chunk()

    def test_set_and_clear_callback(self):
        r = _session_recorder()
        cb = MagicMock()
        r.set_on_audio_chunk(cb)
        assert r._session.on_chunk is cb
        r.clear_on_audio_chunk()
        assert r._session.on_chunk is None

    def test_callback_receives_audio_chunks(self):
        r = _session_recorder()
        chunks: list[bytes] = []
        r.set_on_audio_chunk(lambda data: chunks.append(data))

        r._tap_callback(_voice_buffer(), 1, 3.0, r._session)

        assert len(chunks) == 1
        assert chunks[0]  # resampled int16 bytes
        assert not r._session.queue.empty()

    def test_callback_error_does_not_break_queue(self):
        r = _session_recorder()

        def bad_cb(data):
            raise RuntimeError("callback error")

        r.set_on_audio_chunk(bad_cb)
        r._tap_callback(_voice_buffer(), 1, 3.0, r._session)

        # Audio must still be queued despite the callback error
        assert not r._session.queue.empty()

    def test_callback_not_invoked_when_max_size_reached(self):
        r = _session_recorder(max_session_bytes=2)
        cb = MagicMock()
        r.set_on_audio_chunk(cb)

        r._tap_callback(_voice_buffer(), 1, 3.0, r._session)  # 2 bytes: fits
        r._tap_callback(_voice_buffer(), 1, 3.0, r._session)  # over cap: drop

        assert cb.call_count == 1
        assert r._session.queue.qsize() == 1
