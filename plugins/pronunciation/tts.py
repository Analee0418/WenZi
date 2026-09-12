"""Microsoft Edge TTS using the system speak-selection helper."""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile

FEMALE_VOICE = "en-US-AvaMultilingualNeural"
MALE_VOICE = "en-US-AndrewMultilingualNeural"
DEFAULT_RATE = "-5%"
STREAMER = os.path.expanduser("~/.local/bin/edge-tts-stream.py")
SYNTHESIS_TIMEOUT = 20


async def _generate_bytes(text_path: str, voice: str, rate: str) -> bytes:
    """Generate MP3 bytes through the helper used by the macOS Quick Action."""
    if not os.path.isfile(STREAMER) or not os.access(STREAMER, os.X_OK):
        raise RuntimeError(
            "Microsoft TTS helper is unavailable: ~/.local/bin/edge-tts-stream.py"
        )

    process = await asyncio.create_subprocess_exec(
        STREAMER,
        text_path,
        voice,
        rate,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        audio, stderr = await asyncio.wait_for(
            process.communicate(), timeout=SYNTHESIS_TIMEOUT
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError("Microsoft TTS timed out") from None
    if process.returncode != 0 or not audio:
        detail = stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(detail or "Microsoft TTS returned no audio")
    return audio


async def generate_voice(
    text: str,
    voice: str,
    rate: str = DEFAULT_RATE,
) -> bytes:
    """Generate one MP3 utterance with a specific Microsoft voice."""
    content = text.strip()
    if not content:
        raise ValueError("Text is empty")

    fd, text_path = tempfile.mkstemp(prefix="wenzi-pronunciation-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
        return await _generate_bytes(text_path, voice, rate)
    finally:
        try:
            os.unlink(text_path)
        except FileNotFoundError:
            pass


async def generate_tts(sentence: str) -> dict[str, str]:
    """Generate male and female Microsoft TTS audio as MP3 data URLs."""
    text = sentence.strip()
    if not text:
        raise ValueError("Text is empty")

    female_bytes, male_bytes = await asyncio.gather(
        generate_voice(text, FEMALE_VOICE),
        generate_voice(text, MALE_VOICE),
    )

    def _to_data_url(data: bytes) -> str:
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:audio/mpeg;base64,{encoded}"

    return {
        "female": _to_data_url(female_bytes),
        "male": _to_data_url(male_bytes),
    }
