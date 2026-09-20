"""Multi-speaker dialogue parsing, synthesis, and reader panel."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("pronunciation")

_SPEAKER_RE = re.compile(r"^\s*([^:：\n]{1,40})\s*[:：]\s*(.+?)\s*$")
_NUMBERED_LINE_RE = re.compile(r"^\s*(\d{1,3})\s*[.)]\s*(.+?)\s*$")
_SENTENCE_END_RE = re.compile(r"[.!?。！？][\"'”’)]*$")
_MAX_SEGMENTS = 100
_MAX_CHARACTERS = 20_000
_IMAGE_EXTENSIONS = {".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_TEXT_EXTENSIONS = {".md", ".text", ".txt"}
_dialogue_panel_ref = [None]


class _DialogueSuperseded(Exception):
    """Stop queued work after a newer dialogue request takes ownership."""


class _DialogueAudioOwner:
    __slots__ = ("audio", "request_id")

    def __init__(self, request_id) -> None:
        self.request_id = request_id
        self.audio: bytes | None = None


class _DialogueAudioState:
    """Linearize ownership changes and audio commits across UI and async threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: _DialogueAudioOwner | None = None

    def begin(self, request_id) -> _DialogueAudioOwner:
        owner = _DialogueAudioOwner(request_id)
        with self._lock:
            if self._owner is not None:
                self._owner.audio = None
            self._owner = owner
        return owner

    def is_current(self, owner: _DialogueAudioOwner) -> bool:
        with self._lock:
            return self._owner is owner

    def commit(
        self,
        owner: _DialogueAudioOwner,
        audio: bytes,
        *,
        panel_is_current: Callable[[], bool],
    ) -> bool:
        with self._lock:
            if self._owner is not owner or not panel_is_current():
                return False
            owner.audio = audio
            return True

    def retire(self, owner: _DialogueAudioOwner | None = None) -> bool:
        with self._lock:
            if owner is not None and self._owner is not owner:
                return False
            if self._owner is None:
                return False
            self._owner.audio = None
            self._owner = None
            return True

    def retire_request(self, request_id) -> bool:
        with self._lock:
            if self._owner is None or self._owner.request_id != request_id:
                return False
            self._owner.audio = None
            self._owner = None
            return True

    def audio_for(self, request_id) -> bytes | None:
        with self._lock:
            if self._owner is None or self._owner.request_id != request_id:
                return None
            return self._owner.audio


def _normalize_numbered_dialogue(text: str) -> str:
    """Remove worksheet numbering and OCR noise around a numbered dialogue.

    OCR returns one physical line at a time.  Printed continuation lines are
    joined to the preceding item, while short handwritten notes that appear
    after a complete sentence are ignored.  When OCR detects an out-of-order
    handwritten number, the later real sequence replaces that false entry.
    """
    entries: list[tuple[int, list[str]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _NUMBERED_LINE_RE.match(line)
        if match:
            number = int(match.group(1))
            while entries and entries[-1][0] >= number:
                entries.pop()
            entries.append((number, [match.group(2).strip()]))
            continue
        if entries and not _SENTENCE_END_RE.search(" ".join(entries[-1][1])):
            entries[-1][1].append(line)

    if len(entries) < 2:
        return text

    normalized = [" ".join(parts) for _, parts in entries]
    if len(normalized) < 2 or not any(_SPEAKER_RE.match(line) for line in normalized):
        return text
    return "\n".join(normalized)

# Alternate genders while keeping every role on a stable, distinct voice.
_VOICES = [
    ("en-US-AvaMultilingualNeural", "Female"),
    ("en-US-AndrewMultilingualNeural", "Male"),
    ("en-US-EmmaMultilingualNeural", "Female"),
    ("en-US-BrianMultilingualNeural", "Male"),
    ("en-US-JennyNeural", "Female"),
    ("en-US-GuyNeural", "Male"),
    ("en-US-AriaNeural", "Female"),
    ("en-US-ChristopherNeural", "Male"),
    ("en-US-MichelleNeural", "Female"),
    ("en-US-EricNeural", "Male"),
]


def parse_dialogue(text: str) -> list[dict[str, str]]:
    """Parse ``Speaker: utterance`` lines and attach continuation lines."""
    content = text.strip()
    if not content:
        raise ValueError("Dialogue text is empty")
    if len(content) > _MAX_CHARACTERS:
        raise ValueError(f"Dialogue is longer than {_MAX_CHARACTERS:,} characters")
    content = _normalize_numbered_dialogue(content)

    segments: list[dict[str, str]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _SPEAKER_RE.match(line)
        if match:
            segments.append(
                {"speaker": match.group(1).strip(), "text": match.group(2).strip()}
            )
        elif segments:
            segments[-1]["text"] += " " + line
        else:
            segments.append({"speaker": "Narrator", "text": line})

    if not segments:
        raise ValueError("No dialogue lines were found")
    if len(segments) > _MAX_SEGMENTS:
        raise ValueError(f"Dialogue contains more than {_MAX_SEGMENTS} segments")
    return segments


def assign_voices(segments: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Assign a stable, gender-alternating Microsoft voice to each role."""
    speakers = list(dict.fromkeys(segment["speaker"] for segment in segments))
    if len(speakers) > len(_VOICES):
        raise ValueError(f"Dialogue has more than {len(_VOICES)} distinct roles")

    role_map = {
        speaker: {"voice": _VOICES[index][0], "gender": _VOICES[index][1]}
        for index, speaker in enumerate(speakers)
    }
    roles = [
        {
            "speaker": speaker,
            "voice": role_map[speaker]["voice"],
            "voice_name": role_map[speaker]["voice"].split("-")[2].removesuffix("Neural"),
            "gender": role_map[speaker]["gender"],
        }
        for speaker in speakers
    ]
    voiced_segments = [segment | role_map[segment["speaker"]] for segment in segments]
    return voiced_segments, roles


def _ffmpeg_path() -> str:
    candidates = (
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    )
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise RuntimeError("ffmpeg is required to combine dialogue audio")


async def generate_dialogue_audio_bytes(
    segments: list[dict],
    *,
    should_continue: Callable[[], bool] | None = None,
) -> bytes:
    """Synthesize each segment and return the combined MP3 bytes."""
    from .tts import generate_voice

    semaphore = asyncio.Semaphore(3)
    aborted = False

    def _is_current() -> bool:
        return should_continue is None or should_continue()

    async def _generate(segment: dict) -> bytes | None:
        nonlocal aborted
        async with semaphore:
            if aborted or not _is_current():
                return None
            try:
                return await generate_voice(segment["text"], segment["voice"])
            except Exception:
                aborted = True
                raise

    audio_parts = await asyncio.gather(*(_generate(segment) for segment in segments))
    if not _is_current() or any(audio is None for audio in audio_parts):
        raise _DialogueSuperseded

    completed_parts = [audio for audio in audio_parts if audio is not None]
    with tempfile.TemporaryDirectory(prefix="wenzi-dialogue-") as temp_dir:
        temp_path = Path(temp_dir)
        concat_lines = []
        for index, audio in enumerate(completed_parts):
            part_path = temp_path / f"{index:03d}.mp3"
            part_path.write_bytes(audio)
            concat_lines.append(f"file '{part_path}'")

        if not _is_current():
            raise _DialogueSuperseded

        list_path = temp_path / "inputs.txt"
        list_path.write_text("\n".join(concat_lines), encoding="utf-8")
        output_path = temp_path / "dialogue.mp3"
        process = await asyncio.create_subprocess_exec(
            _ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-y",
            str(output_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0 or not output_path.exists():
            detail = stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(detail or "Failed to combine dialogue audio")
        return output_path.read_bytes()


def _audio_data_url(audio: bytes) -> str:
    """Encode MP3 bytes for playback in the WebView bridge."""
    encoded = base64.b64encode(audio).decode("ascii")
    return f"data:audio/mpeg;base64,{encoded}"


async def generate_dialogue_audio(segments: list[dict]) -> str:
    """Synthesize the dialogue and return an MP3 data URL."""
    return _audio_data_url(await generate_dialogue_audio_bytes(segments))


def _read_source(path: str) -> str:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in _TEXT_EXTENSIONS:
        return source.read_text(encoding="utf-8-sig")
    if suffix in _IMAGE_EXTENSIONS:
        from wenzi.scripting.ocr import recognize_text

        text = recognize_text(str(source), languages=["en-US", "zh-Hans", "zh-Hant"])
        if not text.strip():
            raise RuntimeError("No text was recognized in the image")
        return text
    raise ValueError("Choose a TXT, Markdown, PNG, JPEG, HEIC, TIFF, or WebP file")


_DIALOGUE_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
:root { color-scheme:light dark; --bg:#f4f5f7; --surface:#fff;
        --surface-hover:#f7f8fa; --text:#202124; --muted:#6b7078;
        --border:#d8dce2; --accent:#0969da; --primary:#075fbd;
        --primary-hover:#0969da; --success:#18794e; --danger:#c9372c;
        --focus-ring:#0969da;
        --role-0:#b42368; --role-1:#0875a1; --role-2:#9a6700;
        --role-3:#6f42c1; --role-4:#1f7a4d; --role-5:#c2410c;
        --role-6:#2f5fb3; --role-7:#a23595; --role-8:#466b21;
        --role-9:#9f3a38; }
@media(prefers-color-scheme:dark) { :root { --bg:#18191c; --surface:#24262a;
  --surface-hover:#2b2e33; --text:#f3f4f6; --muted:#a8adb5;
  --border:#3a3e45; --accent:#6aa9ff; --primary:#1769c2;
  --primary-hover:#0d73d5; --success:#56d492; --danger:#ff8a80;
  --focus-ring:#6aa9ff;
  --role-0:#f472b6; --role-1:#38bdf8; --role-2:#fbbf24;
  --role-3:#c4b5fd; --role-4:#4ade80; --role-5:#fb923c;
  --role-6:#93c5fd; --role-7:#e879f9; --role-8:#a3e635;
  --role-9:#fca5a5; } }
* { box-sizing:border-box; }
html, body { min-height:100%; }
body { margin:0; padding:20px; background:var(--bg); color:var(--text);
       font-family:-apple-system,BlinkMacSystemFont,sans-serif; font-size:14px; }
.reader { width:100%; max-width:900px; margin:0 auto; }
.section-head { display:flex; align-items:center; justify-content:space-between;
                gap:12px; margin-bottom:10px; }
.heading-group { min-width:0; }
h2 { margin:0; font-size:15px; line-height:20px; font-weight:650; }
.source-actions, .result-actions { display:flex; align-items:center; gap:7px; }
button { height:34px; border:1px solid var(--border); border-radius:6px;
         background:var(--surface); color:var(--text); padding:0 12px; font-size:13px;
         font-weight:500; cursor:pointer; display:inline-flex; align-items:center;
         justify-content:center; gap:7px; transition:background-color 120ms ease,
         border-color 120ms ease,color 120ms ease,opacity 120ms ease,
         transform 120ms ease; }
button svg { width:16px; height:16px; flex:0 0 auto; fill:none;
             stroke:currentColor; stroke-width:2; stroke-linecap:round;
             stroke-linejoin:round; }
button:not(:disabled):hover { border-color:var(--accent); color:var(--accent);
                              background:var(--surface-hover);
                              transform:translateY(-1px); }
button:not(:disabled):active { transform:scale(.98); }
button:focus-visible, textarea:focus-visible { outline:2px solid var(--focus-ring);
                                                outline-offset:2px; }
button.primary { min-width:136px; background:var(--primary); color:white;
                 border-color:var(--primary); font-weight:600; }
button.primary:not(:disabled):hover { background:var(--primary-hover); color:white;
                                     border-color:var(--primary-hover); }
button.icon-button { width:34px; padding:0; }
button:disabled { opacity:.45; cursor:default; }
textarea { display:block; width:100%; height:176px; resize:vertical; border-radius:6px;
           border:1px solid var(--border); background:var(--surface); color:var(--text);
           padding:12px; font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }
textarea:focus-visible { border-color:var(--accent); }
.source-footer { min-height:46px; display:grid;
                 grid-template-columns:minmax(0,1fr) auto; gap:12px;
                 align-items:center; padding-top:8px; }
.status { min-height:34px; display:flex; align-items:center; gap:8px;
          color:var(--muted); font-size:13px; min-width:0; }
.status-text { min-width:0; overflow-wrap:anywhere; }
.status-mark { width:8px; height:8px; border-radius:50%; flex:0 0 auto;
               display:none; }
.status[data-state="loading"] .status-mark { width:12px; height:12px; display:block;
  border:2px solid var(--border); border-top-color:var(--accent);
  background:transparent; animation:spin 700ms linear infinite; }
.status[data-state="ready"] { color:var(--success); }
.status[data-state="ready"] .status-mark { display:block; background:var(--success);
  animation:status-pop 180ms ease-out; }
.status[data-state="error"] { color:var(--danger); }
.status[data-state="error"] .status-mark { display:block; background:var(--danger); }
.read-spinner { display:none; width:13px; height:13px; border-radius:50%;
                border:2px solid rgba(255,255,255,.45); border-top-color:white; }
#read[aria-busy="true"] .read-spinner { display:block; animation:spin 700ms linear infinite; }
#read[aria-busy="true"] .read-icon { display:none; }
.results { margin-top:12px; padding-top:17px; border-top:1px solid var(--border); }
.summary { display:block; min-height:17px; margin-top:1px; color:var(--muted);
           font-size:12px; }
.roles { display:flex; flex-wrap:wrap; gap:7px; margin:2px 0 14px; }
.role { max-width:100%; border:1px solid var(--border); border-radius:6px;
        padding:5px 9px; background:var(--surface); font-size:12px;
        display:inline-flex; align-items:center; flex-wrap:wrap; gap:7px; }
.role-dot { width:8px; height:8px; border-radius:50%; flex:0 0 auto;
            background:var(--role-color); }
.role strong, .role span:last-child { min-width:0; overflow-wrap:anywhere; }
.role span:last-child { color:var(--muted); }
.script { border-top:1px solid var(--border); }
.line { display:grid; grid-template-columns:minmax(96px,128px) minmax(0,1fr);
        gap:16px; padding:11px 10px 11px 14px; border-bottom:1px solid var(--border);
        border-left:3px solid var(--role-color,transparent); line-height:1.5;
        transition:background-color 120ms ease; }
.line:hover { background:var(--surface-hover); }
.speaker { color:var(--role-color); font-size:13px; font-weight:650;
           overflow-wrap:anywhere; }
.dialogue-text { min-width:0; overflow-wrap:anywhere; }
.rhythm-toolbar { display:flex; align-items:center; flex-wrap:wrap; gap:10px;
                  margin-top:10px; }
.rhythm-status { color:var(--muted); font-size:12px; }
#rhythm-summary { margin:7px 0 12px; }
.role-0 { --role-color:var(--role-0); } .role-1 { --role-color:var(--role-1); }
.role-2 { --role-color:var(--role-2); } .role-3 { --role-color:var(--role-3); }
.role-4 { --role-color:var(--role-4); } .role-5 { --role-color:var(--role-5); }
.role-6 { --role-color:var(--role-6); } .role-7 { --role-color:var(--role-7); }
.role-8 { --role-color:var(--role-8); } .role-9 { --role-color:var(--role-9); }
.empty { color:var(--muted); text-align:center; padding:34px 0; }
.results-enter { animation:results-enter 180ms ease-out; }
@keyframes spin { to { transform:rotate(360deg); } }
@keyframes status-pop { 50% { transform:scale(1.35); } }
@keyframes results-enter { from { opacity:0; transform:translateY(4px); }
                           to { opacity:1; transform:translateY(0); } }
@media(max-width:560px) {
  body { padding:14px; }
  .section-head { align-items:flex-start; flex-wrap:wrap; }
  .line { grid-template-columns:1fr; gap:4px; }
  .source-actions { width:100%; }
  .source-actions button { flex:1; min-width:0; }
}
@media(max-width:480px) {
  .source-footer { grid-template-columns:1fr; }
  #read { width:100%; }
}
@media(prefers-reduced-motion:reduce) {
  *, *::before, *::after { animation-duration:.01ms !important;
    animation-iteration-count:1 !important; transition-duration:.01ms !important;
    scroll-behavior:auto !important; }
  .status[data-state="loading"] .status-mark,
  #read[aria-busy="true"] .read-spinner { animation:none; border-color:currentColor; }
}
</style>
</head>
<body>
<main class="reader">
  <section class="source-section" aria-labelledby="source-title">
    <header class="section-head">
      <h2 id="source-title">Source</h2>
      <div class="source-actions">
        <button onclick="loadClipboard()">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect width="8" height="4" x="8" y="2" rx="1"></rect>
            <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
          </svg>
          <span>Clipboard</span>
        </button>
        <button onclick="chooseFile()">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <path d="M12 18v-6"></path><path d="m9 15 3-3 3 3"></path>
          </svg>
          <span>Open File</span>
        </button>
      </div>
    </header>
    <textarea id="source" maxlength="20000" spellcheck="false"
              aria-labelledby="source-title"
              placeholder="A: ...&#10;B: ...">__INITIAL_TEXT__</textarea>
    <footer class="source-footer">
      <div class="status" id="status" data-state="idle" role="status"
           aria-live="polite" aria-atomic="true">
        <span class="status-mark" aria-hidden="true"></span>
        <span class="status-text" id="status-text"></span>
      </div>
      <button class="primary" id="read" onclick="readDialogue()" aria-busy="false">
        <span class="read-spinner" aria-hidden="true"></span>
        <svg class="read-icon" viewBox="0 0 24 24" aria-hidden="true"><polygon points="6 3 20 12 6 21 6 3"></polygon></svg>
        <span id="read-label">Read Dialogue</span>
      </button>
    </footer>
    <div class="rhythm-toolbar">
      <button id="mark-rhythm" onclick="markRhythm()" aria-busy="false">Mark stress &amp; pauses</button>
      <span class="rhythm-status" id="rhythm-status" role="status" aria-live="polite"></span>
    </div>
  </section>
  <section class="results" id="results" aria-labelledby="dialogue-title">
    <header class="section-head">
      <div class="heading-group">
        <h2 id="dialogue-title">Dialogue</h2>
        <span class="summary" id="summary"></span>
      </div>
      <div class="result-actions">
        <button class="icon-button" id="replay" onclick="replay()" disabled
                title="Replay audio" aria-label="Replay audio">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"></path><path d="M3 3v5h5"></path></svg>
        </button>
        <button class="icon-button" id="download" onclick="saveAudio()" disabled
                title="Save audio" aria-label="Save audio">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" x2="12" y1="15" y2="3"></line>
          </svg>
        </button>
      </div>
    </header>
    <div class="rhythm-legend" id="rhythm-legend" hidden>
      <span class="rhythm-stress">Bold color = stress</span> · | brief pause · || longer pause<br>
      Suggested phrasing, not audio timing. Stress does not require a pause.
    </div>
    <p class="rhythm-summary" id="rhythm-summary" hidden></p>
    <div class="roles" id="roles" role="list"></div>
    <div class="script" id="script" role="list"><div class="empty">No dialogue</div></div>
  </section>
</main>
<script>
var audioUrl = "";
var sourceRequestId = 0;
var dialogueRequestSequence = 0;
var activeDialogueRequestId = 0;
var sourceRollbackState = null;
var dialogueOwned = false;
var sourceErrorMessage = "";
var rhythmRequestSequence = 0;
var activeRhythmRequestId = null;
var rhythmResult = null;
var displayedSegments = [];
function setRhythmBusy(busy) {
  var button = document.getElementById("mark-rhythm");
  button.disabled = busy;
  button.setAttribute("aria-busy", busy ? "true" : "false");
  button.textContent = busy ? "Marking..." : "Mark stress & pauses";
}
function setRhythmStatus(message, error) {
  var status = document.getElementById("rhythm-status");
  status.textContent = message || "";
  status.classList.toggle("rhythm-error", !!error);
}
function renderRhythm() {
  var combinedText = displayedSegments.map(function(item) { return item.text; }).join("\\n\\n");
  var result = rhythmResult && rhythmResult.text === combinedText ? rhythmResult : null;
  var offset = 0;
  document.querySelectorAll("#script .dialogue-text").forEach(function(container, index) {
    var text = displayedSegments[index].text;
    if (result) PhRhythm.render(container, text, result, offset);
    else container.textContent = text;
    offset += (text.match(/\\S+/g) || []).length;
  });
  document.getElementById("rhythm-legend").hidden = !result;
  var summary = document.getElementById("rhythm-summary");
  summary.hidden = !result;
  summary.textContent = result ? result.summary : "";
}
function invalidateRhythm() {
  if (activeRhythmRequestId !== null) {
    wz.send("invalidate_rhythm", {request_id:activeRhythmRequestId});
  }
  activeRhythmRequestId = null;
  rhythmResult = null;
  setRhythmBusy(false);
  setRhythmStatus("");
  renderRhythm();
}
function markRhythm() {
  var requestId = ++rhythmRequestSequence;
  activeRhythmRequestId = requestId;
  rhythmResult = null;
  renderRhythm();
  setRhythmBusy(true);
  setRhythmStatus("Preparing reading guide...");
  wz.send("request_rhythm", {
    text:document.getElementById("source").value, request_id:requestId
  });
}
function setStatus(message, state) {
  var el = document.getElementById("status");
  document.getElementById("status-text").textContent = message || "";
  el.dataset.state = state || "idle";
}
function setReadBusy(isBusy) {
  var read = document.getElementById("read");
  read.setAttribute("aria-busy", isBusy ? "true" : "false");
  document.getElementById("read-label").textContent =
    isBusy ? "Generating..." : "Read Dialogue";
}
function countLabel(count, singular) {
  return count + " " + singular + (count === 1 ? "" : "s");
}
function resetAudio() {
  audioUrl = "";
  document.getElementById("replay").disabled = true;
  document.getElementById("download").disabled = true;
}
function captureDialogueState() {
  var status = document.getElementById("status");
  return {
    requestId: activeDialogueRequestId,
    audioUrl: audioUrl,
    readDisabled: document.getElementById("read").disabled,
    replayDisabled: document.getElementById("replay").disabled,
    downloadDisabled: document.getElementById("download").disabled,
    statusText: document.getElementById("status-text").textContent,
    statusState: status.dataset.state,
    readBusy: document.getElementById("read").getAttribute("aria-busy") === "true",
    dialogueOwned: dialogueOwned,
    sourceErrorMessage: sourceErrorMessage,
    deferredEvents: []
  };
}
function restoreDialogueState(state) {
  if (!state) return;
  activeDialogueRequestId = state.requestId;
  audioUrl = state.audioUrl;
  dialogueOwned = state.dialogueOwned;
  sourceErrorMessage = state.sourceErrorMessage;
  document.getElementById("read").disabled = state.readDisabled;
  document.getElementById("replay").disabled = state.replayDisabled;
  document.getElementById("download").disabled = state.downloadDisabled;
  setReadBusy(state.readBusy);
  setStatus(state.statusText, state.statusState);
  state.deferredEvents.forEach(function(event) {
    handleDialogueEvent(event.name, event.data);
  });
}
function rollbackSourceRequest(errorMessage) {
  var state = sourceRollbackState;
  sourceRollbackState = null;
  restoreDialogueState(state);
  if (errorMessage) {
    sourceErrorMessage = errorMessage;
    setStatus(errorMessage, "error");
  }
}
function beginSourceRequest(message) {
  if (!sourceRollbackState) sourceRollbackState = captureDialogueState();
  sourceErrorMessage = "";
  var requestId = ++sourceRequestId;
  activeDialogueRequestId = ++dialogueRequestSequence;
  resetAudio();
  document.getElementById("read").disabled = false;
  setReadBusy(false);
  setStatus(message, "loading");
  return requestId;
}
function loadClipboard() {
  var requestId = beginSourceRequest("Loading clipboard...");
  wz.send("load_clipboard", {request_id:requestId});
}
function chooseFile() {
  var requestId = beginSourceRequest("Choosing file...");
  wz.send("choose_file", {request_id:requestId});
}
function _startDialogue() {
  var text = document.getElementById("source").value;
  var requestId = ++dialogueRequestSequence;
  activeDialogueRequestId = requestId;
  dialogueOwned = true;
  sourceErrorMessage = "";
  resetAudio();
  document.getElementById("read").disabled = true;
  setReadBusy(true);
  setStatus("Preparing dialogue...", "busy");
  wz.send("read_dialogue", {text:text, request_id:requestId});
}
function readDialogue() {
  ++sourceRequestId;
  sourceRollbackState = null;
  _startDialogue();
}
function replay() { if (audioUrl) wz.playAudio(audioUrl); }
function saveAudio() {
  if (!audioUrl) return;
  document.getElementById("download").disabled = true;
  wz.send("save_audio", {request_id:activeDialogueRequestId});
}
function renderDialogue(data) {
  displayedSegments = data.segments;
  var roles = document.getElementById("roles");
  var roleIndexes = new Map();
  var roleFragment = document.createDocumentFragment();
  data.roles.forEach(function(role, index) {
    var roleIndex = index % 10;
    roleIndexes.set(role.speaker, roleIndex);
    var el = document.createElement("div");
    el.className = "role role-" + roleIndex;
    el.setAttribute("role", "listitem");
    var dot = document.createElement("span");
    dot.className = "role-dot";
    dot.setAttribute("aria-hidden", "true");
    var name = document.createElement("strong");
    name.textContent = role.speaker;
    var detail = document.createElement("span");
    detail.textContent = role.voice_name + " / " + role.gender;
    el.appendChild(dot);
    el.appendChild(name);
    el.appendChild(detail);
    roleFragment.appendChild(el);
  });
  roles.replaceChildren(roleFragment);
  var script = document.getElementById("script");
  var scriptFragment = document.createDocumentFragment();
  data.segments.forEach(function(item) {
    var roleIndex = roleIndexes.get(item.speaker) || 0;
    var row = document.createElement("div");
    row.className = "line role-" + roleIndex;
    row.setAttribute("role", "listitem");
    var speaker = document.createElement("div");
    speaker.className = "speaker";
    speaker.textContent = item.speaker;
    var text = document.createElement("div");
    text.className = "dialogue-text";
    text.textContent = item.text;
    row.appendChild(speaker);
    row.appendChild(text);
    scriptFragment.appendChild(row);
  });
  script.replaceChildren(scriptFragment);
  renderRhythm();
  document.getElementById("summary").textContent =
    countLabel(data.segments.length, "line") + " \\u00b7 " +
    countLabel(data.roles.length, "voice");
  var results = document.getElementById("results");
  results.classList.remove("results-enter");
  void results.offsetWidth;
  results.classList.add("results-enter");
}
wz.on("source_loaded", function(data) {
  if (data.request_id !== sourceRequestId) return;
  invalidateRhythm();
  sourceRollbackState = null;
  document.getElementById("source").value = data.text;
  _startDialogue();
});
wz.on("source_loading", function(data) {
  if (data.request_id !== sourceRequestId) return;
  setStatus(data.message || "Reading file...", "loading");
});
wz.on("source_cancelled", function(data) {
  if (data.request_id !== sourceRequestId) return;
  rollbackSourceRequest("");
});
wz.on("source_error", function(data) {
  if (data.request_id !== sourceRequestId) return;
  rollbackSourceRequest(data.message);
});
function handleDialogueEvent(name, data) {
  if (sourceRollbackState && data.request_id === sourceRollbackState.requestId) {
    sourceRollbackState.deferredEvents.push({name:name, data:data});
    return;
  }
  if (data.request_id !== activeDialogueRequestId) return;
  if (name === "dialogue_parsed") {
    renderDialogue(data);
    if (!sourceErrorMessage) {
      setStatus("Generating " + countLabel(data.segments.length, "line") + "...", "busy");
    }
  } else if (name === "dialogue_audio") {
    audioUrl = data.url;
    dialogueOwned = true;
    document.getElementById("read").disabled = false;
    setReadBusy(false);
    document.getElementById("replay").disabled = false;
    document.getElementById("download").disabled = false;
    if (!sourceErrorMessage) {
      setStatus(countLabel(data.segments, "line") + " ready", "ready");
    }
    wz.playAudio(audioUrl);
  } else if (name === "dialogue_error") {
    dialogueOwned = false;
    document.getElementById("read").disabled = false;
    setReadBusy(false);
    if (!sourceErrorMessage) setStatus(data.message, "error");
  }
}
wz.on("dialogue_parsed", function(data) { handleDialogueEvent("dialogue_parsed", data); });
wz.on("dialogue_audio", function(data) { handleDialogueEvent("dialogue_audio", data); });
wz.on("dialogue_error", function(data) { handleDialogueEvent("dialogue_error", data); });
wz.on("rhythm_parsed", function(data) {
  if (data.request_id !== activeRhythmRequestId) return;
  renderDialogue(data);
});
wz.on("rhythm_progress", function(data) {
  if (data.request_id !== activeRhythmRequestId) return;
  setRhythmStatus("Marking reading guide: " + data.done + " / " + data.total);
});
wz.on("rhythm_result", function(data) {
  if (data.request_id !== activeRhythmRequestId) return;
  rhythmResult = data.result;
  renderRhythm();
  setRhythmBusy(false);
  setRhythmStatus("Suggested everyday American reading");
});
wz.on("rhythm_error", function(data) {
  if (data.request_id !== activeRhythmRequestId) return;
  setRhythmBusy(false);
  document.getElementById("mark-rhythm").textContent = "Retry stress & pauses";
  setRhythmStatus(data.message, true);
});
wz.on("save_complete", function(data) {
  if (data.request_id !== activeDialogueRequestId) return;
  document.getElementById("download").disabled = false;
  sourceErrorMessage = "";
  setStatus("Saved to " + data.path, "ready");
});
wz.on("save_cancelled", function(data) {
  if (data.request_id !== activeDialogueRequestId) return;
  document.getElementById("download").disabled = false;
});
wz.on("save_error", function(data) {
  if (data.request_id !== activeDialogueRequestId) return;
  document.getElementById("download").disabled = false;
  sourceErrorMessage = "";
  setStatus(data.message, "error");
});
document.getElementById("source").addEventListener("input", function() {
  invalidateRhythm();
  ++sourceRequestId;
  var ownedRequestId = null;
  if (sourceRollbackState && sourceRollbackState.dialogueOwned) {
    ownedRequestId = sourceRollbackState.requestId;
  } else if (dialogueOwned) {
    ownedRequestId = activeDialogueRequestId;
  }
  sourceRollbackState = null;
  activeDialogueRequestId = ++dialogueRequestSequence;
  dialogueOwned = false;
  sourceErrorMessage = "";
  if (ownedRequestId !== null) {
    wz.send("invalidate_dialogue", {request_id:ownedRequestId});
  }
  resetAudio();
  document.getElementById("read").disabled = false;
  setReadBusy(false);
  setStatus("", "idle");
});
</script>
</body>
</html>"""


def _build_dialogue_html(initial_text: str) -> str:
    from .rhythm_view import inject_rhythm_assets

    escaped = (
        initial_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return inject_rhythm_assets(_DIALOGUE_HTML.replace("__INITIAL_TEXT__", escaped))


def open_dialogue_panel(wz, initial_text: str = "") -> None:
    """Open the dialogue reader and connect its native handlers."""
    if _dialogue_panel_ref[0] is not None:
        try:
            _dialogue_panel_ref[0].close()
        except Exception:
            pass

    panel = wz.ui.webview_panel(
        title="Dialogue Reader",
        html=_build_dialogue_html(initial_text),
        width=760,
        height=680,
        floating=False,
    )
    panel.show()
    _dialogue_panel_ref[0] = panel
    audio_state = _DialogueAudioState()
    rhythm_request = [None]

    def _on_close() -> None:
        if _dialogue_panel_ref[0] is panel:
            _dialogue_panel_ref[0] = None
        audio_state.retire()
        rhythm_request[0] = None

    panel.on_close(_on_close)

    def _send_panel(event: str, data: dict) -> None:
        """Send WebView events on the AppKit main thread."""
        if threading.current_thread() is threading.main_thread():
            panel.send(event, data)
            return

        from PyObjCTools import AppHelper

        AppHelper.callAfter(lambda: panel.send(event, data))

    def _send_source(path: str, request_id) -> None:
        async def _load() -> None:
            try:
                logger.info("Reading dialogue source: %s", Path(path).name)
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(None, _read_source, path)
                logger.info(
                    "Dialogue source recognized: %s (%d characters)",
                    Path(path).name,
                    len(text),
                )
                _send_panel(
                    "source_loaded",
                    {"text": text, "request_id": request_id},
                )
            except Exception as exc:
                logger.warning(
                    "Failed to read dialogue source %s: %s",
                    Path(path).name,
                    exc,
                )
                _send_panel(
                    "source_error",
                    {"message": str(exc), "request_id": request_id},
                )

        wz.run(_load())

    def _on_choose_file(data) -> None:
        request_id = (data or {}).get("request_id")
        try:
            from AppKit import NSModalResponseOK, NSOpenPanel

            picker = NSOpenPanel.openPanel()
            picker.setCanChooseDirectories_(False)
            picker.setCanChooseFiles_(True)
            picker.setAllowsMultipleSelection_(False)
            picker.setAllowedFileTypes_(
                [
                    "txt",
                    "text",
                    "md",
                    "png",
                    "jpg",
                    "jpeg",
                    "heic",
                    "tif",
                    "tiff",
                    "webp",
                ]
            )

            if picker.runModal() == NSModalResponseOK:
                path = str(picker.URL().path())
                _send_panel(
                    "source_loading",
                    {
                        "message": f"Reading {Path(path).name}...",
                        "request_id": request_id,
                    },
                )
                _send_source(path, request_id)
            else:
                _send_panel("source_cancelled", {"request_id": request_id})
        except Exception as exc:
            _send_panel(
                "source_error",
                {"message": str(exc), "request_id": request_id},
            )

    def _on_load_clipboard(data) -> None:
        request_id = (data or {}).get("request_id")
        text = wz.pasteboard.get() or ""
        history = wz.pasteboard.history(limit=1)
        latest = history[0] if history else {}
        image_path = latest.get("image_path", "")
        if image_path and not text.strip():
            _send_source(image_path, request_id)
        elif text.strip():
            _send_panel(
                "source_loaded",
                {"text": text, "request_id": request_id},
            )
        else:
            _send_panel(
                "source_error",
                {
                    "message": "Clipboard has no text or image",
                    "request_id": request_id,
                },
            )

    def _on_read_dialogue(data) -> None:
        request_id = (data or {}).get("request_id")
        owner = audio_state.begin(request_id)
        try:
            voiced_segments, roles = assign_voices(parse_dialogue((data or {}).get("text", "")))
        except Exception as exc:
            audio_state.retire(owner)
            _send_panel(
                "dialogue_error",
                {"message": str(exc), "request_id": request_id},
            )
            return

        _send_panel(
            "dialogue_parsed",
            {
                "segments": voiced_segments,
                "roles": roles,
                "request_id": request_id,
            },
        )

        async def _generate() -> None:
            try:
                audio = await generate_dialogue_audio_bytes(
                    voiced_segments,
                    should_continue=lambda: (
                        audio_state.is_current(owner)
                        and _dialogue_panel_ref[0] is panel
                    ),
                )
                if not audio_state.commit(
                    owner,
                    audio,
                    panel_is_current=lambda: _dialogue_panel_ref[0] is panel,
                ):
                    return
                _send_panel(
                    "dialogue_audio",
                    {
                        "url": _audio_data_url(audio),
                        "segments": len(voiced_segments),
                        "request_id": request_id,
                    },
                )
            except _DialogueSuperseded:
                audio_state.retire(owner)
                return
            except Exception as exc:
                if audio_state.retire(owner):
                    _send_panel(
                        "dialogue_error",
                        {"message": str(exc), "request_id": request_id},
                    )

        wz.run(_generate())

    def _on_invalidate_dialogue(data) -> None:
        request_id = (data or {}).get("request_id")
        audio_state.retire_request(request_id)

    def _on_request_rhythm(data) -> None:
        from .rhythm import analyze_rhythm

        request_id = (data or {}).get("request_id")
        owner = {"request_id": request_id}
        rhythm_request[0] = owner

        def _is_current() -> bool:
            return rhythm_request[0] is owner and _dialogue_panel_ref[0] is panel

        def _send_rhythm(event: str, payload: dict) -> None:
            def _deliver() -> None:
                if _is_current():
                    panel.send(event, payload)

            if threading.current_thread() is threading.main_thread():
                _deliver()
            else:
                from PyObjCTools import AppHelper

                AppHelper.callAfter(_deliver)

        try:
            segments, roles = assign_voices(parse_dialogue((data or {}).get("text", "")))
        except Exception as exc:
            _send_rhythm(
                "rhythm_error",
                {"message": str(exc), "request_id": request_id},
            )
            return

        # Speaker labels are visual metadata, not words the learner will say.
        text = "\n\n".join(segment["text"] for segment in segments)
        _send_rhythm(
            "rhythm_parsed",
            {"segments": segments, "roles": roles, "request_id": request_id},
        )

        def _on_progress(done: int, total: int) -> None:
            if _is_current():
                _send_rhythm(
                    "rhythm_progress",
                    {"done": done, "total": total, "request_id": request_id},
                )

        async def _analyze() -> None:
            if not _is_current():
                return
            try:
                result = await analyze_rhythm(
                    text, on_progress=_on_progress, should_continue=_is_current
                )
                if _is_current():
                    _send_rhythm(
                        "rhythm_result", {"result": result, "request_id": request_id}
                    )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if _is_current():
                    _send_rhythm(
                        "rhythm_error", {"message": str(exc), "request_id": request_id}
                    )

        wz.run(_analyze())

    def _on_invalidate_rhythm(data) -> None:
        owner = rhythm_request[0]
        if owner is not None and owner["request_id"] == (data or {}).get("request_id"):
            rhythm_request[0] = None

    def _on_save_audio(data) -> None:
        request_id = (data or {}).get("request_id")
        audio = audio_state.audio_for(request_id)
        if audio is None:
            _send_panel(
                "save_error",
                {"message": "No audio is ready to save", "request_id": request_id},
            )
            return

        try:
            from AppKit import NSModalResponseOK, NSSavePanel

            picker = NSSavePanel.savePanel()
            picker.setTitle_("Save Dialogue Audio")
            picker.setCanCreateDirectories_(True)
            picker.setAllowedFileTypes_(["mp3"])
            picker.setNameFieldStringValue_("dialogue.mp3")
            if picker.runModal() != NSModalResponseOK:
                _send_panel("save_cancelled", {"request_id": request_id})
                return

            output_path = Path(str(picker.URL().path()))
            output_path.write_bytes(audio)
            _send_panel(
                "save_complete",
                {"path": str(output_path), "request_id": request_id},
            )
        except Exception as exc:
            _send_panel(
                "save_error",
                {"message": str(exc), "request_id": request_id},
            )

    panel.on("choose_file", _on_choose_file)
    panel.on("load_clipboard", _on_load_clipboard)
    panel.on("read_dialogue", _on_read_dialogue)
    panel.on("invalidate_dialogue", _on_invalidate_dialogue)
    panel.on("request_rhythm", _on_request_rhythm)
    panel.on("invalidate_rhythm", _on_invalidate_rhythm)
    panel.on("save_audio", _on_save_audio)
