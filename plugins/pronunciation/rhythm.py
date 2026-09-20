"""Suggested sentence prominence and thought groups for everyday American speech."""

from __future__ import annotations

import asyncio
import json
import re

from .analyze import _call_llm, _InvalidAnalysisResponse

_CHUNK_WORDS = 120
_MAX_CHARACTERS = 20_000
_PROMPT = """You coach natural, everyday General American speech at a moderate pace.
Suggest one useful reading of the supplied text for a learner to shadow.
Use neutral broad focus unless the text itself establishes contrast. Do not claim
to know a speaker's actual intention or recording: no audio has been supplied.

Return only a JSON object:
{
  "stress": [{"index": 0, "word": "exact supplied token", "note": "Brief Chinese reason"}],
  "pauses": [{"after": 3, "word": "exact supplied token", "kind": "short", "note": "Brief Chinese reason"}],
  "summary": "One brief Chinese explanation of this suggested reading"
}

The supplied tokens are numbered; use their exact index and word, including
punctuation. Never rewrite the input or silently correct its grammar. Surrounding
context, if supplied, is for understanding only and must not be annotated.

Stress means sentence prominence, not dictionary syllable stress. Choose a few
salient information words, typically the final key word in each thought group.
Other content words may receive lighter prominence, but do not highlight every
content word. Unstressed words are still pronounced; color does not mean shouting.
Explain briefly in Chinese why these words carry the message in THIS reading.

Pauses mark thought-group boundaries, not every stressed word. Use kind "short"
for a suggested small boundary and "long" for a sentence/major boundary.
Short sentences can have no internal pauses. Keep determiners with their nouns,
adjectives with their nouns, and prepositions with their objects. Grammar and
collocations do not force a single spoken grouping: "express INTEREST | in a
POSITION" is a possible deliberate reading, even though "interest in" is a
collocation. A fluent reading without that internal break is also possible.
Do not split mechanically after every verb or collocation. Prefer few meaningful
boundaries. Prominence or phrase-final lengthening need not contain silence.
Never provide precise audio timing. Do not add a boundary simply because the
supplied text is one chunk of a longer passage. Empty stress/pauses lists are
allowed when there is nothing useful to mark. Keep notes concise.
"""


def _validate_rhythm(data: dict, text: str) -> dict:
    words = text.split()
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("stress"), list)
        or not isinstance(data.get("pauses"), list)
        or not isinstance(data.get("summary"), str)
        or not data["summary"].strip()
    ):
        raise _InvalidAnalysisResponse("AI returned incomplete stress and pause hints. Please try again.")

    def valid(item, key):
        if not isinstance(item, dict):
            return False
        index = item.get(key)
        return (
            type(index) is int and 0 <= index < len(words)
            and item.get("word") == words[index]
            and any(char.isalpha() for char in words[index])
            and isinstance(item.get("note"), str) and bool(item["note"].strip())
        )

    stress, pauses = {}, {}
    for item in data["stress"]:
        if valid(item, "index"):
            index = item["index"]
            stress.setdefault(index, {"index": index, "note": item["note"].strip()})
    for item in data["pauses"]:
        if valid(item, "after") and item.get("kind") in ("short", "long"):
            index = item["after"]
            pauses.setdefault(index, {"after": index, "kind": item["kind"], "note": item["note"].strip()})
    if data["stress"] and not stress:
        raise _InvalidAnalysisResponse("AI stress hints did not match the supplied words. Please try again.")
    return {
        "text": text, "words": words,
        "stress": [stress[index] for index in sorted(stress)],
        "pauses": [pauses[index] for index in sorted(pauses)],
        "summary": data["summary"].strip(),
    }


def _chunks(text: str, matches: list) -> list[tuple[int, int]]:
    chunks = []
    start = 0
    while start < len(matches):
        end = min(start + _CHUNK_WORDS, len(matches))
        if end < len(matches):
            for candidate in range(end, start + _CHUNK_WORDS // 2, -1):
                previous = matches[candidate - 1]
                gap = text[previous.end():matches[candidate].start()]
                if re.search(r'[.!?;:]["\u201d\u2019\)\]]*$', previous.group()) or "\n\n" in gap:
                    end = candidate
                    break
        chunks.append((start, end))
        start = end
    return chunks


async def analyze_rhythm(text: str, config=None, *, on_progress=None, should_continue=None) -> dict:
    """Analyze bounded chunks; preserve original tokens and ignore superseded work."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to mark stress and pauses.")
    if len(text) > _MAX_CHARACTERS:
        raise ValueError("Use at most 20,000 characters for stress and pause hints.")

    def check_current():
        if should_continue is not None and not should_continue():
            raise asyncio.CancelledError

    check_current()
    if config is None:
        from wenzi.config import load_config

        config, _ = load_config()

    matches = list(re.finditer(r"\S+", text))
    chunks = _chunks(text, matches)
    result = {"text": text, "words": [match.group() for match in matches], "stress": [], "pauses": [], "summary": ""}
    if on_progress:
        on_progress(0, len(chunks))
    for number, (start, end) in enumerate(chunks, 1):
        check_current()
        chunk_text = text[matches[start].start():matches[end - 1].end()]
        payload = {
            "text": chunk_text,
            "tokens": [{"index": index, "word": word} for index, word in enumerate(chunk_text.split())],
            "context_before": " ".join(result["words"][max(0, start - 20):start]),
            "context_after": " ".join(result["words"][end:end + 20]),
        }
        part = await _call_llm(
            _PROMPT, json.dumps(payload, ensure_ascii=False), config,
            validator=lambda data, _: _validate_rhythm(data, chunk_text),
        )
        check_current()
        result["stress"].extend({**item, "index": item["index"] + start} for item in part["stress"])
        for item in part["pauses"]:
            after = item["after"] + start
            # Artificial request boundaries must not become pronunciation advice.
            if after == end - 1 and end < len(matches):
                gap = text[matches[after].end():matches[end].start()]
                if not re.search(r'[.!?;:,]["\u201d\u2019\)\]]*$', matches[after].group()) and "\n\n" not in gap:
                    continue
            result["pauses"].append({**item, "after": after})
        if len(chunks) == 1:
            result["summary"] = part["summary"]
        if on_progress:
            on_progress(number, len(chunks))
    return result
