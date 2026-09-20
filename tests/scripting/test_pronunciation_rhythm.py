"""Sentence rhythm annotations retain their alignment without network access."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def rhythm(monkeypatch):
    package_name = "pronunciation_rhythm_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "plugins" / "pronunciation")]
    monkeypatch.setitem(sys.modules, package_name, package)
    for name in ("analyze", "rhythm"):
        monkeypatch.delitem(sys.modules, f"{package_name}.{name}", raising=False)
    return importlib.import_module(f"{package_name}.rhythm")


def guide(*, stress=None, pauses=None, summary="Keep the main idea clear."):
    return {
        "stress": [] if stress is None else stress,
        "pauses": [] if pauses is None else pauses,
        "summary": summary,
    }


def stress(index, word, note="Give this word more prominence."):
    return {"index": index, "word": word, "note": note}


def pause(after, word, kind="short", note="Briefly separate the thought groups."):
    return {"after": after, "word": word, "kind": kind, "note": note}


def fake_calls(monkeypatch, rhythm, response):
    calls = []

    async def call(prompt, user_text, config, *, validator):
        payload = json.loads(user_text)
        calls.append(payload)
        assert config == {"test": True}
        raw = response(payload) if callable(response) else copy.deepcopy(response)
        return validator(raw, user_text)

    monkeypatch.setattr(rhythm, "_call_llm", call)
    return calls


def analyze(rhythm, text, **kwargs):
    return asyncio.run(rhythm.analyze_rhythm(text, config={"test": True}, **kwargs))


def test_repeated_words_use_exact_indices_and_preserve_source(rhythm):
    text = "  We\twant,\n\nwe want change.  "
    raw = guide(
        stress=[stress(4, "change."), stress(2, "we"), stress(4, "change.")],
        pauses=[pause(4, "change.", "long"), pause(1, "want,"), pause(1, "want,")],
    )
    original = copy.deepcopy(raw)

    result = rhythm._validate_rhythm(raw, text)

    assert result["text"] == text
    assert result["words"] == ["We", "want,", "we", "want", "change."]
    assert [item["index"] for item in result["stress"]] == [2, 4]
    assert [item["after"] for item in result["pauses"]] == [1, 4]
    assert all(set(item) == {"index", "note"} for item in result["stress"])
    assert all(set(item) == {"after", "kind", "note"} for item in result["pauses"])
    assert result["summary"] == raw["summary"]
    assert raw == original


@pytest.mark.parametrize("index", [True, False, -1, 3, 1.0, "1", None])
def test_invalid_stress_indices_are_discarded(rhythm, index):
    result = rhythm._validate_rhythm(
        guide(stress=[stress(2, "you."), stress(index, "see")]), "I see you."
    )
    assert [item["index"] for item in result["stress"]] == [2]


@pytest.mark.parametrize(
    "bad",
    [None, "see", {}, stress(1, "See"), stress(2, "you"), stress(1, "see", " "), stress(1, "see", None)],
)
def test_invalid_stress_annotations_do_not_shift_valid_words(rhythm, bad):
    result = rhythm._validate_rhythm(
        guide(stress=[bad, stress(0, "I")]), "I see you."
    )
    assert [item["index"] for item in result["stress"]] == [0]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        pause(True, "see"),
        pause(-1, "you."),
        pause(3, "you."),
        pause(1.0, "see"),
        pause("1", "see"),
        pause(1, "See"),
        pause(2, "you"),
        pause(1, "see", "medium"),
        pause(1, "see", note=" "),
        pause(1, "see", note=None),
    ],
)
def test_invalid_pauses_are_discarded_without_inventing_stress(rhythm, bad):
    result = rhythm._validate_rhythm(
        guide(pauses=[bad, pause(2, "you.", "long")]), "I see you."
    )
    assert result["stress"] == []
    assert [item["after"] for item in result["pauses"]] == [2]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"stress": [], "summary": "A guide."},
        {"pauses": [], "summary": "A guide."},
        {"stress": [], "pauses": []},
        guide(stress="word"),
        guide(pauses="pause"),
        guide(summary=" "),
        guide(summary=None),
        guide(stress=[stress(1, "wrong")]),
    ],
)
def test_invalid_guides_raise_retryable_analysis_error(rhythm, raw):
    with pytest.raises(rhythm._InvalidAnalysisResponse):
        rhythm._validate_rhythm(raw, "Say it.")


def test_function_word_can_receive_contrastive_stress(rhythm):
    result = rhythm._validate_rhythm(guide(stress=[stress(0, "I")]), "I said it.")
    assert result["stress"][0]["index"] == 0


def test_empty_stress_is_valid_for_a_guide_without_prominent_words(rhythm):
    result = rhythm._validate_rhythm(guide(), "Okay.")
    assert result["stress"] == []
    assert result["pauses"] == []


@pytest.mark.parametrize("text", ["", " \n\t ", "a" * 20001])
def test_invalid_input_never_calls_model(monkeypatch, rhythm, text):
    calls = fake_calls(monkeypatch, rhythm, guide())
    with pytest.raises(ValueError):
        analyze(rhythm, text)
    assert calls == []


def test_sentence_annotations_are_validated_against_numbered_tokens(monkeypatch, rhythm):
    text = "I said I would."
    calls = fake_calls(monkeypatch, rhythm, guide(stress=[stress(2, "I")]))

    result = analyze(rhythm, text)

    assert result["text"] == text
    assert result["words"] == text.split()
    assert [item["index"] for item in result["stress"]] == [2]
    assert calls[0]["tokens"] == [
        {"index": index, "word": word} for index, word in enumerate(text.split())
    ]


def test_chunk_offsets_preserve_words_and_remove_artificial_pauses(monkeypatch, rhythm):
    words = [f"word{index}" for index in range(265)]
    text = "  " + " \t".join(words) + "\n"

    def response(payload):
        chunk_words = payload["text"].split()
        return guide(
            stress=[stress(0, chunk_words[0]), stress(len(chunk_words) - 1, chunk_words[-1])],
            pauses=[pause(len(chunk_words) - 1, chunk_words[-1])],
        )

    calls = fake_calls(monkeypatch, rhythm, response)
    progress = []
    result = analyze(rhythm, text, on_progress=lambda done, total: progress.append((done, total)))

    assert result["text"] == text
    assert result["words"] == words
    assert [word for call in calls for word in call["text"].split()] == words
    assert len(calls) >= 3
    assert all(0 < len(call["text"].split()) <= 120 for call in calls)
    offset = 0
    expected = []
    for call in calls:
        length = len(call["text"].split())
        expected.extend([offset, offset + length - 1])
        offset += length
    assert [item["index"] for item in result["stress"]] == sorted(set(expected))
    assert [item["after"] for item in result["pauses"]] == [264]
    assert progress[0] == (0, len(calls))
    assert progress[-1] == (len(calls), len(calls))
    assert all(total == len(calls) and 0 <= done <= total for done, total in progress)
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)


@pytest.mark.parametrize("separator", [". ", "\n\n"])
def test_chunks_prefer_sentence_or_paragraph_boundaries(monkeypatch, rhythm, separator):
    first = " ".join(f"first{index}" for index in range(100))
    second = " ".join(f"second{index}" for index in range(50))
    text = first + separator + second

    def response(payload):
        chunk_words = payload["text"].split()
        return guide(pauses=[pause(len(chunk_words) - 1, chunk_words[-1])])

    calls = fake_calls(monkeypatch, rhythm, response)
    result = analyze(rhythm, text)

    assert len(calls[0]["text"].split()) == 100
    assert [item["after"] for item in result["pauses"]] == [99, 149]
    assert result["text"] == text


def test_cancellation_before_request_skips_model(monkeypatch, rhythm):
    calls = fake_calls(monkeypatch, rhythm, guide())
    with pytest.raises(asyncio.CancelledError):
        analyze(rhythm, "Say it.", should_continue=lambda: False)
    assert calls == []


def test_cancellation_during_request_discards_result_and_stops_chunks(monkeypatch, rhythm):
    state = {"continue": True}

    def response(payload):
        state["continue"] = False
        return guide()

    calls = fake_calls(monkeypatch, rhythm, response)
    with pytest.raises(asyncio.CancelledError):
        analyze(rhythm, "word " * 250, should_continue=lambda: state["continue"])
    assert len(calls) == 1
