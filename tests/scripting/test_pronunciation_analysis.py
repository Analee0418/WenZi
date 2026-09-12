"""Regression coverage for pronunciation analysis without user config or network."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

SENTENCE = "Air quality has been improving for several years."
VALID_DATA = {
    "words": [
        {"word": "Air", "ipa": "/er/"},
        {"word": "quality", "ipa": "/ˈkwɑləti/"},
        {"word": "has", "ipa": "/hæz/"},
        {"word": "been", "ipa": "/bɪn/"},
        {"word": "improving", "ipa": "/ɪmˈpruvɪŋ/"},
        {"word": "for", "ipa": "/fɔr/"},
        {"word": "several", "ipa": "/ˈsɛvərəl/"},
        {"word": "years.", "ipa": "/jɪrz/"},
    ],
    "connected": "/er ˈkwɑləti/",
    "translation": "空气质量。",
    "links": [],
}
VALID_JSON = json.dumps(VALID_DATA, ensure_ascii=False)
LINKING_DATA = {
    "words": [
        {"word": "Take", "ipa": "/teɪk/"},
        {"word": "it", "ipa": "/ɪt/"},
        {"word": "easy.", "ipa": "/ˈizi/"},
    ],
    "connected": "/teɪk ɪt ˈizi/",
    "translation": "放轻松。",
    "links": [],
}


def completion(content=VALID_JSON, finish_reason="stop"):
    return {
        "choices": [
            {"message": {"content": content}, "finish_reason": finish_reason}
        ]
    }


@pytest.fixture
def analyzer():
    spec = importlib.util.spec_from_file_location(
        "pronunciation_analysis_under_test", Path(__file__).resolve().parents[2] / "plugins" / "pronunciation" / "analyze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    return {
        "ai_enhance": {
            "default_provider": "test-provider",
            "default_model": "deepseek-v4-flash",
            "providers": {
                "test-provider": {
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "dummy-test-key",
                    "extra_body": {
                        "thinking": {"type": "enabled"},
                        "reasoning_effort": "high",
                        "response_format": {"type": "text"},
                        "temperature": 0.3,
                        "metadata": {"label": "pronunciation-test"},
                    },
                }
            },
        }
    }


@pytest.fixture
def fake_client(monkeypatch):
    calls = []
    responses = []
    initializations = []

    class APIError(Exception):
        pass

    class ChatClient:
        def __init__(self, **kwargs):
            initializations.append(kwargs)

        async def create(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            if not responses:
                raise AssertionError("Unexpected additional LLM request")
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return copy.deepcopy(result)

        async def close(self):
            pass

    # Importing the real package can initialize macOS integrations and vaults.
    package = ModuleType("wenzi")
    package.__path__ = []
    http_module = ModuleType("wenzi.llm_http")
    http_module.ChatClient = ChatClient
    http_module.APIError = APIError
    monkeypatch.setitem(sys.modules, "wenzi", package)
    monkeypatch.setitem(sys.modules, "wenzi.llm_http", http_module)
    return responses, calls, initializations, APIError


def run_analysis(analyzer, config, sentence=SENTENCE):
    return asyncio.run(analyzer.analyze_pronunciation(sentence, config=config))


def assert_analysis(result, expected, pairs):
    assert result | {"links": [], "features": []} == expected | {"links": [], "features": []}
    assert [(link["from"], link["to"]) for link in result["links"]] == pairs
    assert all(isinstance(link["note"], str) and link["note"].strip() for link in result["links"])


def assert_standard_analysis(result):
    assert_analysis(result, VALID_DATA, [(3, 4)])


def assert_informative_error(error):
    assert not isinstance(error, (KeyError, IndexError, TypeError, json.JSONDecodeError))
    assert len(str(error).strip()) >= 12


@pytest.mark.parametrize(
    "model", ["deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner", "new-alias"]
)
def test_deepseek_uses_non_thinking_json_without_mutating_config(
    analyzer, config, fake_client, model
):
    responses, calls, initializations, _ = fake_client
    config["ai_enhance"]["default_model"] = model
    original = copy.deepcopy(config)
    responses.append(completion())

    assert_standard_analysis(run_analysis(analyzer, config))

    assert len(calls) == 1
    request = calls[0]
    assert request["model"] == model
    assert request["max_tokens"] == 2048
    assert request["messages"][-1] == {"role": "user", "content": SENTENCE}
    assert request["extra_body"]["thinking"] == {"type": "disabled"}
    assert request["extra_body"]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in request["extra_body"]
    assert request["extra_body"]["temperature"] == 0.3
    assert request["extra_body"]["metadata"] == {"label": "pronunciation-test"}
    assert initializations == [
        {"base_url": "https://api.deepseek.com/v1", "api_key": "dummy-test-key"}
    ]
    assert config == original


@pytest.mark.parametrize("model", ["another-model", "deepseek-v4-flash"])
def test_third_party_provider_preserves_options(analyzer, config, fake_client, model):
    responses, calls, _, _ = fake_client
    config["ai_enhance"]["default_model"] = model
    config["ai_enhance"]["providers"]["test-provider"]["base_url"] = (
        "https://example.invalid/v1"
    )
    original = copy.deepcopy(config)
    responses.append(completion())

    assert_standard_analysis(run_analysis(analyzer, config))

    assert calls[0]["extra_body"] == original["ai_enhance"]["providers"][
        "test-provider"
    ]["extra_body"]
    assert config == original


def test_compatible_provider_may_omit_finish_reason(analyzer, config, fake_client):
    responses, calls, _, _ = fake_client
    config["ai_enhance"]["default_model"] = "another-model"
    config["ai_enhance"]["providers"]["test-provider"]["base_url"] = (
        "https://example.invalid/v1"
    )
    responses.append({"choices": [{"message": {"content": VALID_JSON}}]})

    assert_standard_analysis(run_analysis(analyzer, config))
    assert len(calls) == 1


def test_provider_token_limit_cannot_override_recovery_budget(
    analyzer, config, fake_client
):
    responses, calls, _, _ = fake_client
    config["ai_enhance"]["providers"]["test-provider"]["extra_body"]["max_tokens"] = 1
    original = copy.deepcopy(config)
    responses.extend([completion(""), completion()])

    assert_standard_analysis(run_analysis(analyzer, config))

    # ChatClient merges extra_body last before sending the HTTP request.
    payloads = [call | call["extra_body"] for call in calls]
    assert [payload["max_tokens"] for payload in payloads] == [2048, 4096]
    assert config == original


@pytest.mark.parametrize(
    "content",
    [
        VALID_JSON,
        f"```json\n{VALID_JSON}\n```",
        f"<think>Internal analysis only.</think>\n{VALID_JSON}",
        f"<think>Internal analysis only.</think>\n```json\n{VALID_JSON}\n```",
    ],
    ids=["json", "fenced-json", "think-wrapper", "think-and-fence"],
)
def test_accepts_supported_json_wrappers(analyzer, config, fake_client, content):
    responses, calls, _, _ = fake_client
    responses.append(completion(content))
    assert_standard_analysis(run_analysis(analyzer, config))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        completion(None),
        completion(""),
        completion(" \n\t "),
        completion("<think>No final answer.</think>"),
        completion("<think>Unfinished reasoning"),
        completion("This is not JSON."),
        completion("{\"words\": ["),
        completion("null"),
        completion("42"),
        completion(json.dumps([VALID_DATA])),
        completion({"words": []}),
        {},
        {"choices": []},
        {"choices": None},
        {"choices": [None]},
        {"choices": [{"finish_reason": "stop"}]},
        {"choices": [{"finish_reason": "stop", "message": None}]},
    ],
    ids=[
        "null-content",
        "empty-content",
        "whitespace-content",
        "think-only",
        "unfinished-think",
        "invalid-json",
        "incomplete-json",
        "json-null",
        "json-number",
        "json-array-with-valid-object",
        "non-string-content",
        "missing-choices",
        "empty-choices",
        "null-choices",
        "null-choice",
        "missing-message",
        "null-message",
    ],
)
def test_recovers_once_from_unusable_response(analyzer, config, fake_client, response):
    responses, calls, _, _ = fake_client
    responses.extend([response, completion()])

    assert_standard_analysis(run_analysis(analyzer, config))

    assert [call["max_tokens"] for call in calls] == [2048, 4096]


@pytest.mark.parametrize(
    "field,value",
    [
        ("words", None),
        ("words", []),
        ("words", "Air"),
        ("words", [None]),
        ("words", [{"word": "Air"}]),
        ("words", [{"ipa": "/er/"}]),
        ("words", [{"word": 1, "ipa": "/er/"}]),
        ("words", [{"word": "Air", "ipa": []}]),
        ("words", [{"word": " ", "ipa": "/er/"}]),
        ("words", [{"word": "Air", "ipa": "\n"}]),
        ("connected", None),
        ("connected", []),
        ("connected", " "),
        ("translation", None),
        ("translation", 1),
        ("translation", "\t"),
    ],
)
def test_recovers_once_from_invalid_fields(
    analyzer, config, fake_client, field, value
):
    responses, calls, _, _ = fake_client
    data = copy.deepcopy(VALID_DATA)
    data[field] = value
    responses.extend([completion(json.dumps(data)), completion()])

    assert_standard_analysis(run_analysis(analyzer, config))
    assert [call["max_tokens"] for call in calls] == [2048, 4096]


@pytest.mark.parametrize("field", ["words", "connected", "translation"])
def test_recovers_once_from_missing_field(analyzer, config, fake_client, field):
    responses, calls, _, _ = fake_client
    data = copy.deepcopy(VALID_DATA)
    del data[field]
    responses.extend([completion(json.dumps(data)), completion()])

    assert_standard_analysis(run_analysis(analyzer, config))
    assert [call["max_tokens"] for call in calls] == [2048, 4096]


def test_length_finish_reason_retries_even_with_valid_json(analyzer, config, fake_client):
    responses, calls, _, _ = fake_client
    responses.extend([completion(finish_reason="length"), completion()])

    assert_standard_analysis(run_analysis(analyzer, config))
    assert [call["max_tokens"] for call in calls] == [2048, 4096]


@pytest.mark.parametrize("reason", ["content_filter", "tool_calls", "unexpected"])
def test_non_stop_finish_reason_fails_without_retry(
    analyzer, config, fake_client, reason
):
    responses, calls, _, _ = fake_client
    responses.append(completion(finish_reason=reason))

    with pytest.raises(Exception) as caught:
        run_analysis(analyzer, config)

    assert_informative_error(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [completion(""), completion("not json"), completion(finish_reason="length")],
    ids=["empty", "malformed", "truncated"],
)
def test_stops_after_one_recovery_attempt(analyzer, config, fake_client, response):
    responses, calls, _, _ = fake_client
    responses.extend([response, response])

    with pytest.raises(Exception) as caught:
        run_analysis(analyzer, config)

    assert_informative_error(caught.value)
    assert [call["max_tokens"] for call in calls] == [2048, 4096]


def test_transport_error_is_not_retried(analyzer, config, fake_client):
    responses, calls, _, APIError = fake_client
    error = APIError("HTTP 429: test rate limit")
    responses.append(error)

    with pytest.raises(APIError) as caught:
        run_analysis(analyzer, config)

    assert caught.value is error
    assert len(calls) == 1


def test_recovery_diagnostics_do_not_expose_response_or_credentials(
    analyzer, config, fake_client, caplog
):
    responses, _, _, _ = fake_client
    marker = "PRIVATE_RESPONSE_MARKER"
    responses.extend([completion(marker), completion(marker)])

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception) as caught:
            run_analysis(analyzer, config)

    assert_informative_error(caught.value)
    diagnostics = caplog.text + str(caught.value)
    assert marker not in diagnostics
    assert "dummy-test-key" not in diagnostics


@pytest.mark.parametrize("name,sentence,pairs", [
    ("care-about.json", "I care about the environment", [(1, 2)]),
    ("powers-all.json", "Natural gas powers all city buildings", [(2, 3)]),
    ("get-it-out.json", "Get it out of the way", [(0, 1), (1, 2), (2, 3)]),
    ("water.json", "Water", []),
])
def test_actual_examples_do_not_depend_on_model_links(analyzer, name, sentence, pairs):
    data = json.loads((Path(__file__).with_name("fixtures") / "pronunciation" / name).read_text())
    data["links"] = [{"from": 99, "to": 100, "note": "Incorrect model suggestion"}]
    result = analyzer._validate_analysis(data, sentence)
    assert [(link["from"], link["to"]) for link in result["links"]] == pairs
    assert result["features"]


@pytest.mark.parametrize("ipa", ["/ˈpaʊərz/", "/ˈpaʊə˞z/", "[ˈpaʊɚz]"])
def test_rhotic_notations_keep_powers_all_link(analyzer, ipa):
    words = [{"word": "powers", "ipa": ipa}, {"word": "all", "ipa": "/ɔl/"}]
    assert len(analyzer._derive_links(words, "powers all")) == 1


@pytest.mark.parametrize("left,right", [
    ("powers", "/ɔl/"), ("/powers or power/", "/ɔl/"),
    ("/paʊərz/", "all"), ("/paʊərz/", "/ɔl or ɑl/"),
    ("/paʊərz/", "/sɪti/"), ("/aɪ/", "/əbaʊt/"),
])
def test_spelling_ambiguous_ipa_and_non_cv_pairs_have_no_link(analyzer, left, right):
    words = [{"word": "powers", "ipa": left}, {"word": "all", "ipa": right}]
    assert analyzer._derive_links(words, "powers all") == []


@pytest.mark.parametrize("punct", [".", ",", "!", "?", ";", ":", '."', '?”'])
@pytest.mark.parametrize("source_only", [True, False])
def test_source_or_model_pause_blocks_links(analyzer, punct, source_only):
    words = [{"word": "Take" if source_only else "Take" + punct, "ipa": "/teɪk/"},
             {"word": "it", "ipa": "/ɪt/"}]
    assert analyzer._derive_links(words, "Take" + (punct if source_only else "") + " it") == []


@pytest.mark.parametrize("sentence", ["Take this easy.", "Take it", "it Take easy."])
def test_alignment_mismatch_keeps_core_data_but_omits_links(analyzer, sentence):
    result = analyzer._validate_analysis(copy.deepcopy(LINKING_DATA), sentence)
    assert result["connected"] == LINKING_DATA["connected"]
    assert result["links"] == []


def feature(**changes):
    return {"words": [0], "kind": "flap", "ipa": "/ˈwɔɾɚ/", "note": "A light American flap.", **changes}


@pytest.mark.parametrize("changes", [
    {"words": []}, {"words": [True]}, {"words": [-1]}, {"words": [3]},
    {"words": [0, 2]}, {"words": [1, 0]}, {"words": [0, 1, 2]},
    {"words": "0"}, {"kind": []}, {"kind": "elision"},
    {"ipa": ""}, {"ipa": "water"}, {"ipa": "/ /"}, {"note": " "},
    {"ipa": "/ˈwɔtɚ/"}, {"kind": "unreleased_stop", "ipa": "/stɑp/"},
])
def test_bad_optional_feature_is_ignored(analyzer, changes):
    assert analyzer._validate_features([feature(**changes)], 3) == []


@pytest.mark.parametrize("value", [None, "invalid", {}, 1, [None]])
def test_missing_or_bad_features_are_optional(analyzer, value):
    result = analyzer._validate_analysis(VALID_DATA | {"features": value}, SENTENCE)
    assert result["features"] == []
    assert len(result["links"]) == 1


def test_valid_features_and_duplicates(analyzer):
    flap = feature()
    stop = feature(words=[1, 2], kind="unreleased_stop", ipa="/əbaʊt̚ ðə/")
    assert analyzer._validate_features([flap, flap, stop], 3) == [flap, stop]


def test_single_word_flap_is_independent_of_links(analyzer):
    data = {"words": [{"word": "water", "ipa": "/ˈwɔtər/"}],
            "connected": "/ˈwɔɾɚ/", "translation": "水", "features": [feature()]}
    result = analyzer._validate_analysis(data, "water")
    assert result["links"] == []
    assert result["features"] == [feature()]


def test_cross_sentence_features_are_ignored_even_if_model_drops_pause(analyzer):
    data = {"words": [{"word": "Stop", "ipa": "/stɑp/"}, {"word": "Ask", "ipa": "/æsk/"}],
            "connected": "/stɑp æsk/", "translation": "停。问。",
            "features": [feature(words=[0, 1], kind="unreleased_stop", ipa="/stɑp̚ æsk/")]}
    result = analyzer._validate_analysis(data, "Stop. Ask")
    assert result["features"] == []
    assert result["links"] == []


@pytest.mark.parametrize("word,citation,context,keep", [
    ("powers", "/ˈpaʊərz/", "/ˈpaʊɚz/", False),
    ("powers", "/ˈpaʊə˞z/", "/ˈpaʊɚz/", False),
    ("about", "/əˈbaʊt/", "/əˈbaʊt/", False),
    ("of", "/ʌv/", "/əv/", True),
    ("the", "/ðiː/", "/ði/", True),
])
def test_weak_forms_require_more_than_notation_difference(analyzer, word, citation, context, keep):
    note = feature(kind="weak_form", ipa=context)
    data = {"words": [{"word": word, "ipa": citation}], "connected": context,
            "translation": "示例", "features": [note]}
    result = analyzer._validate_analysis(data, word)
    assert result["features"] == ([note] if keep else [])
