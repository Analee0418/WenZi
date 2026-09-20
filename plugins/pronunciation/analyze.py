"""LLM-based pronunciation analysis — word IPA + connected speech IPA."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# -- Prompts -----------------------------------------------------------------

_SYSTEM_PROMPT_FULL = """\
You are a pronunciation coach specializing in General American English.
Describe one natural everyday conversational reading for a learner to shadow,
at a moderate pace with neutral emphasis. Avoid exaggerated fast-speech reductions.

Given an English sentence, return ONLY valid JSON (no markdown fences, no explanation):

{
  "words": [{"word": "...", "ipa": "/.../"}, ...],
  "connected": "/.../",
  "translation": "Natural Chinese translation",
  "features": [{"words": [0], "kind": "weak_form", "ipa": "/.../", "note": "Brief Chinese explanation"}]
}

Rules:
- "words": standard citation-form IPA for each word (American English).
  Use strong forms for function words, and /t, d/ for underlying stops.
- "connected": natural connected-speech IPA for the whole sentence.
  Include common weak forms and American flapping only where natural.
  Mark primary stress with ˈ.
- Use IPA symbols only (not ARPAbet).
- Preserve the original word order and punctuation in "words".
- "translation": natural Chinese translation of the sentence.
- "features": a few useful American speech features in this reading.
  "words" contains one or two consecutive zero-based indices in the words array.
  "kind" must be weak_form, flap, or unreleased_stop.
  "ipa" gives the contextual pronunciation of just those words; "note" briefly
  explains the change in Chinese. Use [] when there is no clear change.
  Weak forms normally concern unstressed function words such as to, for, and, of.
  Do not label equivalent IPA notation (e.g. /ər/ versus /ɚ/) or accent variants
  as a weak form or other connected-speech change.
  An unstressed syllable already present in a content word's citation form is
  not itself a connected-speech weak form (e.g. the first vowel in about).
  Base explanations on neighboring SOUNDS, not spelling: environment begins
  with a vowel. Unstressed the commonly uses /ði/ before a vowel, /ðə/ before
  a consonant; /ðə/ before a vowel is also possible but is not a consonant case.
  A flap must be shown with [ɾ] in the IPA. A stop before another consonant is
  not automatically a flap. An unreleased stop (t̚, d̚, etc.) is not a deleted stop.
  Keep each feature consistent with "connected". Do not force every phenomenon.
  These are suggested readings; no recorded audio has been supplied."""

# -- JSON extraction ---------------------------------------------------------


class _InvalidAnalysisResponse(ValueError):
    """A model response that can be retried once."""


_IPA_CONSONANTS = "pbtdkgɡfvθðszʃʒhʔmnŋlrɹwjɾ"
_IPA_VOWELS = "aeiouyɑɐɒæɛəɚɜɝɞɪɨʊʉʌɔɶøœɤɯ"


def _ipa_phones(ipa: str) -> str:
    """Accept a single transcription; never interpret spelling as IPA."""
    ipa = ipa.strip()
    if len(ipa) < 3 or (ipa[0], ipa[-1]) not in (("/", "/"), ("[", "]")):
        return ""
    phones = re.sub(r"[ˈˌːˑ˞.'’]", "", ipa[1:-1])
    if any(phone not in _IPA_CONSONANTS + _IPA_VOWELS for phone in phones):
        return ""
    return phones


def _ends_phrase(word: str) -> bool:
    return word.rstrip().rstrip("\"'”’)]}").endswith((".", "!", "?", ";", ":", ","))


def _comparable_ipa(ipa: str) -> str:
    # Rhotic notation and stress marks alone do not establish a weak form.
    ipa = re.sub(r"[\s/\[\]ˈˌ'’]", "", ipa).replace("ɹ", "r")
    return ipa.replace("ər", "ɚ").replace("ə˞", "ɚ")


def _derive_links(words: list[dict[str, str]], sentence: str) -> list[dict[str, Any]]:
    """Find simple C-to-V links from the same IPA displayed by the panel."""
    source_words = sentence.split()
    # Alignment lets original punctuation block links even if the model drops it.
    def letters(word: str) -> str:
        return "".join(char for char in word.casefold() if char.isalnum())

    if len(source_words) != len(words) or any(
        letters(source) != letters(word["word"])
        for source, word in zip(source_words, words, strict=False)
    ):
        return []

    phones = [_ipa_phones(word["ipa"]) for word in words]
    links = []
    for index in range(len(words) - 1):
        left, right = phones[index], phones[index + 1]
        if any(
            _ends_phrase(word)
            for word in (source_words[index], words[index]["word"])
        ):
            continue
        if left and right and left[-1] in _IPA_CONSONANTS and right[0] in _IPA_VOWELS:
            links.append({
                "from": index,
                "to": index + 1,
                "note": "Join the words without a pause. Follow the sentence IPA below for their sounds in context.",
            })
    return links


def _validate_features(features: Any, word_count: int) -> list[dict[str, Any]]:
    """Keep optional speech notes anchored to the words actually displayed."""
    if not isinstance(features, list):
        return []
    kinds = {"weak_form", "flap", "unreleased_stop"}
    clean = []
    seen = set()
    for feature in features:
        if not isinstance(feature, dict):
            continue
        indices = feature.get("words")
        kind, ipa, note = feature.get("kind"), feature.get("ipa"), feature.get("note")
        if (
            not isinstance(indices, list)
            or not 1 <= len(indices) <= 2
            or any(type(index) is not int or not 0 <= index < word_count for index in indices)
            or (len(indices) == 2 and indices[1] != indices[0] + 1)
            or not isinstance(kind, str)
            or kind not in kinds
            or not isinstance(ipa, str)
            or len(ipa.strip()) < 3
            or (ipa.strip()[0], ipa.strip()[-1]) not in (("/", "/"), ("[", "]"))
            or not ipa.strip()[1:-1].strip()
            or not isinstance(note, str)
            or not note.strip()
        ):
            continue
        if kind == "flap" and "ɾ" not in ipa:
            continue
        if kind == "unreleased_stop" and not re.search("[ptkbdɡg]̚", ipa):
            continue
        key = (kind, tuple(indices))
        if key in seen:
            continue
        seen.add(key)
        clean.append({"words": indices, "kind": kind, "ipa": ipa.strip(), "note": note.strip()})
    return clean


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, tolerating code fences and think tags."""
    text = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text:
        raise _InvalidAnalysisResponse("AI returned an empty answer. Please try again.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _InvalidAnalysisResponse(
            "AI returned invalid JSON. Please try again."
        ) from exc
    if not isinstance(data, dict):
        raise _InvalidAnalysisResponse(
            "AI returned an unexpected answer format. Please try again."
        )
    return data


def _validate_analysis(data: dict[str, Any], sentence: str) -> dict[str, Any]:
    """Validate fields before passing model output to the pronunciation panel."""
    words = data.get("words")
    if (
        not isinstance(words, list)
        or not words
        or any(
            not isinstance(word, dict)
            or any(
                not isinstance(word.get(key), str) or not word[key].strip()
                for key in ("word", "ipa")
            )
            for word in words
        )
        or any(
            not isinstance(data.get(key), str) or not data[key].strip()
            for key in ("connected", "translation")
        )
    ):
        raise _InvalidAnalysisResponse(
            "AI returned incomplete pronunciation data. Please try again."
        )
    features = _validate_features(data.get("features"), len(words))
    source_words = sentence.split()
    # A two-word change cannot bridge a pause that the model overlooked.
    features = [
        feature for feature in features
        if len(feature["words"]) == 1 or (
            len(source_words) == len(words)
            and not _ends_phrase(source_words[feature["words"][0]])
            and not _ends_phrase(words[feature["words"][0]]["word"])
        )
    ]
    features = [
        feature for feature in features
        if feature["kind"] != "weak_form" or _comparable_ipa(feature["ipa"]) != _comparable_ipa(
            " ".join(words[index]["ipa"] for index in feature["words"])
        )
    ]
    # Missing or hallucinated model annotations must not suppress obvious links.
    return {
        **data,
        "links": _derive_links(words, sentence),
        "features": features,
    }


def _parse_response(response: dict[str, Any], sentence: str, validator=_validate_analysis) -> dict[str, Any]:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _InvalidAnalysisResponse(
            "AI returned no answer. Please try again."
        )
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    # Even parseable JSON is not trustworthy when the provider reports truncation.
    if finish_reason == "length":
        raise _InvalidAnalysisResponse(
            "AI answer was cut off at the output limit. Try a shorter sentence."
        )
    if finish_reason == "content_filter":
        raise RuntimeError("The AI provider declined this request.")
    if finish_reason not in (None, "stop"):
        raise RuntimeError("The AI provider stopped before completing the answer.")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise _InvalidAnalysisResponse("AI returned an empty answer. Please try again.")
    return validator(_extract_json(content), sentence)


# -- LLM call ----------------------------------------------------------------


async def _call_llm(
    system_prompt: str, user_content: str, config: dict[str, Any], *, validator=_validate_analysis,
) -> dict:
    ai_config = config.get("ai_enhance", {})
    provider_name = ai_config.get("default_provider", "")
    model = ai_config.get("default_model", "")
    providers = ai_config.get("providers", {})

    if not provider_name or provider_name not in providers:
        raise RuntimeError(
            "No LLM provider configured. Enable AI Enhance in Settings first."
        )

    pcfg = providers[provider_name]

    from wenzi.llm_http import ChatClient

    client = ChatClient(
        base_url=pcfg.get("base_url", ""),
        api_key=pcfg.get("api_key", ""),
    )

    extra_body = dict(pcfg.get("extra_body", {}))
    # ChatClient merges extra_body last; keep PH's output budget under its control.
    extra_body.pop("max_tokens", None)
    if urlparse(pcfg.get("base_url", "")).hostname == "api.deepseek.com":
        # PH needs a short answer; shared enhancement settings may enable thinking.
        extra_body.pop("reasoning_effort", None)
        extra_body.update(
            thinking={"type": "disabled"},
            response_format={"type": "json_object"},
        )

    for attempt, max_tokens in enumerate((2048, 4096), start=1):
        response = await client.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        try:
            return _parse_response(response, user_content, validator)
        except _InvalidAnalysisResponse as exc:
            # Log diagnostics without recording the user's sentence or model answer.
            logger.warning(
                "Pronunciation response rejected (model=%s, attempt=%d/2): %s",
                model, attempt, exc,
            )
            if attempt == 2:
                raise


# -- Public API --------------------------------------------------------------


async def analyze_pronunciation(
    sentence: str,
    config: dict[str, Any] | None = None,
) -> dict:
    """Call the user's configured LLM to get IPA + connected speech.

    Returns ``{"words": [{"word": str, "ipa": str}, ...], "connected": str}``.
    Nothing is cached to disk.
    """
    if config is None:
        from wenzi.config import load_config

        config, _ = load_config()

    return await _call_llm(_SYSTEM_PROMPT_FULL, sentence, config)


# -- Local IPA (no LLM) -----------------------------------------------------


def analyze_pronunciation_local(sentence: str) -> dict:
    """Get word-level IPA using eng_to_ipa (no LLM, no connected speech).

    Returns ``{"words": [{"word": str, "ipa": str}, ...]}``.
    """
    try:
        import eng_to_ipa
    except ImportError:
        raise RuntimeError(
            "eng-to-ipa is not installed. Run: uv pip install eng-to-ipa"
        )

    words = sentence.split()
    result_words = []
    for w in words:
        clean = w.lower().strip(".,!?;:\"'()[]")
        if not clean:
            continue
        ipa = eng_to_ipa.convert(clean)
        # eng_to_ipa appends * for unknown words
        if ipa.endswith("*"):
            ipa = ipa[:-1] if len(ipa) > 1 else clean
        result_words.append({"word": w, "ipa": f"/{ipa}/"})

    return {"words": result_words}
