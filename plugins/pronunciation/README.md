# Pronunciation

Use `ph <English sentence>` in the launcher for word IPA, natural everyday
American speech suggestions, and male/female playback. Use `ph` without a
sentence to open the dialogue reader.

## Requirements

- Configure an AI Enhance provider and model in WenZi for **Analyze with AI**.
- Install `eng-to-ipa` in WenZi's Python environment for immediate local word IPA.
- Playback uses the existing executable `~/.local/bin/edge-tts-stream.py`.
  It must accept a UTF-8 text file path, an Edge voice name, and a rate argument,
  and write MP3 bytes to stdout. This personal helper is not bundled here.
- Dialogue playback additionally requires `ffmpeg` to combine utterances.

## Connected speech

The model suggests sentence IPA, Chinese explanations, weak forms, flaps, and
unreleased stops. Local validation filters malformed or contradictory hints.
Consonant-to-vowel curves come from adjacent word IPA, with punctuation boundaries
respected; they do not depend on model-provided link annotations.

Curves illustrate suggested connections, not measured audio timing. The panel
respects reduced motion; **Replay links** explicitly plays one brief demonstration
when that preference is enabled. Reload WenZi scripts and reopen PH after updates.

## Development

The manifest's `source` points this fork-specific plugin at its own repository.
The registry generator preserves that override for installation links.

Run the analysis regressions without network access or user configuration:

```sh
uv run pytest tests/scripting/test_pronunciation_analysis.py -q
```

Fixtures contain recorded AI responses for public example sentences. They test
validation and link derivation; they are not a phonetic accuracy benchmark.
