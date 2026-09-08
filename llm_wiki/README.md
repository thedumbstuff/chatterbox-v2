# Chatterbox-v2 LLM Wiki

Knowledge base for the `chatterbox-v2` repo (fork of Resemble AI's Chatterbox TTS),
written for an LLM coding assistant (and humans) who will modify this codebase heavily.
Everything here was derived by reading the code at commit `5de7a54` (2026-09-08);
where a claim is an inference rather than a verified run, it is marked **(unverified)**.

## How to use this wiki

- Start with `00-overview.md`, then `02-pipeline-architecture.md`. Those two pages
  give the mental model everything else hangs off.
- Before changing a module, read its page (T3 -> `04`, S3Gen -> `05`, tokenizers -> `06`).
- Before touching *anything*, skim `12-gotchas-known-issues.md`. Several bugs and
  sharp edges live in the public entry points.
- `13-constants-and-magic-numbers.md` is the lookup table for every hard-coded number.
- `14-extension-points.md` lists where and how to make the common changes we expect
  to make (new language, new variant, streaming, batching, fine-tuning).
- `changelog.md` is the living log of *our* modifications and decisions. Append to it.

## Maintenance rules

1. When code changes invalidate a page, fix the page in the same change set.
2. Prefer function/class names over line numbers as anchors; line numbers drift.
3. Keep facts and interpretation separate. Mark guesses as **(unverified)**.
4. One topic per page. Cross-link with relative markdown links.

## Page index

| Page | What it covers |
|------|----------------|
| [00-overview.md](00-overview.md) | What Chatterbox is, model zoo, fork provenance, license |
| [01-repo-layout.md](01-repo-layout.md) | Directory tree with one-line purpose per file, line counts |
| [02-pipeline-architecture.md](02-pipeline-architecture.md) | End-to-end data flow: text -> T3 -> speech tokens -> S3Gen -> 24 kHz wav |
| [03-model-variants-and-checkpoints.md](03-model-variants-and-checkpoints.md) | Original / Multilingual v2+v3 / Turbo / Nano: configs, HF repos, checkpoint files |
| [04-t3-text-to-speech-tokens.md](04-t3-text-to-speech-tokens.md) | T3 model internals, conditioning encoder, both inference loops |
| [05-s3gen-tokens-to-waveform.md](05-s3gen-tokens-to-waveform.md) | S3Tokenizer, flow encoder, CFM/meanflow decoder, HiFT vocoder |
| [06-text-tokenizers-and-normalization.md](06-text-tokenizers-and-normalization.md) | punc_norm variants, EnTokenizer, MTLTokenizer per-language preprocessing, GPT-2 tokenizer |
| [07-voice-conditioning.md](07-voice-conditioning.md) | VoiceEncoder, CAMPPlus x-vector, Conditionals, conds.pt, prompt lengths |
| [08-public-api.md](08-public-api.md) | Every public class/method with signatures and defaults |
| [09-generation-parameters.md](09-generation-parameters.md) | Sampling knobs, what each does, per-variant support matrix, tuning tips |
| [10-apps-and-examples.md](10-apps-and-examples.md) | Example scripts and the four Gradio apps |
| [11-environment-and-dependencies.md](11-environment-and-dependencies.md) | pyproject pins, undeclared deps, Python/Windows/GPU notes, HF download quirks |
| [12-gotchas-known-issues.md](12-gotchas-known-issues.md) | Bugs, duplicated code, dead parameters, surprising behaviour |
| [13-constants-and-magic-numbers.md](13-constants-and-magic-numbers.md) | Sample rates, token rates, vocab sizes, special token ids, lengths |
| [14-extension-points.md](14-extension-points.md) | How to make the changes we are likely to make |
| [15-glossary.md](15-glossary.md) | Terms: T3, S3, CFM, meanflow, x-vector, CFG, exaggeration, ... |
| [changelog.md](changelog.md) | Our modification log and decision record (append-only) |
