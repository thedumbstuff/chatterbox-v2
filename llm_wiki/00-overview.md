# 00 - Overview

## What this repo is

`chatterbox-v2` is a clone of **`thedumbstuff/chatterbox-v2`**, which is itself a fork of
**`resemble-ai/chatterbox`** (Resemble AI's open-source TTS family). The git history is the
upstream history (last commit `5de7a54 Update ChatterboxNano (#542)`); there are no
fork-specific commits yet. Branch: `master`. PyPI name: `chatterbox-tts`, version `0.1.7`.

It is a **pure-inference PyTorch library** (`src/chatterbox/`) plus example scripts and
Gradio demo apps at the repo root. There is no training script, no test suite, and the
only CI is a `pip install -e .` smoke test on Python 3.10.

License: MIT for Resemble's code; large parts of `models/s3gen/` are Apache-2.0 code
vendored from CosyVoice / Matcha-TTS / WeNet / ESPnet (headers preserved in files).

## The model family (what the code can load)

| Variant | Python class | T3 backbone | Params | Languages | Special features |
|---------|--------------|-------------|--------|-----------|------------------|
| Chatterbox (original) | `chatterbox.tts.ChatterboxTTS` | Llama 520M | ~500M | English | `exaggeration` + `cfg_weight` controls |
| Chatterbox Multilingual v2 / **v3** | `chatterbox.mtl_tts.ChatterboxMultilingualTTS` | Llama 520M | ~500M | 23 | `language_id`; v3 opt-in via `t3_model="v3"` |
| Single Language Pack (6 finetunes) | same class, `from_local` with a custom `t3_model` filename **(unverified)** | Llama 520M | 500M each | zh-cmn, es-mx-latam, pt-br, es-es, pt-pt, hi | separate HF repos, not wired into code |
| Chatterbox-Turbo | `chatterbox.tts_turbo.ChatterboxTurboTTS` | GPT-2 medium | 350M | English | paralinguistic tags `[laugh]` etc., 1-step distilled mel decoder (meanflow), loudness norm |
| Chatterbox-Nano | same class, `nano=True` | GPT-2 small | 110M | English | Turbo architecture, CPU-friendly (README: 3x realtime on 8 cores) |
| Voice conversion | `chatterbox.vc.ChatterboxVC` | (no T3) | S3Gen only | any | speech -> tokens -> speech in a target voice |

All variants share the same two-stage design: an autoregressive LLM (**T3**) emits discrete
25 Hz speech tokens, and a flow-matching decoder + HiFT vocoder (**S3Gen**) turns those
tokens into 24 kHz audio. See [02-pipeline-architecture.md](02-pipeline-architecture.md).

## Things that are always true

- Output sample rate is **24 000 Hz** (`S3GEN_SR`), mono, returned as a `(1, N)` float tensor.
- Every generated waveform is passed through Resemble's **Perth implicit watermarker**
  (`perth.PerthImplicitWatermarker().apply_watermark`). This is unconditional in all four
  entry points.
- Inference is **batch size 1** end-to-end (asserted in several places).
- Voice cloning is zero-shot from a reference clip; each HF repo also ships a `conds.pt`
  "built-in voice" used when no `audio_prompt_path` is given.
- Weights are downloaded from Hugging Face at first use (`from_pretrained`), cached in the
  standard HF cache. `from_local(ckpt_dir, device)` loads from a folder instead.

## Where to look next

- Directory map: [01-repo-layout.md](01-repo-layout.md)
- Which files each variant loads: [03-model-variants-and-checkpoints.md](03-model-variants-and-checkpoints.md)
- Known bugs before you edit anything: [12-gotchas-known-issues.md](12-gotchas-known-issues.md)
