# chatterbox-v2 — Claude Code instructions

Fork of Resemble AI's **Chatterbox TTS** (`resemble-ai/chatterbox`), cloned from
`thedumbstuff/chatterbox-v2` at upstream commit `5de7a54`. We will modify this repo heavily.
Pure-inference PyTorch library in `src/chatterbox/` + example scripts and Gradio apps at root.

## Knowledge base: `llm_wiki/` — read it, keep it current

- `llm_wiki/README.md` is the index. Start with `00-overview.md` and `02-pipeline-architecture.md`.
- **Before editing a module, read its wiki page** (T3 → `04`, S3Gen → `05`, tokenizers → `06`,
  conditioning → `07`). Before editing anything, skim `12-gotchas-known-issues.md` (G1–G20).
- `13-constants-and-magic-numbers.md` lists every hard-coded number and where it is repeated.
- `14-extension-points.md` has recipes (E1–E15) for the changes we expect to make.
- **After any code change that affects documented behaviour, update the relevant wiki page
  in the same change set, and append an entry to `llm_wiki/changelog.md`** (date, files,
  why, follow-ups). Mark unverified claims as **(unverified)**; remove the mark once verified by a run.

## Architecture in one breath

`text → punc_norm → tokenizer → T3 (Llama-520M or GPT-2 LLM) → 25 Hz speech tokens (vocab 6561)
→ S3Gen (conformer encoder → flow-matching mel decoder → HiFT vocoder) → 24 kHz wav → Perth watermark`.
Reference audio conditions both stages (VoiceEncoder 256-d + prompt tokens for T3;
CAMPPlus x-vector + prompt mel/tokens for S3Gen). Batch size 1 everywhere.

Entry points: `tts.py` (`ChatterboxTTS`, English), `mtl_tts.py` (`ChatterboxMultilingualTTS`,
23 langs, v2 default / v3 opt-in), `tts_turbo.py` (`ChatterboxTurboTTS`, Turbo 350M and
Nano 110M via `nano=True`, meanflow 2-step decoder, `[laugh]`-style tags), `vc.py` (`ChatterboxVC`).

## Environment (this machine)

- Windows 11, PowerShell. RTX 4090 24 GB. **`venv/` exists**: Python 3.12.9, torch 2.6.0+cu124,
  package installed editable (`uv venv venv --python 3.12` + `uv pip install`; recipe in wiki 11).
- **Always run with `.\venv\Scripts\python.exe`** (never the system Python). `pip install -e .`
  is required even to import (`__init__.py` reads the installed version).
- All weights are already in the HF cache (`C:\Users\shwet\.cache\huggingface\hub`): original,
  multilingual v2+v3, turbo, nano. No `HF_TOKEN` needed. Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to quiet the cache warning.
- **Verify changes by generating audio** with our script (outputs land in git-ignored `syn_out/`):
  `.\venv\Scripts\python.exe tests\smoke_generate.py nano|turbo|english|mtl|vc|all [--lang hi --t3 v3] [--ref x.wav] [--cfg 0.0] [--tag=-x]`
  It prints duration, RTF, peak, and watermark detection; exit 1 on failure. Baseline numbers
  (all variants pass, 2026-09-08) are in wiki 09 and `llm_wiki/changelog.md`.
- **Scope: only English and Hindi need testing** (user decision 2026-09-08). Use `mtl --lang hi --t3 v3` for Hindi.
- No pytest suite yet; CI only does `pip install -e .`.

## README one-liner change log (user rule)

The top of `README.md` has a "Fork change log" section. **Every change to this fork gets one
line there** (date + what), newest first. Details go to `llm_wiki/changelog.md`.

## Coding rules for this repo

1. **Do not rename or move `nn.Module` attributes or classes under `src/chatterbox/models/`**
   without re-checking checkpoint loading for every variant: state-dict keys must match
   `ve/t3/s3gen` files on HF. Original loads S3Gen with `strict=False` (silent), Turbo with
   `strict=True`. See `03-model-variants-and-checkpoints.md`.
2. `models/s3gen/**` and `models/s3gen/transformer/**` are vendored CosyVoice / Matcha-TTS /
   WeNet code under Apache-2.0. Keep license headers, keep diffs minimal, do not reformat.
3. The three entry files duplicate `punc_norm` and `Conditionals` with **intentional
   differences** in `punc_norm` (see wiki 06). Fix bugs in all copies or refactor via E1;
   never silently change one variant's normalisation.
4. Keep public class names and `generate()` signatures backward compatible; README, apps,
   and external users rely on them. Add parameters with defaults.
5. Known sharp edges to remember while coding: hard 1000-token = 40 s cap (G5); `temperature=0` is not greedy (G6); `audio_prompt_path`
   recomputes conditionals every call (G7); every `*.wav` is git-ignored (G9).
6. Prefer `logging` over `print`; the library currently prints and shows tqdm bars (G13).
7. Everything runs fp32 by design. Precision/speed changes go through E7 with measurements.
8. Python 3.10+ syntax is fine (`str | None` already used). Match existing style (4 spaces,
   no type-checker config, docstrings sparse).

## Git

- Remote `origin` = our fork (`thedumbstuff/chatterbox-v2`). Upstream not configured; add
  `upstream https://github.com/resemble-ai/chatterbox.git` if we need their fixes.
- Commit when asked. Add `venv/` to `.gitignore` before the first commit that follows setup.
- Commit messages: imperative summary; mention gotcha/recipe ids (G#, E#) when relevant.

## Workspace context

Parent folder `../CLAUDE.md` (workspace `opensource`) describes sibling projects (trading
sims etc.); it is unrelated to this repo except for the shared Windows environment.
