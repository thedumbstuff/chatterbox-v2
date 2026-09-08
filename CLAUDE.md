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

- Windows 11, PowerShell. Python 3.13.1 system; **no venv in this repo yet**. RTX 4090 24 GB.
- Setup (once): `python -m venv venv; .\venv\Scripts\Activate.ps1;`
  `pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124; pip install -e .`
  Install CUDA torch **before** `pip install -e .`. If `spacy-pkuseg` fails on 3.13, use a 3.11 venv.
- Always run with `.\venv\Scripts\python.exe` once the venv exists. `pip install -e .` is
  required even to import (`__init__.py` reads the installed version).
- Weights download from Hugging Face on first `from_pretrained` (several GB). Set `HF_TOKEN`
  if a repo is gated. Cache: `C:\Users\shwet\.cache\huggingface\hub`.
- No tests exist. CI only does `pip install -e .`. Verify changes by actually generating
  audio (`python example_tts_turbo.py` is the cheapest: Turbo/Nano, English, built-in voice).

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
5. Known sharp edges to remember while coding: `ChatterboxTTS(cfg_weight=0)` crashes (G1);
   hard 1000-token = 40 s cap (G5); `temperature=0` is not greedy (G6); `audio_prompt_path`
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
