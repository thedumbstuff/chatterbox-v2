# 11 - Environment and dependencies

## This machine (checked 2026-09-08)

- Windows 11 Home 10.0.26200, Git Bash + PowerShell 5.1.
- System Pythons: 3.13.1 and **3.12.9** (`C:\Users\shwet\AppData\Local\Programs\Python\Python312\`). `uv` 0.8.4 available.
- **`chatterbox-v2/venv` exists (created 2026-09-08 with `uv venv venv --python 3.12`)**: Python 3.12.9,
  torch 2.6.0+cu124, numpy 1.26.4, transformers 5.2.0, diffusers 0.29.0, package installed editable.
  Run everything with `.\venv\Scripts\python.exe`. All pinned deps (incl. `spacy-pkuseg`, `pykakasi`,
  `resemble-perth` from git) installed cleanly on 3.12; 3.13 was not tried.
- HF cache now holds `ResembleAI/chatterbox` (original + multilingual v2 + v3 files), `chatterbox-turbo`, `chatterbox-nano`.
  No `HF_TOKEN` was needed for any of them (unauthenticated warning only). Windows without Developer
  Mode cannot symlink, so the HF cache stores duplicate copies (set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence).
- GPU: **NVIDIA GeForce RTX 4090, 24 GB**, driver 591.86. All variants fit easily in fp32.
- Sibling projects keep their venv in-folder (`../cskr_daytrade/venv`); same here. `venv/` and `.venv/` are git-ignored since commit `d824743`.

## `pyproject.toml` dependency pins (version 0.1.7)

| Package | Pin | Why it matters |
|---------|-----|----------------|
| `numpy` | `<2` on Python <3.13, `>=2` on 3.13+ | 3.13 has no numpy 1.x wheels |
| `torch`, `torchaudio` | `==2.6.0` on <3.14, `>=2.9.0` on 3.14 | torch 2.6 changed `torch.load` default to `weights_only=True`; the code already passes it explicitly where needed. CUDA wheels need the `--index-url https://download.pytorch.org/whl/cu124` (or cu126) index; plain PyPI `torch==2.6.0` on Windows is CPU-only |
| `transformers` | `==5.2.0` | `LlamaModel`, `GPT2Model`, logits processors, `DynamicCache` semantics used by the kv loops |
| `diffusers` | `==0.29.0` | only `matcha/transformer.py` (`Attention`, `FeedForward` pieces, `LoRACompatibleLinear`, `maybe_allow_in_graph`) and `get_activation` in `matcha/decoder.py`. Heavy dependency for a few classes |
| `librosa` | `==0.11.0` | audio load/resample/trim, mel filters |
| `s3tokenizer` | unpinned | provides `S3TokenizerV2`, `ModelConfig`, `utils.padding` |
| `resemble-perth` | **optional extra `[watermark]`** (git `resemble-ai/Perth@master`) | watermarker, off by default since 2026-09-08; needs git available to pip |
| `conformer` | `==0.3.2` | `ConformerBlock` used only by dead `matcha/decoder.py::ConformerWrapper` (still imported at module load) |
| `safetensors` | `==0.5.3` | checkpoint loading |
| `spacy-pkuseg` | unpinned | Chinese word segmentation (multilingual `zh`); has compiled wheels, may lack a 3.13 wheel **(unverified)** |
| `pykakasi` | `==2.3.0` | Japanese kanji -> hiragana |
| `gradio` | `==6.8.0` | demo apps only, but declared as a core dependency |
| `pyloudnorm` | unpinned | Turbo loudness normalisation |
| `omegaconf` | unpinned | `DictConfig` default arg in `flow.py` only |

### Used but **not declared** (arrive transitively; pin them if we ever slim the deps)

`einops` (perceiver, decoder), `tqdm` (progress bars in both T3 loops and CFM), `scipy`
(`get_window`, `signal.lfilter`), `tokenizers` (via transformers), `huggingface_hub`
(via transformers/diffusers), `torchaudio.compliance.kaldi`.

### Optional, imported lazily inside try/except

`dicta_onnx` (Hebrew diacritics), `russian_text_stresser` (Russian stress marks). Absent ->
warning and unprocessed text.

## Python version notes

- Upstream developed/tested on Python 3.11, Debian 11. CI installs on 3.10.
- `requires-python >= 3.10`; PR #495 broadened ranges for 3.13+. `str | None` syntax in
  `mtl_tts.py` needs 3.10+.
- On this machine's 3.13.1 the torch 2.6.0 pin applies (3.13 wheels exist). Risky packages
  for 3.13 wheels on Windows: `spacy-pkuseg`, possibly `pykakasi` deps. If install fails,
  create the venv with Python 3.11 instead (install via `winget install Python.Python.3.11`
  or `uv python install 3.11`).

## Setup as performed on 2026-09-08 (repeat only if the venv is lost)

```powershell
cd C:\Shwetank\Work\Workspace\Python\opensource\chatterbox-v2
uv venv venv --python 3.12
uv pip install --python .\venv\Scripts\python.exe torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\venv\Scripts\python.exe -e .
.\venv\Scripts\python.exe -c "import chatterbox, torch; print(chatterbox.__version__, torch.cuda.is_available())"
.\venv\Scripts\python.exe tests\smoke_generate.py nano     # first real check; downloads weights
```
Install torch **before** `pip install -e .` so the CUDA build satisfies the pin instead of
the CPU wheel. Verify with `torch.cuda.is_available()`.

## Hugging Face download behaviour

- Cache: default `~/.cache/huggingface/hub` (`C:\Users\shwet\.cache\huggingface\hub`). Override with `HF_HOME`.
- `HF_TOKEN` env var is read by Multilingual and Turbo/Nano loaders; the original
  `ChatterboxTTS`/`ChatterboxVC` use `hf_hub_download` without a token argument (falls back
  to the logged-in token from `huggingface-cli login` if any).
- Xet backend crash workaround built into `ChatterboxTurboTTS.from_pretrained` (retries with
  `HF_HUB_DISABLE_XET=True` set on the constants module). Setting the env var before the
  process starts also works for the other loaders.
- Approximate sizes **(unverified)**: T3 Llama 520M fp32 ~2 GB, S3Gen ~ hundreds of MB,
  VE small; Turbo/Nano smaller. First run downloads several GB.
- `S3Tokenizer.__init__` calls `S3TokenizerV2.__init__(name)`; whether the package downloads
  its own checkpoint at construction time (then overwritten by `s3gen` weights) is
  **(unverified)**; if a surprise download to a different cache appears, this is why.

## Device support

- `cuda`: primary. No fp16/bf16 path is implemented; fp32 throughout (24 GB is plenty).
- `cpu`: works for all variants; Nano is the one designed for it.
- `mps`: guarded fallback to cpu if unavailable; `map_location="cpu"` used for `.pt` files.
  Note: `torch.backends.cuda.sdp_kernel` in the Perceiver is CUDA-specific context manager
  but is a no-op on other devices.

## Git hygiene specific to this repo

- `.gitignore` ignores every `*.wav` anywhere. Reference clips or test fixtures in `.wav`
  must be force-added (`git add -f`) or stored as `.flac`/`.mp3`, or the ignore rule narrowed.
- `checkpoints/`, `syn_out/`, `.gradio/` are ignored: use them for local weights and outputs.
- Upstream remote is not configured; only `origin` (thedumbstuff fork). To pull upstream
  fixes later: `git remote add upstream https://github.com/resemble-ai/chatterbox.git`.
