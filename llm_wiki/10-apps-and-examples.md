# 10 - Apps and examples

All scripts live at the repo root and import the installed `chatterbox` package
(`pip install -e .` first). None of them accept CLI arguments; edit the constants.

## Example scripts

| Script | Model | What it does | Notes |
|--------|-------|--------------|-------|
| `example_tts.py` | `ChatterboxTTS` + `ChatterboxMultilingualTTS` (v2 default) | auto-detect cuda/mps/cpu; English line -> `test-1.wav`; French line -> `test-2.wav`; if `YOUR_FILE.wav` exists, cloned English -> `test-3.wav` | comment mentions `t3_model="v3"` opt-in |
| `example_tts_turbo.py` | `ChatterboxTurboTTS` | hard-coded `device="cuda"`; built-in voice; text with `[chuckle]` -> `test-turbo.wav` | cloning line commented out |
| `example_tts_nano.py` | `ChatterboxTurboTTS(nano=True)` | same as Turbo -> `test-nano.wav` | says CPU works too |
| `example_vc.py` | `ChatterboxVC` | converts `YOUR_FILE.wav` into `YOUR_FILE.wav`'s voice -> `testvc.wav` | placeholders must be edited |
| `example_for_mac.py` | `ChatterboxTTS` | picks `mps` if available; **monkeypatches `torch.load`** to inject `map_location`; `exaggeration=2.0` | the patch is now redundant: loaders already pass `map_location` for cpu/mps |

Generated `.wav` files are git-ignored (`**/*.wav`).

## Our test script: `tests/smoke_generate.py` (added 2026-09-08)

```powershell
.\venv\Scripts\python.exe tests\smoke_generate.py nano|turbo|english|mtl|vc|all [--ref x.wav] [--src y.wav]
    [--lang hi] [--t3 v3] [--cfg 0.5] [--exaggeration 0.5] [--temperature 0.8] [--tag=-suffix] [--seed N]
```
Loads the variant, generates one sentence, saves `syn_out/<variant><tag>.wav`, prints duration,
timings, RTF, peak amplitude, and the Perth detector result against the expectation (0.0 by default,
1.0 with `--watermark`). Exit code 1 on failure. `tests/bench_watermark.py` A/B-times the watermark per variant.
Pass `--tag=-x` with `=` (argparse treats a leading `-` value as a flag otherwise).
Project scope decision (user, 2026-09-08): **test only English and Hindi** for multilingual.

`tests/g1_equivalence.py --label old|new` / `--compare`: calls `T3.inference` directly with
near-greedy sampling (temperature 0.05, fixed seed) and saves token ids to `syn_out/g1_*.pt`, to
prove a decoding-loop refactor is output-preserving. Reuse the pattern for future T3 changes.

## Gradio apps

All use `gradio==6.8.0` (pinned in pyproject, imported as a hard dependency of the library
even though the library itself never imports it).

### `gradio_tts_app.py` - original English
- `demo.load(load_model)` stores the model in `gr.State` per session (loads on page open;
  `generate` reloads if the state is `None`).
- Inputs: text (300-char note), reference audio (upload/mic), exaggeration slider 0.25-2,
  CFG/Pace slider **0.0**-1 (0 triggers gotcha G1), accordion: seed, temperature 0.05-5,
  min_p, top_p, repetition_penalty.
- `demo.queue(max_size=50, default_concurrency_limit=1).launch(share=True)` -> creates a
  public Gradio share link by default.

### `gradio_tts_turbo_app.py` - Turbo
- Same session-state pattern. Custom CSS and a JS snippet that inserts a clicked tag at the
  textbox caret (`EVENT_TAGS` list of 9 tags).
- Default reference audio is a remote URL (`chatterbox-demo-samples/prompts/female_random_podcast.wav`).
- Advanced: seed, temperature 0.05-2, top_p, top_k 0-1000, repetition_penalty, min_p, loudness checkbox.
- Loads Turbo only (no `nano` toggle). `launch(share=True)`.

### `gradio_vc_app.py` - voice conversion
- Loads the model at import time (module global), simple `gr.Interface` with two audio
  inputs. `demo.launch()` (no share).

### `multilingual_app.py` - Multilingual
- Module-global `MODEL`, loaded at import with `T3_MODEL = os.getenv("CHATTERBOX_MULTILINGUAL_T3_MODEL", "v2")`.
- `LANGUAGE_CONFIG`: per-language default reference clip URL (Google Cloud Storage
  `chatterbox-demo-samples/mtl_prompts/...`) and a sample sentence (all say "last month we
  reached two billion views on our YouTube channel").
- Changing the language dropdown swaps in that language's default clip and text.
- `generate_tts_audio(...)` has a long docstring because the app is launched with
  `demo.launch(mcp_server=True)`: Gradio exposes the function as an **MCP tool**, so the
  docstring is the tool description. The docstring's language list is outdated (says 7
  languages) relative to `SUPPORTED_LANGUAGES` (23).
- CFG slider minimum is 0.2 in this app (multilingual would accept 0).
- Truncates text to 300 chars; falls back to the language's default clip when none is uploaded.

## Running them (Windows, this machine)

```powershell
cd C:\Shwetank\Work\Workspace\Python\opensource\chatterbox-v2
# after creating/activating a venv and `pip install -e .` (see 11-environment-and-dependencies.md)
python example_tts_turbo.py
python gradio_tts_turbo_app.py      # opens a local URL and (share=True) a public one
```
Set `HF_TOKEN` in the environment if the Turbo/Nano/multilingual repos are gated for the account.
