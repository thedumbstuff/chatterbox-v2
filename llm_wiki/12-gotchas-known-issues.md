# 12 - Gotchas and known issues

Numbered so other pages and commit messages can reference them (G1, G2, ...). Status column:
**open** = still in the code, **fixed** = we changed it (see `changelog.md`). Items marked
**(unverified)** are inferred from reading, not from running.

| # | Severity | Status | Summary |
|---|----------|--------|---------|
| G1 | bug | open (**verified 2026-09-08**) | `ChatterboxTTS.generate(cfg_weight=0)` crashes with a batch-size mismatch |
| G2 | quirk | open | Llama inference feeds the BOS speech token twice |
| G3 | maintenance | open | `punc_norm` and `Conditionals` are copy-pasted three times and have drifted |
| G4 | API | open | Many accepted parameters are silently ignored |
| G5 | limit | open | Hard 1000-token (40 s) generation cap; long text is cut without warning |
| G6 | bug | open | `temperature=0` is not greedy; Llama loop divides by zero |
| G7 | perf | open | Passing `audio_prompt_path` recomputes all conditionals on every call |
| G8 | packaging | open | `import chatterbox` requires the package to be pip-installed |
| G9 | packaging | open | `.gitignore` ignores every `*.wav` in the repo |
| G10 | loading | open | State-dict strictness differs per variant; deleting "dead" modules breaks loading |
| G11 | default | open | Multilingual defaults to v2 although README recommends v3 |
| G12 | limit | open | Turbo tokenizer truncates at GPT-2's 1024-token `model_max_length` silently |
| G13 | noise | open | `print`/`tqdm` chatter on every generation |
| G14 | perf | open | `embed_ref` and `prepare_conditionals` run without `inference_mode` |
| G15 | deprecation | open (warning seen 2026-09-08) | Perceiver uses deprecated `torch.backends.cuda.sdp_kernel` |
| G16 | thread-safety | open | Models mutate shared state; not safe for concurrent requests |
| G17 | warning | open | "Reference mel length is not equal to 2 * reference token length" logs on most clips |
| G18 | deps | open | Heavy mandatory deps (`gradio`, `diffusers`, `conformer`) for tiny usage |
| G19 | quirk | open | `MTLTokenizer` re-downloads `Cangjie5_TC.json` into a nested cache inside the checkpoint dir |
| G20 | quirk | open | Out-of-range speech tokens are logged, not raised, in `flow.inference` |

## Details

### G1 - `cfg_weight=0` crashes in `ChatterboxTTS` **(verified 2026-09-08 with `tests/smoke_generate.py english --cfg 0.0`)**
Observed: `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 1 but got size 2 for tensor number 1 in the list.` at `t3.py` line 313 (`inputs_embeds = torch.cat([embeds, bos_embed], dim=1)`). Multilingual with `cfg_weight=0` generated fine.
`tts.py` only duplicates the text tokens when `cfg_weight > 0`. `T3.inference` then builds
`embeds` with batch 1 but unconditionally does `bos_embed = torch.cat([bos_embed, bos_embed])`
(batch 2) and `torch.cat([embeds, bos_embed], dim=1)` -> `RuntimeError: Sizes of tensors must
match except in dimension 1`. `mtl_tts.py` always duplicates, so multilingual `cfg_weight=0`
works (and is the README-recommended setting for cross-language prompts). The
`gradio_tts_app.py` slider allows 0.0. Fix options: always duplicate in `tts.py` (matches
multilingual; costs a 2x batch even when unused), or make `T3.inference` batch-aware.

### G2 - double BOS
`T3.inference` puts a BOS speech token at the end of `embeds` (via `initial_speech_tokens`)
and appends another `bos_embed` (both with position 0). The models were presumably trained
/ evaluated with this exact code path upstream, so "fixing" it may change output quality.
Treat as intentional unless an A/B says otherwise.

### G3 - triplicated helpers
`punc_norm` differs across the three files (see 06). `Conditionals` is byte-identical in all
three. Any fix to one must be applied to all, or the helpers must be moved to a shared
module while preserving per-variant normalisation.

### G4 - ignored parameters
- `T3.inference`: `num_return_sequences`, `stop_on_eos`, `do_sample`, `length_penalty`,
  `prepend_prompt_speech_tokens` (asserted None), `initial_speech_tokens` (only its shape matters).
- `ChatterboxTurboTTS.generate`: `cfg_weight`, `exaggeration`, `min_p` (warning only).
- `S3Gen.inference(drop_invalid_tokens=...)`: body commented out.
- `S3Gen.embed_ref(ref_fade_out=...)`: unused.
- `S3Tokenizer.forward(accelerator=...)`: only unwraps; fine.
- `ChatterboxMultilingualTTS.from_pretrained(device: torch.device)`: annotated as
  `torch.device` but compared to the string `"mps"`.

### G5 - 40 s cap
`max_new_tokens=1000` in `tts.py`/`mtl_tts.py`, `max_gen_len=1000` in `inference_turbo`.
If no EOS is produced in 1000 steps the loop just ends and the audio stops mid-sentence.
There is also no minimum-length or EOS-suppression logic, so very short texts can
occasionally produce near-empty output. Sentence-chunk long inputs at the caller.

### G6 - temperature 0
Llama loop: `logits / temperature` -> inf. Turbo loop: warper skipped, still samples.
Implement greedy explicitly (argmax) if wanted.

### G7 - conditionals recomputed per call
`generate(audio_prompt_path=...)` always calls `prepare_conditionals`. Cache by calling it
once, or add an in-memory `{path: Conditionals}` cache keyed by path + mtime + exaggeration.

### G8 - `__version__` via `importlib.metadata`
`src/chatterbox/__init__.py` does `version("chatterbox-tts")` at import. Running from a
checkout without `pip install -e .` raises `PackageNotFoundError`. Use `pip install -e .`
(editable) so code edits are live.

### G9 - wav files ignored
`**/*.wav` in `.gitignore`. Test fixtures need `git add -f`, another format, or a rule change.

### G10 - strictness
Original/Multilingual/VC load `s3gen` with `strict=False`, so unexpected or missing keys go
unnoticed (a typo in a module name silently leaves random init weights). Turbo/Nano load
`s3gen_meanflow` with `strict=True`; T3 is always strict. Turbo also deletes `tfmr.wte`
after loading. When refactoring `models/`, run a load of every variant and compare
`missing_keys`/`unexpected_keys` explicitly.

### G11 - v2 default
`DEFAULT_MULTILINGUAL_T3_MODEL = "t3_mtl23ls_v2.safetensors"`. Pass `t3_model="v3"` or set
`CHATTERBOX_MULTILINGUAL_T3_MODEL=v3` for the app.

### G12 - Turbo text truncation
`self.tokenizer(text, padding=True, truncation=True)` with no `max_length`. GPT-2 tokenizers
default `model_max_length=1024`. Long texts are cut silently (and would hit G5 first anyway).

### G13 - console noise
`tqdm` bars in `T3.inference` ("Sampling"), `inference_turbo`, and `basic_euler`; a
`print("S3 Token -> Mel Inference...")`; `print` warnings for MPS, loudness errors, tokenizer
length, "WARNING: s3gen received ref longer than 10s". Replace with `logging` when we
serve this from a process.

### G14 - no `inference_mode` around conditioning
`S3Gen.embed_ref` (CAMPPlus forward, tokenizer quantize) and `prepare_conditionals` run with
autograd enabled in `ChatterboxTTS`/`Multilingual` (only `generate`'s T3/S3Gen part is
inside `torch.inference_mode()`), and in Turbo the top-level `generate` is not wrapped at
all. Wrap `prepare_conditionals` in `torch.inference_mode()`.

### G15 - deprecated SDPA context manager
`perceiver.py::AttentionQKV.flash_attention` uses `torch.backends.cuda.sdp_kernel(**config)`
(deprecated; replacement `torch.nn.attention.sdpa_kernel`). Works on torch 2.6 with a
warning; will break on a future torch.

### G16 - shared mutable state
`self.conds` is replaced per call; `T3.prepare_conditioning` mutates `T3Cond` in place;
`S3Gen.forward` casts `ref_dict` values in place; `mel.py` keeps module-level caches.
One model instance == one request at a time (the Gradio apps use concurrency 1).

### G17 - benign mel/token length warning
Emitted from `embed_ref` whenever the 10-s (or shorter) reference is not an exact multiple
of 40 ms. (Not seen in our 2026-09-08 tests because the references were Chatterbox outputs,
which are token-aligned; expect it with real recordings.) The code truncates tokens to match. Safe to ignore or to pre-pad the clip to a
multiple of 960 samples at 24 kHz.

### G18 - heavy dependencies
`gradio==6.8.0` is required by the library metadata although only the apps use it;
`diffusers==0.29.0` for a handful of attention/FFN helpers; `conformer` for a dead class.
Slimming requires vendoring those helpers (and keeping state-dict key names identical).

### G19 - Cangjie download path
`ChineseCangjieConverter._load_cangjie_mapping` calls `hf_hub_download(cache_dir=model_dir)`
where `model_dir` is the checkpoint directory, even though `from_pretrained` already
downloaded `Cangjie5_TC.json` into the same snapshot. Result: a second copy under
`<ckpt_dir>/models--ResembleAI--chatterbox/...`. Harmless; fix by reading the local file first.

### G20 - out-of-range tokens
`flow.inference` logs `logger.error` if any token >= 6561 but continues; the following
`nn.Embedding(6561, ...)` lookup would then raise an IndexError on CUDA/CPU. Callers must
filter (they do: `drop_invalid_tokens` + `< 6561`). Turbo appends `S3GEN_SIL=4299` which is in range.
