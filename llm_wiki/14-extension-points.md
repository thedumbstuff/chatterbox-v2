# 14 - Extension points: how to make the changes we are likely to make

Each recipe names the files, the minimal edit, and what can break. Update
`12-gotchas-known-issues.md` and `changelog.md` when you implement one.

## E1. Unify the three entry classes (reduce copy-paste)

- Extract `punc_norm` into `src/chatterbox/text_norm.py` with a `variant` argument
  (`"en"`, `"mtl"`, `"turbo"`) preserving the per-variant replacement lists (see 06).
- Extract `Conditionals` into `src/chatterbox/conditionals.py` (identical in all three files).
- Extract the shared `prepare_conditionals` skeleton into a base class
  (`_BaseChatterbox` with `ENC_COND_LEN`, `DEC_COND_LEN`, `sr`, `watermarker`, device
  fallback, `_load_ve`, `_load_conds`).
- Keep the public class names and `generate` signatures unchanged; the apps and README
  depend on them. Keep `from_local` layouts identical so HF snapshots still load.
- Risk: `Conditionals.load` is referenced by name in `conds.pt`? No: it is a plain dict, so
  moving the class is safe.

## E2. Fix G1 (`cfg_weight=0` in `ChatterboxTTS`) - DONE 2026-09-08

Implemented "Option B": `T3.inference` repeats a single text row when `cfg_weight > 0`,
runs one row otherwise, sizes `bos_embed`/step embeddings to `B`, and skips the CFG mix at
cfg 0. Callers no longer duplicate. Verified output-preserving with `tests/g1_equivalence.py`.

## E3. Greedy / deterministic decoding (G6)

Add `do_sample: bool` handling in both loops: `next_token = logits.argmax(-1, keepdim=True)`
when `not do_sample` (Llama loop) or `temperature == 0` (Turbo). Keep repetition penalty
applied before argmax.

## E4. Long-form text (G5) - DONE in Studio (`studio/textsplit.py` + `Engine.generate`)

Do sentence splitting in a new helper (`src/chatterbox/longform.py`): split on
`punc_norm` sentence enders (respect the multilingual set), generate per chunk with cached
conditionals (`prepare_conditionals` once), concatenate with a short silence
(e.g. 200 ms = 4800 samples at 24 kHz) or crossfade. Because each chunk starts with the
`trim_fade` (first 20 ms zeroed), concatenation is already click-free at chunk starts.
Alternatively raise `max_new_tokens` (positional table allows 4096 speech tokens = 164 s)
but the model was likely not trained on such lengths.

## E5. Streaming output

Pieces already present:
- `S3Gen.flow_inference(..., finalize=False)` drops the last `pre_lookahead_len * 2 = 6` mel
  frames so a partial token sequence can be decoded with a 3-token lookahead.
- `HiFTGenerator.inference(cache_source=...)` accepts the source-signal cache of the
  previous chunk to avoid boundary glitches; `hift_inference` exposes it.
- `S3Token2Wav.forward(..., finalize, skip_vocoder)` is the non-streaming placeholder;
  the docstring references an `S3GenStreamer` class that does **not** exist in this repo
  (it lives in Resemble's private/prod code). We would write it.
- `ConditionalCFM.forward` has a disabled `flow_cache` mechanism (prompt + 34-frame overlap
  caching from CosyVoice) that would need reviving for chunked CFM.
Plan: (1) make `T3.inference` a generator yielding tokens; (2) every N tokens (e.g. 25 = 1 s)
run `flow_inference` on `all_tokens_so_far` with `finalize=False`, vocode only the new
frames with `cache_source`, yield audio; (3) on EOS run once with `finalize=True`.
The simple version re-decodes the whole prefix each chunk (O(n^2) but fine for <= 40 s).

## E6. Batched inference

Every stage asserts or assumes batch 1: `drop_invalid_tokens`, `S3Gen.forward` ("designed
for batch_size=1"), `T3.inference` CFG rows, `embed_ref` shapes. `flow.inference` has
`_repeat_batch_dim` helpers and the CFM solver handles B > 1, so S3Gen batching is closer
than T3 batching. Realistic path: keep batch 1 per request and parallelise requests across
model replicas/processes (see G16).

## E7. Half precision / speed

- `S3Gen.dtype` follows the flow parameters and `cast_all` adapts the solver; try
  `s3gen.flow.half()` + `s3gen.mel2wav` in fp32 and measure. `S3Tokenizer` and CAMPPlus
  cast their inputs to float32 explicitly.
- T3: `LlamaModel` in bf16 (`torch.autocast("cuda", dtype=torch.bfloat16)` around the loops)
  is the usual win; the custom embeddings/heads are fp32 Linear layers and autocast handles them.
- `torch.compile` on `t3.tfmr` is plausible; the kv-cache loop rebuilds `T3HuggingfaceBackend`
  every call (`self.compiled = False`), which should be removed first.
- Reduce `n_cfm_timesteps` (10 -> 5) for the non-meanflow S3Gen as a quality/speed trade.

## E8. Add or change a language (multilingual)

1. Add the code and name to `SUPPORTED_LANGUAGES` in `mtl_tts.py` (validation list).
2. The T3 checkpoint must know the `[xx]` tag token; the tokenizer json must contain it
   (`grapheme_mtl_merged_expanded_v1.json`). A new language without retraining = zero-shot
   through shared graphemes only.
3. If the language needs preprocessing, add a branch in `MTLTokenizer.encode` (see 06 for
   the existing `zh/ja/he/ko/ru` pattern: lazy optional import, warn on failure).
4. Extend `punc_norm` sentence enders for its punctuation.
5. Add a `LANGUAGE_CONFIG` entry (default clip URL + sample text) in `multilingual_app.py`.

## E9. Load Single Language Pack checkpoints

`ChatterboxMultilingualTTS.from_local(dir, device, t3_model="<file>.safetensors")` accepts
any `.safetensors` filename, and `from_pretrained` passes the same name into
`allow_patterns` (so only files present in `ResembleAI/chatterbox` can be fetched that way).
For the pack repos, either (a) `snapshot_download` the pack repo separately and symlink/copy
its T3 file into the main snapshot dir, or (b) add a `repo_id` parameter to
`from_pretrained`. Verify that the pack uses the same tokenizer json and `T3Config.multilingual()`
before assuming compatibility **(unverified)**.

## E10. Voice caching / voice library - DONE in Studio (`studio/library.py`, `Engine.apply_voice`)

`Conditionals.save(path)` / `Conditionals.load(path, map_location)` already round-trip the
full conditioning (both T3 and S3Gen parts). Build a library as `{name: path_to_conds.pt}`
and set `model.conds = Conditionals.load(...).to(device)` instead of passing
`audio_prompt_path`. Note `emotion_adv` is stored inside; `generate` rebuilds `T3Cond` if the
requested `exaggeration` differs.

## E11. Reference clip selection / quality

Before `prepare_conditionals`: trim leading silence (`librosa.effects.trim`), pick the
loudest/cleanest 10 s window (the code only uses the first 10 s / 6 s), optionally
loudness-normalise for Llama variants too (reuse `ChatterboxTurboTTS.norm_loudness`).

## E12. Remove or make the watermark optional - DONE 2026-09-08

Implemented as opt-in: `src/chatterbox/watermark.py` (`maybe_watermark`, lazy `get_watermarker`),
`watermark: bool = False` on every class/loader and `watermark: bool | None = None` on every
`generate`; `perth` moved to the `[watermark]` optional extra. Decision and measurements are in the
README "Watermarking" section.

## E13. Serve as an API

Wrap one model instance per worker; serialise requests (G16); disable `tqdm`/prints (G13);
pre-load `conds` for known voices (E10); chunk long text (E4); return 24 kHz PCM or encode
with `torchaudio.save` to an in-memory buffer. `multilingual_app.py` already exposes an MCP
tool via Gradio (`mcp_server=True`) if an MCP interface is wanted quickly.

## E14. Fine-tuning hooks

Training-time code retained from upstream: `T3.forward`/`T3.loss` (text + speech CE),
`CausalMaskedDiffWithXvec.compute_loss` -> `ConditionalCFM.compute_loss` (CFM MSE with
random conditioning dropout `training_cfg_rate=0.2`). Missing: data pipeline, target
shifting for the LM loss, optimiser loop, HiFT discriminators. Freeze S3Gen and fine-tune T3
only for voice/style adaptation; the speech-token targets come from `S3Tokenizer`.

## E15. Testing strategy (there are no tests)

Minimum useful suite (`tests/`):
- unit: `punc_norm` variants, `MTLTokenizer.encode` per language (mock optional deps),
  `drop_invalid_tokens`, `korean_normalize`, `_resolve_multilingual_t3_model`.
- shape tests without weights: `T3(hp)` random init -> `inference` on a tiny config
  (override `LLAMA_CONFIGS` with a 2-layer config) to catch G1-style batch bugs.
- smoke (GPU, slow, opt-in): load each variant, generate 1 sentence, assert 24 kHz length
  and that `perth` detects the watermark.
