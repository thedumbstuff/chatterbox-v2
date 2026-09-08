# 03 - Model variants and checkpoints

## Hugging Face repos and the files each loader pulls

| Variant | HF repo (`REPO_ID`) | Files loaded | Download call |
|---------|---------------------|--------------|---------------|
| Original English (`ChatterboxTTS`) | `ResembleAI/chatterbox` | `ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`, `tokenizer.json`, `conds.pt` | `hf_hub_download` per file; `ckpt_dir` = parent of the last file |
| Multilingual (`ChatterboxMultilingualTTS`) | `ResembleAI/chatterbox` (same repo!) | `ve.pt`, `t3_mtl23ls_v2.safetensors` **or** `t3_mtl23ls_v3.safetensors`, `s3gen.pt`, `grapheme_mtl_merged_expanded_v1.json`, `conds.pt`, `Cangjie5_TC.json` | `snapshot_download(allow_patterns=[...], revision="main", token=HF_TOKEN)` |
| Turbo (`ChatterboxTurboTTS`) | `ResembleAI/chatterbox-turbo` | `ve.safetensors`, `t3_turbo_v1.safetensors`, `s3gen_meanflow.safetensors`, HF tokenizer files (`*.json`, `*.txt`, `*.model`), `conds.pt` | `snapshot_download(allow_patterns=["*.safetensors","*.json","*.txt","*.pt","*.model"], token=HF_TOKEN)` with Xet fallback |
| Nano (`ChatterboxTurboTTS(nano=True)`) | `ResembleAI/chatterbox-nano` | same as Turbo but `t3_nano_v1.safetensors` | same |
| Voice conversion (`ChatterboxVC`) | `ResembleAI/chatterbox` | `s3gen.safetensors`, `conds.pt` (uses only `['gen']`) | `hf_hub_download` |
| Single Language Pack | `ResembleAI/Chatterbox-Multilingual-{zh-cmn, es-mx-latam, pt-br, es-es, pt-pt, hi}` | not wired in. Presumably a T3 `.safetensors` per repo usable via `from_local(dir, device, t3_model="<file>.safetensors")` **(unverified)** | manual |

Notes:
- Original and Multilingual share **one** HF repo but load **different file formats** for
  the same modules (`ve.safetensors` vs `ve.pt`, `s3gen.safetensors` vs `s3gen.pt`).
  Multilingual loads `.pt` with `torch.load(..., weights_only=True)`.
- `conds.pt` is the built-in default voice: a `torch.save`d dict `{"t3": T3Cond.__dict__, "gen": ref_dict}`.
  Loaded with `Conditionals.load(fpath, map_location)`; `map_location="cpu"` is forced for
  `device in ["cpu", "mps"]` because the file was saved from CUDA tensors.
- `HF_TOKEN` env var is passed for Multilingual and Turbo/Nano downloads (repos may be gated).
- The Turbo loader catches Xet-backend download errors ("xet" / "hex hash" in the message)
  and retries after flipping `huggingface_hub.constants.HF_HUB_DISABLE_XET = True` in-process.

## T3 configuration per variant

`T3Config` defaults (in `models/t3/modules/t3_config.py`) describe the **original English** model.
Loaders mutate a `T3Config` for the other variants.

| Field | Original | Multilingual | Turbo | Nano |
|-------|----------|--------------|-------|------|
| `text_tokens_dict_size` | 704 | 2454 | 50276 | 50276 |
| `llama_config_name` | `Llama_520M` | `Llama_520M` | `GPT2_medium` | `GPT2_small` |
| backbone | `transformers.LlamaModel` | same | `transformers.GPT2Model` | same |
| hidden size | 1024 | 1024 | 1024 | 768 |
| layers / heads | 30 / 16 | 30 / 16 | 24 / 16 | 12 / 12 |
| `speech_tokens_dict_size` | 8194 | 8194 | 6563 | 6563 |
| `start_speech_token` / `stop_speech_token` | 6561 / 6562 | same | same | same |
| `start_text_token` / `stop_text_token` | 255 / 0 | 255 / 0 | (unused: GPT-2 tokenizer, no SOT/EOT padding) | same |
| `input_pos_emb` | `"learned"` (own `LearnedPositionEmbeddings` for text and speech) | same | `None` (GPT-2's internal absolute `wpe`) | same |
| `speech_cond_prompt_len` | 150 tokens (6 s) | 150 | 375 (15 s) | 375 |
| `use_perceiver_resampler` | True (32 queries) | True | False | False |
| `emotion_adv` | True | True | False | False |
| `speech_head` bias | False | False | True (`bias=self.is_gpt`) | True |
| `is_multilingual` property | False | True (== dict size 2454) | False | False |
| max text / speech tokens | 2048 / 4096 | same | same (unused) | same |

How the loaders build them:
- `ChatterboxTTS.from_local`: `T3()` -> `T3Config.english_only()`.
- `ChatterboxMultilingualTTS.from_local`: `T3(T3Config.multilingual())`.
- `ChatterboxTurboTTS.from_local`:
  ```python
  hp = T3Config(text_tokens_dict_size=50276)
  hp.llama_config_name = "GPT2_small" if nano else "GPT2_medium"
  hp.speech_tokens_dict_size = 6563
  hp.input_pos_emb = None
  hp.speech_cond_prompt_len = 375
  hp.use_perceiver_resampler = False
  hp.emotion_adv = False
  t3 = T3(hp); t3.load_state_dict(...); del t3.tfmr.wte   # GPT-2's own token table removed after load
  ```
  `del t3.tfmr.wte` happens **after** `load_state_dict`, so the checkpoint contains `wte`;
  re-saving the model after this deletion would produce a checkpoint that no longer loads
  with the current loader. Keep that in mind if we ever export weights.

## T3 checkpoint format

All T3 checkpoints are safetensors. Loaders accept either a flat state dict or a wrapped
one: `if "model" in t3_state.keys(): t3_state = t3_state["model"][0]`.

## S3Gen configuration per variant

| | Original / Multilingual / VC | Turbo / Nano |
|---|---|---|
| constructor | `S3Gen()` (`meanflow=False`) | `S3Gen(meanflow=True)` |
| checkpoint | `s3gen.safetensors` (or `s3gen.pt`), loaded `strict=False` | `s3gen_meanflow.safetensors`, `strict=True` |
| CFM steps | 10 (`flow_inference` default) | 2 (`tts_turbo.py` passes `n_cfm_timesteps=2`; `flow_inference` default would also be 2) |
| solver | `solve_euler` with CFG 0.7, cosine t-schedule | `basic_euler`, no CFG, linear t-schedule, initial noise passed via `noised_mels` |
| estimator extra | none | `time_embed_mixer` (Linear 2*1024 -> 1024, diagonal init) mixing t and r embeddings |

Why `strict=False` for the original: `S3Tokenizer` registers `_mel_filters` and `window`
buffers that are not in the checkpoint (`ignore_state_dict_missing` hints at this). The
meanflow checkpoint evidently contains everything needed for strict loading. If we add
parameters/buffers to S3Gen, the Turbo loader's `strict=True` will break first.

## Voice encoder

Same architecture for all variants (`VoiceEncoder()` with `VoiceEncConfig` defaults).
Weights: `ve.safetensors` (original, turbo, nano) or `ve.pt` (multilingual).

## Text tokenizers

| Variant | Tokenizer | Vocab file | Vocab size |
|---------|-----------|------------|-----------|
| Original | `EnTokenizer` (HF `tokenizers` json) | `tokenizer.json` | 704 |
| Multilingual | `MTLTokenizer` | `grapheme_mtl_merged_expanded_v1.json` (+ `Cangjie5_TC.json` for zh) | 2454 |
| Turbo / Nano | `transformers.AutoTokenizer.from_pretrained(ckpt_dir)` (GPT-2 BPE + extras) | repo's tokenizer files | 50276 (loader warns if `len(tokenizer) != 50276`); `pad_token` set to `eos_token` if missing |

50276 = 50257 (GPT-2) + 19 added tokens, presumably the paralinguistic tags and control
tokens **(unverified; inspect `added_tokens.json` in the HF repo)**.

## Multilingual v2 vs v3 selection

`_resolve_multilingual_t3_model(t3_model)` in `mtl_tts.py`:
- `None` -> `DEFAULT_MULTILINGUAL_T3_MODEL` = `t3_mtl23ls_v2.safetensors` (**v2 is still the default**, README recommends v3).
- `"v2"`, `"t3_mtl23ls_v2"`, `"v3"`, `"t3_mtl23ls_v3"` -> mapped filenames.
- any string ending in `.safetensors` -> used verbatim (this is the hook for Single Language Pack files).
- anything else -> `ValueError`.
`multilingual_app.py` reads `CHATTERBOX_MULTILINGUAL_T3_MODEL` env var (default `v2`).
Both v2 and v3 use the same `T3Config.multilingual()` and the same tokenizer file.

## Device handling (all loaders)

- `device == "mps"` but MPS unavailable -> prints a message and falls back to `"cpu"`.
- `map_location = torch.device("cpu")` for `cpu`/`mps`, else `None` (so CUDA tensors in
  `conds.pt` load straight to GPU when device is `cuda`).
- Modules are moved with `.to(device).eval()`. There is no `.half()` / autocast anywhere;
  everything runs in fp32. `S3Gen.dtype` follows the flow parameters, and `cast_all` in the
  CFM solver casts to the estimator dtype, so half precision is mechanically possible for
  S3Gen but untested here.
