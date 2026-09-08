# 05 - S3Gen: speech tokens -> mel -> waveform

Files: `src/chatterbox/models/s3gen/*` (vendored/modified CosyVoice) and
`src/chatterbox/models/s3tokenizer/*`. Exported as `from .models.s3gen import S3GEN_SR, S3Gen`
where `S3Gen` is an alias of `S3Token2Wav`.

## Class hierarchy

```
S3Token2Mel (nn.Module)                       # s3gen.py
  ├── tokenizer:        S3Tokenizer("speech_tokenizer_v2_25hz")     # audio -> 25 Hz tokens (also used by callers for T3 prompt)
  ├── mel_extractor:    utils.mel.mel_spectrogram (function)        # 24 kHz -> 80-mel @ 50 fps
  ├── speaker_encoder:  CAMPPlus(memory_efficient=False)            # 16 kHz -> 192-d x-vector
  └── flow:             CausalMaskedDiffWithXvec                     # flow.py
        ├── input_embedding:      Embedding(6561, 512)
        ├── spk_embed_affine_layer: Linear(192, 80)
        ├── encoder:              UpsampleConformerEncoder(512, 8 heads, 2048 ffn, 6 blocks + 4 up-blocks)
        ├── encoder_proj:         Linear(512, 80)
        └── decoder:              CausalConditionalCFM(spk_emb_dim=80, CFM_PARAMS, estimator=ConditionalDecoder(...))
S3Token2Wav (S3Token2Mel)
  ├── mel2wav:  HiFTGenerator(24000, upsample [8,5,3], kernels [16,11,7], source resblocks [7,7,11], f0_predictor=ConvRNNF0Predictor())
  └── trim_fade buffer (non-persistent): 960 samples = 480 zeros + 480-sample half-cosine ramp
```

`S3Token2Mel.device` = device of the tokenizer params; `.dtype` = dtype of flow params.

## S3Tokenizer (`s3tokenizer/s3tokenizer.py`)

Subclass of `s3tokenizer.model_v2.S3TokenizerV2` (pip package `s3tokenizer`) named
`speech_tokenizer_v2_25hz`. Overrides:
- registers `_mel_filters` (librosa mel, 16 kHz, n_fft 400, `config.n_mels`=128) and a
  Hann `window` as buffers, so `log_mel_spectrogram` runs on-device with torch.
- `log_mel_spectrogram(audio)`: STFT hop 160 -> power -> mel -> log10, clamp to
  `max - 8`, then `(x + 4) / 4` (Whisper-style normalisation). Returns `(128, n_frames)`.
- `forward(wavs, accelerator=None, max_len=None) -> (tokens, lens)`: accepts a **list** of
  1-D numpy/tensor wavs at 16 kHz, computes mels one by one, truncates to `max_len * 4`
  frames if `max_len` given (4 mel frames per token), pads with `s3tokenizer.utils.padding`,
  calls `self.quantize(mels, mel_lens)`. Returns long tensors, batched (B, T).
- `pad(wavs, sr)` pads to a multiple of 40 ms (unused by callers).
- Weights come inside `s3gen.safetensors` under `tokenizer.*` (the S3Tokenizer package
  otherwise downloads its own; here `super().__init__(name)` may still trigger that
  download at construction **(unverified)**).

Constants: `S3_SR=16000`, `S3_HOP=160`, `S3_TOKEN_HOP=640`, `S3_TOKEN_RATE=25`,
`SPEECH_VOCAB_SIZE=6561`, `SOS=6561`, `EOS=6562`.
`drop_invalid_tokens(x)` (in `s3tokenizer/__init__.py`): keeps the slice after the first
SOS and before the first EOS; batch size must be 1.

## `embed_ref(ref_wav, ref_sr, device="auto", ref_fade_out=True)` -> `ref_dict`

Input: 1-D or (1, L) waveform tensor/ndarray at `ref_sr`. Warns if longer than 10 s (callers
pre-truncate to `DEC_COND_LEN = 10 * 24000`).

| key | how | shape |
|-----|-----|-------|
| `prompt_feat` | resample to 24 kHz -> `mel_spectrogram` -> transpose | (1, T_mel, 80) |
| `prompt_feat_len` | always `None` | |
| `embedding` | resample to 16 kHz -> `CAMPPlus.inference([wav])` (Kaldi fbank 80, mean-normalised) | (1, 192) |
| `prompt_token` | 16 kHz -> `S3Tokenizer` | (1, T_tok) |
| `prompt_token_len` | from tokenizer | (1,) |

If `T_mel != 2 * T_tok` (input not a multiple of 40 ms) it logs a warning and truncates
tokens to `T_mel // 2`. `ref_fade_out` is accepted but unused.
Resamplers are `torchaudio.transforms.Resample` cached by `@lru_cache(100)` on
`(src_sr, dst_sr, device)`.

## Flow: `CausalMaskedDiffWithXvec.inference(...)` (`flow.py`)

Arguments: `token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, finalize, n_timesteps=10, noised_mels=None, meanflow=False`.

1. `embedding` -> L2-normalise -> `spk_embed_affine_layer` -> (B, 80).
2. `_repeat_batch_dim` expands prompt tensors to batch B if they are batch 1.
3. **Prepend** prompt tokens: `token = cat([prompt_token, token])`, lengths added.
   Logs an error (does not raise) if any token >= 6561.
4. `input_embedding(token) * mask` -> encoder -> `h` (B, 2*T, 512). If `finalize is False`,
   drop the last `pre_lookahead_len * token_mel_ratio = 6` frames (streaming lookahead).
5. `encoder_proj(h)` -> `mu` (B, 2T, 80). Build `conds` = zeros with the prompt mel copied into
   the first `mel_len1 = prompt_feat.shape[1]` frames. So the decoder in-paints the
   continuation given the prompt's real mel.
6. `decoder(mu, mask, spks, cond, n_timesteps, noised_mels, meanflow)` -> `feat` (B, 80, 2T)
   then slice off the prompt part: `feat[:, :, mel_len1:]`; asserts remaining length == `mel_len2`.
7. Returns `(feat, None)`.

Encoder details (`transformer/upsample_encoder.py`, `UpsampleConformerEncoder`):
`LinearNoSubsampling` embed + `EspnetRelPositionalEncoding` -> `PreLookaheadLayer`
(conv with 3-frame right-lookahead, residual) -> 6 `ConformerEncoderLayer`
(`RelPositionMultiHeadedAttention`, FFN 2048, no conv module, no macaron) -> `Upsample1D`
(nearest x2 + causal conv) -> second embed/pos-enc -> 4 more conformer layers -> LayerNorm.
Chunk masks are built with `add_optional_chunk_mask` (static_chunk_size 0, full context at
inference).

## CFM decoder: `CausalConditionalCFM` (`flow_matching.py`)

`CFM_PARAMS` (`configs.py`): `sigma_min=1e-6, solver="euler", t_scheduler="cosine",
training_cfg_rate=0.2, inference_cfg_rate=0.7, reg_loss_type="l1"`.

`forward(mu, mask, n_timesteps, temperature=1.0, spks, cond, noised_mels=None, meanflow=False)`:
- `z = randn_like(mu)`; if `noised_mels` is given (meanflow path) it overwrites the
  non-prompt part of `z` (`z[..., prompt_len:] = noised_mels`).
- `t_span = linspace(0, 1, n_timesteps + 1)`; cosine-warped unless `meanflow`.
- `meanflow=True` -> `basic_euler` (no CFG, prints "S3 Token -> Mel Inference..." and a tqdm bar).
- else -> `solve_euler` with CFG: builds a 2B batch where rows B: have zero `mu`, `spks`,
  `cond`; per step `dxdt = (1 + 0.7) * cond_pred - 0.7 * uncond_pred`; `x += (r - t) * dxdt`.
  The estimator is called with `r=None` unless meanflow. Uses preallocated `x_in` etc. and
  slice-assignment (comment says concat breaks TensorRT memory format).
- `cast_all` casts floating tensors to the estimator dtype and back to the input dtype.
- `self.rand_noise = None` with a comment that fixed noise was a bad idea for distillation.

`ConditionalCFM.forward` (the non-causal parent) is dead: raises `NotImplementedError`.

## Estimator: `ConditionalDecoder` (`decoder.py`)

A causal 1-D U-Net with transformer blocks, in the Matcha-TTS style:
- inputs concatenated on channels: `x`(80) + `mu`(80) + `spks` broadcast (80) + `cond`(80) = **320 = in_channels**.
- time: `SinusoidalPosEmb(320)` -> `TimestepEmbedding` (SiLU MLP) to `time_embed_dim = 256*4 = 1024`.
  Meanflow adds an `r` embedding through the same MLP and mixes `[t_emb, r_emb]` with
  `time_embed_mixer` (Linear 2048->1024, initialised to identity on the t half,
  `utils/intmeanflow.py`, paper arXiv 2510.07979 section 3.3).
- `channels=[256]`, so: 1 down stage (`CausalResnetBlock1D` + 4 `BasicTransformerBlock` +
  `CausalConv1d`), 12 mid stages (resnet + 4 transformer blocks each), 1 up stage with skip
  concat, `final_block` (`CausalBlock1D`), `final_proj` Conv1d(256 -> 80).
- `BasicTransformerBlock` comes from `matcha/transformer.py` (diffusers `Attention`,
  `FeedForward` with gelu, `AdaLayerNorm` variants; `attention_head_dim=64`, 8 heads).
- Attention masks are built with `add_optional_chunk_mask(..., static_chunk_size=0)` then
  `mask_to_bias` (0 / -1e10). `static_chunk_size` is hard-set to 0 with a "missing?" note.
- Causality: `CausalConv1d` left-pads `kernel_size - 1`; the transformer masks are full
  (non-causal within the sequence) at inference because chunk size 0 and dynamic chunk off.
  So "causal" here means the conv path only.
- `initialize_weights()` runs at construction (kaiming) and is then overwritten by the checkpoint.

## Vocoder: `HiFTGenerator` (`hifigan.py`)

HiFTNet = Neural Source Filter + iSTFTNet.
- `inference(speech_feat (B,80,T), cache_source)`: `ConvRNNF0Predictor(mel)` -> `f0` (B,T) ->
  `f0_upsamp` (x480) -> `SourceModuleHnNSF` (8 harmonics, sine amp 0.1, noise 0.003, voiced
  threshold 10) -> source `s` (B,1,T*480) -> `decode(mel, s)` -> wav (B, T*480). Returns `(wav, s)`.
- `decode`: `conv_pre` (80->512) -> 3 transposed-conv upsamples (8,5,3) with leaky ReLU,
  each fused with a downsampled STFT of the source (`source_downs`, `source_resblocks`),
  3 `ResBlock`s per stage (kernels 3,7,11, dilations 1,3,5, Snake activations), `conv_post`
  -> 18 channels = n_fft 16: magnitude `exp(x[:9])`, phase `sin(x[9:])`, `torch.istft`
  (n_fft 16, hop 4), clamp to +-0.99.
- `cache_source` supports glitch-free streaming continuation (first N source samples
  replaced); callers always pass an empty cache.
- Weight norm is via `torch.nn.utils.parametrizations.weight_norm`; `remove_weight_norm()` exists
  but is never called (keys in checkpoints are the parametrized names).

## Public inference entry points on `S3Token2Wav`

| Method | Returns | Notes |
|--------|---------|-------|
| `inference(speech_tokens, ref_wav=None, ref_sr=None, ref_dict=None, drop_invalid_tokens=True, n_cfm_timesteps=None, speech_token_lens=None)` | `(wav (1, N), source)` | the one callers use. `drop_invalid_tokens` is accepted but its body is commented out. Applies `trim_fade`. |
| `flow_inference(...)` | mel (1, 80, 2T) | picks `n_cfm_timesteps` default `2 if meanflow else 10`; for meanflow draws the initial noise `(1, 80, 2T)` and passes it as `noised_mels`; forces `finalize=True` when called from `inference` |
| `hift_inference(speech_feat, cache_source=None)` | `(wav, source)` | |
| `forward(...)` | wav or mel | non-inference path with `skip_vocoder`, `finalize` exposed; unused by callers |

Exactly one of `ref_wav`/`ref_dict` must be given (`assert (ref_wav is None) ^ (ref_dict is None)`).
When `ref_dict` values are numpy (prod API), they are converted and cast to model device/dtype **in place**.

Length bookkeeping: N tokens -> 2N mel frames -> 2N * 480 = 960N samples. The prompt part
is removed before the vocoder, so output length is exactly 960 * N_generated_tokens
(Turbo adds 3 silence tokens = 120 ms; Multilingual then chops the last 960 samples).
