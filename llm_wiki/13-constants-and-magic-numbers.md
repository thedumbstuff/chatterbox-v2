# 13 - Constants and magic numbers

Single lookup table. "Defined in" is the canonical location; "also hard-coded in" lists
places that repeat the literal instead of importing it (change all of them together).

## Audio and token rates

| Name | Value | Defined in | Also hard-coded in |
|------|-------|-----------|--------------------|
| `S3_SR` | 16000 | `s3tokenizer/s3tokenizer.py` | `VoiceEncConfig.sample_rate` (16000), `xvector.py` fbank assumes 16 kHz |
| `S3_HOP` | 160 (100 mel frames/s for the S3 tokenizer) | same | `VoiceEncConfig.hop_size` |
| `S3_TOKEN_HOP` | 640 samples (25 tokens/s) | same | |
| `S3_TOKEN_RATE` | 25 tokens/s | same | `flow.py` `input_frame_rate=25`; `s3tokenizer.py` "4 mel frames per token" |
| `S3GEN_SR` | 24000 | `s3gen/const.py` | `mel.py` defaults (`sampling_rate=24000`), HiFT `sampling_rate` arg |
| `S3GEN_SIL` | 4299 (a silence speech token) | `s3gen/const.py` | appended x3 in `tts_turbo.py` |
| S3Gen mel | n_fft 1920, hop 480, win 1920, 80 mels, fmin 0, fmax 8000, center False | `s3gen/utils/mel.py` defaults | `flow.py` `mel_feat_conf` dict says 22050/256 (stale CosyVoice default, unused) |
| mel frames per token | 2 (`token_mel_ratio`) | `flow.py` | `flow_inference` noise shape `speech_tokens.size(-1) * 2`; `embed_ref` check `2 * tokens` |
| samples per token | 960 (= 24000 / 25) | derived | `mtl_tts.py` trims `S3GEN_SR // S3_TOKEN_RATE` |
| HiFT upsample | rates [8, 5, 3], kernels [16, 11, 7], iSTFT n_fft 16 hop 4 -> 480 samples/frame | `s3gen.py` HiFT ctor | |
| `trim_fade` | 960 samples: 480 zeros + 480 half-cosine (20 ms + 20 ms) | `S3Token2Wav.__init__` (`n_trim = S3GEN_SR // 50`) | |
| S3 tokenizer log-mel | n_fft 400, hop 160, 128 mels, Whisper-style `(log10 + 4) / 4`, floor at max - 8 | `s3tokenizer.py` | |
| Voice encoder mel | 40 mels, n_fft 400, hop 160, win 400, fmax 8000, power 2, amplitude scale | `voice_encoder/config.py` | |
| Voice encoder partial | 160 frames (1.6 s), `rate=1.3` -> step 77 frames, `min_coverage=0.8`, `trim_top_db=20` | `voice_encoder.py` | |
| CAMPPlus fbank | 80 bins, mean-normalised | `xvector.py::extract_feature` | |

## Vocabularies and special tokens

| Name | Value | Where |
|------|-------|-------|
| `SPEECH_VOCAB_SIZE` | 6561 (= 3^8) | `s3tokenizer/s3tokenizer.py` |
| `SOS` / `EOS` (speech) | 6561 / 6562 | `s3tokenizer/__init__.py`; same values as `T3Config.start_speech_token` / `stop_speech_token` |
| `T3Config.speech_tokens_dict_size` | 8194 (Llama variants), 6563 (Turbo/Nano) | `t3_config.py`, `tts_turbo.py` |
| `T3Config.text_tokens_dict_size` | 704 (English), 2454 (multilingual; `is_multilingual` checks this), 50276 (Turbo/Nano) | `t3_config.py`, `tts_turbo.py` |
| `T3Config.start_text_token` / `stop_text_token` | 255 / 0 | `t3_config.py` (must match `[START]`/`[STOP]` ids in the tokenizer json) |
| Text special tokens | `[START] [STOP] [UNK] [SPACE] [PAD] [SEP] [CLS] [MASK]` | `tokenizers/tokenizer.py` |
| Language tag tokens | `[ar]` ... `[zh]` prepended by `MTLTokenizer` | `tokenizer.py`, list in `mtl_tts.py::SUPPORTED_LANGUAGES` |
| Cangjie tokens | `[cj_<char>]` ... `[cj_.]` | `tokenizer.py` |
| Turbo tokenizer length | 50276 (warning if different) | `tts_turbo.py` |
| Paralinguistic tags | `[clear throat] [sigh] [shush] [cough] [groan] [sniff] [gasp] [chuckle] [laugh]` | `gradio_tts_turbo_app.py::EVENT_TAGS` (UI list; the model may know more) |
| Speech filter | `speech_tokens < 6561` | `tts.py`, `tts_turbo.py` |
| Cross-entropy ignore id | -100 | `t3.py::loss` |

## Lengths and limits

| Name | Value | Where |
|------|-------|-------|
| `ENC_COND_LEN` | 6 s * 16 kHz (original, multilingual); 15 s (Turbo/Nano) | class constants in each entry file |
| `DEC_COND_LEN` | 10 s * 24 kHz | same |
| `speech_cond_prompt_len` | 150 (Llama), 375 (Turbo/Nano) | `t3_config.py`, `tts_turbo.py` |
| Turbo min reference | > 5.0 s (assert) | `tts_turbo.py::prepare_conditionals` |
| `embed_ref` warning | ref > 10 s | `s3gen.py` |
| `max_new_tokens` / `max_gen_len` | 1000 tokens (40 s) | `tts.py`, `mtl_tts.py`, `t3.py::inference_turbo` default |
| `T3Config.max_text_tokens` / `max_speech_tokens` | 2048 / 4096 (learned pos-emb table sizes +2 / +4) | `t3_config.py` |
| Perceiver queries | 32 x 1024, 4 heads | `perceiver.py` |
| `speaker_embed_size` | 256 | `t3_config.py`, `VoiceEncConfig` |
| CAMPPlus embedding | 192 -> projected to 80 | `xvector.py`, `flow.py` |
| Flow encoder | 512-d, 8 heads, 2048 FFN, 6 + 4 blocks, `pre_lookahead_len` 3 tokens | `s3gen.py`, `upsample_encoder.py`, `flow.py` |
| CFM estimator | in 320, out 80, channels [256], 4 transformer blocks/stage, 12 mid blocks, 8 heads x 64, gelu, time-embed dim 1024 | `s3gen.py` |
| `CFM_PARAMS` | sigma_min 1e-6, euler, cosine, training_cfg 0.2, **inference_cfg_rate 0.7** | `s3gen/configs.py` (a stale duplicate `DictConfig` sits in `flow.py::decoder_conf`) |
| CFM steps | 10 (normal), 2 (meanflow) | `s3gen.py::flow_inference`, `tts_turbo.py` |
| Loudness target | -27 LUFS | `tts_turbo.py::norm_loudness` |
| Gradio text cap | 300 chars | apps |
| HiFT NSF | 8 harmonics, sine amp 0.1, noise std 0.003, voiced threshold 10, audio limit 0.99 | `s3gen.py`, `hifigan.py` |
| `mask_to_bias` | -1e10 for masked | `decoder.py` |

## Sampling defaults

| Parameter | Original | Multilingual | Turbo/Nano | `T3.inference` internal default |
|-----------|----------|--------------|------------|--------------------------------|
| temperature | 0.8 | 0.8 | 0.8 | 0.8 |
| top_p | 1.0 | 1.0 | 0.95 | 0.95 |
| min_p | 0.05 | 0.05 | 0.0 (ignored) | 0.05 |
| top_k | - | - | 1000 | - |
| repetition_penalty | 1.2 | 1.2 | 1.2 | 1.2 |
| cfg_weight | 0.5 | 0.5 | 0.0 (ignored) | 0.5 |
| exaggeration | 0.5 | 0.5 | 0.0 (ignored) | `T3Cond.emotion_adv` default 0.5 |

## Hugging Face identifiers

`ResembleAI/chatterbox`, `ResembleAI/chatterbox-turbo`, `ResembleAI/chatterbox-nano`,
`ResembleAI/Chatterbox-Multilingual-{zh-cmn,es-mx-latam,pt-br,es-es,pt-pt,hi}`.
Filenames: see [03-model-variants-and-checkpoints.md](03-model-variants-and-checkpoints.md).
