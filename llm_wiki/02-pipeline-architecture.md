# 02 - Pipeline architecture

## One-paragraph model

Chatterbox is a **two-stage cascaded TTS**. Stage 1 (**T3**, "Token-To-Token") is a decoder-only
LLM (Llama 520M or GPT-2) that reads `[conditioning] + [text tokens]` and autoregressively
emits **discrete speech tokens** at 25 tokens/second from a 6561-entry codebook
(S3 tokenizer v2 codebook). Stage 2 (**S3Gen**) is CosyVoice's token-to-waveform stack: a
conformer encoder upsamples tokens 2x to 50 Hz, a **conditional flow-matching** (CFM) decoder
turns them into an 80-bin mel spectrogram, and a **HiFT** (NSF + iSTFT HiFi-GAN) vocoder turns
mel into a 24 kHz waveform. Speaker identity is injected twice: a 256-d **VoiceEncoder**
embedding plus prompt speech tokens into T3, and a 192-d **CAMPPlus x-vector** plus prompt
mel/tokens into S3Gen.

## Data flow (TTS, all variants)

```
                reference wav (audio_prompt_path)  ────────────────────────────────────┐
                        │ librosa.load @ 24 kHz                                         │
                        ├── resample -> 16 kHz ─┬─> VoiceEncoder.embeds_from_wavs  -> speaker_emb (1,256)
                        │                       └─> S3Tokenizer (first ENC_COND_LEN s)  -> cond_prompt_speech_tokens (1, <=150|375)
                        │                                                       (exaggeration scalar -> emotion_adv (1,1,1))
                        │                                       ===> T3Cond  ─────────────────────────────┐
                        └── first DEC_COND_LEN=10 s @ 24 kHz -> S3Gen.embed_ref()                          │
                                  ├─ mel_spectrogram @24k (80 x T, hop 480)      -> prompt_feat            │
                                  ├─ resample 16k -> CAMPPlus.inference           -> embedding (1,192)     │
                                  └─ resample 16k -> S3Tokenizer                  -> prompt_token (25 Hz)  │
                                                                 ===> ref_dict (S3Gen conditionals)       │
                                                                                                          │
text ──> punc_norm() ──> tokenizer ──> text_tokens (1, T)  [+ SOT/EOT pads for Llama variants]           │
                                              │                                                           │
                                              ▼                                                           │
        T3.prepare_input_embeds:  [cond_emb | text_emb(+pos) | speech_emb(BOS)(+pos)]  <──────────────────┘
                                              │
                                              ▼
        T3.inference (Llama, CFG, batch 2)  or  T3.inference_turbo (GPT-2, no CFG, batch 1)
              kv-cache loop, sample next speech token, stop at stop_speech_token (6562) or 1000 tokens
                                              │
                                              ▼
        speech_tokens (1, N) in [0, 6561)  ── drop SOS/EOS/invalid ──  (Turbo: append 3x S3GEN_SIL=4299)
                                              │
                                              ▼
        S3Gen.inference(speech_tokens, ref_dict):
              flow.inference:  concat [prompt_token | speech_tokens] -> Embedding(6561,512)
                               -> UpsampleConformerEncoder (x2 -> 50 Hz) -> encoder_proj -> mu (B,80,2N')
                               -> CausalConditionalCFM: Euler ODE from noise, n steps (10 normal / 2 meanflow)
                                  estimator = ConditionalDecoder (causal U-Net + transformers), CFG rate 0.7
                               -> drop the prompt_feat prefix -> mel (1,80,2N)
              hift_inference:  ConvRNNF0Predictor(mel) -> f0 -> SourceModuleHnNSF -> harmonic source
                               HiFTGenerator.decode(mel, source) -> iSTFT -> wav (1, 2N*480)
              zero first 20 ms + 20 ms cosine fade-in (trim_fade)
                                              │
                                              ▼
        (Multilingual only) drop the last token's 960 samples
        perth watermark -> torch tensor (1, N_samples) @ 24 kHz
```

## Stage boundaries and rates (memorize these)

| Quantity | Value | Where |
|----------|-------|-------|
| T3 speech token rate | 25 tokens/s | `S3_TOKEN_RATE` |
| S3 tokenizer input | 16 kHz, log-mel 128 bins, hop 160 (100 fps), 4 mel frames per token | `s3tokenizer.py` |
| Speech codebook size | 6561 (= 3^8, FSQ) ; SOS 6561, EOS 6562 | `SPEECH_VOCAB_SIZE`, `s3tokenizer/__init__.py` |
| S3Gen mel | 24 kHz, n_fft 1920, hop 480, win 1920, 80 mels, fmax 8000 -> 50 frames/s, **2 mel frames per token** | `s3gen/utils/mel.py` |
| HiFT upsampling | 8 x 5 x 3 = 120, times iSTFT hop 4 = 480 samples per mel frame | `s3gen.py` HiFT args |
| Output | 24 kHz -> 960 samples per speech token, 40 ms | `S3GEN_SR` |
| Max T3 output | 1000 tokens = 40 s of audio (hard-coded `max_new_tokens`/`max_gen_len`) | `tts.py`, `mtl_tts.py`, `t3.py` |

## The three conditioning paths

1. **T3 speaker embedding**: `VoiceEncoder` (3-layer LSTM, 40-mel, 16 kHz) -> 256-d, mean over
   partial windows, projected to model dim by `T3CondEnc.spkr_enc`. One token in the prefix.
2. **T3 speech prompt**: the first `ENC_COND_LEN` seconds of the reference are tokenized by
   the S3 tokenizer, capped at `speech_cond_prompt_len` tokens (150 = 6 s for Llama variants,
   375 = 15 s for Turbo/Nano), embedded with `speech_emb` (+ `speech_pos_emb` for Llama) and,
   for Llama variants, compressed by a **Perceiver resampler to 32 tokens**. For Turbo/Nano
   `use_perceiver_resampler=False`, so all <=375 prompt tokens go straight into the prefix.
3. **T3 emotion / exaggeration**: scalar `exaggeration` -> `emotion_adv_fc` (Linear 1->dim, no
   bias) -> one prefix token. Only Llama variants (`hp.emotion_adv=True`). Turbo/Nano set
   `emotion_adv=False`, so the value is ignored (a warning is logged if > 0).
4. **S3Gen**: `ref_dict` = `{prompt_token, prompt_token_len, prompt_feat, prompt_feat_len,
   embedding}`. The prompt tokens are **prepended** to the generated tokens and prompt mel is
   given as `cond` so the CFM in-paints the continuation in the same voice (CosyVoice
   style). `embedding` (192-d CAMPPlus) is projected to 80-d and injected in the estimator.

## Prefix layout T3 actually sees

```
Llama variants:  [spk(1)] [perceiver(32)] [emotion(1)] [SOT text... EOT] [BOS_speech] -> generate...
Turbo/Nano:      [spk(1)] [prompt speech tokens (<=375)] [GPT-2 BPE text tokens] [BOS_speech] -> generate...
```

`T3CondEnc.forward` concatenates `(cond_spkr, cond_clap(empty), cond_prompt_speech_emb, cond_emotion_adv)`.
`clap_emb` is a placeholder that must be `None` (asserted).

## Classifier-free guidance (Llama variants only)

`T3.inference` always runs **batch 2**: row 0 is conditional, row 1 has its text embeddings
zeroed (`text_emb[1].zero_()` in `prepare_input_embeds` when `cfg_weight > 0`). At each step
`logits = cond + cfg_weight * (cond - uncond)`. The callers duplicate the text tokens to make
batch 2 (`torch.cat([text_tokens, text_tokens])`). See gotcha G1 in
[12-gotchas-known-issues.md](12-gotchas-known-issues.md) about `cfg_weight=0` in `tts.py`.

S3Gen has its own, separate CFG inside the flow-matching solver (`inference_cfg_rate=0.7`,
un-conditioned branch has `mu`, `spks`, `cond` zeroed). Not exposed as a user parameter.
The meanflow (Turbo) decoder skips CFG entirely (`basic_euler`).

## Voice conversion path

`ChatterboxVC.generate(audio, target_voice_path)`: load source at 16 kHz -> `S3Gen.tokenizer`
-> speech tokens -> `S3Gen.inference(tokens, ref_dict)` with the target voice's `ref_dict`
(from `embed_ref`). No T3 involved. Content is preserved by the tokens; timbre comes from
`ref_dict`.
