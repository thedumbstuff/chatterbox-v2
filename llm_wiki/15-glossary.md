# 15 - Glossary

| Term | Meaning in this codebase |
|------|--------------------------|
| **T3** | "Token-To-Token": the autoregressive transformer (`models/t3/t3.py`) that maps text tokens + conditioning to speech tokens. Llama 520M (original, multilingual) or GPT-2 (Turbo, Nano). |
| **S3 / S3 tokens** | Discrete 25 Hz speech tokens from the S3Tokenizer v2 (`speech_tokenizer_v2_25hz`, codebook 6561). "S3" is the tokenizer family name (from CosyVoice's supervised semantic speech tokenizer). |
| **S3Gen** | The token-to-waveform generator (`models/s3gen/`): flow-matching mel decoder + HiFT vocoder. Alias of `S3Token2Wav`. |
| **S3Token2Mel / S3Token2Wav** | The two classes in `s3gen.py`; Wav extends Mel with the vocoder. |
| **CFM** | Conditional Flow Matching: the mel decoder learns a velocity field from noise to mel; sampled with an Euler ODE solver in `n_timesteps` steps. |
| **meanflow** | A distilled CFM variant (Turbo/Nano `s3gen_meanflow.safetensors`) whose estimator takes both start time `t` and end time `r`, enabling 1-2 step sampling without CFG. Time embeddings mixed by `time_embed_mixer` (`utils/intmeanflow.py`). |
| **estimator** | The neural net inside the CFM that predicts velocity: `ConditionalDecoder` (causal U-Net + transformer blocks). |
| **HiFT / HiFTNet** | HiFi-GAN variant with a Neural Source Filter and iSTFT output (`hifigan.py::HiFTGenerator`). Mel (50 fps) -> 24 kHz wav. |
| **NSF** | Neural Source Filter: harmonic sine source built from predicted F0 (`SourceModuleHnNSF`, `SineGen`) fused into the vocoder. |
| **F0** | Fundamental frequency, predicted from mel by `ConvRNNF0Predictor`. |
| **x-vector / CAMPPlus** | Speaker embedding network (192-d) used by S3Gen (`xvector.py`). |
| **VoiceEncoder / VE** | LSTM speaker encoder (256-d) used by T3 (`voice_encoder/`). Files `ve.safetensors` / `ve.pt`. |
| **Perceiver (resampler)** | Cross-attention module compressing the speech prompt embeddings to 32 tokens for T3's prefix (Llama variants). |
| **T3Cond** | Dataclass with T3 conditioning: `speaker_emb`, `cond_prompt_speech_tokens`, `cond_prompt_speech_emb`, `emotion_adv`, (`clap_emb` unused). |
| **Conditionals** | Dataclass pairing a `T3Cond` with an S3Gen `ref_dict`; saved as `conds.pt`. |
| **ref_dict** | S3Gen conditioning dict: `prompt_token`, `prompt_token_len`, `prompt_feat` (mel), `prompt_feat_len`, `embedding` (x-vector). |
| **conds.pt** | Built-in default voice shipped in each HF repo (a saved `Conditionals`). |
| **audio prompt / reference clip** | The wav given as `audio_prompt_path` for zero-shot voice cloning. |
| **exaggeration / emotion_adv** | Scalar emotion-intensity control (Llama variants) injected as one prefix token. |
| **CFG / cfg_weight** | Classifier-free guidance. In T3: text-conditional vs text-unconditional logits mixed with `cfg_weight`. In S3Gen: fixed `inference_cfg_rate=0.7` inside the ODE solver. |
| **SOT / EOT** | Start/stop **text** tokens (`[START]`=255, `[STOP]`=0) padded around text for Llama variants. |
| **BOS / EOS (speech)** | `start_speech_token`=6561, `stop_speech_token`=6562. |
| **S3GEN_SIL** | Speech token 4299, treated as silence; Turbo appends three. |
| **prompt token / prompt feat** | The reference clip's speech tokens and mel, prepended in the flow so the CFM continues in the same voice. |
| **finalize** | Flow flag; `False` = streaming chunk, drop lookahead frames. |
| **pre-lookahead** | `PreLookaheadLayer` in the flow encoder: 3-token right context. |
| **trim_fade** | 40 ms zero + fade-in applied to every S3Gen output start to hide prompt spill-over. |
| **Perth watermark** | Resemble's imperceptible neural audio watermark (`resemble-perth`), applied to every output. Detect with `perth.PerthImplicitWatermarker().get_watermark`. |
| **paralinguistic tags** | Bracketed cues like `[laugh]`, `[cough]` understood by Turbo/Nano. |
| **Turbo / Nano** | GPT-2-medium (350M) and GPT-2-small (110M) T3 variants sharing the meanflow S3Gen; English only. |
| **Multilingual v2 / v3** | Two T3 checkpoints (`t3_mtl23ls_v2/v3.safetensors`) for 23 languages; same tokenizer/config. |
| **Single Language Pack** | Six dedicated multilingual-architecture finetunes on separate HF repos. |
| **Cangjie** | Chinese character input-method codes used to spell Chinese characters as `[cj_x]` tokens for the multilingual tokenizer. |
| **Jamo** | Korean letter components; Hangul syllables are decomposed into them before tokenisation. |
| **LUFS** | Loudness unit; Turbo normalises references to -27 LUFS. |
| **Xet** | Hugging Face's newer storage backend; the Turbo loader retries downloads with it disabled on known errors. |
| **kv-cache / past_key_values** | Transformer attention cache used by both T3 loops to generate one token per step. |
| **min_p / top_p / top_k** | Sampling truncation rules (see 09). |
| **repetition_penalty** | Multiplicative logit penalty on previously generated speech tokens. |
