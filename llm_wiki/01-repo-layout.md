# 01 - Repo layout

Line counts are at commit `5de7a54`. Total Python in `src/`: ~8.7k lines across 60 files.

```
chatterbox-v2/
├── CLAUDE.md                     # Claude Code instructions for this repo (ours)
├── llm_wiki/                     # this knowledge base (ours)
├── README.md                     # upstream README: model zoo, usage, language list, tips
├── LICENSE                       # MIT
├── pyproject.toml                # package `chatterbox-tts` 0.1.7, setuptools, src layout
├── .gitignore                    # ignores **/*.wav, checkpoints/, syn_out/, .gradio, build junk
├── .github/workflows/install_check.yml   # CI: pip install -e . on Python 3.10, nothing else
├── Chatterbox-Multilingual.png / Chatterbox-Turbo.jpg   # README banners
│
├── app.py                        # OURS: Chatterbox Studio UI (all models, voice library, history, VC)
├── tests/                        # OURS: smoke_generate.py, g1_equivalence.py, bench_watermark.py
├── voices/                       # OURS (git-ignored): saved voices <slug>/{ref.wav, meta.json, conds_<model>.pt}
├── example_tts.py                # ChatterboxTTS + Multilingual(fr) demo, auto device
├── example_tts_turbo.py          # ChatterboxTurboTTS demo with [chuckle] tag
├── example_tts_nano.py           # same with nano=True
├── example_vc.py                 # ChatterboxVC demo
├── example_for_mac.py            # MPS demo; monkeypatches torch.load map_location
├── gradio_tts_app.py             # Gradio UI for original English model (all sampling knobs)
├── gradio_tts_turbo_app.py       # Gradio UI for Turbo with clickable event-tag buttons
├── gradio_vc_app.py              # Gradio UI for voice conversion
├── multilingual_app.py           # Gradio UI for Multilingual, 23 language presets, MCP server on
│
└── src/chatterbox/
    ├── __init__.py               # exports ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    ├── tts.py            (272)   # ChatterboxTTS: original English model. punc_norm, Conditionals, generate()
    ├── mtl_tts.py        (355)   # ChatterboxMultilingualTTS: 23 languages, v2/v3 T3 selection
    ├── tts_turbo.py      (320)   # ChatterboxTurboTTS: Turbo + Nano, GPT-2 backbone, meanflow S3Gen
    ├── vc.py             (103)   # ChatterboxVC: S3Gen-only voice conversion
    ├── watermark.py      (ours)  # optional Perth watermark: get_watermarker(), maybe_watermark(), resolve()
    ├── studio/           (ours)  # UI-independent app logic
    │   ├── engine.py             # Engine: lazy model cache, apply_voice() with per-(voice,model) conds cache, generate() with chunking, convert()
    │   ├── library.py            # VoiceLibrary (voices/ dir), History (syn_out/history), DEFAULT_VOICE sentinel
    │   └── textsplit.py          # split_text(): sentence/clause/word chunking incl. Devanagari danda
    └── models/
        ├── utils.py      (4)     # AttrDict
        ├── t3/                   # === Stage 1: text -> speech tokens (autoregressive LLM) ===
        │   ├── t3.py             (468)  # T3 nn.Module: embeddings, cond, forward/loss, inference (CFG loop), inference_turbo
        │   ├── llama_configs.py  (109)  # LLAMA_CONFIGS: Llama_520M, GPT2_medium, GPT2_small
        │   ├── inference/t3_hf_backend.py (111)  # T3HuggingfaceBackend: wraps LlamaModel for kv-cache stepping
        │   └── modules/
        │       ├── t3_config.py  (41)   # T3Config hyperparams; english_only() / multilingual()
        │       ├── cond_enc.py   (97)   # T3Cond dataclass + T3CondEnc (speaker proj, perceiver, emotion_adv)
        │       ├── perceiver.py  (212)  # Perceiver resampler (32 learned queries) over prompt speech embeddings
        │       └── learned_pos_emb.py (32) # LearnedPositionEmbeddings (used by Llama variants only)
        ├── s3tokenizer/          # === Speech tokenizer (audio -> 25 Hz discrete tokens) ===
        │   ├── __init__.py       (30)   # constants, SOS/EOS, drop_invalid_tokens()
        │   └── s3tokenizer.py    (168)  # S3Tokenizer(S3TokenizerV2) from the `s3tokenizer` pip package, own log-mel
        ├── s3gen/                # === Stage 2: speech tokens -> mel -> waveform (vendored CosyVoice) ===
        │   ├── __init__.py       (2)    # exports S3Token2Wav as S3Gen, S3GEN_SR
        │   ├── const.py          (2)    # S3GEN_SR=24000, S3GEN_SIL=4299
        │   ├── configs.py        (10)   # CFM_PARAMS
        │   ├── s3gen.py          (362)  # S3Token2Mel (tokenizer+CAMPPlus+flow) and S3Token2Wav (+HiFT), embed_ref(), inference()
        │   ├── flow.py           (198)  # CausalMaskedDiffWithXvec: token embedding -> encoder -> CFM decoder, prompt concat
        │   ├── flow_matching.py  (246)  # ConditionalCFM / CausalConditionalCFM: Euler ODE solver with CFG, meanflow path
        │   ├── decoder.py        (333)  # ConditionalDecoder: causal 1D U-Net + transformer blocks = the CFM estimator
        │   ├── hifigan.py        (474)  # HiFTGenerator: NSF source + iSTFT HiFi-GAN vocoder (mel -> wav)
        │   ├── f0_predictor.py   (55)   # ConvRNNF0Predictor (mel -> f0) used inside HiFT
        │   ├── xvector.py        (428)  # CAMPPlus speaker encoder (x-vector, 192-d) on Kaldi fbank
        │   ├── matcha/           # vendored Matcha-TTS pieces used by decoder.py
        │   │   ├── decoder.py    (443)  # SinusoidalPosEmb, Block1D, ResnetBlock1D, Down/Upsample1D, TimestepEmbedding
        │   │   ├── flow_matching.py (129) # BASECFM / CFM base classes
        │   │   ├── transformer.py (316) # BasicTransformerBlock (diffusers-based), SnakeBeta, FeedForward
        │   │   └── text_encoder.py (413) # Matcha TextEncoder (NOT used at runtime; dead code)
        │   ├── transformer/      # vendored WeNet/ESPnet conformer used by the flow encoder
        │   │   ├── upsample_encoder.py (318) # UpsampleConformerEncoder: 6 blocks -> x2 upsample -> 4 blocks; PreLookaheadLayer
        │   │   ├── encoder_layer.py, attention.py, embedding.py, convolution.py,
        │   │   │   positionwise_feed_forward.py, subsampling.py, activation.py
        │   └── utils/
        │       ├── mel.py        (85)   # mel_spectrogram(): 24 kHz, n_fft 1920, hop 480, 80 mels (Matcha/CosyVoice)
        │       ├── mask.py       (194)  # make_pad_mask, add_optional_chunk_mask, subsequent_chunk_mask
        │       ├── class_utils.py (71)  # registries mapping strings -> encoder building blocks
        │       └── intmeanflow.py (36)  # time-embedding mixer for meanflow (diag init), has a __main__ self-test
        ├── tokenizers/
        │   └── tokenizer.py      (313)  # EnTokenizer, MTLTokenizer, per-language normalizers (ja/he/ko/zh/ru)
        └── voice_encoder/        # === Speaker embedding for T3 conditioning (Real-Time-Voice-Cloning LSTM) ===
            ├── voice_encoder.py  (274)  # VoiceEncoder: 3-layer LSTM over 40-mel partials -> 256-d L2-normed embedding
            ├── melspec.py        (78)   # librosa mel for the voice encoder (16 kHz, 40 mels)
            └── config.py         (18)   # VoiceEncConfig
```

## Which files matter for which task

| Task | Files |
|------|-------|
| Change the public API / add generate options | `tts.py`, `mtl_tts.py`, `tts_turbo.py`, `vc.py` |
| Change sampling / decoding of speech tokens | `models/t3/t3.py` (`inference`, `inference_turbo`) |
| Change speaker / emotion conditioning of T3 | `models/t3/modules/cond_enc.py`, `perceiver.py`, `voice_encoder/` |
| Change text preprocessing or add a language | `models/tokenizers/tokenizer.py`, `punc_norm` in each entry file, `SUPPORTED_LANGUAGES` in `mtl_tts.py` |
| Change audio quality / decoder steps / vocoder | `models/s3gen/s3gen.py`, `flow.py`, `flow_matching.py`, `decoder.py`, `hifigan.py` |
| Add streaming | `s3gen.py` (`finalize`, `flow_inference`, `hift_inference`), `t3.py` loops; see `14-extension-points.md` |
| Model loading / new checkpoint | `from_local` / `from_pretrained` in each entry file; `T3Config`; `llama_configs.py` |
| Demo UI | `gradio_*.py`, `multilingual_app.py` |

## Dead / unused code (safe to ignore, unsafe to delete without checking state dicts)

- `models/s3gen/matcha/text_encoder.py`: not imported by anything at runtime.
- `models/s3gen/matcha/decoder.py::Decoder` and `ConformerWrapper`: only the small building
  blocks are imported by `s3gen/decoder.py`.
- `ConditionalCFM.forward` raises `NotImplementedError` (only `CausalConditionalCFM.forward` is used).
- `T3HuggingfaceBackend.prepare_inputs_for_generation` and the commented-out
  `self.patched_model.generate(...)` call in `T3.inference`: the HF `generate()` path is
  abandoned; a manual kv-cache loop is used instead.
- `T3.forward` / `T3.loss` and `CausalMaskedDiffWithXvec.compute_loss` /
  `ConditionalCFM.compute_loss`: training-time code kept from upstream, never called here.
- `S3Token2Wav.forward` (non-inference path) and `HiFTGenerator.forward(batch, device)`.
- `S3Gen.resamplers = {}` attribute is never used (module-level `get_resampler` lru_cache is).
- Deleting modules that own parameters (even unused ones) changes state-dict keys and
  breaks checkpoint loading. Check `strict=` usage in the loaders first
  ([03-model-variants-and-checkpoints.md](03-model-variants-and-checkpoints.md)).
