# src/chatterbox/models — module-level notes for Claude

Read `../../../llm_wiki/04-t3-text-to-speech-tokens.md` (t3/), `05-s3gen-tokens-to-waveform.md`
(s3gen/, s3tokenizer/), `06-text-tokenizers-and-normalization.md` (tokenizers/),
`07-voice-conditioning.md` (voice_encoder/) before editing here.

## Checkpoint compatibility is the #1 constraint
- Attribute names on every `nn.Module` here are state-dict keys in files on Hugging Face
  (`t3_cfg.safetensors`, `t3_mtl23ls_v{2,3}.safetensors`, `t3_{turbo,nano}_v1.safetensors`,
  `s3gen.safetensors`/`s3gen.pt`, `s3gen_meanflow.safetensors`, `ve.safetensors`/`ve.pt`).
  Renaming a submodule, changing `nn.Sequential` ordering, or swapping `weight_norm`
  parametrisation breaks loading. Buffers registered with `persistent=False` are safe.
- Loading strictness differs: T3 always strict; S3Gen `strict=False` for original/multilingual/VC
  (missing keys go unnoticed) and `strict=True` for Turbo/Nano. After any structural edit,
  load all variants and print `missing_keys` / `unexpected_keys`.
- Turbo/Nano delete `t3.tfmr.wte` after loading; do not rely on it existing.

## Vendored code
- `s3gen/` (except `s3gen.py` glue), `s3gen/matcha/`, `s3gen/transformer/`, `s3gen/utils/`,
  `s3gen/xvector.py`, `s3gen/hifigan.py`, `s3gen/f0_predictor.py`: CosyVoice / Matcha-TTS /
  WeNet / ESPnet / 3D-Speaker, Apache-2.0. Keep headers, minimal diffs, no reformatting.
- `voice_encoder/`: adapted from Real-Time-Voice-Cloning (MIT).
- `s3tokenizer/`: thin subclass of the `s3tokenizer` pip package's `S3TokenizerV2`.
- Dead but parameter-owning code exists (`matcha/text_encoder.py`, `matcha/decoder.py::Decoder`,
  `ConditionalCFM.forward`); deleting is safe only if it owns no keys in the checkpoints.

## Shape and rate invariants
- Speech tokens: 25 Hz, ids in `[0, 6561)`; SOS 6561, EOS 6562; `S3GEN_SIL` 4299.
- Mel: 24 kHz, hop 480 → 50 fps → exactly 2 mel frames per token; HiFT → 480 samples per frame.
- T3 prefix: `[cond | text | speech]`; Llama variants run batch 2 for CFG (row 1 = text zeroed)
  and add learned positional embeddings that restart per segment; GPT-2 variants batch 1, no CFG.
- Everything assumes batch size 1 at inference. Everything is fp32.

## Style
Match the surrounding file (these files come from different upstreams and differ in style).
Use `logging`, not `print`, in new code. Keep `@torch.inference_mode()` on inference methods.
