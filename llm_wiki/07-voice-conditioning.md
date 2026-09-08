# 07 - Voice conditioning (reference audio -> conditionals)

## The `Conditionals` container (defined identically in `tts.py`, `mtl_tts.py`, `tts_turbo.py`)

```python
@dataclass
class Conditionals:
    t3: T3Cond      # speaker_emb, cond_prompt_speech_tokens, emotion_adv (+ cached cond_prompt_speech_emb)
    gen: dict       # S3Gen ref_dict: prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding
    def to(self, device)            # moves both
    def save(self, fpath)           # torch.save({"t3": self.t3.__dict__, "gen": self.gen})
    @classmethod load(cls, fpath, map_location="cpu")   # torch.load(weights_only=True) -> T3Cond(**t3), gen
```
- `conds.pt` shipped in each HF repo is exactly this format (the built-in voice).
- `save` writes `t3.__dict__`, which includes `cond_prompt_speech_emb` if it has been
  populated by a previous generate call. Loading such a file back skips re-embedding.
- `ChatterboxVC` does not use `Conditionals`; it stores only `ref_dict` (`states['gen']` from `conds.pt`).

## `prepare_conditionals(wav_fpath, exaggeration=0.5[, norm_loudness=True])`

Identical structure in all three TTS classes:

```
s3gen_ref_wav = librosa.load(wav_fpath, sr=24000)          # mono float32; librosa resamples any input rate
[Turbo only] assert duration > 5 s ; optional loudness normalisation to -27 LUFS (pyloudnorm), skipped with a printed warning on error
ref_16k_wav   = librosa.resample(24k -> 16k)
s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]               # first 10 s @ 24 kHz
gen  = s3gen.embed_ref(s3gen_ref_wav, 24000, device)       # see 05
if hp.speech_cond_prompt_len:
    tokens, _ = s3gen.tokenizer.forward([ref_16k_wav[:ENC_COND_LEN]], max_len=speech_cond_prompt_len)
    t3_cond_prompt_tokens = atleast_2d(tokens).to(device)
ve_embed = ve.embeds_from_wavs([ref_16k_wav], sample_rate=16000).mean(0, keepdim=True)   # full-length 16 kHz clip
t3 = T3Cond(speaker_emb=ve_embed, cond_prompt_speech_tokens=..., emotion_adv=exaggeration * ones(1,1,1)).to(device)
self.conds = Conditionals(t3, gen)
```

| Constant | Original | Multilingual | Turbo/Nano | Meaning |
|----------|----------|--------------|------------|---------|
| `ENC_COND_LEN` | 6 s * 16 kHz | 6 s | **15 s** | audio fed to the S3 tokenizer for the T3 speech prompt |
| `speech_cond_prompt_len` | 150 tokens | 150 | 375 | cap on those tokens (6 s / 15 s at 25 Hz) |
| `DEC_COND_LEN` | 10 s * 24 kHz | 10 s | 10 s | audio fed to `S3Gen.embed_ref` |
| VoiceEncoder input | whole clip | whole clip | whole clip (after loudness norm) | |

Consequences:
- The **speaker embedding** sees the whole reference (could be minutes; cost is linear).
- The **S3Gen prompt** is the first 10 s; the **T3 prompt** the first 6 s (15 s Turbo).
  A reference whose first seconds are silence or noise gives poor prompts. Consider
  trimming/selecting the best window before calling (see 14-extension-points).
- In `tts.py` (original), the `t3_cond_prompt_tokens` variable is only assigned inside
  `if plen := ...`; since `speech_cond_prompt_len` is always non-zero this never bites, but
  `mtl_tts.py` initialises it to `None` first and `tts_turbo.py` does not.

## `generate()`: when conditionals are (re)computed

- `audio_prompt_path` given -> `prepare_conditionals` runs **every call** (librosa load,
  resample, tokenizer, CAMPPlus, VoiceEncoder). For repeated generation with the same
  voice, call `prepare_conditionals` once and then call `generate(text)` without the path,
  or cache `Conditionals` objects via `save`/`load`.
- No path -> asserts `self.conds is not None` (built-in voice from `conds.pt` or a previous call).
- Llama variants: if `exaggeration` differs from the stored `emotion_adv`, a new `T3Cond` is
  built with the same `speaker_emb` and `cond_prompt_speech_tokens` (**dropping the cached
  `cond_prompt_speech_emb`**, so it is re-embedded once). `tts.py` compares
  `exaggeration != tensor` (tensor comparison, fine for scalars); `mtl_tts.py` compares floats via `.item()`.

## VoiceEncoder (`models/voice_encoder/`)

Adapted from CorentinJ's Real-Time-Voice-Cloning speaker encoder (GE2E-style).
- `VoiceEncConfig`: 40 mels, 16 kHz, n_fft 400, hop 160, win 400, fmin 0, fmax 8000,
  `mel_type="amp"`, `mel_power=2.0`, no preemphasis, `normalized_mels=False`,
  `ve_partial_frames=160` (1.6 s windows), `ve_final_relu=True`, hidden 256, embed 256.
- Network: `LSTM(40 -> 256, 3 layers, batch_first)` -> `Linear(256, 256)` -> ReLU -> L2-normalise.
  `similarity_weight`/`similarity_bias` parameters exist (training loss), unused at inference.
- `embeds_from_wavs(wavs, sample_rate, as_spk=False, batch_size=32, trim_top_db=20, rate=1.3)`:
  resample if needed (kaiser_fast), `librosa.effects.trim(top_db=20)` **trims leading/
  trailing silence**, mel via `melspec.melspectrogram` (librosa STFT, reflect pad), then
  `embeds_from_mels` -> `inference`: splits into overlapping 160-frame partials with
  `frame_step = round(16000 / 1.3 / 160) = 77` frames (~0.77 s hop), runs the LSTM on each,
  averages partial embeddings per utterance, L2-normalises. Returns numpy `(B, 256)` on CPU.
  Callers then `.mean(axis=0, keepdim=True)` (B is 1 anyway).
- Utterances shorter than one partial are zero-padded to 160 frames (`get_num_wins`).

## CAMPPlus x-vector (`models/s3gen/xvector.py`)

- `CAMPPlus(feat_dim=80, embedding_size=192, growth_rate=32, bn_size=4, init_channels=128, memory_efficient=False)`.
- `inference(audio_list)`: for each 16 kHz waveform tensor, `torchaudio.compliance.kaldi.fbank(num_mel_bins=80)`,
  subtract per-utterance mean, pad to batch, forward in float32 -> `(B, 192)`.
- Architecture: FCM (2-D conv front-end) -> TDNN -> 3 CAM-Dense-TDNN blocks (12/24/16 layers)
  with transit layers -> stats pooling -> dense 192. BatchNorm+ReLU everywhere.
- The result is L2-normalised inside `flow.inference` before `spk_embed_affine_layer`.

## Practical rules for reference clips (from code + README)

- Any sample rate/format librosa can read; mono is produced by librosa (stereo averaged).
- Turbo/Nano: **must be > 5 s** (assert). README recommends ~10 s clips.
- 6-10 s of clean, single-speaker speech at the start of the file is what actually gets used.
- Multilingual: the clip's language should match `language_id`, else the accent transfers;
  README says set `cfg_weight=0` to mitigate.
- Loudness: Turbo normalises to -27 LUFS by default; Llama variants do not normalise at all
  (clipping is only warned about by `mel_spectrogram`).
