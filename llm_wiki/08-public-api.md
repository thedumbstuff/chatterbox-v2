# 08 - Public API reference

Package exports (`src/chatterbox/__init__.py`): `ChatterboxTTS`, `ChatterboxVC`,
`ChatterboxMultilingualTTS`, `SUPPORTED_LANGUAGES`, `__version__` (read from installed
package metadata via `importlib.metadata.version("chatterbox-tts")`, so the package **must be
pip-installed** (`pip install -e .`) or the import fails). `ChatterboxTurboTTS` is **not**
re-exported; import it from `chatterbox.tts_turbo`.

All `generate` methods return `torch.FloatTensor` of shape `(1, n_samples)` at `model.sr == 24000`,
already watermarked. Save with `torchaudio.save(path, wav, model.sr)`.

## `chatterbox.tts.ChatterboxTTS` (original English)

```python
ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=None)
ChatterboxTTS.from_pretrained(device) -> ChatterboxTTS
ChatterboxTTS.from_local(ckpt_dir, device) -> ChatterboxTTS
.prepare_conditionals(wav_fpath, exaggeration=0.5) -> None      # sets self.conds
.generate(text, repetition_penalty=1.2, min_p=0.05, top_p=1.0, audio_prompt_path=None,
          exaggeration=0.5, cfg_weight=0.5, temperature=0.8) -> Tensor(1, N)
attributes: sr=24000, t3, s3gen, ve, tokenizer (EnTokenizer), device, conds (Conditionals|None), watermarker
class constants: ENC_COND_LEN = 6*16000, DEC_COND_LEN = 10*24000
```
Flow inside `generate`: conditionals -> exaggeration refresh -> `punc_norm` -> tokenize ->
SOT/EOT pad -> `t3.inference(max_new_tokens=1000, ...)` (T3 adds the CFG row itself when `cfg_weight > 0`)
-> `[0]` -> `drop_invalid_tokens` -> `< 6561` filter -> `s3gen.inference` -> watermark.

## `chatterbox.mtl_tts.ChatterboxMultilingualTTS`

```python
ChatterboxMultilingualTTS.from_pretrained(device, t3_model: str | None = None)
ChatterboxMultilingualTTS.from_local(ckpt_dir, device, t3_model: str | None = None)
ChatterboxMultilingualTTS.get_supported_languages() -> dict[code, name]   # copy of SUPPORTED_LANGUAGES
.prepare_conditionals(wav_fpath, exaggeration=0.5)
.generate(text, language_id, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5,
          temperature=0.8, repetition_penalty=1.2, min_p=0.05, top_p=1.0) -> Tensor(1, N)
module constants: SUPPORTED_LANGUAGES, MULTILINGUAL_T3_MODELS, DEFAULT_MULTILINGUAL_T3_MODEL
```
Differences from `ChatterboxTTS`: `language_id` is **positional-required** (2nd arg) and
validated; trims the last token's audio (960 samples); loads `.pt` weights for VE/S3Gen;
uses `MTLTokenizer`.

## `chatterbox.tts_turbo.ChatterboxTurboTTS` (Turbo and Nano)

```python
ChatterboxTurboTTS.from_pretrained(device, nano=False)
ChatterboxTurboTTS.from_local(ckpt_dir, device, nano=False)
.norm_loudness(wav, sr, target_lufs=-27) -> wav
.prepare_conditionals(wav_fpath, exaggeration=0.5, norm_loudness=True)
.generate(text, repetition_penalty=1.2, min_p=0.00, top_p=0.95, audio_prompt_path=None,
          exaggeration=0.0, cfg_weight=0.0, temperature=0.8, top_k=1000, norm_loudness=True) -> Tensor(1, N)
attributes: model_label ("Turbo" | "Nano"), tokenizer is a HF PreTrainedTokenizer
class constants: ENC_COND_LEN = 15*16000, DEC_COND_LEN = 10*24000
```
`cfg_weight`, `exaggeration`, `min_p` are accepted for signature compatibility and
**ignored** (a warning is logged if any is > 0). Uses `t3.inference_turbo(...)`, appends
3 silence tokens, `s3gen.inference(..., n_cfm_timesteps=2)`. Not wrapped in
`torch.inference_mode()` at this level (the inner methods are decorated).

## `chatterbox.vc.ChatterboxVC` (voice conversion)

```python
ChatterboxVC(s3gen, device, ref_dict=None)
ChatterboxVC.from_pretrained(device)
ChatterboxVC.from_local(ckpt_dir, device)
.set_target_voice(wav_fpath) -> None            # ref_dict from first 10 s
.generate(audio, target_voice_path=None) -> Tensor(1, N)   # audio = path to source speech
```
Source audio is loaded at 16 kHz, tokenized with `s3gen.tokenizer`, and re-synthesised with
the target `ref_dict`. Output length = 960 * n_tokens (roughly the source length rounded to
40 ms). No T3, no text.

## `chatterbox.models.t3.T3` (lower level)

```python
T3(hp: T3Config | None = None)
.inference(*, t3_cond, text_tokens, max_new_tokens=None, temperature=0.8, top_p=0.95, min_p=0.05,
           repetition_penalty=1.2, cfg_weight=0.5, ...ignored...) -> LongTensor(1, n)   # Llama variants
           # text_tokens: 1 row (T3 repeats it for CFG when cfg_weight > 0) or 2 rows [cond, uncond]
.inference_turbo(t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95,
                 repetition_penalty=1.2, max_gen_len=1000) -> LongTensor(1, n)          # GPT-2 variants
.prepare_conditioning(t3_cond) -> Tensor(B, L_cond, dim)
.prepare_input_embeds(*, t3_cond, text_tokens, speech_tokens, cfg_weight=0.0) -> (embeds, len_cond)
.forward(*, t3_cond, text_tokens, text_token_lens, speech_tokens, speech_token_lens, training=False) -> AttrDict
.loss(...) -> (loss_text, loss_speech)
.device (property)
```

## `chatterbox.models.s3gen.S3Gen` (= `S3Token2Wav`)

```python
S3Gen(meanflow=False)
.embed_ref(ref_wav, ref_sr, device="auto", ref_fade_out=True) -> ref_dict
.inference(speech_tokens, ref_wav=None, ref_sr=None, ref_dict=None, drop_invalid_tokens=True,
           n_cfm_timesteps=None, speech_token_lens=None) -> (wav (1, N), source)
.flow_inference(...) -> mel (1, 80, T)
.hift_inference(speech_feat, cache_source=None) -> (wav, source)
.tokenizer (S3Tokenizer), .speaker_encoder (CAMPPlus), .flow, .mel2wav (HiFTGenerator)
.device, .dtype (properties)
```

## `chatterbox.models.s3tokenizer`

`S3Tokenizer`, `S3_SR`, `S3_HOP`, `S3_TOKEN_HOP`, `S3_TOKEN_RATE`, `SPEECH_VOCAB_SIZE`, `SOS`, `EOS`, `drop_invalid_tokens(x)`.
`S3Tokenizer.forward(wavs: list, accelerator=None, max_len=None) -> (tokens, lens)`.

## `chatterbox.models.tokenizers`

`EnTokenizer(vocab_file_path)`: `.text_to_tokens(text)`, `.encode(txt)`, `.decode(seq)`.
`MTLTokenizer(vocab_file_path)`: `.text_to_tokens(text, language_id=None, lowercase=True, nfkd_normalize=True)`, `.encode(...)`, `.decode(seq)`, `.preprocess_text(...)`.

## `chatterbox.models.voice_encoder`

`VoiceEncoder(hp=VoiceEncConfig())`: `.embeds_from_wavs(wavs, sample_rate, as_spk=False, batch_size=32, trim_top_db=20, **kw) -> np.ndarray`,
`.embeds_from_mels(...)`, `.inference(mels, mel_lens, overlap=0.5, rate=None, min_coverage=0.8, batch_size=None)`,
static `utt_to_spk_embed`, `voice_similarity`.

## Minimal usage recipes

```python
# English, built-in voice
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained("cuda"); wav = m.generate("Hello."); ta.save("o.wav", wav, m.sr)

# Same voice, many lines (avoid recomputing conditionals)
m.prepare_conditionals("ref.wav", exaggeration=0.5)
for line in lines: wavs.append(m.generate(line))

# Multilingual v3
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
mm = ChatterboxMultilingualTTS.from_pretrained("cuda", t3_model="v3")
wav = mm.generate("Bonjour.", language_id="fr", audio_prompt_path="ref_fr.wav")

# Turbo / Nano with tags
from chatterbox.tts_turbo import ChatterboxTurboTTS
t = ChatterboxTurboTTS.from_pretrained("cuda", nano=True)
wav = t.generate("Well [chuckle] that was fun.", audio_prompt_path="ref10s.wav")

# Voice conversion
from chatterbox.vc import ChatterboxVC
vc = ChatterboxVC.from_pretrained("cuda"); wav = vc.generate("src.wav", target_voice_path="target.wav")
```
