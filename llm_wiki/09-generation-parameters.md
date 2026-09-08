# 09 - Generation parameters

## Support matrix

| Parameter | Original (`ChatterboxTTS`) | Multilingual | Turbo / Nano | Where it acts |
|-----------|---------------------------|--------------|--------------|---------------|
| `audio_prompt_path` | yes | yes | yes (> 5 s required) | `prepare_conditionals` |
| `exaggeration` | yes, default 0.5 | yes, default 0.5 | **ignored** (default 0.0, warns if > 0) | `T3Cond.emotion_adv` -> `emotion_adv_fc` prefix token |
| `cfg_weight` | yes, default 0.5 (**0 crashes, see G1**) | yes, default 0.5 (0 works) | **ignored** | `T3.inference` CFG mix `cond + w*(cond - uncond)` |
| `temperature` | 0.8 | 0.8 | 0.8 | logits / T before sampling |
| `top_p` | 1.0 (off) | 1.0 (off) | 0.95 | nucleus filter |
| `min_p` | 0.05 | 0.05 | **ignored** (default 0.0, warns if > 0) | `MinPLogitsWarper` |
| `top_k` | n/a | n/a | 1000 | `TopKLogitsWarper` |
| `repetition_penalty` | 1.2 | 1.2 | 1.2 | HF `RepetitionPenaltyLogitsProcessor` over generated speech tokens |
| `language_id` | n/a | required | n/a | tokenizer language tag |
| `norm_loudness` | n/a | n/a | True (-27 LUFS) | reference clip pre-processing |
| seed | not a parameter; apps call `torch.manual_seed` etc. | | | sampling in T3 and CFM noise |
| `n_cfm_timesteps` | not exposed (10) | not exposed (10) | not exposed (2) | `S3Gen.inference` |
| max length | not exposed (1000 tokens = 40 s) | same | same (`max_gen_len`) | `T3.inference` / `inference_turbo` |

## What each knob does mechanically

- **exaggeration** (Llama variants): scalar multiplied into `torch.ones(1,1,1)`, projected by a
  bias-free `Linear(1, 1024)` into one prefix token. Trained range is roughly 0.25-2.0 (the
  Gradio slider); 0.5 = neutral. README: higher values speed up and intensify speech;
  compensate with lower `cfg_weight`. `example_for_mac.py` uses 2.0.
- **cfg_weight** (Llama variants): classifier-free guidance on **text** conditioning
  (the unconditional branch has its text embeddings zeroed but keeps speaker/prompt/emotion).
  Higher = closer adherence to text/pacing, README calls it "CFG/Pace". README tips: 0.3 for
  fast speakers or expressive speech; 0 when the reference language differs from
  `language_id` (multilingual only, given G1).
- **temperature**: applied before min_p/top_p (Llama loop) or via `TemperatureLogitsWarper`
  first in the processor list (Turbo). `temperature=0` is **not** greedy in either loop:
  Llama loop divides by 0 -> inf/NaN logits (avoid); Turbo skips the warper and samples normally.
- **min_p**: keeps tokens with prob >= `min_p * max_prob`. Gradio note: 0.02-0.1 works,
  handles high temperatures better than top_p; 0 disables.
- **top_p**: nucleus sampling; 1.0 disables. Llama default off, Turbo 0.95.
- **top_k** (Turbo): 1000 of 6563 -> mostly off. 0 disables.
- **repetition_penalty**: penalises already-generated speech token ids (multiplicative on
  logits, HF semantics). 1.0 disables. Since speech tokens legitimately repeat (silence,
  sustained vowels), very high values cause artefacts; 1.2 default.
- **S3Gen CFG** (`inference_cfg_rate=0.7`) and **CFM steps** are fixed unless you edit code /
  pass `n_cfm_timesteps` to `s3gen.inference` directly. More steps = slower, marginally
  smoother mel for the 10-step model; the meanflow model was distilled for 1-2 steps.

## Defaults recap (upstream README "Original Chatterbox Tips")

- General: `exaggeration=0.5`, `cfg_weight=0.5` works for most prompts and languages.
- Fast reference speaker: lower `cfg_weight` to ~0.3.
- Dramatic: `cfg_weight ~0.3`, `exaggeration >= 0.7`.
- Reference language mismatch (multilingual): `cfg_weight=0`.

## Reproducibility

Sampling happens in T3 (`torch.multinomial`) and in the CFM initial noise (`torch.randn`).
Seed all of `torch`, `torch.cuda`, `random`, `numpy` (the apps' `set_seed`). The Perceiver's
`sdp_kernel` flash path and cuDNN can still introduce nondeterminism on GPU.

## Runtime cost intuition (not measured here)

- T3 dominates: one transformer step per 40 ms of audio, batch 2 for CFG (Llama variants).
  1000 steps max. Nano (12 layers, 768) is the CPU-viable one per README.
- S3Gen: encoder once, CFM estimator `n_steps * 2` (CFG) forward passes over the full mel
  (10 steps -> 20 U-Net passes; Turbo 2 steps -> 2 passes, no CFG), then HiFT once.
- `prepare_conditionals` adds librosa load/resample + tokenizer + CAMPPlus + VoiceEncoder
  (LSTM over the full clip) and runs every call when `audio_prompt_path` is passed.
