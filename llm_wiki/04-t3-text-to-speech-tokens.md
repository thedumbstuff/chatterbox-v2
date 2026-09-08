# 04 - T3: text -> speech tokens

File: `src/chatterbox/models/t3/t3.py` (class `T3`), plus `modules/` and `inference/`.

T3 ("Token-To-Token") is a decoder-only transformer whose **input embeddings are built
outside the backbone** (the backbone is called with `inputs_embeds=`, never `input_ids=`).
Text and speech each have their own embedding table and their own output head. The
backbone's native token embedding is unused (Llama config sets `vocab_size=8`; Turbo deletes
GPT-2's `wte`).

## Module inventory (`T3.__init__`)

| Attribute | Type | Purpose |
|-----------|------|---------|
| `hp` | `T3Config` | hyperparameters (see 03) |
| `cfg` | `LlamaConfig` or `GPT2Config` | built from `LLAMA_CONFIGS[hp.llama_config_name]` |
| `tfmr` | `LlamaModel` or `GPT2Model` | backbone; `is_gpt = config["model_type"] == "gpt2"` |
| `cond_enc` | `T3CondEnc` | speaker / prompt / emotion prefix (see below) |
| `text_emb` | `nn.Embedding(text_tokens_dict_size, dim)` | text token table |
| `speech_emb` | `nn.Embedding(speech_tokens_dict_size, dim)` | speech token table (also embeds the speech prompt) |
| `text_pos_emb`, `speech_pos_emb` | `LearnedPositionEmbeddings` or `None` | only when `hp.input_pos_emb == "learned"` (Llama variants). Sizes `max_text_tokens+2`, `max_speech_tokens+4` |
| `text_head` | `Linear(dim, text_vocab, bias=False)` | only used in training `forward` |
| `speech_head` | `Linear(dim, speech_vocab, bias=is_gpt)` | logits over speech tokens |
| `patched_model` | `T3HuggingfaceBackend` | created lazily in `inference` (Llama path) |
| `compiled`, `deepspeed_patch_applied` | flags | `compiled` is reset to False on every `inference` call, so the backend is rebuilt every call (cheap: it only wraps existing modules) |

Backbone configs (`llama_configs.py`):
- `Llama_520M`: hidden 1024, 30 layers, 16 heads (head_dim 64, 16 kv heads = MHA), intermediate 4096,
  SiLU, RMSNorm eps 1e-5, RoPE theta 500k with llama3 scaling (factor 8, orig 8192), `attn_implementation="sdpa"`,
  `max_position_embeddings=131072`, `torch_dtype="bfloat16"` in config but weights are loaded fp32.
- `GPT2_medium`: n_embd 1024, 24 layers, 16 heads, `n_positions=8196`, vocab 50276, dropouts 0.1 (inactive in eval).
- `GPT2_small`: n_embd 768, 12 layers, 12 heads, otherwise same.

## Conditioning: `T3Cond` and `T3CondEnc` (`modules/cond_enc.py`)

```python
@dataclass
class T3Cond:
    speaker_emb: Tensor                       # (B, 256) from VoiceEncoder
    clap_emb: Optional[Tensor] = None         # must stay None (asserted "not implemented")
    cond_prompt_speech_tokens: Optional[Tensor] = None   # (B, <=speech_cond_prompt_len) S3 tokens of the reference
    cond_prompt_speech_emb: Optional[Tensor] = None      # filled in by T3.prepare_conditioning
    emotion_adv: Optional[Tensor] = 0.5       # (B,1,1) exaggeration; default is a float, callers always pass a tensor
```
- `T3Cond.to(device=, dtype=)` moves all tensor fields; dtype cast skipped for integer tensors
  (detects via `type(v.view(-1)[0].item()) is not int`, which is a per-call GPU sync).
- `T3.prepare_conditioning(t3_cond)` embeds `cond_prompt_speech_tokens` with `speech_emb`
  (+ `speech_pos_emb` for non-GPT) **and mutates `t3_cond.cond_prompt_speech_emb` in place**,
  then calls `cond_enc`. Because the callers keep one `Conditionals` object across `generate`
  calls, the embedding is computed once and reused (`if ... is None`). If `speech_emb`
  weights ever change at runtime, stale embeddings would persist.
- `T3CondEnc.forward` output = concat of
  `spkr_enc(speaker_emb)` (B,1,dim) | clap (B,0,dim) | prompt emb (B,32,dim after Perceiver, or raw) | `emotion_adv_fc(emotion_adv)` (B,1,dim).

Perceiver (`modules/perceiver.py`): 32 learned queries (1x32x1024), one `AttentionBlock2`
(4 heads, LayerNorm, separate q/k/v Linear, `F.scaled_dot_product_attention` via
`torch.backends.cuda.sdp_kernel`) applied twice: cross-attention queries->prompt, then
self-attention. `sdp_kernel` is deprecated in newer torch (still works in 2.6 with a warning).

## Input assembly: `prepare_input_embeds`

```
cond_emb  = prepare_conditioning(t3_cond)         # (B_c, L_cond, dim)   B_c is 1
text_emb  = text_emb(text_tokens)                 # (B, L_text, dim)
if cfg_weight > 0 and not is_gpt: text_emb[1].zero_()   # row 1 = unconditional branch
speech_emb = speech_emb(speech_tokens)            # (B, L_speech, dim)  (just BOS at inference)
if learned pos emb: add text_pos_emb / speech_pos_emb (position index restarts at 0 for each segment)
cond_emb expanded to B if needed
embeds = per-row concat [cond | text | speech]    # (B, L_cond+L_text+L_speech, dim)
return embeds, len_cond
```
Because positional embeddings restart per segment (text at 0, speech at 0) and Llama uses
RoPE inside the backbone as well, the learned embeddings act as segment-local offsets on
top of RoPE. The class docstring warns the design assumes relative PE.

## Inference loop A: `T3.inference` (Llama variants: original + multilingual)

Signature (keyword-only):
`inference(*, t3_cond, text_tokens, initial_speech_tokens=None, prepend_prompt_speech_tokens=None, num_return_sequences=1, max_new_tokens=None, stop_on_eos=True, do_sample=True, temperature=0.8, top_p=0.95, min_p=0.05, length_penalty=1.0, repetition_penalty=1.2, cfg_weight=0.5)`

What it actually does (several parameters are **ignored**: `num_return_sequences`,
`stop_on_eos`, `do_sample`, `length_penalty`, and `initial_speech_tokens` beyond its shape):

1. `_ensure_BOT_EOT` asserts every row contains `start_text_token` (255) and `stop_text_token` (0).
   Callers pad SOT at the front and EOT at the back with `F.pad`.
2. Builds `embeds` via `prepare_input_embeds` with a single BOS speech token per row.
3. Wraps `tfmr` in `T3HuggingfaceBackend` (`inference/t3_hf_backend.py`): a
   `LlamaPreTrainedModel + GenerationMixin` subclass whose `forward(inputs_embeds, past_key_values, ...)`
   runs the backbone and applies `speech_head` to the last hidden layer. It asserts you
   never pass a multi-token input together with a non-empty cache.
4. **Hard-codes batch 2 for CFG**: `bos_embed = cat([bos_embed, bos_embed])` (with
   `speech_pos_emb.get_fixed_embedding(0)` added), then
   `inputs_embeds = cat([embeds, bos_embed], dim=1)` -> requires `embeds` to have batch 2.
   Note the BOS is therefore present **twice** in the sequence: once from
   `initial_speech_tokens` inside `embeds`, once appended here. This is how upstream ships;
   changing it changes model behaviour.
5. Initial forward pass on the full prefix (`past_key_values=None`), then a loop up to
   `max_new_tokens` (callers pass 1000):
   - `logits = cond + cfg_weight * (cond - uncond)` from rows 0 and 1.
   - `RepetitionPenaltyLogitsProcessor(repetition_penalty)` over `generated_ids` (starts with BOS).
   - divide by `temperature` (if != 1.0), then `MinPLogitsWarper(min_p)`, then `TopPLogitsWarper(top_p)`.
   - `torch.multinomial` sample; append; **break on `stop_speech_token`** (the EOS is included in the output).
   - embed the new token + `speech_pos_emb.get_fixed_embedding(i + 1)`, duplicate for batch 2, step with cache.
6. Returns `(1, n_generated)` tokens (predicted only, BOS excluded, EOS included if hit).
   `tqdm` progress bar "Sampling" is shown.

Callers then apply `drop_invalid_tokens` (strip up to and including SOS, strip from EOS) and
`speech_tokens < 6561` filter, before S3Gen.

## Inference loop B: `T3.inference_turbo` (Turbo / Nano, GPT-2)

`inference_turbo(t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2, max_gen_len=1000)`

- Builds a HF `LogitsProcessorList`: `TemperatureLogitsWarper` (if temperature not in {0, 1}),
  `TopKLogitsWarper` (if top_k > 0), `TopPLogitsWarper` (if top_p < 1), `RepetitionPenaltyLogitsProcessor` (if != 1).
- No SOT/EOT check, no CFG, batch 1. Calls `self.tfmr(inputs_embeds=..., use_cache=True)`
  directly (no backend wrapper) and applies `speech_head` itself.
- First token sampled from the prefix pass; then loop `max_gen_len` steps feeding
  `speech_emb(current_token)` (no explicit positional embedding: GPT-2 adds `wpe` internally
  using the cache length).
- Breaks if all processed logits are `-inf` (prints a warning) or on `stop_speech_token`.
- **Strips the trailing EOS** before returning (unlike loop A).
- `temperature=0` is not greedy decoding here: the temperature warper is skipped and
  `multinomial` still samples from the raw softmax. If greedy is wanted it must be added.

## Training-time methods (unused here)

`forward(...)` splices text and speech latents out of the hidden states and returns
`AttrDict(text_logits, text_latents, speech_logits, speech_latents, hidden_states)`.
`loss(...)` computes cross-entropy on both heads with padding masked (`IGNORE_ID=-100`).
These are the natural starting point for fine-tuning; note `forward` does not shift
targets (next-token alignment would need to be handled by the caller as in upstream
training code, which is not in this repo).
