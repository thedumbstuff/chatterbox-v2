# 06 - Text tokenizers and normalisation

## Step 1: `punc_norm(text)` - three divergent copies

Defined separately in `tts.py`, `mtl_tts.py`, and `tts_turbo.py`. All three:
- return the literal sentence `"You need to add some text for me to talk."` for empty input;
- capitalise the first character if lowercase;
- collapse whitespace (`" ".join(text.split())`);
- apply a replacement list;
- strip trailing spaces and append `"."` if the text does not already end with a sentence ender.

Differences:

| | `tts.py` (original) | `mtl_tts.py` (multilingual) | `tts_turbo.py` (Turbo/Nano) |
|---|---|---|---|
| `"..."` -> `", "` | yes | yes | **no** |
| `"…"` -> `", "` | yes | yes | yes |
| `":"` -> `","` | yes | yes | yes |
| `" - "` -> `", "` | yes | yes | **no** |
| `";"` -> `", "` | yes | yes | **no** |
| em/en dash -> `"-"` | yes | yes | yes |
| `" ,"` -> `","`, smart quotes -> ASCII | yes | yes | yes |
| sentence enders | `. ! ? - ,` | `. ! ? - ,` **plus** `、 ， 。 ？ ！` | `. ! ? - ,` |

Turbo keeps `...`, ` - `, and `;` because its GPT-2 tokenizer and training data handle them.
If we unify these, keep the per-variant behaviour behind a flag; the models were trained on
their respective normalisation.

## Step 2a: `EnTokenizer` (original English) - `models/tokenizers/tokenizer.py`

- Wraps a HF `tokenizers.Tokenizer` loaded from `tokenizer.json` (vocab 704, character/grapheme level).
- Special tokens: `[START]` (SOT), `[STOP]` (EOT), `[UNK]`, `[SPACE]`, plus `[PAD] [SEP] [CLS] [MASK]`.
  Constructor asserts SOT and EOT exist in the vocab.
- `encode(txt)`: replaces `" "` with `[SPACE]` then `tokenizer.encode(txt).ids`. **No lowercasing,
  no unicode normalisation** (the punc_norm output is used as is).
- `text_to_tokens(text)` -> `IntTensor (1, T)`.
- `decode(seq)` reverses `[SPACE]`, strips `[STOP]` and `[UNK]`.
- The caller (`ChatterboxTTS.generate`) then pads `start_text_token=255` at the front and
  `stop_text_token=0` at the back. These ids are `T3Config` fields, not read from the vocab
  file; they must agree with `tokenizer.json` (255 = `[START]`, 0 = `[STOP]` presumably **(unverified)**).

## Step 2b: `MTLTokenizer` (multilingual)

- Same HF `tokenizers` wrapper over `grapheme_mtl_merged_expanded_v1.json` (vocab 2454).
- Constructor also builds `ChineseCangjieConverter(model_dir)`, which downloads
  `Cangjie5_TC.json` from `ResembleAI/chatterbox` via `hf_hub_download(cache_dir=model_dir)`
  (note: `cache_dir` is set to the checkpoint dir, so a nested HF cache may appear inside it)
  and initialises `spacy_pkuseg.pkuseg()` for word segmentation. Failures are logged as
  warnings, not raised.
- `encode(txt, language_id, lowercase=True, nfkd_normalize=True)` pipeline:
  1. `preprocess_text`: lowercase, then `unicodedata.normalize("NFKD", ...)`.
  2. Language-specific step (only for these five ids):
     - `zh`: `ChineseCangjieConverter.__call__`: pkuseg word segmentation (words joined by
       spaces), then every `Lo`-category character with a Cangjie code becomes
       `[cj_X][cj_Y]...[cj_.]` tokens (with a numeric disambiguation index appended to the
       code when several characters share it). Characters without a code pass through
       (e.g. kana).
     - `ja`: `hiragana_normalize`: `pykakasi` converts kanji words to hiragana (prepends a
       space when the reading starts with は/へ), leaves all-katakana words, then NFKD.
       Missing `pykakasi` -> warning, text unchanged.
     - `he`: `add_hebrew_diacritics` via optional `dicta_onnx.Dicta` (not in pyproject).
     - `ko`: `korean_normalize`: decomposes Hangul syllables into Jamo (U+1100 initial,
       U+1161 medial, U+11A7 final) arithmetically. No external dependency.
     - `ru`: `add_russian_stress` via optional `russian_text_stresser` (not in pyproject;
       upstream removed it as a hard dependency in PR #376/#377).
  3. Prepend the language tag token: `f"[{language_id}]"`, e.g. `[fr]bonjour`.
  4. Replace spaces with `[SPACE]`, encode.
- `language_id=None` is allowed by the tokenizer (no tag) and by `generate` (validation is
  `if language_id and ...`), but the model was trained with tags; expect degraded output.
- `SUPPORTED_LANGUAGES` (in `mtl_tts.py`) is the validation list: 23 codes
  `ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh`.
  `generate` raises `ValueError` for anything else (case-insensitive).
- Global module-level singletons `_kakasi`, `_dicta`, `_russian_stresser` are lazily created.

## Step 2c: Turbo / Nano - HF `AutoTokenizer`

- `AutoTokenizer.from_pretrained(ckpt_dir)`: GPT-2 BPE with added tokens, expected length 50276.
- `generate` calls `self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)`
  and uses `input_ids` directly. **No SOT/EOT padding**, no `_ensure_BOT_EOT` check.
  Whether the tokenizer itself adds BOS/EOS depends on the repo's `tokenizer_config.json`
  **(unverified; check `add_bos_token`/post-processor there)**.
- `truncation=True` without `max_length` truncates at the tokenizer's `model_max_length`
  (for GPT-2 that is 1024) -> texts longer than ~1024 BPE tokens are silently cut.
- Paralinguistic tags are plain text in the input: `[laugh]`, `[chuckle]`, `[cough]`,
  `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]` (list from
  `gradio_tts_turbo_app.py::EVENT_TAGS`). They presumably map to the added tokens.
  `punc_norm` does not touch square brackets, so they survive normalisation. Note that
  `punc_norm` capitalises the first character: a text starting with `[laugh]` becomes
  `[laugh]` unchanged (already not lowercase-alpha), fine.

## Length limits

- Gradio apps cap input at 300 characters (label text; `multilingual_app.py` also slices
  `text_input[:300]`). The library itself has no character limit; the practical limit is the
  1000-speech-token (40 s) generation cap and the model's stability on long inputs.
  Long texts should be sentence-split by the caller (see 14-extension-points).
- T3 `max_text_tokens=2048` only sizes the learned positional table; there is no check.
