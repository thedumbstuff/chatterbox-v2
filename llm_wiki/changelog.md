# Changelog and decision record (ours)

Append-only. Newest at the bottom. Reference gotchas as G<n> and recipes as E<n>.
Format: date, who/what, files, why, follow-ups.

## 2026-09-08 - Wiki and Claude files created

- Cloned `thedumbstuff/chatterbox-v2` at upstream commit `5de7a54` (no fork-specific commits).
- Created `llm_wiki/` (this knowledge base, 16 pages) from a full read of `src/`, the
  examples, apps, `pyproject.toml`, CI and README.
- Created `CLAUDE.md` at the repo root and `src/chatterbox/models/CLAUDE.md`.
- No code changes yet. No venv yet (Python 3.13.1 system, RTX 4090 available).

### Decisions
- Wiki lives inside the repo (not in the workspace root) so it travels with the code and
  is visible to any Claude session whose cwd is this repo.
- Keep upstream file layout and public class names stable so README/apps keep working and
  upstream diffs stay mergeable. Refactors (E1) should be additive (new shared modules) first.

### Open follow-ups
- Set up `venv` and verify `pip install -e .` on Python 3.13 (see 11). If `spacy-pkuseg`
  fails to build, fall back to a Python 3.11 venv.
- Run each variant once to convert the **(unverified)** claims (notably G1) into verified ones.
- Add `venv/` to `.gitignore` before the first commit.

## 2026-09-08 - Environment set up, all variants verified, smoke test added

- Commit `e2cbd5e`: wiki + CLAUDE files + `.gitignore` (`venv/`, `.venv/`).
- Created `venv` with `uv venv --python 3.12` (Python 3.12.9), torch 2.6.0+cu124, `pip install -e .`.
  Every pinned dependency installed without issue on 3.12.
- Added `tests/smoke_generate.py`. Results (all OK, watermark detected on every output):
  nano, turbo, english (cfg 0.5), mtl v2/v3 fr, mtl v3 zh/ja, mtl v2/v3 hi, cloning for
  turbo/english/hi with a generated reference, VC. Warm speed table in wiki 09.
- **G1 verified**: `english --cfg 0.0` -> `RuntimeError` at `t3.py:313`; `mtl --cfg 0.0` works.
- G15 confirmed as a `FutureWarning` on torch 2.6 (still functional).
- User scope decision: only English and Hindi need testing going forward.
- Added a "Fork change log" one-liner section at the top of `README.md` (user request; keep updated).

### Open follow-ups
- Fix G1 (recipe E2). Decide whether to keep the double-BOS (G2) when doing so.
- Hindi built-in-voice outputs peak at ~0.99: consider a soft limiter or checking `S3Gen` output gain.
- Turn `tests/smoke_generate.py` into pytest cases (E15) once behaviour changes start.

## 2026-09-08 - G1 fixed (cfg_weight=0 crash)

- `src/chatterbox/models/t3/t3.py::T3.inference`: owns CFG batching now. `use_cfg = cfg_weight > 0`;
  single text row is repeated to `[cond, uncond]` when `use_cfg`; batch 1 otherwise; `bos_embed` and
  per-step embeddings use `.repeat(B, 1, 1)`; CFG mix only when `use_cfg`; clear `ValueError` for
  unsupported batch shapes. Removed a duplicated `top_p_warper` construction.
- `src/chatterbox/tts.py`, `src/chatterbox/mtl_tts.py`: removed caller-side token duplication.
- Verification: `tests/g1_equivalence.py` -> token sequences bit-identical to the old code for
  cfg 0.5 (batch 2) and cfg 0 (old batch-2 workaround vs new batch 1), temperature 0.05, seed 1234.
  Smoke runs pass: english cfg 0 / 0.5 / 0.3+clone, mtl-v3 hi cfg 0 / 0.5, mtl-v3 en cfg 0.
  Warm RTF english: cfg 0.5 -> 0.71, cfg 0 -> 0.65.
- Left G2 (double BOS) untouched on purpose: equivalence would break otherwise.
- Observation: english cfg 0 with the built-in voice peaked at 1.023 (clipping) on one run;
  the near-clipping follow-up from the previous entry stands.
