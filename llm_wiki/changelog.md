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
