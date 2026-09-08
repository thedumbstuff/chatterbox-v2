"""Split long text into chunks the models handle well (they are capped at 40 s of audio per call)."""
import re

# Sentence enders: Latin, plus Devanagari danda, CJK full stops and question/exclamation marks.
_SENTENCE_END = re.compile(r"(?<=[.!?।॥。！？])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:，、])\s*")


def _pack(units, max_chars):
    chunks, cur = [], ""
    for u in units:
        u = u.strip()
        if not u:
            continue
        if not cur:
            cur = u
        elif len(cur) + 1 + len(u) <= max_chars:
            cur = f"{cur} {u}"
        else:
            chunks.append(cur)
            cur = u
    if cur:
        chunks.append(cur)
    return chunks


def _hard_split(s, max_chars):
    words, out, cur = s.split(), [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur = f"{cur} {w}"
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def split_text(text: str, max_chars: int = 300) -> list[str]:
    """Return chunks of at most `max_chars` characters, cut on sentence boundaries first,
    then clause boundaries, then words. Whitespace is normalised; empty input -> []."""
    text = " ".join(text.replace("\r", "\n").split(" ")).strip()
    if not text:
        return []
    if len(text) <= max_chars and "\n" not in text:
        return [text]
    sentences = [s for s in _SENTENCE_END.split(text) if s and s.strip()]
    units = []
    for s in sentences:
        s = " ".join(s.split())
        if len(s) <= max_chars:
            units.append(s)
            continue
        for c in _CLAUSE_SPLIT.split(s):
            if not c.strip():
                continue
            if len(c) <= max_chars:
                units.append(c)
            else:
                units.extend(_hard_split(c, max_chars))
    return _pack(units, max_chars)
