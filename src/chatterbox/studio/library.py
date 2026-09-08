"""Persistent voice library and generation history (plain files, no database).

Layout (defaults, overridable via env):
    CHATTERBOX_VOICES_DIR   (default <repo>/voices)
        <slug>/ref.wav            mono reference clip at its native sample rate
        <slug>/meta.json          {name, slug, added, duration_s, sr, source}
        <slug>/conds_<model>.pt   cached conditionals per model (built on first use)
    CHATTERBOX_HISTORY_DIR  (default <repo>/syn_out/history)
        <id>.wav + index.json
"""
import json
import os
import re
import shutil
import time
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VOICE = "__default__"          # sentinel for each model's built-in voice (conds.pt)
MIN_VOICE_SECONDS = 1.0
TURBO_MIN_SECONDS = 5.0                # ChatterboxTurboTTS asserts the reference is > 5 s


def _slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or f"voice-{int(time.time())}"


@dataclass
class Voice:
    name: str
    slug: str
    added: str
    duration_s: float
    sr: int
    source: str = ""

    @property
    def dir(self) -> Path:
        return VoiceLibrary.current_root() / self.slug

    @property
    def ref_path(self) -> Path:
        return self.dir / "ref.wav"

    @property
    def turbo_ok(self) -> bool:
        return self.duration_s > TURBO_MIN_SECONDS

    def label(self) -> str:
        flag = "" if self.turbo_ok else " (too short for Turbo/Nano)"
        return f"{self.name}  ·  {self.duration_s:.1f}s{flag}"


class VoiceLibrary:
    _root_override: Path | None = None

    def __init__(self, root: str | os.PathLike | None = None):
        root = Path(root) if root else Path(os.environ.get("CHATTERBOX_VOICES_DIR", REPO_ROOT / "voices"))
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        VoiceLibrary._root_override = self.root

    @classmethod
    def current_root(cls) -> Path:
        return cls._root_override or Path(os.environ.get("CHATTERBOX_VOICES_DIR", REPO_ROOT / "voices"))

    # ---- CRUD ----
    def list(self) -> list[Voice]:
        voices = []
        for d in sorted(self.root.iterdir()):
            meta = d / "meta.json"
            if d.is_dir() and meta.exists() and (d / "ref.wav").exists():
                try:
                    voices.append(Voice(**json.loads(meta.read_text(encoding="utf-8"))))
                except (TypeError, ValueError):
                    continue
        return sorted(voices, key=lambda v: v.name.lower())

    def get(self, slug: str) -> Voice | None:
        return next((v for v in self.list() if v.slug == slug), None)

    def add(self, src_path: str | os.PathLike, name: str, overwrite: bool = False) -> Voice:
        """Import an audio file (anything librosa reads) as a mono wav at its native rate."""
        import librosa
        import soundfile as sf

        name = (name or "").strip() or Path(src_path).stem
        slug = _slugify(name)
        vdir = self.root / slug
        if vdir.exists() and not overwrite:
            raise FileExistsError(f"A voice named '{name}' already exists (slug '{slug}').")
        y, sr = librosa.load(str(src_path), sr=None, mono=True)
        dur = len(y) / sr
        if dur < MIN_VOICE_SECONDS:
            raise ValueError(f"Clip is {dur:.2f}s; need at least {MIN_VOICE_SECONDS:.0f}s of speech.")
        peak = float(np.abs(y).max()) if len(y) else 0.0
        if peak > 1.0:
            y = y / peak
        vdir.mkdir(parents=True, exist_ok=True)
        sf.write(str(vdir / "ref.wav"), y.astype(np.float32), sr, subtype="FLOAT")
        v = Voice(name=name, slug=slug, added=time.strftime("%Y-%m-%d %H:%M:%S"),
                  duration_s=round(dur, 2), sr=int(sr), source=Path(src_path).name)
        (vdir / "meta.json").write_text(json.dumps(asdict(v), ensure_ascii=False, indent=2), encoding="utf-8")
        # any cached conditionals from an overwritten voice are stale
        for p in vdir.glob("conds_*.pt"):
            p.unlink()
        for p in vdir.glob("refdict_*.pt"):
            p.unlink()
        return v

    def delete(self, slug: str) -> bool:
        vdir = self.root / slug
        if vdir.is_dir():
            shutil.rmtree(vdir)
            return True
        return False

    def rename(self, slug: str, new_name: str) -> Voice:
        v = self.get(slug)
        if v is None:
            raise KeyError(slug)
        v.name = new_name.strip() or v.name
        (v.dir / "meta.json").write_text(json.dumps(asdict(v), ensure_ascii=False, indent=2), encoding="utf-8")
        return v

    # ---- caches ----
    def conds_path(self, slug: str, model_key: str) -> Path:
        return self.root / slug / f"conds_{model_key}.pt"

    def refdict_path(self, slug: str, model_key: str) -> Path:
        return self.root / slug / f"refdict_{model_key}.pt"


@dataclass
class HistoryEntry:
    id: str
    ts: str
    model: str
    voice: str
    lang: str
    text: str
    file: str
    duration_s: float
    gen_s: float
    params: dict


class History:
    def __init__(self, root: str | os.PathLike | None = None, keep: int = 300):
        self.root = Path(root) if root else Path(os.environ.get("CHATTERBOX_HISTORY_DIR", REPO_ROOT / "syn_out" / "history"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.json"
        self.keep = keep

    def _load(self) -> list[dict]:
        if not self.index.exists():
            return []
        try:
            return json.loads(self.index.read_text(encoding="utf-8"))
        except ValueError:
            return []

    def _save(self, entries: list[dict]):
        self.index.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")

    def add(self, wav, sr: int, *, model: str, voice: str, lang: str, text: str, gen_s: float, params: dict) -> HistoryEntry:
        import torch
        import torchaudio as ta

        eid = time.strftime("%Y%m%d-%H%M%S") + f"-{model}"
        path = self.root / f"{eid}.wav"
        n = 2
        while path.exists():
            path = self.root / f"{eid}-{n}.wav"
            n += 1
        wav_t = wav if isinstance(wav, torch.Tensor) else torch.from_numpy(np.asarray(wav))
        if wav_t.ndim == 1:
            wav_t = wav_t[None]
        ta.save(str(path), wav_t.cpu(), sr)
        e = HistoryEntry(id=path.stem, ts=time.strftime("%Y-%m-%d %H:%M:%S"), model=model, voice=voice, lang=lang,
                         text=text, file=str(path), duration_s=round(wav_t.shape[-1] / sr, 2), gen_s=round(gen_s, 2),
                         params=params)
        entries = self._load()
        entries.insert(0, asdict(e))
        # prune old files beyond `keep`
        for old in entries[self.keep:]:
            try:
                Path(old["file"]).unlink(missing_ok=True)
            except OSError:
                pass
        self._save(entries[: self.keep])
        return e

    def list(self, limit: int = 100) -> list[dict]:
        return [e for e in self._load() if Path(e["file"]).exists()][:limit]

    def delete(self, entry_id: str) -> bool:
        entries = self._load()
        keep = []
        removed = False
        for e in entries:
            if e["id"] == entry_id:
                Path(e["file"]).unlink(missing_ok=True)
                removed = True
            else:
                keep.append(e)
        self._save(keep)
        return removed

    def clear(self):
        for e in self._load():
            Path(e["file"]).unlink(missing_ok=True)
        self._save([])
