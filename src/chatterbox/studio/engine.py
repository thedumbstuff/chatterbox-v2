"""Model manager + generation helpers shared by the Studio UI and scripts.

- Lazy-loads each variant once and keeps it (all four fit in ~11 GiB of VRAM in fp32).
- Applies a library voice to a model, caching the computed Conditionals per (voice, model)
  so a saved voice is instant on every later use (upstream recomputes on every call, G7).
- Generates long text in sentence chunks joined with a short silence (G5: 40 s cap per call).
"""
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .library import DEFAULT_VOICE, Voice, VoiceLibrary
from .textsplit import split_text

logger = logging.getLogger(__name__)

SR = 24000


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str            # "turbo" | "english" | "mtl" | "vc"
    languages: bool        # has a language selector
    supports_exaggeration: bool
    supports_cfg: bool
    supports_min_p: bool
    supports_top_k: bool
    supports_loudness: bool
    needs_5s_ref: bool
    note: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "nano": ModelSpec("nano", "Nano (110M, English, fastest)", "turbo", False, False, False, False, True, True, True,
                      "English only. Supports tags like [laugh] [chuckle] [cough] [sigh]. Reference clip must be > 5 s."),
    "turbo": ModelSpec("turbo", "Turbo (350M, English, best quality/speed)", "turbo", False, False, False, False, True, True, True,
                       "English only. Supports tags like [laugh] [chuckle] [cough] [sigh]. Reference clip must be > 5 s."),
    "english": ModelSpec("english", "Chatterbox (500M, English, exaggeration/CFG controls)", "english", False, True, True, True, False, False, False,
                         "Original English model. Exaggeration 0.5 = neutral; CFG/pace 0.5 default, lower for fast speakers."),
    "mtl": ModelSpec("mtl", "Multilingual v3 (500M, Hindi + 22 languages)", "mtl", True, True, True, True, False, False, False,
                     "Pick the language. Reference clip should be in the same language, or set CFG to 0."),
}


@dataclass
class GenParams:
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    top_p: float = 1.0
    min_p: float = 0.05
    top_k: int = 1000
    repetition_penalty: float = 1.2
    norm_loudness: bool = True
    seed: int = 0
    watermark: bool = False
    split_long_text: bool = True
    max_chars: int = 300
    gap_ms: int = 250

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class GenResult:
    wav: torch.Tensor        # (1, N) float32 at 24 kHz
    sr: int
    gen_s: float
    chunks: list[str] = field(default_factory=list)
    chunk_times: list[float] = field(default_factory=list)

    @property
    def duration_s(self):
        return self.wav.shape[-1] / self.sr


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class Engine:
    def __init__(self, library: VoiceLibrary | None = None, device: str | None = None):
        self.library = library or VoiceLibrary()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, object] = {}
        self._builtin: dict[str, object] = {}     # built-in conds / ref_dict per model key
        self._active_voice: dict[str, str] = {}    # model key -> voice slug currently applied
        self.load_times: dict[str, float] = {}

    # ---------------- models ----------------
    def loaded(self) -> list[str]:
        return list(self._models)

    def unload(self, key: str):
        self._models.pop(key, None)
        self._builtin.pop(key, None)
        self._active_voice.pop(key, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get(self, key: str):
        if key in self._models:
            return self._models[key]
        t0 = time.time()
        if key in ("nano", "turbo"):
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            m = ChatterboxTurboTTS.from_pretrained(self.device, nano=(key == "nano"))
            self._builtin[key] = m.conds
        elif key == "english":
            from chatterbox.tts import ChatterboxTTS
            m = ChatterboxTTS.from_pretrained(self.device)
            self._builtin[key] = m.conds
        elif key == "mtl":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            m = ChatterboxMultilingualTTS.from_pretrained(self.device, t3_model="v3")
            self._builtin[key] = m.conds
        elif key == "vc":
            from chatterbox.vc import ChatterboxVC
            m = ChatterboxVC.from_pretrained(self.device)
            self._builtin[key] = m.ref_dict
        else:
            raise KeyError(key)
        self._models[key] = m
        self._active_voice[key] = DEFAULT_VOICE
        self.load_times[key] = time.time() - t0
        logger.info("loaded %s in %.1fs", key, self.load_times[key])
        return m

    # ---------------- voices ----------------
    def apply_voice(self, key: str, voice_slug: str | None, exaggeration: float = 0.5) -> str:
        """Point model `key` at a library voice (or the built-in default). Returns a status string."""
        m = self.get(key)
        slug = voice_slug or DEFAULT_VOICE
        if slug == DEFAULT_VOICE:
            if key == "vc":
                m.ref_dict = self._builtin[key]
            else:
                m.conds = self._builtin[key]
            self._active_voice[key] = slug
            return "built-in voice"
        v = self.library.get(slug)
        if v is None:
            raise KeyError(f"voice '{slug}' not in library")
        if MODEL_SPECS.get(key, None) and MODEL_SPECS[key].needs_5s_ref and not v.turbo_ok:
            raise ValueError(f"'{v.name}' is {v.duration_s:.1f}s; Turbo/Nano need a reference longer than 5 s.")
        if key == "vc":
            p = self.library.refdict_path(slug, key)
            if p.exists():
                rd = torch.load(p, map_location="cpu", weights_only=True)
                m.ref_dict = {k: (t.to(self.device) if torch.is_tensor(t) else t) for k, t in rd.items()}
                status = "cached ref"
            else:
                m.set_target_voice(str(v.ref_path))
                torch.save({k: (t.detach().cpu() if torch.is_tensor(t) else t) for k, t in m.ref_dict.items()}, p)
                status = "computed ref (cached for next time)"
        else:
            p = self.library.conds_path(slug, key)
            conds_cls = type(self._builtin[key]) if self._builtin.get(key) is not None else None
            if p.exists() and conds_cls is not None:
                m.conds = conds_cls.load(p, map_location="cpu").to(self.device)
                status = "cached conditionals"
            else:
                with torch.inference_mode():
                    m.prepare_conditionals(str(v.ref_path), exaggeration=exaggeration)
                try:
                    m.conds.save(p)
                except Exception as e:  # noqa: BLE001
                    logger.warning("could not cache conditionals: %s", e)
                status = "computed conditionals (cached for next time)"
        self._active_voice[key] = slug
        return f"{v.name}: {status}"

    # ---------------- generation ----------------
    def generate(self, key: str, text: str, voice_slug: str | None, lang: str | None, params: GenParams,
                 progress: Callable[[float, str], None] | None = None) -> GenResult:
        spec = MODEL_SPECS[key]
        m = self.get(key)
        voice_status = self.apply_voice(key, voice_slug, params.exaggeration)
        if params.seed:
            set_seed(params.seed)
        chunks = split_text(text, params.max_chars) if params.split_long_text else [" ".join(text.split())]
        if not chunks:
            raise ValueError("Please enter some text.")

        kwargs = dict(temperature=params.temperature, top_p=params.top_p,
                      repetition_penalty=params.repetition_penalty, watermark=params.watermark)
        if spec.family == "turbo":
            kwargs.update(top_k=int(params.top_k), norm_loudness=params.norm_loudness)
        else:
            kwargs.update(exaggeration=params.exaggeration, cfg_weight=params.cfg_weight, min_p=params.min_p)
        if spec.family == "mtl":
            kwargs["language_id"] = (lang or "en").lower()

        pieces, times = [], []
        gap = torch.zeros(1, int(SR * params.gap_ms / 1000))
        t_all = time.time()
        for i, chunk in enumerate(chunks):
            if progress:
                progress(i / len(chunks), f"chunk {i + 1}/{len(chunks)}")
            t0 = time.time()
            wav = m.generate(chunk, **kwargs)
            times.append(time.time() - t0)
            pieces.append(wav.float().cpu())
            if i < len(chunks) - 1 and params.gap_ms > 0:
                pieces.append(gap)
        wav = torch.cat(pieces, dim=-1)
        res = GenResult(wav=wav, sr=SR, gen_s=time.time() - t_all, chunks=chunks, chunk_times=times)
        logger.info("%s | %s | %d chunk(s) | %.2fs audio in %.2fs", key, voice_status, len(chunks), res.duration_s, res.gen_s)
        return res

    def convert(self, source_path: str, voice_slug: str | None, watermark: bool = False) -> GenResult:
        m = self.get("vc")
        self.apply_voice("vc", voice_slug)
        t0 = time.time()
        wav = m.generate(source_path, watermark=watermark).float().cpu()
        return GenResult(wav=wav, sr=SR, gen_s=time.time() - t0, chunks=[source_path], chunk_times=[time.time() - t0])
