"""Optional Perth audio watermarking.

Upstream Chatterbox applies Resemble's Perth implicit watermark to every generated waveform.
This fork turns it OFF by default (see README "Watermarking" for the measurements) and keeps
it available behind a flag: pass ``watermark=True`` to ``from_pretrained``/``from_local`` or
to ``generate``. The ``perth`` package is only imported when the watermark is actually used.
"""
import numpy as np

_WATERMARKER = None


def get_watermarker():
    """Lazily construct the (CPU, ~9M parameter) Perth watermarker. Raises if perth is missing."""
    global _WATERMARKER
    if _WATERMARKER is None:
        try:
            import perth
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Watermarking requested but the 'perth' package is not installed. "
                "Install it with: pip install 'resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master'"
            ) from e
        _WATERMARKER = perth.PerthImplicitWatermarker()
    return _WATERMARKER


def maybe_watermark(wav: np.ndarray, sample_rate: int, enabled: bool) -> np.ndarray:
    """Return ``wav`` unchanged when ``enabled`` is False, else the Perth-watermarked waveform."""
    if not enabled:
        return wav
    return get_watermarker().apply_watermark(wav, sample_rate=sample_rate)


def resolve(default: bool, override):
    """Per-call override semantics: ``None`` means use the instance default."""
    return default if override is None else bool(override)
