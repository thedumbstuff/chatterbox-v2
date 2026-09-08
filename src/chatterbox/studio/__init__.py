"""Chatterbox Studio: model manager, persistent voice library, generation history, long-text chunking.

UI-independent; `app.py` at the repo root builds the Gradio interface on top of this.
"""
from .textsplit import split_text
from .library import VoiceLibrary, Voice, History
from .engine import Engine, MODEL_SPECS

__all__ = ["split_text", "VoiceLibrary", "Voice", "History", "Engine", "MODEL_SPECS"]
