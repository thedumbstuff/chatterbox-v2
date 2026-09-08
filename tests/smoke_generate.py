"""Smoke test: load a Chatterbox variant, generate one sentence, report timings and checks.

Usage (from repo root, inside the venv):
    python tests/smoke_generate.py nano
    python tests/smoke_generate.py turbo --ref path/to/ref.wav
    python tests/smoke_generate.py english --cfg 0.0        # reproduces gotcha G1
    python tests/smoke_generate.py mtl --lang fr --t3 v3
    python tests/smoke_generate.py vc --src in.wav --ref target.wav
    python tests/smoke_generate.py all

Outputs go to syn_out/<variant>[-tag].wav (git-ignored). Exit code 1 on any failure.
"""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
import torchaudio as ta

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "syn_out"
OUT_DIR.mkdir(exist_ok=True)

TEXT_EN = "The quick brown fox jumps over the lazy dog, and honestly, nobody was surprised."
TEXT_TAGS = "Oh, that's hilarious! [chuckle] Anyway, the new model is in store, would you like a price?"
TEXT_BY_LANG = {
    "en": TEXT_EN,
    "fr": "Bonjour, comment ça va? Ceci est un test du modèle multilingue Chatterbox.",
    "de": "Guten Tag, dies ist ein kurzer Test des mehrsprachigen Chatterbox-Modells.",
    "hi": "नमस्ते, यह चैटरबॉक्स बहुभाषी मॉडल का एक छोटा परीक्षण है।",
    "zh": "你好，这是Chatterbox多语言模型的一个简短测试。",
    "ja": "こんにちは、これはチャッターボックス多言語モデルの短いテストです。",
}


def device_arg(s):
    if s == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s


def report(name, wav, sr, t_load, t_gen, extra=""):
    n = wav.shape[-1]
    dur = n / sr
    rtf = t_gen / dur if dur > 0 else float("inf")
    peak = float(wav.abs().max())
    print(f"[{name}] OK  samples={n} dur={dur:.2f}s sr={sr} peak={peak:.3f} "
          f"load={t_load:.1f}s gen={t_gen:.1f}s RTF={rtf:.2f} {extra}")
    if dur < 0.5:
        print(f"[{name}] WARNING: output shorter than 0.5 s")
    if peak > 1.0:
        print(f"[{name}] WARNING: clipping (peak {peak:.3f})")


def check_watermark(path, sr):
    try:
        import perth
        import librosa
        y, sr2 = librosa.load(str(path), sr=None)
        wm = perth.PerthImplicitWatermarker().get_watermark(y, sample_rate=sr2)
        return f"watermark={wm}"
    except Exception as e:  # noqa: BLE001
        return f"watermark_check_failed={type(e).__name__}"


def run_turbo(args, nano):
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    name = "nano" if nano else "turbo"
    t0 = time.time()
    model = ChatterboxTurboTTS.from_pretrained(device=args.device, nano=nano)
    t_load = time.time() - t0
    kwargs = {}
    if args.ref:
        kwargs["audio_prompt_path"] = args.ref
    t0 = time.time()
    wav = model.generate(TEXT_TAGS, **kwargs)
    t_gen = time.time() - t0
    out = OUT_DIR / f"{name}{args.tag}.wav"
    ta.save(str(out), wav, model.sr)
    report(name, wav, model.sr, t_load, t_gen, check_watermark(out, model.sr))
    return model


def run_english(args):
    from chatterbox.tts import ChatterboxTTS
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=args.device)
    t_load = time.time() - t0
    kwargs = dict(cfg_weight=args.cfg, exaggeration=args.exaggeration, temperature=args.temperature)
    if args.ref:
        kwargs["audio_prompt_path"] = args.ref
    t0 = time.time()
    wav = model.generate(TEXT_EN, **kwargs)
    t_gen = time.time() - t0
    out = OUT_DIR / f"english{args.tag}.wav"
    ta.save(str(out), wav, model.sr)
    report("english", wav, model.sr, t_load, t_gen, f"cfg={args.cfg} " + check_watermark(out, model.sr))
    return model


def run_mtl(args):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    t0 = time.time()
    model = ChatterboxMultilingualTTS.from_pretrained(device=args.device, t3_model=args.t3)
    t_load = time.time() - t0
    text = TEXT_BY_LANG.get(args.lang, TEXT_EN)
    kwargs = dict(cfg_weight=args.cfg, exaggeration=args.exaggeration, temperature=args.temperature)
    if args.ref:
        kwargs["audio_prompt_path"] = args.ref
    t0 = time.time()
    wav = model.generate(text, language_id=args.lang, **kwargs)
    t_gen = time.time() - t0
    out = OUT_DIR / f"mtl-{args.t3 or 'v2'}-{args.lang}{args.tag}.wav"
    ta.save(str(out), wav, model.sr)
    report(f"mtl-{args.t3 or 'v2'}-{args.lang}", wav, model.sr, t_load, t_gen, check_watermark(out, model.sr))
    return model


def run_vc(args):
    from chatterbox.vc import ChatterboxVC
    if not args.src:
        raise SystemExit("vc needs --src <source speech wav>")
    t0 = time.time()
    model = ChatterboxVC.from_pretrained(device=args.device)
    t_load = time.time() - t0
    t0 = time.time()
    wav = model.generate(args.src, target_voice_path=args.ref)
    t_gen = time.time() - t0
    out = OUT_DIR / f"vc{args.tag}.wav"
    ta.save(str(out), wav, model.sr)
    report("vc", wav, model.sr, t_load, t_gen, check_watermark(out, model.sr))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["nano", "turbo", "english", "mtl", "vc", "all"])
    p.add_argument("--device", default="auto")
    p.add_argument("--ref", default=None, help="reference wav for voice cloning / VC target")
    p.add_argument("--src", default=None, help="source wav for VC")
    p.add_argument("--lang", default="fr")
    p.add_argument("--t3", default=None, help="multilingual T3: v2 (default) or v3")
    p.add_argument("--cfg", type=float, default=0.5)
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--tag", default="", help="suffix for output filename")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.device = device_arg(args.device)
    logging.basicConfig(level=logging.WARNING)
    if args.seed:
        torch.manual_seed(args.seed)
    print(f"device={args.device} torch={torch.__version__}")

    variants = ["nano", "turbo", "english", "mtl"] if args.variant == "all" else [args.variant]
    failures = 0
    for v in variants:
        try:
            if v == "nano":
                run_turbo(args, nano=True)
            elif v == "turbo":
                run_turbo(args, nano=False)
            elif v == "english":
                run_english(args)
            elif v == "mtl":
                run_mtl(args)
            elif v == "vc":
                run_vc(args)
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"[{v}] FAILED")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
