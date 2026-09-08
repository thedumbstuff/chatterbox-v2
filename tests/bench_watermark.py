"""Benchmark: generation time with vs without the Perth watermark, per variant.

Warm runs (first call discarded), same text, same process. Prints a markdown table.
    python tests/bench_watermark.py            # nano, turbo, english, mtl-hi (v3)
    python tests/bench_watermark.py --runs 5
"""
import argparse
import time
import warnings

import torch

warnings.filterwarnings("ignore")
TEXT = "The quick brown fox jumps over the lazy dog, and honestly, nobody was surprised."
HI = "नमस्ते, यह चैटरबॉक्स बहुभाषी मॉडल का एक छोटा परीक्षण है।"


def timed(fn):
    torch.cuda.synchronize()
    t = time.time()
    wav = fn()
    torch.cuda.synchronize()
    return time.time() - t, wav.shape[-1] / 24000


def bench(name, gen, runs):
    rows = {}
    for wm in (False, True):
        gen(wm)  # warmup (also loads perth on first True)
        tot_t = tot_d = 0.0
        for _ in range(runs):
            dt, dur = timed(lambda: gen(wm))
            tot_t += dt
            tot_d += dur
        rows[wm] = (tot_t / runs, tot_d / runs, tot_t / tot_d)
    off, on = rows[False], rows[True]
    saved = on[0] - off[0]
    print(f"| {name} | {on[0]:.2f} s | {off[0]:.2f} s | {saved:.2f} s ({saved / on[0] * 100:.0f}%) | {on[2]:.2f} | {off[2]:.2f} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()
    print(f"device=cuda ({torch.cuda.get_device_name(0)}), fp32, warm, {args.runs} runs each, ~4-5 s sentences\n")
    print("| Variant | gen time WITH watermark | gen time WITHOUT | saved | RTF with | RTF without |")
    print("|---|---|---|---|---|---|")

    from chatterbox.tts_turbo import ChatterboxTurboTTS
    m = ChatterboxTurboTTS.from_pretrained("cuda", nano=True)
    bench("Nano", lambda wm: m.generate(TEXT, watermark=wm), args.runs)
    del m
    m = ChatterboxTurboTTS.from_pretrained("cuda")
    bench("Turbo", lambda wm: m.generate(TEXT, watermark=wm), args.runs)
    del m
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained("cuda")
    bench("English (cfg 0.5)", lambda wm: m.generate(TEXT, watermark=wm), args.runs)
    del m
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    m = ChatterboxMultilingualTTS.from_pretrained("cuda", t3_model="v3")
    bench("Multilingual v3 Hindi", lambda wm: m.generate(HI, language_id="hi", watermark=wm), args.runs)


if __name__ == "__main__":
    main()
