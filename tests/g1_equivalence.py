"""Regression check for the G1 fix (cfg_weight=0 in T3.inference).

Calls T3.inference directly with near-greedy sampling (temperature 0.05, fixed seed) and
saves the generated speech-token ids, so the sequence produced before and after the fix
can be compared. Run once with the old code (`--label old`), once with the new (`--label new`),
then `--compare`.

    python tests/g1_equivalence.py --label old
    python tests/g1_equivalence.py --label new
    python tests/g1_equivalence.py --compare
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parents[1] / "syn_out"
OUT.mkdir(exist_ok=True)
TEXT = "The quick brown fox jumps over the lazy dog."


def gen(model, cfg_weight, duplicate, seed=1234):
    from chatterbox.tts import punc_norm
    text = punc_norm(TEXT)
    tokens = model.tokenizer.text_to_tokens(text).to(model.device)
    if duplicate:
        tokens = torch.cat([tokens, tokens], dim=0)
    tokens = F.pad(tokens, (1, 0), value=model.t3.hp.start_text_token)
    tokens = F.pad(tokens, (0, 1), value=model.t3.hp.stop_text_token)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        out = model.t3.inference(
            t3_cond=model.conds.t3, text_tokens=tokens, max_new_tokens=300,
            temperature=0.05, cfg_weight=cfg_weight, repetition_penalty=1.2, min_p=0.05, top_p=1.0,
        )
    return out[0].cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default=None)
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()

    if args.compare:
        for case in ["cfg05", "cfg0"]:
            a = torch.load(OUT / f"g1_old_{case}.pt")
            b = torch.load(OUT / f"g1_new_{case}.pt")
            same = a.shape == b.shape and torch.equal(a, b)
            n = min(len(a), len(b))
            first_diff = next((i for i in range(n) if a[i] != b[i]), None)
            print(f"{case}: old_len={len(a)} new_len={len(b)} identical={same} first_diff_idx={first_diff}")
        return

    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained("cuda")
    # cfg 0.5: both old and new code expect/handle batch 2; feed duplicated tokens to both
    # (old code requires it; new code accepts it).
    t = gen(model, cfg_weight=0.5, duplicate=True)
    torch.save(t, OUT / f"g1_{args.label}_cfg05.pt")
    print(f"[{args.label}] cfg0.5 tokens={len(t)} head={t[:12].tolist()}")
    # cfg 0: old code only works when the caller duplicates (multilingual style);
    # new code should give the same result from a single row.
    t = gen(model, cfg_weight=0.0, duplicate=(args.label == "old"))
    torch.save(t, OUT / f"g1_{args.label}_cfg0.pt")
    print(f"[{args.label}] cfg0 tokens={len(t)} head={t[:12].tolist()}")


if __name__ == "__main__":
    main()
