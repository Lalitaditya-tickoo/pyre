"""Where does the KV cache start paying?

Week 2 measured a 1.03x cache speedup at 128 generated tokens and batch 1,
which looks like a failure and is not. At short sequences decode is bound by
streaming ~3.1 GB of fp16 weights per forward pass, not by attention. The naive
path's extra O(n^2) attention work hides inside that bandwidth floor.

This sweeps generation length to find the crossover -- the point where the
quadratic term finally exceeds the floor and the cache earns its memory.

Usage:
    python bench/sweep_length.py --out bench/results/w2_sweep.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import harness, prompts  # noqa: E402
from pyre.loader import load_model  # noqa: E402


def timed(fn, *args, repeats=2):
    for _ in range(1):
        fn(*args)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument("--out", default="bench/results/w2_sweep.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model, cfg = load_model(args.model, device="cuda", dtype=torch.float16)
    ids = tok(prompts.get_suite("short", 1)[0], return_tensors="pt").input_ids.cuda()

    print(f"{'gen_len':>8} {'naive t/s':>11} {'cached t/s':>12} {'speedup':>9} {'cache MB':>10}")
    print("-" * 54)

    rows = []
    for n in args.lengths:
        t_naive = timed(lambda: model.generate_greedy(ids, max_new_tokens=n))
        t_cached = timed(lambda: model.generate_cached(ids, max_new_tokens=n))
        naive_tps, cached_tps = n / t_naive, n / t_cached
        cache_mb = (
            cfg.num_hidden_layers * 2 * cfg.num_key_value_heads
            * (ids.shape[1] + n) * cfg.head_dim * 2 / 1024**2
        )
        print(f"{n:>8} {naive_tps:>11.1f} {cached_tps:>12.1f} "
              f"{cached_tps / naive_tps:>8.2f}x {cache_mb:>9.1f}")
        rows.append({
            "gen_len": n, "naive_tps": round(naive_tps, 2),
            "cached_tps": round(cached_tps, 2),
            "speedup": round(cached_tps / naive_tps, 3),
            "cache_mb": round(cache_mb, 1),
        })

    cross = next((r["gen_len"] for r in rows if r["speedup"] >= 1.25), None)
    print(f"\ncrossover (cache >= 1.25x): {cross if cross else 'not reached in this range'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(
        {"environment": harness.environment(), "model": args.model, "rows": rows}, indent=2
    ))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
