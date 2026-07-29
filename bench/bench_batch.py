"""Week 4 headline: continuous-batching throughput vs the HF baseline.

The week-2 sweep proved batch-1 decode is weight-bandwidth-bound, so throughput
only rises by putting more sequences through each weight read. This measures
exactly that: PYRE's scheduler running N identical-length requests to
completion, tokens/sec as N grows, against HF's 904 tok/s at batch 32.

Usage:
    python bench/bench_batch.py --out bench/results/w4_batch.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import harness, prompts  # noqa: E402
from pyre.loader import load_model  # noqa: E402
from pyre.scheduler import Scheduler  # noqa: E402


def run_batch(model, cfg, tok, prompt_ids, batch, gen_len, device):
    """One scheduler run of `batch` copies of the prompt; return tokens/sec."""
    # size the block pool for the worst case of this run
    max_len = len(prompt_ids) + gen_len
    num_blocks = (max_len // 16 + 2) * batch + 8
    sched = Scheduler(model, cfg, num_blocks=num_blocks, max_batch=batch, device=device)
    for _ in range(batch):
        sched.add_request(prompt_ids, gen_len)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = sched.run()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    total_tokens = sum(len(g) for g in results.values())
    return total_tokens / dt, total_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--out", default="bench/results/w4_batch.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model, cfg = load_model(args.model, device="cuda", dtype=torch.float16)
    prompt_ids = tok(prompts.get_suite("short", 1)[0], return_tensors="pt").input_ids[0].tolist()

    print(f"model {args.model}, gen_len {args.gen_len}, HF baseline @ bs32 = 904 tok/s\n")
    print(f"{'batch':>6} {'tok/s':>10} {'vs HF bs32':>11} {'tokens':>8}")
    print("-" * 40)

    rows = []
    for b in args.batches:
        # warmup at this batch size, then measure
        run_batch(model, cfg, tok, prompt_ids, min(b, 4), args.gen_len, "cuda")
        tps, toks = run_batch(model, cfg, tok, prompt_ids, b, args.gen_len, "cuda")
        print(f"{b:>6} {tps:>10.1f} {tps / 904:>10.2f}x {toks:>8}")
        rows.append({"batch": b, "tok_s": round(tps, 2), "tokens": toks})

    best = max(r["tok_s"] for r in rows)
    print(f"\npeak PYRE throughput: {best:.1f} tok/s ({best / 904:.2f}x HF's 904 at bs32)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"environment": harness.environment(), "model": args.model,
         "gen_len": args.gen_len, "hf_bs32_baseline": 904, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
