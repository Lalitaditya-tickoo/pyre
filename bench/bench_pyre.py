"""Week 2 benchmark: naive decode vs cached decode, against the HF baseline.

Batch size 1 only, deliberately. Batching prompts of different lengths needs
left-padding plus a padding mask plus position-id correction -- three things
that belong to the scheduler, not the cache. Week 4 does batching properly with
continuous batching. Single-stream is also where the cache effect is clearest:
the naive loop's per-token cost grows with every token, the cached loop's does
not.

Usage:
    python bench/bench_pyre.py --out bench/results/w2_pyre.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import harness, prompts  # noqa: E402
from pyre.loader import load_model  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_backends(model_id: str, device: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model, cfg = load_model(model_id, device=device, dtype=torch.float16)

    @torch.inference_mode()
    def naive(batch_prompts, max_new_tokens):
        ids = tok(batch_prompts[0], return_tensors="pt").input_ids.to(device)
        out = model.generate_greedy(ids, max_new_tokens=max_new_tokens)
        return [tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)]

    @torch.inference_mode()
    def cached(batch_prompts, max_new_tokens):
        ids = tok(batch_prompts[0], return_tensors="pt").input_ids.to(device)
        out = model.generate_cached(ids, max_new_tokens=max_new_tokens)
        return [tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)]

    return {"pyre-naive": naive, "pyre-cached": cached}, tok, model, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--suites", nargs="+", default=["short", "shared_prefix"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-naive", action="store_true")
    ap.add_argument("--out", default="bench/results/w2_pyre.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device. CPU numbers are not comparable to anything.")

    backends, tok, model, cfg = build_backends(args.model, device)
    if args.skip_naive:
        backends.pop("pyre-naive")

    ids = tok(prompts.get_suite("short", 1)[0], return_tensors="pt").input_ids.to(device)
    a = model.generate_greedy(ids, max_new_tokens=16)[0, ids.shape[1]:].tolist()
    b = model.generate_cached(ids, max_new_tokens=16)[0, ids.shape[1]:].tolist()
    if a != b:
        sys.exit("naive and cached decode disagree -- fix correctness before benchmarking")
    print("naive/cached agreement: OK\n")

    results = []
    for name, fn in backends.items():
        for suite in args.suites:
            batch = prompts.get_suite(suite, 1)
            m = harness.measure(
                fn, batch, args.max_new_tokens,
                backend=name, suite=suite, repeats=args.repeats,
            )
            results.append(m)
            print(f"  done {name} suite={suite}: {m.total_tps:.1f} tok/s, {m.ttft_s:.3f}s TTFT")

    print()
    harness.print_table(results)

    if "pyre-naive" in backends:
        nv = next(m.total_tps for m in results if m.backend == "pyre-naive" and m.suite == "short")
        cd = next(m.total_tps for m in results if m.backend == "pyre-cached" and m.suite == "short")
        print(f"\ncache speedup on suite=short: {cd / nv:.2f}x")

    reserved = (
        cfg.num_hidden_layers * 2 * cfg.num_key_value_heads
        * (128 + args.max_new_tokens) * cfg.head_dim * 2
    )
    print(f"contiguous cache reserved for one sequence: {reserved / 1024**2:.1f} MB")

    path = harness.save(results, args.out, extra={"model": args.model, "batch_size": 1})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
