"""Week 3 headline: reserved vs actually-used KV memory.

The contiguous cache reserves prompt_len + max_new_tokens per sequence up
front. At batch 32 that is a large fixed reservation for a worst case that most
sequences never reach. The paged cache allocates 16-token blocks on demand, so
it holds only what sequences actually use.

This computes both for a realistic batch where sequence lengths vary, and
reports the reduction. No GPU strictly needed -- it is arithmetic over the model
config -- but it runs on Kaggle alongside everything else.

Usage:
    python bench/bench_memory.py
"""

from __future__ import annotations

import argparse
import random

from pyre.config import ModelConfig
from pyre.paged_cache import BLOCK_SIZE


def kv_bytes_per_token(cfg: ModelConfig) -> int:
    # keys + values, all layers, all kv heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to a HF config.json")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Qwen2.5-1.5B values, so this runs without downloading anything.
    if args.config:
        cfg = ModelConfig.from_json(args.config)
    else:
        cfg = ModelConfig(
            hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
            num_attention_heads=12, num_key_value_heads=2, vocab_size=151936,
            rms_norm_eps=1e-6, rope_theta=1e6, tie_word_embeddings=True,
            max_position_embeddings=32768, head_dim=128,
        )

    per_tok = kv_bytes_per_token(cfg)

    mb = 1024 ** 2
    random.seed(args.seed)

    # Contiguous reserves max_len for every sequence regardless of workload.
    reserved = args.batch * args.max_len * per_tok

    def paged_mb(lengths):
        blocks = sum((n + BLOCK_SIZE - 1) // BLOCK_SIZE for n in lengths)
        return blocks * BLOCK_SIZE * per_tok / mb, blocks

    # Real inference traffic is short-skewed: most requests are brief, a few are
    # long. Uniform is the adversarial worst case for paging (still wins). The
    # exponential and short-heavy rows are what production actually looks like.
    workloads = {
        "uniform 20..max":   [random.randint(20, args.max_len) for _ in range(args.batch)],
        "exponential (mean ~200)":
            [min(args.max_len, max(20, int(random.expovariate(1 / 200)))) for _ in range(args.batch)],
        "short-heavy (90% <128)":
            [random.randint(20, 128) if random.random() < 0.9 else random.randint(128, args.max_len)
             for _ in range(args.batch)],
    }

    print(f"model: {cfg.num_hidden_layers}L, {cfg.num_key_value_heads} kv heads, "
          f"head_dim {cfg.head_dim}")
    print(f"batch {args.batch}, max_len {args.max_len}, block {BLOCK_SIZE}, "
          f"KV/token {per_tok / 1024:.1f} KB")
    print(f"\ncontiguous reserves {reserved / mb:.0f} MB for ALL workloads "
          f"(worst case, {args.batch} x {args.max_len})\n")
    print(f"{'workload':<26} {'mean len':>9} {'paged MB':>10} {'reduction':>10}")
    print("-" * 58)
    for name, lengths in workloads.items():
        used_mb, blocks = paged_mb(lengths)
        print(f"{name:<26} {sum(lengths) // len(lengths):>9} "
              f"{used_mb:>10.1f} {reserved / mb / used_mb:>9.1f}x")

    print(f"\nContiguous caps batch at "
          f"{int(16 * 1024 * mb / (args.max_len * per_tok))} sequences on a 16 GB T4 "
          f"(each reserving {args.max_len} tokens).")
    print("Paged caps only on tokens actually generated, so a short-heavy batch "
          "fits several times more sequences on the same card.")


if __name__ == "__main__":
    main()
