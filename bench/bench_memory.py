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

    # Realistic mixed workload: most sequences finish well short of max_len.
    random.seed(args.seed)
    actual_lengths = [random.randint(20, args.max_len) for _ in range(args.batch)]

    # Contiguous: every sequence reserves the worst case.
    reserved = args.batch * args.max_len * per_tok

    # Paged: each sequence rounds up to whole blocks of its actual length.
    used_blocks = sum((n + BLOCK_SIZE - 1) // BLOCK_SIZE for n in actual_lengths)
    used = used_blocks * BLOCK_SIZE * per_tok

    mb = 1024 ** 2
    print(f"model: {cfg.num_hidden_layers}L, {cfg.num_key_value_heads} kv heads, "
          f"head_dim {cfg.head_dim}")
    print(f"batch {args.batch}, max_len {args.max_len}, block {BLOCK_SIZE}")
    print(f"KV per token: {per_tok / 1024:.1f} KB\n")
    print(f"actual sequence lengths (sample): {sorted(actual_lengths)[:8]} ... "
          f"mean {sum(actual_lengths) // args.batch}")
    print(f"\ncontiguous reserved: {reserved / mb:8.1f} MB  (worst case for all {args.batch})")
    print(f"paged used:          {used / mb:8.1f} MB  ({used_blocks} blocks)")
    print(f"reduction:           {reserved / used:8.1f}x")
    print(f"\nOn a 16 GB T4, contiguous caps batch at "
          f"{int(16 * 1024 * mb / (args.max_len * per_tok))} sequences; "
          f"paged fits far more by not reserving the worst case.")


if __name__ == "__main__":
    main()
