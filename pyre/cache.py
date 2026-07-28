"""Contiguous KV cache.

Week 2. The naive loop re-runs the full forward for every token, recomputing
keys and values for the entire prefix each step -- O(n^2) for work that is
provably identical every time. This caches it.

"Contiguous" means each sequence gets one flat slab sized for its worst case.
That is the simple design, and it is exactly what paged attention exists to fix:

  * The slab is sized for max_seq, but most sequences stop far short, so most
    of the reservation is never touched.
  * Two sequences sharing a prefix each store their own copy.
  * Freed slabs leave holes that only fit sequences of similar size.

Week 3 replaces this with fixed-size blocks and an indirection table. The
numbers this produces are what that gets measured against.

Layout: (n_layers, batch, n_kv_heads, max_seq, head_dim). Sequence position is
dim 3 so appending is a contiguous slice-assign and reading back is a view.
"""

from __future__ import annotations

import torch


class KVCache:
    """Pre-allocated key/value store for a fixed batch and maximum length."""

    def __init__(self, n_layers, batch, n_kv_heads, head_dim, max_seq, device, dtype):
        shape = (n_layers, batch, n_kv_heads, max_seq, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq = max_seq
        self.n_layers = n_layers
        self.batch = batch

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def update(self, layer: int, start_pos: int, k: torch.Tensor, v: torch.Tensor):
        """Write k/v at start_pos, return the full prefix 0 .. start_pos+seq.

        start_pos is passed in rather than tracked internally on purpose. Every
        layer in one forward pass writes the same position range, so a
        self-incrementing counter would have to know it is called n_layers times
        per token -- a bug waiting to happen. The caller owns the position.
        """
        seq = k.shape[2]
        end = start_pos + seq
        if end > self.max_seq:
            raise ValueError(
                f"sequence length {end} exceeds cache capacity {self.max_seq}. "
                "Allocate with max_seq >= prompt_len + max_new_tokens."
            )
        self.k[layer, :, :, start_pos:end] = k
        self.v[layer, :, :, start_pos:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    @classmethod
    def for_model(cls, cfg, batch: int, max_seq: int, device, dtype) -> "KVCache":
        return cls(
            n_layers=cfg.num_hidden_layers,
            batch=batch,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            max_seq=max_seq,
            device=device,
            dtype=dtype,
        )
