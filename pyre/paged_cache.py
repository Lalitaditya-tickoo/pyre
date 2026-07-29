"""Paged KV cache.

Week 3. The contiguous cache reserves prompt_len + max_new_tokens per sequence
up front, so a sequence that stops early still holds memory for a length it
never reached, and two sequences sharing a prefix each store their own copy.
At batch 32 that is ~900 MB reserved for a worst case that mostly will not
happen.

Paging fixes both. KV memory is one pool of fixed-size blocks (BLOCK_SIZE
tokens each). Each sequence holds a *block table* -- an ordered list of physical
block indices -- and grows it one block at a time as it generates. Physical
blocks need not be contiguous; the block table is the indirection that hides
that from attention. This is the same idea as virtual memory paging, which is
where the name comes from.

Two structures:

  BlockAllocator  owns the free list. hand out blocks, take them back. This is
                  pure bookkeeping -- no tensors, no CUDA -- so it is fully
                  testable on a laptop, which is where its correctness is
                  established before any GPU sees it.

  PagedKVCache    owns the physical KV tensors and the per-sequence block
                  tables. Translates (sequence, logical position) to a physical
                  slot and reads back a sequence's keys/values gathered across
                  its scattered blocks.

Copy-on-write for shared prefixes is deferred to week 6 (the radix cache); the
ref_count scaffolding for it lives here so the allocator does not need reworking
then.
"""

from __future__ import annotations

import torch

BLOCK_SIZE = 16


class BlockAllocator:
    """A free list over [0, num_blocks). Pure bookkeeping, no tensors."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # Hand out low indices first: deterministic, and easier to read in tests
        # than a set. A list used as a stack is O(1) at both ends we touch.
        self._free: list[int] = list(range(num_blocks))
        self._ref_count: dict[int, int] = {}

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def allocate(self) -> int:
        """Take one block. Raises if the pool is exhausted -- the caller
        (the scheduler, later) is expected to handle this by preempting a
        sequence, not by crashing. For now it surfaces as a clear error."""
        if not self._free:
            raise MemoryError(
                f"KV block pool exhausted ({self.num_blocks} blocks all in use). "
                "Either lower batch size / max length, or wait for week 4's "
                "scheduler to preempt a sequence."
            )
        block = self._free.pop()
        self._ref_count[block] = 1
        return block

    def free(self, block: int) -> None:
        """Return a block. With copy-on-write a block may be shared; only the
        last owner actually returns it to the pool."""
        rc = self._ref_count.get(block, 0)
        if rc <= 0:
            raise ValueError(f"double free or freeing unallocated block {block}")
        rc -= 1
        if rc == 0:
            self._ref_count.pop(block)
            self._free.append(block)
        else:
            self._ref_count[block] = rc

    def incref(self, block: int) -> None:
        """Mark a block as shared by one more sequence (copy-on-write, week 6)."""
        if block not in self._ref_count:
            raise ValueError(f"increfing unallocated block {block}")
        self._ref_count[block] += 1

    def ref_count(self, block: int) -> int:
        return self._ref_count.get(block, 0)


class PagedKVCache:
    """Physical KV blocks plus per-sequence block tables.

    Physical tensor shape, keys and values each:
        (n_layers, num_blocks, BLOCK_SIZE, n_kv_heads, head_dim)

    Block is dim 1 so a single (layer, block) selects a contiguous
    (BLOCK_SIZE, n_kv_heads, head_dim) tile -- the unit that gets written and
    gathered.
    """

    def __init__(self, n_layers, num_blocks, n_kv_heads, head_dim, device, dtype):
        shape = (n_layers, num_blocks, BLOCK_SIZE, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.allocator = BlockAllocator(num_blocks)
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        # seq_id -> list of physical block indices, in logical order
        self.block_tables: dict[int, list[int]] = {}
        # seq_id -> number of tokens currently stored
        self.seq_len: dict[int, int] = {}
        # tokens written in the current pass; advances on layer 0 so gather()
        # sees the full prefix on every layer, not just after the last one.
        self.write_len: dict[int, int] = {}

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    @classmethod
    def for_model(cls, cfg, num_blocks, device, dtype) -> "PagedKVCache":
        return cls(
            n_layers=cfg.num_hidden_layers,
            num_blocks=num_blocks,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            device=device,
            dtype=dtype,
        )

    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self.block_tables:
            raise ValueError(f"sequence {seq_id} already exists")
        self.block_tables[seq_id] = []
        self.seq_len[seq_id] = 0
        self.write_len[seq_id] = 0

    def add_sequence_with_prefix(self, seq_id: int, shared_blocks: list) -> int:
        """Start a sequence that reuses already-cached prefix blocks.

        The shared blocks are ref-counted (incref), not copied: several
        sequences point at the same physical KV for the shared prefix. Returns
        the number of prefix tokens already covered, so the caller prefills only
        the suffix. This is the payoff of week 3's ref_count scaffolding.
        """
        if seq_id in self.block_tables:
            raise ValueError(f"sequence {seq_id} already exists")
        for blk in shared_blocks:
            self.allocator.incref(blk)
        self.block_tables[seq_id] = list(shared_blocks)
        covered = len(shared_blocks) * BLOCK_SIZE
        self.seq_len[seq_id] = covered
        return covered

    def free_sequence(self, seq_id: int) -> None:
        for block in self.block_tables.pop(seq_id):
            self.allocator.free(block)
        self.seq_len.pop(seq_id)
        self.write_len.pop(seq_id, None)

    def _ensure_capacity(self, seq_id: int, new_len: int) -> None:
        """Grow a sequence's block table so it can hold new_len tokens."""
        have = len(self.block_tables[seq_id]) * BLOCK_SIZE
        while have < new_len:
            self.block_tables[seq_id].append(self.allocator.allocate())
            have += BLOCK_SIZE

    def append(self, layer: int, seq_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write ``k``/``v`` for the next tokens of one sequence.

        k, v: (seq, n_kv_heads, head_dim) -- the new tokens only. On layer 0 the
        block table grows if needed; later layers reuse it, since every layer of
        one forward pass writes the same logical positions.
        """
        start = self.seq_len[seq_id]
        seq = k.shape[0]
        end = start + seq

        if layer == 0:
            self._ensure_capacity(seq_id, end)

        table = self.block_tables[seq_id]
        for i in range(seq):
            pos = start + i
            block = table[pos // BLOCK_SIZE]
            off = pos % BLOCK_SIZE
            self.k[layer, block, off] = k[i]
            self.v[layer, block, off] = v[i]

        if layer == 0:
            self.write_len[seq_id] = end
        if layer == self.n_layers - 1:
            self.seq_len[seq_id] = end

    def gather(self, layer: int, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this sequence's keys/values so far as contiguous tensors,
        shape (seq_len, n_kv_heads, head_dim).

        This copies scattered blocks into a dense tensor so week 2's attention
        math can run unchanged. It is deliberately the slow, obvious version:
        week 5's Triton kernel reads the blocks in place via the block table
        and never materialises this gather. Correct first, fused later.
        """
        n = self.write_len[seq_id]
        table = self.block_tables[seq_id]
        n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        idx = torch.tensor(table[:n_blocks], device=self.device, dtype=torch.long)
        k = self.k[layer, idx].reshape(-1, self.n_kv_heads, self.head_dim)[:n]
        v = self.v[layer, idx].reshape(-1, self.n_kv_heads, self.head_dim)[:n]
        return k, v
