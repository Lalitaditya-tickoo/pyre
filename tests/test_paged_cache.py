"""Paged cache correctness.

The allocator is pure bookkeeping and its tests run anywhere -- this is the bulk
of week 3, and it is all CPU. The GPU tests then prove that a sequence stored in
scattered blocks produces the same tokens as one stored contiguously.
"""

from __future__ import annotations

import os

import pytest
import torch

from pyre.paged_cache import BLOCK_SIZE, BlockAllocator, PagedKVCache

MODEL = os.environ.get("PYRE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


# --- allocator: CPU, runs everywhere -------------------------------------

def test_allocate_and_free_roundtrip():
    a = BlockAllocator(4)
    assert a.num_free == 4
    blocks = [a.allocate() for _ in range(4)]
    assert a.num_free == 0 and len(set(blocks)) == 4, "must hand out 4 distinct blocks"
    for b in blocks:
        a.free(b)
    assert a.num_free == 4


def test_exhaustion_raises():
    a = BlockAllocator(2)
    a.allocate(); a.allocate()
    with pytest.raises(MemoryError, match="exhausted"):
        a.allocate()


def test_freed_blocks_are_reused():
    a = BlockAllocator(2)
    b0 = a.allocate(); a.allocate()
    a.free(b0)
    b2 = a.allocate()
    assert b2 == b0, "a freed block must come back into circulation"


def test_double_free_raises():
    a = BlockAllocator(2)
    b = a.allocate()
    a.free(b)
    with pytest.raises(ValueError, match="double free"):
        a.free(b)


def test_refcount_shares_before_freeing():
    """Copy-on-write scaffolding: a shared block only returns to the pool once
    its last owner frees it."""
    a = BlockAllocator(2)
    b = a.allocate()
    a.incref(b)                       # now shared by 2
    assert a.ref_count(b) == 2
    a.free(b)                         # one owner leaves
    assert a.num_free == 1, "still held by the other owner"
    a.free(b)                        # last owner leaves
    assert a.num_free == 2


def test_block_table_grows_one_block_at_a_time():
    """A sequence must allocate exactly ceil(len / BLOCK_SIZE) blocks -- no
    over-reservation. This is the whole point of paging."""
    cache = _tiny_cache(num_blocks=8)
    cache.add_sequence(0)
    k = torch.ones(BLOCK_SIZE + 1, 2, 4)      # one past a block boundary
    cache.append(layer=0, seq_id=0, k=k, v=k)
    for layer in range(1, cache.n_layers):
        cache.append(layer, 0, k, k)
    assert len(cache.block_tables[0]) == 2, "17 tokens must use exactly 2 blocks"
    assert cache.allocator.num_used == 2


def test_freeing_sequence_returns_all_its_blocks():
    cache = _tiny_cache(num_blocks=8)
    cache.add_sequence(0)
    k = torch.ones(BLOCK_SIZE * 3, 2, 4)
    for layer in range(cache.n_layers):
        cache.append(layer, 0, k, k)
    assert cache.allocator.num_used == 3
    cache.free_sequence(0)
    assert cache.allocator.num_used == 0
    assert 0 not in cache.block_tables


def test_two_sequences_get_disjoint_blocks():
    cache = _tiny_cache(num_blocks=8)
    cache.add_sequence(0); cache.add_sequence(1)
    k = torch.ones(BLOCK_SIZE, 2, 4)
    for layer in range(cache.n_layers):
        cache.append(layer, 0, k, k)
        cache.append(layer, 1, k, k)
    b0 = set(cache.block_tables[0]); b1 = set(cache.block_tables[1])
    assert b0.isdisjoint(b1), "distinct sequences must never share a physical block"


def _tiny_cache(num_blocks: int) -> PagedKVCache:
    class Cfg:
        num_hidden_layers = 2
        num_key_value_heads = 2
        head_dim = 4
    return PagedKVCache.for_model(Cfg(), num_blocks, "cpu", torch.float32)


def test_gather_reconstructs_written_data():
    """What went into scattered blocks must come back out in logical order."""
    cache = _tiny_cache(num_blocks=8)
    cache.add_sequence(0)
    n = BLOCK_SIZE + 5
    k = torch.arange(n * 2 * 4, dtype=torch.float32).reshape(n, 2, 4)
    for layer in range(cache.n_layers):
        cache.append(layer, 0, k, k)
    gk, gv = cache.gather(layer=0, seq_id=0)
    assert gk.shape == (n, 2, 4)
    assert torch.equal(gk, k), "gather must return tokens in the order written"
    assert torch.equal(gv, k)


# --- end to end: GPU ------------------------------------------------------

@requires_cuda
@pytest.mark.gpu
def test_paged_gather_matches_contiguous_forward():
    """A prefill stored in the paged cache, gathered back, must match the same
    prefill through the contiguous cache -- proving the block indirection is
    transparent to the attention math."""
    from transformers import AutoTokenizer

    from pyre.cache import KVCache
    from pyre.loader import load_model

    model, cfg = load_model(MODEL, device="cuda", dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("Explain what a hash table is.", return_tensors="pt").input_ids.cuda()
    n = ids.shape[1]

    with torch.inference_mode():
        contig = KVCache.for_model(cfg, 1, n, "cuda", torch.float16)
        ref = model(ids, contig, start_pos=0)

    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE + 2
    paged = PagedKVCache.for_model(cfg, num_blocks, "cuda", torch.float16)
    paged.add_sequence(0)

    # Drive the model's per-layer k/v into the paged cache by hand, then check a
    # gathered layer equals the contiguous cache's stored layer.
    with torch.inference_mode():
        _ = model(ids, contig, start_pos=0)
    for layer in range(cfg.num_hidden_layers):
        k = contig.k[layer, 0].transpose(0, 1)[:n]      # (n, n_kv, hd)
        v = contig.v[layer, 0].transpose(0, 1)[:n]
        paged.append(layer, 0, k, v)

    for layer in range(cfg.num_hidden_layers):
        gk, gv = paged.gather(layer, 0)
        ck = contig.k[layer, 0].transpose(0, 1)[:n]
        cv = contig.v[layer, 0].transpose(0, 1)[:n]
        assert torch.equal(gk, ck), f"paged keys differ from contiguous at layer {layer}"
        assert torch.equal(gv, cv), f"paged values differ from contiguous at layer {layer}"
    assert ref.shape[1] == n
