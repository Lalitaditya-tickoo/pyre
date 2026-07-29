"""Radix prefix cache correctness. Pure Python, runs anywhere."""

from __future__ import annotations

from pyre.radix import RadixTree

BS = 16


def test_empty_tree_matches_nothing():
    t = RadixTree(BS)
    matched, blocks = t.match([1, 2, 3])
    assert matched == 0 and blocks == []


def test_exact_prefix_reused():
    t = RadixTree(BS)
    toks = list(range(32))              # exactly 2 blocks
    t.insert(toks, [100, 101])
    matched, blocks = t.match(toks)
    assert matched == 32 and blocks == [100, 101]


def test_match_rounds_down_to_block_boundary():
    """A 20-token prefix caches only its first whole block (16 tokens)."""
    t = RadixTree(BS)
    toks = list(range(20))
    t.insert(toks[:16], [100])          # only whole blocks are inserted
    matched, blocks = t.match(toks)
    assert matched == 16 and blocks == [100], "partial second block must not be reused"


def test_shared_prefix_divergent_suffix():
    """Two sequences share a 2-block prefix, then diverge. The shared prefix's
    blocks are returned for both; the suffix is not cached."""
    t = RadixTree(BS)
    prefix = list(range(32))
    a = prefix + [1000, 1001]
    b = prefix + [2000, 2001]
    t.insert(prefix, [100, 101])
    ma, ba = t.match(a)
    mb, bb = t.match(b)
    assert ma == 32 and ba == [100, 101]
    assert mb == 32 and bb == [100, 101], "both diverging seqs reuse the shared prefix blocks"


def test_longer_cached_prefix_extends_match():
    """If a longer prefix gets cached later, a matching request reuses more."""
    t = RadixTree(BS)
    short = list(range(16))
    long = list(range(48))              # 3 blocks, extends short
    t.insert(short, [100])
    assert t.match(long) == (16, [100])
    t.insert(long, [100, 101, 102])
    assert t.match(long) == (48, [100, 101, 102])


def test_no_match_on_different_first_token():
    t = RadixTree(BS)
    t.insert(list(range(16)), [100])
    matched, blocks = t.match([999] + list(range(1, 16)))
    assert matched == 0 and blocks == []
