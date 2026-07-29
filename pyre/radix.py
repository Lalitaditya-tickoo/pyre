"""Radix prefix cache.

Week 6. The week-1 baseline showed the cost this fixes: at batch 32 the
shared_prefix suite had 0.398s TTFT vs 0.080s for short, a 5x penalty, because
32 requests each re-prefilled an identical 297-character preamble. That work is
byte-for-byte redundant across requests.

A radix tree indexes cached prefixes by their token sequence. When a new request
arrives, we walk the tree matching its prompt tokens against what is already
cached; the matched prefix's KV blocks are reused (ref-counted, not recomputed),
and only the divergent suffix is prefilled.

This is what the ref_count scaffolding in the block allocator (week 3) was for:
a shared block must not be freed while another sequence still points at it.

The tree stores, per node, the run of tokens on the edge into it and the list of
physical KV blocks covering the prefix ending at that node. Matching is greedy
along edges; a partial edge match splits the node.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RadixNode:
    # tokens on the edge from parent to this node
    tokens: list[int] = field(default_factory=list)
    # physical KV block ids covering the prefix that ends at this node
    blocks: list[int] = field(default_factory=list)
    children: dict[int, "RadixNode"] = field(default_factory=dict)  # keyed by first token of child edge


class RadixTree:
    """Maps token-sequence prefixes to cached KV block lists.

    Block granularity: prefixes are cached in whole BLOCK_SIZE-token units, so
    the number of blocks for a prefix of length n is n // BLOCK_SIZE (the final
    partial block is never shared -- it is still being written).
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = RadixNode()

    def match(self, tokens: list[int]) -> tuple[int, list[int]]:
        """Longest cached prefix of ``tokens`` that lands on a block boundary.

        Returns (matched_len, blocks) where matched_len is a multiple of
        block_size and blocks are the physical blocks covering those tokens.
        Only whole blocks are reusable; a half-filled block is still mutating.
        """
        node = self.root
        matched = 0
        blocks: list[int] = []
        i = 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            edge = child.tokens
            # how many tokens of this edge match the query
            j = 0
            while j < len(edge) and i + j < len(tokens) and edge[j] == tokens[i + j]:
                j += 1
            if j == len(edge):
                # full edge consumed
                node = child
                i += j
                matched = i
                blocks = list(child.blocks)
            else:
                # partial edge match -- cannot reuse a partial block, stop here
                break
        # round down to a block boundary
        usable = (matched // self.block_size) * self.block_size
        n_blocks = usable // self.block_size
        return usable, blocks[:n_blocks]

    def insert(self, tokens: list[int], blocks: list[int]) -> None:
        """Record that ``tokens`` (a whole-block-aligned prefix) is cached in
        ``blocks``. Splits edges as needed so future matches can share sub-prefixes."""
        node = self.root
        i = 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                new = RadixNode(tokens=tokens[i:], blocks=list(blocks))
                node.children[tokens[i]] = new
                return
            edge = child.tokens
            j = 0
            while j < len(edge) and i + j < len(tokens) and edge[j] == tokens[i + j]:
                j += 1
            if j == len(edge):
                node = child
                i += j
                if i == len(tokens):
                    if not child.blocks:
                        child.blocks = list(blocks)
                    return
            else:
                # split child at j
                mid = RadixNode(tokens=edge[:j])
                child.tokens = edge[j:]
                mid.children[edge[j]] = child
                # blocks covering the split point
                nb = (i + j) // self.block_size
                mid.blocks = list(blocks[:nb])
                node.children[tokens[i]] = mid
                if i + j < len(tokens):
                    leaf = RadixNode(tokens=tokens[i + j:], blocks=list(blocks))
                    mid.children[tokens[i + j]] = leaf
                return
