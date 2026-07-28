"""KV cache correctness.

The cache is a pure optimisation: it must not change a single token. Week 1
proved the naive forward matches HuggingFace, so the naive path is now a
trusted reference, and the cheapest strong check is cached-vs-naive on the same
model -- no HF download, no tolerance, exact integer equality.

The failure mode this is built to catch is the quiet one. A wrong position
offset in RoPE, or a mask that lets a query see one token too far, still
produces fluent text. It just produces different fluent text.
"""

from __future__ import annotations

import os

import pytest
import torch

from pyre.cache import KVCache

MODEL = os.environ.get("PYRE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

PROMPTS = [
    "Explain what a hash table is.",
    "The capital of France is",
    "def fibonacci(n):",
]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_cache_roundtrip():
    """Writes land at the right positions and reads return the right prefix."""
    c = KVCache(2, 1, 2, 4, 8, "cpu", torch.float32)

    prefill = torch.ones(1, 2, 3, 4)
    k, v = c.update(layer=0, start_pos=0, k=prefill, v=prefill * 2)
    assert k.shape == (1, 2, 3, 4)
    assert torch.equal(k, prefill) and torch.equal(v, prefill * 2)

    step = torch.full((1, 2, 1, 4), 7.0)
    k, v = c.update(layer=0, start_pos=3, k=step, v=step)
    assert k.shape == (1, 2, 4, 4), "read must cover the whole prefix"
    assert torch.equal(k[:, :, :3], prefill), "earlier positions must be untouched"
    assert torch.equal(k[:, :, 3:], step)


def test_cache_layers_are_independent():
    c = KVCache(2, 1, 1, 2, 4, "cpu", torch.float32)
    c.update(0, 0, torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))
    k1, _ = c.update(1, 0, torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))
    assert torch.equal(k1, torch.zeros(1, 1, 2, 2))


def test_cache_capacity_is_enforced():
    """Overrun must raise, not silently corrupt or truncate."""
    c = KVCache(1, 1, 1, 2, 4, "cpu", torch.float32)
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        c.update(0, 3, torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))


@pytest.fixture(scope="module")
def pyre_model():
    from pyre.loader import load_model

    model, _ = load_model(MODEL, device="cuda", dtype=torch.float16)
    return model


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("prompt", PROMPTS)
def test_cached_matches_naive(prompt, pyre_model):
    """The headline invariant: the cache changes speed, not output."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    n = 32

    naive = pyre_model.generate_greedy(ids, max_new_tokens=n)
    cached = pyre_model.generate_cached(ids, max_new_tokens=n)

    a = naive[0, ids.shape[1]:].tolist()
    b = cached[0, ids.shape[1]:].tolist()

    if a != b:
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), 0)
        pytest.fail(
            f"cached diverged from naive at token {first}\n"
            f"  naive:  {a[max(0, first - 3):first + 3]}\n"
            f"  cached: {b[max(0, first - 3):first + 3]}\n"
            "  token 0 means prefill is wrong; later means the RoPE position "
            "offset or the causal mask is off by one."
        )


@requires_cuda
@pytest.mark.gpu
def test_incremental_prefill_matches_full_prefill(pyre_model):
    """Feeding a prompt in chunks must equal feeding it whole.

    This is week 4 chunked prefill checked early, and it is the sharpest test of
    the start_pos plumbing: it exercises S > 1 with a non-zero offset, which
    neither prefill (offset 0) nor decode (S == 1) does.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPTS[0] + " It maps keys to values.", return_tensors="pt").input_ids.cuda()
    split = ids.shape[1] // 2

    whole_cache = KVCache.for_model(pyre_model.cfg, 1, ids.shape[1], "cuda", torch.float16)
    whole = pyre_model(ids, whole_cache, start_pos=0)

    chunk_cache = KVCache.for_model(pyre_model.cfg, 1, ids.shape[1], "cuda", torch.float16)
    pyre_model(ids[:, :split], chunk_cache, start_pos=0)
    chunked = pyre_model(ids[:, split:], chunk_cache, start_pos=split)

    diff = (whole[:, split:].float() - chunked.float()).abs().max().item()
    assert diff < 1e-2, f"chunked prefill differs from full prefill by {diff:.6f}"


@requires_cuda
@pytest.mark.gpu
def test_cached_matches_hf(pyre_model):
    """Closing the loop: cached pyre still matches HuggingFace exactly."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="eager"
    ).cuda().eval()
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.cuda()
    n = 32

    with torch.inference_mode():
        hf_out = hf.generate(
            ids, max_new_tokens=n, min_new_tokens=n, do_sample=False,
            use_cache=True, pad_token_id=tok.eos_token_id, repetition_penalty=1.0,
        )
    pyre_out = pyre_model.generate_cached(ids, max_new_tokens=n)

    assert hf_out[0, ids.shape[1]:].tolist() == pyre_out[0, ids.shape[1]:].tolist()
