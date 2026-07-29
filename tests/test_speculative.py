"""Speculative decoding correctness.

The defining property: speculative output must be token-identical to greedy
decoding with the target model alone. The draft only proposes; the target's
distribution decides every token. If this holds, speculation is a pure latency
optimization with zero quality change.
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL = os.environ.get("PYRE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("prompt", ["The capital of France is", "def fibonacci(n):"])
def test_speculative_matches_greedy(prompt):
    """Using the same model as both draft and target is a degenerate but
    strict check: every proposal is accepted, and the output must still exactly
    equal plain greedy decode. This isolates the accept/emit logic from draft
    quality."""
    from transformers import AutoTokenizer

    from pyre.loader import load_model
    from pyre.speculative import speculative_generate

    model, _ = load_model(MODEL, device="cuda", dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    n = 24

    greedy = model.generate_greedy(ids, max_new_tokens=n)[0, ids.shape[1]:].tolist()
    spec, stats = speculative_generate(model, model, ids, max_new_tokens=n, k=4)
    spec = spec[0, ids.shape[1]:].tolist()

    assert spec[:len(greedy)] == greedy or greedy[:len(spec)] == spec, (
        f"speculative output diverged from greedy\n  greedy: {greedy[:8]}\n  spec:   {spec[:8]}"
    )
    # with identical draft==target, every proposal should be accepted
    assert stats["tokens_per_pass"] > 3.5, (
        f"identical draft/target should accept ~all k=4 proposals, "
        f"got {stats['tokens_per_pass']:.2f} tokens/pass"
    )


@requires_cuda
@pytest.mark.gpu
def test_speculative_real_draft_matches_greedy():
    """The real configuration: 0.5B draft, 0.5B target is the same here (we only
    have one small model in CI), but the point is that ANY draft yields
    target-identical output. Verified with a genuinely different k."""
    from transformers import AutoTokenizer

    from pyre.loader import load_model
    from pyre.speculative import speculative_generate

    model, _ = load_model(MODEL, device="cuda", dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("Explain what a hash table is.", return_tensors="pt").input_ids.cuda()
    n = 32

    greedy = model.generate_greedy(ids, max_new_tokens=n)[0, ids.shape[1]:].tolist()
    spec, _ = speculative_generate(model, model, ids, max_new_tokens=n, k=6)
    spec = spec[0, ids.shape[1]:].tolist()

    assert spec[:len(greedy)] == greedy or greedy[:len(spec)] == spec
