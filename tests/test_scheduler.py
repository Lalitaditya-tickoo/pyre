"""Continuous batching correctness.

The invariant that makes batching safe: a sequence decoded inside a batch must
produce the exact same tokens as that sequence decoded alone. Batching changes
throughput, never output. If this holds, the scheduler is correct regardless of
what else is in the batch or when other sequences finish.
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL = os.environ.get("PYRE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@requires_cuda
@pytest.mark.gpu
def test_batched_matches_single():
    """Three prompts of different lengths, decoded together, must match each
    decoded alone. Different lengths force sequences to finish at different
    steps -- the exact case static batching handles badly and continuous
    batching must handle correctly."""
    from transformers import AutoTokenizer

    from pyre.loader import load_model
    from pyre.scheduler import Scheduler

    model, cfg = load_model(MODEL, device="cuda", dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL)

    prompts = ["Explain what a hash table is.", "The capital of France is", "def fibonacci(n):"]
    max_new = {0: 24, 1: 16, 2: 32}   # different caps -> finish at different steps

    # Reference: each prompt decoded alone via the paged path.
    reference = {}
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        num_blocks = (ids.shape[1] + max_new[i]) // 16 + 4
        from pyre.paged_cache import PagedKVCache
        c = PagedKVCache.for_model(cfg, num_blocks, "cuda", torch.float16)
        out = model.generate_paged(ids, max_new[i], paged_cache=c, seq_id=0)
        reference[i] = out[0, ids.shape[1]:].tolist()

    # Batched: all three through one scheduler.
    sched = Scheduler(model, cfg, num_blocks=256, max_batch=8, device="cuda")
    id_map = {}
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids[0].tolist()
        sid = sched.add_request(ids, max_new[i])
        id_map[sid] = i

    results = sched.run()

    for sid, gen in results.items():
        i = id_map[sid]
        assert gen == reference[i], (
            f"prompt {i} differs batched vs alone\n"
            f"  alone:   {reference[i][:8]}\n"
            f"  batched: {gen[:8]}"
        )


@requires_cuda
@pytest.mark.gpu
def test_scheduler_frees_all_blocks():
    """After draining, every block must be back in the pool -- no leaks across
    requests, or long-running servers would slowly run out of memory."""
    from transformers import AutoTokenizer

    from pyre.loader import load_model
    from pyre.scheduler import Scheduler

    model, cfg = load_model(MODEL, device="cuda", dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL)

    sched = Scheduler(model, cfg, num_blocks=256, max_batch=4, device="cuda")
    for p in ["Hello world", "def add(a, b):", "The year is"]:
        sched.add_request(tok(p, return_tensors="pt").input_ids[0].tolist(), 16)
    sched.run()

    assert sched.cache.allocator.num_used == 0, "scheduler leaked KV blocks"
    assert not sched.running and not sched.waiting


@requires_cuda
@pytest.mark.gpu
def test_radix_prefix_matches_no_radix():
    """Requests sharing a long prefix, served with prefix reuse, must produce
    token-identical output to the same requests with radix disabled. Reuse
    changes what is recomputed, never what is generated."""
    from transformers import AutoTokenizer

    from pyre.loader import load_model
    from pyre.scheduler import Scheduler

    model, cfg = load_model(MODEL, device="cuda", dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL)

    preamble = "You are a careful assistant. Answer precisely and concisely. "
    prompts = [preamble + q for q in ["What is a hash table?", "What is a stack?", "What is a queue?"]]

    def run(use_radix):
        sched = Scheduler(model, cfg, num_blocks=512, max_batch=8, device="cuda")
        sched.use_radix = use_radix
        idmap = {}
        for i, p in enumerate(prompts):
            sid = sched.add_request(tok(p, return_tensors="pt").input_ids[0].tolist(), 24)
            idmap[sid] = i
        res = sched.run()
        return {idmap[sid]: g for sid, g in res.items()}

    with_radix = run(True)
    without = run(False)
    for i in range(len(prompts)):
        assert with_radix[i] == without[i], f"prompt {i}: radix reuse changed the output"
