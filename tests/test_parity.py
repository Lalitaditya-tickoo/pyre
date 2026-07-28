"""The correctness gate.

This is the most important file in the repo. Fast and subtly wrong is the
default failure mode for a hand-written inference engine, and it is very hard
to notice by reading output — a broken GQA mapping or an off-by-one RoPE
offset still produces fluent English.

So: greedy decoding is deterministic, therefore PYRE and HuggingFace must emit
*byte-identical token ids* from the same prompt. Not similar. Identical. This
test stays green from week 1 through week 8; every optimisation is checked
against it before its benchmark number is allowed into RESULTS.md.

Run on a GPU box (Kaggle / RunPod):
    pytest tests/ -v -m gpu

Skipped automatically without CUDA, so local runs on the Mac stay green.
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL = os.environ.get("PYRE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

PROMPTS = [
    "Explain what a hash table is.",
    "The capital of France is",
    "def fibonacci(n):",
]

pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@pytest.fixture(scope="module")
def hf_pair():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="eager").cuda().eval()
    return tok, model


@pytest.fixture(scope="module")
def pyre_model():
    from pyre.loader import load_model

    model, _ = load_model(MODEL, device="cuda", dtype=torch.float16)
    return model


@requires_cuda
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_token_parity(prompt, hf_pair, pyre_model):
    """PYRE and HF must produce identical greedy token ids."""
    tok, hf = hf_pair
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    n = 32

    with torch.inference_mode():
        hf_out = hf.generate(
            ids, max_new_tokens=n, min_new_tokens=n,
            do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id,
            repetition_penalty=1.0,   # Qwen ships 1.1; pyre does pure argmax
        )
    pyre_out = pyre_model.generate_greedy(ids, max_new_tokens=n)

    hf_new = hf_out[0, ids.shape[1]:].tolist()
    pyre_new = pyre_out[0, ids.shape[1]:].tolist()

    if hf_new != pyre_new:
        # Report where they diverge — the index is the single most useful
        # debugging signal. Divergence at step 0 means the prefill/forward is
        # wrong; divergence later usually means cache or position handling.
        first = next((i for i, (a, b) in enumerate(zip(hf_new, pyre_new)) if a != b), 0)
        pytest.fail(
            f"diverged at generated token {first}\n"
            f"  hf:   {hf_new[max(0, first-3):first+3]}\n"
            f"  pyre: {pyre_new[max(0, first-3):first+3]}\n"
            f"  hf text:   {tok.decode(hf_new)!r}\n"
            f"  pyre text: {tok.decode(pyre_new)!r}"
        )


@requires_cuda
def test_logits_close(hf_pair, pyre_model):
    """Tighter than token parity: raw prefill logits must match numerically.

    Catches drift that argmax would hide — two backends can agree on every
    argmax while one of them is quietly accumulating error that will bite at
    longer contexts.
    """
    tok, hf = hf_pair
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.cuda()

    with torch.inference_mode():
        hf_logits = hf(ids).logits.float()
        pyre_logits = pyre_model(ids).float()

    assert hf_logits.shape == pyre_logits.shape
    max_abs = (hf_logits - pyre_logits).abs().max().item()
    assert max_abs < 1e-2, f"max abs logit diff {max_abs:.6f} — forward pass differs from HF eager"


def test_repeat_kv_head_mapping():
    """Query head i must attend to kv head i // n_rep.

    Pure CPU, no model needed, runs everywhere. Guards the single easiest bug
    to introduce in GQA: using repeat() instead of repeat_interleave(), which
    produces plausible-looking but wrong output.
    """
    from pyre.model import repeat_kv

    n_kv, n_rep = 2, 6
    x = torch.arange(n_kv, dtype=torch.float32).view(1, n_kv, 1, 1)
    out = repeat_kv(x, n_rep)

    assert out.shape == (1, n_kv * n_rep, 1, 1)
    for i in range(n_kv * n_rep):
        assert out[0, i, 0, 0].item() == i // n_rep, f"query head {i} mapped to wrong kv head"
