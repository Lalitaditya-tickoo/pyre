"""Localise where PYRE diverges from HuggingFace.

The parity test tells you *that* the forward pass differs. This tells you
*where*. It hooks every decoder layer in both models, runs the same input
through both, and reports the max absolute difference in hidden states after
each layer.

Reading the output:

  Divergence grows smoothly, roughly doubling every few layers
      -> accumulated floating-point noise. Two numerically different but
         individually correct implementations. Expected in fp16.

  Divergence is ~0 for layers 0..k then jumps by orders of magnitude at k+1
      -> a real bug, localised to layer k+1.

  Divergence is large at layer 0
      -> the bug is in embeddings, RoPE, or the first attention block.

Two flags matter:

  --dtype float32   Removes fp16 noise entirely. If tokens match exactly in
                    fp32, the implementation is correct and the fp16 gap is
                    numerical, not a bug. This is the decisive test.

  --attn eager      HF defaults to SDPA, which fuses attention and accumulates
                    the softmax and PV matmul differently to a hand-written
                    matmul chain. "eager" makes HF do what pyre/model.py does,
                    so it isolates implementation differences from kernel
                    differences.

Usage:
    python scripts/diagnose_parity.py --dtype float32 --attn eager
    python scripts/diagnose_parity.py --dtype float16 --attn eager
    python scripts/diagnose_parity.py --dtype float16 --attn sdpa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyre.loader import load_model  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain what a hash table is."


def _tensor_of(out):
    """HF decoder layers return a tensor in newer versions, a tuple in older."""
    return out[0] if isinstance(out, tuple) else out


def collect(model, layers, input_ids):
    """Run a forward pass, capturing the output of every layer."""
    captured: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(lambda m, i, o: captured.append(_tensor_of(o).detach().float()))
        for layer in layers
    ]
    try:
        with torch.inference_mode():
            logits = model(input_ids)
            logits = logits.logits if hasattr(logits, "logits") else logits
    finally:
        for h in handles:
            h.remove()
    return captured, logits.detach().float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="eager")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--tokens", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a CUDA device")

    dtype = getattr(torch, args.dtype)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation=args.attn
        )
        .cuda()
        .eval()
    )
    pyre, _ = load_model(args.model, device="cuda", dtype=dtype)

    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()

    hf_hidden, hf_logits = collect(hf, hf.model.layers, ids)
    pyre_hidden, pyre_logits = collect(pyre, pyre.model.layers, ids)

    print(f"\nmodel={args.model}  dtype={args.dtype}  hf_attn={args.attn}")
    print(f"prompt={args.prompt!r}  seq_len={ids.shape[1]}\n")
    print(f"{'layer':>6} {'max_abs_diff':>14} {'rel_growth':>12}")
    print("-" * 34)

    prev = None
    for i, (a, b) in enumerate(zip(hf_hidden, pyre_hidden)):
        d = (a - b).abs().max().item()
        growth = f"{d / prev:.2f}x" if prev and prev > 0 else "-"
        print(f"{i:>6} {d:>14.6f} {growth:>12}")
        prev = d

    logit_diff = (hf_logits - pyre_logits).abs().max().item()
    print(f"\nfinal logits max abs diff: {logit_diff:.6f}")
    print(f"logit magnitude (hf max abs): {hf_logits.abs().max().item():.2f}")

    # Greedy agreement is what actually matters downstream.
    with torch.inference_mode():
        hf_gen = hf.generate(
            ids, max_new_tokens=args.tokens, min_new_tokens=args.tokens,
            do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id,
        )
    pyre_gen = pyre.generate_greedy(ids, max_new_tokens=args.tokens)

    hf_new = hf_gen[0, ids.shape[1]:].tolist()
    pyre_new = pyre_gen[0, ids.shape[1]:].tolist()

    if hf_new == pyre_new:
        print(f"\ngreedy tokens: IDENTICAL over {args.tokens} tokens")
    else:
        first = next((i for i, (a, b) in enumerate(zip(hf_new, pyre_new)) if a != b), 0)
        print(f"\ngreedy tokens: diverge at index {first} of {args.tokens}")
        # How close was the decision at the divergence point? A near-tie means
        # noise flipped an argmax; a wide margin means something is wrong.
        with torch.inference_mode():
            ctx = torch.cat([ids, torch.tensor([hf_new[:first]], device=ids.device, dtype=ids.dtype)], dim=1)
            lg = pyre(ctx)[0, -1].float()
        top = lg.topk(2)
        print(f"  pyre top-2 logits at that step: {top.values.tolist()} "
              f"(margin {(top.values[0] - top.values[1]).item():.4f})")
        print(f"  hf chose {hf_new[first]}, pyre chose {pyre_new[first]}")


if __name__ == "__main__":
    main()
