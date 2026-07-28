"""HuggingFace ``generate()`` baseline.

This produces row one of RESULTS.md. Everything PYRE does for the next seven
weeks is measured as a multiple of this number, so run it first and commit the
JSON before touching anything else.

Usage (Kaggle T4):
    python bench/baseline_hf.py --batch-sizes 1 4 8 16 32 --out bench/results/w1_hf.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import harness, prompts

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_backend(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=None
    ).to(device).eval()

    @torch.inference_mode()
    def generate(batch_prompts: list[str], max_new_tokens: int) -> list[str]:
        enc = tok(batch_prompts, return_tensors="pt", padding=True).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,   # no early EOS, so token counts are exact
            do_sample=False,                 # greedy: comparable across backends
            use_cache=True,
            pad_token_id=tok.pad_token_id,
        )
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate, tok, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--suites", nargs="+", default=["short", "shared_prefix"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="bench/results/w1_hf.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device. Numbers from CPU are not comparable to anything.")

    generate, _, _ = build_backend(args.model, device)

    results = []
    for suite in args.suites:
        for bs in args.batch_sizes:
            batch = prompts.get_suite(suite, bs)
            try:
                m = harness.measure(
                    generate, batch, args.max_new_tokens,
                    backend="hf", suite=suite, repeats=args.repeats,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at suite={suite} bs={bs} — recording as the HF ceiling and stopping this suite.")
                torch.cuda.empty_cache()
                break
            results.append(m)
            print(f"  done suite={suite} bs={bs}: {m.total_tps:.1f} tok/s total, {m.ttft_s:.3f}s TTFT")

    print()
    harness.print_table(results)
    path = harness.save(results, args.out, extra={"model": args.model})
    print(f"\nwrote {path}")
    print("Now commit this file and paste the numbers into RESULTS.md.")


if __name__ == "__main__":
    main()
