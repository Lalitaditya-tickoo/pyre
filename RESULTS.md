# PYRE — Results

All numbers: **Qwen2.5-1.5B-Instruct, fp16, single NVIDIA T4 (sm_75)**,
128 new tokens, greedy, median of 3 runs after 1 warmup.
Reproduce any row with the command in its section. Raw JSON in `bench/results/`.

Every row is gated on `tests/test_parity.py` passing — a backend that does not
produce token-identical output to HuggingFace does not get a row.

---

## Headline

| | HF `generate()` | PYRE | speedup |
|---|---|---|---|
| Throughput @ bs=32 (tok/s) | _pending_ | — | — |
| TTFT @ bs=1 (s) | _pending_ | — | — |
| Max concurrent sequences | _pending_ | — | — |

Targets: **≥4× HF throughput at bs=32**, within **15%** of vLLM aggregate,
**beat vLLM on shared-prefix TTFT**.

---

## Week 1 — baseline

```bash
python bench/baseline_hf.py --out bench/results/w1_hf.json
```

| backend | suite | bs | TTFT (s) | decode tok/s | total tok/s | peak GB |
|---|---|---|---|---|---|---|
| hf | short | 1 | | | | |
| hf | short | 4 | | | | |
| hf | short | 8 | | | | |
| hf | short | 16 | | | | |
| hf | short | 32 | | | | |
| hf | shared_prefix | 32 | | | | |

Notes:
-

---

## Week 2 — contiguous KV cache
## Week 3 — paged KV cache
## Week 4 — continuous batching
## Week 5 — Triton paged-attention kernel
## Week 6 — radix prefix cache
## Week 7 — speculative decoding
## Week 8 — vLLM comparison

---

## Methodology

- **TTFT** — wall time of a generate call capped at 1 new token; includes
  tokenisation, prefill, one decode step.
- **decode tok/s** — `(N-1) * batch / (t_total - t_ttft)`. Excludes prefill so
  it does not flatter itself on short prompts.
- **total tok/s** — `N * batch / t_total`. The honest end-to-end number.
- Timing is `torch.cuda.synchronize()`-fenced on both sides.
- Prompt suites are frozen in `bench/prompts.py` and never edited, so a week-8
  number is comparable to a week-1 number.

## Known limits

- fp16 only. Turing has no bf16 tensor-core path.
- Qwen2 architecture only.
- Single GPU.
