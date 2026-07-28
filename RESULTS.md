# PYRE — Results

All numbers: **Qwen2.5-1.5B-Instruct, fp16, single NVIDIA T4 (sm_75)**,
128 new tokens, greedy, median of 3 runs after 1 warmup.
Raw JSON in `bench/results/`.

---

## Headline

| | HF `generate()` | PYRE | speedup |
|---|---|---|---|
| Throughput @ bs=1 (tok/s) | 28.2 | **35.4** | **1.26×** |
| Throughput @ bs=1, 1024 tok (tok/s) | — | **35.3** | — |
| TTFT @ bs=1 (s) | 0.040 | **0.033** | 1.21× |
| Throughput @ bs=32 (tok/s) | **904.1** | — | — |
| Peak memory @ bs=32 (GB) | 3.11 / 16.0 | — | — |

Targets: **≥3,616 tok/s** (4× HF), within **15%** of vLLM, **beat vLLM on shared-prefix TTFT**.

---

## Week 1 — baseline

```bash
python bench/baseline_hf.py --out bench/results/w1_hf.json
```

| backend | suite | bs | TTFT (s) | decode tok/s | total tok/s | peak GB |
|---|---|---|---|---|---|---|
| hf | short | 1 | 0.040 | 28.2 | 28.2 | 2.89 |
| hf | short | 4 | 0.035 | 126.2 | 126.1 | 2.92 |
| hf | short | 8 | 0.037 | 240.5 | 240.3 | 2.94 |
| hf | short | 16 | 0.044 | 480.4 | 479.2 | 3.03 |
| hf | short | 32 | 0.080 | 913.2 | 904.1 | 3.11 |
| hf | shared_prefix | 1 | 0.042 | 27.8 | 27.8 | 2.90 |
| hf | shared_prefix | 4 | 0.054 | 123.3 | 122.7 | 2.92 |
| hf | shared_prefix | 8 | 0.128 | 242.6 | 237.3 | 2.95 |
| hf | shared_prefix | 16 | 0.218 | 477.5 | 457.7 | 3.03 |
| hf | shared_prefix | 32 | 0.398 | 845.0 | 786.5 | 3.15 |

**Observations**

- **Shared-prefix TTFT is 5× worse than short at bs=32** (0.398s vs 0.080s) despite
  near-identical decode throughput. All 32 requests share an identical 297-character
  preamble, and HF re-prefills it 32 times. This is the target for the week-6 radix
  prefix cache.
- **Peak memory is 3.11 GB of 16 GB.** HF leaves ~80% of the card unused because the
  KV cache is allocated contiguously per sequence. This is the headroom the paged
  cache (week 3) reclaims.
- Decode throughput scales close to linearly to bs=32, so HF is not yet decode-bound
  at this batch size — the ceiling is memory layout, not compute.

---

## Week 2 — contiguous KV cache

```bash
python bench/sweep_length.py --lengths 32 64 128 256 512 1024
```

Batch 1, Qwen2.5-1.5B, fp16, T4. Prompt fixed; generation length swept.

| gen tokens | naive tok/s | cached tok/s | speedup | cache reserved |
|---|---|---|---|---|
| 32 | 32.3 | 33.9 | 1.05× | 1.1 MB |
| 64 | 32.4 | 35.2 | 1.09× | 2.0 MB |
| 128 | 32.9 | 35.6 | 1.08× | 3.7 MB |
| 256 | 27.7 | 34.9 | 1.26× | 7.2 MB |
| 512 | 17.5 | 34.8 | 1.99× | 14.2 MB |
| 1024 | 8.9 | **35.3** | **3.95×** | 28.2 MB |

**Cached throughput is flat at ~35 tok/s across a 32× range in sequence length.**
That is the result. Constant per-token cost is what O(n) decode looks like; the
naive path collapses from 32.3 to 8.9 tok/s over the same range as its O(n²)
attention term grows.

**The cache does almost nothing below 256 tokens, and that is not a defect.**
At batch 1 decode is bound by streaming ~3.1 GB of fp16 weights from HBM per
forward pass. On a T4 at ~320 GB/s that is ~10 ms of unavoidable memory traffic.
Attention over a few hundred tokens costs microseconds by comparison, so the
naive path's redundant work hides entirely inside the bandwidth floor. The cache
only starts paying once the quadratic term climbs out from under it — measured
crossover, 256 tokens.

Two consequences for the rest of the project:

- **Batching, not caching, is the lever for throughput.** One weight read
  amortised across many sequences is where a 4× lives. Week 4 is load-bearing.
- The contiguous cache reserves `prompt_len + max_new_tokens` up front —
  28.2 MB per sequence at 1024 tokens, whether or not the sequence gets there.
  At batch 32 that is ~900 MB reserved for a worst case that mostly will not
  happen. Week 3 is the fix.

PYRE-cached also beats HF `generate()` at batch 1 (35.4 vs 28.2 tok/s, 1.26×),
from lower per-step Python overhead — attention is still unfused until week 5.

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
- **decode tok/s** — `(N-1) * batch / (t_total - t_ttft)`. Excludes prefill.
- **total tok/s** — `N * batch / t_total`. The honest end-to-end number.
- Timing is `torch.cuda.synchronize()`-fenced on both sides.
- Prompt suites are frozen in `bench/prompts.py` and never edited, so a week-8
  number is comparable to a week-1 number.

## Correctness

Greedy decoding is deterministic, so PYRE must match HuggingFace token-for-token
in fp32. In fp16 exact agreement is **not** achievable against HF's fused SDPA
kernel — different accumulation order produces ~4 ULP of logit drift, which flips
near-tie argmaxes. The gate is therefore: exact token match in fp32, plus a
documented logit tolerance in fp16.

## Known limits

- fp16 only. Turing has no bf16 tensor-core path.
- Qwen2 architecture only.
- Single GPU.
