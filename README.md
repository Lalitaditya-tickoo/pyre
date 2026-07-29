# PYRE

A from-scratch LLM inference engine for **Turing-generation GPUs (T4, sm_75)** —
paged KV cache, continuous batching, and a hand-written Triton attention kernel.

FlashAttention-2 requires Ampere (sm_80+). The T4 — what Colab, Kaggle, and the
cheapest cloud tiers actually hand out — is Turing. So the fused attention kernel
here is written by hand, because on this hardware there is no library to fall
back on. That constraint is the whole point of the project.

No vLLM, no TensorRT, no FlashAttention. Python + Triton, ~1500 lines, 29 tests.

## Results

All numbers: Qwen2.5, fp16, single NVIDIA T4. Full methodology in [RESULTS.md](RESULTS.md).

**Memory — paged vs contiguous KV cache** (batch 32, max_len 1024):

| workload | contiguous | paged | reduction |
|---|---|---|---|
| short-heavy (real traffic) | 896 MB | 145 MB | **6.2×** |
| exponential lengths | 896 MB | 170 MB | 5.3× |

**Throughput — fused kernel vs unfused batching** (Qwen2.5-0.5B, gen_len 64):

| batch | unfused | fused kernel | speedup |
|---|---|---|---|
| 1 | 27.9 | 37.9 | 1.36× |
| 32 | 59.1 | **138.2** | **2.34×** |

The speedup grows with batch size — the signature of removing a launch-bound
bottleneck.

**Correctness:** the forward pass is bit-identical to HuggingFace across all
layers; every optimization (KV cache, paging, the Triton kernel) is gated on
producing token-identical output to the reference path before its benchmark
counts.

## How it's built

| | Component | What it does |
|---|---|---|
| 1 | Qwen2 forward pass | From scratch: RMSNorm, RoPE, GQA, SwiGLU. Bit-exact vs HF. |
| 2 | KV cache | O(n) decode. Flat throughput across a 32× range in sequence length. |
| 3 | Paged KV cache | 16-token blocks from a shared pool, per-sequence block tables. 6.2× less memory. |
| 4 | Continuous batching | Sequences join and leave mid-flight; finished slots reused immediately. |
| 5 | Triton paged-attention kernel | Fused, block-table-aware, online-softmax. 2.34× throughput on sm_75. |

## Run it

```bash
git clone https://github.com/Lalitaditya-tickoo/pyre.git
cd pyre && pip install -e ".[dev]"
pytest tests/ -v          # CPU tests run anywhere; GPU tests skip without CUDA
```

On a GPU box:

```bash
python bench/bench_memory.py       # paged vs contiguous memory
python bench/bench_batch.py        # continuous-batching throughput
```

## Design notes

- **fp16 only** — Turing has no bf16 tensor-core path.
- **RMSNorm and softmax upcast to fp32** — fp16 variance overflows on large
  activations and NaNs a few layers down.
- **Correctness before speed** — every layer has a token-identical reference,
  and no benchmark number is recorded unless its correctness gate is green.
- **The kernel reads scattered blocks in place** via the block table — no gather,
  which is what makes paging free at attention time.

## License

Apache-2.0
