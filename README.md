# PYRE

**A from-scratch LLM inference engine for Turing-generation GPUs (T4, sm_75).**
Paged KV cache, continuous batching, a hand-written Triton attention kernel,
prefix caching, exact speculative decoding, and an OpenAI-compatible server —
built end to end in Python + Triton, ~2000 lines, 39 tests.

FlashAttention-2 requires Ampere (sm_80+). The T4 — what Colab, Kaggle, and the
cheapest cloud tiers actually hand out — is Turing. So the fused attention kernel
here is written by hand, because on this hardware there is no library to fall
back on. That constraint is the whole point of the project.

```bash
python scripts/serve.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
curl localhost:8000/v1/chat/completions \
  -d '{"model":"pyre","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Any OpenAI client library points at this and works unchanged.

## Results

All numbers: Qwen2.5, fp16, single NVIDIA T4. Full methodology in [RESULTS.md](RESULTS.md).

**Memory — paged vs contiguous KV cache** (batch 32, max_len 1024):

| workload | contiguous | paged | reduction |
|---|---|---|---|
| short-heavy (real traffic) | 896 MB | 145 MB | **6.2×** |
| exponential lengths | 896 MB | 170 MB | 5.3× |

**Throughput — fused Triton kernel vs unfused batching** (Qwen2.5-0.5B):

| batch | unfused | fused kernel | speedup |
|---|---|---|---|
| 1 | 27.9 | 37.9 | 1.36× |
| 32 | 59.1 | **138.2** | **2.34×** |

The speedup *grows* with batch size — the signature of removing a launch-bound
bottleneck.

**Correctness is the through-line.** The forward pass is bit-identical to
HuggingFace across all layers. Every optimization — KV cache, paging, the Triton
kernel, batching, prefix reuse, speculation — is gated on producing
token-identical output to a trusted reference before its benchmark counts.

## What's inside

| | Component | What it does |
|---|---|---|
| 1 | Qwen2 forward pass | From scratch: RMSNorm, RoPE, GQA, SwiGLU. Bit-exact vs HF. |
| 2 | KV cache | O(n) decode. Throughput flat across a 32× range in sequence length. |
| 3 | Paged KV cache | 16-token blocks from a shared pool + per-sequence block tables. 6.2× less memory. |
| 4 | Continuous batching | Sequences join and leave mid-flight; finished slots reused immediately. |
| 5 | Triton paged-attention kernel | Fused, block-table-aware, online-softmax. 2.34× on sm_75. |
| 6 | Radix prefix cache | Shared prefixes reuse KV blocks via ref-counting instead of recomputing. |
| 7 | Speculative decoding | 0.5B draft proposes, 1.5B target verifies. Exact; 84% draft acceptance on prose. |
| 8 | OpenAI-compatible server | FastAPI, `/v1/chat/completions`, live SSE streaming. |

## Run it

```bash
git clone https://github.com/Lalitaditya-tickoo/pyre.git
cd pyre && pip install -e ".[dev]"
pytest tests/ -v          # CPU tests run anywhere; GPU tests skip without CUDA
```

On a GPU box:

```bash
python bench/bench_memory.py     # paged vs contiguous memory
python bench/bench_batch.py      # continuous-batching throughput
python scripts/serve.py          # start the OpenAI-compatible server
```

## Design notes

- **fp16 only** — Turing has no bf16 tensor-core path.
- **RMSNorm and softmax upcast to fp32** — fp16 variance overflows on large
  activations and NaNs a few layers down.
- **The kernel reads scattered blocks in place** via the block table — no gather,
  which is what makes paging free at attention time.
- **Correctness before speed** — every layer has a token-identical reference, and
  no benchmark is recorded unless its correctness gate is green.

## Scope, honestly

This is a from-scratch engine built to implement and demonstrate the core
techniques of modern LLM serving — not a drop-in replacement for vLLM or TGI.
It is single-GPU, fp16-only, and its kernel is correct and fused but not tuned to
the metal like FlashAttention. Production serving would additionally need tensor/
pipeline parallelism, quantization, and the operational layer (load-aware
batching, observability, autoscaling). Those are deliberately out of scope: the
goal here was to build and prove the ideas, and to be able to say precisely what
separates this from a production stack.

## License

Apache-2.0
