# PYRE

An LLM inference engine for **Turing-generation GPUs (T4, sm_75)**, written from scratch.

FlashAttention-2 requires Ampere or newer, and modern inference stacks have
deprioritised the older path. Meanwhile T4s are what Colab, Kaggle, and the
cheapest cloud tiers actually hand out. PYRE targets that gap: a paged KV
cache, a continuous-batching scheduler, and a hand-written Triton attention
kernel, tuned for the hardware most people learning this stuff are stuck with.

Python + Triton. No vLLM, no TensorRT, no FlashAttention.

> **Status: week 1 of 8.** Forward pass and benchmark harness only.
> See [RESULTS.md](RESULTS.md) for current numbers.

## What is actually implemented here

| | Component | Week |
|---|---|---|
| ✅ | Qwen2 forward pass, from scratch (RMSNorm, RoPE, GQA, SwiGLU) | 1 |
| ✅ | Benchmark harness + HF baseline | 1 |
| ✅ | Greedy-token parity gate vs HuggingFace | 1 |
| ⬜ | Contiguous KV cache | 2 |
| ⬜ | Paged KV cache — block allocator, block tables, copy-on-write | 3 |
| ⬜ | Continuous batching — chunked prefill, preemption + recompute | 4 |
| ⬜ | Triton paged-attention decode kernel, fused RMSNorm/RoPE/SwiGLU | 5 |
| ⬜ | Radix-tree prefix cache | 6 |
| ⬜ | Speculative decoding | 7 |
| ⬜ | OpenAI-compatible server | 8 |

## Correctness

Greedy decoding is deterministic, so PYRE must emit **token-identical** output
to HuggingFace from the same prompt. Not similar — identical. Fast-and-subtly-wrong
is the default failure mode for hand-written inference code, and a broken GQA
head mapping still produces fluent English. The parity test is the gate: no
benchmark number enters RESULTS.md unless it is green.

```bash
pytest tests/ -v          # CPU tests run anywhere; GPU tests skip without CUDA
```

## Install

```bash
git clone https://github.com/Lalitaditya-tickoo/pyre.git
cd pyre
pip install -e ".[dev]"
```

## Run the baseline

```bash
python bench/baseline_hf.py --out bench/results/w1_hf.json
```

On Kaggle or RunPod, the notebook holds no logic:

```python
!git clone https://github.com/Lalitaditya-tickoo/pyre.git
%cd pyre
!pip install -e . -q
!python scripts/kaggle_run.py --stage baseline
```

Download `bench/results/`, commit from your local machine, update RESULTS.md.
Nothing worth keeping ever lives in a notebook session.

## Design notes

- **fp16, never bf16.** Turing has no bf16 tensor-core path; it falls back to
  a slow emulated route. Every number here is fp16.
- **RMSNorm and softmax upcast to fp32.** fp16 variance overflows on
  large-magnitude activations and produces NaNs several layers downstream.
- **RoPE tables built in fp32.** At `rope_theta=1e6` the low-frequency terms
  lose too much fp16 precision to stay stable over long contexts.
- **Attention is written out by hand**, not delegated to SDPA, because it is
  the exact computation the Triton kernel replaces in week 5.

## License

Apache-2.0
