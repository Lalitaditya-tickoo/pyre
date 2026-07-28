"""Measurement harness.

Built in week 1, before any optimisation, because an optimisation you cannot
measure on day three is a guess for the next seven weeks.

Methodology, stated explicitly so the numbers in RESULTS.md are auditable:

  TTFT        wall time of a generate call capped at 1 new token. Includes
              tokenisation, prefill, and one decode step.
  decode_tps  (N-1) * batch / (t_total - t_ttft). Deliberately excludes
              prefill so it does not flatter itself on short prompts.
  total_tps   N * batch / t_total. The honest end-to-end number.

Every measurement is a median over ``repeats`` runs after ``warmup`` discarded
runs. The first CUDA call in a process pays kernel autotuning and allocator
warmup costs that have nothing to do with the code being measured.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

GenerateFn = Callable[[list[str], int], list[str]]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def environment() -> dict:
    """Capture everything needed to reproduce a number, or to explain why it moved."""
    gpu = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {
            "name": p.name,
            "capability": f"sm_{p.major}{p.minor}",
            "total_memory_gb": round(p.total_memory / 1024**3, 2),
            "count": torch.cuda.device_count(),
        }
    return {
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@dataclass
class Measurement:
    backend: str
    suite: str
    batch_size: int
    max_new_tokens: int
    ttft_s: float
    total_s: float
    decode_tps: float
    total_tps: float
    peak_mem_gb: float
    samples_s: list[float] = field(default_factory=list)


def measure(
    generate: GenerateFn,
    prompts: list[str],
    max_new_tokens: int,
    backend: str,
    suite: str,
    warmup: int = 1,
    repeats: int = 3,
) -> Measurement:
    batch = len(prompts)

    for _ in range(warmup):
        generate(prompts, max_new_tokens)
    _sync()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # TTFT — separate call, capped at one token.
    ttfts = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        generate(prompts, 1)
        _sync()
        ttfts.append(time.perf_counter() - t0)

    totals = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        generate(prompts, max_new_tokens)
        _sync()
        totals.append(time.perf_counter() - t0)

    ttft = statistics.median(ttfts)
    total = statistics.median(totals)
    decode_time = max(total - ttft, 1e-6)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

    return Measurement(
        backend=backend,
        suite=suite,
        batch_size=batch,
        max_new_tokens=max_new_tokens,
        ttft_s=round(ttft, 4),
        total_s=round(total, 4),
        decode_tps=round((max_new_tokens - 1) * batch / decode_time, 2),
        total_tps=round(max_new_tokens * batch / total, 2),
        peak_mem_gb=round(peak, 2),
        samples_s=[round(x, 4) for x in totals],
    )


def save(measurements: list[Measurement], out_path: str | Path, extra: dict | None = None) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": environment(),
        "extra": extra or {},
        "measurements": [asdict(m) for m in measurements],
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def print_table(measurements: list[Measurement]) -> None:
    hdr = f"{'backend':<10} {'suite':<14} {'bs':>4} {'ttft(s)':>9} {'decode t/s':>11} {'total t/s':>10} {'mem GB':>7}"
    print(hdr)
    print("-" * len(hdr))
    for m in measurements:
        print(
            f"{m.backend:<10} {m.suite:<14} {m.batch_size:>4} {m.ttft_s:>9.3f} "
            f"{m.decode_tps:>11.1f} {m.total_tps:>10.1f} {m.peak_mem_gb:>7.2f}"
        )
