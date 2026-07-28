"""Load HuggingFace Qwen2 weights into PyreQwen2.

Uses safetensors directly rather than going through ``transformers`` so that
pyre has no runtime dependency on the HF modelling code — only on the file
format. ``transformers`` stays a dev/benchmark dependency.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from pyre.config import ModelConfig
from pyre.model import PyreQwen2


def resolve_model_path(model_id_or_path: str) -> Path:
    """Return a local directory containing config.json and *.safetensors.

    If given a HF repo id, downloads it via huggingface_hub. On Kaggle, set
    HF_HOME to a working directory so the download survives within a session.
    """
    p = Path(model_id_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download  # imported lazily

    return Path(snapshot_download(model_id_or_path))


def load_model(
    model_id_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[PyreQwen2, ModelConfig]:
    """Build PyreQwen2 and fill it with checkpoint weights.

    fp16, not bf16: Turing (sm_75) has no bf16 tensor-core path, so bf16 falls
    back to a slow emulated route. Every number in RESULTS.md is fp16.
    """
    path = resolve_model_path(model_id_or_path)
    cfg = ModelConfig.from_json(path / "config.json")

    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors found in {path}")

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(shard)))

    # Qwen2.5-1.5B ties lm_head to the embedding matrix, so the checkpoint
    # simply omits lm_head.weight. Materialise it rather than leaving the
    # randomly-initialised parameter in place — that failure mode produces
    # fluent-looking garbage and is miserable to debug.
    if "lm_head.weight" not in state:
        if not cfg.tie_word_embeddings:
            raise KeyError("lm_head.weight missing but tie_word_embeddings is false")
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    model = PyreQwen2(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Anything unexpected means the checkpoint layout drifted from what
    # model.py assumes. Fail loudly here instead of producing wrong tokens.
    hard_missing = [k for k in missing if not k.startswith("_")]
    if hard_missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch.\n  missing: {hard_missing[:8]}\n  unexpected: {unexpected[:8]}"
        )

    model = model.to(device=device, dtype=dtype).eval()
    # Inference only: nothing here needs a gradient. Without this, autograd
    # tracks every forward -- which costs time, and makes RoPE tables built
    # under inference_mode unusable in a normally-tracked call later.
    model.requires_grad_(False)
    return model, cfg
