"""Model configuration, read from a HuggingFace ``config.json``.

Nothing here is hardcoded to a specific checkpoint. Values are pulled from the
config file so the same code path works for Qwen2.5-0.5B (the draft model used
later for speculative decoding) and Qwen2.5-1.5B (the target model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    head_dim: int

    @property
    def n_rep(self) -> int:
        """How many query heads share each key/value head (GQA factor)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        raw = json.loads(Path(path).read_text())

        arch = raw.get("architectures", ["<unknown>"])[0]
        if "Qwen2" not in arch:
            raise ValueError(
                f"pyre currently implements the Qwen2 architecture only, got {arch!r}. "
                "Adding a new architecture means adding a new block in pyre/model.py."
            )

        hidden = raw["hidden_size"]
        n_heads = raw["num_attention_heads"]
        # Qwen2 does not always emit head_dim; it is hidden_size // n_heads by default.
        head_dim = raw.get("head_dim") or hidden // n_heads

        return cls(
            hidden_size=hidden,
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=raw.get("num_key_value_heads", n_heads),
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            max_position_embeddings=raw.get("max_position_embeddings", 32768),
            head_dim=head_dim,
        )
