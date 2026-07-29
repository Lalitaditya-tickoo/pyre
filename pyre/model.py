"""Qwen2 forward pass, written from scratch.

Module and parameter names mirror the HuggingFace Qwen2 checkpoint layout so
weights load with a plain load_state_dict. Do not rename anything here without
also updating pyre/loader.py.

Two decode paths coexist on purpose:
  generate_greedy  -- no cache, recomputes everything, O(n^2). The trusted
                      reference, proven bit-identical to HF in week 1.
  generate_cached  -- KV cache, O(n). Checked against the reference for exact
                      token equality in tests/test_cache.py.
"""

from __future__ import annotations

import torch
from torch import nn

from pyre.config import ModelConfig


class RMSNorm(nn.Module):
    """The float32 upcast is not optional: fp16 variance overflows on
    large-magnitude activations and produces NaNs several layers downstream.
    HF does the same thing; we match it exactly because parity is checked at
    the token level."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(dtype)


def build_rope_cache(seq_len, head_dim, theta, device, dtype):
    """cos/sin tables of shape (seq_len, head_dim), built in fp32 then cast.
    At rope_theta=1e6 the low-frequency terms lose too much fp16 precision to
    stay stable over long contexts."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: (B, H, S, D). cos, sin: (S, D) -- broadcast over batch and heads."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, KV, S, D) -> (B, KV*n_rep, S, D). Query head i must attend to kv head
    i // n_rep. repeat_interleave gives that; repeat gives the wrong mapping and
    produces output that is subtly wrong rather than obviously broken."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        # Qwen2 carries bias on q/k/v and none on o_proj.
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, cache=None, layer_idx: int = 0, start_pos: int = 0):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)

        # RoPE is applied to the new tokens only, at their absolute positions.
        # Cached keys were already rotated when written, which is why the cache
        # can be reused verbatim instead of re-rotated every step.
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, start_pos, k, v)

        k = repeat_kv(k, self.cfg.n_rep)
        v = repeat_kv(v, self.cfg.n_rep)

        # Written out by hand rather than calling SDPA: this is the exact
        # computation the Triton kernel replaces in week 5.
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Query i sits at absolute position start_pos + i and may attend to keys
        # 0 .. start_pos + i. During decode S == 1 and every key is visible, so
        # the mask is all-zero -- skip building it rather than allocate a (1, T)
        # tensor of zeros on every single token.
        if S > 1:
            T = scores.shape[-1]
            neg = torch.finfo(scores.dtype).min
            mask = torch.full((S, T), neg, device=x.device, dtype=scores.dtype)
            mask = torch.triu(mask, diagonal=start_pos + 1)
            scores = scores + mask

        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None, layer_idx: int = 0, start_pos: int = 0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx, start_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _Inner(nn.Module):
    """Exists only so parameter names come out as model.layers.0.… , matching
    the HuggingFace checkpoint."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class PyreQwen2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = _Inner(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._rope_len = 0
        self._cos = None
        self._sin = None

    def _rope(self, start_pos: int, seq_len: int, device, dtype):
        """cos/sin for absolute positions start_pos .. start_pos+seq_len."""
        need = start_pos + seq_len
        if self._cos is None or need > self._rope_len or self._cos.device != device:
            n = max(need, 512)
            self._cos, self._sin = build_rope_cache(
                n, self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
            self._rope_len = n
        return self._cos[start_pos:need], self._sin[start_pos:need]

    def forward(self, input_ids: torch.Tensor, cache=None, start_pos: int = 0) -> torch.Tensor:
        """input_ids: (B, S) -> logits (B, S, vocab).

        With a cache, input_ids holds only the new tokens and start_pos is how
        many are already cached. Without one, the full sequence is recomputed --
        the week-1 reference path, kept working so the two can be compared.
        """
        B, S = input_ids.shape
        x = self.model.embed_tokens(input_ids)
        cos, sin = self._rope(start_pos, S, x.device, x.dtype)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, i, start_pos)
        x = self.model.norm(x)
        return self.lm_head(x)

    @torch.inference_mode()
    def generate_greedy(self, input_ids, max_new_tokens: int, eos_id: int | None = None):
        """Naive greedy decode, recomputing the whole forward every step.
        Do not optimise this -- its job is to stay obviously correct so later
        versions have something to be checked against."""
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return ids

    @torch.inference_mode()
    def generate_cached(self, input_ids, max_new_tokens: int, eos_id: int | None = None, cache=None):
        """Greedy decode with a KV cache. Same tokens as generate_greedy,
        without the quadratic recompute: one prefill over the prompt, then one
        forward per token with S=1.

        The cache is sized prompt_len + max_new_tokens because a contiguous
        cache must reserve its worst case up front -- the thing week 3 fixes.
        """
        from pyre.cache import KVCache

        B, S = input_ids.shape
        if cache is None:
            cache = KVCache.for_model(
                self.cfg, batch=B, max_seq=S + max_new_tokens,
                device=input_ids.device, dtype=self.lm_head.weight.dtype,
            )

        logits = self.forward(input_ids, cache, start_pos=0)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = [nxt]
        pos = S

        for _ in range(max_new_tokens - 1):
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            logits = self.forward(nxt, cache, start_pos=pos)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out.append(nxt)
            pos += 1

        return torch.cat([input_ids, *out], dim=1)

    @torch.inference_mode()
    def forward_paged(self, input_ids, paged_cache, seq_id, start_pos):
        """Forward pass reading/writing KV through the paged cache.

        Same math as forward(); the only difference is where keys and values
        live. Each layer appends its new k/v to the sequence's blocks, then
        gathers the full prefix back out. gather() is the slow, obvious version
        that materialises a contiguous tensor so the week-2 attention runs
        unchanged; week 5's kernel reads the blocks in place and deletes it.
        """
        B, S = input_ids.shape
        assert B == 1, "paged decode path is single-sequence; batching is week 4"
        x = self.model.embed_tokens(input_ids)
        cos, sin = self._rope(start_pos, S, x.device, x.dtype)

        for i, layer in enumerate(self.model.layers):
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(h).view(B, S, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(B, S, attn.n_kv, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(h).view(B, S, attn.n_kv, attn.head_dim).transpose(1, 2)
            q, k = apply_rope(q, k, cos, sin)

            # store the new tokens (S, n_kv, hd), then read the whole prefix back
            paged_cache.append(i, seq_id, k[0].transpose(0, 1), v[0].transpose(0, 1))
            gk, gv = paged_cache.gather(i, seq_id)          # (T, n_kv, hd)
            kk = gk.transpose(0, 1).unsqueeze(0)            # (1, n_kv, T, hd)
            vv = gv.transpose(0, 1).unsqueeze(0)

            kk = repeat_kv(kk, self.cfg.n_rep)
            vv = repeat_kv(vv, self.cfg.n_rep)

            scores = torch.matmul(q, kk.transpose(-1, -2)) * attn.scale
            if S > 1:
                T = scores.shape[-1]
                neg = torch.finfo(scores.dtype).min
                mask = torch.full((S, T), neg, device=x.device, dtype=scores.dtype)
                mask = torch.triu(mask, diagonal=start_pos + 1)
                scores = scores + mask
            probs = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
            o = torch.matmul(probs, vv).transpose(1, 2).reshape(B, S, attn.n_heads * attn.head_dim)
            x = x + attn.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        return self.lm_head(self.model.norm(x))

    @torch.inference_mode()
    def forward_paged_batch(self, tokens, paged_cache, seq_ids, positions):
        """One batched decode step, attention fused in the Triton paged kernel.

        Week 5. Replaces the week-4 Python loop over sequences. append() runs
        first (it may allocate a new block when a sequence crosses a block
        boundary), then the block table is read, then one paged_attention call
        computes all B sequences. Token-identical to single-sequence decode;
        ~2.3x faster than the unfused path at batch 32, growing with batch size.
        """
        from pyre.kernels.paged_attn import paged_attention

        B, S = tokens.shape
        assert S == 1, "batched path is decode-only"
        x = self.model.embed_tokens(tokens)
        device = x.device
        seq_lens = [positions[i] + 1 for i in range(B)]

        for li, layer in enumerate(self.model.layers):
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(h).view(B, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(B, 1, attn.n_kv, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(h).view(B, 1, attn.n_kv, attn.head_dim).transpose(1, 2)

            # append first — may grow a block table on a boundary crossing
            for i, sid in enumerate(seq_ids):
                cos, sin = self._rope(positions[i], 1, device, x.dtype)
                qi, ki = apply_rope(q[i:i + 1], k[i:i + 1], cos, sin)
                q[i:i + 1] = qi
                paged_cache.append(li, sid, ki[0].transpose(0, 1), v[i].transpose(0, 1))

            # read the block table AFTER appends, so new blocks are included
            max_blocks = max(len(paged_cache.block_tables[s]) for s in seq_ids)
            bt = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
            for i, sid in enumerate(seq_ids):
                tbl = paged_cache.block_tables[sid]
                bt[i, :len(tbl)] = torch.tensor(tbl, dtype=torch.int32, device=device)

            qk = q[:, :, 0, :]
            o = paged_attention(qk, paged_cache.k[li], paged_cache.v[li],
                                bt, seq_lens, self.cfg.n_rep, attn.scale)
            o = o.reshape(B, 1, attn.n_heads * attn.head_dim)
            x = x + attn.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        return self.lm_head(self.model.norm(x))

    @torch.inference_mode()
    def generate_paged(self, input_ids, max_new_tokens, paged_cache, seq_id, eos_id=None):
        """Greedy decode against the paged cache. Must produce identical tokens
        to generate_cached; the storage layout is the only thing that changed."""
        B, S = input_ids.shape
        paged_cache.add_sequence(seq_id)

        logits = self.forward_paged(input_ids, paged_cache, seq_id, start_pos=0)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = [nxt]
        pos = S
        for _ in range(max_new_tokens - 1):
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            logits = self.forward_paged(nxt, paged_cache, seq_id, start_pos=pos)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out.append(nxt)
            pos += 1
        return torch.cat([input_ids, *out], dim=1)
