"""Continuous batching scheduler (week 4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pyre.paged_cache import PagedKVCache


@dataclass
class Request:
    seq_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    eos_id: int | None = None
    generated: list[int] = field(default_factory=list)
    done: bool = False

    @property
    def cur_len(self) -> int:
        return len(self.prompt_ids) + len(self.generated)


class Scheduler:
    def __init__(self, model, cfg, num_blocks, max_batch, device="cuda", dtype=torch.float16):
        self.model = model
        self.cfg = cfg
        self.max_batch = max_batch
        self.device = device
        self.cache = PagedKVCache.for_model(cfg, num_blocks, device, dtype)
        self.waiting = []
        self.running = []
        self._next_seq = 0

    def add_request(self, prompt_ids, max_new_tokens, eos_id=None):
        r = Request(self._next_seq, list(prompt_ids), max_new_tokens, eos_id)
        self._next_seq += 1
        self.waiting.append(r)
        return r.seq_id

    def _admit(self):
        while self.waiting and len(self.running) < self.max_batch:
            r = self.waiting.pop(0)
            self.cache.add_sequence(r.seq_id)
            ids = torch.tensor([r.prompt_ids], device=self.device)
            logits = self.model.forward_paged(ids, self.cache, r.seq_id, start_pos=0)
            nxt = int(logits[0, -1].argmax())
            r.generated.append(nxt)
            if nxt == r.eos_id or len(r.generated) >= r.max_new_tokens:
                r.done = True
            self.running.append(r)

    def _decode_step(self):
        if not self.running:
            return
        seq_ids = [r.seq_id for r in self.running]
        last = torch.tensor([[r.generated[-1]] for r in self.running], device=self.device)
        positions = [r.cur_len - 1 for r in self.running]
        logits = self.model.forward_paged_batch(last, self.cache, seq_ids, positions)
        for i, r in enumerate(self.running):
            nxt = int(logits[i, -1].argmax())
            r.generated.append(nxt)
            if nxt == r.eos_id or len(r.generated) >= r.max_new_tokens:
                r.done = True

    def _evict(self):
        for r in self.running:
            if r.done:
                self.cache.free_sequence(r.seq_id)
        self.running = [r for r in self.running if not r.done]

    def run(self):
        results = {}
        while self.waiting or self.running:
            self._admit()
            self._decode_step()
            for r in self.running:
                if r.done:
                    results[r.seq_id] = r.generated
            self._evict()
        return results
