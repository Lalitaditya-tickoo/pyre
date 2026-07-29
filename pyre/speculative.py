"""Speculative decoding.

Week 7. Plain decode runs one target forward pass per token. Speculative
decoding uses a cheap draft model to propose K tokens ahead, then verifies all K
with a SINGLE target forward pass. Because decode is memory-bound, the target
processes K positions for nearly the cost of one — so when the draft guesses
right, several tokens are produced per target pass.

The accept/reject rule makes it EXACT: the output is token-identical to greedy
decoding with the target alone. The draft only ever proposes; the target's own
distribution decides. This is the property that matters — a latency win with
zero quality change.

Algorithm (greedy variant):
  1. draft proposes tokens d_1..d_K autoregressively from the current context
  2. target runs once over [context, d_1..d_K], giving its own greedy token at
     each of the K+1 positions: t_0 (at the last context pos), t_1..t_K
  3. accept d_i while d_i == t_{i-1}; stop at the first mismatch
  4. emit the accepted draft tokens plus t_j at the mismatch (the target's
     correction), which is always valid because the target computed it
  5. repeat from the new context

Expected speedup ≈ average accepted length per target pass. High on predictable
text where the small draft agrees with the target; ~1x worst case (never worse
in tokens, only in wasted draft compute).
"""

from __future__ import annotations

import torch


@torch.inference_mode()
def speculative_generate(target, draft, input_ids, max_new_tokens, k=4, eos_id=None):
    """Greedy speculative decoding. Returns (output_ids, stats).

    target, draft: PyreQwen2 models sharing a tokenizer/vocab.
    input_ids: (1, S). k: draft lookahead length.
    stats: dict with total target passes and accepted-token counts, for
    computing the realised speedup.
    """
    device = input_ids.device
    ids = input_ids
    generated = 0
    target_passes = 0
    accepted_total = 0

    while generated < max_new_tokens:
        # 1. draft proposes k tokens autoregressively (cheap model)
        draft_ctx = ids
        proposals = []
        for _ in range(k):
            dlogits = draft.forward(draft_ctx)
            dtok = int(dlogits[0, -1].argmax())
            proposals.append(dtok)
            draft_ctx = torch.cat([draft_ctx, torch.tensor([[dtok]], device=device)], dim=1)

        # 2. target verifies all k in ONE pass over [ids, proposals]
        cand = torch.cat([ids, torch.tensor([proposals], device=device)], dim=1)
        tlogits = target.forward(cand)                       # (1, S+k, vocab)
        target_passes += 1

        # target's greedy token at each position from the last context token on
        S = ids.shape[1]
        target_toks = [int(tlogits[0, S - 1 + i].argmax()) for i in range(k + 1)]

        # 3. accept proposals while they match the target's own choice
        n_accept = 0
        for i in range(k):
            if proposals[i] == target_toks[i]:
                n_accept += 1
            else:
                break

        # 4. emit accepted drafts + the target's correction token
        emit = proposals[:n_accept] + [target_toks[n_accept]]
        accepted_total += n_accept

        for t in emit:
            ids = torch.cat([ids, torch.tensor([[t]], device=device)], dim=1)
            generated += 1
            if generated >= max_new_tokens or (eos_id is not None and t == eos_id):
                stats = {"target_passes": target_passes,
                         "accepted": accepted_total,
                         "generated": generated,
                         "tokens_per_pass": generated / target_passes}
                return ids, stats

    stats = {"target_passes": target_passes,
             "accepted": accepted_total,
             "generated": generated,
             "tokens_per_pass": generated / target_passes}
    return ids, stats
