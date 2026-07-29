"""OpenAI-compatible inference server.

Week 8. Wraps the pyre engine behind the OpenAI Chat Completions API, so any
OpenAI client library — the official SDK, curl, LangChain, whatever — points at
this and just works. This is what turns the engine from code into a service:

    curl localhost:8000/v1/chat/completions \\
      -d '{"model":"pyre","messages":[{"role":"user","content":"hi"}],"stream":true}'

Endpoints:
  GET  /health                    liveness
  GET  /v1/models                 lists the served model
  POST /v1/chat/completions       chat, with optional SSE streaming

Streaming uses Server-Sent Events in the exact OpenAI delta format, so a client
sees tokens appear one at a time. Non-streaming returns the standard completion
object.

The model is loaded once at startup and held in a single Scheduler; requests are
served through the same continuous-batching engine built across weeks 1–7.
"""

from __future__ import annotations

import json
import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="pyre", description="An LLM inference engine for Turing GPUs")

# populated at startup by load()
STATE: dict = {"model": None, "tok": None, "cfg": None, "name": None}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "pyre"
    messages: list[Message]
    max_tokens: int = 128
    stream: bool = False


def load(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cuda"):
    """Load the model once. Called at startup or manually before serving."""
    from transformers import AutoTokenizer

    from pyre.loader import load_model

    model, cfg = load_model(model_id, device=device, dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id)
    STATE.update(model=model, tok=tok, cfg=cfg, name=model_id)


def _prompt_ids(messages: list[Message]) -> list[int]:
    """Apply the model's chat template to the message list."""
    tok = STATE["tok"]
    chat = [{"role": m.role, "content": m.content} for m in messages]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids[0].tolist()


def _generate(prompt_ids: list[int], max_tokens: int):
    """Yield decoded token strings one at a time using the paged cache."""
    model, tok, cfg = STATE["model"], STATE["tok"], STATE["cfg"]
    from pyre.paged_cache import BLOCK_SIZE, PagedKVCache

    device = next(model.parameters()).device
    num_blocks = (len(prompt_ids) + max_tokens) // BLOCK_SIZE + 8
    cache = PagedKVCache.for_model(cfg, num_blocks, device, torch.float16)
    cache.add_sequence(0)

    ids = torch.tensor([prompt_ids], device=device)
    logits = model.forward_paged(ids, cache, 0, start_pos=0)
    nxt = int(logits[0, -1].argmax())
    pos = len(prompt_ids)
    eos = tok.eos_token_id

    for _ in range(max_tokens):
        if nxt == eos:
            break
        yield tok.decode([nxt], skip_special_tokens=True)
        step = torch.tensor([[nxt]], device=device)
        logits = model.forward_paged(step, cache, 0, start_pos=pos)
        nxt = int(logits[0, -1].argmax())
        pos += 1


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["model"] is not None}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": STATE["name"] or "pyre", "object": "model", "owned_by": "pyre"}
    ]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if STATE["model"] is None:
        return {"error": "model not loaded"}
    prompt_ids = _prompt_ids(req.messages)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        def event_stream():
            for piece in _generate(prompt_ids, req.max_tokens):
                chunk = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": piece},
                                 "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            done = {
                "id": cid, "object": "chat.completion.chunk",
                "created": created, "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = "".join(_generate(prompt_ids, req.max_tokens))
    return {
        "id": cid, "object": "chat.completion", "created": created, "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt_ids),
                  "completion_tokens": len(text.split()),
                  "total_tokens": len(prompt_ids) + len(text.split())},
    }
