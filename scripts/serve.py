"""Start the pyre server.

    python scripts/serve.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8000

Loads the model, then serves the OpenAI-compatible API on the given port.
"""

import argparse

import uvicorn

from pyre.server.app import app, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"loading {args.model} on {args.device} ...")
    load(args.model, args.device)
    print(f"pyre serving on http://{args.host}:{args.port}  (OpenAI-compatible)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
