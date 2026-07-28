"""Single entry point for Kaggle / RunPod runs.

The Kaggle notebook stays four lines and holds no logic, so a session wipe
costs nothing. Everything that matters is in git.

Notebook cell:
    !git clone https://github.com/Lalitaditya-tickoo/pyre.git
    %cd pyre
    !pip install -e . -q
    !python scripts/kaggle_run.py --stage baseline

Then download bench/results/*.json, commit it from your Mac, and update
RESULTS.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        sys.exit(rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baseline", "parity", "all"], default="all")
    ap.add_argument("--tag", default="w1", help="prefix for the results filename")
    args = ap.parse_args()

    if args.stage in ("baseline", "all"):
        run([sys.executable, "bench/baseline_hf.py", "--out", f"bench/results/{args.tag}_hf.json"])

    if args.stage in ("parity", "all"):
        run([sys.executable, "-m", "pytest", "tests/", "-v"])

    print("\nDone. Download bench/results/ and commit it.")


if __name__ == "__main__":
    main()
