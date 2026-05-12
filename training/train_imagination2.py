#!/usr/bin/env python3
"""
Top-level training entrypoint for Imagination 2.

This wrapper delegates to the baseline runtime training script in versions/1:
  versions/1/training/scripts/train_imagination2_coding.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "versions" / "1" / "training" / "scripts" / "train_imagination2_coding.py"
    if not target.is_file():
        print(f"[train_imagination2] missing training target:\n  {target}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(target), *sys.argv[1:]]
    print("[train_imagination2] forwarding to:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
