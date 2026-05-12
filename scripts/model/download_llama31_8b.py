#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def ensure_pkg(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        print(f"[download] installing {name} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", name])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Llama-3.1-8B-Instruct into local model folder")
    p.add_argument("--repo-id", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--target-dir", required=True)
    p.add_argument("--revision", default="main")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target = Path(args.target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    print(f"[download] Repo: {args.repo_id}", flush=True)
    print(f"[download] Target: {target}", flush=True)
    print(f"[download] Revision: {args.revision}", flush=True)

    ensure_pkg("huggingface_hub")
    from huggingface_hub import snapshot_download

    try:
        local = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            revision=args.revision,
            resume_download=True,
        )
    except Exception:  # noqa: BLE001
        print("[download] failed. If auth is missing, run: huggingface-cli login", file=sys.stderr)
        raise

    print(f"[download] complete: {local}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
