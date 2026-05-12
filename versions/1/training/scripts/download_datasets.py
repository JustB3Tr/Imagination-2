#!/usr/bin/env python3
"""
Download / build coding instruction JSONL for Imagination 2 training.

Default: runs the monorepo ``scripts/hf_build_train_jsonl.py`` (fact-dense HF coding mix)
and writes under ``versions/v1.4/training/datasets/downloaded/``.

Prereq::

  pip install -r versions/v1.4/training/requirements.txt
  pip install -r scripts/requirements_train_export.txt

Examples::

  cd versions/v1.4/training
  python scripts/download_datasets.py --target 2000

  # Forward extra flags to hf_build_train_jsonl.py after ``--``::
  python scripts/download_datasets.py --target 5000 -- \\
    --min-fact-score 2.5 --backfill-floor 1.5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _training_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    # training/scripts/this.py -> parents[1]=training, [2]=v1.4, [3]=versions, [4]=repo
    return Path(__file__).resolve().parents[4]


def main() -> None:
    p = argparse.ArgumentParser(description="Build coding JSONL via hf_build_train_jsonl.py")
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL path (default: datasets/downloaded/coding_hf_export.jsonl under training/)",
    )
    p.add_argument("--target", type=int, default=5000, help="Row target passed to hf_build_train_jsonl.py")
    p.add_argument(
        "--hf-script",
        type=str,
        default="",
        help="Override path to hf_build_train_jsonl.py (default: <repo>/scripts/hf_build_train_jsonl.py)",
    )
    args, forwarded = p.parse_known_args()
    training = _training_root()
    repo = _repo_root()
    hf_script = Path(args.hf_script) if args.hf_script else repo / "scripts" / "hf_build_train_jsonl.py"
    if not hf_script.is_file():
        raise SystemExit(
            f"hf_build_train_jsonl.py not found at:\n  {hf_script}\n"
            "Clone the full imagination-ai monorepo or pass --hf-script explicitly."
        )

    out = Path(args.output) if args.output else training / "datasets" / "downloaded" / "coding_hf_export.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(hf_script),
        "--output",
        str(out.resolve()),
        "--target",
        str(int(args.target)),
        # Never pull random repo .md/.txt unless user passes e.g. -- --docs-dir ... after this.
        "--doc-quota",
        "0",
    ]
    if forwarded:
        cmd.extend(forwarded)

    print(
        "\n[download_datasets] Hugging Face often prints several small README.md downloads first — "
        "those are dataset *cards* going into the Hub cache, not your JSONL training rows. "
        "Streaming the ``train`` split comes next; this can take many minutes on Colab.\n",
        flush=True,
    )
    print("[download_datasets] Running:\n  " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    rc = subprocess.call(cmd, cwd=str(repo), env=env)
    if rc != 0:
        raise SystemExit(rc)
    print(f"[download_datasets] Done → {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
