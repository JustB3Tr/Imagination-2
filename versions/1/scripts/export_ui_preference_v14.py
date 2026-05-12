#!/usr/bin/env python3
"""
Validate UI Artisan preference JSONL (Stage C / DPO-style pairs).

Each line must decode as JSON with pair_id, prompt, chosen, rejected where chosen/rejected are objects
with at least ``messages`` (list) and optionally ``screenshot_path`` (str).

Usage::

  cd versions/v1.4
  python scripts/export_ui_preference_v14.py --input datasets/ui_preference_pair_example.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _validate_row(obj: Dict[str, Any], base: Path) -> None:
    for k in ("pair_id", "prompt", "chosen", "rejected"):
        if k not in obj:
            raise ValueError(f"missing key {k}")
    for side in ("chosen", "rejected"):
        blob = obj[side]
        if not isinstance(blob, dict):
            raise ValueError(f"{side} must be object")
        msgs = blob.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"{side}.messages required")
        sp = blob.get("screenshot_path")
        if isinstance(sp, str) and sp.strip():
            p = (base / sp.strip()).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"missing screenshot for {side}: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--base_dir", type=str, default="", help="Resolve screenshot_path from this dir (default: versions/v1.4)")
    args = ap.parse_args()
    base = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parents[1]
    path = Path(args.input)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        obj = json.loads(ln)
        if not isinstance(obj, dict):
            raise ValueError(f"line {i}: not an object")
        _validate_row(obj, base)
    print(f"[export_ui_preference_v14] OK — {len(lines)} valid pair(s) in {path}")


if __name__ == "__main__":
    main()
