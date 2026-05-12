#!/usr/bin/env python3
"""
Stage B helper: validate multimodal-style JSONL (screenshot + messages) for UI Artisan.

Rows are expected from ``scripts/generate_ui_artisan_synthetic.py``:

  {"messages": [...], "screenshot_path": "datasets/.../screenshot.png", ...}

This script:
  1) Verifies each ``screenshot_path`` exists relative to ``versions/v1.4`` (or ``--base_dir``).
  2) Optionally embeds the path into the first user message for text-only QLoRA audit trails.
  3) Writes ``messages``-only JSONL suitable for ``train_ui_v14.py`` when you are not yet running
     true multimodal SFT (CLIP projector / native VLM batches differ by model — integrate there).

Usage::

  cd versions/v1.4
  python train_ui_vision_v14.py --input datasets/ui_artisan_synthetic/stage_b_multimodal.jsonl \\
      --output datasets/ui_artisan_synthetic/stage_b_text_proxy.jsonl

Then::

  python train_ui_v14.py --dataset datasets/ui_artisan_synthetic/stage_b_text_proxy.jsonl --output_dir adapters/0002
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _embed_path_in_user(msgs: List[Dict[str, str]], shot: str) -> List[Dict[str, str]]:
    out = [dict(m) for m in msgs]
    for i, m in enumerate(out):
        if (m.get("role") or "").lower() == "user":
            out[i] = {
                "role": "user",
                "content": f"[screenshot_file={shot}]\n" + str(m.get("content") or ""),
            }
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--base_dir", type=str, default="", help="Directory to resolve screenshot_path from (default: parent of this script)")
    ap.add_argument("--embed_path", action="store_true", help="Prefix user message with screenshot path for text-only training")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent
    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(inp.resolve())
    export: List[Dict[str, Any]] = []
    for r in rows:
        shot = str(r.get("screenshot_path") or "").strip()
        if not shot:
            raise SystemExit(f"Row missing screenshot_path: {r.get('case_id')}")
        abs_shot = (base / shot).resolve()
        if not abs_shot.is_file():
            raise SystemExit(f"Screenshot not found: {abs_shot}")
        msgs = r.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise SystemExit(f"Row missing messages: {r.get('case_id')}")
        typed_msgs = [{"role": str(m.get("role")), "content": str(m.get("content") or "")} for m in msgs if isinstance(m, dict)]
        if args.embed_path:
            typed_msgs = _embed_path_in_user(typed_msgs, shot)
        export.append({"messages": typed_msgs, "source_case": r.get("case_id"), "screenshot_path": shot})

    with open(out_path, "w", encoding="utf-8") as f:
        for row in export:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[train_ui_vision_v14] wrote {len(export)} rows → {out_path}", flush=True)


if __name__ == "__main__":
    main()
