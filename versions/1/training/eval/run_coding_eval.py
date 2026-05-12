#!/usr/bin/env python3
"""
Lightweight coding task checks.

1) **Schema mode (default):** validate JSONL task file structure.
2) **Substring mode:** with ``--predictions``, each line must be ``{"id": "...", "text": "..."}``
   and ``checks`` from the task with the same ``id`` must all appear in ``text`` (case-sensitive).

Usage::

  cd versions/v1.4/training
  python eval/run_coding_eval.py --tasks eval/coding_tasks_smoke.jsonl
  python eval/run_coding_eval.py --tasks eval/coding_tasks_smoke.jsonl --predictions eval/predictions_smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, required=True)
    ap.add_argument("--predictions", type=str, default="", help="JSONL: {id, text} per line")
    args = ap.parse_args()
    tasks_path = Path(args.tasks)
    tasks = _load_jsonl(tasks_path)

    for i, t in enumerate(tasks):
        tid = str(t.get("id") or "").strip()
        if not tid:
            raise SystemExit(f"task line {i}: missing id")
        msgs = t.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise SystemExit(f"task {tid}: messages required")
        if not any((m.get("role") or "").lower() == "user" for m in msgs if isinstance(m, dict)):
            raise SystemExit(f"task {tid}: need at least one user message")

    print(f"[eval] OK: {len(tasks)} tasks validated in {tasks_path}", flush=True)

    pred_path = (args.predictions or "").strip()
    if not pred_path:
        return

    preds = _load_jsonl(Path(pred_path))
    by_id = {str(p.get("id")): str(p.get("text") or "") for p in preds if str(p.get("id") or "").strip()}
    failed: List[Tuple[str, str]] = []
    for t in tasks:
        tid = str(t.get("id"))
        text = by_id.get(tid, "")
        checks = t.get("checks") or []
        if not isinstance(checks, list):
            continue
        for c in checks:
            s = str(c)
            if s and s not in text:
                failed.append((tid, f"missing substring {s!r}"))

    if failed:
        for tid, msg in failed:
            print(f"[eval] FAIL {tid}: {msg}", flush=True)
        raise SystemExit(1)
    print(f"[eval] All substring checks passed ({len(tasks)} tasks, predictions from {pred_path})", flush=True)


if __name__ == "__main__":
    main()
