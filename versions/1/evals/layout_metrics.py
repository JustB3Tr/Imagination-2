#!/usr/bin/env python3
"""
Lightweight layout checks on synthetic or exported HTML fixtures (no browser required).

Parses a small JSON case file and runs string-level heuristics (presence of margin-left, gap patterns).
For pixel-accurate metrics, use Playwright in your own harness or extend this script.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _score_html(html: str, expect_defect: str) -> Dict[str, Any]:
    html_l = html.lower()
    out: Dict[str, Any] = {"expect_defect": expect_defect, "signals": {}}
    m = re.search(r"margin-left:\s*(\d+)px", html_l)
    out["signals"]["margin_left_px"] = int(m.group(1)) if m else None
    if "gap:" in html_l:
        out["signals"]["has_grid_gap"] = True
        g = re.search(r"gap:\s*([^;]+);", html_l)
        out["signals"]["gap_raw"] = g.group(1).strip() if g else None
    if expect_defect == "off_center":
        out["pass"] = bool(m and int(m.group(1)) > 0)
    elif expect_defect == "uneven_gutters":
        out["pass"] = bool(out["signals"].get("gap_raw"))
    else:
        out["pass"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=str, required=True, help="JSON file: list of {name, html_path, expect_defect}")
    args = ap.parse_args()
    p = Path(args.cases)
    data = json.loads(p.read_text(encoding="utf-8"))
    cases: List[Dict[str, Any]] = data if isinstance(data, list) else data.get("cases", [])
    base = p.parent
    results = []
    for c in cases:
        hp = base / str(c.get("html_path") or "")
        html = hp.read_text(encoding="utf-8") if hp.is_file() else ""
        results.append({"name": c.get("name"), **_score_html(html, str(c.get("expect_defect") or ""))})
    passed = sum(1 for r in results if r.get("pass"))
    print(json.dumps({"total": len(results), "passed": passed, "results": results}, indent=2))


if __name__ == "__main__":
    main()
