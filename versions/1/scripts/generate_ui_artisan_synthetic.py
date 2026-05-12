#!/usr/bin/env python3
"""
Synthetic UI Artisan dataset: HTML with known layout defects → Playwright screenshot → JSONL for Stage B.

Usage (from repo root or versions/v1.4):

  python scripts/generate_ui_artisan_synthetic.py --out_dir datasets/ui_artisan_synthetic --count 12

Requires: playwright (`pip install playwright` && `playwright install chromium`).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _case_html(defect: str, offset_px: int) -> str:
    """Single-page HTML with an obvious hero title for centroid heuristics."""
    extra = ""
    if defect == "off_center":
        extra = f"#hero h1 {{ margin-left: {offset_px}px; }}"
    elif defect == "uneven_gutters":
        extra = ".grid {{ gap: 12px 24px 12px; }}"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>ui_artisan synthetic</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }}
  .wrap {{ max-width: 720px; margin: 0 auto; padding: 48px 24px; }}
  #hero h1 {{ font-size: 2rem; text-align: center; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; margin-top: 32px; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 16px; min-height: 80px; }}
  {extra}
</style></head>
<body><div class="wrap"><section id="hero"><h1>Synthetic Hero</h1></section>
<div class="grid"><div class="card">A</div><div class="card">B</div><div class="card">C</div></div>
</div></body></html>"""


def _auto_visual_report(defect: str, offset_px: int, w: int, h: int) -> Dict[str, Any]:
    from imagination_runtime.ui_artisan_protocol import VISUAL_REPORT_SCHEMA_VERSION

    if defect == "off_center":
        summary = f"Hero title shifted right by ~{offset_px}px relative to symmetric center (synthetic)."
        align = [f"Detected defect kind=off_center margin-left={offset_px}px in CSS."]
    elif defect == "uneven_gutters":
        summary = "Card grid shows uneven horizontal spacing between columns (synthetic)."
        align = ["Gutters alternate narrow/wide between columns."]
    else:
        summary = "Synthetic layout sample."
        align = []
    return {
        "schema_version": VISUAL_REPORT_SCHEMA_VERSION,
        "viewport": {"width_px": w, "height_px": h},
        "summary": summary,
        "alignment_notes": align,
        "spacing_notes": [],
        "a11y_flags": ["synthetic_fixture"],
        "regions": [
            {
                "id": "hero_title",
                "label": "Hero title",
                "bbox_norm": {"x": 0.25, "y": 0.08, "w": 0.5, "h": 0.12},
                "role": "heading",
            }
        ],
        "expected_after_next_edit": "Remove the intentional defect CSS so the hero is centered and gutters are even.",
    }


def _screenshot(html_path: Path, png_path: Path, w: int, h: int) -> None:
    from playwright.sync_api import sync_playwright

    url = html_path.resolve().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": w, "height": h})
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(300)
        page.screenshot(path=str(png_path), full_page=True)
        page.close()
        ctx.close()
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="datasets/ui_artisan_synthetic")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--jsonl", type=str, default="stage_b_multimodal.jsonl")
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1]
    os.chdir(here)
    # Ensure imagination_runtime imports resolve when run as script
    import sys

    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    out_root = (here / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / args.jsonl

    patterns: List[Tuple[str, int]] = [("off_center", 20), ("off_center", 32), ("uneven_gutters", 0)]
    rows: List[Dict[str, Any]] = []

    for i in range(int(args.count)):
        defect, offset = patterns[i % len(patterns)]
        case_dir = out_root / f"case_{i:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        html_path = case_dir / "index.html"
        png_path = case_dir / "screenshot.png"
        html_path.write_text(_case_html(defect, offset), encoding="utf-8")
        _screenshot(html_path, png_path, args.width, args.height)
        rel_shot = str(png_path.relative_to(here)).replace("\\", "/")
        rel_html = str(html_path.relative_to(here)).replace("\\", "/")
        report = _auto_visual_report(defect, offset, args.width, args.height)
        user = (
            f"You are reviewing a static UI fixture. Screenshot on disk: {rel_shot}. "
            f"HTML source path: {rel_html}. Describe layout issues and the fix."
        )
        assistant = json.dumps(
            {
                "text": f"Identified synthetic defect {defect}.",
                "visual_report": report,
                "code_fix_hint": "Adjust CSS to remove intentional defect; re-capture to verify.",
            },
            ensure_ascii=False,
        )
        rows.append(
            {
                "case_id": f"case_{i:04d}",
                "defect": defect,
                "offset_px": offset,
                "screenshot_path": rel_shot,
                "html_path": rel_html,
                "messages": [
                    {"role": "system", "content": "You are Imagination UI Artisan. Output JSON with text + visual_report."},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ui_artisan_synthetic] wrote {len(rows)} cases under {out_root}", flush=True)
    print(f"[ui_artisan_synthetic] JSONL: {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
