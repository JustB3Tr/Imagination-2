"""
v1.4 UI Artisan — structured visual self-review protocol (NDJSON + training metadata).

Defines ``visual_report`` JSON shape so the model, frontend, and SFT/DPO datasets share one vocabulary
(regions, alignment, spacing) independent of a specific UI framework.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

VISUAL_REPORT_SCHEMA_VERSION = "1.0"

# Keys expected on a well-formed visual_report (subset enforced by validate_visual_report).
REQUIRED_TOP_LEVEL = ("schema_version", "viewport", "regions")
OPTIONAL_TOP_LEVEL = (
    "summary",
    "alignment_notes",
    "spacing_notes",
    "a11y_flags",
    "expected_after_next_edit",
)


def new_code_revision_id(session_id: str, seq: int) -> str:
    """Stable id for a UI code revision within an agent session."""
    sid = re.sub(r"[^a-zA-Z0-9._-]", "_", (session_id or "session").strip())[:64]
    return f"{sid}-r{int(seq)}"


def _is_region(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    rid = str(obj.get("id") or "").strip()
    label = str(obj.get("label") or "").strip()
    if not rid or not label:
        return False
    box = obj.get("bbox_norm")
    if not isinstance(box, dict):
        return False
    for k in ("x", "y", "w", "h"):
        v = box.get(k)
        if not isinstance(v, (int, float)):
            return False
        if float(v) < 0 or float(v) > 1.0001:
            return False
    return True


def validate_visual_report(obj: Any) -> Tuple[bool, List[str]]:
    """
    Return (ok, issues). Soft validation — allows partial reports but flags missing pieces.
    """
    issues: List[str] = []
    if not isinstance(obj, dict):
        return False, ["visual_report must be a JSON object"]

    ver = str(obj.get("schema_version") or "")
    if ver and ver != VISUAL_REPORT_SCHEMA_VERSION:
        issues.append(f"schema_version {ver!r} != {VISUAL_REPORT_SCHEMA_VERSION!r}")

    vp = obj.get("viewport")
    if not isinstance(vp, dict):
        issues.append("viewport object required")
    else:
        for k in ("width_px", "height_px"):
            v = vp.get(k)
            if not isinstance(v, int) or v <= 0:
                issues.append(f"viewport.{k} must be positive int")

    regions = obj.get("regions")
    if not isinstance(regions, list) or not regions:
        issues.append("regions must be a non-empty array")
    else:
        for i, r in enumerate(regions):
            if not _is_region(r):
                issues.append(f"regions[{i}] invalid (need id, label, bbox_norm.x/y/w/h in 0..1)")

    ok = not any(msg.startswith("viewport") for msg in issues) and not any(
        msg.startswith("regions") for msg in issues
    )
    return ok, issues


def example_visual_report() -> Dict[str, Any]:
    """Reference payload for docs, tests, and prompt few-shots."""
    return {
        "schema_version": VISUAL_REPORT_SCHEMA_VERSION,
        "viewport": {"width_px": 1280, "height_px": 720},
        "summary": "Hero title appears shifted slightly left of the content column center.",
        "alignment_notes": [
            "Title block centroid ~12px left of column center (visually).",
            "Primary CTA aligns with right gutter.",
        ],
        "spacing_notes": ["Card grid gutters: 24px / 22px / 24px — middle column tighter."],
        "a11y_flags": ["heading_contrast_ok", "tap_targets_ok"],
        "regions": [
            {
                "id": "hero_title",
                "label": "Hero title",
                "bbox_norm": {"x": 0.31, "y": 0.12, "w": 0.38, "h": 0.08},
                "role": "heading",
            },
            {
                "id": "cta_primary",
                "label": "Primary CTA",
                "bbox_norm": {"x": 0.62, "y": 0.42, "w": 0.14, "h": 0.06},
                "role": "button",
            },
        ],
        "expected_after_next_edit": "Title centered in the 720px content column; gutters even at 24px.",
    }


def parse_visual_report_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extract a visual_report object from assistant prose (fenced ```json or raw object).
    """
    if not text or not text.strip():
        return None
    t = text.strip()
    for fence in ("```json", "```"):
        if fence in t:
            chunk = t.split(fence, 1)[-1]
            chunk = chunk.split("```", 1)[0].strip()
            t = chunk
            break
    if not (t.startswith("{") and "visual_report" in t):
        # try whole string as JSON
        pass
    try:
        outer = json.loads(t)
        if isinstance(outer, dict) and "visual_report" in outer:
            inner = outer["visual_report"]
            return inner if isinstance(inner, dict) else None
        if isinstance(outer, dict) and outer.get("schema_version") == VISUAL_REPORT_SCHEMA_VERSION:
            return outer
    except Exception:
        pass
    return None


def new_preference_pair_id() -> str:
    return f"ui_pref_{uuid.uuid4().hex[:16]}"


def preference_jsonl_schema_doc() -> str:
    """Documentation string for Stage C DPO-style rows."""
    return (
        "Each JSONL line: {\n"
        '  "pair_id": "ui_pref_...",\n'
        '  "prompt": "user instruction / design brief",\n'
        '  "chosen": {"messages": [...], "screenshot_path": "..."},\n'
        '  "rejected": {"messages": [...], "screenshot_path": "..."}\n'
        "}\n"
        "Screenshots should be renders of the chosen vs rejected HTML for the same prompt."
    )
