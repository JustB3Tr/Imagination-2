"""
HTTP API chat: prepend the same system prompt as the Gradio path (identity + global_memory.txt + rules).

Without this, /api/chat only saw raw user/assistant turns and missed inject_internal_instructions().
"""
from __future__ import annotations

from typing import Dict, List

from imagination_runtime.think_tags import THINK_FORMAT_INSTRUCTION


def enrich_messages_with_imagination_system(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Prepend server system prompt; drop any client-sent system (server is source of truth)."""
    from imagination_runtime.paths import resolve_root_path
    from imagination_runtime.users import load_global_memory
    from imagination_v1_4 import build_system_prompt

    root = resolve_root_path(None)
    gm = load_global_memory(root)
    system_prompt = build_system_prompt(
        root,
        gm,
        "",
        "",
        "",
        "",
        "",
        "",
        thread_kind=None,
    )
    system_prompt = f"{system_prompt}\n\n{THINK_FORMAT_INSTRUCTION}"
    rest = [dict(m) for m in msgs if (m.get("role") or "").strip().lower() != "system"]
    return [{"role": "system", "content": system_prompt}] + rest
