"""
System-prompt hints for web vs no-web turns, plus HTML for the in-chat collapsible trace UI.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List

_BOLD_SEGMENT = re.compile(r"\*\*(.+?)\*\*")


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _format_thinking_step_line(s: str) -> str:
    """
    Escape HTML, but turn **like this** into bold for the thinking UI (inner text escaped).
    """
    text = s or ""
    parts: List[str] = []
    pos = 0
    for m in _BOLD_SEGMENT.finditer(text):
        parts.append(_esc(text[pos : m.start()]))
        parts.append(f'<strong class="thinking-em">{_esc(m.group(1))}</strong>')
        pos = m.end()
    parts.append(_esc(text[pos:]))
    return "".join(parts)


def build_thinking_path(
    *,
    use_web: bool,
    reason: str,
    query: str,
    source_cards: List[Dict[str, Any]],
    conversation_turns: int,
) -> str:
    """
    Short system-prompt hint for the model only. Detailed search/source trace lives in the UI;
    do not repeat step labels or source lists here (avoids the model echoing them in replies).
    """
    _ = (reason, query, source_cards)  # retained for API compatibility with callers
    if use_web:
        return (
            "Web search was used for this turn; excerpts may appear below as retrieved notes. "
            "Use them for factual grounding when relevant. "
            "Answer in plain, concise prose only: do not narrate search steps, label steps, "
            "evaluate sources in the reply, use inline [1]/[2] citations, list URLs, or add a "
            '"Sources used" section—the app UI already shows that trace.'
        )
    lines = ["No web search for this turn; answer from model and conversation context."]
    if conversation_turns > 0:
        lines.append(f"Prior context: {conversation_turns} user/assistant turn(s) in history.")
    return " ".join(lines)


def build_thinking_path_no_web(conversation_turns: int = 0) -> str:
    """Short system hint when no web search is used."""
    return build_thinking_path(
        use_web=False,
        reason="",
        query="",
        source_cards=[],
        conversation_turns=conversation_turns,
    )


def build_thinking_html_open(
    *,
    step_lines: List[str],
    summary_label: str = "Thinking",
    pulse_last_step: bool = False,
) -> str:
    """
    Collapsible block shown while reasoning (open). Uses safe HTML only.
    If pulse_last_step, the last step gets a loading animation (subagent weight load).
    """
    n = len(step_lines)
    parts: List[str] = []
    for i, line in enumerate(step_lines):
        cls = "thinking-step"
        if pulse_last_step and n > 0 and i == n - 1:
            cls += " thinking-step--loading"
        parts.append(f'<div class="{cls}">{_format_thinking_step_line(line)}</div>')
    body = "".join(parts)
    return (
        f'<details class="thinking-block" open><summary class="thinking-summary">'
        f'<span class="thinking-summary-text">{_format_thinking_step_line(summary_label)}</span>'
        f'<span class="thinking-dots" aria-hidden="true"><span></span><span></span><span></span></span>'
        f"</summary><div class=\"thinking-body\">{body}</div></details>"
    )


def build_thinking_html_collapsed(*, summary: str, step_lines: List[str]) -> str:
    """Collapsed details after answer; user can expand to see steps."""
    body = "".join(f'<div class="thinking-step">{_format_thinking_step_line(line)}</div>' for line in step_lines)
    return (
        f'<details class="thinking-block thinking-done"><summary class="thinking-summary">'
        f"{_format_thinking_step_line(summary)}</summary><div class=\"thinking-body\">{body}</div></details>"
    )


def compose_assistant_display(thinking_html: str, answer_part: str) -> str:
    """Visible assistant bubble: thinking block + answer (markdown/plain)."""
    ap = answer_part or ""
    if not ap.strip():
        return thinking_html
    return f"{thinking_html}\n\n{ap}"
