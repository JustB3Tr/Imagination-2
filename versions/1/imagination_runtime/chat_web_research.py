"""
Optional web follow-ups for HTTP chat: model may end a reply with [[WEB_QUERY:...]]; we fetch
DDGS snippets and continue the conversation in-process (no extra user round-trip).

Disabled when IMAGINATION_CHAT_WEB=0.
Max web fetches: IMAGINATION_CHAT_WEB_FETCHES (default 8).
Unknown / obscure questions add IMAGINATION_CHAT_WEB_UNKNOWN_FETCH_BONUS (default 4), capped at 12.

When IMAGINATION_CHAT_WEB_PREFETCH=1 (default), almost every user turn gets a *small* DDGS prefetch
(IMAGINATION_CHAT_WEB_PREFETCH_LIGHT, default 4 results). Time-sensitive or “unknown” turns use a
larger prefetch (IMAGINATION_CHAT_WEB_PREFETCH_HEAVY, default 12).
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterator, List, Optional

_WEB_OPEN = "[[WEB_QUERY:"
_WEB_RE = re.compile(r"\[\[WEB_QUERY:([^\]]+)\]\]\s*$", re.DOTALL)


def web_followups_enabled() -> bool:
    return (os.getenv("IMAGINATION_CHAT_WEB") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def max_web_fetches() -> int:
    raw = (os.getenv("IMAGINATION_CHAT_WEB_FETCHES") or "8").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 8
    return max(0, min(n, 12))


def effective_max_web_fetches(user_text: str) -> int:
    """Cap follow-up [[WEB_QUERY:...]] rounds; higher when the question looks obscure or hard to satisfy."""
    base = max_web_fetches()
    if base <= 0:
        return 0
    if not query_seems_unknown_or_obscure(user_text):
        return base
    try:
        bonus = int((os.getenv("IMAGINATION_CHAT_WEB_UNKNOWN_FETCH_BONUS") or "4").strip())
    except ValueError:
        bonus = 4
    return max(0, min(12, base + max(0, bonus)))


def prefetch_snippet_budget(*, heavy: bool) -> int:
    """DDGS result count for initial prefetch: small by default, larger when web is strongly indicated."""
    key = "IMAGINATION_CHAT_WEB_PREFETCH_HEAVY" if heavy else "IMAGINATION_CHAT_WEB_PREFETCH_LIGHT"
    default = "12" if heavy else "4"
    raw = (os.getenv(key) or default).strip()
    try:
        n = int(raw)
    except ValueError:
        n = int(default)
    return max(2, min(16, n))


def followup_snippet_budget(*, heavy: bool) -> int:
    """DDGS result count for each [[WEB_QUERY:...]] continuation."""
    raw = (os.getenv("IMAGINATION_CHAT_WEB_FOLLOWUP_RESULTS") or "").strip()
    if raw:
        try:
            n = int(raw)
            return max(2, min(16, n))
        except ValueError:
            pass
    return 12 if heavy else 8


# Broad heuristics: prefer web unless the question is clearly timeless / historical-only.
_TIME_SENSITIVE = re.compile(
    r"(?i)\b("
    r"202[3-9]|203[0-9]|20[4-9][0-9]|"  # years 2023+
    r"today|tonight|yesterday|tomorrow|right now|currently|as of|latest|recent|recently|breaking|"
    r"this week|last week|this month|last month|this year|last year|upcoming|"
    r"news|headline|announced|released|launch|changelog|version\s*\d|patch|update|upgrade|"
    r"stock|share price|market cap|crypto|bitcoin|ethereum|"
    r"weather|forecast|temperature|hurricane|earthquake|"
    r"who won|final score|playoff|super bowl|world cup|olympics|medal|"
    r"mlb|nba|nfl|nhl|ncaa|home runs?|standings|season stats|box score|"
    r"election|poll|senate|house vote|president\b|prime minister|resigned|impeach|"
    r"ceo\b|ipo\b|earnings|quarterly|fed rate|inflation|gdp|"
    r"according to (who|what)|cite|source|verify|fact[- ]check"
    r")\b"
)

_HISTORICAL_ANCHOR = re.compile(
    r"(?i)\b("
    r"ancient|roman empire|medieval|renaissance|world war i\b|wwi|world war ii\b|wwii|"
    r"19th century|18th century|before christ|bc\b|julius caesar|napoleon\s+1812"
    r")\b"
)

_TRIVIAL_USER = re.compile(
    r"(?is)^\s*(hi+|hello|hey+|thanks?|thank you|bye|ok(ay)?|cool|nice)\b[!.]?\s*$"
)


def query_seems_unknown_or_obscure(user_text: str) -> bool:
    """
    Heuristic: topic is niche, ambiguous, or likely underrepresented in training — warrant extra web depth.
    """
    t = (user_text or "").strip()
    if len(t) < 12:
        return False
    if re.search(
        r"(?i)\b("
        r"obscure|niche|arcane|esoteric|little-known|hard to find|rarely (mentioned|covered)|"
        r"never heard|can't find|cannot find|no one knows|who even is|what even is|"
        r"obscure (book|band|paper|library|api|tool)|"
        r"edge case|corner case|undocumented"
        r")\b",
        t,
    ):
        return True
    if len(t) > 140 and ("?" in t or t.count(",") >= 3):
        return True
    if len(t) > 90 and re.search(r"(?i)\b(or|versus|vs\.?)\b", t) and "?" in t:
        return True
    return False


def _skip_lightweight_prefetch(user_text: str) -> bool:
    """No DDGS call for trivial turns (saves latency); model can still request [[WEB_QUERY:...]]."""
    t = (user_text or "").strip()
    if len(t) < 4:
        return True
    if _TRIVIAL_USER.match(t):
        return True
    return False


def _needs_large_prefetch(user_text: str) -> bool:
    """Wider DDGS pull: live or fast-moving facts, or niche/ambiguous questions (not every generic why-question)."""
    t = (user_text or "").strip()
    if not t or _skip_lightweight_prefetch(t):
        return False
    if query_seems_unknown_or_obscure(t):
        return True
    if _HISTORICAL_ANCHOR.search(t) and not _TIME_SENSITIVE.search(t):
        return False
    return bool(_TIME_SENSITIVE.search(t))


def user_likely_needs_web(user_text: str) -> bool:
    """True when the question almost certainly needs current or verifiable web facts."""
    t = (user_text or "").strip()
    if len(t) < 4:
        return False
    if _HISTORICAL_ANCHOR.search(t) and not _TIME_SENSITIVE.search(t):
        return False
    if _TIME_SENSITIVE.search(t):
        return True
    # Default: bias toward web for general knowledge questions (model cutoff ~2022).
    if len(t) > 12 and "?" in t:
        return True
    if len(t) > 40:
        return True
    return False


def web_prefetch_enabled() -> bool:
    return (os.getenv("IMAGINATION_CHAT_WEB_PREFETCH") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _last_user_content(msgs: List[Dict[str, str]]) -> str:
    for m in reversed(msgs):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _prefetch_query_from_user(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) > 220:
        return t[:220].rsplit(" ", 1)[0] + "…"
    return t or "recent news"


def _augment_last_user_with_prefetch(msgs: List[Dict[str, str]], ctx: str, *, heavy: bool) -> List[Dict[str, str]]:
    if not msgs:
        return msgs
    out: List[Dict[str, str]] = [dict(m) for m in msgs]
    if out[-1].get("role") != "user":
        return msgs
    u = (out[-1].get("content") or "").strip()
    depth = "broader" if heavy else "brief"
    block = (
        f"[Auto web excerpts — {depth} prefetch; use when relevant; say if insufficient.]\n\n"
        f"{ctx}\n\n---\n\nUser message:\n{u}"
    )
    out[-1] = {**out[-1], "content": block}
    return out


def inject_web_system_message(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    last_u = _last_user_content(msgs)
    needs = user_likely_needs_web(last_u)
    deep = query_seems_unknown_or_obscure(last_u)
    strong_web = needs or deep
    extra = (
        "Your general knowledge may be incomplete or outdated (training data roughly through Dec 2022). "
        "For anything after that window, current events, live data, prices, sports results, politics, "
        "weather, product versions, or verifiable facts, do not invent specifics. "
        "A short web prefetch may already appear under the user message; use it to ground answers. "
        "When a web lookup would still help, end your assistant message with a single final line exactly "
        "(nothing after it):\n"
        f"{_WEB_OPEN}your concise English search query]]\n"
    )
    if strong_web:
        extra += (
            "\n\nFor **this** turn, prioritize web-backed answers: include that "
            f"{_WEB_OPEN}...]] line unless the question is purely timeless (e.g. established math, "
            "ancient history with no current-events angle). If excerpts are already injected above your "
            "message, use them honestly; add [[WEB_QUERY:...]] for another angle if needed."
        )
        if deep:
            extra += (
                "\n\nThis question looks niche, ambiguous, or easy to get wrong from memory alone — "
                "prefer multiple targeted searches over guessing."
            )
    else:
        extra += (
            "\n\nFor simple or timeless questions you may omit the web line if memory is enough. "
            "If unsure, use [[WEB_QUERY:...]] rather than inventing specifics."
        )
    if not msgs:
        return [{"role": "system", "content": extra}]
    if msgs[0].get("role") == "system":
        merged = list(msgs)
        first = (merged[0].get("content") or "").strip()
        merged[0] = {**merged[0], "content": (extra + "\n\n" + first).strip()}
        return merged
    return [{"role": "system", "content": extra}] + list(msgs)


def strip_web_marker_for_stream(cumulative: str) -> str:
    """Hide partial or complete [[WEB_QUERY:...]] suffix from streamed display."""
    i = cumulative.find(_WEB_OPEN)
    if i == -1:
        return cumulative
    return cumulative[:i].rstrip()


def remove_web_marker(full: str) -> str:
    return _WEB_RE.sub("", (full or "").strip()).strip()


def extract_web_query(full: str) -> Optional[str]:
    m = _WEB_RE.search((full or "").strip())
    if not m:
        return None
    q = (m.group(1) or "").strip()
    return q or None


def fetch_web_context(query: str, *, max_results: int = 8) -> str:
    query = (query or "").strip()
    if not query:
        return "(empty search query)"
    lines: List[str] = []
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results), 1):
                title = (r.get("title") or "").strip()
                snippet = (r.get("body") or "").strip()
                url = (r.get("href") or "").strip()
                lines.append(f"{i}. {title}\n   {snippet}\n   {url}")
    except Exception as e:
        return f"(web search failed: {e})"
    if not lines:
        return "(no search results; try a shorter or broader query)"
    return "\n".join(lines)


BUDGET_EXHAUSTED_USER = (
    "Web search budget for this reply is exhausted. Do not request another [[WEB_QUERY:...]]. "
    "Summarize whatever was useful from prior web excerpts (if any), clearly state if there was "
    "not enough relevant or verified information for a complete answer, and do not fabricate facts."
)


def iterate_display_text_with_web(
    *,
    base_messages: List[Dict[str, str]],
    max_new_tokens: int,
    stream_native,
) -> Iterator[str]:
    """
    Yields cumulative **visible** assistant text for the UI (web marker stripped while streaming).

    ``stream_native(messages, max_new_tokens)`` must be the existing _stream_native iterator.
    """
    working = inject_web_system_message(base_messages)
    last_u = _last_user_content(base_messages)
    heavy = bool(last_u and _needs_large_prefetch(last_u))
    if web_prefetch_enabled() and last_u and not _skip_lightweight_prefetch(last_u):
        try:
            q = _prefetch_query_from_user(last_u)
            n_results = prefetch_snippet_budget(heavy=heavy)
            ctx = fetch_web_context(q, max_results=n_results)
            working = _augment_last_user_with_prefetch(working, ctx, heavy=heavy)
        except Exception as e:
            ctx = f"(prefetch failed: {e})"
            working = _augment_last_user_with_prefetch(working, ctx, heavy=heavy)

    total_shown = ""
    fetches = 0
    max_f = effective_max_web_fetches(last_u) if last_u else max_web_fetches()
    followup_heavy = heavy

    while True:
        full_text = ""
        for chunk in stream_native(working, max_new_tokens):
            full_text = chunk
            disp = strip_web_marker_for_stream(full_text)
            yield total_shown + ("\n\n" if total_shown else "") + disp

        clean_assistant = remove_web_marker(full_text)
        query = extract_web_query(full_text)

        if not query:
            break

        if fetches >= max_f:
            working.append({"role": "assistant", "content": clean_assistant})
            working.append({"role": "user", "content": BUDGET_EXHAUSTED_USER})
            total_shown = total_shown + ("\n\n" if total_shown else "") + clean_assistant
            full_text2 = ""
            for chunk in stream_native(working, max_new_tokens):
                full_text2 = chunk
                disp2 = strip_web_marker_for_stream(full_text2)
                yield total_shown + ("\n\n" if total_shown else "") + disp2
            break

        fetches += 1
        ctx = fetch_web_context(query, max_results=followup_snippet_budget(heavy=followup_heavy))
        pack = (
            f"Web search results for query: {query}\n\n{ctx}\n\n"
            "Use the above to answer accurately. If excerpts conflict or are insufficient, say so."
        )
        working.append({"role": "assistant", "content": clean_assistant})
        working.append({"role": "user", "content": pack})
        total_shown = total_shown + ("\n\n" if total_shown else "") + clean_assistant


def final_text_from_stream(gen: Iterator[str]) -> str:
    last = ""
    for s in gen:
        last = s
    return last
