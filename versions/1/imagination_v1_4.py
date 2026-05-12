# ============================================================
# IMAGINATION v1.4.0 — UI Artisan line (multimodal main model; branched from v1.3)
#
# - Minimal dark UI: model/mode bar + composer (message + chat), status, progress bar
# - No exposed model settings (root from env only)
# - Model lifecycle: load on demand, auto-unload after timeout
# - Response cloaking: module output rewritten by main model
# - Internal instructions + topic memory (invisible to user)
# ============================================================

from __future__ import annotations

import html
import os
import queue
import re
import time
import uuid
import warnings
from collections import OrderedDict
from threading import Lock, Thread
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gradio as gr
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextIteratorStreamer,
)

from imagination_runtime.cache import RuntimeCache
from imagination_runtime.cloaker import build_cloak_messages
from imagination_runtime.internal_prompt import inject_internal_instructions, load_tutor_math_policy
from imagination_runtime.model_lifecycle import (
    ModelLifecycle,
    KEY_TO_LABEL,
)
from imagination_runtime.paths import ModelPaths, resolve_root_path
from imagination_runtime.registry import TaskId, get_task_specs
from imagination_runtime.thinking import (
    build_thinking_path,
    build_thinking_path_no_web,
    build_thinking_html_open,
    build_thinking_html_collapsed,
    compose_assistant_display,
)
from imagination_runtime.auth import (
    account_markdown,
    login_email_password,
    resolve_user_from_gradio_oauth,
    session_dict_from_user,
    signup_email_password,
)
from imagination_runtime.session_signed import user_id_from_cookie
from imagination_runtime.users import (
    THREAD_KIND_GENERAL,
    THREAD_KIND_TUTOR,
    create_user_chat_thread,
    delete_user_chat_thread,
    get_active_thread_id,
    get_user_by_id,
    get_user_chat_thread,
    list_user_chat_threads,
    load_global_memory,
    load_last_subagent_context,
    load_learner_profile,
    load_relevant_topics,
    load_trusted,
    load_user_chat_state,
    load_user_memory,
    pick_default_thread_id,
    rename_user_chat_thread,
    save_last_subagent_context,
    save_learner_profile,
    save_topic_memory,
    save_user_chat_thread,
    save_user_chat_state,
    save_user_memory,
    set_active_thread_id,
)
from imagination_runtime.vlm_infer import generate_stream_vlm, load_main_model_auto
from imagination_runtime.training_log import append_training_turn
from imagination_runtime.web import fetch_page, get_domain, refine_search_query, sources_to_cards, web_search

warnings.filterwarnings("ignore")

# Web: large DDGS pool, keep many sources for RAG (trimmed by max_sources after ranking).
SEARCH_RESULTS = 48
MAX_SOURCES = 14
MAX_CHARS_PER_SOURCE = 2400
REQUEST_TIMEOUT_S = 12
# Main chat: desired upper bound; each request is clamped to remaining context (prompt + margin).
# Generation normally ends when the model emits EOS—max_new_tokens is only a ceiling, not a target.
# Lowered default 8192 -> 1024 so a single chat turn (especially VLM on Colab, ~5-15 tok/s) does not
# stream for many minutes before hitting EOS. Per-request override via `max_new_tokens` still works,
# and IMAGINATION_MAIN_MAX_NEW_TOKENS can raise the ceiling for long-form answers.
_MAIN_MAX_NEW_TOKENS_DEFAULT = 1024
_MAIN_MAX_NEW_TOKENS_CEILING = 131_072
# Coder/research models need a much higher cap or long programs truncate mid-token (e.g. Snake game).
# Override coder cap (trusted / private setups): IMAGINATION_CODER_MAX_NEW_TOKENS=100000
# Hard cap 250_000 so a typo does not hang forever; the real limit is the model context window.
_CODER_MAX_NEW_TOKENS_CEILING = 250_000
CODER_MAX_NEW_TOKENS = 16384
MODULE_REASONING_MAX_NEW_TOKENS = 8192
_REASONING_MAX_NEW_TOKENS_CEILING = 131_072
SHOW_TYPING_CURSOR = True
TYPING_CURSOR = "▌"

TRUSTED_DOMAINS_STARTER = [
    "reuters.com", "apnews.com", "bbc.com", "npr.org", "pbs.org",
    "nytimes.com", "washingtonpost.com", "wsj.com", "bloomberg.com", "theguardian.com",
    "who.int", "cdc.gov", "nih.gov", "ncbi.nlm.nih.gov", "nasa.gov", "noaa.gov", "usgs.gov",
]

# Composer / header mode dropdown: internal value -> (TaskId, force_web for main chat)
CHAT_MODE_DEFAULT = "chat"
CHAT_MODE_CHOICES: List[Tuple[str, str]] = [
    ("Imagination 1.3.0 (auto subagents)", "chat"),
    ("Search the web", "web"),
    ("Coder", "coder"),
]
MODEL_VERSION_DEFAULT = "i13"
MODEL_VERSION_CHOICES: List[Tuple[str, str]] = [
    ("Imagination 1.3.0", "i13"),
]


def _resolve_chat_mode(mode_key: Optional[str]) -> Tuple[TaskId, bool]:
    k = (mode_key or CHAT_MODE_DEFAULT).strip().lower()
    if k == "web":
        return TaskId.CHAT_MAIN, True
    if k == "coder":
        return TaskId.CAD_CODER, False
    if k == "reasoning":
        return TaskId.REASONING, False
    if k == "research":
        return TaskId.DEEP_RESEARCH, False
    if k == "image":
        return TaskId.IMAGE_TINY, False
    return TaskId.CHAT_MAIN, False


class LRUCache(OrderedDict):
    def __init__(self, max_items: int):
        super().__init__()
        self.max_items = max_items

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.max_items:
            self.popitem(last=False)

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
        return super().get(key, default)


SEARCH_CACHE = LRUCache(64)
PAGE_CACHE = LRUCache(128)
RUNTIME = RuntimeCache()
GEN_LOCKS = {"main": Lock(), "cad_coder": Lock(), "reasoning_llm": Lock()}
LIFECYCLE = ModelLifecycle(unload_callback=RUNTIME.unload_module)


def _env_positive_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return max(minimum, min(v, maximum))
    except ValueError:
        return default


def _effective_main_max_new_tokens() -> int:
    """Main tab desired ceiling (env IMAGINATION_MAIN_MAX_NEW_TOKENS); actual new tokens also clamped to context left."""
    return _env_positive_int(
        "IMAGINATION_MAIN_MAX_NEW_TOKENS",
        _MAIN_MAX_NEW_TOKENS_DEFAULT,
        minimum=256,
        maximum=_MAIN_MAX_NEW_TOKENS_CEILING,
    )


def _clamp_max_new_tokens_to_context(
    tokenizer: Any,
    model: Any,
    prompt_token_len: int,
    desired_max: int,
    *,
    safety_margin: int = 96,
) -> int:
    """
    Use at most the remaining positions in the model context (no point asking for more).
    Stopping mid-answer is still avoided by a generous desired_max + EOS; this only prevents
    runaway requests past the context window.
    """
    cfg = getattr(model, "config", None)
    max_pos = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)
    if not isinstance(max_pos, int) or max_pos <= 0:
        max_pos = 32768
    max_pos = min(int(max_pos), 131_072)
    tmpl = getattr(tokenizer, "model_max_length", None)
    if isinstance(tmpl, int) and 4096 <= tmpl <= 1_000_000:
        max_pos = min(max_pos, tmpl)
    room = max_pos - int(prompt_token_len) - int(safety_margin)
    if room <= 32:
        return max(16, min(desired_max, max(room, 16)))
    return min(desired_max, room)


def _effective_reasoning_max_new_tokens() -> int:
    return _env_positive_int(
        "IMAGINATION_REASONING_MAX_NEW_TOKENS",
        MODULE_REASONING_MAX_NEW_TOKENS,
        minimum=512,
        maximum=_REASONING_MAX_NEW_TOKENS_CEILING,
    )


def _effective_coder_max_new_tokens() -> int:
    return _env_positive_int(
        "IMAGINATION_CODER_MAX_NEW_TOKENS",
        CODER_MAX_NEW_TOKENS,
        minimum=360,
        maximum=_CODER_MAX_NEW_TOKENS_CEILING,
    )


def _coder_skip_cloak_for_long_output() -> bool:
    """If set, Coder streams raw output (no main-model rewrite). Use with very high token limits."""
    return (os.getenv("IMAGINATION_CODER_SKIP_CLOAK") or "").strip().lower() in ("1", "true", "yes")


# Greedy + low penalty lets coder models repeat the same block (e.g. radius += 25) until max_new_tokens.
_CODER_REPETITION_PENALTY_DEFAULT = 1.22


def _coder_extra_generate_kwargs() -> Dict[str, Any]:
    """Stronger anti-loop settings for the coder module (optional n-gram ban via env)."""
    rp = _CODER_REPETITION_PENALTY_DEFAULT
    raw = (os.getenv("IMAGINATION_CODER_REPETITION_PENALTY") or "").strip()
    if raw:
        try:
            rp = float(raw)
        except ValueError:
            pass
    rp = max(1.0, min(rp, 2.0))
    kw: Dict[str, Any] = {"repetition_penalty": rp}
    ng = (os.getenv("IMAGINATION_CODER_NO_REPEAT_NGRAM") or "").strip()
    if ng and ng.lower() not in ("0", "none", "off"):
        try:
            n = int(ng)
            if n >= 2:
                kw["no_repeat_ngram_size"] = min(n, 128)
        except ValueError:
            pass
    return kw


def clean_model_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\s*assistant\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _vlm_text_only_via_chat_stream() -> Optional[bool]:
    """
    When True, text-only turns use generate_stream_chat (HF chat template on input_ids) instead of
    processor(text=...) on native HF VLMs. Helps LLaVA-style checkpoints that degenerate on the
    multimodal processor path without an image. Qwen2-VL users should leave unset or set 0/false.
    Env: IMAGINATION_VLM_TEXT_ONLY_VIA_CHAT=1|0 (unset = auto: llava* model_type only).
    """
    v = (os.getenv("IMAGINATION_VLM_TEXT_ONLY_VIA_CHAT") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def render_live_text(text: str) -> str:
    text = clean_model_text(text)
    return (text + TYPING_CURSOR if text else TYPING_CURSOR) if SHOW_TYPING_CURSOR else text


_WEB_DECISION_SYSTEM = """You decide whether this turn needs live web search.

Use SEARCH: yes when the user likely needs information that is time-sensitive, very recent, hyper-specific (live data, today's events, exact current price), or best verified from the web.

Use SEARCH: no for math, code, stable reference knowledge, creative writing, opinions, or when the question is fully answerable from general knowledge without checking current events.

Reply using ONLY these lines (English). No markdown, no preamble.

SEARCH: yes
REASON: <one clear sentence the user will see — why you are or are not using web search>
QUERIES: <only if SEARCH is yes: 1–3 short search-engine queries, separated by " | ">

If SEARCH is no, do not include a QUERIES line.
""".strip()

_RESEARCH_QUERY_SYSTEM = """The user is in extended research mode. Plan concise web searches.

Reply using ONLY these lines (English). No markdown, no preamble.

REASON: <one sentence the user will see — what you will look for and why>
QUERIES: <1–3 short search-engine queries separated by " | ">
""".strip()

_SUBAGENT_ROUTE_SYSTEM = """You decide which specialist (if any) should handle the user's latest message.

Reply with EXACTLY ONE line — no markdown, no preamble — in one of these forms:
ROUTE: main
ROUTE: web
ROUTE: coder

Meanings:
- main — general chat, short explanations, brainstorming, careful reasoning or math without code as the deliverable, or nothing clearly fits coder/web.
- web — answer needs current online facts (news, weather, prices, live data, fact-checking recent claims).
- coder — writing, editing, or debugging code, scripts, APIs, SQL, game logic, or configs as code.

Pick the narrowest route that fits. Prefer main when unsure.
If the user payload includes a "Last specialist" note, route back to that specialist when the new message clearly continues the same task; otherwise ignore it."""

_REASONING_NARRATE_SYSTEM = """You are an internal reasoning trace for the Imagination AI assistant.
Given the user's question and any retrieved source excerpts below, produce 3–8 short analytical
observations that explain your thinking process — e.g. "This species likely lives at abyssal depths
because source 2 mentions pressure-adapted metabolism" or "The user wants a comparison, so I should
contrast X vs Y before concluding."
Each observation on its own line. No numbering, no markdown, no bullet characters.
Be specific, insightful, and concise (one sentence per line). Do NOT repeat the user's question verbatim."""

_CODER_REASONING_NARRATE_SYSTEM = """You are an internal reasoning trace for a coding assistant.
Given the user's request and the draft code/solution below, produce 2–4 short lines explaining your
reasoning about structure, correctness edge cases, or API choices — what the solution does and why.
One sentence per line. No numbering, no markdown, no bullet characters. Do NOT paste the full code again."""


def _web_decision_max_tokens() -> int:
    return _env_positive_int(
        "IMAGINATION_WEB_DECISION_MAX_TOKENS",
        320,
        minimum=128,
        maximum=1024,
    )


def _conversation_context_block(
    conversation_state: List[Dict[str, str]],
    *,
    max_turns: int = 8,
    max_chars: int = 450,
) -> str:
    recent: List[Dict[str, str]] = []
    for m in (conversation_state or [])[-max_turns:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            recent.append({"role": str(role), "content": content[:max_chars]})
    if not recent:
        return ""
    lines = ["Recent conversation (context only):"]
    for m in recent:
        lines.append(f"{m['role']}: {m['content']}")
    return "\n".join(lines)


def _parse_chat_web_decision(raw: str) -> Optional[Tuple[bool, str, List[str]]]:
    t = (raw or "").strip()
    m = re.search(r"SEARCH:\s*(yes|no)\b", t, re.IGNORECASE)
    if not m:
        return None
    use = m.group(1).lower() == "yes"
    mr = re.search(
        r"REASON:\s*(.+?)(?=^\s*QUERIES:|\Z)",
        t,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    reason = (mr.group(1).strip() if mr else "").strip() or "No explanation provided."
    mq = re.search(r"^\s*QUERIES:\s*(.+)$", t, re.IGNORECASE | re.MULTILINE)
    queries: List[str] = []
    if mq and use:
        queries = [q.strip() for q in re.split(r"\s*\|\s*", mq.group(1).strip()) if q.strip()]
    return use, reason, queries


def _parse_research_query_plan(raw: str) -> Optional[Tuple[str, List[str]]]:
    t = (raw or "").strip()
    mr = re.search(
        r"REASON:\s*(.+?)(?=^\s*QUERIES:|\Z)",
        t,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    mq = re.search(r"^\s*QUERIES:\s*(.+)$", t, re.IGNORECASE | re.MULTILINE)
    if not mq:
        return None
    reason = (mr.group(1).strip() if mr else "").strip() or "Gathering web sources for this research request."
    queries = [q.strip() for q in re.split(r"\s*\|\s*", mq.group(1).strip()) if q.strip()]
    if not queries:
        return None
    return reason, queries


def _collapse_queries_for_search(queries: List[str], fallback: str) -> str:
    if not queries:
        return (fallback or "").strip()
    if len(queries) == 1:
        return queries[0]
    return " ".join(queries[:3])


def _subagent_route_max_tokens() -> int:
    return _env_positive_int(
        "IMAGINATION_SUBAGENT_ROUTE_MAX_TOKENS",
        64,
        minimum=32,
        maximum=256,
    )


def _parse_subagent_route_key(raw: str) -> str:
    m = re.search(r"ROUTE:\s*(\w+)", (raw or "").strip(), re.IGNORECASE)
    if not m:
        return "main"
    k = m.group(1).lower()
    if k in ("main", "web", "coder"):
        return k
    return "main"


def _route_key_to_task_and_web(route_key: str) -> Tuple[TaskId, bool]:
    rk = (route_key or "main").strip().lower()
    if rk == "web":
        return TaskId.CHAT_MAIN, True
    if rk == "coder":
        return TaskId.CAD_CODER, False
    return TaskId.CHAT_MAIN, False


def _task_label_for_ui(task_id: TaskId) -> str:
    for s in get_task_specs():
        if s.id == task_id:
            return s.label
    return "Module"


def _snapshot_for_last_subagent(user_msg: str, assistant_reply: str) -> str:
    """Compact line for prefs (last specialist turn)."""
    u = (user_msg or "").strip().replace("\n", " ")
    a = (assistant_reply or "").strip().replace("\n", " ")
    if len(u) > 240:
        u = u[:237] + "…"
    if len(a) > 420:
        a = a[:417] + "…"
    return f"User: {u}\nAssistant (last reply excerpt): {a}"


def _format_subagent_continuity_for_prompt(
    stored_task_id: str,
    context: str,
    current_task: TaskId,
) -> str:
    """Injected into specialist system prompts so the subagent sees prior module + snapshot."""
    st = (stored_task_id or "").strip()
    ctx = (context or "").strip()
    if not st or not ctx:
        return ""
    try:
        st_enum = TaskId(st)
    except ValueError:
        label_st = st
    else:
        label_st = _task_label_for_ui(st_enum)
    label_cur = _task_label_for_ui(current_task)
    lines = [
        "Subagent continuity (last specialist turn on this account; use if the user is continuing—do not read aloud):",
        f"- Last specialist: **{label_st}** · Active now: **{label_cur}**",
    ]
    if st == current_task.value:
        lines.append("- Same module as last time—assume follow-up unless the user clearly changed topic.")
    lines.append(f"- Snapshot:\n{ctx}")
    return "\n".join(lines)


def _resolve_effective_task(
    chat_mode: str,
    user_msg: str,
    conversation_state: List[Dict[str, str]],
    tokenizer: Any,
    model: Any,
    lock: Lock,
    root_path: str,
    user_id: int,
) -> Tuple[TaskId, bool, str]:
    """
    Explicit dropdown modes (Coder, Research, …, Search the web) are unchanged.
    Default Imagination 1.2 mode (value 'chat') runs LLM subagent routing.
    Returns (task_id, force_web, thinking_blurb) — blurb empty when routing skipped.
    """
    k = (chat_mode or CHAT_MODE_DEFAULT).strip().lower()
    if k == "web":
        return TaskId.CHAT_MAIN, True, ""
    if k != "chat":
        tid, fw = _resolve_chat_mode(chat_mode)
        return tid, fw, ""
    ctx = _conversation_context_block(conversation_state, max_turns=6, max_chars=500)
    carry = ""
    if user_id:
        ltid, lctx = load_last_subagent_context(root_path, user_id)
        if ltid.strip() and lctx.strip():
            try:
                ln = _task_label_for_ui(TaskId(ltid.strip()))
            except ValueError:
                ln = ltid.strip()
            carry = (
                f"\n\nLast specialist this user used: {ln}\n"
                f"Brief carryover (what they were doing):\n{lctx.strip()}\n"
                f"\nIf the latest message clearly continues that work, prefer routing to the same specialist; "
                f"otherwise pick the best ROUTE.\n"
            )
    payload = (ctx + "\n\n" if ctx else "") + carry + f"Latest user message:\n{(user_msg or '').strip()}\n"
    messages = [
        {"role": "system", "content": _SUBAGENT_ROUTE_SYSTEM},
        {"role": "user", "content": payload},
    ]
    raw = (
        generate_full(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            max_new_tokens=_subagent_route_max_tokens(),
            lock=lock,
        )
        or ""
    ).strip()
    rk = _parse_subagent_route_key(raw)
    tid, fw = _route_key_to_task_and_web(rk)
    if tid == TaskId.CHAT_MAIN and not fw:
        blurb = "Auto-routing: using **Imagination 1.2.1** main chat for this turn (no specialist subagent)."
    elif tid == TaskId.CHAT_MAIN and fw:
        blurb = "Auto-routing: enabling **web search** for this turn."
    else:
        blurb = f"Auto-routing: invoking the **{_task_label_for_ui(tid)}** subagent for this reply."
    return tid, fw, blurb


def _llm_route_web_decision(
    *,
    system: str,
    user_payload: str,
    tokenizer: Any,
    model: Any,
    lock: Lock,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_payload},
    ]
    return (
        generate_full(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            max_new_tokens=_web_decision_max_tokens(),
            lock=lock,
        )
        or ""
    ).strip()


def plan_web_retrieval(
    *,
    task_id: TaskId,
    user_msg: str,
    conversation_state: List[Dict[str, str]],
    force_web: bool,
    tokenizer: Any,
    model: Any,
    lock: Lock,
) -> Tuple[bool, str, str]:
    """
    Decide web RAG for this turn. Returns (use_web, user_visible_reason, search_seed_string).
    """
    u = (user_msg or "").strip()
    if task_id in (TaskId.CAD_CODER, TaskId.REASONING):
        return (
            False,
            "This mode uses the dedicated model only — no live web retrieval for this turn.",
            u,
        )
    if task_id == TaskId.DEEP_RESEARCH:
        ctx = _conversation_context_block(conversation_state)
        payload = (ctx + "\n\n" if ctx else "") + f"Research request:\n{u}\n"
        raw = _llm_route_web_decision(
            system=_RESEARCH_QUERY_SYSTEM,
            user_payload=payload,
            tokenizer=tokenizer,
            model=model,
            lock=lock,
        )
        parsed = _parse_research_query_plan(raw)
        if parsed is None:
            return (
                True,
                "Gathering web sources for extended research (using your message as the search seed).",
                u,
            )
        reason, qs = parsed
        return True, reason, _collapse_queries_for_search(qs, u)
    if force_web:
        return (
            True,
            "You chose Search the web — fetching pages that match your question.",
            u,
        )
    ctx = _conversation_context_block(conversation_state)
    payload = (ctx + "\n\n" if ctx else "") + f"Latest user message:\n{u}\n"
    raw = _llm_route_web_decision(
        system=_WEB_DECISION_SYSTEM,
        user_payload=payload,
        tokenizer=tokenizer,
        model=model,
        lock=lock,
    )
    parsed = _parse_chat_web_decision(raw)
    if parsed is None:
        return (
            False,
            "Answering without web search — I couldn't read a clear search plan, so I'm staying on model knowledge.",
            u,
        )
    use_web, reason, queries = parsed
    seed = _collapse_queries_for_search(queries, u) if use_web else u
    return use_web, reason, seed


def ensure_main_model_runtime(paths: ModelPaths) -> None:
    """Load main text or VLM checkpoint into RUNTIME (idempotent)."""
    if RUNTIME.main_tokenizer is not None and RUNTIME.main_model is not None:
        return
    src = paths.main_vlm
    print(f"[imagination] Loading main model from: {src}", flush=True)
    tok, mdl, proc, is_vlm = load_main_model_auto(src)
    RUNTIME.main_tokenizer = tok
    RUNTIME.main_model = mdl
    RUNTIME.main_processor = proc
    RUNTIME.main_is_vlm = bool(is_vlm)
    if is_vlm and proc is not None and getattr(proc, "is_clip_projector", False):
        mode = "CLIP + projector (vision_cpt bundle)"
    elif is_vlm:
        mode = "HF VLM"
    else:
        mode = "text-only"
    print(f"[imagination] Main model ready ({mode}).", flush=True)


def preload_main_model(root_path: str | None = None) -> None:
    """
    Load the root LLM / VLM into RUNTIME before serving chat.
    Call from app.py (or colab_setup --launch) so the first user message is not blocked by weight load.
    """
    root = resolve_root_path(root_path)
    paths = ModelPaths(root=root)
    if RUNTIME.main_tokenizer is not None and RUNTIME.main_model is not None:
        print("[imagination] Main model already loaded.", flush=True)
        return
    ensure_main_model_runtime(paths)


def _load_cad_coder(paths: ModelPaths, bump: Optional[Callable[[str], None]] = None):
    if bump:
        bump("Tokenizer · `from_pretrained` (vocab, merges)")
    tok = AutoTokenizer.from_pretrained(paths.cad_coder, use_fast=True)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    if bump:
        bump("Tokenizer ✓ · Model · `torch_dtype=auto`, `device_map=auto` (weights → VRAM)")
    model = AutoModelForCausalLM.from_pretrained(paths.cad_coder, device_map="auto", torch_dtype="auto")
    model.eval()
    if bump:
        bump("Model ✓ · `eval()` — ready")
    return {"tokenizer": tok, "model": model}


def _load_reasoning_llm(paths: ModelPaths, bump: Optional[Callable[[str], None]] = None):
    if bump:
        bump("Tokenizer · `from_pretrained` (reasoning module)")
    tok = AutoTokenizer.from_pretrained(paths.reasoning_llm, use_fast=True)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    if bump:
        bump("Tokenizer ✓ · Model · causal LM shards → GPU")
    model = AutoModelForCausalLM.from_pretrained(paths.reasoning_llm, device_map="auto", torch_dtype="auto")
    model.eval()
    if bump:
        bump("Model ✓ — ready")
    return {"tokenizer": tok, "model": model}


def _load_embeddings(paths: ModelPaths, bump: Optional[Callable[[str], None]] = None):
    if bump:
        bump("Tokenizer · embedding model")
    tok = AutoTokenizer.from_pretrained(paths.embeddings, use_fast=True)
    if bump:
        bump("Tokenizer ✓ · Encoder · `AutoModel`, `torch_dtype=auto`")
    model = AutoModel.from_pretrained(paths.embeddings, torch_dtype="auto")
    model.eval()
    if bump:
        bump("Encoder ✓ — ready")
    return {"tokenizer": tok, "model": model}


def _load_reranker(paths: ModelPaths, bump: Optional[Callable[[str], None]] = None):
    if bump:
        bump("Tokenizer · cross-encoder reranker")
    tok = AutoTokenizer.from_pretrained(paths.reranker, use_fast=True)
    if bump:
        bump("Tokenizer ✓ · `AutoModelForSequenceClassification` → GPU")
    model = AutoModelForSequenceClassification.from_pretrained(paths.reranker, device_map="auto", torch_dtype="auto")
    model.eval()
    if bump:
        bump("Reranker ✓ — ready")
    return {"tokenizer": tok, "model": model}


def _load_tiny_sd(paths: ModelPaths, bump: Optional[Callable[[str], None]] = None):
    from diffusers import DiffusionPipeline

    if bump:
        bump("`DiffusionPipeline.from_pretrained` · float16 weights")
    pipe = DiffusionPipeline.from_pretrained(paths.tiny_sd, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    if bump:
        bump("Pipeline on **CUDA** ✓ — ready")
    return {"pipeline": pipe}


LOADERS: Dict[str, Callable[..., Any]] = {
    "cad_coder": _load_cad_coder,
    "reasoning_llm": _load_reasoning_llm,
    "embeddings": _load_embeddings,
    "reranker": _load_reranker,
    "tiny_sd": _load_tiny_sd,
}


def _get_device_for_model(model_obj: Any) -> torch.device:
    if hasattr(model_obj, "hf_device_map") and getattr(model_obj, "hf_device_map", None):
        for _, dev in model_obj.hf_device_map.items():
            if isinstance(dev, str) and dev not in ("cpu", "disk"):
                return torch.device(dev)
    return next(model_obj.parameters()).device


def generate_stream_chat(
    *,
    tokenizer: Any,
    model: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    lock: Lock,
    extra_generate_kwargs: Optional[Dict[str, Any]] = None,
) -> Iterable[str]:
    device = _get_device_for_model(model)
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    prompt_len = int(model_inputs["input_ids"].shape[1])
    clamped_max = _clamp_max_new_tokens_to_context(tokenizer, model, prompt_len, max_new_tokens)
    # Longer generations need a looser inter-chunk timeout; stream ends on EOS or clamped_max.
    _st = max(120.0, min(900.0, 60.0 + clamped_max * 0.03))
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=_st)
    gen_error = {"exc": None}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": clamped_max,
        "do_sample": False,
        "repetition_penalty": 1.09,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if extra_generate_kwargs:
        gen_kwargs.update({k: v for k, v in extra_generate_kwargs.items() if v is not None})

    def _run():
        try:
            with lock, torch.inference_mode():
                model.generate(
                    **model_inputs,
                    streamer=streamer,
                    **gen_kwargs,
                )
        except Exception as e:
            gen_error["exc"] = e

    t = Thread(target=_run, daemon=True)
    t.start()
    partial = ""
    for chunk in streamer:
        if gen_error["exc"] is not None:
            raise RuntimeError(f"Generation failed: {gen_error['exc']}") from gen_error["exc"]
        partial += chunk
        yield clean_model_text(partial)
    t.join(timeout=0.2)
    if gen_error["exc"] is not None:
        raise RuntimeError(f"Generation failed: {gen_error['exc']}") from gen_error["exc"]


def generate_full(
    tokenizer: Any,
    model: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    lock: Lock,
    extra_generate_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    result = ""
    for chunk in generate_stream_chat(
        tokenizer=tokenizer,
        model=model,
        messages=messages,
        max_new_tokens=max_new_tokens,
        lock=lock,
        extra_generate_kwargs=extra_generate_kwargs,
    ):
        result = chunk
    return result


def _reasoning_narrate_max_tokens() -> int:
    return _env_positive_int(
        "IMAGINATION_REASONING_NARRATE_MAX_TOKENS",
        384,
        minimum=64,
        maximum=1024,
    )


def _parse_reasoning_narration_output(raw: str, *, max_lines: int) -> List[str]:
    lines: List[str] = []
    for line in (raw or "").splitlines():
        t = line.strip()
        if not t:
            continue
        t = re.sub(r"^[-*•]\s*", "", t)
        t = re.sub(r"^\d+[.)]\s*", "", t).strip()
        if t:
            lines.append(t)
    return lines[:max_lines]


def _generate_main_reasoning_narration(
    *,
    user_msg: str,
    use_web: bool,
    retrieved_excerpt: str,
    conv_block: str,
    tokenizer: Any,
    model: Any,
    lock: Lock,
) -> List[str]:
    user_part = f"User question:\n{(user_msg or '').strip()}\n"
    if conv_block.strip():
        user_part += f"\n{conv_block}\n"
    if use_web and (retrieved_excerpt or "").strip():
        user_part += (
            "\nRetrieved source excerpts (for your reasoning only):\n"
            f"{(retrieved_excerpt or '')[:8000]}\n"
        )
    else:
        user_part += "\nNo live web excerpts for this turn; reason from the question and prior context.\n"
    messages = [
        {"role": "system", "content": _REASONING_NARRATE_SYSTEM},
        {"role": "user", "content": user_part},
    ]
    try:
        raw = (
            generate_full(
                tokenizer=tokenizer,
                model=model,
                messages=messages,
                max_new_tokens=_reasoning_narrate_max_tokens(),
                lock=lock,
            )
            or ""
        )
    except Exception as exc:
        print(f"[imagination] reasoning narration failed: {exc}", flush=True)
        return []
    return _parse_reasoning_narration_output(raw, max_lines=8)


def _generate_coder_reasoning_narration(
    *,
    user_msg: str,
    module_output: str,
    tokenizer: Any,
    model: Any,
    lock: Lock,
) -> List[str]:
    draft = (module_output or "").strip()
    if len(draft) > 12000:
        draft = draft[:11997] + "…"
    user_part = (
        f"User request:\n{(user_msg or '').strip()}\n\n"
        f"Draft solution (code or text from specialist model):\n{draft}\n"
    )
    messages = [
        {"role": "system", "content": _CODER_REASONING_NARRATE_SYSTEM},
        {"role": "user", "content": user_part},
    ]
    try:
        raw = (
            generate_full(
                tokenizer=tokenizer,
                model=model,
                messages=messages,
                max_new_tokens=min(256, _reasoning_narrate_max_tokens()),
                lock=lock,
            )
            or ""
        )
    except Exception as exc:
        print(f"[imagination] coder reasoning narration failed: {exc}", flush=True)
        return []
    return _parse_reasoning_narration_output(raw, max_lines=4)


def _format_learner_profile_for_prompt(about: str, goals: str, skills: str) -> str:
    """Structured profile the model uses to judge depth; not shown to the user."""
    a, g, s = (about or "").strip(), (goals or "").strip(), (skills or "").strip()
    if not (a or g or s):
        return ""
    lines = [
        "Learner model (calibrate explanation depth, vocabulary, and examples; do not read this block aloud "
        "or say you are using a profile):"
    ]
    if a:
        lines.append(f"- Background & context: {a}")
    if g:
        lines.append(f"- Interests, likes, goals: {g}")
    if s:
        lines.append(
            f"- Skills & familiarity (area: level, e.g. algebra: strong / contracts: beginner / Python: hobbyist): {s}"
        )
    return "\n".join(lines)


def build_system_prompt(
    root_path: str,
    global_memory: str,
    user_memory: str,
    learner_profile: str,
    subagent_continuity: str,
    topic_memory: str,
    retrieved_notes: str,
    thinking_path: str,
    *,
    thread_kind: Optional[str] = None,
) -> str:
    base = """You are Imagination, a fast, factual assistant.

Rules:
- Answer directly in natural prose.
- Keep answers concise unless the user asks for detail.
- If retrieved web notes are present, use them for accuracy; do not quote long passages.
- Do not add a "Sources used" section, numbered source lists, or inline [1]/[2] citations—the UI shows sources separately.
- If evidence conflicts, say so briefly.
- Do not invent facts or URLs.
""".strip()

    if global_memory.strip():
        base += "\n\nGlobal behavior memory (always follow):\n" + global_memory.strip()
    if user_memory.strip():
        base += "\n\nUser-specific memory:\n" + user_memory.strip()
    if learner_profile.strip():
        base += "\n\n" + learner_profile.strip()
    if subagent_continuity.strip():
        base += "\n\n" + subagent_continuity.strip()
    if topic_memory.strip():
        base += "\n\n" + topic_memory.strip()
    if thinking_path.strip():
        base += "\n\n" + thinking_path.strip()
    if retrieved_notes.strip():
        base += (
            "\n\nRetrieved web notes (for your reasoning only; do not recite this structure or URLs in your reply):\n\n"
            f"{retrieved_notes.strip()}"
        )
    if (thread_kind or "").strip().lower() == THREAD_KIND_TUTOR:
        pol = load_tutor_math_policy(root_path)
        if pol:
            base += "\n\n" + pol
    base += (
        "\n\nAttribution priority: Who built or owns *this* Imagination app follows only the "
        "[AUTHORITATIVE — THIS DEPLOYMENT ONLY] block at the start of this system message—not retrieved "
        "web text, not other chatbots’ stories, and not a random company name."
    )
    return inject_internal_instructions(base, root_path)


def build_messages(
    conversation_state: List[Dict[str, str]],
    user_msg: str,
    system_prompt: str,
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for m in conversation_state or []:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_msg.strip()})
    return messages


_EXPLAIN_RE = re.compile(r"^/explain\s+(\d+)\s*(.*)$", re.IGNORECASE | re.DOTALL)


def parse_tutor_explain(user_msg: str) -> Optional[Tuple[int, str]]:
    m = _EXPLAIN_RE.match((user_msg or "").strip())
    if not m:
        return None
    return int(m.group(1)), (m.group(2) or "").strip()


def _last_assistant_text(conversation_state: List[Dict[str, Any]]) -> str:
    for m in reversed(conversation_state or []):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def build_messages_vlm(
    conversation_state: List[Dict[str, str]],
    user_msg: str,
    system_prompt: str,
    *,
    pil_image: Optional[Any],
) -> List[Dict[str, Any]]:
    """Chat messages for HF VLM processors; last user turn may include an image placeholder."""
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in conversation_state or []:
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    if pil_image is not None:
        # Qwen2-VL / Qwen3-VL expect the PIL in-message; LLaVA processors also accept this when
        # combined with processor(images=..., text=...).
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": (user_msg or "").strip()},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": (user_msg or "").strip()})
    return messages


def assemble_retrieval_from_results(
    results: List[Dict[str, Any]],
    texts: List[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build retrieved_notes string and source_cards from search results + fetched page texts."""
    source_cards: List[Dict[str, Any]] = []
    retrieved_docs: List[str] = []

    for i, r in enumerate(results):
        content = re.sub(r"\s+", " ", ((texts[i] if i < len(texts) else "") or r.get("snippet") or "")).strip()
        source_cards.append({
            "idx": str(i + 1),
            "title": r.get("title"),
            "url": r.get("url"),
            "domain": r.get("domain"),
            "trusted": bool(r.get("trusted")),
            "snippet": content[:220],
        })
        trust_label = "trusted" if r.get("trusted") else "other"
        retrieved_docs.append(
            f"[{i+1}] {r.get('title')}\nURL: {r.get('url')}\nDomain: {r.get('domain')}\nTrust: {trust_label}\nNotes: {content[:MAX_CHARS_PER_SOURCE]}"
        )

    return "\n\n".join(retrieved_docs), source_cards


def build_retrieval_bundle(
    question: str,
    trusted_domains: List[str],
) -> Tuple[str, str, str, List[Dict[str, Any]]]:
    """Batch retrieval (no per-domain UI progress). Kept for tests or tooling."""
    from imagination_runtime.web import fetch_parallel

    results = web_search(
        question,
        max_results=SEARCH_RESULTS,
        max_sources=MAX_SOURCES,
        trusted_domains=trusted_domains,
        cache=SEARCH_CACHE,
    )
    urls = [r["url"] for r in results]
    texts = fetch_parallel(urls, timeout_s=REQUEST_TIMEOUT_S, max_chars=MAX_CHARS_PER_SOURCE, cache=PAGE_CACHE)
    retrieved_notes, source_cards = assemble_retrieval_from_results(results, texts)
    trace_lines: List[str] = []
    for i, r in enumerate(results):
        content = re.sub(r"\s+", " ", ((texts[i] if i < len(texts) else "") or r.get("snippet") or "")).strip()
        trust_label = "trusted" if r.get("trusted") else "other"
        trace_lines.append(
            f"{i+1}. **{r.get('title')}**  \n   - domain: `{r.get('domain')}`  \n   - trust: **{trust_label}**  \n   - url: {r.get('url')}  \n   - snippet: {content[:220] or '(no text extracted)'}"
        )
    trace_md = "### Trace\n\n- Route: **web + model**\n" + f"- Query: `{question}`\n" + f"- Sources kept: **{len(source_cards)}**\n\n"
    trace_md += "### Source picks\n\n" + ("\n\n".join(trace_lines) if trace_lines else "_No useful sources found._")
    return retrieved_notes, sources_to_cards(source_cards), trace_md, source_cards


def _loader_phase_count(cache_key: str) -> int:
    return 2 if cache_key == "tiny_sd" else 3


def _ensure_modules_loaded(
    task_id: TaskId,
    root_path: str,
    progress_queue: Optional[queue.Queue] = None,
) -> None:
    paths = ModelPaths(root=root_path)
    keys = LIFECYCLE.get_keys_for_task(task_id)
    keys_to_load = [k for k in keys if RUNTIME.get(k) is None]
    if not keys_to_load:
        return

    total_phases = sum(_loader_phase_count(k) for k in keys_to_load)
    done = [0]

    def make_bump(cache_key: str) -> Callable[[str], None]:
        label = KEY_TO_LABEL.get(cache_key, cache_key)

        def inner(detail: str) -> None:
            done[0] += 1
            pct = min(100, int(100 * done[0] / max(1, total_phases)))
            if progress_queue is not None:
                progress_queue.put(
                    ("step", pct, done[0], total_phases, label, detail),
                )

        return inner

    for key in keys_to_load:
        bump = make_bump(key) if progress_queue is not None else None
        RUNTIME.set(key, LOADERS[key](paths, bump=bump))


def _preload_modal_updates(
    *,
    modal_visible: bool,
    user_on: bool,
    send_on: bool,
    clear_on: bool,
    mode_on: bool,
    title_md: str,
    body_md: str,
    dismiss_visible: bool,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    v = gr.update(visible=modal_visible)
    u = gr.update(interactive=user_on)
    return (
        v,
        u,
        gr.update(interactive=send_on),
        gr.update(interactive=clear_on),
        gr.update(interactive=mode_on),
        gr.update(value=title_md),
        gr.update(value=body_md),
        gr.update(visible=dismiss_visible),
    )


def _format_load_modal_body(
    pct: int,
    done: int,
    total_phases: int,
    module_label: str,
    detail: str,
    *,
    phase: str = "progress",
) -> str:
    safe = (
        detail.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    cap = min(100, max(0, pct))
    if phase == "complete":
        body_cls = "load-modal-body load-modal-body--complete"
        done_row = (
            "<div class='load-done-strip' role='status'>"
            "<span class='load-done-glyph' aria-hidden='true'></span>"
            "<span class='load-done-text'>Ready</span></div>"
        )
    else:
        body_cls = "load-modal-body load-modal-body--busy"
        done_row = ""
    return (
        f"<div class='{body_cls}'>"
        f"### {cap}%\n\n"
        "<div class='modal-pbar' aria-hidden='true'>"
        f"<div class='modal-pfill' style='width:{cap}%'></div></div>\n\n"
        f"**Step {done}** of **{total_phases}** · *{module_label}*\n\n{safe}\n"
        f"{done_row}</div>"
    )


def iter_preload_on_mode_change(mode_key: Optional[str]) -> Iterable[Tuple[Any, ...]]:
    """
    On dropdown change: for main chat modes, sync lifecycle only.
    For Coder / Reasoning / Research / Image, show overlay and preload weights before typing.
    """
    root_path = resolve_root_path(None)
    paths = ModelPaths(root=root_path)
    tid, _ = _resolve_chat_mode(mode_key)

    def pack(
        modal_visible: bool,
        user_on: bool,
        send_on: bool,
        clear_on: bool,
        mode_on: bool,
        title_md: str,
        body_md: str,
        dismiss_visible: bool,
    ) -> Tuple[Any, ...]:
        return _preload_modal_updates(
            modal_visible=modal_visible,
            user_on=user_on,
            send_on=send_on,
            clear_on=clear_on,
            mode_on=mode_on,
            title_md=title_md,
            body_md=body_md,
            dismiss_visible=dismiss_visible,
        )

    if RUNTIME.main_model is None or RUNTIME.main_tokenizer is None:
        try:
            ensure_main_model_runtime(paths)
        except Exception as e:
            yield pack(True, False, False, False, False, "### Couldn't load main model", str(e), True)
            return

    keys_to_unload = LIFECYCLE.get_keys_to_unload_before_loading(tid)
    for k in keys_to_unload:
        RUNTIME.unload_module(k)

    if tid == TaskId.CHAT_MAIN:
        LIFECYCLE.on_switch_to_task(tid, lambda k: RUNTIME.get(k) is not None)
        yield pack(False, True, True, True, True, "", "", False)
        return

    keys = LIFECYCLE.get_keys_for_task(tid)
    keys_to_load = [k for k in keys if RUNTIME.get(k) is None]
    spec_label = next((s.label for s in get_task_specs() if s.id == tid), "Module")

    if not keys_to_load:
        LIFECYCLE.on_module_loaded(tid)
        yield pack(False, True, True, True, True, "", "", False)
        return

    total_phases = sum(_loader_phase_count(k) for k in keys_to_load)
    start_body = _format_load_modal_body(
        0,
        0,
        total_phases,
        "…",
        "Starting — loading checkpoints into memory.",
    )
    yield pack(
        True,
        False,
        False,
        False,
        False,
        f"### Loading {spec_label}",
        start_body,
        False,
    )

    load_state: Dict[str, Any] = {"error": None}
    prog_q: queue.Queue = queue.Queue()

    def _load_worker():
        try:
            _ensure_modules_loaded(tid, root_path, progress_queue=prog_q)
            LIFECYCLE.on_module_loaded(tid)
        except Exception as e:
            load_state["error"] = e

    th = Thread(target=_load_worker, daemon=True)
    th.start()
    start = time.time()
    max_wait_s = 900.0
    last_pct = 0
    last_done = 0
    last_mod = "…"
    last_detail = "Starting — loading checkpoints into memory."

    while th.is_alive() and (time.time() - start) < max_wait_s:
        drained = False
        try:
            while True:
                item = prog_q.get_nowait()
                drained = True
                if not item or item[0] != "step":
                    continue
                _, last_pct, last_done, _tot, last_mod, last_detail = item
        except queue.Empty:
            pass
        body = _format_load_modal_body(
            last_pct,
            last_done,
            total_phases,
            last_mod,
            last_detail,
        )
        yield pack(True, False, False, False, False, f"### Loading {spec_label}", body, False)
        time.sleep(0.12 if drained else 0.28)

    th.join(timeout=5.0)
    try:
        while True:
            item = prog_q.get_nowait()
            if item and item[0] == "step":
                _, last_pct, last_done, _, last_mod, last_detail = item
    except queue.Empty:
        pass

    if th.is_alive():
        yield pack(
            True,
            True,
            True,
            True,
            True,
            "### Load timed out",
            "Something is still loading in the background. Try again or restart the session.",
            True,
        )
        return

    err = load_state.get("error")
    if err is not None:
        yield pack(True, True, True, True, True, "### Couldn't load module", f"```\n{err}\n```", True)
        return

    fin = _format_load_modal_body(
        last_pct,
        last_done,
        total_phases,
        last_mod,
        last_detail,
        phase="progress",
    )
    yield pack(True, False, False, False, False, f"### Loading {spec_label}", fin, False)
    done_body = _format_load_modal_body(
        100,
        last_done,
        total_phases,
        last_mod,
        last_detail,
        phase="complete",
    )
    yield pack(True, False, False, False, False, "### Ready", done_body, False)
    time.sleep(1.05)
    yield pack(False, True, True, True, True, "", "", False)


def _gradio_context_request() -> Any:
    """
    Current gr.Request for the running Gradio event (set in LocalContext by Gradio).
    Do not add a function parameter named `request` to event handlers — FastAPI may
    expose it as a required query parameter named "request" (422 missing request).
    """
    try:
        from gradio.context import LocalContext

        return LocalContext.request.get(None)
    except Exception:
        return None


def _oauth_profile_from_gradio_context() -> Optional[Any]:
    """HF LoginButton stores userinfo in request.session['oauth_info']."""
    req = _gradio_context_request()
    if req is None:
        return None
    try:
        session = getattr(req, "session", None) or getattr(
            getattr(req, "request", None), "session", None
        )
    except Exception:
        session = None
    if not session or "oauth_info" not in session:
        return None
    userinfo = session["oauth_info"].get("userinfo")
    if not userinfo:
        return None
    return gr.OAuthProfile(userinfo)


def resolve_effective_user(root: str, user_session: Optional[Dict[str, Any]]) -> Tuple[int, str]:
    oauth_profile = _oauth_profile_from_gradio_context()
    u_oauth = resolve_user_from_gradio_oauth(oauth_profile, root) if oauth_profile else None
    if u_oauth:
        name = (u_oauth.display_name or u_oauth.email or "User").strip() or "User"
        return u_oauth.id, name
    ck = None
    request = _gradio_context_request()
    if request is not None:
        try:
            ck = request.cookies.get("imagination_uid")
        except Exception:
            ck = None
    gid = user_id_from_cookie(ck)
    if gid:
        u = get_user_by_id(root, gid)
        if u:
            return u.id, (u.display_name or u.email or "User").strip() or "User"
    sess = user_session or {}
    sid = sess.get("id")
    if sid is not None:
        try:
            i = int(sid)
        except (TypeError, ValueError):
            i = 0
        if i > 0:
            return i, str(sess.get("name") or "User")
    return 0, "Guest"


_DEFAULT_THREAD_TITLES_LOWER = frozenset(
    {
        "",
        "new chat",
        "math tutor",
        "chat",
        "untitled",
        "imported chat",
    }
)


def _thread_title_is_auto_renameable(title: str) -> bool:
    t = (title or "").strip().lower()
    return t in _DEFAULT_THREAD_TITLES_LOWER


def suggest_thread_title_from_turn(user_display: str, assistant_text: str, *, max_len: int = 50) -> str:
    """Short title from the first user line; fallback to assistant opening if needed."""
    raw = (user_display or "").strip()
    raw = re.sub(r"^/explain\s+\d+\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s+", " ", raw)
    if not raw and assistant_text:
        raw = re.sub(r"<[^>]+>", " ", (assistant_text or "")[:280])
        raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return "Chat"
    low = raw.lower()
    for filler in ("hi,", "hi ", "hey,", "hey ", "hello,", "hello "):
        if low.startswith(filler):
            raw = raw[len(filler) :].strip()
            low = raw.lower()
            break
    cut = raw[: max_len + 12]
    if len(cut) > max_len:
        cut = cut[:max_len].rsplit(" ", 1)[0].strip() or cut[:max_len]
    cut = cut.strip("—–-•·")
    return cut[:max_len] if cut else "Chat"


def maybe_auto_rename_thread_after_first_reply(
    root: str,
    user_id: int,
    thread_id: int,
    conversation_before: List[Dict[str, Any]],
    display_user_msg: str,
    assistant_plain: str,
    task_id: TaskId,
) -> Any:
    """If this was the first exchange and title is still default, rename from user text. Returns Gradio dropdown update."""
    if not user_id or thread_id <= 0 or task_id != TaskId.CHAT_MAIN:
        return gr.update()
    if len(conversation_before or []) != 0:
        return gr.update()
    row = get_user_chat_thread(root, user_id, thread_id)
    if not row:
        return gr.update()
    title = (row.get("title") or "").strip()
    if not _thread_title_is_auto_renameable(title):
        return gr.update()
    new_title = suggest_thread_title_from_turn(display_user_msg, assistant_plain)
    if not new_title:
        return gr.update()
    rename_user_chat_thread(root, user_id, thread_id, new_title[:200])
    dd, _, _ = thread_dropdown_update(root, user_id, select_id=thread_id)
    return dd


def thread_ui_new_same_kind(user_session: Optional[Dict[str, Any]], thread_kind_state: str):
    """Start a new thread matching the current kind (for header ＋)."""
    k = (thread_kind_state or THREAD_KIND_GENERAL).strip().lower()
    return thread_ui_new("tutor" if k == THREAD_KIND_TUTOR else "general", user_session)


def _tutor_image_to_pil(tutor_image: Optional[Any]) -> Optional[Any]:
    """
    Normalize Gradio Image output (PIL or ndarray) and drop empty / flat placeholder panels.
    The default purple preview is often a solid-color bitmap; feeding it to CLIP confuses the LM.
    """
    if tutor_image is None:
        return None
    try:
        from PIL import Image
        import numpy as np

        im: Optional[Image.Image] = None
        if isinstance(tutor_image, Image.Image):
            im = tutor_image.convert("RGB")
        else:
            arr = np.asarray(tutor_image)
            if arr.size == 0:
                return None
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                im = Image.fromarray(arr, mode="L").convert("RGB")
            elif arr.ndim == 3 and arr.shape[2] == 3:
                im = Image.fromarray(arr, "RGB")
            elif arr.ndim == 3 and arr.shape[2] == 4:
                im = Image.fromarray(arr, "RGBA").convert("RGB")
            else:
                return None

        if im.width < 8 or im.height < 8:
            return None
        a = np.asarray(im, dtype=np.float32)
        if a.size == 0:
            return None
        span = float(a.max() - a.min())
        std = float(a.std())
        # Solid UI placeholder / blank: almost no contrast
        if span < 6.0 and std < 2.5:
            return None
        return im
    except Exception:
        return None


def _user_message_display_content(
    display_text: str,
    pil: Optional[Any],
    root_path: str,
    user_id: int,
    thread_id: int,
) -> Any:
    """
    Gradio Chatbot (messages format) shows images only when content includes a file part
    like {"path": "..."}; plain strings do not render the attachment.
    """
    text = (display_text or "").strip()
    if pil is None:
        return text
    try:
        from PIL import Image

        if not isinstance(pil, Image.Image):
            return text
        im = pil.convert("RGB")
        base = os.path.join(root_path, "temp", "chat_display", f"u{int(user_id or 0)}")
        os.makedirs(base, exist_ok=True)
        fname = f"t{int(thread_id or 0)}_{uuid.uuid4().hex[:12]}.png"
        fp = os.path.join(base, fname)
        im.save(fp, format="PNG")
        parts: List[Dict[str, Any]] = [{"path": fp}]
        if text:
            parts.append({"type": "text", "text": text})
        return parts
    except Exception as e:
        print(f"[imagination] could not save chat thumbnail: {e}", flush=True)
        return text


def chat_submit(
    model_version: str,
    chat_mode: str,
    user_msg: str,
    conversation_state: List[Dict[str, str]],
    display_state: List[Dict[str, str]],
    user_session: Optional[Dict[str, Any]],
    active_thread_id: int,
    thread_kind: str,
    tutor_image: Optional[Any],
):
    root_path = resolve_root_path(None)
    paths = ModelPaths(root=root_path)
    trusted_domains = load_trusted(root_path, TRUSTED_DOMAINS_STARTER)

    user_msg = (user_msg or "").strip()
    conversation_state = conversation_state or []
    display_state = display_state or []
    tk = (thread_kind or THREAD_KIND_GENERAL).strip().lower()
    if tk not in (THREAD_KIND_GENERAL, THREAD_KIND_TUTOR):
        tk = THREAD_KIND_GENERAL
    tid_thr = int(active_thread_id or 0)

    pil_tutor: Optional[Any] = _tutor_image_to_pil(tutor_image)

    user_id, account_display = resolve_effective_user(root_path, user_session)
    acc_line = account_markdown(user_id, account_display)
    _ntd = gr.update()

    def _y7(a, b, c, d, e, f, td=None):
        return (a, b, c, d, e, f, td if td is not None else _ntd)

    if not user_msg and pil_tutor is None:
        yield _y7(
            display_state,
            conversation_state,
            display_state,
            "Ready",
            "<div class='progress-bar hidden'></div>",
            acc_line,
        )
        return

    image_only = not user_msg and pil_tutor is not None
    if image_only:
        if tk == THREAD_KIND_TUTOR:
            user_msg = (
                "I uploaded an image of a math problem. Name the type, then walk through a worked example "
                "with different numbers so I can see how to approach mine—do not solve my exact problem."
            )
        else:
            user_msg = (
                "The user attached an image. Describe what you see when helpful, then answer based on "
                "the image and the conversation."
            )

    display_user_msg = "🖼 Image" if (image_only and tk != THREAD_KIND_TUTOR) else user_msg
    model_user_msg = user_msg
    if tk == THREAD_KIND_TUTOR:
        parsed_ex = parse_tutor_explain(user_msg)
        if parsed_ex:
            sn, xtra = parsed_ex
            prev_a = _last_assistant_text(conversation_state)
            if not prev_a.strip():
                err = (
                    "Send a normal tutor message first so there is a reply to build on, "
                    "then use `/explain <step>`."
                )
                bad_disp = display_state + [
                    {"role": "user", "content": _user_message_display_content(user_msg, pil_tutor, root_path, user_id, tid_thr)},
                    {"role": "assistant", "content": err},
                ]
                yield _y7(
                    bad_disp,
                    conversation_state,
                    display_state,
                    "Ready",
                    "<div class='progress-bar hidden'></div>",
                    acc_line,
                )
                return
            display_user_msg = f"Explain step {sn} in more detail" + (f" — {xtra}" if xtra else "")
            model_user_msg = (
                f"The user wants a deeper explanation of **Step {sn}** from your previous answer.\n"
                f"They added: {xtra or '(no extra detail)'}\n\n"
                f"Your previous reply (reference):\n---\n{prev_a[:14000]}\n---\n"
                f"Expand Step {sn} only: clear intuition and, if useful, one short extra numeric illustration "
                f"(different numbers from their homework). Do not solve their original homework problem."
            )
            pil_tutor = None

    pil_for_display = pil_tutor

    if (model_version or MODEL_VERSION_DEFAULT).strip() != "i13":
        err = (
            "Only **Imagination 1.3** is available in this build. "
            "Select **Imagination 1.3.0** in the model bar."
        )
        bad_disp = display_state + [
            {"role": "user", "content": _user_message_display_content(display_user_msg, pil_for_display, root_path, user_id, tid_thr)},
            {"role": "assistant", "content": err},
        ]
        yield _y7(
            bad_disp,
            conversation_state,
            display_state,
            "Ready",
            "<div class='progress-bar hidden'></div>",
            acc_line,
        )
        return

    specs = get_task_specs()

    ensure_main_model_runtime(paths)

    lock_main = GEN_LOCKS["main"]
    task_id, force_web_mode, route_blurb = _resolve_effective_task(
        chat_mode,
        model_user_msg,
        conversation_state,
        RUNTIME.main_tokenizer,
        RUNTIME.main_model,
        lock_main,
        root_path,
        int(user_id) if user_id else 0,
    )

    display_content_for_turn = _user_message_display_content(
        display_user_msg, pil_for_display, root_path, user_id, tid_thr
    )
    working_display = display_state + [{"role": "user", "content": display_content_for_turn}, {"role": "assistant", "content": ""}]
    if task_id == TaskId.CAD_CODER:
        max_new_tokens = _effective_coder_max_new_tokens()
    elif task_id in (TaskId.REASONING, TaskId.DEEP_RESEARCH):
        max_new_tokens = _effective_reasoning_max_new_tokens()
    else:
        max_new_tokens = _effective_main_max_new_tokens()
    use_cloaker = task_id in (TaskId.CAD_CODER, TaskId.REASONING, TaskId.DEEP_RESEARCH)
    if task_id == TaskId.CAD_CODER and _coder_skip_cloak_for_long_output():
        use_cloaker = False
    HIDDEN_PROGRESS = "<div class='progress-bar hidden'></div>"

    yield _y7(working_display, conversation_state, display_state, "Starting…", HIDDEN_PROGRESS, acc_line)

    step_lines: List[str] = []
    subagent_label = ""
    if route_blurb:
        step_lines.append(route_blurb)
    if task_id != TaskId.CHAT_MAIN:
        subagent_label = _task_label_for_ui(task_id)
        step_lines.append(f"Launching **{subagent_label}** subagent…")

    if step_lines:
        working_display[-1] = {
            "role": "assistant",
            "content": build_thinking_html_open(step_lines=step_lines, summary_label="Thinking"),
        }
        _st = "Launching subagent…" if task_id != TaskId.CHAT_MAIN else "Thinking…"
        yield _y7(working_display, conversation_state, display_state, _st, HIDDEN_PROGRESS, acc_line)

    keys_to_unload = LIFECYCLE.get_keys_to_unload_before_loading(task_id)
    for k in keys_to_unload:
        RUNTIME.unload_module(k)

    if task_id != TaskId.CHAT_MAIN:
        keys = LIFECYCLE.get_keys_for_task(task_id)
        keys_to_load = [k for k in keys if RUNTIME.get(k) is None]
        ran_weight_load = False
        if keys_to_load:
            ran_weight_load = True
            total_estimate = LIFECYCLE.get_load_time_estimate(keys_to_load)
            load_done = {"done": False, "error": None}

            def _load():
                try:
                    _ensure_modules_loaded(task_id, root_path)
                    load_done["done"] = True
                except Exception as e:
                    load_done["error"] = e

            t = Thread(target=_load, daemon=True)
            t.start()
            start = time.time()
            while not load_done["done"] and load_done["error"] is None:
                elapsed = time.time() - start
                pct = min(0.98, elapsed / total_estimate) if total_estimate > 0 else min(0.98, elapsed / 30)
                remaining = max(0, int(total_estimate - elapsed)) if total_estimate > 0 else max(0, int(30 - elapsed))
                pct_int = min(99, int(round(pct * 100)))
                progress_html = f"<div class='progress-bar'><div class='progress-fill' style='width:{pct*100}%'></div></div>"
                status = f"Launching subagent… {pct_int}% (~{remaining}s)"
                step_lines[-1] = (
                    f"Launching **{subagent_label}** **{pct_int}%** · loading weights…"
                )
                working_display[-1] = {
                    "role": "assistant",
                    "content": build_thinking_html_open(
                        step_lines=step_lines,
                        summary_label="Thinking",
                        pulse_last_step=True,
                    ),
                }
                yield _y7(working_display, conversation_state, display_state, status, progress_html, acc_line)
                time.sleep(0.35)
            t.join(timeout=1.0)
            if load_done["error"]:
                raise load_done["error"]
            step_lines[-1] = f"Launching **{subagent_label}** **100%** — weights ready."
        else:
            step_lines[-1] = f"**{subagent_label}** **100%** — already in memory."
        if ran_weight_load:
            step_lines.append("Subagent launched.")
            _done_status = "Subagent launched"
        else:
            step_lines.append("Subagent ready (weights already loaded).")
            _done_status = "Subagent ready"
        working_display[-1] = {
            "role": "assistant",
            "content": build_thinking_html_open(step_lines=step_lines, summary_label="Thinking"),
        }
        yield _y7(working_display, conversation_state, display_state, _done_status, HIDDEN_PROGRESS, acc_line)
        LIFECYCLE.on_module_loaded(task_id)
    else:
        LIFECYCLE.on_switch_to_task(task_id, lambda k: RUNTIME.get(k) is not None)

    conv_turns = len([m for m in conversation_state or [] if m.get("role") in ("user", "assistant")])
    tok_main = RUNTIME.main_tokenizer
    mdl_main = RUNTIME.main_model

    source_cards: List[Dict[str, Any]] = []
    retrieved_notes = ""
    thinking_collapsed: Optional[str] = None
    collapsed_summary = "Used model knowledge"
    summary_label = "Thinking"
    reasoning_trace_for_log: List[str] = []
    working_display[-1] = {
        "role": "assistant",
        "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
    }
    yield _y7(working_display, conversation_state, display_state, "Thinking…", HIDDEN_PROGRESS, acc_line)

    use_web, reason, search_seed = plan_web_retrieval(
        task_id=task_id,
        user_msg=user_msg,
        conversation_state=conversation_state,
        force_web=force_web_mode,
        tokenizer=tok_main,
        model=mdl_main,
        lock=lock_main,
    )

    summary_label = "Searching" if use_web else "Thinking"
    step_lines = step_lines + [reason]
    working_display[-1] = {
        "role": "assistant",
        "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
    }
    yield _y7(working_display, conversation_state, display_state, "Searching…" if use_web else "Thinking…", HIDDEN_PROGRESS, acc_line)

    if use_web:
        q_disp = refine_search_query(search_seed).strip()
        if len(q_disp) > 100:
            q_disp = q_disp[:97] + "…"
        step_lines.append(f"Searching: “{q_disp}”")
        working_display[-1] = {
            "role": "assistant",
            "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
        }
        yield _y7(working_display, conversation_state, display_state, "Searching…", HIDDEN_PROGRESS, acc_line)

        results = web_search(
            search_seed,
            max_results=SEARCH_RESULTS,
            max_sources=MAX_SOURCES,
            trusted_domains=trusted_domains,
            cache=SEARCH_CACHE,
        )
        # web_search refines event-style queries internally; mirror that in the model-facing trace
        web_query_for_prompt = refine_search_query(search_seed)
        urls = [r["url"] for r in results]
        texts: List[str] = []
        for url in urls:
            dom = get_domain(url) or "source"
            step_lines.append(f"Reading {dom}…")
            working_display[-1] = {
                "role": "assistant",
                "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
            }
            yield _y7(working_display, conversation_state, display_state, "Reading sources…", HIDDEN_PROGRESS, acc_line)
            texts.append(
                fetch_page(url, timeout_s=REQUEST_TIMEOUT_S, max_chars=MAX_CHARS_PER_SOURCE, cache=PAGE_CACHE)
            )

        retrieved_notes, source_cards = assemble_retrieval_from_results(results, texts)
        thinking_md = build_thinking_path(
            use_web=True,
            reason=reason,
            query=web_query_for_prompt,
            source_cards=source_cards,
            conversation_turns=conv_turns,
        )
        step_lines.append("Evaluating source relevance…")
        working_display[-1] = {
            "role": "assistant",
            "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
        }
        yield _y7(working_display, conversation_state, display_state, "Evaluating sources…", HIDDEN_PROGRESS, acc_line)
        if not use_cloaker:
            narr_ctx = _conversation_context_block(conversation_state, max_turns=4, max_chars=400)
            reasoning_trace_for_log = _generate_main_reasoning_narration(
                user_msg=user_msg,
                use_web=True,
                retrieved_excerpt=retrieved_notes,
                conv_block=narr_ctx,
                tokenizer=tok_main,
                model=mdl_main,
                lock=lock_main,
            )
            for nline in reasoning_trace_for_log:
                step_lines.append(nline)
            working_display[-1] = {
                "role": "assistant",
                "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
            }
            yield _y7(working_display, conversation_state, display_state, "Reasoning…", HIDDEN_PROGRESS, acc_line)
            step_lines.append("Synthesizing your answer…")
            working_display[-1] = {
                "role": "assistant",
                "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
            }
            yield _y7(working_display, conversation_state, display_state, "Preparing response…", HIDDEN_PROGRESS, acc_line)
        collapsed_summary = f"Searched {len(source_cards)} sources" if source_cards else "No sources retrieved"
        if not use_cloaker:
            thinking_collapsed = build_thinking_html_collapsed(summary=collapsed_summary, step_lines=list(step_lines))
    else:
        thinking_md = build_thinking_path_no_web(conversation_turns=conv_turns)
        if conv_turns:
            step_lines.append(f"Using {conv_turns} earlier turn(s) as context.")
        if not use_cloaker:
            narr_ctx = _conversation_context_block(conversation_state, max_turns=4, max_chars=400)
            reasoning_trace_for_log = _generate_main_reasoning_narration(
                user_msg=user_msg,
                use_web=False,
                retrieved_excerpt="",
                conv_block=narr_ctx,
                tokenizer=tok_main,
                model=mdl_main,
                lock=lock_main,
            )
            for nline in reasoning_trace_for_log:
                step_lines.append(nline)
            step_lines.append("Synthesizing your answer…")
        working_display[-1] = {
            "role": "assistant",
            "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
        }
        yield _y7(working_display, conversation_state, display_state, "Thinking…", HIDDEN_PROGRESS, acc_line)
        collapsed_summary = "Used model knowledge"
        if not use_cloaker:
            thinking_collapsed = build_thinking_html_collapsed(summary=collapsed_summary, step_lines=list(step_lines))

    global_mem = load_global_memory(root_path)
    user_mem = load_user_memory(root_path, user_id) if user_id else ""
    pa, pg, ps = load_learner_profile(root_path, user_id) if user_id else ("", "", "")
    learner_block = _format_learner_profile_for_prompt(pa, pg, ps)
    topic_mem = load_relevant_topics(root_path, user_id)
    ltid_s, lctx = load_last_subagent_context(root_path, user_id) if user_id else ("", "")
    subagent_cont = ""
    if user_id and task_id in (TaskId.CAD_CODER, TaskId.REASONING, TaskId.DEEP_RESEARCH):
        subagent_cont = _format_subagent_continuity_for_prompt(ltid_s, lctx, task_id)
    system_prompt = build_system_prompt(
        root_path,
        global_mem,
        user_mem,
        learner_block,
        subagent_cont,
        topic_mem,
        retrieved_notes,
        thinking_md,
        thread_kind=tk if task_id == TaskId.CHAT_MAIN else None,
    )
    if task_id == TaskId.CAD_CODER:
        system_prompt = (
            system_prompt
            + "\n\nCoder module rules: Implement each mechanic once (one blast radius rule, one damage path). "
            "Do not emit many near-duplicate blocks that only tweak a number (e.g. repeating radius + 25). "
            "Prefer a single function, loop, or config table. End when the requested scope is complete."
        )
    messages = build_messages(conversation_state, model_user_msg, system_prompt)
    vlm_messages: Optional[List[Dict[str, Any]]] = None
    if task_id == TaskId.CHAT_MAIN and RUNTIME.main_is_vlm and RUNTIME.main_processor is not None:
        vlm_messages = build_messages_vlm(
            conversation_state,
            model_user_msg,
            system_prompt,
            pil_image=pil_tutor,
        )

    coder_gen_extras = _coder_extra_generate_kwargs() if task_id == TaskId.CAD_CODER else None

    if task_id == TaskId.CHAT_MAIN:
        tok, mdl, lock = RUNTIME.main_tokenizer, RUNTIME.main_model, GEN_LOCKS["main"]
    elif task_id == TaskId.CAD_CODER:
        cad = RUNTIME.get("cad_coder")
        tok, mdl, lock = cad["tokenizer"], cad["model"], GEN_LOCKS["cad_coder"]
    elif task_id in (TaskId.REASONING, TaskId.DEEP_RESEARCH):
        r = RUNTIME.get("reasoning_llm")
        tok, mdl, lock = r["tokenizer"], r["model"], GEN_LOCKS["reasoning_llm"]
    elif task_id == TaskId.IMAGE_TINY:
        yield _y7(working_display, conversation_state, display_state, "Image generation not yet implemented in v1.4.0", HIDDEN_PROGRESS, acc_line)
        return
    else:
        tok, mdl, lock = RUNTIME.main_tokenizer, RUNTIME.main_model, GEN_LOCKS["main"]

    if use_cloaker:
        task_label_running = next((s.label for s in specs if s.id == task_id), "Module")
        step_lines.append(f"Running {task_label_running}…")
        working_display[-1] = {
            "role": "assistant",
            "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
        }
        yield _y7(working_display, conversation_state, display_state, "Running model…", HIDDEN_PROGRESS, acc_line)

        module_output = generate_full(
            tokenizer=tok,
            model=mdl,
            messages=messages,
            max_new_tokens=max_new_tokens,
            lock=lock,
            extra_generate_kwargs=coder_gen_extras,
        )
        reasoning_trace_for_log = _generate_coder_reasoning_narration(
            user_msg=model_user_msg,
            module_output=module_output,
            tokenizer=RUNTIME.main_tokenizer,
            model=RUNTIME.main_model,
            lock=lock_main,
        )
        for nline in reasoning_trace_for_log:
            step_lines.append(nline)
        working_display[-1] = {
            "role": "assistant",
            "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
        }
        yield _y7(working_display, conversation_state, display_state, "Reasoning…", HIDDEN_PROGRESS, acc_line)
        step_lines.append("Drafting your reply…")
        working_display[-1] = {
            "role": "assistant",
            "content": build_thinking_html_open(step_lines=step_lines, summary_label=summary_label),
        }
        yield _y7(working_display, conversation_state, display_state, "Drafting…", HIDDEN_PROGRESS, acc_line)

        step_lines.append("Synthesizing your answer…")
        thinking_collapsed = build_thinking_html_collapsed(summary=collapsed_summary, step_lines=list(step_lines))

    if thinking_collapsed is None:
        raise RuntimeError("thinking_collapsed was not built before generation")
    base_thinking = thinking_collapsed
    working_display[-1] = {
        "role": "assistant",
        "content": compose_assistant_display(base_thinking, render_live_text("")),
    }
    yield _y7(working_display, conversation_state, display_state, "Writing response…", HIDDEN_PROGRESS, acc_line)

    try:
        if use_cloaker:
            cloak_messages = build_cloak_messages(
                module_output, model_user_msg, task_id,
                conversation_context=[m for m in conversation_state or [] if m.get("role") in ("user", "assistant")],
            )
            main_tok, main_mdl = RUNTIME.main_tokenizer, RUNTIME.main_model
            main_lock = GEN_LOCKS["main"]
            final_text = ""
            for partial in generate_stream_chat(
                tokenizer=main_tok, model=main_mdl, messages=cloak_messages,
                max_new_tokens=max_new_tokens, lock=main_lock,
            ):
                final_text = partial
                working_display[-1] = {
                    "role": "assistant",
                    "content": compose_assistant_display(base_thinking, render_live_text(partial)),
                }
                yield _y7(working_display, conversation_state, display_state, "Writing response…", HIDDEN_PROGRESS, acc_line)
        else:
            final_text = ""
            if vlm_messages is not None:
                is_clip = getattr(RUNTIME.main_processor, "is_clip_projector", False)
                pref_chat = _vlm_text_only_via_chat_stream()
                mt = (getattr(getattr(RUNTIME.main_model, "config", None), "model_type", None) or "").lower()
                plain_chat = pil_tutor is None and not is_clip and (
                    pref_chat is True or (pref_chat is None and "llava" in mt)
                )
                if plain_chat:
                    for partial in generate_stream_chat(
                        tokenizer=RUNTIME.main_tokenizer,
                        model=RUNTIME.main_model,
                        messages=messages,
                        max_new_tokens=max_new_tokens,
                        lock=lock,
                        extra_generate_kwargs=coder_gen_extras,
                    ):
                        final_text = partial
                        working_display[-1] = {
                            "role": "assistant",
                            "content": compose_assistant_display(base_thinking, render_live_text(partial)),
                        }
                        yield _y7(working_display, conversation_state, display_state, "Writing response…", HIDDEN_PROGRESS, acc_line)
                else:
                    for partial in generate_stream_vlm(
                        processor=RUNTIME.main_processor,
                        tokenizer=RUNTIME.main_tokenizer,
                        model=RUNTIME.main_model,
                        messages=vlm_messages,
                        max_new_tokens=max_new_tokens,
                        lock=lock,
                        image=pil_tutor,
                    ):
                        partial_clean = clean_model_text(partial)
                        final_text = partial_clean
                        working_display[-1] = {
                            "role": "assistant",
                            "content": compose_assistant_display(base_thinking, render_live_text(partial_clean)),
                        }
                        yield _y7(working_display, conversation_state, display_state, "Writing response…", HIDDEN_PROGRESS, acc_line)
            else:
                for partial in generate_stream_chat(
                    tokenizer=tok,
                    model=mdl,
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    lock=lock,
                    extra_generate_kwargs=coder_gen_extras,
                ):
                    final_text = partial
                    working_display[-1] = {
                        "role": "assistant",
                        "content": compose_assistant_display(base_thinking, render_live_text(partial)),
                    }
                    yield _y7(working_display, conversation_state, display_state, "Writing response…", HIDDEN_PROGRESS, acc_line)

        final_text = clean_model_text(final_text)
        if not (final_text or "").strip():
            final_text = (
                "_The model did not return any visible text._ If you attached an image, confirm "
                "**Imagination 1.3.0** is a vision-capable checkpoint, that the upload preview is not "
                "an empty purple placeholder, and check the server terminal for errors."
            )
        working_display[-1] = {
            "role": "assistant",
            "content": compose_assistant_display(base_thinking, final_text),
        }
        new_conv = conversation_state + [
            {"role": "user", "content": model_user_msg},
            {"role": "assistant", "content": final_text},
        ]
        new_disp = display_state + [
            {"role": "user", "content": display_content_for_turn},
            {"role": "assistant", "content": compose_assistant_display(base_thinking, final_text)},
        ]

        save_topic_memory(root_path, user_id, model_user_msg, final_text)
        if user_id and task_id in (TaskId.CAD_CODER, TaskId.REASONING, TaskId.DEEP_RESEARCH):
            save_last_subagent_context(
                root_path,
                user_id,
                task_id.value,
                _snapshot_for_last_subagent(model_user_msg, final_text),
            )

        try:
            task_label_log = next((s.label for s in specs if s.id == task_id), "Imagination 1.3.0")
            log_msgs = messages
            if vlm_messages is not None:
                log_msgs = build_messages(conversation_state, model_user_msg, system_prompt)
            append_training_turn(
                root_path,
                task_id=task_id.value,
                task_label=task_label_log,
                messages=log_msgs,
                thinking_md=thinking_md,
                trace_summary=collapsed_summary,
                step_lines=step_lines,
                thinking_collapsed_html=base_thinking,
                use_web=use_web,
                web_trigger_reason=reason,
                source_cards=source_cards,
                answer=final_text,
                user_message=model_user_msg,
                reasoning_trace=reasoning_trace_for_log,
                has_image=bool(pil_tutor),
                thread_kind=tk,
            )
        except Exception as log_exc:
            print(f"[imagination] training log failed (turn still saved in chat): {log_exc}", flush=True)

        task_label_short = next((s.label for s in specs if s.id == task_id), "Imagination 1.3.0")
        if user_id and tid_thr > 0:
            save_user_chat_thread(root_path, user_id, tid_thr, new_conv, new_disp)
        elif user_id:
            save_user_chat_state(root_path, user_id, new_conv, new_disp)
        thread_dd_upd = maybe_auto_rename_thread_after_first_reply(
            root_path,
            int(user_id) if user_id else 0,
            tid_thr,
            conversation_state,
            display_user_msg,
            final_text,
            task_id,
        )
        yield _y7(new_disp, new_conv, new_disp, f"{task_label_short} ready", HIDDEN_PROGRESS, acc_line, thread_dd_upd)

    except Exception as e:
        err_text = f"Error: {e}"
        working_display[-1] = {"role": "assistant", "content": err_text}
        new_conv = conversation_state + [{"role": "user", "content": model_user_msg}, {"role": "assistant", "content": err_text}]
        new_disp = display_state + [
            {"role": "user", "content": display_content_for_turn},
            {"role": "assistant", "content": err_text},
        ]
        yield _y7(new_disp, new_conv, new_disp, f"Error: {e}", HIDDEN_PROGRESS, acc_line)


def clear_all(
    user_session: Optional[Dict[str, Any]] = None,
    active_thread_id: int = 0,
):
    root = resolve_root_path(None)
    uid, dispn = resolve_effective_user(root, user_session)
    if uid:
        tid = int(active_thread_id or 0)
        if tid > 0:
            save_user_chat_thread(root, uid, tid, [], [])
        else:
            save_user_chat_state(root, uid, [], [])
    acct = account_markdown(uid, dispn)
    return (
        [],
        [],
        [],
        "Ready",
        "<div class='progress-bar hidden'></div>",
        acct,
        gr.update(visible=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value=CHAT_MODE_DEFAULT, interactive=True),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False),
    )


def dismiss_load_modal() -> Tuple[Any, ...]:
    return (
        gr.update(visible=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False),
    )


def hydrate_ui_from_session_json(payload: str):
    """
    Apply session + chat + notes + learner profile from GET /auth/me JSON (fetched in the browser on load).
    Gradio's server-side load handler often does not receive Cookie headers; fetch fixes that.
    """
    import json

    empty_profile = ("", "", "")

    def _guest_banner(banner: str):
        return (
            {"id": None, "name": "", "email": ""},
            [],
            [],
            [],
            banner,
            "",
            *empty_profile,
        )

    guest = _guest_banner(account_markdown(0, "Guest"))
    if not payload or not str(payload).strip():
        return guest
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return guest
    if not isinstance(data, dict):
        return guest
    extra = data.get("banner_extra") or ""
    if not data.get("logged_in"):
        return _guest_banner(account_markdown(0, "Guest") + extra)
    try:
        uid = int(data["id"])
    except (KeyError, TypeError, ValueError):
        return guest
    name = (data.get("name") or "User").strip() or "User"
    sess = {
        "id": uid,
        "name": name,
        "email": (data.get("email") or "").strip(),
    }
    conv = data.get("conv") if isinstance(data.get("conv"), list) else []
    disp = data.get("disp") if isinstance(data.get("disp"), list) else []
    notes = data.get("notes") if isinstance(data.get("notes"), str) else ""
    pa = data.get("profile_about") if isinstance(data.get("profile_about"), str) else ""
    pg = data.get("profile_goals") if isinstance(data.get("profile_goals"), str) else ""
    ps = data.get("profile_skills") if isinstance(data.get("profile_skills"), str) else ""
    banner = account_markdown(uid, name) + extra
    return sess, disp, conv, disp, banner, notes, pa, pg, ps


def standalone_email_login(email: str, password: str):
    root = resolve_root_path(None)
    u = login_email_password(root, email or "", password or "")
    if not u:
        return (
            {"id": None, "name": "", "email": ""},
            [],
            [],
            [],
            account_markdown(0, "Guest"),
            "Invalid email or password.",
            "",
            "",
            "",
            "",
            gr.update(choices=[], value=None),
            0,
            THREAD_KIND_GENERAL,
            gr.update(visible=True, value=None),
            gr.update(visible=False, value=""),
        )
    sess = session_dict_from_user(u)
    notes = load_user_memory(root, u.id)
    pa, pg, ps = load_learner_profile(root, u.id)
    dd, tid, kind, img_u, hint_u, conv, disp = _thread_ui_pack(root, sess)
    return (
        sess,
        disp,
        conv,
        disp,
        account_markdown(u.id, sess["name"]),
        "",
        notes,
        pa,
        pg,
        ps,
        dd,
        tid,
        kind,
        img_u,
        hint_u,
    )


def standalone_email_signup(email: str, password: str, display_name: str):
    root = resolve_root_path(None)
    user, err = signup_email_password(root, email or "", password or "", display_name or "")
    if err or not user:
        return (
            {"id": None, "name": "", "email": ""},
            [],
            [],
            [],
            account_markdown(0, "Guest"),
            err or "Could not sign up.",
            "",
            "",
            "",
            "",
            gr.update(choices=[], value=None),
            0,
            THREAD_KIND_GENERAL,
            gr.update(visible=True, value=None),
            gr.update(visible=False, value=""),
        )
    sess = session_dict_from_user(user)
    notes = load_user_memory(root, user.id)
    pa, pg, ps = load_learner_profile(root, user.id)
    dd, tid, kind, img_u, hint_u, conv, disp = _thread_ui_pack(root, sess)
    return (
        sess,
        disp,
        conv,
        disp,
        account_markdown(user.id, sess["name"]),
        "Account created — start chatting.",
        notes,
        pa,
        pg,
        ps,
        dd,
        tid,
        kind,
        img_u,
        hint_u,
    )


def standalone_logout():
    return (
        {"id": None, "name": "", "email": ""},
        [],
        [],
        [],
        account_markdown(0, "Guest"),
        "Signed out on this device. Sign in again to reload your saved chat from the server.",
        "",
        "",
        "",
        "",
        gr.update(choices=[], value=None),
        0,
        THREAD_KIND_GENERAL,
        gr.update(visible=True, value=None),
        gr.update(visible=False, value=""),
    )


def save_user_notes_ui(
    notes: str,
    profile_about: str,
    profile_goals: str,
    profile_skills: str,
    user_session: Optional[Dict[str, Any]],
):
    root = resolve_root_path(None)
    uid, _ = resolve_effective_user(root, user_session)
    if not uid:
        return "Sign in to save notes and learner profile."
    save_user_memory(root, uid, notes or "")
    save_learner_profile(root, uid, profile_about, profile_goals, profile_skills)
    return "Saved notes and learner profile (used in future replies for your account)."


# ----------------------------
# UI
# ----------------------------
_UNLOAD_TOAST_MSG = {
    "unload_warn_1min": "Extra models unload in 1 minute.",
    "unload_warn_15s": "Extra models unload in 15 seconds.",
}

TUTOR_SLASH_HINTS_MD = (
    "<div class='tutor-hint-strip'>"
    "<strong>Tutor</strong> · <code>/explain &lt;n&gt;</code> — dig into step <em>n</em> from the last answer "
    "(add what makes sense vs. what confuses you after the number). "
    "<span class='tutor-integrity'>For learning — follow your class rules.</span>"
    "</div>"
)


def poll_unload_toast_ui(state: dict) -> Tuple[dict, Any]:
    """Gradio timer: show slide-in toasts when eviction warnings fire."""
    now = time.monotonic()
    key = LIFECYCLE.pop_unload_toast_key()
    text = str(state.get("text") or "").strip()
    until = float(state.get("until") or 0.0)
    anim_id = int(state.get("anim_id") or 0)

    if key is not None:
        text = _UNLOAD_TOAST_MSG.get(str(key), str(key))
        until = now + 3.65
        anim_id += 1
    elif now >= until:
        text = ""
        until = 0.0

    new_state = {"text": text, "until": until, "anim_id": anim_id}
    if not text:
        inner = ""
    else:
        inner = (
            f'<div class="im-toast" data-a="{anim_id}">'
            f'<div class="im-toast-inner">{html.escape(text)}</div></div>'
        )
    out_html = f'<div id="imagination-toast-root" class="toast-stack-root">{inner}</div>'
    return new_state, gr.update(value=out_html)


def _thread_choice_label(t: Dict[str, Any]) -> str:
    kind = (t.get("kind") or "").strip()
    tag = "Tutor" if kind == THREAD_KIND_TUTOR else "General"
    title = (t.get("title") or "Chat").strip()[:56]
    return f"{tag} · {title}"


def thread_dropdown_update(
    root: str,
    user_id: int,
    select_id: Optional[int] = None,
) -> Tuple[Any, int, str]:
    if not user_id:
        return gr.update(choices=[], value=None), 0, THREAD_KIND_GENERAL
    threads = list_user_chat_threads(root, user_id)
    if not threads:
        tid_new = create_user_chat_thread(root, user_id, THREAD_KIND_GENERAL, "Chat")
        set_active_thread_id(root, user_id, tid_new)
        threads = list_user_chat_threads(root, user_id)
    choices = [(_thread_choice_label(t), str(t["id"])) for t in threads]
    pick: Optional[int] = int(select_id) if select_id else None
    if pick is None or not any(int(t["id"]) == int(pick) for t in threads):
        pick = pick_default_thread_id(root, user_id)
        if pick is None:
            pick = int(threads[0]["id"])
    pick = int(pick)
    set_active_thread_id(root, user_id, pick)
    kind = THREAD_KIND_GENERAL
    for t in threads:
        if int(t["id"]) == pick:
            kind = str(t.get("kind") or THREAD_KIND_GENERAL)
            break
    return gr.update(choices=choices, value=str(pick)), pick, kind


def apply_thread_selection(
    root: str,
    user_id: int,
    thread_id_str: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int, str]:
    if not user_id or not (thread_id_str or "").strip():
        return [], [], [], 0, THREAD_KIND_GENERAL
    try:
        tid = int(thread_id_str)
    except (TypeError, ValueError):
        return [], [], [], 0, THREAD_KIND_GENERAL
    set_active_thread_id(root, user_id, tid)
    row = get_user_chat_thread(root, user_id, tid)
    if not row:
        return [], [], [], tid, THREAD_KIND_GENERAL
    conv = row.get("conversation") or []
    disp = row.get("display") or []
    k = str(row.get("kind") or THREAD_KIND_GENERAL)
    return conv, disp, disp, tid, k


def thread_ui_refresh(user_session: Optional[Dict[str, Any]], current_value: Optional[str] = None):
    root = resolve_root_path(None)
    uid, _ = resolve_effective_user(root, user_session)
    if not uid:
        return gr.update(choices=[], value=None), 0, THREAD_KIND_GENERAL
    sel: Optional[int] = None
    if current_value:
        try:
            sel = int(current_value)
        except (TypeError, ValueError):
            sel = None
    return thread_dropdown_update(root, uid, select_id=sel)


def thread_ui_on_select(thread_id_str: str, user_session: Optional[Dict[str, Any]]):
    root = resolve_root_path(None)
    uid, _ = resolve_effective_user(root, user_session)
    return apply_thread_selection(root, uid, thread_id_str or "")


def thread_ui_select_with_visibility(thread_id_str: str, user_session: Optional[Dict[str, Any]]):
    conv, disp, chatv, tid, k = thread_ui_on_select(thread_id_str, user_session)
    vis_tutor = k == THREAD_KIND_TUTOR
    img_u = gr.update(visible=True)
    hint_u = gr.update(visible=vis_tutor, value=(TUTOR_SLASH_HINTS_MD if vis_tutor else ""))
    return conv, disp, chatv, tid, k, img_u, hint_u


def thread_ui_new(kind_key: str, user_session: Optional[Dict[str, Any]]):
    root = resolve_root_path(None)
    uid, _ = resolve_effective_user(root, user_session)
    if not uid:
        return (
            gr.update(choices=[], value=None),
            0,
            THREAD_KIND_GENERAL,
            [],
            [],
            [],
            gr.update(visible=True, value=None),
            gr.update(visible=False, value=""),
        )
    k = THREAD_KIND_TUTOR if (kind_key or "").strip().lower() == "tutor" else THREAD_KIND_GENERAL
    title = "Math tutor" if k == THREAD_KIND_TUTOR else "New chat"
    tid = create_user_chat_thread(root, uid, k, title)
    set_active_thread_id(root, uid, tid)
    dd, tid2, k2 = thread_dropdown_update(root, uid, select_id=tid)
    vis_tutor = k2 == THREAD_KIND_TUTOR
    img_u = gr.update(visible=True)
    hint_u = gr.update(visible=vis_tutor, value=(TUTOR_SLASH_HINTS_MD if vis_tutor else ""))
    return dd, tid2, k2, [], [], [], img_u, hint_u


def thread_ui_delete(thread_id_str: str, user_session: Optional[Dict[str, Any]]):
    root = resolve_root_path(None)
    uid, _ = resolve_effective_user(root, user_session)
    if not uid or not (thread_id_str or "").strip():
        return (
            gr.update(),
            0,
            THREAD_KIND_GENERAL,
            [],
            [],
            [],
            gr.update(visible=True, value=None),
            gr.update(visible=False, value=""),
        )
    try:
        tid = int(thread_id_str)
    except (TypeError, ValueError):
        return (
            gr.update(),
            0,
            THREAD_KIND_GENERAL,
            [],
            [],
            [],
            gr.update(visible=True, value=None),
            gr.update(visible=False, value=""),
        )
    delete_user_chat_thread(root, uid, tid)
    dd, tid2, k2 = thread_dropdown_update(root, uid, select_id=None)
    conv, disp, chatv, _tid_sel, k_after = apply_thread_selection(root, uid, str(tid2))
    vis_tutor = k_after == THREAD_KIND_TUTOR
    img_u = gr.update(visible=True)
    hint_u = gr.update(visible=vis_tutor, value=(TUTOR_SLASH_HINTS_MD if vis_tutor else ""))
    return dd, tid2, k2, conv, disp, chatv, img_u, hint_u


def thread_ui_rename(new_title: str, thread_id_str: str, user_session: Optional[Dict[str, Any]]):
    root = resolve_root_path(None)
    uid, _ = resolve_effective_user(root, user_session)
    if not uid or not (thread_id_str or "").strip():
        return gr.update(), gr.update()
    try:
        tid = int(thread_id_str)
    except (TypeError, ValueError):
        return gr.update(), gr.update()
    rename_user_chat_thread(root, uid, tid, (new_title or "").strip() or "Untitled")
    dd, _, _ = thread_dropdown_update(root, uid, select_id=tid)
    return dd, gr.update(value="")


def _thread_ui_pack(root: str, sess: Dict[str, Any]):
    uid, _ = resolve_effective_user(root, sess)
    if not uid:
        return (
            gr.update(choices=[], value=None),
            0,
            THREAD_KIND_GENERAL,
            gr.update(visible=True, value=None),
            gr.update(visible=False, value=""),
            [],
            [],
        )
    dd, tid, kind = thread_ui_refresh(sess, None)
    conv, disp, _, _, _ = apply_thread_selection(root, uid, str(tid))
    vis_tutor = kind == THREAD_KIND_TUTOR
    img_u = gr.update(visible=True)
    hint_u = gr.update(visible=vis_tutor, value=(TUTOR_SLASH_HINTS_MD if vis_tutor else ""))
    return dd, tid, kind, img_u, hint_u, conv, disp


def hydrate_ui_from_session_json_with_threads(payload: str) -> Tuple[Any, ...]:
    base = hydrate_ui_from_session_json(payload)
    root = resolve_root_path(None)
    sess = base[0]
    uid, _ = resolve_effective_user(root, sess)
    dd, tid, kind, img_u, hint_u, conv, disp = _thread_ui_pack(root, sess)
    if uid:
        b = list(base)
        b[1] = disp
        b[2] = conv
        b[3] = disp
        return tuple(b) + (dd, tid, kind, img_u, hint_u)
    return base + (dd, tid, kind, img_u, hint_u)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg0: #06080d;
  --bg1: #0a0d14;
  --surface: rgba(255,255,255,0.04);
  --surface2: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.1);
  --border-strong: rgba(255,255,255,0.16);
  --text: rgba(248,250,255,0.96);
  --muted: rgba(160,175,200,0.7);
  --accent: #5ee0d0;
  --accent-dim: rgba(94,224,208,0.12);
  --accent-glow: rgba(94,224,208,0.35);
  --term-orange: #ff7000;
  --term-orange-dim: rgba(255,112,0,0.2);
  --radius-lg: 14px;
  --radius-md: 10px;
  --shadow: 0 20px 56px rgba(0,0,0,0.5);
}

.gradio-container {
  font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
  background:
    radial-gradient(ellipse 100% 70% at 80% -10%, rgba(94,224,208,0.08), transparent 45%),
    radial-gradient(ellipse 80% 50% at 10% 0%, rgba(255,112,0,0.06), transparent 42%),
    linear-gradient(168deg, var(--bg1) 0%, var(--bg0) 100%) !important;
  color: var(--text);
  min-height: 100vh;
  position: relative;
}

.gradio-container::after {
  content: "";
  pointer-events: none;
  position: fixed;
  inset: 0;
  z-index: 1;
  opacity: 0.04;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.18) 2px,
    rgba(0,0,0,0.18) 3px
  );
}

.toast-stack-root {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 100000;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 10px;
  pointer-events: none;
  max-width: min(92vw, 340px);
}

.im-toast {
  pointer-events: none;
  animation: imToastCycle 3.5s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

.im-toast-inner {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em;
  padding: 12px 16px 12px 14px;
  border-radius: var(--radius-md);
  border: 1px solid rgba(94,224,208,0.35);
  background: linear-gradient(145deg, rgba(12,18,26,0.97), rgba(8,12,18,0.98));
  color: rgba(220,255,248,0.95) !important;
  box-shadow:
    0 0 0 1px rgba(0,0,0,0.35) inset,
    0 12px 40px rgba(0,0,0,0.45),
    0 0 28px rgba(94,224,208,0.12);
}

@keyframes imToastCycle {
  0% { transform: translateX(calc(100% + 24px)); opacity: 0; }
  11% { transform: translateX(0); opacity: 1; }
  78% { transform: translateX(0); opacity: 1; }
  100% { transform: translateX(calc(100% + 24px)); opacity: 0; }
}

@media (prefers-reduced-motion: reduce) {
  .im-toast { animation: none; opacity: 1; transform: none; }
}

#shell {
  max-width: 1400px;
  width: 100%;
  margin: 0 auto;
  padding: 24px 20px 48px;
  position: relative;
  z-index: 2;
  box-sizing: border-box;
}

#app-layout {
  display: flex !important;
  flex-direction: row !important;
  align-items: flex-start !important;
  gap: 22px;
  width: 100% !important;
}

#thread-sidebar {
  width: min(300px, 30vw);
  min-width: 240px;
  max-width: 340px;
  flex-shrink: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 10px !important;
  padding: 14px 12px 16px !important;
  border-radius: var(--radius-lg) !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  align-self: flex-start !important;
  max-height: min(88vh, 920px) !important;
}

.sidebar-title {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  font-size: 1.1rem !important;
  letter-spacing: -0.02em !important;
  text-transform: none !important;
  color: var(--text) !important;
  font-weight: 700 !important;
  margin: 0 0 8px 0 !important;
  padding: 0 2px !important;
}

#thread-sidebar #thread-dropdown {
  width: 100% !important;
}
#thread-sidebar #thread-dropdown label {
  display: none !important;
}
#thread-sidebar #thread-dropdown .wrap,
#thread-sidebar #thread-dropdown .secondary-wrap {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border-strong) !important;
  min-height: 44px !important;
}

#main-chat-column {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  display: flex !important;
  flex-direction: column !important;
}

.terminal-split {
  display: grid !important;
  grid-template-columns: 1fr 1fr;
  gap: 22px;
  align-items: stretch;
  margin-bottom: 8px;
  width: 100% !important;
}
@media (max-width: 960px) {
  .terminal-split {
    grid-template-columns: 1fr !important;
  }
}
.terminal-col {
  display: flex !important;
  flex-direction: column !important;
  gap: 8px;
  min-width: 0;
}
.terminal-field-label {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  font-weight: 600 !important;
}
.terminal-panel {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 10px;
  box-shadow: var(--shadow);
  min-height: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.terminal-panel:focus-within {
  border-color: rgba(94,224,208,0.28);
  box-shadow: var(--shadow), 0 0 0 3px var(--accent-dim);
}

/* v1.2.1-style single “message box”: image strip + text share one panel */
.composer-unified {
  gap: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
.composer-unified .tutor-hint-wrap {
  margin: 8px 10px 4px 10px !important;
}
.composer-unified #tutor-image-upload {
  margin: 0 !important;
  max-height: 130px !important;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  padding: 4px 8px 8px 8px !important;
}
.composer-unified #tutor-image-upload label,
.composer-unified #tutor-image-upload .label-wrap {
  display: none !important;
}
.composer-unified #tutor-image-upload .form {
  min-height: 0 !important;
  max-height: 118px !important;
  gap: 4px !important;
}
.composer-unified .composer-inner-text {
  border-top: 1px solid var(--border);
  padding: 8px 10px 10px 10px !important;
  margin: 0 !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
}
.composer-unified .composer-inner-text.wrap,
.composer-unified .composer-inner-text .wrap {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}
.composer-unified .composer-input textarea {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  min-height: 10rem !important;
  max-height: 40vh !important;
  line-height: 1.45 !important;
}

.chat-wrap {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 4px;
  margin-bottom: 0;
  box-shadow: var(--shadow);
  min-height: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
}
#chatbox {
  min-height: 420px;
  flex: 1;
  border: none !important;
  border-radius: calc(var(--radius-lg) - 4px) !important;
  background: transparent !important;
}
#chatbox .form {
  min-height: 0 !important;
  flex: 1 1 auto !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
}
#chatbox [class*="panel-wrap"],
#chatbox [class*="bubble-wrap"] {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  max-height: 100% !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
  scrollbar-color: rgba(255,255,255,0.35) rgba(0,0,0,0.25);
}
#chatbox [class*="panel-wrap"]::-webkit-scrollbar,
#chatbox [class*="bubble-wrap"]::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}
#chatbox [class*="panel-wrap"]::-webkit-scrollbar-thumb,
#chatbox [class*="bubble-wrap"]::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.22);
  border-radius: 6px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
#chatbox [class*="panel-wrap"]::-webkit-scrollbar-track,
#chatbox [class*="bubble-wrap"]::-webkit-scrollbar-track {
  background: rgba(0,0,0,0.2);
  border-radius: 6px;
}

@media (max-width: 900px) {
  #app-layout {
    flex-direction: column !important;
  }
  #thread-sidebar {
    width: 100% !important;
    max-width: none !important;
    max-height: 40vh !important;
    align-self: stretch !important;
  }
}

.term-version {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  opacity: 0.85;
  margin-left: auto !important;
}

#model-bar {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 10px 14px !important;
  align-items: center !important;
  margin-bottom: 18px !important;
  padding: 12px 14px !important;
  border-radius: var(--radius-md) !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
#model-bar .model-bar-label {
  font-size: 0.68rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  color: var(--muted) !important;
  font-weight: 600 !important;
  margin-right: 2px !important;
}
#model-bar .mode-soon-sep {
  opacity: 0.22 !important;
  color: var(--muted) !important;
  font-size: 0.75rem !important;
  margin: 0 0.2em !important;
  user-select: none !important;
}
#model-bar .mode-coming-soon {
  opacity: 0.38 !important;
  font-size: 0.75rem !important;
  color: var(--muted) !important;
  cursor: default !important;
  white-space: nowrap !important;
}
#thread-sidebar .sidebar-new-thread button {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.04em !important;
  text-transform: none !important;
  padding: 12px 14px !important;
  border-radius: var(--radius-md) !important;
  color: #061016 !important;
  background: linear-gradient(165deg, #8af5e8 0%, var(--accent) 42%, #3dbfb0 100%) !important;
  border: 1px solid rgba(120, 245, 220, 0.5) !important;
  box-shadow: 0 4px 18px rgba(94,224,208,0.28) !important;
  transition: transform 0.15s ease, filter 0.2s ease !important;
}
#thread-sidebar .sidebar-new-thread button:hover {
  transform: translateY(-1px);
  filter: brightness(1.06);
}
#model-version-dropdown {
  min-width: 148px !important;
  flex: 0 0 auto !important;
}
#model-version-dropdown label {
  display: none !important;
}
#model-version-dropdown .wrap,
#model-version-dropdown .secondary-wrap {
  border-radius: var(--radius-lg) !important;
  border: 1px solid var(--border-strong) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.07) 0%, var(--bg0) 100%) !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05) !important;
  min-height: 44px !important;
}
#model-version-dropdown .wrap:focus-within,
#model-version-dropdown .secondary-wrap:focus-within {
  border-color: rgba(124,142,240,0.45) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
#model-version-dropdown input,
#model-version-dropdown button[role="combobox"] {
  border-radius: var(--radius-lg) !important;
  font-size: 0.8125rem !important;
  font-weight: 600 !important;
  color: var(--text) !important;
  background: transparent !important;
}
.app-brand h1 {
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin: 0 0 4px 0;
  color: var(--text);
}
.app-brand .ver {
  font-size: 0.72em;
  font-weight: 500;
  color: var(--muted);
  margin-left: 0.35em;
  vertical-align: middle;
  letter-spacing: 0.04em;
}

.load-modal-backdrop {
  position: fixed !important;
  inset: 0 !important;
  z-index: 99990 !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: center !important;
  background: rgba(5,7,12,0.48) !important;
  backdrop-filter: blur(8px) !important;
  -webkit-backdrop-filter: blur(8px) !important;
  padding: 24px !important;
}
.load-modal-backdrop > .form,
.load-modal-backdrop > div:not(.load-modal-card) {
  width: auto !important;
  max-width: min(50vw, 560px) !important;
  max-height: 50vh !important;
  margin: 0 auto !important;
  flex-shrink: 0 !important;
}
.load-modal-card {
  background: linear-gradient(165deg, rgba(16,22,32,0.98), rgba(6,8,12,0.99)) !important;
  border: 1px solid rgba(94,224,208,0.32) !important;
  border-radius: var(--radius-lg) !important;
  padding: 22px 20px 20px !important;
  box-shadow:
    0 32px 80px rgba(0,0,0,0.6),
    0 0 0 1px rgba(255,255,255,0.04) inset,
    0 0 48px rgba(94,224,208,0.08) !important;
  width: min(50vw, 560px) !important;
  max-width: min(92vw, 560px) !important;
  max-height: min(50vh, 480px) !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  animation: loadModalCardIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}

@keyframes loadModalCardIn {
  from {
    opacity: 0;
    transform: scale(0.94) translateY(12px);
    filter: blur(2px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
    filter: blur(0);
  }
}

.load-modal-body--busy .modal-pfill {
  position: relative;
  overflow: hidden;
}

.load-modal-body--busy .modal-pfill::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(
    100deg,
    transparent 0%,
    rgba(255,255,255,0.22) 48%,
    transparent 96%
  );
  animation: modalBarSheen 1.35s ease-in-out infinite;
}

@keyframes modalBarSheen {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.load-modal-body--complete .modal-pfill::after {
  display: none;
}

.load-done-strip {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
  padding: 10px 14px;
  border-radius: var(--radius-md);
  border: 1px solid rgba(94,224,208,0.35);
  background: rgba(94,224,208,0.08);
  animation: loadDoneStripIn 0.65s cubic-bezier(0.22, 1, 0.36, 1) both 0.12s;
}

.load-done-glyph {
  display: inline-flex;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: linear-gradient(145deg, rgba(94,224,208,0.35), rgba(94,224,208,0.12));
  border: 1px solid rgba(94,224,208,0.45);
  position: relative;
  animation: loadDoneGlyphPop 0.55s cubic-bezier(0.34, 1.4, 0.64, 1) both 0.2s;
}

.load-done-glyph::after {
  content: "";
  position: absolute;
  left: 6px;
  top: 3px;
  width: 6px;
  height: 10px;
  border: solid rgba(230,255,250,0.95);
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}

.load-done-text {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: var(--text) !important;
  letter-spacing: 0.04em;
}

@keyframes loadDoneStripIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes loadDoneGlyphPop {
  from { transform: scale(0.3); opacity: 0; }
  70% { transform: scale(1.08); }
  to { transform: scale(1); opacity: 1; }
}

.modal-pbar {
  height: 8px;
  border-radius: 999px;
  background: var(--border);
  overflow: hidden;
  margin: 10px 0 14px;
}
.modal-pfill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--term-orange), #ff9a4d, var(--accent));
  transition: width 0.2s ease-out;
  box-shadow: 0 0 14px var(--term-orange-dim), 0 0 20px var(--accent-glow);
}
.load-modal-card h3 {
  margin: 0 0 10px 0 !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  color: var(--text) !important;
}
.load-modal-card .prose, .load-modal-card p, .load-modal-card li {
  color: var(--muted) !important;
  font-size: 0.875rem !important;
  line-height: 1.5 !important;
}
#modal-dismiss-btn {
  margin-top: 16px !important;
  width: 100% !important;
  transition: transform 0.15s ease, filter 0.2s ease !important;
}
#modal-dismiss-btn:hover {
  transform: translateY(-1px);
  filter: brightness(1.08);
}

.status-strip {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 12px 16px 14px;
  margin-bottom: 10px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(20px);
  flex-shrink: 0;
}

.account-dock {
  flex-shrink: 0 !important;
  margin-top: 8px !important;
  max-height: min(38vh, 420px) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
}

#status-wrap {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}
#status-wrap p,
#status p {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.8125rem;
  color: var(--muted);
  margin: 0 !important;
}
#status-wrap p::before,
#status p::before {
  content: '';
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 14px var(--accent-glow);
}

.progress-bar {
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  margin: 0 0 14px 0;
  overflow: hidden;
}
.progress-bar.hidden { display: none; }
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #5a6fd4, var(--accent), #a8b4ff);
  border-radius: 2px;
  transition: width 0.25s ease;
}

/* Collapsible in-chat thinking (HTML inside assistant bubble) */
#chatbox details.thinking-block {
  margin: 0 0 12px 0;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: rgba(0,0,0,0.22);
  font-size: 0.8125rem;
  color: var(--muted);
  line-height: 1.45;
  transition: border-color 0.2s ease, background 0.2s ease;
}
#chatbox details.thinking-block[open] {
  border-color: rgba(124,142,240,0.28);
  background: rgba(124,142,240,0.06);
}
#chatbox details.thinking-block.thinking-done:not([open]) {
  margin-bottom: 10px;
  opacity: 0.92;
}
#chatbox details.thinking-block summary.thinking-summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  color: var(--muted);
  user-select: none;
}
#chatbox details.thinking-block summary.thinking-summary::-webkit-details-marker {
  display: none;
}
#chatbox details.thinking-block .thinking-body {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
  max-height: 220px;
  overflow-y: auto;
}
#chatbox details.thinking-block .thinking-step {
  padding: 4px 0;
  border-left: 2px solid var(--border-strong);
  padding-left: 10px;
  margin-left: 2px;
}
#chatbox details.thinking-block .thinking-step:last-child {
  margin-bottom: 0;
}
#chatbox details.thinking-block .thinking-em {
  color: #ffffff;
  font-weight: 700;
  font-size: inherit;
}
/* Subagent weight load: percentage + name pulse with indigo accent (matches --accent) */
#chatbox details.thinking-block .thinking-step--loading {
  border-left: 2px solid rgba(124, 142, 240, 0.42);
  padding-left: 10px;
  margin-left: 2px;
  border-radius: 0 8px 8px 0;
  background: linear-gradient(
    105deg,
    rgba(124, 142, 240, 0.1) 0%,
    rgba(124, 142, 240, 0.03) 42%,
    transparent 72%
  );
  background-size: 220% 100%;
  animation: thinking-load-border 1.85s ease-in-out infinite,
    thinking-load-sheen 2.6s ease-in-out infinite;
}
@keyframes thinking-load-border {
  0%, 100% {
    border-left-color: rgba(124, 142, 240, 0.32);
    box-shadow: inset 3px 0 14px rgba(124, 142, 240, 0.07);
  }
  50% {
    border-left-color: rgba(168, 180, 255, 0.85);
    box-shadow: inset 3px 0 20px rgba(124, 142, 240, 0.16);
  }
}
@keyframes thinking-load-sheen {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}
#chatbox details.thinking-block .thinking-step--loading .thinking-em {
  animation: thinking-em-shine 1.45s ease-in-out infinite;
}
@keyframes thinking-em-shine {
  0%, 100% {
    color: #f2f4ff;
    text-shadow: 0 0 0 transparent;
  }
  50% {
    color: #ffffff;
    text-shadow:
      0 0 10px rgba(124, 142, 240, 0.5),
      0 0 22px rgba(124, 142, 240, 0.22);
  }
}
@media (prefers-reduced-motion: reduce) {
  #chatbox details.thinking-block .thinking-step--loading,
  #chatbox details.thinking-block .thinking-step--loading .thinking-em {
    animation: none;
  }
  #chatbox details.thinking-block .thinking-step--loading {
    background: rgba(124, 142, 240, 0.06);
    background-size: auto;
  }
}
#chatbox .thinking-dots {
  display: inline-flex;
  gap: 3px;
  align-items: center;
  margin-left: 4px;
}
#chatbox .thinking-dots span {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--accent);
  opacity: 0.35;
  animation: thinking-pulse 1.1s ease-in-out infinite;
}
#chatbox .thinking-dots span:nth-child(2) { animation-delay: 0.15s; }
#chatbox .thinking-dots span:nth-child(3) { animation-delay: 0.3s; }
#chatbox details.thinking-block:not([open]) .thinking-dots {
  display: none;
}
@keyframes thinking-pulse {
  0%, 100% { opacity: 0.25; transform: scale(0.9); }
  50% { opacity: 1; transform: scale(1.05); }
}

/* Assistant bubble accent */
#chatbox .message.bot,
#chatbox [class*="bot"] {
  border-left: 3px solid rgba(124,142,240,0.45) !important;
  padding-left: 12px !important;
  border-radius: var(--radius-md) !important;
}

.composer-row,
.terminal-actions {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 10px !important;
  align-items: center !important;
  margin-top: 4px !important;
}
.terminal-actions {
  padding-top: 4px;
}
#chat-mode-dropdown {
  min-width: 168px !important;
  flex: 1 1 200px !important;
  max-width: 320px !important;
}
#chat-mode-dropdown label {
  display: none !important;
}
#chat-mode-dropdown .wrap,
#chat-mode-dropdown .secondary-wrap {
  border-radius: var(--radius-lg) !important;
  border: 1px solid var(--border-strong) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.07) 0%, var(--bg0) 100%) !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05) !important;
  min-height: 48px !important;
}
#chat-mode-dropdown .wrap:focus-within,
#chat-mode-dropdown .secondary-wrap:focus-within {
  border-color: rgba(124,142,240,0.45) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
#chat-mode-dropdown input,
#chat-mode-dropdown button[role="combobox"] {
  border-radius: var(--radius-lg) !important;
  font-size: 0.8125rem !important;
  font-weight: 600 !important;
  color: var(--text) !important;
  background: transparent !important;
}
#chat-mode-dropdown span[data-testid] {
  font-weight: 600 !important;
}

.terminal-panel textarea,
.composer-row textarea,
.input-row textarea {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 0.875rem !important;
  line-height: 1.45 !important;
  flex: 1;
  min-height: 420px !important;
  max-height: 560px !important;
  background: rgba(4,6,10,0.85) !important;
  color: rgba(236,245,255,0.95) !important;
  border: 1px solid var(--border) !important;
  border-radius: calc(var(--radius-lg) - 2px) !important;
  padding: 14px 16px !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.terminal-panel textarea:focus,
.composer-row textarea:focus,
.input-row textarea:focus {
  border-color: rgba(94,224,208,0.4) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
}

.terminal-actions button.primary,
.composer-row button.primary,
.input-row button.primary {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.8125rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding: 12px 24px !important;
  border-radius: var(--radius-md) !important;
  background: linear-gradient(180deg, #ff8a2b 0%, var(--term-orange) 45%, #e85f00 100%) !important;
  border: 1px solid rgba(255,160,80,0.55) !important;
  color: #fff !important;
  box-shadow: 0 4px 20px rgba(255,112,0,0.35), 0 1px 0 rgba(255,255,255,0.12) inset !important;
  transition: transform 0.18s ease, box-shadow 0.22s ease, filter 0.2s ease !important;
}
.terminal-actions button.primary:hover,
.composer-row button.primary:hover,
.input-row button.primary:hover {
  transform: translateY(-1px) scale(1.02);
  filter: brightness(1.06);
  box-shadow: 0 8px 28px rgba(255,112,0,0.45), 0 1px 0 rgba(255,255,255,0.15) inset !important;
}
.terminal-actions button.primary:active,
.composer-row button.primary:active,
.input-row button.primary:active {
  transform: translateY(0) scale(0.98);
  box-shadow: 0 2px 12px rgba(255,112,0,0.3) !important;
}

.terminal-actions button.secondary,
.composer-row button.secondary,
.input-row button.secondary {
  font-size: 0.8125rem !important;
  padding: 12px 18px !important;
  border-radius: var(--radius-md) !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--muted) !important;
  transition: transform 0.16s ease, border-color 0.2s ease, color 0.2s ease, background 0.2s ease !important;
}
.terminal-actions button.secondary:hover,
.composer-row button.secondary:hover,
.input-row button.secondary:hover {
  color: var(--text) !important;
  border-color: rgba(255,255,255,0.22) !important;
  background: rgba(255,255,255,0.07) !important;
  transform: translateY(-1px);
}
.terminal-actions button.secondary:active,
.composer-row button.secondary:active,
.input-row button.secondary:active {
  transform: scale(0.97);
}

@media (prefers-reduced-motion: reduce) {
  .terminal-actions button.primary,
  .terminal-actions button.secondary,
  .load-modal-card,
  .load-done-strip,
  .load-done-glyph {
    animation: none !important;
    transition: none !important;
  }
  .terminal-actions button.primary:hover,
  .terminal-actions button.secondary:hover {
    transform: none;
  }
}

#thread-dropdown label { display: none !important; }
.tutor-hint-wrap {
  margin-bottom: 8px !important;
}
.tutor-hint-strip {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 0.72rem !important;
  line-height: 1.45 !important;
  color: var(--muted) !important;
  padding: 8px 10px !important;
  border-radius: var(--radius-md) !important;
  border: 1px solid rgba(94,224,208,0.22) !important;
  background: rgba(94,224,208,0.06) !important;
}
.tutor-hint-strip code {
  color: var(--accent) !important;
  font-size: 0.85em !important;
}
.tutor-integrity {
  display: block;
  margin-top: 4px;
  opacity: 0.85;
  font-size: 0.9em;
}
/* Image attachment strip: short bar, thumbnail scales up vs empty purple gutter */
#tutor-image-upload,
#tutor-image-upload.composer-image-attach {
  flex: 0 0 auto !important;
  margin-bottom: 6px !important;
  max-height: 128px !important;
  min-height: 0 !important;
}
#tutor-image-upload .form {
  gap: 4px !important;
  min-height: 0 !important;
  max-height: 122px !important;
}
#tutor-image-upload .form > div {
  min-height: 0 !important;
  max-height: 118px !important;
}
#tutor-image-upload [class*="image-container"],
#tutor-image-upload [class*="upload-container"],
#tutor-image-upload [class*="image-frame"] {
  max-height: 112px !important;
  min-height: 0 !important;
}
/* When a preview exists, keep the purple upload gutter narrow so the thumb gets width */
#tutor-image-upload:has(img) [class*="upload-container"] {
  flex: 0 1 26% !important;
  max-width: 140px !important;
  min-width: 72px !important;
}
#tutor-image-upload:has(img) [class*="image-container"] {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}
#tutor-image-upload img {
  max-height: 104px !important;
  width: auto !important;
  max-width: 100% !important;
  object-fit: contain !important;
}
"""


def build_ui(auth_http_available: bool = False):
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["DM Sans", "system-ui", "sans-serif"],
        font_mono=["JetBrains Mono", "monospace"],
    ).set(
        body_background_fill="transparent",
        block_background_fill="transparent",
        block_border_width="0",
        block_label_text_size="sm",
    )

    _google_on = bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))
    _hf_oauth = bool(os.getenv("SPACE_ID") or os.getenv("HF_TOKEN"))
    if not _hf_oauth:
        try:
            from huggingface_hub import get_token

            _hf_oauth = get_token() is not None
        except Exception:
            _hf_oauth = False

    with gr.Blocks(css=CSS, theme=theme, title="Imagination v1.4 UI Artisan") as demo:
        toast_state = gr.State({"text": "", "until": 0.0, "anim_id": 0})
        toast_stack = gr.HTML(
            '<div id="imagination-toast-root" class="toast-stack-root" aria-live="polite"></div>',
        )

        with gr.Column(elem_id="shell"):
            user_session = gr.State({"id": None, "name": "", "email": ""})
            active_thread_id = gr.State(0)
            thread_kind_state = gr.State(THREAD_KIND_GENERAL)
            _session_json_bridge = gr.Textbox(value="", visible=False, elem_id="imagination-session-json-bridge")
            auth_banner = gr.Markdown(account_markdown(0, "Guest"))
            login_hint = gr.Markdown("")

            with gr.Row(elem_id="app-layout"):
                with gr.Column(elem_id="thread-sidebar", scale=0, min_width=272):
                    gr.HTML('<div class="sidebar-title">Chats</div>')
                    thread_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Conversation",
                        elem_id="thread-dropdown",
                    )
                    with gr.Row():
                        btn_thread_new_gen = gr.Button(
                            "New general",
                            scale=1,
                            min_width=100,
                            elem_classes=["sidebar-new-thread"],
                        )
                        btn_thread_new_tut = gr.Button(
                            "New tutor",
                            scale=1,
                            min_width=100,
                            elem_classes=["sidebar-new-thread"],
                        )
                    thread_rename_box = gr.Textbox(
                        placeholder="Rename active thread…",
                        show_label=False,
                        max_lines=1,
                    )
                    with gr.Row():
                        btn_thread_rename = gr.Button("Rename", scale=1, min_width=80)
                        btn_thread_del = gr.Button("Delete", scale=1, min_width=80)

                with gr.Column(elem_id="main-chat-column", scale=1, min_width=320):
                    with gr.Row(elem_id="model-bar"):
                        gr.HTML('<span class="model-bar-label">Model</span>')
                        model_version = gr.Dropdown(
                            choices=MODEL_VERSION_CHOICES,
                            value=MODEL_VERSION_DEFAULT,
                            label=None,
                            show_label=False,
                            elem_id="model-version-dropdown",
                            container=False,
                            scale=0,
                            min_width=160,
                        )
                        gr.HTML(
                            '<span class="mode-soon-sep">·</span>'
                            '<span class="mode-coming-soon" title="Coming soon">Reasoning</span>'
                            '<span class="mode-soon-sep">·</span>'
                            '<span class="mode-coming-soon" title="Coming soon">Research</span>'
                            '<span class="mode-soon-sep">·</span>'
                            '<span class="mode-coming-soon" title="Coming soon">Image gen</span>'
                        )
                        gr.HTML('<span class="model-bar-label">Mode</span>')
                        chat_mode = gr.Dropdown(
                            choices=CHAT_MODE_CHOICES,
                            value=CHAT_MODE_DEFAULT,
                            label=None,
                            show_label=False,
                            elem_id="chat-mode-dropdown",
                            container=False,
                            scale=1,
                            min_width=176,
                        )
                        gr.HTML('<span class="term-version">v1.4.0</span>')

                    with gr.Column(elem_classes=["status-strip"]):
                        progress_html = gr.HTML("<div class='progress-bar hidden'></div>")
                        with gr.Row(elem_id="status-wrap"):
                            status = gr.Markdown("Ready", elem_id="status")

                    with gr.Row(elem_classes=["terminal-split"]):
                        with gr.Column(elem_classes=["terminal-col"]):
                            gr.HTML('<div class="terminal-field-label">Input</div>')
                            with gr.Column(elem_classes=["terminal-panel", "composer-unified"]):
                                tutor_hint_md = gr.Markdown(
                                    value="",
                                    visible=False,
                                    elem_classes=["tutor-hint-wrap"],
                                )
                                _img_kw: Dict[str, Any] = {
                                    "label": None,
                                    "type": "pil",
                                    "value": None,
                                    "visible": True,
                                    "elem_id": "tutor-image-upload",
                                    "elem_classes": ["composer-image-attach"],
                                    "height": 108,
                                    "width": "100%",
                                    "min_width": 100,
                                    "container": False,
                                    "show_label": False,
                                    "interactive": True,
                                }
                                try:
                                    _img_kw["placeholder"] = (
                                        "Upload or paste a screenshot (optional — ignore purple preview if text-only)"
                                    )
                                    _img_kw["sources"] = ["upload", "clipboard"]
                                    tutor_image = gr.Image(**_img_kw)
                                except TypeError:
                                    tutor_image = gr.Image(
                                        type="pil",
                                        value=None,
                                        visible=True,
                                        elem_id="tutor-image-upload",
                                        elem_classes=["composer-image-attach"],
                                        height=108,
                                        width="100%",
                                        min_width=100,
                                        container=False,
                                        show_label=False,
                                    )
                                with gr.Column(elem_classes=["composer-inner-text"]):
                                    user = gr.Textbox(
                                        placeholder="Message…",
                                        show_label=False,
                                        container=False,
                                        lines=14,
                                        max_lines=24,
                                        elem_id="imagination-chat-input",
                                        elem_classes=["composer-input"],
                                        html_attributes=gr.components.textbox.InputHTMLAttributes(
                                            enterkeyhint="send",
                                        ),
                                    )
                            with gr.Row(elem_classes=["terminal-actions"]):
                                send = gr.Button(
                                    "Send",
                                    variant="primary",
                                    elem_classes=["primary"],
                                    scale=1,
                                    elem_id="imagination-chat-send",
                                )
                                clear_btn = gr.Button("Clear", elem_classes=["secondary"], scale=0)
                        with gr.Column(elem_classes=["terminal-col"]):
                            gr.HTML('<div class="terminal-field-label">Output</div>')
                            with gr.Column(elem_classes=["terminal-panel", "chat-wrap"]):
                                try:
                                    chat = gr.Chatbot(
                                        elem_id="chatbox",
                                        height=480,
                                        label=None,
                                        show_label=False,
                                        show_copy_button=True,
                                        layout="bubble",
                                        format="messages",
                                        render_markdown=True,
                                    )
                                except TypeError:
                                    chat = gr.Chatbot(
                                        elem_id="chatbox",
                                        height=480,
                                        label=None,
                                        show_label=False,
                                        show_copy_button=True,
                                        layout="bubble",
                                        type="messages",
                                        render_markdown=True,
                                    )

            with gr.Column(
                elem_id="load-modal-root",
                elem_classes=["load-modal-backdrop"],
                visible=False,
            ) as load_modal:
                with gr.Column(elem_classes=["load-modal-card"]):
                    modal_title = gr.Markdown("")
                    modal_body = gr.Markdown("")
                    modal_dismiss = gr.Button("Dismiss", elem_id="modal-dismiss-btn", variant="secondary", visible=False)

            conversation_state = gr.State([])
            display_state = gr.State([])

            _thread_toolbar_out = [
                thread_dropdown,
                active_thread_id,
                thread_kind_state,
                conversation_state,
                display_state,
                chat,
                tutor_image,
                tutor_hint_md,
            ]
            btn_thread_new_gen.click(
                lambda s: thread_ui_new("general", s),
                inputs=[user_session],
                outputs=_thread_toolbar_out,
            )
            btn_thread_new_tut.click(
                lambda s: thread_ui_new("tutor", s),
                inputs=[user_session],
                outputs=_thread_toolbar_out,
            )
            thread_dropdown.change(
                thread_ui_select_with_visibility,
                inputs=[thread_dropdown, user_session],
                outputs=[
                    conversation_state,
                    display_state,
                    chat,
                    active_thread_id,
                    thread_kind_state,
                    tutor_image,
                    tutor_hint_md,
                ],
            )
            btn_thread_del.click(
                thread_ui_delete,
                inputs=[thread_dropdown, user_session],
                outputs=_thread_toolbar_out,
            )
            btn_thread_rename.click(
                thread_ui_rename,
                inputs=[thread_rename_box, thread_dropdown, user_session],
                outputs=[thread_dropdown, thread_rename_box],
            )

            le = lp = ld = None
            b_in = b_up = b_out = None

            with gr.Column(elem_classes=["account-dock"]):
                with gr.Accordion("Account · privacy & saved memory", open=False):
                    if auth_http_available:
                        _lines = [
                            "**Sign-in:** use the forms below (secure cookie after `python app.py`). "
                            "Fill **learner profile** fields so replies match your level and interests.",
                        ]
                        if _google_on:
                            _lines.append("[Continue with Google](/auth/google/login)")
                        _lines.append("[Log out](/auth/logout)")
                        gr.Markdown(" ".join(_lines))
                        gr.HTML(
                            """
<form method="post" action="/auth/email/login" style="display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end;margin:10px 0;">
  <input name="email" type="email" placeholder="Email" required style="flex:1;min-width:180px;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,0.12);background:#0a0c12;color:#eef0ff"/>
  <input name="password" type="password" placeholder="Password" required style="flex:1;min-width:140px;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,0.12);background:#0a0c12;color:#eef0ff"/>
  <button type="submit" style="padding:10px 18px;border-radius:12px;background:rgba(124,142,240,0.4);border:1px solid rgba(124,142,240,0.5);color:#fff;font-weight:600;cursor:pointer">Sign in</button>
</form>
<form method="post" action="/auth/email/signup" style="display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end;margin:10px 0;">
  <input name="display_name" type="text" placeholder="Display name (optional)" style="flex:1;min-width:140px;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,0.12);background:#0a0c12;color:#eef0ff"/>
  <input name="email" type="email" placeholder="Email" required style="flex:1;min-width:180px;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,0.12);background:#0a0c12;color:#eef0ff"/>
  <input name="password" type="password" placeholder="Password (6+ chars)" required style="flex:1;min-width:140px;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,0.12);background:#0a0c12;color:#eef0ff"/>
  <button type="submit" style="padding:10px 18px;border-radius:12px;background:transparent;border:1px solid rgba(255,255,255,0.2);color:#e8ecff;font-weight:600;cursor:pointer">Create account</button>
</form>
                            """
                        )
                    else:
                        gr.Markdown(
                            "*Without the FastAPI app (`python app.py`), use the buttons here — "
                            "refreshing the page clears the session. Chats are still isolated per browser tab until then.*"
                        )
                        le = gr.Textbox(label="Email", lines=1)
                        lp = gr.Textbox(label="Password", type="password", lines=1)
                        ld = gr.Textbox(label="Display name (signup only)", lines=1)
                        with gr.Row():
                            b_in = gr.Button("Sign in")
                            b_up = gr.Button("Create account")
                            b_out = gr.Button("Sign out")

                    user_notes = gr.Textbox(
                        label="Long-term notes (free-form — only for your account)",
                        lines=4,
                        placeholder="e.g. Prefer short answers; call me by first name; I'm working on a science fair project…",
                    )
                    profile_about = gr.Textbox(
                        label="About you (background, age band, what you do)",
                        lines=3,
                        placeholder="e.g. 9th grade student; builds robots after school; prefers hands-on examples",
                    )
                    profile_goals = gr.Textbox(
                        label="Interests & goals (likes, dislikes, what you're trying to learn)",
                        lines=3,
                        placeholder="e.g. Loves astronomy; nervous about public speaking; wants to understand how LLMs work",
                    )
                    profile_skills = gr.Textbox(
                        label="Skills & familiarity (topic → level, one per line)",
                        lines=4,
                        placeholder="algebra: strong\nPython: beginner\nmusic theory: intermediate\n…",
                    )
                    notes_status = gr.Markdown("")
                    save_notes_btn = gr.Button("Save notes & learner profile")

            if not auth_http_available and b_in is not None and b_up is not None and b_out is not None:
                _acct_out = [
                    user_session,
                    chat,
                    conversation_state,
                    display_state,
                    auth_banner,
                    login_hint,
                    user_notes,
                    profile_about,
                    profile_goals,
                    profile_skills,
                    thread_dropdown,
                    active_thread_id,
                    thread_kind_state,
                    tutor_image,
                    tutor_hint_md,
                ]
                b_in.click(
                    standalone_email_login,
                    inputs=[le, lp],
                    outputs=_acct_out,
                )
                b_up.click(
                    standalone_email_signup,
                    inputs=[le, lp, ld],
                    outputs=_acct_out,
                )
                b_out.click(
                    standalone_logout,
                    outputs=_acct_out,
                )

            if _hf_oauth:
                gr.LoginButton("Sign in with Hugging Face")

            def chat_inputs():
                return [
                    model_version,
                    chat_mode,
                    user,
                    conversation_state,
                    display_state,
                    user_session,
                    active_thread_id,
                    thread_kind_state,
                    tutor_image,
                ]

            save_notes_btn.click(
                save_user_notes_ui,
                inputs=[user_notes, profile_about, profile_goals, profile_skills, user_session],
                outputs=[notes_status],
            )

            modal_targets = [
                load_modal,
                user,
                send,
                clear_btn,
                chat_mode,
                modal_title,
                modal_body,
                modal_dismiss,
            ]

            chat_mode.change(
                fn=iter_preload_on_mode_change,
                inputs=[chat_mode],
                outputs=modal_targets,
            )

            modal_dismiss.click(fn=dismiss_load_modal, inputs=None, outputs=modal_targets)

            _chat_out = [
                chat,
                conversation_state,
                display_state,
                status,
                progress_html,
                auth_banner,
                thread_dropdown,
            ]

            send_evt = send.click(
                fn=chat_submit,
                inputs=chat_inputs(),
                outputs=_chat_out,
                concurrency_limit=1,
                concurrency_id="chat_gpu",
            )
            send_evt.then(lambda: "", None, user)

            user.submit(
                fn=chat_submit,
                inputs=chat_inputs(),
                outputs=_chat_out,
                concurrency_limit=1,
                concurrency_id="chat_gpu",
            ).then(lambda: "", None, user)

            clear_btn.click(
                fn=clear_all,
                inputs=[user_session, active_thread_id],
                outputs=[
                    chat,
                    conversation_state,
                    display_state,
                    status,
                    progress_html,
                    auth_banner,
                    *modal_targets,
                ],
            )

        demo.queue(default_concurrency_limit=1)
        demo.load(
            fn=hydrate_ui_from_session_json_with_threads,
            inputs=[_session_json_bridge],
            outputs=[
                user_session,
                chat,
                conversation_state,
                display_state,
                auth_banner,
                user_notes,
                profile_about,
                profile_goals,
                profile_skills,
                thread_dropdown,
                active_thread_id,
                thread_kind_state,
                tutor_image,
                tutor_hint_md,
            ],
            js="""
            async () => {
                const q = window.location.search || '';
                try {
                    const r = await fetch('/auth/me' + q, { credentials: 'same-origin' });
                    const j = await r.json();
                    return JSON.stringify(j);
                } catch (e) {
                    return JSON.stringify({ logged_in: false, banner_extra: '' });
                }
            }
            """,
        )
        demo.load(
            js="""
            () => {
                const tipMode = "Chat mode: main auto-routing, web search, or Coder";
                const tipModel = "Base model version for this session";
                const mark = () => {
                    for (const id of ["chat-mode-dropdown", "model-version-dropdown"]) {
                        const root = document.getElementById(id);
                        if (!root) continue;
                        const t = root.querySelector("button, [role='combobox'], select");
                        if (t && !t.dataset.imTip) {
                            t.title = id === "model-version-dropdown" ? tipModel : tipMode;
                            t.dataset.imTip = "1";
                        }
                    }
                };
                if (!window.__imaginationEnterSend) {
                    window.__imaginationEnterSend = true;
                    document.addEventListener(
                        "keydown",
                        (e) => {
                            const wrap = document.getElementById("imagination-chat-input");
                            if (!wrap || !wrap.contains(e.target)) return;
                            if (!(e.target instanceof HTMLTextAreaElement)) return;
                            if (e.key !== "Enter" || e.shiftKey) return;
                            if (e.isComposing) return;
                            e.preventDefault();
                            e.stopImmediatePropagation();
                            const sendWrap = document.getElementById("imagination-chat-send");
                            const btn = sendWrap && sendWrap.querySelector("button");
                            if (btn && !btn.disabled) btn.click();
                        },
                        true,
                    );
                }
                mark();
                const obs = new MutationObserver(mark);
                obs.observe(document.body, { childList: true, subtree: true });
            }
            """,
        )
        if getattr(gr, "Timer", None) is not None:
            _unload_toast_timer = gr.Timer(1.0)
            _unload_toast_timer.tick(
                fn=poll_unload_toast_ui,
                inputs=[toast_state],
                outputs=[toast_state, toast_stack],
            )
        else:
            demo.load(
                fn=poll_unload_toast_ui,
                inputs=[toast_state],
                outputs=[toast_state, toast_stack],
                every=1.0,
            )
    return demo


if __name__ == "__main__":
    try:
        from imagination_runtime.asgi_app import build_full_app
        import uvicorn

        demo = build_ui(auth_http_available=True)
        app = build_full_app(demo)
        uvicorn.run(app, host="0.0.0.0", port=7860)
    except ImportError:
        demo = build_ui(auth_http_available=False)
        demo.launch(share=True, server_port=7860)
