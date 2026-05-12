from __future__ import annotations

import ast
import base64
import difflib
import hashlib
import json
import os
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from imagination_runtime.chat_web_research import fetch_web_context
from imagination_runtime.paths import resolve_root_path
from imagination_runtime.ui_artisan_protocol import new_code_revision_id

JsonDict = Dict[str, Any]
ModelGenerateFn = Callable[[List[Dict[str, str]], int], str]


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def ui_artisan_mode_from_env() -> bool:
    """UI-only agent tools + visual-review protocol (also set per-request via AgentChatRequest)."""
    return _env_truthy("IMAGINATION_UI_ARTISAN_MODE")


def ui_loop_file_capture_from_env() -> bool:
    """When True, ``capture_ui`` may use file:/// URLs pointing at files under the session workspace."""
    return _env_truthy("IMAGINATION_UI_LOOP")


_SESSIONS_LOCK = Lock()
_SESSIONS: Dict[str, "AgentSessionState"] = {}


_FULL_TOOL_NAMES = frozenset({"read_file", "write_file", "run_shell", "web_search", "capture_ui"})
_UI_ARTISAN_TOOL_NAMES = frozenset({"read_file", "write_file", "capture_ui"})

_DEFAULT_SYSTEM_PROMPT = """You are Imagination Composer, an autonomous coding agent.

Each assistant turn must be ONE JSON object only (no markdown fences, no prose before or after the object).
Put brief plans in the "thought" string, but every turn must ALSO include exactly one of:
- "tool_call": { "name": "<tool>", "args": { ... } }  — to act, OR
- "final_answer": "<string>" — to finish the task for this session.

Allowed tool_call.name values ONLY (copy exactly, lowercase):
read_file | write_file | run_shell | web_search | capture_ui

There are NO other tools (no set_score, update_snake, game APIs, etc.). Implement games and logic with
write_file and/or run_shell on files in the workspace.

Schema for write_file (content is REQUIRED — the full file text, not just a path):
{"thought":"...","tool_call":{"name":"write_file","args":{"path":"snake_game.py","content":"#!/usr/bin/env python3\\n...","reason":"create game"}}}

Schema for read_file:
{"thought":"...","tool_call":{"name":"read_file","args":{"path":"snake_game.py"}}}

Schema for run_shell — use EITHER args.command as ONE shell string OR args.argv as a JSON array of tokens (preferred for python -c):
{"thought":"...","tool_call":{"name":"run_shell","args":{"argv":["python3","-c","print(42)"]}}}
{"thought":"...","tool_call":{"name":"run_shell","args":{"command":"python3 -c \"print(42)\""}}}
Do not pass argv elements joined with commas inside a single string (that breaks parsing).

When done:
{"thought":"...","final_answer":"what was done and what to do next","summary_report":{"files_modified":[{"path":"...","why":"..."}],"commands_run":[],"captures":[]}}

final_answer must summarize ONLY real actions from this session (files written, commands run, tools used).
Never describe game scores, movement, or UI behavior that you did not actually implement in file contents yet.

If the model uses chain-of-thought, end the message with the JSON object as the last characters.
"""


_UI_ARTISAN_SYSTEM_PROMPT = """You are Imagination UI Artisan — a specialist for graphical UI (HTML/CSS first).

Each assistant turn must be ONE JSON object only (no markdown fences, no prose before or after).
Put brief plans in "thought". Include exactly one of:
- "tool_call": { "name": "<tool>", "args": { ... } }  — to act, OR
- "final_answer": "<string>" — to finish, OR
- "final_answer" as an object when you need to attach a structured visual self-review (preferred for UI turns).

Allowed tool_call.name values ONLY (copy exactly, lowercase):
read_file | write_file | capture_ui

There are NO run_shell or web_search tools in this mode. Implement previews with static HTML/CSS files
under the workspace (e.g. index.html + styles.css) and open them with capture_ui.

When you finish a UI iteration, use final_answer as an OBJECT with keys:
- "text": short user-facing summary
- "visual_report": REQUIRED object following schema_version "1.0" with viewport {width_px,height_px},
  regions[{id,label,bbox_norm{x,y,w,h}}], summary, alignment_notes[], spacing_notes[], a11y_flags[],
  expected_after_next_edit (one sentence of what the user should see after your last edit).

After every write_file that changes HTML/CSS, you MUST call capture_ui before final_answer when possible.
For local files use file:/// URLs (three slashes) pointing at html under the workspace when the environment allows it.

Schema for write_file:
{"thought":"...","tool_call":{"name":"write_file","args":{"path":"index.html","content":"<!DOCTYPE html>...","reason":"create UI"}}}

Schema for capture_ui on a local preview file:
{"thought":"...","tool_call":{"name":"capture_ui","args":{"url":"file:///ABS_PATH_UNDER_WORKSPACE/index.html","width":1280,"height":720,"wait_ms":800}}}

When done (with visual_report):
{"thought":"...","final_answer":{"text":"...","visual_report":{"schema_version":"1.0","viewport":{"width_px":1280,"height_px":720},"summary":"...","alignment_notes":[],"spacing_notes":[],"a11y_flags":[],"regions":[{"id":"hero","label":"Hero","bbox_norm":{"x":0.2,"y":0.1,"w":0.6,"h":0.2},"role":"section"}],"expected_after_next_edit":"..."},"summary_report":{"files_modified":[{"path":"index.html","why":"..."}],"commands_run":[],"captures":[{"artifact_id":"...","kind":"screenshot"}]}}}
"""


def _normalize_tool_name(raw: str) -> str:
    """Map common model variants (runShell, run Shell) to canonical snake_case tool ids."""
    s = (raw or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _coerce_shell_argv(args: Dict[str, Any]) -> Tuple[Optional[List[str]], str, Optional[str]]:
    """Build argv for subprocess.run(..., shell=False). Returns (argv, display, error)."""
    posix = os.name != "nt"

    if "argv" in args and isinstance(args["argv"], (list, tuple)):
        argv = [str(x) for x in args["argv"] if str(x).strip()]
        if not argv:
            return None, "", "run_shell args.argv must be a non-empty JSON array of strings"
        display = " ".join(shlex.quote(a) for a in argv) if posix else subprocess.list2cmdline(argv)
        return argv, display, None

    raw_cmd = args.get("command") or args.get("cmd")
    if isinstance(raw_cmd, dict):
        return None, "", "run_shell args.command must be a string or JSON array, not an object"
    if isinstance(raw_cmd, (list, tuple)):
        argv = [str(x) for x in raw_cmd if str(x).strip()]
        if not argv:
            return None, "", "run_shell args.command as array must be non-empty"
        display = " ".join(shlex.quote(a) for a in argv) if posix else subprocess.list2cmdline(argv)
        return argv, display, None

    command = str(raw_cmd or "").strip()
    if not command:
        return None, "", "command is required (string or argv array)"

    if command.startswith("[") and command.rstrip().endswith("]"):
        try:
            loaded = ast.literal_eval(command)
        except (SyntaxError, ValueError, TypeError):
            loaded = None
        if isinstance(loaded, (list, tuple)):
            return _coerce_shell_argv({"argv": [str(x) for x in loaded]})

    try:
        argv = shlex.split(command, posix=posix)
    except ValueError:
        argv = [command]
    if not argv:
        return None, "", "command is empty after parsing"
    return argv, command, None


@dataclass
class DiffProposal:
    proposal_id: str
    rel_path: str
    old_content: str
    new_content: str
    reason: str = ""


@dataclass
class CaptureArtifact:
    artifact_id: str
    kind: str
    abs_path: str
    mime_type: str
    rel_path: str


@dataclass
class AgentSessionState:
    session_id: str
    workspace_root: str
    proposals: Dict[str, DiffProposal] = field(default_factory=dict)
    artifacts: Dict[str, CaptureArtifact] = field(default_factory=dict)
    modified_files: Dict[str, str] = field(default_factory=dict)
    commands_run: List[str] = field(default_factory=list)
    code_revision_seq: int = 0


def _normalize_session_id(raw: Optional[str]) -> str:
    base = (raw or "").strip()
    if base:
        return re.sub(r"[^a-zA-Z0-9._-]", "_", base)[:80]
    return f"session_{uuid.uuid4().hex[:10]}"


def _default_workspace_for_session(session_id: str) -> str:
    root = resolve_root_path(None)
    base = Path(root) / "temp" / "agent_workspaces" / session_id
    base.mkdir(parents=True, exist_ok=True)
    return str(base.resolve())


def get_or_create_session(session_id: Optional[str], workspace_root: Optional[str]) -> AgentSessionState:
    sid = _normalize_session_id(session_id)
    with _SESSIONS_LOCK:
        state = _SESSIONS.get(sid)
        if state is not None:
            return state

        if workspace_root:
            ws = str(Path(workspace_root).expanduser().resolve())
            Path(ws).mkdir(parents=True, exist_ok=True)
        else:
            ws = _default_workspace_for_session(sid)

        state = AgentSessionState(session_id=sid, workspace_root=ws)
        _SESSIONS[sid] = state
        return state


def workspace_tree(workspace_root: str, *, max_depth: int = 5, max_entries: int = 1000) -> JsonDict:
    root = Path(workspace_root).resolve()
    entries = 0

    def build(node: Path, depth: int) -> Optional[JsonDict]:
        nonlocal entries
        if entries >= max_entries:
            return None
        if depth > max_depth:
            return {"name": node.name, "path": str(node.relative_to(root)) if node != root else "", "type": "dir", "truncated": True, "children": []}

        entries += 1
        rel = "" if node == root else str(node.relative_to(root)).replace("\\", "/")
        if node.is_file():
            return {"name": node.name, "path": rel, "type": "file", "size": node.stat().st_size}

        children: List[JsonDict] = []
        try:
            for child in sorted(node.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if child.name.startswith(".git"):
                    continue
                built = build(child, depth + 1)
                if built is not None:
                    children.append(built)
                if entries >= max_entries:
                    break
        except PermissionError:
            pass

        return {"name": node.name or ".", "path": rel, "type": "dir", "children": children}

    tree = build(root, 0)
    return {"root": str(root), "tree": tree, "truncated": entries >= max_entries}


def list_workspace_tree(session_id: Optional[str], workspace_root: Optional[str]) -> JsonDict:
    state = get_or_create_session(session_id, workspace_root)
    payload = workspace_tree(state.workspace_root)
    payload["session_id"] = state.session_id
    return payload


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text_safe(path: Path, *, max_bytes: int = 300_000) -> Tuple[str, bool]:
    if not path.exists():
        return "", False
    raw = path.read_bytes()
    clipped = len(raw) > max_bytes
    if clipped:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="replace"), clipped


def _normalize_jsonish_quotes(t: str) -> str:
    """Replace curly/smart quotes that break strict JSON.parse."""
    return (
        (t or "")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u00ab", '"')
        .replace("\u00bb", '"')
    )


def _balanced_object_from(t: str, start: int) -> Optional[str]:
    """Return substring t[start:end+1] for first balanced `{` … `}` brace run from ``start``."""
    if start < 0 or start >= len(t) or t[start] != "{":
        return None
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
            if depth < 0:
                return None
    return None


def _last_balanced_json_slice(t: str) -> Optional[str]:
    """
    Many models emit prose then a JSON object at the end. Try balanced slices from each ``{``
    position (right-to-left) until ``json.loads`` succeeds.
    """
    if not t:
        return None
    pos = len(t)
    while True:
        start = t.rfind("{", 0, pos)
        if start < 0:
            return None
        blob = _balanced_object_from(t, start)
        if blob and len(blob) > 2:
            for cand in (blob, _normalize_jsonish_quotes(blob)):
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        return cand
                except Exception:
                    pass
        pos = start - 1
        if pos < 0:
            break
    return None


def _try_ast_literal_dict(t: str) -> Optional[JsonDict]:
    """Some checkpoints emit Python dict syntax (single quotes). ``ast.literal_eval`` is safe here."""
    blob = (t or "").strip()
    if not (blob.startswith("{") and blob.endswith("}")):
        return None
    if blob.count("'") < 2:
        return None
    try:
        v = ast.literal_eval(blob)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _strip_composer_preamble(text: str) -> str:
    """Remove common reasoning wrappers so a trailing JSON object can be parsed."""
    t = (text or "").strip()
    t = re.sub(r"<thinking>.*?</thinking>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE)
    # Some checkpoints emit  /  blocks (avoid angle brackets in source so editors don't strip).
    _ot, _ct = "\u003cthink\u003e", "\u003c/think\u003e"
    t = re.sub(_ot + r".*?" + _ct, "", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<reasoning>.*?</reasoning>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"(?is)^\s*thinking\s*[:.]?\s*\n+", "", t)
    t = re.sub(r"(?is)^\s*reasoning\s*[:.]?\s*\n+", "", t)
    return t.strip()


def _collect_json_dict_candidates(t: str) -> List[JsonDict]:
    """Parse every top-level JSON object in the string (handles prose + trailing JSON)."""
    out: List[JsonDict] = []
    dec = json.JSONDecoder()
    i = 0
    n = len(t)
    while i < n:
        while i < n and t[i] != "{":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(t, i)
            if isinstance(obj, dict):
                out.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    return out


def _pick_agent_turn_dict(candidates: List[JsonDict]) -> Optional[JsonDict]:
    if not candidates:
        return None
    for d in reversed(candidates):
        fa = str(d.get("final_answer") or "").strip()
        if fa and len(fa) > 8:
            return d
    for d in reversed(candidates):
        tc = d.get("tool_call")
        if isinstance(tc, dict) and str(tc.get("name") or "").strip():
            return d
    return candidates[-1]


def _extract_json_object(text: str) -> Optional[JsonDict]:
    t = _strip_composer_preamble(text)
    if not t:
        return None
    if t.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
        if m:
            t = m.group(1).strip()

    t_norm = _normalize_jsonish_quotes(t)
    for variant in (t, t_norm):
        try:
            loaded = json.loads(variant)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass

    for variant in (t, t_norm):
        candidates = _collect_json_dict_candidates(variant)
        picked = _pick_agent_turn_dict(candidates)
        if picked is not None:
            return picked

    for variant in (t, t_norm):
        start = variant.find("{")
        end = variant.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                loaded = json.loads(variant[start : end + 1])
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                pass

    slice_s = _last_balanced_json_slice(t) or _last_balanced_json_slice(t_norm)
    if slice_s:
        try:
            loaded = json.loads(slice_s)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
        lit = _try_ast_literal_dict(slice_s)
        if lit is not None:
            return lit

    lit2 = _try_ast_literal_dict(t) or _try_ast_literal_dict(t_norm)
    if lit2 is not None:
        return lit2

    return None


def _safe_rel(workspace: Path, input_path: str) -> Tuple[Path, str]:
    rel = (input_path or "").strip().replace("\\", "/").lstrip("/")
    if not rel:
        raise ValueError("path is required")
    candidate = (workspace / rel).resolve()
    try:
        candidate.relative_to(workspace)
    except Exception as exc:
        raise ValueError("path escapes workspace") from exc
    rel_norm = str(candidate.relative_to(workspace)).replace("\\", "/")
    return candidate, rel_norm


def _diff_unified(old_text: str, new_text: str, rel_path: str) -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=3,
        )
    )


def apply_proposals(session_id: str, proposal_ids: List[str]) -> JsonDict:
    state = get_or_create_session(session_id, None)
    applied: List[JsonDict] = []
    skipped: List[str] = []
    ws = Path(state.workspace_root).resolve()
    for pid in proposal_ids:
        prop = state.proposals.get(pid)
        if prop is None:
            skipped.append(pid)
            continue
        abs_path, rel_path = _safe_rel(ws, prop.rel_path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(prop.new_content, encoding="utf-8")
        state.modified_files[rel_path] = prop.reason or "Applied diff proposal"
        applied.append({"proposal_id": pid, "path": rel_path})
        del state.proposals[pid]
    return {"session_id": state.session_id, "applied": applied, "skipped": skipped}


def capture_artifact(session_id: str, artifact_id: str) -> Optional[CaptureArtifact]:
    state = get_or_create_session(session_id, None)
    return state.artifacts.get(artifact_id)


class AgenticLoop:
    def __init__(
        self,
        *,
        model_generate: ModelGenerateFn,
        session_id: Optional[str] = None,
        workspace_root: Optional[str] = None,
        max_tool_iters: int = 28,
        max_new_tokens: int = 1024,
        confirm_apply: bool = False,
        allow_network_tools: bool = True,
        ui_artisan_mode: Optional[bool] = None,
    ) -> None:
        self.state = get_or_create_session(session_id, workspace_root)
        self.model_generate = model_generate
        self.max_tool_iters = max(1, min(64, int(max_tool_iters)))
        self.max_new_tokens = max(128, min(4096, int(max_new_tokens)))
        self.confirm_apply = bool(confirm_apply)
        self.allow_network_tools = bool(allow_network_tools)
        self.workspace = Path(self.state.workspace_root).resolve()
        if ui_artisan_mode is None:
            ui_artisan_mode = ui_artisan_mode_from_env()
        self.ui_artisan_mode = bool(ui_artisan_mode)
        self._allowed_tool_names = _UI_ARTISAN_TOOL_NAMES if self.ui_artisan_mode else _FULL_TOOL_NAMES

    def run(self, *, user_prompt: str, messages: Optional[List[Dict[str, str]]] = None) -> Iterator[JsonDict]:
        sys_prompt = _UI_ARTISAN_SYSTEM_PROMPT if self.ui_artisan_mode else _DEFAULT_SYSTEM_PROMPT
        convo: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
        if messages:
            prev_key: Optional[Tuple[str, str]] = None
            for m in messages:
                role = (m.get("role") or "").strip().lower()
                if role not in ("user", "assistant"):
                    continue
                content = str(m.get("content") or "")
                key = (role, content)
                if key == prev_key:
                    continue
                prev_key = key
                convo.append({"role": role, "content": content})
        elif user_prompt.strip():
            convo.append({"role": "user", "content": user_prompt.strip()})
        else:
            convo.append({"role": "user", "content": "Inspect workspace and propose first coding step."})

        yield {"type": "session", "session_id": self.state.session_id, "workspace_root": self.state.workspace_root}
        yield {"type": "workspace_tree", **workspace_tree(self.state.workspace_root)}

        final_answer = ""
        fail_streak = 0
        _allowed = ", ".join(sorted(self._allowed_tool_names))
        _break_after_failures = 8

        for _ in range(self.max_tool_iters):
            raw = self.model_generate(convo, self.max_new_tokens)
            parsed = _extract_json_object(raw)
            if not parsed:
                fail_streak += 1
                err = "Model did not return valid JSON object"
                yield {"type": "error", "message": err}
                convo.append({"role": "assistant", "content": raw})
                convo.append(
                    {
                        "role": "system",
                        "content": (
                            "Reply with ONE JSON object only (no markdown fences, no prose after the object). "
                            "Include tool_call OR final_answer in the same object as thought. "
                            'Minimal valid example: {"thought":"create file","tool_call":'
                            '{"name":"write_file","args":{"path":"hello.py","content":"print(1)\\n","reason":"demo"}}}'
                        ),
                    }
                )
                if fail_streak >= _break_after_failures:
                    final_answer = (
                        "Stopped after repeated non-JSON replies. Try a smaller step, disable long chain-of-thought "
                        "if your stack allows it, or use a checkpoint that follows strict JSON instructions."
                    )
                    break
                continue

            thought = str(parsed.get("thought") or "").strip()
            if thought:
                yield {"type": "thought", "text": thought}

            fa_raw = parsed.get("final_answer")
            if fa_raw is not None and fa_raw != "":
                if isinstance(fa_raw, dict):
                    final_answer = json.dumps(fa_raw, ensure_ascii=False)
                    if self.ui_artisan_mode:
                        yield {"type": "ui_artisan_final", "final_answer": fa_raw}
                else:
                    final_answer = str(fa_raw).strip()
                break

            tool_call = parsed.get("tool_call")
            if not isinstance(tool_call, dict):
                fail_streak += 1
                yield {"type": "error", "message": "Missing tool_call or final_answer"}
                convo.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
                convo.append(
                    {
                        "role": "system",
                        "content": (
                            "The JSON must include either final_answer (string or, in UI Artisan mode, object "
                            "with text + visual_report) or tool_call "
                            f'(object with name in [{_allowed}] and args). "thought" alone is not enough.'
                        ),
                    }
                )
                if fail_streak >= _break_after_failures:
                    final_answer = (
                        "Stopped: the model kept returning JSON without tool_call or final_answer. "
                        "Try rephrasing the task or lowering complexity."
                    )
                    break
                continue

            tool_name = _normalize_tool_name(str(tool_call.get("name") or ""))
            args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}

            if not tool_name:
                fail_streak += 1
                yield {"type": "error", "message": "tool_call.name is empty"}
                convo.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
                convo.append(
                    {
                        "role": "system",
                        "content": f"tool_call.name is required. Use one of: {_allowed}.",
                    }
                )
                if fail_streak >= _break_after_failures:
                    final_answer = "Stopped after repeated invalid tool_call.name values."
                    break
                continue

            if tool_name not in self._allowed_tool_names:
                fail_streak += 1
                msg = f"Invalid tool {tool_name!r}; allowed: {_allowed}"
                yield {"type": "error", "message": msg}
                convo.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
                convo.append(
                    {
                        "role": "system",
                        "content": (
                            f"Tool {tool_name!r} does not exist. Use only: {_allowed}. "
                            + (
                                "Use write_file and capture_ui on workspace files."
                                if self.ui_artisan_mode
                                else "Implement behavior with write_file / run_shell on workspace files."
                            )
                        ),
                    }
                )
                if fail_streak >= _break_after_failures:
                    final_answer = "Stopped after repeated invalid tool names."
                    break
                continue

            fail_streak = 0
            call_id = uuid.uuid4().hex[:10]
            yield {"type": "tool_call", "id": call_id, "name": tool_name, "args": args}

            ok, result = self._execute_tool(tool_name, args)
            yield {"type": "tool_result", "id": call_id, "name": tool_name, "ok": ok, "data": result}

            if self.ui_artisan_mode and tool_name == "write_file" and ok and isinstance(result, dict):
                crid = result.get("code_revision_id")
                rel_p = result.get("path") or ""
                if crid:
                    hint = (
                        f"UI Artisan: revision {crid} for {rel_p}. "
                        "Call capture_ui with a file:/// URL under the workspace (requires IMAGINATION_UI_LOOP=1) "
                        "or serve the page over http(s) if available."
                    )
                    yield {
                        "type": "ui_artisan_revision",
                        "code_revision_id": crid,
                        "path": rel_p,
                        "screenshot_artifact_id": None,
                        "mandatory_capture_hint": hint,
                    }

            convo.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            convo.append(
                {
                    "role": "system",
                    "content": f"Tool result for {tool_name}: {json.dumps(result, ensure_ascii=False)[:12000]}",
                }
            )

            if tool_name == "write_file" and not ok:
                convo.append(
                    {
                        "role": "system",
                        "content": (
                            "write_file was rejected: include the full file source in args.content "
                            r'(a string; use JSON escapes like \n for newlines). Then call write_file again.'
                        ),
                    }
                )

            if tool_name == "run_shell" and not ok:
                convo.append(
                    {
                        "role": "system",
                        "content": (
                            "The shell command failed. Diagnose stderr/stdout and propose a corrected command "
                            "or an alternate fix path."
                        ),
                    }
                )

        if not final_answer:
            final_answer = "Reached tool iteration budget; reporting current status and pending actions."

        report = self._build_summary_report()
        yield {"type": "summary", "report": report}
        yield {"type": "final", "text": final_answer}

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, JsonDict]:
        try:
            if tool_name == "read_file":
                return True, self._tool_read_file(args)
            if tool_name == "write_file":
                if "content" not in args:
                    return False, {
                        "error": (
                            'write_file requires args["content"] — the complete file source as a string. '
                            "You passed only a path."
                        ),
                    }
                body = str(args.get("content") or "")
                if not body.strip():
                    return False, {
                        "error": (
                            "write_file requires non-empty content (full file body). "
                            "Omitting or empty content creates useless empty files."
                        ),
                    }
                return True, self._tool_write_file(args)
            if tool_name == "run_shell":
                return self._tool_run_shell(args)
            if tool_name == "web_search":
                return True, self._tool_web_search(args)
            if tool_name == "capture_ui":
                return self._tool_capture_ui(args)
            return False, {"error": f"unknown_tool: {tool_name}"}
        except Exception as exc:
            return False, {"error": str(exc)}

    def _tool_read_file(self, args: Dict[str, Any]) -> JsonDict:
        path = str(args.get("path") or "")
        abs_path, rel = _safe_rel(self.workspace, path)
        text, clipped = _read_text_safe(abs_path)
        return {
            "path": rel,
            "exists": abs_path.exists(),
            "content": text,
            "truncated": clipped,
            "sha256": _sha256_text(text),
        }

    def _tool_write_file(self, args: Dict[str, Any]) -> JsonDict:
        path = str(args.get("path") or "")
        content = str(args.get("content") or "")
        reason = str(args.get("reason") or "").strip()
        abs_path, rel = _safe_rel(self.workspace, path)
        old_text, _ = _read_text_safe(abs_path, max_bytes=2_000_000)
        diff_text = _diff_unified(old_text, content, rel)
        proposal_id = uuid.uuid4().hex[:12]
        event: JsonDict = {
            "proposal_id": proposal_id,
            "path": rel,
            "diff": diff_text,
            "old_sha256": _sha256_text(old_text),
            "new_sha256": _sha256_text(content),
            "applied": False,
        }
        if self.confirm_apply:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
            self.state.modified_files[rel] = reason or "Agent write_file"
            event["applied"] = True
        else:
            self.state.proposals[proposal_id] = DiffProposal(
                proposal_id=proposal_id,
                rel_path=rel,
                old_content=old_text,
                new_content=content,
                reason=reason,
            )
        if self.ui_artisan_mode:
            self.state.code_revision_seq += 1
            event["code_revision_id"] = new_code_revision_id(
                self.state.session_id, self.state.code_revision_seq
            )
        return {"type": "diff_preview", **event}

    def _tool_run_shell(self, args: Dict[str, Any]) -> Tuple[bool, JsonDict]:
        allow_shell = (os.getenv("IMAGINATION_AGENT_ALLOW_SHELL") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if not allow_shell:
            return False, {"error": "run_shell disabled by IMAGINATION_AGENT_ALLOW_SHELL=0"}

        argv, command_display, argv_err = _coerce_shell_argv(args)
        if argv_err or not argv:
            return False, {"error": argv_err or "command is required"}

        cwd_raw = str(args.get("cwd") or "").strip()
        timeout_ms = int(args.get("timeout_ms") or 90_000)
        timeout_sec = max(1, min(600, timeout_ms // 1000))

        if cwd_raw:
            cwd_abs, _ = _safe_rel(self.workspace, cwd_raw)
            cwd = str(cwd_abs)
        else:
            cwd = str(self.workspace)

        self.state.commands_run.append(command_display)
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            shell=False,
        )
        stdout = (completed.stdout or "")[:10000]
        stderr = (completed.stderr or "")[:10000]
        ok = completed.returncode == 0
        return ok, {
            "command": command_display,
            "cwd": cwd,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "status": "success" if ok else "fail",
        }

    def _tool_web_search(self, args: Dict[str, Any]) -> JsonDict:
        if not self.allow_network_tools:
            return {"error": "network tools disabled for this request"}
        query = str(args.get("query") or "").strip()
        max_results = int(args.get("max_results") or 8)
        snippets = fetch_web_context(query, max_results=max(1, min(12, max_results)))
        return {"query": query, "results": snippets}

    def _tool_capture_ui(self, args: Dict[str, Any]) -> Tuple[bool, JsonDict]:
        capture_on = (os.getenv("IMAGINATION_AGENT_ENABLE_CAPTURE") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if not capture_on:
            return False, {"error": "capture_ui disabled by IMAGINATION_AGENT_ENABLE_CAPTURE=0"}

        url = str(args.get("url") or "").strip()
        ulow = url.lower()
        file_mode = ulow.startswith("file:")
        if file_mode and not ui_loop_file_capture_from_env():
            return False, {"error": "file:// capture requires IMAGINATION_UI_LOOP=1"}

        if not self.allow_network_tools and not file_mode:
            return False, {"error": "network tools disabled for this request"}

        goto_url = url
        if file_mode:
            from urllib.parse import unquote, urlparse

            parsed = urlparse(url)
            raw_path = unquote(parsed.path or "")
            if os.name == "nt" and len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
                raw_path = raw_path.lstrip("/")
            path_obj = Path(raw_path).resolve()
            try:
                path_obj.relative_to(self.workspace.resolve())
            except ValueError:
                return False, {"error": "file url must resolve under workspace"}
            if not path_obj.is_file():
                return False, {"error": f"file not found: {path_obj}"}
            goto_url = path_obj.as_uri()

        if not file_mode and not (url.startswith("http://") or url.startswith("https://")):
            return False, {"error": "url must start with http:// or https:// (or file:// with IMAGINATION_UI_LOOP=1)"}

        allow_hosts = (os.getenv("IMAGINATION_AGENT_CAPTURE_ALLOW_HOSTS") or "").strip()
        if allow_hosts and not file_mode:
            from urllib.parse import urlparse

            host = (urlparse(url).hostname or "").lower()
            allowed = {h.strip().lower() for h in allow_hosts.split(",") if h.strip()}
            if host not in allowed:
                return False, {"error": f"host not allowlisted: {host}"}

        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception as exc:
            return False, {"error": f"playwright import failed: {exc}"}

        out_dir = Path(self.state.workspace_root) / ".captures"
        out_dir.mkdir(parents=True, exist_ok=True)
        shot_id = uuid.uuid4().hex[:12]
        image_path = out_dir / f"{shot_id}.png"
        video_path: Optional[Path] = None

        want_video = bool(args.get("video"))
        wait_ms = int(args.get("wait_ms") or 1200)
        wait_ms = max(0, min(15_000, wait_ms))
        width = int(args.get("width") or 1400)
        height = int(args.get("height") or 900)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context_kwargs: Dict[str, Any] = {"viewport": {"width": width, "height": height}}
                if want_video:
                    context_kwargs["record_video_dir"] = str(out_dir)
                context = browser.new_context(**context_kwargs)
                page = context.new_page()
                page.goto(goto_url, wait_until="domcontentloaded", timeout=60_000)
                if wait_ms:
                    page.wait_for_timeout(wait_ms)
                page.screenshot(path=str(image_path), full_page=True)
                if want_video:
                    page.wait_for_timeout(1000)
                    video = page.video
                else:
                    video = None
                page.close()
                context.close()
                browser.close()
                if video is not None:
                    try:
                        video_path = Path(video.path())
                    except Exception:
                        video_path = None
        except Exception as exc:
            return False, {"error": f"capture failed: {exc}"}

        rel_img = str(image_path.relative_to(self.workspace)).replace("\\", "/")
        img_artifact = CaptureArtifact(
            artifact_id=shot_id,
            kind="screenshot",
            abs_path=str(image_path),
            mime_type="image/png",
            rel_path=rel_img,
        )
        self.state.artifacts[shot_id] = img_artifact

        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload: JsonDict = {
            "kind": "screenshot",
            "artifact_id": shot_id,
            "mime_type": "image/png",
            "relative_path": rel_img,
            "base64": image_b64,
        }
        if video_path and video_path.exists():
            vid_id = uuid.uuid4().hex[:12]
            rel_vid = str(video_path.relative_to(self.workspace)).replace("\\", "/") if self.workspace in video_path.parents else video_path.name
            self.state.artifacts[vid_id] = CaptureArtifact(
                artifact_id=vid_id,
                kind="video",
                abs_path=str(video_path),
                mime_type="video/webm",
                rel_path=rel_vid,
            )
            payload["video"] = {
                "artifact_id": vid_id,
                "mime_type": "video/webm",
                "relative_path": rel_vid,
            }
        return True, {"type": "media", **payload}

    def _build_summary_report(self) -> JsonDict:
        files_modified = [{"path": path, "why": why} for path, why in sorted(self.state.modified_files.items())]
        captures = [
            {"artifact_id": c.artifact_id, "kind": c.kind, "path": c.rel_path}
            for c in self.state.artifacts.values()
        ]
        pending = [
            {"proposal_id": p.proposal_id, "path": p.rel_path, "reason": p.reason}
            for p in self.state.proposals.values()
        ]
        return {
            "session_id": self.state.session_id,
            "workspace_root": self.state.workspace_root,
            "files_modified": files_modified,
            "commands_run": list(self.state.commands_run),
            "captures": captures,
            "pending_proposals": pending,
        }
