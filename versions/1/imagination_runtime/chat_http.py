"""
Shared HTTP routes for text/VLM chat: /generate, /api/chat, /health.

Used by ``colab_backend`` (FastAPI-only) and ``api_server`` (auth + API, no Gradio).
"""
from __future__ import annotations

import json
import os
import base64
from io import BytesIO
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from imagination_runtime.agent_loop import (
    AgenticLoop,
    apply_proposals,
    capture_artifact,
    list_workspace_tree,
)
from imagination_runtime.chat_web_research import (
    final_text_from_stream,
    iterate_display_text_with_web,
    web_followups_enabled,
)
from imagination_runtime.http_chat_system import enrich_messages_with_imagination_system
from imagination_runtime.think_tags import preserve_think_tags

_GEN_LOCK = Lock()


def _stream_heartbeat_seconds() -> float:
    """Seconds between NDJSON keepalive lines while waiting for the first model chunk.

    Prevents proxies (ngrok, cloud LBs) and ``curl -m`` from killing the connection during
    cold-start prefill on the VLM path, which can exceed 60-120s on Colab. Values <=0 disable
    heartbeats entirely. Env override: ``IMAGINATION_STREAM_HEARTBEAT_S`` (default 15s).
    """
    raw = (os.getenv("IMAGINATION_STREAM_HEARTBEAT_S") or "").strip()
    if not raw:
        return 15.0
    try:
        v = float(raw)
    except ValueError:
        return 15.0
    if v <= 0:
        return 0.0
    return max(1.0, min(v, 120.0))


def _iter_stream_with_heartbeat(
    stream_iter: Iterator[str],
    *,
    prelude: str,
    heartbeat_s: float,
) -> Iterator[bytes]:
    """Yield NDJSON bytes from a text iterator, injecting keepalive lines during long waits.

    ``stream_iter`` yields cumulative assistant text (not deltas), matching ``_stream_native`` /
    ``iterate_display_text_with_web``. While no new chunk arrives, the wrapper re-emits the
    current best text every ``heartbeat_s`` seconds so that TCP bytes keep flowing even before
    the model's first generated token. Exceptions from the producer are surfaced as
    ``{"error": "..."}`` lines so the client still sees a terminal event.
    """
    if heartbeat_s <= 0:
        try:
            for text in stream_iter:
                line = json.dumps({"text": preserve_think_tags(prelude + text)}, ensure_ascii=False) + "\n"
                yield line.encode("utf-8")
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
            yield err.encode("utf-8")
        return

    q: "Queue[Any]" = Queue()
    SENTINEL = object()
    err_holder: Dict[str, Any] = {"exc": None}

    def _pump() -> None:
        try:
            for text in stream_iter:
                q.put(("text", text))
        except Exception as e:
            err_holder["exc"] = e
        finally:
            q.put(SENTINEL)

    t = Thread(target=_pump, daemon=True)
    t.start()

    last_text = prelude
    while True:
        try:
            item = q.get(timeout=heartbeat_s)
        except Empty:
            # Heartbeat: re-emit current cumulative text so intermediary proxies see bytes.
            yield (json.dumps({"text": preserve_think_tags(last_text)}, ensure_ascii=False) + "\n").encode("utf-8")
            continue
        if item is SENTINEL:
            break
        kind, payload = item
        if kind == "text":
            last_text = prelude + payload
            yield (json.dumps({"text": preserve_think_tags(last_text)}, ensure_ascii=False) + "\n").encode("utf-8")

    if err_holder["exc"] is not None:
        err = json.dumps({"error": str(err_holder["exc"])}, ensure_ascii=False) + "\n"
        yield err.encode("utf-8")


def _web_followups_active() -> bool:
    """Web research loop is only enabled for text (non-VLM) stacks; VLM templates are unchanged."""
    if not web_followups_enabled():
        return False
    try:
        from imagination_v1_4 import RUNTIME

        if getattr(RUNTIME, "main_is_vlm", False):
            return False
    except Exception:
        return False
    return True


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: Optional[int] = None


class ChatApiRequest(BaseModel):
    """Payload from Next.js / v0 frontends (prompt + model + optional full history)."""

    model_config = ConfigDict(extra="ignore")

    prompt: str = ""
    currentModel: str = "imagination-1.3"
    messages: Optional[List[ChatMessage]] = None
    image: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    max_new_tokens: Optional[int] = None


def coerce_chat_api_request_dict(raw: Any) -> Dict[str, Any]:
    """
    Normalize arbitrary client/JSON into fields ``ChatApiRequest`` understands.
    Fills ``prompt`` / ``messages`` from common aliases and avoids empty bodies that
    break proxies or Pydantic when only legacy keys (``query``, ``message``) are set.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = dict(raw)
    p = (
        out.get("prompt")
        or out.get("message")
        or out.get("text")
        or out.get("query")
        or out.get("q")
        or ""
    )
    p = p.strip() if isinstance(p, str) else (str(p).strip() if p is not None else "")
    if p:
        out["prompt"] = p

    messages = out.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        if p:
            out["messages"] = [{"role": "user", "content": p}]
    if not p:
        if isinstance(messages, list):
            for m in reversed(messages):
                if not isinstance(m, dict):
                    continue
                if (str(m.get("role") or "").lower() != "user"):
                    continue
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    out["prompt"] = c.strip()
                    break
    return out


class AgentChatRequest(ChatApiRequest):
    session_id: Optional[str] = None
    workspace_root: Optional[str] = None
    max_tool_iters: int = 28
    confirm_apply: bool = False
    allow_network_tools: bool = True
    """When True, agent tools are read_file / write_file / capture_ui only and the UI Artisan system prompt applies."""
    ui_artisan_mode: bool = False


class AgentApplyRequest(BaseModel):
    session_id: str
    proposal_ids: List[str] = Field(default_factory=list)


def add_cors_from_env(app: FastAPI) -> None:
    _cors = (os.getenv("COLAB_BACKEND_CORS") or os.getenv("IMAGINATION_API_CORS") or "*").strip()
    if _cors == "*":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


def _messages_dicts(items: List[ChatMessage]) -> List[Dict[str, str]]:
    return [{"role": m.role.strip(), "content": (m.content or "").strip()} for m in items]


def _inject_clip_image_placeholder(
    messages: List[Dict[str, str]], image: Optional[Any], processor: Any
) -> List[Dict[str, str]]:
    """
    Ensure CLIP+projector prompts carry exactly one image placeholder token.

    The clip_projector path validates that the rendered prompt contains one image token id
    when ``image`` is present. API chat messages are plain text, so we inject the literal token
    onto the latest user turn if it is currently missing.
    """
    if image is None or not getattr(processor, "is_clip_projector", False):
        return messages
    meta = getattr(getattr(processor, "bundle", None), "meta", None) or {}
    token = str(meta.get("image_token_literal") or meta.get("image_token") or "").strip()
    if not token:
        return messages

    out = [dict(m) for m in messages]
    has_token = any(token in (m.get("content") or "") for m in out)
    if has_token:
        return out

    for i in range(len(out) - 1, -1, -1):
        if (out[i].get("role") or "").strip().lower() != "user":
            continue
        prior = (out[i].get("content") or "").strip()
        out[i]["content"] = f"{prior}\n\n{token}" if prior else token
        return out

    out.append({"role": "user", "content": token})
    return out


def _first_image_data_url_from_attachments(attachments: Optional[List[Dict[str, Any]]]) -> str:
    if not attachments:
        return ""
    for att in attachments:
        if not isinstance(att, dict):
            continue
        if str(att.get("type", "")).lower() != "image":
            continue
        url = att.get("url")
        if isinstance(url, str) and url.strip().lower().startswith("data:image/"):
            return url.strip()
    return ""


def _parse_image_data_url(image_data: str) -> Optional[Any]:
    """
    Parse a data URL (``data:image/...;base64,...``) into a PIL image.

    Returns ``None`` on parse/decode errors so text-only requests keep working.
    """
    if not image_data:
        return None
    s = image_data.strip()
    if not s.lower().startswith("data:image/"):
        return None
    comma = s.find(",")
    if comma <= 0:
        return None
    meta = s[:comma].lower()
    if "base64" not in meta:
        return None
    b64 = "".join(s[comma + 1 :].split())
    try:
        raw = base64.b64decode(b64, validate=False)
        from PIL import Image

        bio = BytesIO(raw)
        img = Image.open(bio)
        img.load()
        return img.convert("RGB")
    except Exception as e:
        print(f"[imagination] image decode failed ({type(e).__name__}): {e}", flush=True)
        return None


def _resolve_chat_pil_image(body: ChatApiRequest) -> tuple[Optional[Any], str]:
    """
    Prefer ``body.image``, else first ``attachments[].url`` that looks like a data URL.
    Returns ``(pil_or_none, raw_data_url_or_empty)``.
    """
    raw = (body.image or "").strip()
    if not raw:
        raw = _first_image_data_url_from_attachments(body.attachments)
    if not raw:
        return None, ""
    return _parse_image_data_url(raw), raw


def _stream_native(
    messages: List[Dict[str, str]], max_new_tokens: int, image: Optional[Any] = None
) -> Iterator[str]:
    """Yield cumulative decoded text (same as ``generate_stream_chat`` / VLM stream)."""
    from imagination_v1_4 import RUNTIME, _effective_main_max_new_tokens, generate_stream_chat
    from imagination_runtime.vlm_infer import generate_stream_vlm

    if RUNTIME.main_tokenizer is None or RUNTIME.main_model is None:
        raise RuntimeError("Main model not loaded")

    cap = _effective_main_max_new_tokens()
    want = min(max(32, int(max_new_tokens)), cap)

    if RUNTIME.main_is_vlm and RUNTIME.main_processor is not None:
        msgs_for_vlm = _inject_clip_image_placeholder(messages, image, RUNTIME.main_processor)
        yield from generate_stream_vlm(
            processor=RUNTIME.main_processor,
            tokenizer=RUNTIME.main_tokenizer,
            model=RUNTIME.main_model,
            messages=msgs_for_vlm,
            max_new_tokens=want,
            lock=_GEN_LOCK,
            image=image,
        )
        return

    yield from generate_stream_chat(
        tokenizer=RUNTIME.main_tokenizer,
        model=RUNTIME.main_model,
        messages=messages,
        max_new_tokens=want,
        lock=_GEN_LOCK,
        extra_generate_kwargs=None,
    )


def _generate_native(
    messages: List[Dict[str, str]], max_new_tokens: int, image: Optional[Any] = None
) -> str:
    out = ""
    for chunk in _stream_native(messages, max_new_tokens, image=image):
        out = chunk
    return preserve_think_tags(out)


def attach_generation_routes(app: FastAPI) -> None:
    """Register /generate, /api/chat, and a simple /health (in addition to /api/health from ASGI core)."""

    @app.get("/health")
    def health() -> Dict[str, Any]:
        from imagination_runtime.runtime_health import health_version_dict, vision_health_dict

        out: Dict[str, Any] = {"status": "ok", "model_name": "Imagination 1.3"}
        out.update(health_version_dict())
        out.update(vision_health_dict())
        return out

    @app.post("/generate")
    def generate_completion(body: GenerateRequest) -> Dict[str, Any]:
        from imagination_v1_4 import _effective_main_max_new_tokens

        raw = _messages_dicts(body.messages)
        if not any(m["content"] for m in raw):
            raise HTTPException(status_code=400, detail="Empty message content")
        msgs = enrich_messages_with_imagination_system(raw)
        default_max = _effective_main_max_new_tokens()
        max_nt = body.max_new_tokens if body.max_new_tokens is not None else default_max
        try:
            text = _generate_native(msgs, max_nt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {
            "response": text,
            "model": "Imagination 1.3",
            "max_new_tokens_effective": min(max_nt, default_max),
        }

    @app.post("/api/chat")
    def api_chat(body: ChatApiRequest) -> Dict[str, Any]:
        from imagination_v1_4 import _effective_main_max_new_tokens

        prompt = (body.prompt or "").strip()
        if body.messages:
            raw = _messages_dicts(body.messages)
            if not any(m["content"] for m in raw):
                raise HTTPException(status_code=400, detail="Empty message content")
            msgs = enrich_messages_with_imagination_system(raw)
        elif prompt:
            msgs = enrich_messages_with_imagination_system([{"role": "user", "content": prompt}])
        else:
            raise HTTPException(status_code=400, detail="Provide `prompt` and/or `messages`")
        default_max = _effective_main_max_new_tokens()
        max_nt = body.max_new_tokens if body.max_new_tokens is not None else default_max
        pil_image, img_raw = _resolve_chat_pil_image(body)
        if img_raw and pil_image is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not decode the image payload (WEBP/JPEG/PNG). "
                    "Try PNG or JPEG, or upgrade Pillow with WEBP support on the server."
                ),
            )
        try:
            if _web_followups_active():
                text = final_text_from_stream(
                    iterate_display_text_with_web(
                        base_messages=msgs,
                        max_new_tokens=max_nt,
                        stream_native=_stream_native,
                    )
                )
            else:
                text = _generate_native(msgs, max_nt, image=pil_image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {
            "response": text,
            "model": body.currentModel or "Imagination 1.3",
            "max_new_tokens_effective": min(max_nt, default_max),
        }

    @app.post("/api/chat/stream")
    def api_chat_stream(body: ChatApiRequest) -> StreamingResponse:
        """NDJSON stream: one JSON object per line: ``{\"text\": \"...\"}`` (cumulative assistant text)."""
        from imagination_v1_4 import _effective_main_max_new_tokens

        prompt = (body.prompt or "").strip()
        if body.messages:
            raw = _messages_dicts(body.messages)
            if not any(m["content"] for m in raw):
                raise HTTPException(status_code=400, detail="Empty message content")
            msgs = enrich_messages_with_imagination_system(raw)
        elif prompt:
            msgs = enrich_messages_with_imagination_system([{"role": "user", "content": prompt}])
        else:
            raise HTTPException(status_code=400, detail="Provide `prompt` and/or `messages`")
        default_max = _effective_main_max_new_tokens()
        max_nt = body.max_new_tokens if body.max_new_tokens is not None else default_max
        pil_image, img_raw = _resolve_chat_pil_image(body)

        def ndjson_chunks() -> Iterator[bytes]:
            if img_raw and pil_image is None:
                err = json.dumps(
                    {
                        "error": (
                            "Could not decode the image payload (WEBP/JPEG/PNG). "
                            "Try PNG or JPEG, or upgrade Pillow with WEBP support on the server."
                        )
                    },
                    ensure_ascii=False,
                ) + "\n"
                yield err.encode("utf-8")
                return

            # No status preamble: older builds sent _Generating…_ / vision here; the UI shows it
            # merged into the assistant bubble. Heartbeats + stream chunks carry content only.
            prelude = ""
            try:
                if _web_followups_active():
                    stream_iter = iterate_display_text_with_web(
                        base_messages=msgs,
                        max_new_tokens=max_nt,
                        stream_native=_stream_native,
                    )
                else:
                    stream_iter = _stream_native(msgs, max_nt, image=pil_image)

                yield from _iter_stream_with_heartbeat(
                    stream_iter,
                    prelude=prelude,
                    heartbeat_s=_stream_heartbeat_seconds(),
                )
            except Exception as e:
                err = json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
                yield err.encode("utf-8")

        return StreamingResponse(
            ndjson_chunks(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/chat/agent")
    def api_chat_agent(body: AgentChatRequest) -> StreamingResponse:
        from imagination_v1_4 import _effective_main_max_new_tokens

        prompt = (body.prompt or "").strip()
        if body.messages:
            raw = _messages_dicts(body.messages)
            if not any(m["content"] for m in raw):
                raise HTTPException(status_code=400, detail="Empty message content")
            # Do NOT use enrich_messages_with_imagination_system here: that prepends the long
            # chat-assistant system prompt, which fights the Composer JSON tool schema and causes
            # non-JSON replies → every loop iteration burns until max_tool_iters.
            msgs = [
                dict(m)
                for m in raw
                if (m.get("role") or "").strip().lower() in ("user", "assistant")
            ]
            if not msgs:
                raise HTTPException(status_code=400, detail="No user/assistant messages for agent")
        elif prompt:
            msgs = [{"role": "user", "content": prompt}]
        else:
            raise HTTPException(status_code=400, detail="Provide `prompt` and/or `messages`")
        default_max = _effective_main_max_new_tokens()
        max_nt = body.max_new_tokens if body.max_new_tokens is not None else default_max
        # Composer JSON + write_file bodies need headroom; small caps truncate tool JSON mid-stream.
        max_nt = min(default_max, max(int(max_nt), min(4096, default_max)))

        def _model_generate(model_messages: List[Dict[str, str]], req_max_new_tokens: int) -> str:
            return _generate_native(model_messages, req_max_new_tokens)

        def ndjson_chunks() -> Iterator[bytes]:
            try:
                loop = AgenticLoop(
                    model_generate=_model_generate,
                    session_id=body.session_id,
                    workspace_root=body.workspace_root,
                    max_tool_iters=body.max_tool_iters,
                    max_new_tokens=max_nt,
                    confirm_apply=body.confirm_apply,
                    allow_network_tools=body.allow_network_tools,
                    ui_artisan_mode=body.ui_artisan_mode,
                )
                for event in loop.run(user_prompt=prompt, messages=msgs):
                    line = json.dumps(event, ensure_ascii=False) + "\n"
                    yield line.encode("utf-8")
                    if event.get("type") == "tool_result":
                        data = event.get("data")
                        if isinstance(data, dict) and data.get("type") in {"diff_preview", "media"}:
                            nested = {"type": data.get("type"), **data}
                            nested_line = json.dumps(nested, ensure_ascii=False) + "\n"
                            yield nested_line.encode("utf-8")
            except Exception as e:
                err = json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n"
                yield err.encode("utf-8")

        return StreamingResponse(
            ndjson_chunks(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/agent/workspace")
    def api_agent_workspace(
        session_id: Optional[str] = Query(default=None),
        workspace_root: Optional[str] = Query(default=None),
    ) -> Dict[str, Any]:
        return list_workspace_tree(session_id, workspace_root)

    @app.post("/api/agent/apply")
    def api_agent_apply(body: AgentApplyRequest) -> Dict[str, Any]:
        if not body.proposal_ids:
            raise HTTPException(status_code=400, detail="proposal_ids is required")
        return apply_proposals(body.session_id, body.proposal_ids)

    @app.get("/api/agent/capture/{session_id}/{artifact_id}")
    def api_agent_capture(session_id: str, artifact_id: str) -> FileResponse:
        artifact = capture_artifact(session_id, artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(path=artifact.abs_path, media_type=artifact.mime_type, filename=artifact.rel_path)

    from imagination_runtime.deep_research import attach_deep_and_sync_routes

    attach_deep_and_sync_routes(app)
