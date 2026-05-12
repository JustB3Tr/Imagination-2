"""
Deep research pipeline: 3-step cap (broad search → scrape+embed → verify+answer),
trafilatura Markdown-first scrapes, Chroma on IMAGINATION_ROOT/research_db,
CPU embeddings via multiprocessing pool, preliminary stream after 5 chunks.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Optional, Tuple

from imagination_runtime.http_chat_system import enrich_messages_with_imagination_system
from imagination_runtime.internal_prompt import load_identity_first_block
from imagination_runtime.paths import resolve_root_path
from imagination_runtime.web import get_domain, web_search

# ---------------------------------------------------------------------------
# Markdown fetch (trafilatura primary)
# ---------------------------------------------------------------------------


def _fetch_html(url: str, timeout_s: int = 20) -> str:
    import requests

    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout_s)
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if "text/html" not in ctype and "application/xhtml" not in ctype:
            return ""
        return r.text or ""
    except Exception:
        return ""


def html_to_markdown(html: str, max_chars: int) -> str:
    if not html.strip():
        return ""
    try:
        import trafilatura

        md = trafilatura.extract(
            html,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
        )
        if md and isinstance(md, str):
            text = re.sub(r"\s+\n", "\n", md).strip()
            return text[:max_chars]
    except Exception:
        pass
    # Fallback: lossy plain text (same spirit as legacy BeautifulSoup path)
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]):
        tag.decompose()
    chunks: List[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        if txt:
            chunks.append(txt)
    text = " ".join(chunks)
    text = re.sub(r"\s+", " ", text).strip()[:max_chars]
    return text


def chunk_markdown(text: str, max_len: int = 720, overlap: int = 80) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_len:
        return [t] if len(t) >= 40 else []
    out: List[str] = []
    i = 0
    while i < len(t):
        piece = t[i : i + max_len]
        if len(piece) >= 40:
            out.append(piece.strip())
        i += max_len - overlap
    return out


def _rank_urls(sources: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    seen_domains: Dict[str, int] = {}
    for s in sources:
        url = (s.get("url") or "").strip()
        if not url:
            continue
        dom = get_domain(url)
        snip = (s.get("snippet") or "")
        title = (s.get("title") or "")
        diversity_penalty = seen_domains.get(dom, 0) * 2
        seen_domains[dom] = seen_domains.get(dom, 0) + 1
        score = len(snip) + 0.5 * len(title) - diversity_penalty - (10 if "wikipedia.org" in url else 0)
        scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [x[1] for x in scored[:k]]


# ---------------------------------------------------------------------------
# Multiprocess CPU embeddings (SentenceTransformer in worker only)
# ---------------------------------------------------------------------------

_embed_model = None


def _embed_worker_init() -> None:
    global _embed_model
    from sentence_transformers import SentenceTransformer

    _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def _embed_worker_encode(texts: List[str]) -> List[List[float]]:
    global _embed_model
    import torch

    if _embed_model is None:
        _embed_worker_init()
    with torch.inference_mode():
        arr = _embed_model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return arr.tolist()


def _trace_line(
    step: str,
    summary: str,
    detail: Optional[Dict[str, Any]] = None,
) -> bytes:
    eid = uuid.uuid4().hex[:12]
    payload: Dict[str, Any] = {
        "type": "trace",
        "id": eid,
        "step": step,
        "summary": summary,
        "detail": detail or {},
    }
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _yield_trace(step: str, summary: str, detail: Optional[Dict[str, Any]] = None) -> Iterator[bytes]:
    yield _trace_line(step, summary, detail)


def _messages_for_deep(body: Any) -> List[Dict[str, str]]:
    prompt = (body.prompt or "").strip()
    if body.messages:
        return [{"role": m.role.strip(), "content": (m.content or "").strip()} for m in body.messages]
    if prompt:
        return [{"role": "user", "content": prompt}]
    return []


def _extract_queries_json(text: str) -> List[str]:
    t = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}\s*$", t)
    blob = m.group(0) if m else t
    try:
        obj = json.loads(blob)
        q = obj.get("queries")
        if isinstance(q, list):
            out = [str(x).strip() for x in q if str(x).strip()]
            return out[:5]
    except Exception:
        pass
    return [t[:200]] if t else ["research"]


def run_deep_ndjson(body: Any) -> Iterator[bytes]:
    """
    Yields NDJSON lines: trace, preliminary_ready, token (phase preliminary|final), trace done.
    """
    from imagination_runtime.chat_http import _generate_native, _stream_native
    from imagination_v1_4 import _effective_main_max_new_tokens

    base_msgs = enrich_messages_with_imagination_system(_messages_for_deep(body))
    user_question = (body.prompt or "").strip() or (
        base_msgs[-1]["content"] if base_msgs else ""
    )
    if not user_question:
        yield (json.dumps({"type": "error", "error": "empty_prompt"}, ensure_ascii=False) + "\n").encode("utf-8")
        return

    max_nt = body.max_new_tokens if body.max_new_tokens is not None else _effective_main_max_new_tokens()
    max_nt = min(max(256, int(max_nt)), _effective_main_max_new_tokens())

    root = resolve_root_path(None)
    db_path = os.path.join(root, "research_db")
    os.makedirs(db_path, exist_ok=True)

    yield from _yield_trace("plan", "Planning search queries", {"phase": "plan"})

    plan_user = (
        "Reply with ONLY valid JSON (no markdown): "
        '{"queries":["...","...","..."]} with 2–4 short web search queries '
        f"to research: {user_question[:500]}"
    )
    plan_msgs = [{"role": "user", "content": plan_user}]
    try:
        plan_raw = _generate_native(plan_msgs, 256)
    except Exception as e:
        yield (json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n").encode("utf-8")
        return

    queries = _extract_queries_json(plan_raw)
    yield from _yield_trace("plan", "Search queries ready", {"queries": queries})

    # --- Step 1: broad search (3-step cap) ---
    cache: Dict[str, List[Dict[str, Any]]] = {}
    all_hits: List[Dict[str, Any]] = []
    for q in queries[:3]:
        hits = web_search(
            q,
            max_results=12,
            max_sources=10,
            trusted_domains=[],
            cache=cache,
        )
        for h in hits:
            h = dict(h)
            h["query"] = q
            all_hits.append(h)

    yield from _yield_trace(
        "search",
        f"Broad search: {len(all_hits)} hits",
        {"queries": queries, "hit_count": len(all_hits)},
    )

    ranked = _rank_urls(all_hits, k=5)
    urls = [(h.get("url") or "").strip() for h in ranked if (h.get("url") or "").strip()][:5]
    yield from _yield_trace(
        "select",
        f"Selected {len(urls)} URLs",
        {"urls": urls},
    )

    # --- Step 2: targeted fetch + markdown + chunk ---
    chunks: List[Dict[str, Any]] = []
    max_page_chars = int(os.getenv("IMAGINATION_DEEP_PAGE_CHARS", "120000"))
    for url in urls:
        html = _fetch_html(url)
        md = html_to_markdown(html, max_chars=max_page_chars)
        for i, ch in enumerate(chunk_markdown(md)):
            if len(ch) < 50:
                continue
            chunks.append({"url": url, "text": ch, "chunk_index": i})

    yield from _yield_trace(
        "fetch",
        f"Fetched {len(urls)} pages → {len(chunks)} chunks",
        {"url_count": len(urls), "chunk_count": len(chunks)},
    )

    if not chunks:
        yield from _yield_trace("embed", "No chunks to embed; answering from model only", {})
        msgs = base_msgs + [{"role": "user", "content": user_question}]
        try:
            for text in _stream_native(msgs, max_nt):
                line = json.dumps({"type": "token", "phase": "final", "text": text}, ensure_ascii=False) + "\n"
                yield line.encode("utf-8")
        except Exception as e:
            yield (json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n").encode("utf-8")
        yield from _yield_trace("done", "Deep research finished", {})
        return

    import chromadb

    coll_name = f"deep_{uuid.uuid4().hex[:20]}"
    client = chromadb.PersistentClient(path=db_path)
    collection = client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})

    pool_size = max(1, min(2, int(os.getenv("IMAGINATION_EMBED_POOL", "1"))))
    pool: Optional[Pool] = None
    try:
        pool = Pool(processes=pool_size, initializer=_embed_worker_init)
    except Exception:
        pool = None
        _embed_worker_init()

    def encode_batch(texts: List[str]) -> List[List[float]]:
        if pool is not None:
            return pool.apply(_embed_worker_encode, (texts,))
        return _embed_worker_encode(texts)

    embedded = 0
    rest_thread: Optional[threading.Thread] = None
    rest_exc: List[Optional[BaseException]] = [None]
    lock = threading.Lock()

    def embed_slice(start: int, end: int) -> None:
        nonlocal embedded
        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_meta: List[Dict[str, Any]] = []
        for j in range(start, end):
            c = chunks[j]
            tid = f"{j}"
            batch_ids.append(tid)
            batch_docs.append(c["text"])
            batch_meta.append({"url": c["url"], "chunk_index": str(c["chunk_index"])})
            if len(batch_ids) >= 8 or j == end - 1:
                embs = encode_batch(batch_docs)
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=embs)
                with lock:
                    embedded += len(batch_ids)
                batch_ids, batch_docs, batch_meta = [], [], []
        if batch_ids:
            embs = encode_batch(batch_docs)
            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=embs)
            with lock:
                embedded += len(batch_ids)

    first_end = min(5, len(chunks))
    try:
        embed_slice(0, first_end)
    except Exception as e:
        yield (json.dumps({"type": "error", "error": f"embed: {e}"}, ensure_ascii=False) + "\n").encode("utf-8")
        if pool is not None:
            pool.close()
            pool.join()
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass
        return

    if first_end < len(chunks):

        def _rest() -> None:
            try:
                embed_slice(first_end, len(chunks))
            except BaseException as e:
                rest_exc[0] = e

        rest_thread = threading.Thread(target=_rest, name="deep-embed-rest", daemon=True)
        rest_thread.start()

    pre_threshold = min(5, len(chunks))
    if pre_threshold > 0 and embedded >= pre_threshold:
        yield (
            json.dumps(
                {"type": "preliminary_ready", "chunk_count": embedded},
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")

        pre_sys = (
            "You are writing Preliminary Findings from early retrieved notes only. "
            "Be concise; label uncertainty; do not claim verification yet."
        )
        pre_ctx = "\n\n".join(c["text"][:1200] for c in chunks[:pre_threshold])
        pre_msgs = [
            {"role": "system", "content": pre_sys},
            {
                "role": "user",
                "content": f"Question:\n{user_question}\n\nEarly excerpts:\n{pre_ctx}\n\nDraft preliminary findings.",
            },
        ]
        pre_cap = min(512, max_nt)
        try:
            for text in _stream_native(pre_msgs, pre_cap):
                line = (
                    json.dumps({"type": "token", "phase": "preliminary", "text": text}, ensure_ascii=False) + "\n"
                )
                yield line.encode("utf-8")
        except Exception as e:
            yield (json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n").encode("utf-8")

    if rest_thread is not None:
        rest_thread.join(timeout=600)
        if rest_exc[0] is not None:
            yield (
                json.dumps({"type": "trace", "step": "embed", "summary": f"Rest embed error: {rest_exc[0]}"}, ensure_ascii=False)
                + "\n"
            ).encode("utf-8")

    yield from _yield_trace("embed", f"Embedded {embedded} chunks", {"embedded": embedded})

    # --- Step 3: verify (short, bounded) ---
    yield from _yield_trace("verify", "Running fact verification pass", {"phase": "verify"})
    ctx_parts: List[str] = []
    try:
        q_emb = encode_batch([user_question])[0]
        qres = collection.query(query_embeddings=[q_emb], n_results=min(12, max(4, len(chunks))))
        docs = (qres.get("documents") or [[]])[0]
        metas = (qres.get("metadatas") or [[]])[0]
        for d, meta in zip(docs, metas):
            if not d:
                continue
            u = (meta or {}).get("url", "")
            ctx_parts.append(f"(Source: {u})\n{d[:2000]}")
        ctx_block = "\n\n---\n\n".join(ctx_parts[:8])
    except Exception as e:
        ctx_block = f"(retrieval failed: {e})"

    verify_msgs = [
        {
            "role": "system",
            "content": "List 3–6 bullet claims from the excerpts that are most relevant; note conflicts or gaps. JSON only: {\"claims\":[...]}",
        },
        {"role": "user", "content": ctx_block[:8000]},
    ]
    try:
        verify_out = _generate_native(verify_msgs, 384)
    except Exception:
        verify_out = ""
    yield from _yield_trace("verify", "Verification notes", {"raw": verify_out[:800]})

    yield from _yield_trace(
        "retrieve",
        "Retrieved context for final answer",
        {"snippets": len(ctx_parts)},
    )

    if pool is not None:
        pool.close()
        pool.join()

    final_sys = (
        "Answer using the retrieved excerpts and verification notes. "
        "Address counterarguments and uncertainty explicitly; do not only confirm the user's prior bias. "
        "Cite sources by URL when possible."
    )
    final_user = f"Question:\n{user_question}\n\nRetrieved context:\n{ctx_block[:12000]}\n\nVerification:\n{verify_out[:2000]}"
    hist_only = [m for m in base_msgs if m.get("role") in ("user", "assistant")]
    tail_hist = hist_only[-6:] if len(hist_only) > 6 else hist_only
    combined_sys = (load_identity_first_block(root) + "\n\n" + final_sys).strip()
    final_msgs = [{"role": "system", "content": combined_sys}, *tail_hist, {"role": "user", "content": final_user}]

    try:
        for text in _stream_native(final_msgs, max_nt):
            line = json.dumps({"type": "token", "phase": "final", "text": text}, ensure_ascii=False) + "\n"
            yield line.encode("utf-8")
    except Exception as e:
        yield (json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n").encode("utf-8")

    try:
        client.delete_collection(coll_name)
    except Exception:
        pass

    yield from _yield_trace("done", "Deep research finished", {})


def attach_deep_and_sync_routes(app: Any) -> None:
    """Register on FastAPI app (called from chat_http after core routes)."""
    from fastapi import HTTPException, Header
    from fastapi.responses import StreamingResponse
    from starlette.requests import Request as IncomingRequest

    from pydantic import ValidationError

    from imagination_runtime.chat_http import ChatApiRequest, coerce_chat_api_request_dict

    # Use starlette IncomingRequest explicitly: some FastAPI builds mis-bind names like ``http_request``
    # or ``request`` as *query* parameters (422 Field required loc: query) when ``Request`` is not
    # resolved correctly from ``fastapi``.
    @app.post("/api/chat/deep")
    async def api_chat_deep(incoming: IncomingRequest) -> StreamingResponse:
        try:
            raw: Any = await incoming.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid_json")
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="expected_json_object")
        merged = coerce_chat_api_request_dict(raw)
        try:
            body = ChatApiRequest.model_validate(merged)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail={"errors": e.errors()}) from e

        uq = (body.prompt or "").strip()
        if not uq and body.messages:
            for m in reversed(body.messages):
                c = (m.content or "").strip()
                if (m.role or "").lower() == "user" and c:
                    uq = c
                    break
        if not uq:
            raise HTTPException(
                status_code=400,
                detail="empty_prompt: send prompt, message, text, query, or messages with content",
            )

        def gen() -> Iterator[bytes]:
            yield from run_deep_ndjson(body)

        return StreamingResponse(
            gen(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _user_dir(user_id: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", user_id)[:120] or "anonymous"
        d = os.path.join(resolve_root_path(None), "users", safe)
        os.makedirs(d, exist_ok=True)
        return d

    def _chats_path(user_id: str) -> str:
        return os.path.join(_user_dir(user_id), "chats.jsonl")

    @app.get("/api/sync/chats")
    def sync_chats_get(x_imagination_user_id: Optional[str] = Header(None)) -> Dict[str, Any]:
        uid = (x_imagination_user_id or "").strip() or "anonymous"
        path = _chats_path(uid)
        if not os.path.isfile(path):
            return {"chats": []}
        chats: List[Dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chats.append(json.loads(line))
        except Exception:
            return {"chats": []}
        return {"chats": chats}

    @app.post("/api/sync/chats")
    async def sync_chats_post(
        incoming: IncomingRequest,
        x_imagination_user_id: Optional[str] = Header(None),
    ) -> Dict[str, str]:
        uid = (x_imagination_user_id or "").strip() or "anonymous"
        try:
            payload = await incoming.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid_json")

        chats = payload.get("chats")
        if not isinstance(chats, list):
            raise HTTPException(status_code=400, detail="expected_chats_array")

        path = _chats_path(uid)
        with open(path, "w", encoding="utf-8") as f:
            for obj in chats:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return {"status": "ok"}
