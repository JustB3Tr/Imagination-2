"""
Flask bridge for Colab full-stack: REST + proxy to the Imagination v1.4 AI backend
(FastAPI/Gradio on PORT, default 7860).

Run on 0.0.0.0 so Colab's proxy can reach you. Model lifecycle (on-demand modules) is
enforced inside the Gradio app (imagination_v1_4 + model_lifecycle); this server only
forwards HTTP where needed.
"""
from __future__ import annotations

import os
import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

FLASK_HOST = os.getenv("FLASK_BRIDGE_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_BRIDGE_PORT", "5000"))
AI_BACKEND_URL = os.getenv("IMAGINATION_AI_BACKEND_URL", "http://127.0.0.1:7860").rstrip("/")

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)
_SKIP_RESP = frozenset({"transfer-encoding", "connection", "content-encoding"})


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": os.getenv("FLASK_CORS_ORIGINS", "*")}})

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "imagination-v1.4-flask-bridge"})

    @app.get("/api/health")
    def api_health():
        ai_ok = False
        detail = ""
        try:
            r = requests.get(f"{AI_BACKEND_URL}/api/health", timeout=5)
            ai_ok = r.ok
            if not r.ok:
                detail = r.text[:200]
        except Exception as e:
            detail = str(e)
        return jsonify(
            {
                "ok": True,
                "bridge": "ok",
                "ai_backend": AI_BACKEND_URL,
                "ai_reachable": ai_ok,
                "ai_error": detail if not ai_ok else None,
            }
        )

    @app.route("/api/upstream/<path:subpath>", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    def proxy_upstream(subpath: str):
        """Forward to the AI backend (e.g. /api/upstream/api/health -> backend /api/health)."""
        return _forward(subpath)

    return app


def _forward(subpath: str):
    url = f"{AI_BACKEND_URL}/{subpath}"
    if request.query_string:
        url += "?" + request.query_string.decode()

    headers = {k: v for k, v in request.headers if k.lower() not in _HOP_BY_HOP}
    try:
        r = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=int(os.getenv("AI_BACKEND_TIMEOUT", "300")),
        )
    except requests.RequestException as e:
        return jsonify({"error": "upstream_unreachable", "detail": str(e), "url": url}), 502

    resp = Response(r.content, status=r.status_code)
    for k, v in r.headers.items():
        kl = k.lower()
        if kl in _SKIP_RESP:
            continue
        resp.headers[k] = v
    return resp


app = create_app()

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=os.getenv("FLASK_DEBUG", "0") == "1", threaded=True)
