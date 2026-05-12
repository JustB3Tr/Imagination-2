"""
Flask backend for the v1.4 Next.js + Tailwind frontend (front&back/frontend).

Inspected frontend (read-only): Next.js 16 App Router, client-only chat state in
lib/chat-context.tsx — no fetch() to an API yet. Wiring is:

  • /api/*     → handled here (health, info, config; extend for chat/auth later)
  • everything else → reverse-proxied to the Next dev/start server (default :3000)

Colab + single tunnel (dev, needs Node + two processes):
  Terminal 1:  cd ../frontend && npm install && npm run dev -- -H 0.0.0.0
  Terminal 2:  cd backend && pip install -r requirements.txt && python app.py
  Tunnel:      public URL → port 5000 (Flask)

One Python server with the model + Next static (no Flask, no Node at runtime): build the
frontend with IMAGINATION_NEXT_EXPORT=1 and IMAGINATION_NEXT_BASE_PATH=/next, then run
versions/v1.4/app.py — see imagination_runtime/asgi_app.py.

Static export (no Node at runtime): set FLASK_PROXY_NEXT=0 and serve from
frontend/out (next export) or frontend/dist — see FRONTEND_STATIC_ROOT.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from flask import Blueprint, Flask, Response, abort, jsonify, request, send_from_directory
from flask_cors import CORS

BACKEND_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = BACKEND_ROOT.parent / "frontend"
# Next static export default output; some stacks use dist/
FRONTEND_OUT = FRONTEND_ROOT / "out"
FRONTEND_DIST = FRONTEND_ROOT / "dist"

_HTTP_METHODS = ["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
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
    }
)
_SKIP_RESPONSE_HEADERS = frozenset({"transfer-encoding", "connection", "content-encoding"})


def _static_root() -> Path | None:
    env = (os.getenv("FRONTEND_STATIC_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    if FRONTEND_OUT.is_dir():
        return FRONTEND_OUT
    if FRONTEND_DIST.is_dir():
        return FRONTEND_DIST
    return None


def _forward_to_next(upstream_base: str) -> Response:
    """Proxy the current request to Next (npm run dev / start)."""
    path = request.path or "/"
    if path.startswith("/api"):
        abort(404)

    url = upstream_base.rstrip("/") + path
    if request.query_string:
        url += "?" + request.query_string.decode()

    out_headers = {
        k: v for k, v in request.headers if k.lower() not in _HOP_BY_HOP
    }

    try:
        r = requests.request(
            method=request.method,
            url=url,
            headers=out_headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=int(os.getenv("NEXT_UPSTREAM_TIMEOUT", "120")),
        )
    except requests.RequestException as e:
        return jsonify(
            {
                "error": "next_upstream_unreachable",
                "detail": str(e),
                "upstream": upstream_base,
                "hint": "Start Next on 0.0.0.0:3000: cd ../frontend && npm run dev -- -H 0.0.0.0",
            }
        ), 502

    resp = Response(r.content, status=r.status_code)
    for k, v in r.headers.items():
        if k.lower() in _SKIP_RESPONSE_HEADERS:
            continue
        resp.headers[k] = v
    return resp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    origins = os.getenv("FLASK_CORS_ORIGINS", "*")
    cors_list = [o.strip() for o in origins.split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": cors_list or "*"}})

    upstream = os.getenv("NEXT_UPSTREAM", "http://127.0.0.1:3000").strip()
    use_proxy = os.getenv("FLASK_PROXY_NEXT", "1").lower() not in ("0", "false", "no", "off")
    static_dir = _static_root()

    api_bp = Blueprint("api", __name__, url_prefix="/api")

    @api_bp.get("/health")
    def health():
        return jsonify({"ok": True, "service": "imagination-v1.4-ui-backend"})

    @api_bp.get("/info")
    def info():
        return jsonify(
            {
                "version": "0.2.0-next-proxy",
                "frontend_root": str(FRONTEND_ROOT),
                "proxy_next": use_proxy,
                "next_upstream": upstream if use_proxy else None,
                "static_dir": str(static_dir) if static_dir else None,
                "static_dir_present": bool(static_dir),
            }
        )

    @api_bp.get("/config")
    def config():
        """For browser checks without changing the Next app."""
        return jsonify(
            {
                "apiBase": "/api",
                "proxyMode": use_proxy,
                "nextUpstream": upstream if use_proxy else None,
            }
        )

    app.register_blueprint(api_bp)

    if use_proxy and upstream:

        @app.route("/", methods=_HTTP_METHODS)
        def proxy_root():
            return _forward_to_next(upstream)

        @app.route("/<path:subpath>", methods=_HTTP_METHODS)
        def proxy_subpath(subpath: str):
            return _forward_to_next(upstream)

    elif static_dir is not None:
        root = static_dir

        @app.get("/", defaults={"path": ""})
        @app.get("/<path:path>")
        def spa(path: str):
            if path.startswith("api"):
                abort(404)
            candidate = root / path
            if path and candidate.is_file():
                return send_from_directory(root, path)
            index = root / "index.html"
            if index.is_file():
                return send_from_directory(root, "index.html")
            return jsonify({"error": "static root missing index.html", "root": str(root)}), 404

    else:

        @app.get("/")
        def root_placeholder():
            return jsonify(
                {
                    "message": "No Next proxy target and no static export found.",
                    "health": "/api/health",
                    "info": "/api/info",
                    "config": "/api/config",
                    "hints": [
                        "Proxy mode (default): FLASK_PROXY_NEXT=1, start Next with npm run dev -- -H 0.0.0.0",
                        "Static mode: FLASK_PROXY_NEXT=0 and next build && next export → frontend/out, or set FRONTEND_STATIC_ROOT",
                    ],
                }
            )

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    app.run(host=host, port=port, debug=debug)
