"""Imagination v1.4 — HTTP API only (no Gradio). Same preload / SKIP_PRELOAD / PORT as app.py.

Runs Tailscale on startup when ``TAILSCALE_KEY`` is set; serves auth (``/auth/*``), ``/api/health``,
optional Next static export at ``/next/``, and chat (``POST /api/chat``, ``POST /generate``).

**Public URL (Colab / remote):** set ``NGROK_AUTHTOKEN`` and install the ``ngrok`` binary; this script
starts a tunnel and exposes ``public_base_url`` on ``GET /api/info`` for your frontend.

Stable hostname: set ``NGROK_DOMAIN`` to a **reserved** ngrok domain (dashboard). Free tier URLs
change each session unless reserved.

You can instead set ``IMAGINATION_PUBLIC_BASE_URL`` only (no ngrok process) if you tunnel by other means.

Usage (from ``versions/v1.4`` with ``IMAGINATION_ROOT`` set):

  python api_server.py
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_V14 = Path(__file__).resolve().parent
if str(_V14) not in sys.path:
    sys.path.insert(0, str(_V14))

os.environ.setdefault("IMAGINATION_ROOT", "/content/imagination-v1.1.0")
os.environ.setdefault("IMAGINATION_MAIN_4BIT", "0")
os.environ.setdefault("IMAGINATION_MAIN_LOAD_BF16", "1")


def _skip_preload() -> bool:
    return os.getenv("SKIP_PRELOAD", "").strip().lower() in ("1", "true", "yes")


def _wait_api_health(port: int, timeout_s: float = 300.0) -> bool:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.35)
    return False


if __name__ == "__main__":
    import threading

    import uvicorn

    if _skip_preload():
        print("[imagination] SKIP_PRELOAD=1 — main model loads on server startup.", flush=True)
    else:
        from colab_backend import _load_model_stack

        _load_model_stack()
        os.environ["IMAGINATION_MAIN_ALREADY_LOADED"] = "1"

    from imagination_runtime.asgi_app import build_backend_only_app

    app = build_backend_only_app()
    port = int(os.getenv("PORT", "7860"))

    manual_pub = (os.getenv("IMAGINATION_PUBLIC_BASE_URL") or "").strip()
    ngrok_token = (os.getenv("NGROK_AUTHTOKEN") or os.getenv("NGROK_AUTH_TOKEN") or "").strip()
    need_ngrok_subprocess = bool(ngrok_token) and not manual_pub

    print()
    print("=" * 50)
    print("  Imagination v1.4.0 — API only (no Gradio)")
    print("  Local:  http://127.0.0.1:" + str(port))
    print("  Auth:   /auth/me · email + cookies · optional Google OAuth")
    print("  Chat:   POST /api/chat · POST /generate")
    print("  Tailscale on startup if TAILSCALE_KEY is set (often blocked on Colab; see docs)")
    print("  Ngrok:  NGROK_AUTHTOKEN + ngrok on PATH; optional NGROK_DOMAIN (reserved = stable URL)")
    print("=" * 50)
    print()

    ngrok_proc = None

    if need_ngrok_subprocess:

        def _run_uvicorn() -> None:
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

        th = threading.Thread(target=_run_uvicorn, daemon=False, name="imagination-uvicorn")
        th.start()
        print("[imagination] Waiting for /api/health …", flush=True)
        if not _wait_api_health(port):
            print("[imagination] Server did not become healthy in time.", flush=True)
            raise SystemExit(1)

        from imagination_runtime.ngrok_tunnel import get_public_base_url, start_ngrok_subprocess

        ngrok_proc = start_ngrok_subprocess(port)
        pub = get_public_base_url()
        if pub:
            print(
                f"[imagination] Frontend: point base URL at {pub} "
                f"or read GET {pub}/api/info → public_base_url",
                flush=True,
            )
        try:
            th.join()
        except KeyboardInterrupt:
            print("\n[imagination] Shutting down.", flush=True)
        finally:
            if ngrok_proc is not None:
                try:
                    ngrok_proc.terminate()
                except Exception:
                    pass
    else:
        if manual_pub:
            print(f"[imagination] IMAGINATION_PUBLIC_BASE_URL={manual_pub} (no ngrok subprocess)", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
