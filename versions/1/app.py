"""Imagination v1.4 — Entry point. Run with: python app.py"""
from __future__ import annotations

import os
import secrets
import threading
import time
from typing import Optional


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _wait_for_server(port: int, timeout_s: float = 90.0) -> bool:
    """Wait until something answers on 127.0.0.1:port (uvicorn + mounted Gradio)."""
    deadline = time.time() + timeout_s
    local = f"http://127.0.0.1:{port}/"
    while time.time() < deadline:
        try:
            from gradio import networking

            if networking.url_ok(local):
                return True
        except Exception:
            pass
        try:
            import urllib.request

            urllib.request.urlopen(local, timeout=2)
            return True
        except Exception:
            time.sleep(0.35)
    return False


def _gradio_public_url(port: int) -> Optional[str]:
    """Same FRP tunnel Gradio uses for demo.launch(share=True)."""
    try:
        from gradio import networking
    except ImportError:
        print("[imagination] Gradio not installed; cannot create share link.", flush=True)
        return None
    try:
        return networking.setup_tunnel(
            "127.0.0.1",
            int(port),
            secrets.token_urlsafe(32),
            None,
            None,
        )
    except Exception as e:
        print(f"[imagination] Could not create Gradio share tunnel: {e}", flush=True)
        return None


def _run_uvicorn_with_share(app, port: int) -> None:
    """Colab / remote: public *.gradio.live URL while keeping FastAPI auth routes."""

    def _serve() -> None:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    th = threading.Thread(target=_serve, daemon=False, name="imagination-uvicorn")
    th.start()
    print("[imagination] Waiting for local server…", flush=True)
    if _wait_for_server(port):
        pub = _gradio_public_url(port)
        if pub:
            print(f"\n* Running on public URL: {pub}\n", flush=True)
            print(
                "(Gradio tunnel — link expires after ~1 week. Set GRADIO_SHARE=false to disable.)\n",
                flush=True,
            )
        else:
            print(
                f"[imagination] Local only: http://127.0.0.1:{port}/ "
                "(tunnel failed; in Colab try GRADIO_SHARE=true after fixing errors above.)",
                flush=True,
            )
    else:
        print(f"[imagination] Server on port {port} did not respond in time.", flush=True)
    try:
        th.join()
    except KeyboardInterrupt:
        print("\n[imagination] Shutting down.", flush=True)


if __name__ == "__main__":
    from imagination_v1_4 import build_ui, preload_main_model

    if os.getenv("SKIP_PRELOAD", "").strip().lower() not in ("1", "true", "yes"):
        preload_main_model()
    else:
        print("[imagination] SKIP_PRELOAD=1 — main model loads on first message.", flush=True)

    demo = build_ui(auth_http_available=True)
    port = int(os.getenv("PORT", "7860"))
    share_env = (os.getenv("GRADIO_SHARE") or "").strip().lower()
    if share_env in ("0", "false", "no"):
        share = False
    elif share_env in ("1", "true", "yes"):
        share = True
    else:
        # Default: share in Colab (localhost is useless there); opt-in elsewhere
        share = _in_colab()

    print()
    print("=" * 50)
    print("  Imagination v1.4.0")
    print("  Local:  http://127.0.0.1:" + str(port))
    if share:
        print("  Share:  Gradio tunnel URL will print once the server is up")
    print("  Auth:   email forms + cookies · optional Google (GOOGLE_CLIENT_ID/SECRET)")
    print("=" * 50)
    print()

    try:
        from imagination_runtime.asgi_app import build_full_app
        import uvicorn

        app = build_full_app(demo)
        if share:
            _run_uvicorn_with_share(app, port)
        else:
            uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError as e:
        print(f"[imagination] ASGI stack missing ({e}); falling back to demo.launch().", flush=True)
        print("  Install: pip install fastapi uvicorn authlib python-multipart itsdangerous", flush=True)
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share or _in_colab(),
            show_error=True,
        )
