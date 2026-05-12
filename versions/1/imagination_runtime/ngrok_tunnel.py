"""
Optional ngrok HTTP tunnel for Colab / dev: expose the API on a public https URL.

- Set ``NGROK_AUTHTOKEN`` (from https://dashboard.ngrok.com/get-started/your-authtoken).
- **Stable hostname (for your frontend):** use a **reserved domain** on your ngrok plan and set
  ``NGROK_DOMAIN`` (e.g. ``my-api.ngrok-free.dev``). Free tier URLs change every session unless you use a
  reserved name.
- Or set ``IMAGINATION_PUBLIC_BASE_URL`` manually (no ngrok subprocess) if you tunnel elsewhere.

Requires the ``ngrok`` binary on ``PATH`` (install from ngrok.com/download).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

_lock = threading.Lock()
_public_base_url: Optional[str] = None


def get_public_base_url() -> Optional[str]:
    """HTTPS origin without trailing slash, or None."""
    with _lock:
        if _public_base_url:
            return _public_base_url.rstrip("/")
    manual = (os.getenv("IMAGINATION_PUBLIC_BASE_URL") or "").strip()
    if manual:
        return manual.rstrip("/")
    return None


def set_public_base_url(url: Optional[str]) -> None:
    global _public_base_url
    with _lock:
        _public_base_url = (url or "").strip().rstrip("/") or None


def _poll_ngrok_api(timeout_s: float = 60.0) -> Optional[str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:4040/api/tunnels",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
            for t in data.get("tunnels") or []:
                u = (t.get("public_url") or "").strip()
                if u.startswith("https://"):
                    return u.rstrip("/")
                if u.startswith("http://"):
                    return u.rstrip("/")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
            pass
        time.sleep(0.4)
    return None


def start_ngrok_subprocess(local_port: int) -> Optional[subprocess.Popen]:
    """
    Start ``ngrok http <port>`` if authtoken is set and ngrok exists.
    Returns the Popen handle or None if skipped / binary missing.
    """
    if (os.getenv("IMAGINATION_PUBLIC_BASE_URL") or "").strip():
        set_public_base_url(os.getenv("IMAGINATION_PUBLIC_BASE_URL"))
        return None

    token = (os.getenv("NGROK_AUTHTOKEN") or os.getenv("NGROK_AUTH_TOKEN") or "").strip()
    if not token:
        return None

    exe = shutil.which("ngrok")
    if not exe:
        print(
            "[ngrok] Binary not found on PATH. Install from https://ngrok.com/download "
            "or: apt install ngrok / unzip the Linux zip on Colab.",
            flush=True,
        )
        return None

    cmd = [exe, "http", str(int(local_port)), "--log=stdout"]
    domain = (os.getenv("NGROK_DOMAIN") or "").strip()
    if domain:
        cmd.extend(["--domain", domain])

    env = os.environ.copy()
    env["NGROK_AUTHTOKEN"] = token

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as e:
        print(f"[ngrok] Failed to start: {e}", flush=True)
        return None

    url = _poll_ngrok_api(timeout_s=float(os.getenv("NGROK_POLL_TIMEOUT", "90")))
    if url:
        set_public_base_url(url)
        print(f"[ngrok] Tunnel → {url}", flush=True)
        if not domain:
            print(
                "[ngrok] URL changes each run on the free tier. "
                "Set NGROK_DOMAIN to a reserved hostname for a stable URL (ngrok dashboard).",
                flush=True,
            )
    else:
        print("[ngrok] Timed out waiting for http://127.0.0.1:4040/api/tunnels — check logs.", flush=True)
        try:
            proc.terminate()
        except Exception:
            pass
        return None

    return proc


def tunnel_info_dict() -> dict:
    """Fields merged into ``/api/info``."""
    u = get_public_base_url()
    manual = bool((os.getenv("IMAGINATION_PUBLIC_BASE_URL") or "").strip())
    tunnel = None
    if u:
        tunnel = "manual" if manual else "ngrok"
    return {
        "public_base_url": u,
        "public_api_health": f"{u}/api/health" if u else None,
        "tunnel": tunnel,
    }
