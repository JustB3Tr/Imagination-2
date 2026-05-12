#!/usr/bin/env python3
"""
Imagination v1.4 — Colab FastAPI backend for a remote frontend (e.g. v0) over Tailscale.

- Loads the main model like the Gradio app: `preload_main_model()` + optional LoRA (PEFT).
- Text path: `generate_full()` → `apply_chat_template` + same generate kwargs as main chat.
- VLM / CLIP+projector path: `generate_stream_vlm()` (same as production).

For **auth + same HTTP surface as ``app.py`` without Gradio**, use:

  python api_server.py

Before import, set (recommended on Colab with ~20GB+ VRAM for bf16, or "1" for 4-bit to save VRAM):

  os.environ["IMAGINATION_ROOT"] = "/content/imagination-v1.1.0"
  # Single switch for the *main* causal LM (CLIP+projector stay float). True/1 = 4-bit; False/0 = bf16.
  os.environ["IMAGINATION_MAIN_4BIT"] = "0"   # bfloat16 main LM (or "1" for bitsandbytes 4-bit)
  # Legacy alias (if IMAGINATION_MAIN_4BIT is unset): IMAGINATION_MAIN_LOAD_BF16=1 => bf16, no 4-bit

On startup, the first main-model load prints a line like::

  [imagination] Main LM: bfloat16 (no 4-bit quant) — …
  [imagination] Vision: CLIP tower + MLP projector (float; not 4-bit); loading CLIP+projector+LM…

Tailscale auth key (in order):
  1) Env `TAILSCALE_KEY` (recommended: set in the **notebook kernel** before starting — see below)
  2) File path in env `TAILSCALE_KEY_FILE` (first line = key)
  3) Colab **Secrets** → `TAILSCALE_KEY` via `google.colab.userdata` (only works in the **same
     Python process** as the notebook kernel — **not** in a `!python colab_backend.py` subprocess)

  If you use `!python colab_backend.py`, either export the key for that shell or start the server
  from a Python cell after `os.environ["TAILSCALE_KEY"] = userdata.get("TAILSCALE_KEY")`.

  In **Jupyter/Colab**, do not call `uvicorn.run()` in the **main** notebook thread — IPython already
  has an asyncio loop and uvicorn would call `asyncio.run()` (RuntimeError). Prefer
  ``run_colab_backend_subprocess()`` (separate process, no notebook asyncio) or
  ``run_colab_backend_server()`` (uvicorn in a background thread with explicit ``asyncio.run``).

MagicDNS hostname: `--hostname=ai-backend` → https://ai-backend.<your-tailnet>.ts.net:8000

Run from the v1.4 directory (or ensure `imagination_runtime` resolves):

  cd versions/v1.4 && python colab_backend.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Thread
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths & Colab-friendly defaults (before importing imagination_*)
# ---------------------------------------------------------------------------
_V13 = Path(__file__).resolve().parent
if str(_V13) not in sys.path:
    sys.path.insert(0, str(_V13))

os.environ.setdefault("IMAGINATION_ROOT", "/content/imagination-v1.1.0")
# Default Colab: bf16 main LM. Set IMAGINATION_MAIN_4BIT=1 *before* import to use 4-bit and save VRAM.
os.environ.setdefault("IMAGINATION_MAIN_4BIT", "0")
# Legacy: still honored when IMAGINATION_MAIN_4BIT is unset; redundant when MAIN_4BIT=0
os.environ.setdefault("IMAGINATION_MAIN_LOAD_BF16", "1")

from fastapi import FastAPI

from imagination_runtime.chat_http import add_cors_from_env, attach_generation_routes


def _read_tailscale_key() -> Optional[str]:
    """
    Colab Secrets are only readable from the notebook kernel process. A subprocess started with
    `!python colab_backend.py` cannot see `userdata` unless you inject the key into the environment
    first (see module docstring).
    """
    env_key = (os.environ.get("TAILSCALE_KEY") or "").strip()
    if env_key:
        return env_key

    path = (os.environ.get("TAILSCALE_KEY_FILE") or "").strip()
    if path:
        try:
            p = Path(path).expanduser()
            if p.is_file():
                line = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                if line:
                    return line[0].strip()
        except OSError as e:
            print(f"[colab_backend] Could not read TAILSCALE_KEY_FILE {path}: {e}", flush=True)

    try:
        from google.colab import userdata
    except ImportError:
        return None

    try:
        return (userdata.get("TAILSCALE_KEY") or "").strip() or None
    except Exception as e:
        name = type(e).__name__
        if "Secret" in name or "NotFound" in name:
            print(
                "[colab_backend] Colab Secret TAILSCALE_KEY not found or notebook access disabled. "
                "Open the key icon (Secrets), add TAILSCALE_KEY, and turn ON access for this notebook.",
                flush=True,
            )
        else:
            print(f"[colab_backend] userdata.get(TAILSCALE_KEY) failed: {e}", flush=True)
        return None


def _tailscale_connect() -> None:
    """Install + `tailscale up` if a key is available from env, file, or Colab userdata."""
    key = _read_tailscale_key()
    if not key:
        print(
            "[colab_backend] No TAILSCALE_KEY — skipping Tailscale.\n"
            "  Fix: In a **Python** cell run:\n"
            "    from google.colab import userdata\n"
            "    import os\n"
            "    os.environ['TAILSCALE_KEY'] = userdata.get('TAILSCALE_KEY')\n"
            "  Then start the server from **Python** (same process), e.g.:\n"
            "    from colab_backend import run_colab_backend_subprocess\n"
            "    run_colab_backend_subprocess()\n"
            "  Or: run_colab_backend_server() (thread). Do not call uvicorn.run() in the main cell.\n"
            "  Or set env in the shell before `!python`: not all Colab versions pass kernel env to `!python`.",
            flush=True,
        )
        return

    if shutil.which("tailscale") is None:
        print("[colab_backend] Installing Tailscale…", flush=True)
        subprocess.run(
            "curl -fsSL https://tailscale.com/install.sh | sh",
            shell=True,
            check=False,
        )
    hostname = (os.environ.get("TAILSCALE_HOSTNAME") or "ai-backend").strip()
    print(f"[colab_backend] Running tailscale up (hostname={hostname})…", flush=True)
    rc = subprocess.run(
        [
            "sudo",
            "tailscale",
            "up",
            "--auth-key",
            key,
            f"--hostname={hostname}",
            "--accept-dns=false",
        ],
        check=False,
    )
    if rc.returncode != 0:
        print(
            "[colab_backend] tailscale up failed — check auth key and sudo. "
            "Continuing without Tailscale.",
            flush=True,
        )
    else:
        port = int(os.getenv("COLAB_BACKEND_PORT", os.getenv("PORT", "8000")))
        print(
            "[colab_backend] Tailscale up. Open (example) "
            f"https://{hostname}.<your-tailnet>.ts.net:{port} from your client.",
            flush=True,
        )


def _maybe_apply_peft_adapter(root: str, model: Any) -> Any:
    """Optional LoRA / PEFT on top of the loaded main causal LM (train_imagination output)."""
    raw = (os.environ.get("IMAGINATION_PEFT_ADAPTER_PATH") or "").strip()
    if not raw:
        return model
    path = raw if os.path.isabs(raw) else os.path.join(root, raw)
    if not os.path.isdir(path):
        print(f"[colab_backend] PEFT path missing, skipping: {path}", flush=True)
        return model
    try:
        from peft import PeftModel
    except ImportError as e:
        print(f"[colab_backend] peft not installed, skipping adapter: {e}", flush=True)
        return model
    print(f"[colab_backend] Loading PEFT adapter from: {path}", flush=True)
    m = PeftModel.from_pretrained(model, path)
    m.eval()
    if (os.environ.get("IMAGINATION_PEFT_MERGE") or "").strip().lower() in ("1", "true", "yes"):
        print("[colab_backend] merge_and_unload()", flush=True)
        m = m.merge_and_unload()
        m.eval()
    return m


def _load_model_stack() -> None:
    from imagination_runtime.paths import resolve_root_path
    from imagination_v1_4 import RUNTIME, preload_main_model

    preload_main_model()
    root = resolve_root_path(None)
    if RUNTIME.main_model is not None:
        RUNTIME.main_model = _maybe_apply_peft_adapter(root, RUNTIME.main_model)
    print("[colab_backend] Model stack ready.", flush=True)


def create_app() -> FastAPI:
    app = FastAPI(title="Imagination 1.3 Colab Backend", version="1.3.0")
    add_cors_from_env(app)
    attach_generation_routes(app)

    @app.on_event("startup")
    def _startup() -> None:
        _tailscale_connect()
        _load_model_stack()

    return app


app = create_app()


def run_colab_backend_server(host: Optional[str] = None, port: Optional[int] = None) -> Thread:
    """
    Run uvicorn in a **background thread** so the main notebook thread never calls ``asyncio.run``.

    Uses ``asyncio.run(server.serve())`` explicitly (same as ``uvicorn.run``) to avoid odd interactions
    with IPython. If you still see ``RuntimeWarning: coroutine 'Server.serve' was never awaited``,
    try ``run_colab_backend_subprocess()`` instead — a clean process with no Jupyter event loop.

    The CLI entrypoint ``python colab_backend.py`` still uses ``uvicorn.run`` in a fresh process.
    """
    import asyncio
    import uvicorn

    h = host if host is not None else os.getenv("COLAB_BACKEND_HOST", "0.0.0.0")
    p = port if port is not None else int(os.getenv("COLAB_BACKEND_PORT", "8000"))

    def _serve() -> None:
        config = uvicorn.Config(app, host=h, port=p)
        server = uvicorn.Server(config)
        asyncio.run(server.serve())

    th = Thread(target=_serve, daemon=True, name="colab-backend-uvicorn")
    th.start()
    print(
        f"[colab_backend] Uvicorn starting on http://{h}:{p}/ (background thread). "
        "When logs show model ready, try GET /health on 127.0.0.1.",
        flush=True,
    )
    return th


def run_colab_backend_subprocess(
    host: Optional[str] = None,
    port: Optional[int] = None,
    *,
    cwd: Optional[Path] = None,
) -> subprocess.Popen:
    """
    Start ``python -m uvicorn colab_backend:app`` in a **child process**.

    Best Colab option when the notebook shows asyncio/coroutine warnings: no IPython loop in that
    process. Set ``TAILSCALE_KEY`` and ``IMAGINATION_ROOT`` on ``os.environ`` in the kernel **before**
    calling — the child inherits the environment (Colab ``userdata`` is not re-read in the child).
    """
    h = host if host is not None else os.getenv("COLAB_BACKEND_HOST", "0.0.0.0")
    p = port if port is not None else int(os.getenv("COLAB_BACKEND_PORT", "8000"))
    workdir = cwd if cwd is not None else _V13
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "colab_backend:app",
        "--host",
        h,
        "--port",
        str(p),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        env=os.environ.copy(),
    )
    print(
        f"[colab_backend] uvicorn pid={proc.pid} {h}:{p} cwd={workdir} "
        "(subprocess — tailscale + model load run in this process)",
        flush=True,
    )
    return proc


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("COLAB_BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("COLAB_BACKEND_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
