#!/usr/bin/env python3
"""
Imagination v1.4 — Colab single-cell full stack (or run: python colab_fullstack_orchestrator.py).

Launches in order:
  1) AI backend: app.py (preload_main_model() runs before uvicorn binds — wait on /api/health)
  2) Flask bridge: server.py on 0.0.0.0
  3) Next.js dev server on 0.0.0.0 (local node_modules/.bin/next)

If node_modules is missing, on **Colab** the orchestrator runs `npm install` in the frontend
folder automatically (disable with COLAB_STACK_AUTO_NPM_INSTALL=0).

Paste into one Colab cell after Drive mount + Python deps + Node/npm, or run from a terminal.

Model lifecycle (on-demand auxiliary modules) lives in imagination_runtime/model_lifecycle.py
and is used by the Gradio/FastAPI app inside app.py — do not bypass it for real inference.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Paths — edit REPO_ROOT for your Drive layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(os.environ.get("IMAGINATION_ROOT", "/content/imagination-v1.1.0")).resolve()
V14_DIR = REPO_ROOT / "versions" / "v1.4"
MODEL_LIFECYCLE_PY = V14_DIR / "imagination_runtime" / "model_lifecycle.py"
APP_PY = V14_DIR / "app.py"
SERVER_PY = V14_DIR / "server.py"
FRONTEND_DIR = V14_DIR / "front&back" / "frontend"

# Ports (AI uses PORT env; default 7860)
AI_PORT = int(os.environ.get("PORT", "7860"))
FLASK_PORT = int(os.environ.get("FLASK_BRIDGE_PORT", "5000"))
NEXT_PORT = int(os.environ.get("NEXT_PORT", "3000"))

# Wait for Imagination to finish preload + bind uvicorn (seconds)
AI_READY_TIMEOUT_S = float(os.environ.get("AI_READY_TIMEOUT_S", "7200"))
AI_READY_POLL_S = float(os.environ.get("AI_READY_POLL_S", "3.0"))
# Heartbeat while waiting (Colab looked "frozen" with no prints before)
AI_WAIT_STATUS_INTERVAL_S = float(os.environ.get("AI_WAIT_STATUS_INTERVAL_S", "30"))

# Logs (AI/Flask to files so pipes never fill; Next streamed in the main loop)
LOG_DIR = Path(os.environ.get("COLAB_STACK_LOG_DIR", "/tmp/imagination_colab_stack_logs"))
AI_LOG = LOG_DIR / "ai_backend.log"
FLASK_LOG = LOG_DIR / "flask_bridge.log"


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _auto_npm_install_enabled() -> bool:
    """
    If unset: auto-run npm install when Next is missing — on in Colab, off elsewhere.
    Override with COLAB_STACK_AUTO_NPM_INSTALL=1 or =0.
    """
    raw = (os.environ.get("COLAB_STACK_AUTO_NPM_INSTALL") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return _in_colab()


def _run_npm_install_frontend() -> bool:
    """Return True if node_modules now contains the Next CLI."""
    npm = shutil.which("npm") or (os.environ.get("NPM_PATH") or "").strip()
    if not npm:
        print(
            "[orchestrator] No `npm` on PATH. Install Node.js in Colab, e.g.\n"
            "  !apt-get update -qq && apt-get install -qq -y nodejs npm",
            flush=True,
        )
        return False
    fe = FRONTEND_DIR.resolve()
    print(f"[orchestrator] Running `npm install` under:\n  {fe}\n  (can take several minutes; large folder on Drive may be slow)…\n", flush=True)
    r = subprocess.run([npm, "install"], cwd=str(fe), env=os.environ.copy())
    if r.returncode != 0:
        print(f"[orchestrator] npm install failed (exit {r.returncode}).", flush=True)
        return False
    return _next_dev_command() is not None


def _next_dev_command() -> Optional[List[str]]:
    """
    Prefer local node_modules CLI (Colab: `npm run dev` often fails with `next: not found`
    when node_modules/.bin is not on PATH inside npm's sh -c wrapper).
    """
    args_tail = ["dev", "-H", "0.0.0.0", "-p", str(NEXT_PORT)]

    if platform.system() == "Windows":
        win = FRONTEND_DIR / "node_modules" / ".bin" / "next.cmd"
        if win.is_file():
            return [str(win), *args_tail]
    else:
        nix = FRONTEND_DIR / "node_modules" / ".bin" / "next"
        if nix.is_file():
            return [str(nix), *args_tail]

    # Same as `node node_modules/next/dist/bin/next` (works if .bin wrapper missing)
    dist = FRONTEND_DIR / "node_modules" / "next" / "dist" / "bin" / "next"
    if dist.is_file():
        return ["node", str(dist), *args_tail]

    return None


def _tail_log(path: Path, *, max_lines: int = 18, max_bytes: int = 48_000) -> str:
    """Last lines of a growing log file (for Colab visibility while app.py preloads)."""
    try:
        if not path.is_file():
            return "(log file not created yet — process may still be starting)"
        with open(path, "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - max_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        tail = lines[-max_lines:] if len(lines) > max_lines else lines
        return "\n".join(tail) if tail else "(log empty so far)"
    except Exception as e:
        return f"(could not read log: {e})"


def _wait_ai_ready(p_ai: subprocess.Popen, log_path: Path) -> bool:
    import urllib.request

    url = f"http://127.0.0.1:{AI_PORT}/api/health"
    deadline = time.time() + AI_READY_TIMEOUT_S
    t0 = time.time()
    last_status = 0.0

    print(
        f"[orchestrator] Waiting for {url} — main model preload + server bind can take **15–45+ minutes** "
        f"the first time; status every {AI_WAIT_STATUS_INTERVAL_S:.0f}s below.\n",
        flush=True,
    )

    while time.time() < deadline:
        code = p_ai.poll()
        if code is not None:
            print(
                f"[orchestrator] ERROR: app.py exited early with code {code}. Last log lines:\n",
                flush=True,
            )
            print(_tail_log(log_path), flush=True)
            return False

        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"[orchestrator] /api/health OK after {time.time() - t0:.0f}s.", flush=True)
                    return True
        except Exception:
            pass

        now = time.time()
        if now - last_status >= AI_WAIT_STATUS_INTERVAL_S:
            last_status = now
            elapsed = now - t0
            print(
                f"[orchestrator] Still waiting… {elapsed:.0f}s / {AI_READY_TIMEOUT_S:.0f}s — "
                f"tail {log_path.name}:",
                flush=True,
            )
            print("---", flush=True)
            print(_tail_log(log_path), flush=True)
            print("---\n", flush=True)

        time.sleep(AI_READY_POLL_S)

    print(f"[orchestrator] Timed out after {AI_READY_TIMEOUT_S:.0f}s.", flush=True)
    print(_tail_log(log_path), flush=True)
    return False


def _colab_proxy_url(port: int) -> Optional[str]:
    try:
        from google.colab.output import eval_js

        # Colab exposes the notebook kernel proxy for this port
        return eval_js(f"google.colab.kernel.proxyPort({port})")
    except Exception:
        return None


def _terminate_all(procs: List[subprocess.Popen], *, timeout: float = 15.0) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    for p in procs:
        if p.poll() is None:
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception:
                    pass


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_LIFECYCLE_PY.is_file():
        print(f"[orchestrator] WARNING: model_lifecycle not found: {MODEL_LIFECYCLE_PY}", flush=True)
    if not APP_PY.is_file():
        print(f"[orchestrator] ERROR: app.py missing: {APP_PY}", flush=True)
        return 1
    if not SERVER_PY.is_file():
        print(f"[orchestrator] ERROR: server.py missing: {SERVER_PY}", flush=True)
        return 1
    if not (FRONTEND_DIR / "package.json").is_file():
        print(f"[orchestrator] ERROR: frontend package.json missing: {FRONTEND_DIR}", flush=True)
        return 1

    env_ai = os.environ.copy()
    env_ai.setdefault("IMAGINATION_ROOT", str(REPO_ROOT))
    env_ai.setdefault("PORT", str(AI_PORT))
    # Line-buffer Python stdout/stderr into ai_backend.log (otherwise tail can look empty for a long time)
    env_ai.setdefault("PYTHONUNBUFFERED", "1")
    # Gradio share tunnel is optional; orchestrator uses Colab proxy for Next primarily
    env_ai.setdefault("GRADIO_SHARE", os.environ.get("GRADIO_SHARE", "false"))

    env_flask = os.environ.copy()
    env_flask.setdefault("IMAGINATION_AI_BACKEND_URL", f"http://127.0.0.1:{AI_PORT}")
    env_flask.setdefault("FLASK_BRIDGE_HOST", "0.0.0.0")
    env_flask.setdefault("FLASK_BRIDGE_PORT", str(FLASK_PORT))

    env_next = os.environ.copy()
    # Next binds 0.0.0.0 for Colab
    env_next.setdefault("HOSTNAME", "0.0.0.0")

    ai_log_f = open(AI_LOG, "w", encoding="utf-8", errors="replace")
    flask_log_f = open(FLASK_LOG, "w", encoding="utf-8", errors="replace")

    procs: List[subprocess.Popen] = []

    print("[orchestrator] Starting AI backend (app.py) — preload runs before port opens…", flush=True)
    print(f"[orchestrator] AI log: {AI_LOG}", flush=True)
    p_ai = subprocess.Popen(
        [sys.executable, str(APP_PY)],
        cwd=str(V14_DIR),
        env=env_ai,
        stdout=ai_log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    procs.append(p_ai)

    if not _wait_ai_ready(p_ai, AI_LOG):
        print(
            f"[orchestrator] ERROR: AI backend not ready on :{AI_PORT}/api/health within {AI_READY_TIMEOUT_S}s",
            flush=True,
        )
        _terminate_all(procs)
        ai_log_f.close()
        flask_log_f.close()
        return 1

    print(f"[orchestrator] AI backend ready — http://127.0.0.1:{AI_PORT}/", flush=True)

    print(f"[orchestrator] Starting Flask bridge on 0.0.0.0:{FLASK_PORT}…", flush=True)
    print(f"[orchestrator] Flask log: {FLASK_LOG}", flush=True)
    p_flask = subprocess.Popen(
        [sys.executable, str(SERVER_PY)],
        cwd=str(V14_DIR),
        env=env_flask,
        stdout=flask_log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    procs.append(p_flask)
    time.sleep(1.5)
    if p_flask.poll() is not None:
        print("[orchestrator] ERROR: Flask exited early; check flask log.", flush=True)
        _terminate_all(procs)
        ai_log_f.close()
        flask_log_f.close()
        return 1

    next_cmd = _next_dev_command()
    if next_cmd is None and _auto_npm_install_enabled():
        if _run_npm_install_frontend():
            next_cmd = _next_dev_command()

    if next_cmd is None:
        fe = FRONTEND_DIR.resolve()
        print(
            "[orchestrator] ERROR: Next.js dependencies still missing (no local `next` CLI).\n"
            "  Install manually, then re-run the orchestrator:",
            flush=True,
        )
        print(f'    %cd "{fe}"', flush=True)
        print("    !npm install", flush=True)
        if not _auto_npm_install_enabled():
            print(
                "\n  Tip: on Colab, auto-install is on by default; if you disabled it, unset\n"
                "  COLAB_STACK_AUTO_NPM_INSTALL or set it to 1.",
                flush=True,
            )
        _terminate_all(procs)
        ai_log_f.close()
        flask_log_f.close()
        return 1

    print(f"[orchestrator] Starting Next.js on 0.0.0.0:{NEXT_PORT}…", flush=True)
    print(f"[orchestrator] Next command: {' '.join(next_cmd)}", flush=True)
    p_next = subprocess.Popen(
        next_cmd,
        cwd=str(FRONTEND_DIR),
        env=env_next,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    procs.append(p_next)

    proxy_url = _colab_proxy_url(NEXT_PORT)
    print("\n" + "=" * 60, flush=True)
    if proxy_url:
        print(f"  Colab proxy URL (Next.js, port {NEXT_PORT}):", flush=True)
        print(f"  {proxy_url}", flush=True)
    else:
        print("  (eval_js proxy unavailable — open Colab’s port popup for", NEXT_PORT, "or use localhost)", flush=True)
    print(f"  AI backend:    http://127.0.0.1:{AI_PORT}/", flush=True)
    print(f"  Flask bridge:  http://127.0.0.1:{FLASK_PORT}/health", flush=True)
    print("=" * 60 + "\n", flush=True)

    try:
        assert p_next.stdout is not None
        while True:
            line = p_next.stdout.readline()
            if line:
                print(line, end="", flush=True)
            elif p_next.poll() is not None:
                print("[orchestrator] Next.js process ended.", flush=True)
                break
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[orchestrator] KeyboardInterrupt — stopping all processes…", flush=True)
    finally:
        _terminate_all(list(reversed(procs)))
        ai_log_f.close()
        flask_log_f.close()
        print("[orchestrator] Shutdown complete.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
