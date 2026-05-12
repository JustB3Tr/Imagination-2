"""
Hidden internal system instructions. Loaded from file and injected into
the system prompt. Never exposed to the user.

The first block is always loaded from IMAGINATION_ROOT/temp/assistant_identity.txt
(seeded from default_assistant_identity.txt on first run) so deployment owners can
override attribution without editing code.
"""
from __future__ import annotations

import json
import os

_FALLBACK_IDENTITY = (
    "[AUTHORITATIVE — THIS DEPLOYMENT ONLY]\n"
    "Create or edit temp/assistant_identity.txt under your Imagination root (IMAGINATION_ROOT) "
    "to state who built and runs this deployment. Do not claim Meta, OpenAI, Google, Anthropic, "
    "or Hugging Face as the author of this assistant UI unless that is explicitly true for this instance."
)


def assistant_identity_path(root: str) -> str:
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "assistant_identity.txt")


def _bundled_default_identity_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "default_assistant_identity.txt")


def load_identity_first_block(root: str) -> str:
    """
    Authoritative identity / attribution text (prepended first in the system prompt).

    Resolution order:
    1. IMAGINATION_ROOT/temp/assistant_identity.json — keys ``identity_text`` or ``text``
    2. IMAGINATION_ROOT/temp/assistant_identity.txt
    3. Seed (2) from packaged default_assistant_identity.txt if missing or empty
    4. Built-in fallback string
    """
    json_path = os.path.join(root, "temp", "assistant_identity.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                t = (obj.get("identity_text") or obj.get("text") or "").strip()
                if t:
                    return t
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
            pass

    path = assistant_identity_path(root)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                t = f.read().strip()
            if t:
                return t
        except (OSError, UnicodeDecodeError):
            pass

    bundled = _bundled_default_identity_path()
    if os.path.isfile(bundled):
        try:
            with open(bundled, "r", encoding="utf-8") as f:
                default_text = f.read()
            try:
                with open(path, "w", encoding="utf-8") as out:
                    out.write(default_text)
            except (OSError, UnicodeDecodeError):
                pass
            return default_text.strip()
        except (OSError, UnicodeDecodeError):
            pass

    return _FALLBACK_IDENTITY


def internal_instructions_path(root: str) -> str:
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "internal_instructions.txt")


def load_internal_instructions(root: str) -> str:
    path = internal_instructions_path(root)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (IOError, OSError):
            pass
    return ""


def tutor_math_policy_path(root: str) -> str:
    """Shipped tutor policy next to package root (v1.3) or under temp/."""
    pkg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tutor_math_policy.txt")
    if os.path.isfile(pkg):
        return pkg
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "tutor_math_policy.txt")


def load_tutor_math_policy(root: str) -> str:
    path = tutor_math_policy_path(root)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, UnicodeDecodeError):
            pass
    return ""


def inject_internal_instructions(base_prompt: str, root: str) -> str:
    """Prepend identity block, optional temp/internal_instructions.txt, then base prompt."""
    identity = load_identity_first_block(root)
    internal = load_internal_instructions(root)
    parts = [identity]
    if internal:
        parts.append(internal)
    parts.append(base_prompt)
    return "\n\n".join(parts)
