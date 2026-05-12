"""
Email/password and OAuth (Gradio / Google) account resolution.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from .users import (
    fetch_email_credentials,
    get_or_create_user,
    get_user_by_id,
    User,
)


def hash_password(password: str) -> str:
    try:
        import bcrypt

        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except ImportError:
        import hashlib

        return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    try:
        import bcrypt

        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except (ImportError, ValueError):
        import hashlib

        return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored_hash


def login_email_password(root: str, email: str, password: str) -> Optional[User]:
    email_key = (email or "").strip().lower()
    if not email_key or "@" not in email_key:
        return None
    row = fetch_email_credentials(root, email_key)
    if not row:
        return None
    user_id, stored_hash = row
    if not verify_password(password or "", stored_hash):
        return None
    return get_user_by_id(root, user_id)


def signup_email_password(
    root: str,
    email: str,
    password: str,
    display_name: str = "",
) -> Tuple[Optional[User], Optional[str]]:
    provider = "email"
    provider_uid = (email or "").strip().lower()
    if not provider_uid or "@" not in provider_uid:
        return None, "Invalid email"
    if not password or len(password) < 6:
        return None, "Password must be at least 6 characters"
    if fetch_email_credentials(root, provider_uid):
        return None, "That email is already registered — use Sign in."

    pw_hash = hash_password(password)
    user = get_or_create_user(
        root,
        provider=provider,
        provider_uid=provider_uid,
        display_name=(display_name or "").strip() or provider_uid.split("@")[0],
        email=provider_uid,
        password_hash=pw_hash,
    )
    return user, None


def resolve_user_from_gradio_oauth(profile: Any, root: str) -> Optional[User]:
    """
    Map Gradio OAuthProfile (HF, or Google when enabled on the Space) to a DB user.
    """
    if profile is None:
        return None
    sub = getattr(profile, "sub", None) or getattr(profile, "id", None)
    name = getattr(profile, "name", None) or getattr(profile, "username", None) or ""
    email = getattr(profile, "email", None) or ""
    prov_raw = (getattr(profile, "provider", None) or "").lower()
    if "google" in prov_raw:
        prov = "google"
    elif prov_raw in ("github",):
        prov = "github"
    else:
        prov = "huggingface"
    uid_key = str(sub or email or name or id(profile))
    display = (name or "").strip() or (email.split("@")[0] if email else "User")
    return get_or_create_user(
        root,
        provider=prov,
        provider_uid=uid_key,
        display_name=display,
        email=email or "",
    )


def session_dict_from_user(u: User) -> dict:
    return {
        "id": u.id,
        "name": u.display_name or u.email or f"user-{u.id}",
        "email": u.email or "",
    }


def account_markdown(user_id: int, display_name: str) -> str:
    safe = (display_name or "User").replace("*", "").strip() or "User"
    if user_id and user_id > 0:
        return f"**Signed in as {safe}** — your chats and saved topic memory stay on this account only."
    return "**Guest** — sign in so chat history and memory are not shared with other visitors."
