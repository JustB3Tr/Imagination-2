"""Signed cookie payload for Google OAuth session (optional)."""
from __future__ import annotations

import os
from typing import Optional

_SALT = "imagination-v12-uid"


def _secret() -> str:
    return (os.environ.get("IMAGINATION_SESSION_SECRET") or "imagination-dev-insecure-change-me").strip()


def sign_user_id(user_id: int) -> str:
    from itsdangerous import URLSafeTimedSerializer

    s = URLSafeTimedSerializer(_secret(), salt=_SALT)
    return s.dumps({"u": int(user_id)})


def user_id_from_cookie(token: Optional[str], max_age_seconds: int = 86400 * 60) -> Optional[int]:
    if not token or not str(token).strip():
        return None
    from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

    s = URLSafeTimedSerializer(_secret(), salt=_SALT)
    try:
        data = s.loads(str(token), max_age=max_age_seconds)
        uid = data.get("u")
        if isinstance(uid, int) and uid > 0:
            return uid
    except (BadSignature, SignatureExpired, TypeError, ValueError):
        return None
    return None
