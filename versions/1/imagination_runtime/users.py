"""
Per-user memory, preferences, global admin memory, and topic memory.
SQLite-backed user data; global memory from a text file.
Topic memory: auto-extracted key topics from conversations, invisible to user.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


STOPWORDS = frozenset(
    "a an the and or but in on at to for of with by from as is was are were been be have has had do does did will would could should may might must shall can need dare ought used".split()
)

# Long-lived learner profile (SQLite user_prefs) — calibrate explanations / tone per account.
PREF_PROFILE_ABOUT = "im_profile_about"
PREF_PROFILE_GOALS = "im_profile_goals"
PREF_PROFILE_SKILLS = "im_profile_skills"

# Last specialist module + short snapshot (for routing and subagent prompts).
PREF_LAST_SUBAGENT = "im_last_subagent"
PREF_LAST_SUBAGENT_CTX = "im_last_subagent_ctx"
LAST_SUBAGENT_CTX_MAX_CHARS = 720

# v1.3: multi-thread chat library + last-open thread for signed-in users
PREF_ACTIVE_THREAD_ID = "im_active_thread_id"
THREAD_KIND_GENERAL = "general"
THREAD_KIND_TUTOR = "tutor"


@dataclass
class User:
    id: int
    provider: str
    provider_uid: str
    display_name: str
    email: str
    created_at: str


def _db_path(root: str) -> str:
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "users.db")


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            provider_uid TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            email TEXT DEFAULT '',
            password_hash TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(provider, provider_uid)
        );
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id INTEGER PRIMARY KEY,
            memory_text TEXT DEFAULT '',
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS user_prefs (
            user_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (user_id, key),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS topic_memory (
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            detail TEXT DEFAULT '',
            last_mentioned TEXT DEFAULT (datetime('now')),
            mention_count INTEGER DEFAULT 1,
            PRIMARY KEY (user_id, topic),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS user_chat_state (
            user_id INTEGER PRIMARY KEY,
            conversation_json TEXT NOT NULL DEFAULT '[]',
            display_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS user_chat_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            kind TEXT NOT NULL DEFAULT 'general',
            title TEXT NOT NULL DEFAULT '',
            conversation_json TEXT NOT NULL DEFAULT '[]',
            display_json TEXT NOT NULL DEFAULT '[]',
            archived INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_users_provider_uid ON users(provider, provider_uid);
        CREATE INDEX IF NOT EXISTS idx_topic_memory_user_mentioned ON topic_memory(user_id, last_mentioned);
        CREATE INDEX IF NOT EXISTS idx_chat_threads_user_updated ON user_chat_threads(user_id, updated_at DESC);
    """)


def get_or_create_user(
    root: str,
    *,
    provider: str,
    provider_uid: str,
    display_name: str = "",
    email: str = "",
    password_hash: Optional[str] = None,
) -> User:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "SELECT id, provider, provider_uid, display_name, email, created_at FROM users WHERE provider = ? AND provider_uid = ?",
            (provider, provider_uid),
        )
        row = cur.fetchone()
        if row:
            return User(id=row[0], provider=row[1], provider_uid=row[2], display_name=row[3] or "", email=row[4] or "", created_at=row[5] or "")
        conn.execute(
            "INSERT INTO users (provider, provider_uid, display_name, email, password_hash) VALUES (?, ?, ?, ?, ?)",
            (provider, provider_uid, display_name, email, password_hash),
        )
        conn.commit()
        uid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        cur2 = conn.execute("SELECT id, provider, provider_uid, display_name, email, created_at FROM users WHERE id = ?", (uid,))
        r = cur2.fetchone()
        return User(id=r[0], provider=r[1], provider_uid=r[2], display_name=r[3] or "", email=r[4] or "", created_at=r[5] or "")
    finally:
        conn.close()


def get_user_by_id(root: str, user_id: int) -> Optional[User]:
    if not user_id:
        return None
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "SELECT id, provider, provider_uid, display_name, email, created_at FROM users WHERE id = ?",
            (int(user_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        return User(
            id=row[0],
            provider=row[1],
            provider_uid=row[2],
            display_name=row[3] or "",
            email=row[4] or "",
            created_at=row[5] or "",
        )
    finally:
        conn.close()


def fetch_email_credentials(root: str, email: str) -> Optional[Tuple[int, str]]:
    """Return (user_id, password_hash) for provider=email, or None."""
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "SELECT id, password_hash FROM users WHERE provider = 'email' AND lower(email) = lower(?) AND password_hash IS NOT NULL",
            ((email or "").strip(),),
        )
        row = cur.fetchone()
        if not row or not row[1]:
            return None
        return int(row[0]), str(row[1])
    finally:
        conn.close()


def get_user_by_email_password(root: str, email: str, password_hash: str) -> Optional[User]:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "SELECT id, provider, provider_uid, display_name, email, created_at FROM users WHERE provider = 'email' AND email = ? AND password_hash = ?",
            (email, password_hash),
        )
        row = cur.fetchone()
        if row:
            return User(id=row[0], provider=row[1], provider_uid=row[2], display_name=row[3] or "", email=row[4] or "", created_at=row[5] or "")
        return None
    finally:
        conn.close()


def load_user_memory(root: str, user_id: int) -> str:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute("SELECT memory_text FROM user_memory WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        return (row[0] or "").strip() if row else ""
    finally:
        conn.close()


def save_user_chat_state(
    root: str,
    user_id: int,
    conversation: List[Dict[str, Any]],
    display: List[Dict[str, Any]],
) -> None:
    """Persist chat transcripts for signed-in users only."""
    if not user_id:
        return
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        conn.execute(
            """INSERT INTO user_chat_state (user_id, conversation_json, display_json, updated_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(user_id) DO UPDATE SET
                 conversation_json = excluded.conversation_json,
                 display_json = excluded.display_json,
                 updated_at = datetime('now')""",
            (
                int(user_id),
                json.dumps(conversation or [], ensure_ascii=False),
                json.dumps(display or [], ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_user_chat_state(root: str, user_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not user_id:
        return [], []
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "SELECT conversation_json, display_json FROM user_chat_state WHERE user_id = ?",
            (int(user_id),),
        )
        row = cur.fetchone()
        if not row:
            return [], []
        try:
            conv = json.loads(row[0] or "[]")
            disp = json.loads(row[1] or "[]")
            if not isinstance(conv, list):
                conv = []
            if not isinstance(disp, list):
                disp = []
            return conv, disp
        except (json.JSONDecodeError, TypeError):
            return [], []
    finally:
        conn.close()


def _migrate_legacy_chat_to_threads(conn: sqlite3.Connection, user_id: int) -> None:
    cur = conn.execute("SELECT COUNT(*) FROM user_chat_threads WHERE user_id = ?", (int(user_id),))
    if cur.fetchone()[0] > 0:
        return
    cur = conn.execute(
        "SELECT conversation_json, display_json FROM user_chat_state WHERE user_id = ?",
        (int(user_id),),
    )
    row = cur.fetchone()
    if not row:
        return
    conv_s, disp_s = row[0] or "[]", row[1] or "[]"
    if conv_s.strip() in ("[]", "null", "") and disp_s.strip() in ("[]", "null", ""):
        return
    conn.execute(
        """INSERT INTO user_chat_threads (user_id, kind, title, conversation_json, display_json, updated_at)
           VALUES (?, ?, ?, ?, ?, datetime('now'))""",
        (int(user_id), THREAD_KIND_GENERAL, "Imported chat", conv_s, disp_s),
    )


def ensure_user_threads_migrated(root: str, user_id: int) -> None:
    if not user_id:
        return
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        _migrate_legacy_chat_to_threads(conn, user_id)
        conn.commit()
    finally:
        conn.close()


def list_user_chat_threads(
    root: str,
    user_id: int,
    *,
    kind: Optional[str] = None,
    include_archived: bool = False,
) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    ensure_user_threads_migrated(root, user_id)
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        q = (
            "SELECT id, kind, title, archived, created_at, updated_at "
            "FROM user_chat_threads WHERE user_id = ?"
        )
        params: List[Any] = [int(user_id)]
        if kind:
            q += " AND kind = ?"
            params.append(str(kind).strip().lower())
        if not include_archived:
            q += " AND archived = 0"
        q += " ORDER BY updated_at DESC"
        cur = conn.execute(q, tuple(params))
        out: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            out.append(
                {
                    "id": int(row[0]),
                    "kind": row[1] or THREAD_KIND_GENERAL,
                    "title": row[2] or "",
                    "archived": bool(row[3]),
                    "created_at": row[4] or "",
                    "updated_at": row[5] or "",
                }
            )
        return out
    finally:
        conn.close()


def create_user_chat_thread(
    root: str,
    user_id: int,
    kind: str,
    title: str = "",
) -> int:
    if not user_id:
        return 0
    ensure_user_threads_migrated(root, user_id)
    k = (kind or THREAD_KIND_GENERAL).strip().lower()
    if k not in (THREAD_KIND_GENERAL, THREAD_KIND_TUTOR):
        k = THREAD_KIND_GENERAL
    ttl = (title or "").strip() or ("Math tutor" if k == THREAD_KIND_TUTOR else "New chat")
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        conn.execute(
            """INSERT INTO user_chat_threads (user_id, kind, title, conversation_json, display_json, updated_at)
               VALUES (?, ?, ?, '[]', '[]', datetime('now'))""",
            (int(user_id), k, ttl),
        )
        conn.commit()
        tid = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        return tid
    finally:
        conn.close()


def get_user_chat_thread(
    root: str,
    user_id: int,
    thread_id: int,
) -> Optional[Dict[str, Any]]:
    if not user_id or not thread_id:
        return None
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "SELECT id, kind, title, conversation_json, display_json, archived "
            "FROM user_chat_threads WHERE id = ? AND user_id = ?",
            (int(thread_id), int(user_id)),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            conv = json.loads(row[3] or "[]")
            disp = json.loads(row[4] or "[]")
            if not isinstance(conv, list):
                conv = []
            if not isinstance(disp, list):
                disp = []
        except (json.JSONDecodeError, TypeError):
            conv, disp = [], []
        return {
            "id": int(row[0]),
            "kind": row[1] or THREAD_KIND_GENERAL,
            "title": row[2] or "",
            "conversation": conv,
            "display": disp,
            "archived": bool(row[5]),
        }
    finally:
        conn.close()


def save_user_chat_thread(
    root: str,
    user_id: int,
    thread_id: int,
    conversation: List[Dict[str, Any]],
    display: List[Dict[str, Any]],
) -> None:
    if not user_id or not thread_id:
        return
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        conn.execute(
            """UPDATE user_chat_threads SET
                 conversation_json = ?,
                 display_json = ?,
                 updated_at = datetime('now')
               WHERE id = ? AND user_id = ?""",
            (
                json.dumps(conversation or [], ensure_ascii=False),
                json.dumps(display or [], ensure_ascii=False),
                int(thread_id),
                int(user_id),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def rename_user_chat_thread(root: str, user_id: int, thread_id: int, title: str) -> bool:
    if not user_id or not thread_id:
        return False
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "UPDATE user_chat_threads SET title = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?",
            ((title or "").strip()[:200], int(thread_id), int(user_id)),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_user_chat_thread(root: str, user_id: int, thread_id: int) -> bool:
    if not user_id or not thread_id:
        return False
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "DELETE FROM user_chat_threads WHERE id = ? AND user_id = ?",
            (int(thread_id), int(user_id)),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_active_thread_id(root: str, user_id: int) -> Optional[int]:
    if not user_id:
        return None
    raw = get_user_pref(root, user_id, PREF_ACTIVE_THREAD_ID)
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def set_active_thread_id(root: str, user_id: int, thread_id: Optional[int]) -> None:
    if not user_id:
        return
    if thread_id is None or thread_id <= 0:
        set_user_pref(root, user_id, PREF_ACTIVE_THREAD_ID, "")
    else:
        set_user_pref(root, user_id, PREF_ACTIVE_THREAD_ID, str(int(thread_id)))


def pick_default_thread_id(root: str, user_id: int) -> Optional[int]:
    threads = list_user_chat_threads(root, user_id)
    if not threads:
        return None
    active = get_active_thread_id(root, user_id)
    if active and any(t["id"] == active for t in threads):
        return active
    return int(threads[0]["id"])


def save_user_memory(root: str, user_id: int, text: str) -> None:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        conn.execute(
            "INSERT INTO user_memory (user_id, memory_text, updated_at) VALUES (?, ?, datetime('now')) ON CONFLICT(user_id) DO UPDATE SET memory_text = excluded.memory_text, updated_at = datetime('now')",
            (user_id, (text or "").strip()),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_pref(root: str, user_id: int, key: str) -> Optional[str]:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute("SELECT value FROM user_prefs WHERE user_id = ? AND key = ?", (user_id, key))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_user_pref(root: str, user_id: int, key: str, value: str) -> None:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        conn.execute(
            "INSERT INTO user_prefs (user_id, key, value) VALUES (?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
            (user_id, key, value),
        )
        conn.commit()
    finally:
        conn.close()


def load_learner_profile(root: str, user_id: int) -> Tuple[str, str, str]:
    """Background, interests/goals, and topic→level lines for prompt injection."""
    if not user_id:
        return "", "", ""
    return (
        get_user_pref(root, user_id, PREF_PROFILE_ABOUT) or "",
        get_user_pref(root, user_id, PREF_PROFILE_GOALS) or "",
        get_user_pref(root, user_id, PREF_PROFILE_SKILLS) or "",
    )


def save_learner_profile(root: str, user_id: int, about: str, goals: str, skills: str) -> None:
    if not user_id:
        return
    set_user_pref(root, user_id, PREF_PROFILE_ABOUT, (about or "").strip())
    set_user_pref(root, user_id, PREF_PROFILE_GOALS, (goals or "").strip())
    set_user_pref(root, user_id, PREF_PROFILE_SKILLS, (skills or "").strip())


def save_last_subagent_context(root: str, user_id: int, task_id_value: str, context_text: str) -> None:
    """Persist last specialist task id (e.g. cad_coder) and a short user/reply snapshot."""
    if not user_id:
        return
    tid = (task_id_value or "").strip()
    ctx = (context_text or "").strip()
    if len(ctx) > LAST_SUBAGENT_CTX_MAX_CHARS:
        ctx = ctx[: LAST_SUBAGENT_CTX_MAX_CHARS - 1] + "…"
    set_user_pref(root, user_id, PREF_LAST_SUBAGENT, tid)
    set_user_pref(root, user_id, PREF_LAST_SUBAGENT_CTX, ctx)


def load_last_subagent_context(root: str, user_id: int) -> Tuple[str, str]:
    if not user_id:
        return "", ""
    return (
        get_user_pref(root, user_id, PREF_LAST_SUBAGENT) or "",
        get_user_pref(root, user_id, PREF_LAST_SUBAGENT_CTX) or "",
    )


def global_memory_path(root: str) -> str:
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "global_memory.txt")


def _bundled_default_global_memory() -> str:
    """Shipped with v1.2; copied to temp/global_memory.txt on first load if missing."""
    pkg = Path(__file__).resolve().parent.parent
    p = pkg / "default_global_memory.txt"
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            pass
    return ""


def load_global_memory(root: str) -> str:
    path = global_memory_path(root)
    existing = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = f.read().strip()
    if existing:
        return existing
    default = _bundled_default_global_memory()
    if default:
        with open(path, "w", encoding="utf-8") as f:
            f.write(default + "\n")
        return default
    return ""


def save_global_memory(root: str, text: str) -> None:
    path = global_memory_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write((text or "").strip())


def trusted_sources_path(root: str) -> str:
    base = os.path.join(root, "temp", "imagination1")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "trusted_sources.txt")


def load_trusted(root: str, starter_domains: Optional[List[str]] = None) -> List[str]:
    path = trusted_sources_path(root)
    if not os.path.exists(path):
        domains = starter_domains or [
            "reuters.com", "apnews.com", "bbc.com", "npr.org", "pbs.org",
            "nytimes.com", "washingtonpost.com", "wsj.com", "bloomberg.com", "theguardian.com",
            "who.int", "cdc.gov", "nih.gov", "ncbi.nlm.nih.gov", "nasa.gov", "noaa.gov", "usgs.gov",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(domains) + "\n")
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip().lower() for x in f if x.strip()]


# --- Topic memory (auto-extract, invisible to user) ---


def _extract_topics_simple(user_msg: str, assistant_msg: str) -> List[Tuple[str, str]]:
    """
    Extract key topics from a conversation turn using simple heuristics.
    Returns list of (topic, detail) where topic is a short key and detail is context.
    """
    combined = f"{user_msg}\n{assistant_msg}".lower()
    words = re.findall(r"[a-z0-9]{4,}", combined)
    seen = set()
    topics: List[Tuple[str, str]] = []
    for w in words:
        if w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        detail = (user_msg[:150] + "..." if len(user_msg) > 150 else user_msg).strip()
        topics.append((w, detail))
    if not topics and user_msg.strip():
        topic = user_msg[:60].strip().replace("\n", " ")
        topics.append((topic, user_msg[:300]))
    return topics[:5]


def save_topic_memory(
    root: str,
    user_id: int,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """
    After each conversation turn, extract and save topics.
    Skips guests (no signed-in user) so topic keys are not shared globally.
    """
    if user_id is None or user_id == 0:
        return
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        for topic, detail in _extract_topics_simple(user_msg, assistant_msg):
            topic_clean = topic[:100].strip()
            if not topic_clean:
                continue
            conn.execute(
                """INSERT INTO topic_memory (user_id, topic, detail, last_mentioned, mention_count)
                   VALUES (?, ?, ?, datetime('now'), 1)
                   ON CONFLICT(user_id, topic) DO UPDATE SET
                     detail = excluded.detail,
                     last_mentioned = datetime('now'),
                     mention_count = mention_count + 1""",
                (user_id, topic_clean, (detail or "")[:500]),
            )
        conn.commit()
    finally:
        conn.close()


def load_relevant_topics(root: str, user_id: Optional[int], limit: int = 10) -> str:
    """
    Load recent topic memory for prompt injection.
    Returns a formatted string for the system prompt, or empty if none.
    """
    if user_id is None or user_id == 0:
        return ""
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            """SELECT topic, detail FROM topic_memory
               WHERE user_id = ?
               ORDER BY last_mentioned DESC
               LIMIT ?""",
            (user_id, limit),
        )
        rows = cur.fetchall()
        if not rows:
            return ""
        lines = ["User context (auto — recurring topics from recent chats):"]
        for topic, detail in rows:
            if detail:
                lines.append(f"- {topic}: {detail[:200]}")
            else:
                lines.append(f"- {topic}")
        return "\n".join(lines)
    finally:
        conn.close()
