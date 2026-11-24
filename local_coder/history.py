"""Chat history saved locally in SQLite (~/.local-coder/history.db). Nothing leaves the laptop."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    mode TEXT NOT NULL,
    model TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def make_title(text: str, limit: int = 60) -> str:
    first = " ".join((text or "").split())
    for prefix in ("Task:", "Question:", "Code:"):
        if first.startswith(prefix):
            first = first[len(prefix):].strip()
    first = first.replace("```", "").strip() or "New chat"
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


class History:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.db.commit()
        if str(self.path) != ":memory:":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    def close(self) -> None:
        self.db.close()

    # sessions -----------------------------------------------------------
    def create_session(self, title: str, mode: str, model: str, language: str = "en") -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO sessions(title, mode, model, language, created, updated) VALUES (?,?,?,?,?,?)",
            (make_title(title), mode, model, language, now, now))
        self.db.commit()
        return int(cur.lastrowid)

    def session_exists(self, session_id: int) -> bool:
        return self.db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is not None

    def list_sessions(self, limit: int = 50, query: str = "") -> list[dict]:
        sql = ("SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS n FROM sessions s")
        args: list = []
        if query:
            sql += (" WHERE s.title LIKE ? OR EXISTS (SELECT 1 FROM messages m WHERE m.session_id=s.id "
                    "AND m.content LIKE ?)")
            args += [f"%{query}%", f"%{query}%"]
        sql += " ORDER BY s.updated DESC, s.id DESC LIMIT ?"
        args.append(int(limit))
        return [dict(r) for r in self.db.execute(sql, args)]

    def get_session(self, session_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            return None
        msgs = [dict(r) | {"meta": json.loads(r["meta"] or "{}")} for r in self.db.execute(
            "SELECT id, role, content, meta, created FROM messages WHERE session_id=? ORDER BY id", (session_id,))]
        return dict(row) | {"messages": msgs}

    def delete_session(self, session_id: int) -> bool:
        cur = self.db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self.db.commit()
        return cur.rowcount > 0

    def clear(self) -> int:
        cur = self.db.execute("DELETE FROM sessions")
        self.db.commit()
        return cur.rowcount

    # messages -----------------------------------------------------------
    def add_message(self, session_id: int, role: str, content: str, meta: dict | None = None) -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO messages(session_id, role, content, meta, created) VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(meta or {}, ensure_ascii=False), now))
        self.db.execute("UPDATE sessions SET updated=? WHERE id=?", (now, session_id))
        self.db.commit()
        return int(cur.lastrowid)

    def chat_messages(self, session_id: int) -> list[dict]:
        """Messages in Ollama format (role/content) for continuing a conversation."""
        return [{"role": r["role"], "content": r["content"]} for r in self.db.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id", (session_id,))]

    def stats(self) -> dict:
        s = self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        m = self.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"sessions": s, "messages": m, "bytes": size, "path": str(self.path)}

    def export_markdown(self, session_id: int) -> str:
        s = self.get_session(session_id)
        if not s:
            raise KeyError(session_id)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["created"]))
        out = [f"# {s['title']}", "",
               f"- Mode: {s['mode']}", f"- Model: {s['model']}", f"- Language: {s['language']}",
               f"- Started: {when}", ""]
        for m in s["messages"]:
            out += [f"## {'You' if m['role'] == 'user' else 'Assistant'}", "", m["content"], ""]
            stats = m["meta"].get("stats") if isinstance(m["meta"], dict) else None
            if stats and stats.get("tokens_per_second"):
                out += [f"_{stats.get('output_tokens', 0)} tokens at {stats['tokens_per_second']} tokens/s_", ""]
        return "\n".join(out).rstrip() + "\n"
