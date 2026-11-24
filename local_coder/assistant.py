"""The assistant: builds prompts, keeps history, streams answers, checks code blocks."""
from __future__ import annotations

import time
from typing import Callable, Iterator

from .codeutils import check_blocks
from .config import Settings
from .context import fit_messages
from .history import History
from .ollama_client import make_client
from .prompts import MODES, system_prompt, user_prompt


def session_title(mode: str, question: str = "", filename: str | None = None, code: str = "", error: str = "") -> str:
    """A readable title for the history list."""
    label = MODES.get(mode, mode)
    if question.strip():
        return question.strip() if mode in ("chat", "generate") else f"{label}: {question.strip()}"
    err = error.strip().splitlines()[-1].strip() if error.strip() else ""
    if filename:
        return f"{label}: {filename}" + (f" ({err[:40]})" if err else "")
    first = next((ln.strip() for ln in code.splitlines() if ln.strip()), "")
    return f"{label}: {first or err or 'untitled'}"


class Assistant:
    def __init__(self, settings: Settings, client=None, history: History | None = None):
        self.settings = settings
        self.client = client or make_client(settings)
        self.history = history

    def build_messages(self, mode: str, code: str = "", question: str = "", error: str = "",
                       filename: str | None = None, language: str | None = None, level: str | None = None,
                       session_id: int | None = None) -> tuple[list[dict], str, dict]:
        """Return (messages for the model, the user text to save, meta)."""
        language = language or self.settings.language
        level = level or self.settings.level
        budget = max(400, int(self.settings.num_ctx * 0.55))
        text, meta = user_prompt(mode, code, question, error, filename, code_budget_tokens=budget)
        system = {"role": "system", "content": system_prompt(mode, language, level, meta.get("code_language", ""))}
        past = self.history.chat_messages(session_id) if (self.history and session_id) else []
        reserve = min(1024, self.settings.num_ctx // 3)
        messages, dropped = fit_messages([system, *past, {"role": "user", "content": text}],
                                         self.settings.num_ctx, reserve)
        meta["dropped_messages"] = dropped
        return messages, text, meta

    def stream(self, mode: str = "chat", code: str = "", question: str = "", error: str = "",
               filename: str | None = None, model: str | None = None, language: str | None = None,
               level: str | None = None, session_id: int | None = None, save: bool = True,
               think: bool | None = None) -> Iterator[dict]:
        """Yield events: session, token, thinking, done (with stats + code checks) or error."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(MODES)}")
        model = model or self.settings.model
        language = language or self.settings.language
        think = self.settings.think if think is None else think
        messages, user_text, meta = self.build_messages(mode, code, question, error, filename, language,
                                                        level, session_id)
        if self.history and save:
            if not session_id or not self.history.session_exists(session_id):
                session_id = self.history.create_session(
                    session_title(mode, question, filename, code, error), mode, model, language)
            self.history.add_message(session_id, "user", user_text, {"mode": mode, "model": model})
        yield {"type": "session", "session_id": session_id, "model": model, "mode": mode, "meta": meta}
        parts: list[str] = []
        started = time.time()
        stats: dict = {}
        try:
            for event in self.client.chat_stream(model, messages, think=think):
                if event["type"] == "token":
                    parts.append(event["text"])
                    yield event
                elif event["type"] == "thinking":
                    yield event
                elif event["type"] == "done":
                    stats = event.get("stats", {})
        except Exception as exc:  # noqa: BLE001 - shown to the student, not swallowed
            hint = getattr(exc, "hint", "")
            yield {"type": "error", "message": str(exc.args[0] if exc.args else exc), "hint": hint}
            if self.history and save and session_id and parts:
                self.history.add_message(session_id, "assistant", "".join(parts) + "\n\n[stopped: error]")
            return
        answer = "".join(parts)
        stats["wall_seconds"] = round(time.time() - started, 2)
        checks = check_blocks(answer)
        if self.history and save and session_id:
            self.history.add_message(session_id, "assistant", answer,
                                     {"model": model, "stats": stats, "checks": checks})
        yield {"type": "done", "answer": answer, "stats": stats, "checks": checks, "session_id": session_id}

    def ask(self, on_token: Callable[[str], None] | None = None, **kwargs) -> dict:
        """Non-streaming helper: run stream() and return the final event (or the error)."""
        result: dict = {}
        for event in self.stream(**kwargs):
            if event["type"] == "token" and on_token:
                on_token(event["text"])
            elif event["type"] == "session":
                result["session_id"] = event["session_id"]
            elif event["type"] in ("done", "error"):
                result.update(event)
        return result
