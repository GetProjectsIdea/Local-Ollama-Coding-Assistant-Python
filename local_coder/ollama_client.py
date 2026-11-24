"""A tiny Ollama client (standard library only) plus an offline mock backend.

Only the read-only and chat endpoints are used: /api/version, /api/tags,
/api/ps and /api/chat. The assistant never pulls, deletes or copies models.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Iterator

from .config import Settings


class OllamaError(RuntimeError):
    """A problem talking to Ollama, with a student-friendly hint."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = super().__str__()
        return f"{base} (hint: {self.hint})" if self.hint else base


def stats_from_final(chunk: dict) -> dict:
    """Turn Ollama's final chunk into human numbers (durations are nanoseconds)."""
    ns = 1_000_000_000
    eval_count = int(chunk.get("eval_count") or 0)
    eval_dur = int(chunk.get("eval_duration") or 0)
    prompt_count = int(chunk.get("prompt_eval_count") or 0)
    prompt_dur = int(chunk.get("prompt_eval_duration") or 0)
    return {
        "output_tokens": eval_count,
        "prompt_tokens": prompt_count,
        "tokens_per_second": round(eval_count / (eval_dur / ns), 1) if eval_dur else 0.0,
        "prompt_tokens_per_second": round(prompt_count / (prompt_dur / ns), 1) if prompt_dur else 0.0,
        "load_seconds": round(int(chunk.get("load_duration") or 0) / ns, 2),
        "total_seconds": round(int(chunk.get("total_duration") or 0) / ns, 2),
        "done_reason": chunk.get("done_reason") or "stop",
    }


class OllamaClient:
    """Talks to a local Ollama server over HTTP."""

    name = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base = settings.ollama_host.rstrip("/")

    # -- low level -------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method="POST" if payload is not None else "GET",
            headers={"Content-Type": "application/json", "User-Agent": "local-coder/1.0"},
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.settings.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                msg = json.loads(body).get("error", body)
            except ValueError:
                msg = body or exc.reason
            hint = ""
            if exc.code == 404 and "not found" in str(msg).lower():
                model = (payload or {}).get("model", "the model")
                hint = f"download it first with: ollama pull {model}"
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {msg}", hint) from None
        except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise OllamaError(
                f"Cannot reach Ollama at {self.base} ({reason})",
                "start it with 'ollama serve' (or open the Ollama app) and try again",
            ) from None

    def _json(self, path: str, timeout: float = 10.0) -> dict:
        with self._request(path, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")

    # -- read-only info ----------------------------------------------------
    def version(self) -> str:
        return str(self._json("/api/version").get("version", "unknown"))

    def list_models(self) -> list[dict]:
        models = self._json("/api/tags").get("models") or []
        out = []
        for m in models:
            details = m.get("details") or {}
            out.append({
                "name": m.get("name") or m.get("model"),
                "size_gb": round((m.get("size") or 0) / 1e9, 2),
                "family": details.get("family", ""),
                "parameters": details.get("parameter_size", ""),
                "quantization": details.get("quantization_level", ""),
            })
        return sorted(out, key=lambda m: m["name"] or "")

    def running(self) -> list[dict]:
        models = self._json("/api/ps").get("models") or []
        return [{"name": m.get("name"), "size_gb": round((m.get("size") or 0) / 1e9, 2),
                 "vram_gb": round((m.get("size_vram") or 0) / 1e9, 2)} for m in models]

    # -- chat --------------------------------------------------------------
    def chat_stream(self, model: str, messages: list[dict], options: dict | None = None,
                    think: bool | None = None) -> Iterator[dict]:
        """Yield {"type": "token"|"thinking"|"done", ...} events while Ollama streams."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.settings.keep_alive,
            "options": {"num_ctx": self.settings.num_ctx, "temperature": self.settings.temperature,
                        **(options or {})},
        }
        if think is not None:
            payload["think"] = think
        with self._request("/api/chat", payload) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise OllamaError(f"Ollama error: {chunk['error']}")
                msg = chunk.get("message") or {}
                if msg.get("thinking"):
                    yield {"type": "thinking", "text": msg["thinking"]}
                if msg.get("content"):
                    yield {"type": "token", "text": msg["content"]}
                if chunk.get("done"):
                    yield {"type": "done", "stats": stats_from_final(chunk)}
                    return
        yield {"type": "done", "stats": stats_from_final({})}


# ---------------------------------------------------------------------------
# Offline mock backend: deterministic answers so the demo and tests run anywhere
# ---------------------------------------------------------------------------
MOCK_MODELS = [
    {"name": "mock-coder:tiny", "size_gb": 0.0, "family": "mock", "parameters": "0B", "quantization": "none"},
]

MOCK_ANSWERS = {
    "explain": (
        "**What this code does**\n\n"
        "1. `def average(nums)` defines a function that takes a list of numbers.\n"
        "2. `sum(nums)` adds them and `len(nums)` counts them.\n"
        "3. It returns the total divided by the count, i.e. the mean.\n\n"
        "**Watch out:** an empty list makes `len(nums)` zero, so Python raises `ZeroDivisionError`."
    ),
    "explain_hi": (
        "**यह कोड क्या करता है**\n\n"
        "1. `def average(nums)` एक function बनाता है जो numbers की list लेता है।\n"
        "2. `sum(nums)` सबको जोड़ता है और `len(nums)` गिनती करता है।\n"
        "3. यह total को count से divide करके average लौटाता है।\n\n"
        "**ध्यान दें:** खाली list पर `ZeroDivisionError` आएगा।"
    ),
    "generate": (
        "Here is a simple, commented solution:\n\n"
        "```python\n"
        "def is_palindrome(text: str) -> bool:\n"
        "    \"\"\"Return True if text reads the same forwards and backwards.\"\"\"\n"
        "    cleaned = \"\".join(ch.lower() for ch in text if ch.isalnum())\n"
        "    return cleaned == cleaned[::-1]\n\n\n"
        "print(is_palindrome(\"Nitin\"))  # True\n"
        "```\n\n"
        "It ignores spaces, punctuation and capital letters."
    ),
    "fix": (
        "**Why it fails:** `average([])` divides by `len([])`, which is 0, so Python raises "
        "`ZeroDivisionError` on line 2.\n\n"
        "**Fixed code:**\n\n"
        "```python\n"
        "def average(nums):\n"
        "    if not nums:\n"
        "        return 0.0\n"
        "    return sum(nums) / len(nums)\n"
        "```\n\n"
        "**Tip:** always decide what an empty input should return."
    ),
    "docstring": (
        "```python\n"
        "def average(nums):\n"
        "    \"\"\"Return the arithmetic mean of a list of numbers.\n\n"
        "    Args:\n"
        "        nums: A non-empty list of int or float values.\n\n"
        "    Returns:\n"
        "        float: The average value.\n\n"
        "    Raises:\n"
        "        ZeroDivisionError: If ``nums`` is empty.\n"
        "    \"\"\"\n"
        "    return sum(nums) / len(nums)\n"
        "```"
    ),
    "tests": (
        "```python\n"
        "import pytest\n"
        "from average import average\n\n\n"
        "def test_average_of_integers():\n"
        "    assert average([2, 4, 6]) == 4\n\n\n"
        "def test_average_of_floats():\n"
        "    assert average([1.5, 2.5]) == pytest.approx(2.0)\n\n\n"
        "def test_empty_list_raises():\n"
        "    with pytest.raises(ZeroDivisionError):\n"
        "        average([])\n"
        "```"
    ),
    "review": (
        "**Review**\n\n"
        "- Works for normal lists.\n"
        "- Missing: empty-list handling and a docstring.\n"
        "- Naming is clear. Consider type hints: `def average(nums: list[float]) -> float`."
    ),
    "chat": "Sure! Ask me about any code, error or concept and I will explain it step by step.",
}


class MockClient:
    """Pretends to be Ollama. Used by `--backend mock`, the demo and the test-suite."""

    name = "mock"

    def __init__(self, settings: Settings | None = None, delay: float = 0.0):
        self.settings = settings or Settings(backend="mock")
        self.delay = delay
        self.calls: list[dict] = []

    def version(self) -> str:
        return "mock"

    def list_models(self) -> list[dict]:
        return [dict(m) for m in MOCK_MODELS]

    def running(self) -> list[dict]:
        return []

    @staticmethod
    def pick_answer(messages: list[dict]) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        tags = {}
        for line in system.splitlines():
            if line.startswith(("MODE:", "LANGUAGE:")):
                key, _, value = line.partition(":")
                tags[key.strip()] = value.strip()
        mode = tags.get("MODE", "chat")
        if mode == "explain" and tags.get("LANGUAGE") == "hi":
            return MOCK_ANSWERS["explain_hi"]
        return MOCK_ANSWERS.get(mode, MOCK_ANSWERS["chat"])

    def chat_stream(self, model: str, messages: list[dict], options: dict | None = None,
                    think: bool | None = None) -> Iterator[dict]:
        self.calls.append({"model": model, "messages": messages, "options": options, "think": think})
        answer = self.pick_answer(messages)
        words = answer.split(" ")
        for i, word in enumerate(words):
            if self.delay:
                time.sleep(self.delay)
            yield {"type": "token", "text": word + (" " if i < len(words) - 1 else "")}
        yield {"type": "done", "stats": {
            "output_tokens": len(words), "prompt_tokens": sum(len(m["content"]) // 4 for m in messages),
            "tokens_per_second": 0.0, "prompt_tokens_per_second": 0.0,
            "load_seconds": 0.0, "total_seconds": 0.0, "done_reason": "stop"}}


def make_client(settings: Settings):
    return MockClient(settings) if settings.backend == "mock" else OllamaClient(settings)
