"""Settings, read from environment variables with safe local defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"
LANGUAGES = ("en", "hi", "hinglish")
LEVELS = ("beginner", "intermediate", "advanced")


def _int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(lo, min(hi, value))


def _float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(lo, min(hi, value))


@dataclass
class Settings:
    """Everything the assistant needs to know. All values have offline-safe defaults."""

    ollama_host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    home: Path = field(default_factory=lambda: Path.home() / ".local-coder")
    num_ctx: int = 4096          # small context = less RAM; raise it if you have memory
    temperature: float = 0.2     # low temperature = steadier code
    language: str = "en"
    level: str = "beginner"
    keep_alive: str = "10m"
    timeout: float = 600.0
    backend: str = "ollama"      # "ollama" or "mock" (offline demo / tests)
    think: bool | None = None    # None = model default; False = faster answers on thinking models

    @property
    def db_path(self) -> Path:
        return self.home / "history.db"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        host = os.environ.get("OLLAMA_HOST", "").strip()
        if host:
            s.ollama_host = normalise_host(host)
        s.model = os.environ.get("LOCAL_CODER_MODEL", s.model).strip() or s.model
        home = os.environ.get("LOCAL_CODER_HOME", "").strip()
        if home:
            s.home = Path(home).expanduser()
        s.num_ctx = _int("LOCAL_CODER_NUM_CTX", s.num_ctx, 1024, 262144)
        s.temperature = _float("LOCAL_CODER_TEMPERATURE", s.temperature, 0.0, 2.0)
        lang = os.environ.get("LOCAL_CODER_LANGUAGE", s.language).strip().lower()
        s.language = lang if lang in LANGUAGES else "en"
        level = os.environ.get("LOCAL_CODER_LEVEL", s.level).strip().lower()
        s.level = level if level in LEVELS else "beginner"
        backend = os.environ.get("LOCAL_CODER_BACKEND", s.backend).strip().lower()
        s.backend = backend if backend in ("ollama", "mock") else "ollama"
        think = os.environ.get("LOCAL_CODER_THINK", "").strip().lower()
        if think in ("1", "true", "on", "yes"):
            s.think = True
        elif think in ("0", "false", "off", "no"):
            s.think = False
        return s


def normalise_host(host: str) -> str:
    """Accept the same OLLAMA_HOST forms Ollama itself accepts (host, host:port, URL)."""
    from urllib.parse import urlsplit

    host = host.strip().rstrip("/")
    if not host:
        return DEFAULT_HOST
    if "://" not in host:
        host = "http://" + host
    parts = urlsplit(host)
    hostname = parts.hostname or "127.0.0.1"
    if hostname == "0.0.0.0":
        # 0.0.0.0 is a bind address, not something a client can connect to
        hostname = "127.0.0.1"
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port = parts.port or 11434
    return f"{parts.scheme}://{hostname}:{port}"
