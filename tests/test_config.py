import pytest

from local_coder.config import DEFAULT_HOST, Settings, normalise_host


@pytest.mark.parametrize("raw,expected", [
    ("", DEFAULT_HOST),
    ("127.0.0.1", "http://127.0.0.1:11434"),
    ("localhost:8080", "http://localhost:8080"),
    ("0.0.0.0", "http://127.0.0.1:11434"),
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    ("http://192.168.1.5:11434/", "http://192.168.1.5:11434"),
    ("https://ollama.lan", "https://ollama.lan:11434"),
    ("[::1]:9000", "http://[::1]:9000"),
])
def test_normalise_host(raw, expected):
    assert normalise_host(raw) == expected


def test_defaults_are_low_ram_friendly():
    s = Settings()
    assert s.num_ctx == 4096
    assert s.model == "qwen2.5-coder:3b"
    assert s.temperature <= 0.3
    assert s.backend == "ollama"


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    monkeypatch.setenv("LOCAL_CODER_MODEL", "gemma4:e2b")
    monkeypatch.setenv("LOCAL_CODER_HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_CODER_NUM_CTX", "8192")
    monkeypatch.setenv("LOCAL_CODER_LANGUAGE", "hi")
    monkeypatch.setenv("LOCAL_CODER_THINK", "off")
    s = Settings.from_env()
    assert s.ollama_host == "http://127.0.0.1:11434"
    assert s.model == "gemma4:e2b"
    assert s.db_path == tmp_path / "history.db"
    assert s.num_ctx == 8192
    assert s.language == "hi"
    assert s.think is False


def test_from_env_rejects_bad_values(monkeypatch):
    monkeypatch.setenv("LOCAL_CODER_NUM_CTX", "lots")
    monkeypatch.setenv("LOCAL_CODER_TEMPERATURE", "99")
    monkeypatch.setenv("LOCAL_CODER_LANGUAGE", "fr")
    monkeypatch.setenv("LOCAL_CODER_BACKEND", "cloud")
    s = Settings.from_env()
    assert s.num_ctx == 4096
    assert s.temperature == 2.0
    assert s.language == "en"
    assert s.backend == "ollama"


def test_num_ctx_is_clamped(monkeypatch):
    monkeypatch.setenv("LOCAL_CODER_NUM_CTX", "10")
    assert Settings.from_env().num_ctx == 1024
