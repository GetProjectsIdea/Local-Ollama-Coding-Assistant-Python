import pytest

from local_coder.config import Settings
from local_coder.ollama_client import MockClient, OllamaClient, OllamaError, make_client, stats_from_final


def client_for(url, **kw):
    return OllamaClient(Settings(ollama_host=url, **kw))


def test_version_models_ps(fake_ollama):
    url, _ = fake_ollama
    c = client_for(url)
    assert c.version() == "0.40.16"
    models = c.list_models()
    assert [m["name"] for m in models] == ["gemma4-e2b-q4km:latest", "qwen2.5-coder:3b"]
    assert models[1]["size_gb"] == 1.9 and models[1]["quantization"] == "Q4_K_M"
    assert c.running()[0]["name"] == "qwen2.5-coder:3b"


def test_chat_stream_tokens_and_stats(fake_ollama):
    url, fake = fake_ollama
    c = client_for(url, num_ctx=2048, temperature=0.1)
    events = list(c.chat_stream("qwen2.5-coder:3b", [{"role": "user", "content": "hi"}]))
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello student!"
    done = events[-1]
    assert done["type"] == "done"
    assert done["stats"]["tokens_per_second"] == 15.0
    assert done["stats"]["load_seconds"] == 1.5
    sent = fake.requests[-1]["payload"]
    assert sent["stream"] is True
    assert sent["options"] == {"num_ctx": 2048, "temperature": 0.1}
    assert "think" not in sent


def test_think_flag_is_sent_only_when_set(fake_ollama):
    url, fake = fake_ollama
    c = client_for(url)
    events = list(c.chat_stream("m", [{"role": "user", "content": "x"}], think=True))
    assert fake.requests[-1]["payload"]["think"] is True
    assert events[0] == {"type": "thinking", "text": "hmm"}
    list(c.chat_stream("m", [{"role": "user", "content": "x"}], think=False))
    assert fake.requests[-1]["payload"]["think"] is False


def test_missing_model_gives_pull_hint(fake_ollama):
    url, _ = fake_ollama
    with pytest.raises(OllamaError) as exc:
        list(client_for(url).chat_stream("missing:1b", [{"role": "user", "content": "x"}]))
    assert "ollama pull missing:1b" in exc.value.hint


def test_stream_error_line(fake_ollama):
    url, _ = fake_ollama
    with pytest.raises(OllamaError, match="out of memory"):
        list(client_for(url).chat_stream("broken:1b", [{"role": "user", "content": "x"}]))


def test_unreachable_server_hint():
    c = client_for("http://127.0.0.1:9", timeout=2)
    with pytest.raises(OllamaError) as exc:
        c.version()
    assert "ollama serve" in exc.value.hint


def test_stats_from_final_handles_missing():
    st = stats_from_final({})
    assert st["tokens_per_second"] == 0.0 and st["output_tokens"] == 0


def test_make_client():
    assert isinstance(make_client(Settings(backend="mock")), MockClient)
    assert isinstance(make_client(Settings()), OllamaClient)


def test_mock_answers_by_mode():
    m = MockClient()
    for mode, needle in [("explain", "What this code does"), ("fix", "Fixed code"), ("tests", "pytest"),
                         ("docstring", "Args:"), ("generate", "is_palindrome"), ("review", "Review")]:
        sys = {"role": "system", "content": f"x\nMODE: {mode}\nLANGUAGE: en"}
        text = "".join(e.get("text", "") for e in m.chat_stream("x", [sys]))
        assert needle in text
    sys = {"role": "system", "content": "x\nMODE: explain\nLANGUAGE: hi"}
    assert "कोड" in "".join(e.get("text", "") for e in m.chat_stream("x", [sys]))
    assert len(m.calls) == 7
