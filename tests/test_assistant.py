import pytest

from local_coder.assistant import Assistant
from local_coder.config import Settings
from local_coder.ollama_client import MockClient, OllamaError

CODE = "def average(nums):\n    return sum(nums) / len(nums)\n"


def test_stream_event_order_and_saving(settings, history):
    bot = Assistant(settings, client=MockClient(), history=history)
    events = list(bot.stream(mode="fix", code=CODE, error="ZeroDivisionError: division by zero"))
    types = [e["type"] for e in events]
    assert types[0] == "session" and types[-1] == "done" and "token" in types
    done = events[-1]
    assert "Fixed code" in done["answer"]
    assert done["checks"][0]["ok"] is True
    s = history.get_session(done["session_id"])
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]
    assert s["messages"][1]["meta"]["checks"][0]["language"] == "python"


def test_continue_session_sends_history(settings, history):
    mock = MockClient()
    bot = Assistant(settings, client=mock, history=history)
    first = bot.ask(mode="chat", question="what is a list?")
    bot.ask(mode="chat", question="and a tuple?", session_id=first["session_id"])
    sent = mock.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert "tuple" in sent[-1]["content"]


def test_unknown_session_creates_new(settings, history):
    bot = Assistant(settings, client=MockClient(), history=history)
    res = bot.ask(mode="chat", question="hi", session_id=12345)
    assert res["session_id"] != 12345 and history.session_exists(res["session_id"])


def test_no_save(settings, history):
    bot = Assistant(settings, client=MockClient(), history=history)
    res = bot.ask(mode="explain", code=CODE, save=False)
    assert res["type"] == "done" and res["session_id"] is None
    assert history.list_sessions() == []


def test_without_history(settings):
    bot = Assistant(settings, client=MockClient())
    res = bot.ask(mode="explain", code=CODE)
    assert "What this code does" in res["answer"]


def test_language_and_model_passed(settings, history):
    mock = MockClient()
    bot = Assistant(settings, client=mock, history=history)
    res = bot.ask(mode="explain", code=CODE, language="hi", model="gemma4:e2b")
    assert "कोड" in res["answer"]
    assert mock.calls[-1]["model"] == "gemma4:e2b"
    assert "LANGUAGE: hi" in mock.calls[-1]["messages"][0]["content"]


def test_think_setting_used(settings):
    mock = MockClient()
    settings.think = False
    Assistant(settings, client=mock).ask(mode="chat", question="x")
    assert mock.calls[-1]["think"] is False
    Assistant(settings, client=mock).ask(mode="chat", question="x", think=True)
    assert mock.calls[-1]["think"] is True


def test_unknown_mode():
    with pytest.raises(ValueError):
        list(Assistant(Settings(backend="mock")).stream(mode="hack"))


def test_error_event_and_partial_save(settings, history):
    class Broken(MockClient):
        def chat_stream(self, *a, **k):
            yield {"type": "token", "text": "partial "}
            raise OllamaError("Cannot reach Ollama", "start it with 'ollama serve'")

    bot = Assistant(settings, client=Broken(), history=history)
    res = bot.ask(mode="chat", question="hi")
    assert res["type"] == "error" and "ollama serve" in res["hint"]
    msgs = history.get_session(res["session_id"])["messages"]
    assert msgs[-1]["content"].startswith("partial") and "[stopped: error]" in msgs[-1]["content"]


def test_on_token_callback(settings):
    got = []
    Assistant(settings, client=MockClient()).ask(on_token=got.append, mode="chat", question="x")
    assert "".join(got).startswith("Sure!")


def test_long_history_is_trimmed(settings, history):
    settings.num_ctx = 1024
    mock = MockClient()
    bot = Assistant(settings, client=mock, history=history)
    sid = history.create_session("long", "chat", "m")
    for i in range(30):
        history.add_message(sid, "user", f"q{i} " + "x" * 400)
        history.add_message(sid, "assistant", f"a{i} " + "y" * 400)
    events = list(bot.stream(mode="chat", question="last", session_id=sid))
    assert events[0]["meta"]["dropped_messages"] > 40
    sent = mock.calls[-1]["messages"]
    assert sent[0]["role"] == "system" and "last" in sent[-1]["content"]
    assert len(sent) < 10


def test_real_ollama_protocol(fake_ollama, tmp_path):
    from local_coder.history import History
    url, fake = fake_ollama
    s = Settings(ollama_host=url, home=tmp_path, model="qwen2.5-coder:3b")
    h = History(s.db_path)
    res = Assistant(s, history=h).ask(mode="explain", code=CODE)
    assert res["answer"] == "Hello student!"
    assert res["stats"]["tokens_per_second"] == 15.0
    payload = fake.requests[-1]["payload"]
    assert payload["model"] == "qwen2.5-coder:3b"
    assert payload["messages"][0]["role"] == "system"
    assert "```python" in payload["messages"][-1]["content"]
    h.close()


def test_session_titles():
    from local_coder.assistant import session_title
    assert session_title("chat", "what is a list?") == "what is a list?"
    assert session_title("explain", "why n-i-1?") == "Explain code: why n-i-1?"
    assert session_title("fix", filename="a.py", error="Traceback...\nZeroDivisionError: division by zero") == \
        "Fix an error: a.py (ZeroDivisionError: division by zero)"
    assert session_title("tests", code="\n\ndef f():\n  pass") == "Write unit tests: def f():"
    assert session_title("review") == "Review my code: untitled"
