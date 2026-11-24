import json

import pytest

from local_coder.config import Settings
from local_coder.history import History
from local_coder.ollama_client import MockClient, OllamaClient
from local_coder.web import create_app, hostname_of

H = {"X-Local-Coder": "1"}
CODE = "def average(nums):\n    return sum(nums) / len(nums)\n"


@pytest.fixture
def app(tmp_path):
    s = Settings(backend="mock", model="mock-coder:tiny", home=tmp_path)
    return create_app(s, client=MockClient(), history=History(s.db_path))


@pytest.fixture
def client(app):
    return app.test_client()


def sse(resp):
    events = []
    for block in resp.get_data(as_text=True).strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.mark.parametrize("raw,host", [("localhost:5050", "localhost"), ("127.0.0.1", "127.0.0.1"),
                                      ("[::1]:5050", "[::1]"), ("Evil.com:80", "evil.com")])
def test_hostname_of(raw, host):
    assert hostname_of(raw) == host


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    for text in ("Local Coder", "Fix an error", "Write unit tests", "हिंदी", "Hinglish", "Saved chats" if False else "New chat"):
        assert text in html
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Frame-Options"] == "DENY"


def test_static_assets(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_dns_rebinding_blocked(client):
    assert client.get("/", headers={"Host": "attacker.example"}).status_code == 403
    assert client.get("/api/models", headers={"Host": "localhost:5050"}).status_code == 200


def test_post_needs_custom_header(client):
    r = client.post("/api/chat", json={"mode": "chat", "question": "hi"})
    assert r.status_code == 403


def test_health_and_models(client):
    j = client.get("/api/health").get_json()
    assert j["ok"] is True and j["backend"] == "mock"
    j = client.get("/api/models").get_json()
    assert j["models"][0]["name"] == "mock-coder:tiny" and j["default"] == "mock-coder:tiny"


def test_health_when_ollama_down(tmp_path):
    s = Settings(ollama_host="http://127.0.0.1:9", home=tmp_path, timeout=2)
    c = create_app(s, client=OllamaClient(s)).test_client()
    r = c.get("/api/health")
    assert r.status_code == 503 and "ollama serve" in r.get_json()["hint"]
    r = c.get("/api/models")
    assert r.status_code == 503 and r.get_json()["models"] == []


def test_hardware_endpoint(client):
    j = client.get("/api/hardware").get_json()
    assert "tier" in j and "best" in j


def test_chat_streams_sse(client):
    r = client.post("/api/chat", headers=H, json={"mode": "fix", "code": CODE, "error": "ZeroDivisionError: x",
                                                  "filename": "a.py", "language": "en"})
    assert r.status_code == 200 and r.mimetype == "text/event-stream"
    events = sse(r)
    assert events[0][0] == "session" and events[-1][0] == "done"
    tokens = "".join(d["text"] for e, d in events if e == "token")
    assert tokens == events[-1][1]["answer"]
    assert events[-1][1]["checks"][0]["ok"] is True


def test_chat_validation(client):
    assert client.post("/api/chat", headers=H, data="nope", content_type="text/plain").status_code == 400
    assert client.post("/api/chat", headers=H, json={"mode": "hack", "question": "x"}).status_code == 400
    assert client.post("/api/chat", headers=H, json={"mode": "chat"}).status_code == 400
    r = client.post("/api/chat", headers=H, json={"mode": "chat", "code": "x" * 130_000})
    assert r.status_code == 413


def test_body_limit(client):
    r = client.post("/api/chat", headers={**H, "Content-Type": "application/json"}, data="x" * 500_000)
    assert r.status_code == 413


def test_sessions_api(client):
    events = sse(client.post("/api/chat", headers=H, json={"mode": "explain", "code": CODE, "language": "hi"}))
    sid = events[0][1]["session_id"]
    sse(client.post("/api/chat", headers=H, json={"mode": "chat", "question": "more?", "session_id": sid}))
    lst = client.get("/api/sessions").get_json()["sessions"]
    assert lst[0]["id"] == sid and lst[0]["n"] == 4
    s = client.get(f"/api/sessions/{sid}").get_json()
    assert s["language"] == "hi" and len(s["messages"]) == 4
    assert client.get("/api/sessions?q=more").get_json()["sessions"][0]["id"] == sid
    md = client.get(f"/api/sessions/{sid}/export.md")
    assert md.status_code == 200 and "attachment" in md.headers["Content-Disposition"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 403  # no header
    assert client.delete(f"/api/sessions/{sid}", headers=H).status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/sessions/{sid}", headers=H).status_code == 404
    assert client.get(f"/api/sessions/{sid}/export.md").status_code == 404


def test_bad_language_falls_back(client, app):
    events = sse(client.post("/api/chat", headers=H, json={"mode": "chat", "question": "x", "language": "xx",
                                                          "session_id": "abc"}))
    assert events[-1][0] == "done"
