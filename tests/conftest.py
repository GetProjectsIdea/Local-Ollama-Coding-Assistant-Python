import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from local_coder.config import Settings
from local_coder.history import History
from local_coder.ollama_client import MockClient


class FakeOllama(BaseHTTPRequestHandler):
    """A fake Ollama server that streams NDJSON like the real one."""

    models = [
        {"name": "qwen2.5-coder:3b", "size": 1_900_000_000,
         "details": {"family": "qwen2", "parameter_size": "3.1B", "quantization_level": "Q4_K_M"}},
        {"name": "gemma4-e2b-q4km:latest", "size": 3_100_000_000,
         "details": {"family": "gemma4", "parameter_size": "4.6B", "quantization_level": "Q4_K_M"}},
    ]
    requests: list = []
    reply = ["Hello", " student", "!"]

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send(200, {"version": "0.40.16"})
        if self.path == "/api/tags":
            return self._send(200, {"models": self.models})
        if self.path == "/api/ps":
            return self._send(200, {"models": [{"name": "qwen2.5-coder:3b", "size": 2_000_000_000, "size_vram": 0}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        FakeOllama.requests.append({"path": self.path, "payload": payload})
        if self.path != "/api/chat":
            return self._send(404, {"error": "not found"})
        if payload.get("model") == "missing:1b":
            return self._send(404, {"error": "model 'missing:1b' not found"})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        if payload.get("model") == "broken:1b":
            self.wfile.write(json.dumps({"error": "out of memory"}).encode() + b"\n")
            return
        if payload.get("think"):
            self.wfile.write(json.dumps({"message": {"role": "assistant", "content": "", "thinking": "hmm"}}).encode() + b"\n")
        for word in self.reply:
            self.wfile.write(json.dumps({"message": {"role": "assistant", "content": word}, "done": False}).encode() + b"\n")
            self.wfile.flush()
        final = {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                 "eval_count": 30, "eval_duration": 2_000_000_000, "prompt_eval_count": 100,
                 "prompt_eval_duration": 500_000_000, "load_duration": 1_500_000_000, "total_duration": 4_000_000_000}
        self.wfile.write(json.dumps(final).encode() + b"\n")


@pytest.fixture
def fake_ollama():
    FakeOllama.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", FakeOllama
    server.shutdown()
    server.server_close()


@pytest.fixture
def settings(tmp_path):
    return Settings(backend="mock", model="mock-coder:tiny", home=tmp_path)


@pytest.fixture
def history(tmp_path):
    h = History(tmp_path / "h.db")
    yield h
    h.close()


@pytest.fixture
def mock():
    return MockClient()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("OLLAMA_HOST", "LOCAL_CODER_MODEL", "LOCAL_CODER_HOME", "LOCAL_CODER_NUM_CTX",
                "LOCAL_CODER_TEMPERATURE", "LOCAL_CODER_LANGUAGE", "LOCAL_CODER_LEVEL",
                "LOCAL_CODER_BACKEND", "LOCAL_CODER_THINK", "FORCE_COLOR"):
        monkeypatch.delenv(key, raising=False)
