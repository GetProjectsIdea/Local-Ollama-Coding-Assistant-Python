"""Flask UI: chat-style coding help with streaming (Server-Sent Events)."""
from __future__ import annotations

import json

from flask import Flask, Response, abort, jsonify, render_template, request, stream_with_context

from . import __version__
from .assistant import Assistant
from .config import LANGUAGES, LEVELS, Settings
from .hardware import report
from .history import History
from .ollama_client import OllamaError, make_client
from .prompts import MODES

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
MAX_FIELD = 120_000


def hostname_of(host_header: str) -> str:
    """'localhost:5050' -> 'localhost', '[::1]:5050' -> '[::1]'."""
    h = host_header.strip().lower()
    if h.startswith("["):
        return h[: h.find("]") + 1] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def create_app(settings: Settings | None = None, client=None, history: History | None = None) -> Flask:
    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 400_000
    client = client or make_client(settings)
    history = history or History(settings.db_path)
    bot = Assistant(settings, client=client, history=history)
    app.config.update(SETTINGS=settings, HISTORY=history, CLIENT=client)

    @app.before_request
    def local_only():
        # Block DNS-rebinding: a web page on another site cannot talk to us through a fake hostname.
        host = hostname_of(request.host or "")
        if host not in LOCAL_HOSTS and not app.config.get("TESTING_ANY_HOST"):
            abort(403)
        # Changes need our custom header, which a cross-site form cannot send (simple CSRF guard).
        if request.method in ("POST", "DELETE") and request.headers.get("X-Local-Coder") != "1":
            abort(403)

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return resp

    @app.get("/")
    def index():
        return render_template("index.html", modes=MODES, languages=LANGUAGES, levels=LEVELS,
                               settings=settings, version=__version__)

    @app.get("/api/health")
    def health():
        try:
            version = client.version()
            return jsonify(ok=True, backend=settings.backend, ollama=version, host=settings.ollama_host)
        except OllamaError as exc:
            return jsonify(ok=False, backend=settings.backend, error=exc.args[0], hint=exc.hint), 503

    @app.get("/api/models")
    def models():
        try:
            items = client.list_models()
        except OllamaError as exc:
            return jsonify(models=[], default=settings.model, error=exc.args[0], hint=exc.hint), 503
        return jsonify(models=items, default=settings.model)

    @app.get("/api/hardware")
    def hardware():
        return jsonify(report())

    @app.get("/api/sessions")
    def sessions():
        q = request.args.get("q", "")[:100]
        return jsonify(sessions=history.list_sessions(limit=50, query=q))

    @app.get("/api/sessions/<int:sid>")
    def session(sid: int):
        s = history.get_session(sid)
        if not s:
            abort(404)
        return jsonify(s)

    @app.delete("/api/sessions/<int:sid>")
    def delete_session(sid: int):
        if not history.delete_session(sid):
            abort(404)
        return jsonify(ok=True)

    @app.get("/api/sessions/<int:sid>/export.md")
    def export(sid: int):
        try:
            md = history.export_markdown(sid)
        except KeyError:
            abort(404)
        return Response(md, mimetype="text/markdown",
                        headers={"Content-Disposition": f'attachment; filename="local-coder-session-{sid}.md"'})

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="send JSON"), 400
        mode = str(data.get("mode", "chat"))
        if mode not in MODES:
            return jsonify(error=f"unknown mode {mode}"), 400
        fields = {k: str(data.get(k) or "") for k in ("code", "question", "error", "filename")}
        if any(len(v) > MAX_FIELD for v in fields.values()):
            return jsonify(error="input too large (limit 120 KB per field)"), 413
        if not (fields["code"].strip() or fields["question"].strip() or fields["error"].strip()):
            return jsonify(error="type a question or paste some code"), 400
        language = data.get("language") if data.get("language") in LANGUAGES else settings.language
        level = data.get("level") if data.get("level") in LEVELS else settings.level
        model = str(data.get("model") or settings.model)[:200]
        sid = data.get("session_id")
        sid = int(sid) if isinstance(sid, int) or (isinstance(sid, str) and sid.isdigit()) else None
        think = data.get("think") if isinstance(data.get("think"), bool) else None

        def events():
            for ev in bot.stream(mode=mode, code=fields["code"], question=fields["question"],
                                 error=fields["error"], filename=fields["filename"] or None, model=model,
                                 language=language, level=level, session_id=sid, think=think):
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

        return Response(stream_with_context(events()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def wsgi_app():  # pragma: no cover - for `flask --app local_coder.web:wsgi_app run`
    return create_app()
