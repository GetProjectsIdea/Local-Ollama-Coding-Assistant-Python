"""Command line interface: python -m local_coder <command>."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .assistant import Assistant
from .config import LANGUAGES, LEVELS, Settings, normalise_host
from .hardware import MODEL_CATALOG, report
from .history import History
from .ollama_client import OllamaError, make_client
from .prompts import MODES

USE_COLOR = (sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR"))) and not os.environ.get("NO_COLOR")


def c(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if USE_COLOR else text


def settings_from_args(args) -> Settings:
    s = Settings.from_env()
    if getattr(args, "host", None):
        s.ollama_host = normalise_host(args.host)
    if getattr(args, "model", None):
        s.model = args.model
    if getattr(args, "backend", None):
        s.backend = args.backend
    if getattr(args, "lang", None):
        s.language = args.lang
    if getattr(args, "level", None):
        s.level = args.level
    if getattr(args, "num_ctx", None):
        s.num_ctx = args.num_ctx
    if getattr(args, "temperature", None) is not None:
        s.temperature = args.temperature
    if getattr(args, "home", None):
        s.home = Path(args.home).expanduser()
    if getattr(args, "think", None) is not None:
        s.think = args.think
    return s


def read_input(path: str | None) -> tuple[str, str | None]:
    if not path:
        return "", None
    if path == "-":
        return sys.stdin.read(), None
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"error: file not found: {path}")
    if p.stat().st_size > 400_000:
        raise SystemExit(f"error: {path} is larger than 400 KB; pass a smaller file or a function")
    return p.read_text(encoding="utf-8", errors="replace"), p.name


def print_stats(event: dict) -> None:
    st = event.get("stats", {})
    bits = []
    if st.get("output_tokens"):
        bits.append(f"{st['output_tokens']} tokens")
    if st.get("tokens_per_second"):
        bits.append(f"{st['tokens_per_second']} tokens/s")
    if st.get("wall_seconds") is not None:
        bits.append(f"{st['wall_seconds']} s")
    if st.get("load_seconds"):
        bits.append(f"model load {st['load_seconds']} s")
    if bits:
        print(c("  [" + " · ".join(bits) + "]", "2"))
    for chk in event.get("checks", []):
        if chk["ok"] is True:
            print(c(f"  ✓ code block {chk['index']} ({chk['language']}): syntax OK", "32"))
        elif chk["ok"] is False:
            print(c(f"  ✗ code block {chk['index']} ({chk['language']}): line {chk['line']}: {chk['message']}", "31"))


def run_task(args, mode: str) -> int:
    s = settings_from_args(args)
    code, filename = read_input(getattr(args, "file", None))
    error = ""
    if getattr(args, "error", None):
        error, _ = read_input(args.error)
    question = " ".join(getattr(args, "question", None) or [])
    if mode == "generate" and not question:
        raise SystemExit("error: describe what to generate, e.g. generate \"palindrome checker in Python\"")
    if mode != "generate" and mode != "chat" and not code:
        raise SystemExit(f"error: {mode} needs a code file (or - to read from stdin)")
    history = None if args.no_save else History(s.db_path)
    bot = Assistant(s, history=history)
    print(c(f"▶ {MODES[mode]} · model {s.model} · {s.backend} · lang {s.language}", "1;36"))
    failed = False
    for ev in bot.stream(mode=mode, code=code, question=question, error=error, filename=filename,
                         session_id=getattr(args, "session", None), save=not args.no_save):
        if ev["type"] == "session":
            if ev["meta"].get("clipped"):
                print(c("  (long file: the middle was skipped to fit the context window)", "33"))
            err = ev["meta"].get("error") or {}
            if err.get("type"):
                print(c(f"  detected {err['type']}" + (f" at line {err['line']}" if err.get("line") else ""), "33"))
        elif ev["type"] == "token":
            sys.stdout.write(ev["text"])
            sys.stdout.flush()
        elif ev["type"] == "error":
            print(c(f"\n✗ {ev['message']}", "1;31"))
            if ev.get("hint"):
                print(c(f"  hint: {ev['hint']}", "33"))
            failed = True
        elif ev["type"] == "done":
            print()
            print_stats(ev)
            if ev.get("session_id"):
                print(c(f"  saved to history as session #{ev['session_id']}", "2"))
    return 1 if failed else 0


def cmd_chat(args) -> int:
    s = settings_from_args(args)
    history = None if args.no_save else History(s.db_path)
    bot = Assistant(s, history=history)
    mode, session = "chat", None
    print(c(f"Local Coder chat · model {s.model} · type /help for commands, /quit to exit", "1;36"))
    while True:
        try:
            line = input(c(f"[{mode}] you> ", "1"))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return 0
        if line == "/help":
            print("/mode <" + "|".join(MODES) + ">  /model <name>  /lang <en|hi|hinglish>  /new  /quit")
            print("Paste multi-line code between lines that contain only ``` (three backticks).")
            continue
        if line.startswith("/mode "):
            new = line.split(None, 1)[1].strip()
            if new in MODES:
                mode = new
            else:
                print("unknown mode")
            continue
        if line.startswith("/model "):
            s.model = line.split(None, 1)[1].strip()
            session = None
            continue
        if line.startswith("/lang "):
            lang = line.split(None, 1)[1].strip()
            if lang in LANGUAGES:
                s.language = lang
            continue
        if line == "/new":
            session = None
            print("new conversation")
            continue
        code = ""
        if line == "```":
            buf = []
            while True:
                try:
                    nxt = input()
                except EOFError:
                    break
                if nxt.strip() == "```":
                    break
                buf.append(nxt)
            code = "\n".join(buf)
            line = input(c("question (Enter to skip)> ", "1")).strip()
        for ev in bot.stream(mode=mode, code=code, question=line, session_id=session, save=not args.no_save):
            if ev["type"] == "session":
                session = ev["session_id"]
            elif ev["type"] == "token":
                sys.stdout.write(ev["text"])
                sys.stdout.flush()
            elif ev["type"] == "error":
                print(c(f"\n✗ {ev['message']}", "1;31"), c(ev.get("hint", ""), "33"))
            elif ev["type"] == "done":
                print()
                print_stats(ev)


def cmd_models(args) -> int:
    s = settings_from_args(args)
    client = make_client(s)
    try:
        models = client.list_models()
    except OllamaError as exc:
        print(c(f"✗ {exc.args[0]}", "31"))
        print(c(f"  hint: {exc.hint}", "33"))
        return 1
    if args.json:
        print(json.dumps(models, indent=2))
        return 0
    if not models:
        print("No models installed yet. Try: ollama pull qwen2.5-coder:3b")
        return 0
    print(c(f"{'MODEL':44} {'SIZE':>8}  {'PARAMS':>7}  QUANT", "1"))
    for m in models:
        mark = c(" ←  default", "32") if m["name"] == s.model else ""
        print(f"{m['name']:44} {m['size_gb']:>6} GB  {m['parameters']:>7}  {m['quantization']}{mark}")
    return 0


def cmd_hardware(args) -> int:
    r = report()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0
    print(c("Your machine", "1;36"))
    print(f"  OS          {r['os']} ({r['machine']}), {r['cpus']} CPU threads")
    print(f"  RAM         {r['ram_gb']} GB total" + (f", {r['available_gb']} GB available now" if r['available_gb'] else ""))
    print(f"  GPU         {r['gpu']}")
    print(f"  Tier        {r['tier']['label']}")
    print(f"  Advice      {r['tier']['advice']}")
    print(f"  num_ctx     start with {r['suggested_num_ctx']} (LOCAL_CODER_NUM_CTX)")
    print(c(f"  Best pick   {r['best']}", "1;32"))
    print()
    print(c(f"  {'MODEL':42} {'DOWNLOAD':>9} {'NEEDS RAM':>10}  FIT", "1"))
    names = {m["model"] for m in r["fits"]}
    tight = {m["model"] for m in r["tight"]}
    for m in MODEL_CATALOG:
        fit = c("yes", "32") if m["model"] in names else (c("tight", "33") if m["model"] in tight else c("no", "31"))
        print(f"  {m['model']:42} {m['download_gb']:>7} GB {m['needs_gb']:>7} GB  {fit}")
    return 0


def cmd_doctor(args) -> int:
    s = settings_from_args(args)
    client = make_client(s)
    ok = True
    print(c("Local Coder doctor", "1;36"))
    print(f"  version          {__version__}")
    print(f"  backend          {s.backend}  ({s.ollama_host})")
    try:
        v = client.version()
        print(c(f"  ✓ Ollama         reachable, version {v}", "32"))
        models = client.list_models()
        names = [m["name"] for m in models]
        print(c(f"  ✓ models         {len(names)} installed", "32") if names else c("  ✗ models         none installed", "31"))
        if s.model in names or (s.backend == "mock"):
            print(c(f"  ✓ default model  {s.model}", "32"))
        else:
            ok = False
            print(c(f"  ✗ default model  {s.model} is not installed", "31"))
            print(c(f"    fix: ollama pull {s.model}   or set LOCAL_CODER_MODEL to one of: {', '.join(names) or '-'}", "33"))
    except OllamaError as exc:
        ok = False
        print(c(f"  ✗ Ollama         {exc.args[0]}", "31"))
        print(c(f"    fix: {exc.hint}", "33"))
    r = report()
    print(f"  RAM              {r['ram_gb']} GB → {r['tier']['label']}; best pick {r['best']}")
    print(f"  history          {s.db_path}")
    print(f"  num_ctx          {s.num_ctx}   temperature {s.temperature}   language {s.language}")
    return 0 if ok else 1


def cmd_history(args) -> int:
    s = settings_from_args(args)
    h = History(s.db_path)
    if args.action == "list":
        rows = h.list_sessions(limit=args.limit, query=args.query or "")
        if not rows:
            print("No saved chats yet.")
        for r in rows:
            print(f"#{r['id']:<4} {r['mode']:9} {r['model'][:28]:28} {r['n']:>3} msgs  {r['title']}")
    elif args.action == "show":
        sess = h.get_session(args.id)
        if not sess:
            print(f"session {args.id} not found")
            return 1
        print(h.export_markdown(args.id))
    elif args.action == "export":
        try:
            md = h.export_markdown(args.id)
        except KeyError:
            print(f"session {args.id} not found")
            return 1
        out = Path(args.out or f"local-coder-session-{args.id}.md")
        out.write_text(md, encoding="utf-8")
        print(f"exported to {out}")
    elif args.action == "delete":
        print("deleted" if h.delete_session(args.id) else "not found")
    elif args.action == "clear":
        print(f"deleted {h.clear()} sessions")
    elif args.action == "stats":
        print(json.dumps(h.stats(), indent=2))
    return 0


def cmd_web(args) -> int:
    from .web import create_app

    s = settings_from_args(args)
    if args.bind not in ("127.0.0.1", "localhost", "::1"):
        print(c("Refusing to listen on a public address: Local Coder has no login and is meant for your own laptop.", "31"))
        return 2
    app = create_app(s)
    print(c(f"Local Coder UI on http://{args.bind}:{args.port}  (model {s.model}, backend {s.backend})", "1;36"))
    app.run(host=args.bind, port=args.port, debug=False, threaded=True)
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo

    return run_demo()


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", "-m", help="Ollama model name (default: LOCAL_CODER_MODEL or qwen2.5-coder:3b)")
    common.add_argument("--backend", choices=["ollama", "mock"], help="mock = offline demo answers")
    common.add_argument("--host", help="Ollama URL (default: OLLAMA_HOST or http://127.0.0.1:11434)")
    common.add_argument("--lang", choices=LANGUAGES, help="explanation language: en, hi (Hindi) or hinglish")
    common.add_argument("--level", choices=LEVELS)
    common.add_argument("--num-ctx", type=int, dest="num_ctx", help="context window in tokens (RAM!)")
    common.add_argument("--temperature", type=float)
    common.add_argument("--home", help="folder for history.db (default ~/.local-coder)")
    common.add_argument("--no-save", action="store_true", help="do not save this chat to history")
    common.add_argument("--think", dest="think", action="store_true", default=None, help="enable thinking on thinking models")
    common.add_argument("--no-think", dest="think", action="store_false", help="disable thinking (faster on Gemma 4)")

    p = argparse.ArgumentParser(prog="local-coder", description="Local AI coding assistant for students (Ollama).")
    p.add_argument("--version", action="version", version=f"local-coder {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    for mode, needs_file in (("explain", True), ("fix", True), ("docstring", True), ("tests", True), ("review", True)):
        sp = sub.add_parser(mode, parents=[common], help=MODES[mode])
        sp.add_argument("file", help="code file, or - for stdin")
        sp.add_argument("question", nargs="*", help="optional extra question")
        sp.add_argument("--session", type=int, help="continue a saved session")
        if mode == "fix":
            sp.add_argument("--error", "-e", help="file with the error/traceback (or - for stdin)")
        sp.set_defaults(func=lambda a, m=mode: run_task(a, m))
    sp = sub.add_parser("generate", parents=[common], help=MODES["generate"])
    sp.add_argument("question", nargs="+", help="what to build")
    sp.add_argument("--file", help="existing code to extend")
    sp.add_argument("--session", type=int)
    sp.set_defaults(func=lambda a: run_task(a, "generate"))
    sp = sub.add_parser("ask", parents=[common], help="one question, no code needed")
    sp.add_argument("question", nargs="+")
    sp.add_argument("--file")
    sp.add_argument("--session", type=int)
    sp.set_defaults(func=lambda a: run_task(a, "chat"))

    sub.add_parser("chat", parents=[common], help="interactive chat").set_defaults(func=cmd_chat)
    sp = sub.add_parser("models", parents=[common], help="list installed Ollama models")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_models)
    sp = sub.add_parser("hardware", help="RAM check and model advice")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_hardware)
    sub.add_parser("doctor", parents=[common], help="check Ollama, models and settings").set_defaults(func=cmd_doctor)
    sp = sub.add_parser("history", parents=[common], help="saved chats")
    sp.add_argument("action", choices=["list", "show", "export", "delete", "clear", "stats"])
    sp.add_argument("id", type=int, nargs="?")
    sp.add_argument("--query", "-q")
    sp.add_argument("--limit", type=int, default=30)
    sp.add_argument("--out", "-o")
    sp.set_defaults(func=cmd_history)
    sp = sub.add_parser("web", parents=[common], help="start the Flask UI")
    sp.add_argument("--port", type=int, default=5050)
    sp.add_argument("--bind", default="127.0.0.1")
    sp.set_defaults(func=cmd_web)
    sub.add_parser("demo", help="offline demo of every mode (mock model)").set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "history" and args.action in ("show", "export", "delete") and args.id is None:
        parser.error(f"history {args.action} needs a session id")
    return int(args.func(args) or 0)
