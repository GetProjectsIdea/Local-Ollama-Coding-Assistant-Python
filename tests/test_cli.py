import json

import pytest

from local_coder.cli import build_parser, main

CODE = "def average(nums):\n    return sum(nums) / len(nums)\n"
TB = 'Traceback (most recent call last):\n  File "a.py", line 2, in average\nZeroDivisionError: division by zero\n'


@pytest.fixture
def files(tmp_path):
    (tmp_path / "a.py").write_text(CODE)
    (tmp_path / "err.txt").write_text(TB)
    return tmp_path


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_parser_has_all_commands():
    p = build_parser()
    for cmd in ("explain", "fix", "docstring", "tests", "review", "generate", "ask", "chat", "models",
                "hardware", "doctor", "history", "web", "demo"):
        assert cmd in p._subparsers._group_actions[0].choices


def test_explain_mock(capsys, files):
    code, out = run(capsys, "explain", str(files / "a.py"), "--backend", "mock", "--home", str(files))
    assert code == 0
    assert "What this code does" in out and "saved to history as session #1" in out


def test_fix_mock_detects_error(capsys, files):
    code, out = run(capsys, "fix", str(files / "a.py"), "-e", str(files / "err.txt"),
                    "--backend", "mock", "--home", str(files))
    assert code == 0
    assert "detected ZeroDivisionError at line 2" in out
    assert "syntax OK" in out


def test_generate_and_no_save(capsys, files):
    code, out = run(capsys, "generate", "palindrome", "checker", "--backend", "mock", "--home", str(files), "--no-save")
    assert code == 0 and "is_palindrome" in out and "saved to history" not in out


def test_generate_needs_text(capsys, files):
    with pytest.raises(SystemExit):
        main(["generate", "--backend", "mock"])


def test_missing_file(files):
    with pytest.raises(SystemExit, match="file not found"):
        main(["explain", str(files / "nope.py"), "--backend", "mock", "--home", str(files)])


def test_stdin(capsys, files, monkeypatch):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO(CODE))
    code, out = run(capsys, "docstring", "-", "--backend", "mock", "--home", str(files))
    assert code == 0 and "Args:" in out


def test_history_commands(capsys, files):
    run(capsys, "explain", str(files / "a.py"), "--backend", "mock", "--home", str(files))
    code, out = run(capsys, "history", "list", "--home", str(files))
    assert "#1" in out and "explain" in out
    code, out = run(capsys, "history", "show", "1", "--home", str(files))
    assert "## Assistant" in out
    dest = files / "out.md"
    run(capsys, "history", "export", "1", "-o", str(dest), "--home", str(files))
    assert dest.read_text().startswith("# ")
    code, out = run(capsys, "history", "stats", "--home", str(files))
    assert json.loads(out)["sessions"] == 1
    code, out = run(capsys, "history", "delete", "1", "--home", str(files))
    assert "deleted" in out
    code, out = run(capsys, "history", "show", "1", "--home", str(files))
    assert code == 1


def test_history_id_required(files):
    with pytest.raises(SystemExit):
        main(["history", "show", "--home", str(files)])


def test_models_and_doctor_against_fake(capsys, fake_ollama, files):
    url, _ = fake_ollama
    code, out = run(capsys, "models", "--host", url)
    assert code == 0 and "qwen2.5-coder:3b" in out and "default" in out
    code, out = run(capsys, "models", "--host", url, "--json")
    assert json.loads(out)[0]["name"] == "gemma4-e2b-q4km:latest"
    code, out = run(capsys, "doctor", "--host", url, "--home", str(files))
    assert code == 0 and "0.40.16" in out
    code, out = run(capsys, "doctor", "--host", url, "--home", str(files), "--model", "flint")
    assert code == 1 and "ollama pull flint" in out


def test_doctor_when_ollama_is_down(capsys, files):
    code, out = run(capsys, "doctor", "--host", "http://127.0.0.1:9", "--home", str(files))
    assert code == 1 and "ollama serve" in out


def test_fix_reports_backend_error(capsys, files):
    code, out = run(capsys, "fix", str(files / "a.py"), "--host", "http://127.0.0.1:9", "--home", str(files))
    assert code == 1 and "hint" in out


def test_hardware(capsys):
    code, out = run(capsys, "hardware")
    assert code == 0 and "Best pick" in out and "flint" in out
    code, out = run(capsys, "hardware", "--json")
    assert "tier" in json.loads(out)


def test_web_refuses_public_bind(capsys, files):
    code, out = run(capsys, "web", "--bind", "0.0.0.0", "--backend", "mock", "--home", str(files))
    assert code == 2 and "Refusing" in out


def test_demo(capsys):
    code, out = run(capsys, "demo")
    assert code == 0
    assert "[6] generate" in out and "6 sessions, 12 messages" in out


def test_chat_repl(capsys, files, monkeypatch):
    lines = iter(["/help", "/mode explain", "```", "x = 1", "```", "what is this?", "/lang hi", "/new", "hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(lines))
    code = main(["chat", "--backend", "mock", "--home", str(files)])
    out = capsys.readouterr().out
    assert code == 0
    assert "What this code does" in out
    assert "new conversation" in out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "local-coder 1.0.0" in capsys.readouterr().out
