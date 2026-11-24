import os
import stat

from local_coder.history import History, make_title


def test_make_title():
    assert make_title("Task: write a bubble sort") == "write a bubble sort"
    assert make_title("") == "New chat"
    assert make_title("x" * 100).endswith("…") and len(make_title("x" * 100)) == 60


def test_session_roundtrip(history):
    sid = history.create_session("Explain recursion", "explain", "qwen2.5-coder:3b", "hi")
    history.add_message(sid, "user", "what is recursion?")
    history.add_message(sid, "assistant", "A function calling itself.", {"stats": {"tokens_per_second": 12.5, "output_tokens": 5}})
    s = history.get_session(sid)
    assert s["title"] == "Explain recursion" and s["language"] == "hi"
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]
    assert s["messages"][1]["meta"]["stats"]["tokens_per_second"] == 12.5
    assert history.chat_messages(sid) == [{"role": "user", "content": "what is recursion?"},
                                          {"role": "assistant", "content": "A function calling itself."}]


def test_list_search_and_order(history):
    a = history.create_session("sorting", "chat", "m")
    b = history.create_session("graphs", "chat", "m")
    history.add_message(a, "user", "explain quicksort pivot")
    rows = history.list_sessions()
    assert rows[0]["id"] == a  # most recently updated first
    assert [r["id"] for r in history.list_sessions(query="pivot")] == [a]
    assert [r["id"] for r in history.list_sessions(query="graphs")] == [b]
    assert rows[0]["n"] == 1


def test_delete_cascades(history):
    sid = history.create_session("x", "chat", "m")
    history.add_message(sid, "user", "hi")
    assert history.delete_session(sid) is True
    assert history.get_session(sid) is None
    assert history.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert history.delete_session(sid) is False


def test_clear_and_stats(history):
    for i in range(3):
        history.add_message(history.create_session(f"s{i}", "chat", "m"), "user", "q")
    st = history.stats()
    assert st["sessions"] == 3 and st["messages"] == 3 and st["bytes"] > 0
    assert history.clear() == 3


def test_export_markdown(history):
    sid = history.create_session("Fix my code", "fix", "gemma4:e2b")
    history.add_message(sid, "user", "it crashes")
    history.add_message(sid, "assistant", "```python\nx = 1\n```", {"stats": {"tokens_per_second": 9.9, "output_tokens": 3}})
    md = history.export_markdown(sid)
    assert md.startswith("# Fix my code")
    assert "- Model: gemma4:e2b" in md and "## You" in md and "## Assistant" in md
    assert "9.9 tokens/s" in md


def test_export_missing_raises(history):
    import pytest
    with pytest.raises(KeyError):
        history.export_markdown(999)


def test_private_permissions(tmp_path):
    if os.name == "nt":
        return
    h = History(tmp_path / "priv" / "history.db")
    mode = stat.S_IMODE(os.stat(tmp_path / "priv" / "history.db").st_mode)
    assert mode == 0o600
    assert stat.S_IMODE(os.stat(tmp_path / "priv").st_mode) == 0o700
    h.close()


def test_role_check(history):
    import sqlite3
    import pytest
    sid = history.create_session("x", "chat", "m")
    with pytest.raises(sqlite3.IntegrityError):
        history.add_message(sid, "system", "nope")
