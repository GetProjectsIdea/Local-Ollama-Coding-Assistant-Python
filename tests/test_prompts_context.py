from local_coder.context import fit_messages
from local_coder.prompts import MODES, system_prompt, user_prompt

CODE = "def average(nums):\n    return sum(nums) / len(nums)\n\n\nprint(average([]))\n"
TB = 'Traceback (most recent call last):\n  File "a.py", line 2, in average\nZeroDivisionError: division by zero'


def test_all_modes_have_rules():
    for mode in MODES:
        sp = system_prompt(mode)
        assert f"MODE: {mode}" in sp
        assert "Local Coder" in sp


def test_unknown_mode_falls_back_to_chat():
    assert "MODE: chat" in system_prompt("hack")


def test_language_rules():
    assert "Devanagari" in system_prompt("explain", "hi")
    assert "Hinglish" in system_prompt("explain", "hinglish")
    assert "simple English" in system_prompt("explain", "en")
    assert "LANGUAGE: en" in system_prompt("explain", "klingon")


def test_tests_mode_picks_framework():
    assert "pytest" in system_prompt("tests", code_language="python")
    assert "JUnit 5" in system_prompt("tests", code_language="java")
    assert "Jest" in system_prompt("tests", code_language="javascript")


def test_academic_honesty_line():
    assert "viva" in system_prompt("generate")


def test_fix_prompt_includes_hot_spot():
    text, meta = user_prompt("fix", CODE, error=TB, filename="a.py")
    assert meta["error"]["type"] == "ZeroDivisionError"
    assert ">>2 |" in text
    assert "Detected: ZeroDivisionError" in text
    assert "```python" in text


def test_fix_without_error_asks_for_bug():
    text, _ = user_prompt("fix", CODE)
    assert "wrong output" in text


def test_generate_prompt():
    text, meta = user_prompt("generate", question="bubble sort", filename="sort.java")
    assert "Task: bubble sort" in text
    assert "Language: java" in text


def test_default_questions():
    assert "explain this code" in user_prompt("explain", CODE)[0]
    assert "docstrings" in user_prompt("docstring", CODE)[0]
    assert "unit tests" in user_prompt("tests", CODE)[0]
    assert "review" in user_prompt("review", CODE)[0]


def test_long_code_is_clipped():
    big = "\n".join(f"v{i} = {i}" for i in range(5000))
    text, meta = user_prompt("explain", big, code_budget_tokens=500)
    assert meta["clipped"] is True
    assert len(text) < 4000


def _msgs(n, size=400):
    out = [{"role": "system", "content": "sys"}]
    for i in range(n):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"{i} " + "x" * size})
    return out


def test_fit_keeps_everything_when_small():
    msgs = _msgs(4, 10)
    kept, dropped = fit_messages(msgs, 4096)
    assert kept == msgs and dropped == 0


def test_fit_drops_oldest_but_keeps_system_and_newest():
    msgs = _msgs(41, 400)  # ~100 tokens each
    kept, dropped = fit_messages(msgs, 2048, reserve_for_answer=1024)
    assert kept[0]["role"] == "system"
    assert kept[-1] == msgs[-1]
    assert dropped > 0
    assert kept[1]["role"] == "user"
    assert sum(len(m["content"]) for m in kept) // 4 <= 1100


def test_fit_empty():
    assert fit_messages([], 4096) == ([], 0)
