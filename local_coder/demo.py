"""Offline demo: every mode against the mock model, using a temporary history file."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .assistant import Assistant
from .config import Settings
from .history import History

BUGGY = '''def average(nums):
    return sum(nums) / len(nums)


print(average([]))
'''
TRACEBACK = '''Traceback (most recent call last):
  File "average.py", line 5, in <module>
    print(average([]))
  File "average.py", line 2, in average
    return sum(nums) / len(nums)
ZeroDivisionError: division by zero
'''


def run_demo(out=sys.stdout) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(backend="mock", model="mock-coder:tiny", home=Path(tmp))
        h = History(s.db_path)
        bot = Assistant(s, history=h)
        steps = [
            ("explain", dict(code=BUGGY, filename="average.py"), "en"),
            ("explain", dict(code=BUGGY, filename="average.py"), "hi"),
            ("fix", dict(code=BUGGY, error=TRACEBACK, filename="average.py"), "en"),
            ("docstring", dict(code=BUGGY, filename="average.py"), "en"),
            ("tests", dict(code=BUGGY, filename="average.py"), "en"),
            ("generate", dict(question="palindrome checker in Python"), "en"),
        ]
        for i, (mode, kw, lang) in enumerate(steps, 1):
            print(f"\n[{i}] {mode} ({lang})", file=out)
            print("-" * 60, file=out)
            res = bot.ask(mode=mode, language=lang, **kw)
            if res.get("type") == "error":
                print("error:", res["message"], file=out)
                return 1
            print(res["answer"], file=out)
            for chk in res.get("checks", []):
                status = "syntax OK" if chk["ok"] else ("not checked" if chk["ok"] is None else f"line {chk['line']}: {chk['message']}")
                print(f"  check block {chk['index']} ({chk['language']}): {status}", file=out)
        st = h.stats()
        print(f"\nhistory: {st['sessions']} sessions, {st['messages']} messages saved locally (temporary demo file)", file=out)
        h.close()
    return 0
