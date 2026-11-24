"""Helpers for code: language detection, traceback parsing, code-block extraction,
syntax checks and token estimates. Nothing here ever executes student code."""
from __future__ import annotations

import ast
import re
from pathlib import Path

EXT_LANG = {
    ".py": "python", ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp",
    ".hpp": "cpp", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript",
    ".html": "html", ".css": "css", ".php": "php", ".sql": "sql", ".go": "go", ".rs": "rust",
    ".kt": "kotlin", ".swift": "swift", ".cs": "csharp", ".rb": "ruby", ".sh": "bash",
    ".dart": "dart", ".r": "r", ".m": "matlab", ".ipynb": "json", ".json": "json",
}

TEST_FRAMEWORK = {
    "python": "pytest", "java": "JUnit 5", "javascript": "Jest", "typescript": "Jest",
    "c": "a small assert-based main()", "cpp": "GoogleTest", "php": "PHPUnit", "go": "the testing package",
    "rust": "#[test] functions", "kotlin": "JUnit 5", "csharp": "xUnit", "ruby": "RSpec", "dart": "package:test",
}

MAX_CODE_CHARS = 60_000


def detect_language(filename: str | None = None, code: str = "") -> str:
    """Guess the programming language from the file name, then from the code."""
    if filename:
        lang = EXT_LANG.get(Path(filename).suffix.lower())
        if lang:
            return lang
    head = code[:2000]
    checks = [
        ("java", r"\bpublic\s+(static\s+)?(class|void)\b|System\.out\.println"),
        ("cpp", r"#include\s*<(iostream|vector|string)>|std::"),
        ("c", r"#include\s*<stdio\.h>|\bprintf\s*\("),
        ("php", r"<\?php"),
        ("html", r"<!DOCTYPE html>|<html[\s>]"),
        ("javascript", r"\bconsole\.log\(|\bfunction\s+\w+\s*\(|=>\s*\{|\bconst\s+\w+\s*="),
        ("sql", r"(?i)^\s*(select|insert|create\s+table|update)\b"),
        ("python", r"^\s*(def|class|import|from)\s+\w+|print\(|if __name__ =="),
    ]
    for lang, pattern in checks:
        if re.search(pattern, head, re.M):
            return lang
    return "text"


_FRAME = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>\S+))?')
_ERRLINE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit)|KeyboardInterrupt)"
                      r"(?::\s*(?P<msg>.*))?$")
_JAVA = re.compile(r"(?P<type>(?:[a-z]\w*\.)+[A-Z]\w*(?:Exception|Error))(?::\s*(?P<msg>.*))?")
_JAVA_AT = re.compile(r"at\s+[\w.$]+\((?P<file>[\w.]+):(?P<line>\d+)\)")
_GCC = re.compile(r"^(?P<file>[^:\s]+):(?P<line>\d+):(?:\d+:)?\s*(?:fatal\s+)?error:\s*(?P<msg>.+)$", re.M)
_JS = re.compile(r"(?P<type>(?:Type|Reference|Syntax|Range)Error):\s*(?P<msg>.+)")


def parse_error(text: str) -> dict:
    """Pull out the error type, message and line number from a traceback or compiler output.

    Works for Python tracebacks, Java stack traces, gcc/g++ errors and JavaScript errors.
    Returns {} when nothing recognisable is found.
    """
    text = (text or "").strip()
    if not text:
        return {}
    info: dict = {}
    frames = [m.groupdict() for m in _FRAME.finditer(text)]
    if frames:
        info["language"] = "python"
        info["frames"] = [{"file": f["file"], "line": int(f["line"]), "function": f["func"] or ""} for f in frames]
        last = frames[-1]
        info["file"], info["line"] = last["file"], int(last["line"])
        for line in reversed(text.splitlines()):
            m = _ERRLINE.match(line.strip())
            if m:
                info["type"], info["message"] = m.group("type"), (m.group("msg") or "").strip()
                break
        if "type" not in info:
            info["type"], info["message"] = "Error", text.splitlines()[-1].strip()
        return info
    m = _ERRLINE.match(text.splitlines()[-1].strip())
    if m:  # a bare one-line Python error such as "NameError: name 'x' is not defined"
        info.update(language="python", type=m.group("type"), message=(m.group("msg") or "").strip())
        sm = re.search(r"line (\d+)", text)
        if sm:
            info["line"] = int(sm.group(1))
        return info
    m = _JAVA.search(text)
    if m:
        info.update(language="java", type=m.group("type").rsplit(".", 1)[-1], message=(m.group("msg") or "").strip())
        at = _JAVA_AT.search(text)
        if at:
            info["file"], info["line"] = at.group("file"), int(at.group("line"))
        return info
    m = _GCC.search(text)
    if m:
        info.update(language="c/cpp", type="CompileError", message=m.group("msg").strip(),
                    file=m.group("file"), line=int(m.group("line")))
        return info
    m = _JS.search(text)
    if m:
        info.update(language="javascript", type=m.group("type"), message=m.group("msg").strip())
        jm = re.search(r":(\d+):\d+\)?\s*$", text, re.M)
        if jm:
            info["line"] = int(jm.group(1))
        return info
    return {}


def numbered(code: str, start: int = 1) -> str:
    lines = code.splitlines()
    width = len(str(start + len(lines)))
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, start))


def snippet_around(code: str, line: int, radius: int = 6) -> str:
    """Numbered lines around `line` with a >> marker, so a small model sees the hot spot."""
    lines = code.splitlines()
    if not lines or line < 1:
        return ""
    lo, hi = max(1, line - radius), min(len(lines), line + radius)
    width = len(str(hi))
    out = []
    for n in range(lo, hi + 1):
        mark = ">>" if n == line else "  "
        out.append(f"{mark}{n:>{width}} | {lines[n - 1]}")
    return "\n".join(out)


_FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)```", re.S)


def extract_code_blocks(markdown: str) -> list[dict]:
    """Return [{"language": "python", "code": "..."}] for every fenced block in a reply."""
    blocks = []
    for m in _FENCE.finditer(markdown or ""):
        lang = (m.group(1) or "").lower() or "text"
        lang = {"py": "python", "js": "javascript", "c++": "cpp", "sh": "bash"}.get(lang, lang)
        blocks.append({"language": lang, "code": m.group(2).rstrip("\n")})
    return blocks


def check_python_syntax(code: str) -> dict:
    """Parse (never run) Python code and report the first syntax error, if any."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {"ok": False, "message": exc.msg, "line": exc.lineno or 0}
    except ValueError as exc:  # e.g. null bytes
        return {"ok": False, "message": str(exc), "line": 0}
    return {"ok": True, "message": "syntax OK", "line": 0}


def check_blocks(markdown: str) -> list[dict]:
    """Syntax-check every Python block in a model reply. Other languages are listed as 'not checked'."""
    results = []
    for i, block in enumerate(extract_code_blocks(markdown), 1):
        if block["language"] == "python":
            res = check_python_syntax(block["code"])
        else:
            res = {"ok": None, "message": "not checked (only Python is parsed)", "line": 0}
        results.append({"index": i, "language": block["language"], **res})
    return results


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token for English and code).

    Devanagari text usually needs more tokens per character, so it counts double.
    """
    if not text:
        return 0
    deva = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    return max(1, (len(text) - deva) // 4 + deva // 2)


def clip_code(code: str, max_tokens: int) -> tuple[str, bool]:
    """Trim code to fit a token budget, keeping the start and the end of the file."""
    if estimate_tokens(code) <= max_tokens:
        return code, False
    lines = code.splitlines()
    keep_head, keep_tail = [], []
    budget = max_tokens * 4
    i, j = 0, len(lines) - 1
    used = 0
    while i <= j:
        head = lines[i]
        if used + len(head) + 1 > budget * 0.6 and keep_head:
            break
        keep_head.append(head)
        used += len(head) + 1
        i += 1
    while j >= i:
        tail = lines[j]
        if used + len(tail) + 1 > budget:
            break
        keep_tail.insert(0, tail)
        used += len(tail) + 1
        j -= 1
    skipped = j - i + 1
    middle = [f"# ... {skipped} lines skipped to fit the model's context window ..."] if skipped > 0 else []
    return "\n".join(keep_head + middle + keep_tail), True
