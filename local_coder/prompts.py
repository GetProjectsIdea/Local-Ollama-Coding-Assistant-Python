"""Prompt templates for each assistant mode, in English, Hindi or Hinglish."""
from __future__ import annotations

from .codeutils import TEST_FRAMEWORK, clip_code, detect_language, parse_error, snippet_around

MODES = {
    "explain": "Explain code",
    "generate": "Generate code",
    "fix": "Fix an error",
    "docstring": "Write docstrings / comments",
    "tests": "Write unit tests",
    "review": "Review my code",
    "chat": "Ask anything",
}

BASE = (
    "You are Local Coder, a patient programming tutor for college students in India. "
    "You run fully offline on the student's own laptop. Be correct, short and practical. "
    "Use simple words, numbered steps and small examples. Put every piece of code in a fenced "
    "code block with the language name. Never invent library functions; if you are not sure, say so. "
    "Do not do graded work dishonestly: teach the idea so the student can explain it in a viva."
)

MODE_RULES = {
    "explain": "Explain what the code does line by line or block by block, then give the big idea, "
               "the time complexity if relevant, and one common mistake to avoid.",
    "generate": "Write clean, commented, runnable code for the request. Prefer the standard library. "
                "After the code, explain how it works in 3-5 bullet points and show sample input/output.",
    "fix": "Find the root cause of the error. First say in one or two sentences WHY it happens, "
           "pointing to the exact line. Then give the corrected code in one block, and finally one tip "
           "to avoid this mistake next time. Change as little as possible.",
    "docstring": "Return the same code with clear docstrings (Google style for Python, Javadoc/JSDoc "
                 "for Java/JavaScript) and short comments on tricky lines. Do not change the logic.",
    "tests": "Write unit tests for the code using {framework}. Cover normal cases, edge cases "
             "(empty input, zero, negative, large values) and expected errors. Return one code block "
             "and then list what each test checks.",
    "review": "Review the code like a friendly senior: correctness bugs first, then readability, naming, "
              "edge cases and performance. Give a short prioritised list and an improved version only if needed.",
    "chat": "Answer the student's programming question clearly.",
}

LANGUAGE_RULES = {
    "en": "Write the explanation in simple English.",
    "hi": "Write the explanation in Hindi using Devanagari script. Keep code, identifiers, error names "
          "and technical terms (function, loop, list, class) in English.",
    "hinglish": "Write the explanation in Hinglish (Hindi words in Roman script mixed with English), "
                "the way Indian students talk. Keep code and technical terms in English.",
}

LEVEL_RULES = {
    "beginner": "The student is a beginner (first or second year). Avoid jargon or explain it.",
    "intermediate": "The student knows the basics of programming and data structures.",
    "advanced": "The student is advanced; be concise and technical.",
}


def system_prompt(mode: str, language: str = "en", level: str = "beginner", code_language: str = "") -> str:
    mode = mode if mode in MODES else "chat"
    rule = MODE_RULES[mode]
    if mode == "tests":
        rule = rule.format(framework=TEST_FRAMEWORK.get(code_language, "the usual test framework"))
    return "\n".join([
        BASE,
        f"MODE: {mode}",
        f"LANGUAGE: {language if language in LANGUAGE_RULES else 'en'}",
        f"LEVEL: {level if level in LEVEL_RULES else 'beginner'}",
        rule,
        LANGUAGE_RULES.get(language, LANGUAGE_RULES["en"]),
        LEVEL_RULES.get(level, LEVEL_RULES["beginner"]),
    ])


def user_prompt(mode: str, code: str = "", question: str = "", error: str = "",
                filename: str | None = None, code_budget_tokens: int = 2500) -> tuple[str, dict]:
    """Build the user message. Returns (text, meta) where meta has language/error info."""
    code = (code or "").replace("\r\n", "\n").strip("\n")
    lang = detect_language(filename, code) if (code or filename) else ""
    lang = "" if lang == "text" and not code else lang
    meta: dict = {"code_language": lang, "clipped": False}
    parts = []
    if mode == "generate":
        parts.append(f"Task: {question.strip() or 'Write a small example program.'}")
        if lang or filename:
            parts.append(f"Language: {lang or 'python'}")
        if code:
            clipped, meta["clipped"] = clip_code(code, code_budget_tokens)
            parts.append(f"Existing code to build on:\n```{lang}\n{clipped}\n```")
        return "\n\n".join(parts), meta
    if code:
        clipped, meta["clipped"] = clip_code(code, code_budget_tokens)
        name = f" ({filename})" if filename else ""
        parts.append(f"Code{name}:\n```{lang}\n{clipped}\n```")
    if mode == "fix":
        err = parse_error(error)
        meta["error"] = err
        if error.strip():
            parts.append(f"Error message:\n```\n{error.strip()[-3000:]}\n```")
        if err.get("line") and code:
            hot = snippet_around(code, err["line"])
            if hot:
                parts.append(f"The error points at line {err['line']} (marked >>):\n```\n{hot}\n```")
        if err.get("type"):
            parts.append(f"Detected: {err['type']}" + (f" - {err['message']}" if err.get("message") else ""))
        if not error.strip():
            parts.append("There is no error message; the program gives the wrong output. Find the bug.")
    if question.strip():
        parts.append(f"Question: {question.strip()}")
    elif mode == "explain":
        parts.append("Please explain this code.")
    elif mode == "docstring":
        parts.append("Please add docstrings and helpful comments.")
    elif mode == "tests":
        parts.append("Please write unit tests for this code.")
    elif mode == "review":
        parts.append("Please review this code.")
    return "\n\n".join(parts) or (question or "Hello"), meta
