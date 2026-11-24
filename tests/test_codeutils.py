import pytest

from local_coder.codeutils import (check_blocks, check_python_syntax, clip_code, detect_language,
                                   estimate_tokens, extract_code_blocks, numbered, parse_error,
                                   snippet_around)

PY_TB = '''Traceback (most recent call last):
  File "average.py", line 5, in <module>
    print(average([]))
  File "average.py", line 2, in average
    return sum(nums) / len(nums)
ZeroDivisionError: division by zero
'''


@pytest.mark.parametrize("name,lang", [
    ("a.py", "python"), ("Main.java", "java"), ("x.cpp", "cpp"), ("x.c", "c"), ("app.js", "javascript"),
    ("index.html", "html"), ("q.sql", "sql"), ("m.PHP", "php"),
])
def test_detect_language_by_extension(name, lang):
    assert detect_language(name) == lang


@pytest.mark.parametrize("code,lang", [
    ("def f():\n    return 1\n", "python"),
    ("public class Main { public static void main(String[] a) {} }", "java"),
    ("#include <iostream>\nint main(){ std::cout << 1; }", "cpp"),
    ("#include <stdio.h>\nint main(){ printf(\"hi\"); }", "c"),
    ("const x = 5;\nconsole.log(x);", "javascript"),
    ("<?php echo 1; ?>", "php"),
    ("SELECT * FROM students;", "sql"),
    ("just some words", "text"),
])
def test_detect_language_by_content(code, lang):
    assert detect_language(None, code) == lang


def test_parse_python_traceback():
    info = parse_error(PY_TB)
    assert info["type"] == "ZeroDivisionError"
    assert info["message"] == "division by zero"
    assert info["line"] == 2
    assert info["file"] == "average.py"
    assert [f["line"] for f in info["frames"]] == [5, 2]
    assert info["frames"][1]["function"] == "average"


def test_parse_python_one_liner():
    info = parse_error("NameError: name 'totl' is not defined")
    assert info["type"] == "NameError"
    assert "totl" in info["message"]


def test_parse_python_syntax_error_traceback():
    tb = '  File "x.py", line 3\n    if x == 1\n             ^\nSyntaxError: expected \':\''
    info = parse_error(tb)
    assert info["type"] == "SyntaxError" and info["line"] == 3


def test_parse_java_stack_trace():
    tb = ('Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: Index 5 out of bounds for length 5\n'
          '\tat Main.main(Main.java:7)')
    info = parse_error(tb)
    assert info["type"] == "ArrayIndexOutOfBoundsException"
    assert info["line"] == 7 and info["file"] == "Main.java"


def test_parse_gcc_error():
    info = parse_error("main.c:4:5: error: expected ';' before 'return'")
    assert info["type"] == "CompileError" and info["line"] == 4 and info["file"] == "main.c"


def test_parse_js_error():
    info = parse_error("TypeError: Cannot read properties of undefined (reading 'map')\n    at app.js:12:9")
    assert info["type"] == "TypeError" and info["line"] == 12


def test_parse_nothing():
    assert parse_error("") == {}
    assert parse_error("my program prints the wrong answer") == {}


def test_snippet_marks_the_line():
    code = "\n".join(f"line{i}" for i in range(1, 21))
    snip = snippet_around(code, 10, radius=2)
    assert snip.splitlines()[2].startswith(">>10")
    assert len(snip.splitlines()) == 5
    assert snippet_around(code, 0) == ""
    assert snippet_around("", 3) == ""


def test_numbered():
    assert numbered("a\nb") == "1 | a\n2 | b"


def test_extract_code_blocks():
    md = "text\n```py\nprint(1)\n```\nmore\n```javascript\nconsole.log(1)\n```\n```\nplain\n```"
    blocks = extract_code_blocks(md)
    assert [b["language"] for b in blocks] == ["python", "javascript", "text"]
    assert blocks[0]["code"] == "print(1)"


def test_check_python_syntax():
    assert check_python_syntax("x = 1\n")["ok"] is True
    bad = check_python_syntax("def f(:\n    pass\n")
    assert bad["ok"] is False and bad["line"] == 1


def test_check_python_syntax_never_executes(tmp_path):
    marker = tmp_path / "ran"
    code = f"open({str(marker)!r}, 'w').write('x')\n"
    assert check_python_syntax(code)["ok"] is True
    assert not marker.exists()


def test_check_blocks():
    md = "```python\nprint('ok')\n```\n```python\nprint('bad'\n```\n```java\nclass A {}\n```"
    res = check_blocks(md)
    assert [r["ok"] for r in res] == [True, False, None]


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 100) == 100
    assert estimate_tokens("नमस्ते" * 10) > estimate_tokens("hello!" * 10) // 2


def test_clip_code_keeps_head_and_tail():
    code = "\n".join(f"x{i} = {i}" for i in range(2000))
    clipped, was = clip_code(code, 300)
    assert was is True
    assert clipped.startswith("x0 = 0")
    assert clipped.rstrip().endswith("x1999 = 1999")
    assert "lines skipped" in clipped
    assert estimate_tokens(clipped) <= 330


def test_clip_code_small_is_untouched():
    assert clip_code("x = 1", 100) == ("x = 1", False)
