"""Tests for language features added in the v1.1.0 completion pass: match/case, dict,
range(), string/math builtins, `not`, and compound/plain assignment."""
import pytest

from neuva.parser import NeuvaParser
from neuva.interpreter.interpreter import NeuvaInterpreter, RuntimeError_


def _run(source: str) -> NeuvaInterpreter:
    interp = NeuvaInterpreter()
    interp.visit(NeuvaParser().parse(source))
    return interp


# ── match / case ─────────────────────────────────────────────────────────────

def test_match_case_hits_matching_case(capsys):
    _run(
        'let result = 1\n'
        'match result {\n'
        '    case 0: print "negative"\n'
        '    case 1: print "positive"\n'
        '    default: print "unknown"\n'
        '}\n'
    )
    assert capsys.readouterr().out.strip() == "positive"


def test_match_case_falls_through_to_default(capsys):
    _run(
        'let result = 99\n'
        'match result {\n'
        '    case 0: print "negative"\n'
        '    case 1: print "positive"\n'
        '    default: print "unknown"\n'
        '}\n'
    )
    assert capsys.readouterr().out.strip() == "unknown"


def test_match_case_no_default_no_match_is_noop(capsys):
    _run(
        'let result = 99\n'
        'match result {\n'
        '    case 0: print "negative"\n'
        '}\n'
        'print "after"\n'
    )
    assert capsys.readouterr().out.strip() == "after"


def test_match_case_string_scrutinee(capsys):
    _run(
        'let label = "b"\n'
        'match label {\n'
        '    case "a": print "first"\n'
        '    case "b": print "second"\n'
        '}\n'
    )
    assert capsys.readouterr().out.strip() == "second"


# ── dict ─────────────────────────────────────────────────────────────────────

def test_dict_literal_and_indexing():
    interp = _run('let config = {"lr": 0.001, "epochs": 20}\nlet lr = config["lr"]\n')
    assert interp.env.get("config") == {"lr": 0.001, "epochs": 20}
    assert interp.env.get("lr") == 0.001


def test_empty_dict_literal():
    interp = _run('let d = {}\n')
    assert interp.env.get("d") == {}


def test_dict_missing_key_raises():
    with pytest.raises(RuntimeError_):
        _run('let d = {"a": 1}\nlet x = d["missing"]\n')


def test_nested_dict_and_list():
    interp = _run('let d = {"a": [1, 2, 3], "b": {"c": 4}}\nlet x = d["b"]["c"]\n')
    assert interp.env.get("x") == 4


# ── range() ──────────────────────────────────────────────────────────────────

def test_range_with_step_in_for_loop(capsys):
    _run('for i in range(0, 10, 2) {\n    print i\n}\n')
    out = capsys.readouterr().out.split()
    assert out == ["0", "2", "4", "6", "8"]


# ── string builtins ──────────────────────────────────────────────────────────

def test_string_builtins():
    interp = _run(
        'let a = upper("hi")\n'
        'let b = lower("HI")\n'
        'let c = strip("  x  ")\n'
        'let d = replace("abc", "b", "z")\n'
        'let e = split("a,b,c", ",")\n'
        'let f = join("-", [1, 2, 3])\n'
        'let g = len("hello")\n'
    )
    assert interp.env.get("a") == "HI"
    assert interp.env.get("b") == "hi"
    assert interp.env.get("c") == "x"
    assert interp.env.get("d") == "azc"
    assert interp.env.get("e") == ["a", "b", "c"]
    assert interp.env.get("f") == "1-2-3"
    assert interp.env.get("g") == 5


# ── math builtins ────────────────────────────────────────────────────────────

def test_math_builtins():
    interp = _run(
        'let a = abs(-5)\n'
        'let b = sqrt(16)\n'
        'let c = pow(2, 3)\n'
        'let d = round(3.14159, 2)\n'
        'let e = min(3, 1, 2)\n'
        'let f = max(3, 1, 2)\n'
        'let g = sum([1, 2, 3])\n'
        'let h = mean([1, 2, 3])\n'
        'let i = log(1)\n'
        'let j = exp(0)\n'
    )
    assert interp.env.get("a") == 5
    assert interp.env.get("b") == 4.0
    assert interp.env.get("c") == 8
    assert interp.env.get("d") == 3.14
    assert interp.env.get("e") == 1
    assert interp.env.get("f") == 3
    assert interp.env.get("g") == 6
    assert interp.env.get("h") == 2.0
    assert interp.env.get("i") == 0.0
    assert interp.env.get("j") == 1.0


# ── not ──────────────────────────────────────────────────────────────────────

def test_not_keyword(capsys):
    _run('let ready = false\nif not ready {\n    print "not ready"\n}\n')
    assert capsys.readouterr().out.strip() == "not ready"


def test_not_keyword_matches_bang_operator():
    interp = _run('let a = not true\nlet b = !true\n')
    assert interp.env.get("a") == interp.env.get("b") == False


# ── compound / plain assignment ─────────────────────────────────────────────

def test_plain_reassignment():
    interp = _run('let x = 5\nx = 10\n')
    assert interp.env.get("x") == 10


def test_compound_assignment_operators():
    interp = _run('let x = 10\nx += 5\nx -= 2\nx *= 3\nx /= 2\n')
    assert interp.env.get("x") == 19.5


def test_compound_assignment_string_concat():
    interp = _run('let s = "a"\ns += "b"\n')
    assert interp.env.get("s") == "ab"


def test_assignment_to_undeclared_name_raises():
    with pytest.raises(RuntimeError_):
        _run('x = 5\n')


def test_augmented_assignment_mutates_outer_scope_from_function():
    interp = _run(
        'let counter = 0\n'
        'fn bump() {\n'
        '    counter += 1\n'
        '    return counter\n'
        '}\n'
        'let a = bump()\n'
        'let b = bump()\n'
    )
    assert interp.env.get("a") == 1
    assert interp.env.get("b") == 2
    assert interp.env.get("counter") == 2


# ── triple-quoted strings ────────────────────────────────────────────────────

def test_triple_quoted_string():
    interp = _run('let msg = """\nhello\nworld"""\n')
    assert interp.env.get("msg") == "\nhello\nworld"
