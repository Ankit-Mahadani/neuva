import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser import NeuvaParser
from interpreter.interpreter import NeuvaInterpreter


def _run(source: str) -> NeuvaInterpreter:
    interp = NeuvaInterpreter()
    interp.visit(NeuvaParser().parse(source))
    return interp


def test_print_variable():
    interp = _run("let x = 42\nprint(x)")
    assert interp.env.get("x") == 42


def test_addition():
    interp = _run("let x = 10 + 5")
    assert interp.env.get("x") == 15


def test_string_literal():
    interp = _run('let name = "Neuva"')
    assert interp.env.get("name") == "Neuva"


def test_boolean_literal():
    interp = _run("let x = true")
    assert interp.env.get("x") is True


def test_arithmetic_precedence():
    interp = _run("let x = 10 * 3 + 2")
    assert interp.env.get("x") == 32


if __name__ == "__main__":
    tests = [
        test_print_variable,
        test_addition,
        test_string_literal,
        test_boolean_literal,
        test_arithmetic_precedence,
    ]
    for t in tests:
        t()
    print("All 5 interpreter tests passed!")
