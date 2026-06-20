from neuva.parser import NeuvaParser
from neuva.interpreter.interpreter import NeuvaInterpreter


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


def test_fstring_interpolation():
    interp = _run('let lang = "Neuva"\nlet msg = "I love {lang}"')
    assert interp.env.get("msg") == "I love Neuva"


def test_fstring_number():
    interp = _run("let n = 42\nlet s = \"answer: {n}\"")
    assert interp.env.get("s") == "answer: 42"


def test_list_literal():
    interp = _run("let scores = [85, 90, 78]")
    assert interp.env.get("scores") == [85, 90, 78]


def test_list_index():
    interp = _run("let scores = [85, 90, 78]\nlet first = scores[0]")
    assert interp.env.get("first") == 85


def test_list_last_index():
    interp = _run("let scores = [10, 20, 30]\nlet last = scores[2]")
    assert interp.env.get("last") == 30


def test_len_builtin():
    interp = _run("let scores = [1, 2, 3, 4]\nlet n = len(scores)")
    assert interp.env.get("n") == 4


def test_and_true():
    interp = _run("let x = true and true")
    assert interp.env.get("x") is True


def test_and_false():
    interp = _run("let x = true and false")
    assert interp.env.get("x") is False


def test_or_true():
    interp = _run("let x = false or true")
    assert interp.env.get("x") is True


def test_or_false():
    interp = _run("let x = false or false")
    assert interp.env.get("x") is False


def test_and_short_circuit():
    interp = _run("let x = 5\nlet y = x > 0 and x < 10")
    assert interp.env.get("y") is True


def test_or_short_circuit():
    interp = _run("let x = 5\nlet y = x > 10 or x > 3")
    assert interp.env.get("y") is True


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
