"""Regression tests for grammar ambiguity risk introduced by adding `{`-delimited dict
literals and match/case blocks to a grammar that already uses `{`/`}` heavily for
if/for/while/fn/model bodies. The Earley parser's `ambiguity="resolve"` mode can pick a
wrong-but-valid parse silently rather than erroring, so these are worth pinning down
explicitly rather than trusting "it parsed" alone — each program is also *run* and its
result checked, not just parsed."""
import glob
import os

from neuva.parser import NeuvaParser
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def _run(source: str) -> NeuvaInterpreter:
    interp = NeuvaInterpreter()
    interp.visit(NeuvaParser().parse(source))
    return interp


def test_all_example_files_still_parse():
    files = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.nva")))
    assert len(files) > 0
    for path in files:
        NeuvaParser().parse_file(path)  # raises on failure


def test_dict_literal_immediately_after_if_block():
    interp = _run(
        'let x = 1\n'
        'if x == 1 {\n'
        '    print "in if"\n'
        '}\n'
        'let d = {"a": 1}\n'
    )
    assert interp.env.get("d") == {"a": 1}


def test_dict_literal_immediately_after_for_block():
    interp = _run(
        'for i in range(0, 2) {\n'
        '    print i\n'
        '}\n'
        'let d = {"k": 2}\n'
    )
    assert interp.env.get("d") == {"k": 2}


def test_match_with_parenthesized_scrutinee(capsys):
    _run(
        'fn f() {\n'
        '    return 1\n'
        '}\n'
        'match (f()) {\n'
        '    case 1: print "one"\n'
        '    default: print "other"\n'
        '}\n'
    )
    assert capsys.readouterr().out.strip() == "one"


def test_model_block_unaffected_by_dict_grammar():
    interp = _run(
        'model M {\n'
        '    layer dense(4 -> 3, relu)\n'
        '}\n'
    )
    from neuva.parser.ast_nodes import ModelStatement
    assert isinstance(interp.env.get("M"), ModelStatement)


def test_nested_dict_and_list_literal():
    interp = _run('let d = {"a": [1, 2, 3], "b": {"c": 4}}\n')
    assert interp.env.get("d") == {"a": [1, 2, 3], "b": {"c": 4}}


def test_if_condition_itself_a_dict_expression():
    interp = _run(
        'let d = {"flag": true}\n'
        'if d["flag"] {\n'
        '    let x = 1\n'
        '} else {\n'
        '    let x = 0\n'
        '}\n'
    )
    assert interp.env.get("x") == 1


def test_match_block_followed_by_dict_literal():
    interp = _run(
        'let n = 1\n'
        'match n {\n'
        '    case 1: print "one"\n'
        '}\n'
        'let d = {"after": "match"}\n'
    )
    assert interp.env.get("d") == {"after": "match"}
