import os

from neuva.parser import NeuvaParser
from neuva.parser.ast_nodes import ImportStatement
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
STDLIB_DIR = os.path.join(os.path.dirname(__file__), "..", "neuva", "stdlib")


def test_import_statement_parses():
    tree = NeuvaParser().parse("import metrics\n")
    stmt = tree.body[0]
    assert isinstance(stmt, ImportStatement)
    assert stmt.module == "metrics"


def test_stdlib_files_exist():
    for name in ("metrics", "preprocessing", "visualization"):
        assert os.path.isfile(os.path.join(STDLIB_DIR, f"{name}.nva"))


def test_stdlib_files_parse():
    for name in ("metrics", "preprocessing", "visualization"):
        tree = NeuvaParser().parse_file(os.path.join(STDLIB_DIR, f"{name}.nva"))
        assert tree is not None


def test_import_metrics_defines_functions():
    interp = NeuvaInterpreter()
    interp.visit(NeuvaParser().parse("import metrics\n"))
    assert interp.env.get("f1_from_pr") is not None
    assert interp.env.get("average") is not None
    assert interp.env.get("report") is not None


def test_import_preprocessing_one_hot():
    interp = NeuvaInterpreter()
    interp.visit(NeuvaParser().parse('import preprocessing\nlet x = one_hot(2, 4)\n'))
    assert interp.env.get("x") == [0, 0, 1, 0]


def test_import_visualization_print_loss_curve(capsys):
    interp = NeuvaInterpreter()
    interp.visit(NeuvaParser().parse('import visualization\nprint_loss_curve([0.5, 0.25])\n'))
    out = capsys.readouterr().out
    assert "#" in out


def test_import_unknown_module_raises():
    from neuva.interpreter.interpreter import RuntimeError_
    interp = NeuvaInterpreter()
    try:
        interp.visit(NeuvaParser().parse("import does_not_exist\n"))
        assert False, "expected RuntimeError_"
    except RuntimeError_:
        pass


def test_stdlib_demo_example_runs():
    path = os.path.join(EXAMPLES_DIR, "stdlib_demo.nva")
    tree = NeuvaParser().parse_file(path)
    NeuvaInterpreter().visit(tree)
