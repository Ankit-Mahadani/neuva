import os
import subprocess
import sys

from neuva.parser import NeuvaParser
from neuva.typechecker import TypeChecker

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _check(src):
    tree = NeuvaParser().parse(src)
    return TypeChecker().check(tree)


def test_valid_program_has_no_errors():
    errors = _check(
        'model M {\n    layer dense(4 -> 8, relu)\n    layer dense(8 -> 3, softmax)\n}\n'
        'let data = load("examples/data/iris.csv")\n'
        'train M on data for 5 epochs, lr=0.01, loss=crossentropy\n'
        'let acc = accuracy(M, data)\nprint acc\n'
    )
    assert errors == []


def test_undefined_variable_is_caught():
    errors = _check("let x = y\n")
    assert len(errors) == 1
    assert "undefined variable 'y'" in str(errors[0])


def test_undefined_model_in_train_is_caught():
    errors = _check('let data = load("x.csv")\ntrain Ghost on data for 5 epochs, lr=0.01, loss=mse\n')
    messages = [str(e) for e in errors]
    assert any("undefined model 'Ghost'" in m for m in messages)


def test_non_model_used_as_model_is_caught():
    errors = _check('let Foo = 5\nlet data = load("x.csv")\ntrain Foo on data for 5 epochs, lr=0.01, loss=mse\n')
    messages = [str(e) for e in errors]
    assert any("is used as a model but is not a model definition" in m for m in messages)


def test_invalid_loss_name_is_caught():
    errors = _check(
        'model M {\n    layer dense(4 -> 3, softmax)\n}\n'
        'let data = load("x.csv")\n'
        'train M on data for 5 epochs, lr=0.01, loss=not_a_real_loss\n'
    )
    messages = [str(e) for e in errors]
    assert any("not a valid loss" in m for m in messages)


def test_custom_fn_loss_is_accepted():
    errors = _check(
        'fn my_loss(pred, target) {\n    return mse(pred, target)\n}\n'
        'model M {\n    layer dense(4 -> 3, softmax)\n}\n'
        'let data = load("x.csv")\n'
        'train M on data for 5 epochs, lr=0.01, loss=my_loss\n'
    )
    assert errors == []


def test_dense_layer_dimension_mismatch_is_caught():
    errors = _check('model M {\n    layer dense(4 -> 8, relu)\n    layer dense(16 -> 3, softmax)\n}\n')
    messages = [str(e) for e in errors]
    assert any("dimension mismatch" in m for m in messages)


def test_matching_dense_dimensions_pass():
    errors = _check('model M {\n    layer dense(4 -> 8, relu)\n    layer dense(8 -> 3, softmax)\n}\n')
    assert errors == []


def test_stdlib_import_registers_names():
    errors = _check('import metrics\nlet f = f1_from_pr(0.8, 0.6)\nprint f\n')
    assert errors == []


def test_unknown_stdlib_module_is_caught():
    errors = _check("import does_not_exist\n")
    messages = [str(e) for e in errors]
    assert any("stdlib module" in m for m in messages)


def test_function_body_may_forward_reference_later_globals():
    # fn bodies run later via a shared closure environment, so referencing a global
    # defined further down the file is legitimate and should not be flagged.
    errors = _check(
        'fn use_later() {\n    return later_value\n}\n'
        'let later_value = 42\n'
    )
    assert errors == []


def test_all_example_files_pass_typecheck_except_intentional_error_demo():
    failures = {}
    for fname in sorted(os.listdir(EXAMPLES_DIR)):
        if not fname.endswith(".nva"):
            continue
        if fname == "error_demo.nva":
            continue  # intentionally contains a typo to demonstrate error messages
        path = os.path.join(EXAMPLES_DIR, fname)
        tree = NeuvaParser().parse_file(path)
        errors = TypeChecker().check(tree)
        if errors:
            failures[fname] = [str(e) for e in errors]
    assert failures == {}


def test_cli_exits_nonzero_on_type_error(tmp_path):
    bad_file = tmp_path / "bad.nva"
    bad_file.write_text(
        "model M {\n    layer dense(4 -> 8, relu)\n    layer dense(16 -> 3, softmax)\n}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "neuva.cli", str(bad_file)],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode != 0
    assert "dimension mismatch" in result.stderr


def test_cli_no_check_flag_skips_typechecking(tmp_path):
    bad_file = tmp_path / "bad.nva"
    bad_file.write_text(
        "model M {\n    layer dense(4 -> 8, relu)\n    layer dense(16 -> 3, softmax)\n}\nprint M\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "neuva.cli", str(bad_file), "--no-check"],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert "dimension mismatch" not in result.stderr
    assert result.returncode == 0
