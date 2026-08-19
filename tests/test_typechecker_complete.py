"""Tests for type checker rules added in the v1.1.0 completion pass: return-type
checking, list/dict indexing, for-loop shadow warnings, train lr/epochs validation,
and the --strict CLI flag / summary line."""
import os
import subprocess
import sys
import tempfile

from neuva.parser import NeuvaParser
from neuva.typechecker import TypeChecker

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _check(src):
    tree = NeuvaParser().parse(src)
    checker = TypeChecker()
    errors = checker.check(tree)
    return checker, errors


# ── return type checking ─────────────────────────────────────────────────────

def test_return_type_mismatch_is_error():
    _, errors = _check(
        'fn f() -> int {\n    return "hello"\n}\n'
    )
    messages = [str(e) for e in errors]
    assert any("declares return type 'int'" in m and "'string' value" in m for m in messages)


def test_return_type_match_has_no_error():
    _, errors = _check(
        'fn f() -> int {\n    return 5\n}\nlet x = f()\n'
    )
    assert errors == []


def test_return_type_not_checked_for_non_literal_returns():
    # can't statically know the type of a variable/expression return, so no false positive
    _, errors = _check(
        'fn f(x) -> int {\n    return x\n}\nlet y = f(5)\n'
    )
    assert errors == []


def test_return_type_checked_through_if_branches():
    _, errors = _check(
        'fn f(cond) -> int {\n'
        '    if cond {\n'
        '        return "oops"\n'
        '    }\n'
        '    return 1\n'
        '}\n'
    )
    messages = [str(e) for e in errors]
    assert any("declares return type 'int'" in m for m in messages)


# ── list/dict indexing ───────────────────────────────────────────────────────

def test_indexing_int_literal_is_error():
    _, errors = _check('print 5[0]\n')
    messages = [str(e) for e in errors]
    assert any("cannot index" in m for m in messages)


def test_indexing_list_literal_is_fine():
    _, errors = _check('let x = [1, 2, 3][0]\n')
    assert errors == []


def test_indexing_variable_is_not_flagged():
    # can't know a variable's runtime type statically, so no false positive
    _, errors = _check('let d = {"a": 1}\nlet x = d["a"]\n')
    assert errors == []


# ── for-loop shadow warning ──────────────────────────────────────────────────

def test_for_loop_shadow_produces_warning_not_error():
    checker, errors = _check('let i = 5\nfor i in range(0, 3) {\n    print i\n}\n')
    assert errors == []
    assert any("shadows" in str(w) for w in checker.warnings)


def test_for_loop_no_shadow_no_warning():
    checker, errors = _check('for i in range(0, 3) {\n    print i\n}\n')
    assert errors == []
    assert checker.warnings == []


# ── train lr / epochs validation ─────────────────────────────────────────────

def test_train_zero_epochs_is_error():
    _, errors = _check(
        'model M {\n    layer dense(2 -> 2, relu)\n}\n'
        'let data = load("x.csv")\n'
        'train M on data for 0 epochs, lr = 0.01\n'
    )
    messages = [str(e) for e in errors]
    assert any("epochs must be a positive integer" in m for m in messages)


def test_train_zero_lr_is_error():
    _, errors = _check(
        'model M {\n    layer dense(2 -> 2, relu)\n}\n'
        'let data = load("x.csv")\n'
        'train M on data for 5 epochs, lr = 0.0\n'
    )
    messages = [str(e) for e in errors]
    assert any("lr must be a positive number" in m for m in messages)


def test_train_lr_as_variable_is_not_checked_statically():
    _, errors = _check(
        'model M {\n    layer dense(2 -> 2, relu)\n}\n'
        'let data = load("x.csv")\n'
        'let candidate_lr = 0.01\n'
        'train M on data for 5 epochs, lr = candidate_lr\n'
    )
    assert errors == []


def test_train_zero_early_stop_is_error():
    _, errors = _check(
        'model M {\n    layer dense(2 -> 2, relu)\n}\n'
        'let data = load("x.csv")\n'
        'train M on data for 5 epochs, lr = 0.01, early_stop = 0\n'
    )
    messages = [str(e) for e in errors]
    assert any("early_stop must be a positive integer" in m for m in messages)


def test_train_zero_lr_warmup_is_error():
    _, errors = _check(
        'model M {\n    layer dense(2 -> 2, relu)\n}\n'
        'let data = load("x.csv")\n'
        'train M on data for 5 epochs, lr = 0.01, lr_warmup = 0\n'
    )
    messages = [str(e) for e in errors]
    assert any("lr_warmup must be a positive integer" in m for m in messages)


# ── layer dimension mismatch (pre-existing, regression-covered here) ────────

def test_layer_dimension_mismatch_still_detected():
    _, errors = _check(
        'model M {\n    layer dense(4 -> 8, relu)\n    layer dense(16 -> 3, softmax)\n}\n'
    )
    messages = [str(e) for e in errors]
    assert any("dimension mismatch" in m for m in messages)


# ── model-name checking doesn't false-positive on load()-bound variables ────

def test_predict_on_loaded_variable_is_not_flagged():
    _, errors = _check(
        'let reloaded = load("model.nva")\n'
        'let data = load("x.csv")\n'
        'predict reloaded on data\n'
    )
    assert errors == []


def test_train_on_non_model_literal_is_still_flagged():
    _, errors = _check(
        'let Foo = 5\nlet data = load("x.csv")\ntrain Foo on data for 5 epochs, lr = 0.01\n'
    )
    messages = [str(e) for e in errors]
    assert any("is used as a model but is not a model definition" in m for m in messages)


# ── CLI: --strict flag and summary line ──────────────────────────────────────

def _run_cli(src, extra_args=None):
    with tempfile.NamedTemporaryFile("w", suffix=".nva", delete=False, dir=REPO_ROOT) as f:
        f.write(src)
        path = f.name
    try:
        args = [sys.executable, "-m", "neuva.cli", path] + (extra_args or [])
        return subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace", cwd=REPO_ROOT)
    finally:
        os.unlink(path)


def test_cli_clean_program_prints_summary_line():
    result = _run_cli('let x = 1\nprint x\n')
    assert result.returncode == 0
    assert "Type check passed: 0 warnings, 0 errors" in result.stdout


def test_cli_warning_without_strict_exits_zero():
    result = _run_cli('let i = 5\nfor i in range(0, 3) {\n    print i\n}\n')
    assert result.returncode == 0
    assert "shadows" in result.stderr
    assert "Type check passed: 1 warnings, 0 errors" in result.stdout


def test_cli_warning_with_strict_exits_nonzero():
    result = _run_cli('let i = 5\nfor i in range(0, 3) {\n    print i\n}\n', ["--strict"])
    assert result.returncode == 1
    assert "shadows" in result.stderr
