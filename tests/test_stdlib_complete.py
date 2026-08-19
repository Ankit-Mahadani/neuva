"""Tests for the stdlib modules added in the v1.1.0 completion pass: datasets,
optimizers, callbacks."""
import os

from neuva.parser import NeuvaParser
from neuva.interpreter.interpreter import NeuvaInterpreter

STDLIB_DIR = os.path.join(os.path.dirname(__file__), "..", "neuva", "stdlib")


def _run(source: str) -> NeuvaInterpreter:
    interp = NeuvaInterpreter()
    interp.visit(NeuvaParser().parse(source))
    return interp


def test_new_stdlib_files_exist():
    for name in ("datasets", "optimizers", "callbacks"):
        assert os.path.isfile(os.path.join(STDLIB_DIR, f"{name}.nva"))


def test_new_stdlib_files_parse():
    for name in ("datasets", "optimizers", "callbacks"):
        tree = NeuvaParser().parse_file(os.path.join(STDLIB_DIR, f"{name}.nva"))
        assert tree is not None


# ── datasets ─────────────────────────────────────────────────────────────────

def test_import_datasets_defines_functions():
    interp = _run("import datasets\n")
    assert interp.env.get("load_iris") is not None
    assert interp.env.get("load_housing") is not None
    assert interp.env.get("load_mnist_sample") is not None


def test_load_iris_returns_150_rows():
    interp = _run("import datasets\nlet data = load_iris()\nlet n = len(data)\n")
    assert interp.env.get("n") == 150


def test_load_mnist_sample_returns_100_rows_by_default():
    interp = _run("import datasets\nlet data = load_mnist_sample()\nlet n = len(data)\n")
    assert interp.env.get("n") == 100


def test_load_iris_dataset_is_trainable():
    interp = _run(
        'import datasets\n'
        'let data = load_iris()\n'
        'let train_data, test_data = data.split(0.8)\n'
        'model Net {\n'
        '    layer dense(4 -> 8, relu)\n'
        '    layer dense(8 -> 3, softmax)\n'
        '}\n'
        'train Net on train_data for 2 epochs, lr = 0.01, loss = crossentropy\n'
        'let acc = accuracy(Net, test_data)\n'
    )
    acc = interp.env.get("acc")
    assert 0.0 <= acc <= 1.0


# ── optimizers ───────────────────────────────────────────────────────────────

def test_import_optimizers_defines_functions():
    interp = _run("import optimizers\n")
    assert interp.env.get("use_sgd") is not None
    assert interp.env.get("use_adam") is not None
    assert interp.env.get("use_rmsprop") is not None


def test_use_sgd_sets_pending_optimizer():
    interp = _run("import optimizers\nuse_sgd(0.05, 0.9)\n")
    assert interp.pending_optimizer == ("sgd", {"lr": 0.05, "momentum": 0.9})


def test_use_adam_sets_pending_optimizer():
    interp = _run("import optimizers\nuse_adam(0.01, 0.8, 0.99)\n")
    assert interp.pending_optimizer == ("adam", {"lr": 0.01, "betas": (0.8, 0.99)})


def test_optimizer_override_is_consumed_after_one_train_call():
    interp = _run(
        'import optimizers\n'
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'model Net {\n'
        '    layer dense(4 -> 8, relu)\n'
        '    layer dense(8 -> 3, softmax)\n'
        '}\n'
        'use_sgd(0.05, 0.9)\n'
        'train Net on train_data for 1 epochs, lr = 0.01, loss = crossentropy\n'
    )
    assert interp.pending_optimizer is None


# ── callbacks ────────────────────────────────────────────────────────────────

def test_import_callbacks_defines_functions():
    interp = _run("import callbacks\n")
    assert interp.env.get("on_epoch_end") is not None
    assert interp.env.get("on_improvement") is not None


def test_on_epoch_end_is_called_each_epoch(capsys):
    _run(
        'import callbacks\n'
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'model Net {\n'
        '    layer dense(4 -> 8, relu)\n'
        '    layer dense(8 -> 3, softmax)\n'
        '}\n'
        'fn log_epoch(epoch, loss) {\n'
        '    print "cb-epoch", epoch\n'
        '}\n'
        'on_epoch_end(log_epoch)\n'
        'train Net on train_data for 3 epochs, lr = 0.01, loss = crossentropy\n'
    )
    out = capsys.readouterr().out
    assert out.count("cb-epoch") == 3


def test_callbacks_are_consumed_after_one_train_call():
    interp = _run(
        'import callbacks\n'
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'model Net {\n'
        '    layer dense(4 -> 8, relu)\n'
        '    layer dense(8 -> 3, softmax)\n'
        '}\n'
        'fn noop(epoch, loss) {\n'
        '    return 0\n'
        '}\n'
        'on_epoch_end(noop)\n'
        'train Net on train_data for 1 epochs, lr = 0.01, loss = crossentropy\n'
    )
    assert interp.epoch_end_callbacks == []
