import os
import torch

from neuva.parser import NeuvaParser
from neuva.parser.ast_nodes import LayerStatement
from neuva.backend.torch_backend import NeuvaModel, NeuvaTrainer
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_early_stop_parses():
    src = "train M on data for 100 epochs, lr=0.001, loss=mse, early_stop=10\n"
    stmt = NeuvaParser().parse(src).body[0]
    opts = {o.key: o.value for o in stmt.options}
    assert opts["early_stop"] == 10


def test_no_early_stop_defaults_to_none():
    src = "train M on data for 100 epochs, lr=0.001, loss=mse\n"
    stmt = NeuvaParser().parse(src).body[0]
    assert all(o.key != "early_stop" for o in stmt.options)


def test_early_stop_halts_before_epoch_limit(capsys):
    torch.manual_seed(0)
    layers = [LayerStatement(name="dense", args=[4, 1, "linear"])]
    model = NeuvaModel(layers)

    class FakeDS:
        X = torch.zeros(16, 4)
        y = torch.zeros(16, 1)

    NeuvaTrainer().train(model, FakeDS(), epochs=200, lr=0.05, loss_fn="mse", early_stop=3)
    out = capsys.readouterr().out
    assert "Early stopping" in out
    assert "Epoch 200/200" not in out


def test_no_early_stop_runs_full_epochs(capsys):
    layers = [LayerStatement(name="dense", args=[4, 1, "linear"])]
    model = NeuvaModel(layers)

    class FakeDS:
        X = torch.randn(16, 4)
        y = torch.randn(16, 1)

    NeuvaTrainer().train(model, FakeDS(), epochs=5, lr=0.01, loss_fn="mse")
    out = capsys.readouterr().out
    assert "Epoch 5/5" in out
    assert "Early stopping" not in out


def test_early_stop_in_train_statement_runs():
    src = """
model M {
    layer dense(4 -> 3, softmax)
}
let data = load("examples/data/iris.csv")
train M on data for 50 epochs, lr=0.05, loss=crossentropy, early_stop=3
"""
    tree = NeuvaParser().parse(src)
    NeuvaInterpreter().visit(tree)


def test_scheduler_demo_example_has_early_stop_option():
    path = os.path.join(EXAMPLES_DIR, "scheduler_early_stop_demo.nva")
    tree = NeuvaParser().parse_file(path)
    train_stmts = [s for s in tree.body if type(s).__name__ == "TrainStatement"]
    assert train_stmts
    opts = {o.key for o in train_stmts[0].options}
    assert "early_stop" in opts
