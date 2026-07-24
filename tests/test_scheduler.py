import os

from neuva.parser import NeuvaParser
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_lr_schedule_step_parses():
    src = "train M on data for 10 epochs, lr=0.01, loss=mse, lr_schedule=step\n"
    stmt = NeuvaParser().parse(src).body[0]
    opts = {o.key: o.value for o in stmt.options}
    assert opts["lr_schedule"] == "step"


def test_lr_schedule_cosine_parses():
    src = "train M on data for 10 epochs, lr=0.01, loss=mse, lr_schedule=cosine\n"
    stmt = NeuvaParser().parse(src).body[0]
    opts = {o.key: o.value for o in stmt.options}
    assert opts["lr_schedule"] == "cosine"


def test_no_lr_schedule_defaults_to_none():
    src = "train M on data for 10 epochs, lr=0.01, loss=mse\n"
    stmt = NeuvaParser().parse(src).body[0]
    assert all(o.key != "lr_schedule" for o in stmt.options)


def test_train_with_step_schedule_runs_past_boundary():
    # step_size=10 halves lr every 10 epochs — run past that boundary without error.
    src = """
model M {
    layer dense(4 -> 3, softmax)
}
let data = load("examples/data/iris.csv")
train M on data for 12 epochs, lr=0.05, loss=crossentropy, lr_schedule=step
"""
    tree = NeuvaParser().parse(src)
    NeuvaInterpreter().visit(tree)  # should not raise


def test_train_with_cosine_schedule_runs():
    src = """
model M {
    layer dense(4 -> 3, softmax)
}
let data = load("examples/data/iris.csv")
train M on data for 5 epochs, lr=0.05, loss=crossentropy, lr_schedule=cosine
"""
    tree = NeuvaParser().parse(src)
    NeuvaInterpreter().visit(tree)


def test_scheduler_demo_example_runs():
    path = os.path.join(EXAMPLES_DIR, "scheduler_early_stop_demo.nva")
    tree = NeuvaParser().parse_file(path)
    NeuvaInterpreter().visit(tree)
