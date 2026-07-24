import os
import torch

from neuva.parser import NeuvaParser
from neuva.parser.ast_nodes import LayerStatement, OutputLayerStatement
from neuva.backend.data_loader import DataSet
from neuva.backend.torch_backend import NeuvaModel, NeuvaTrainer, NeuvaDataset, evaluate_accuracy
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
DATA_DIR = os.path.join(EXAMPLES_DIR, "data")


def _multitask_model():
    trunk = [LayerStatement(name="dense", args=[4, 32, "relu"])]
    outputs = [
        OutputLayerStatement(output_name="classification", layer=LayerStatement(name="dense", args=[32, 3, "softmax"])),
        OutputLayerStatement(output_name="regression", layer=LayerStatement(name="dense", args=[32, 1, "linear"])),
    ]
    return NeuvaModel(trunk, outputs=outputs)


def test_multitask_dataset_loads_2d_target():
    ds = DataSet(path=os.path.join(DATA_DIR, "multitask.csv"), n_targets=2)
    assert ds.y.dim() == 2
    assert ds.y.shape[1] == 2


def test_multi_output_forward_returns_dict():
    model = _multitask_model()
    out = model(torch.randn(5, 4))
    assert isinstance(out, dict)
    assert set(out.keys()) == {"classification", "regression"}
    assert out["classification"].shape == (5, 3)
    assert out["regression"].shape == (5, 1)


def test_multitask_training_reduces_loss(capsys):
    model = _multitask_model()
    ds = DataSet(path=os.path.join(DATA_DIR, "multitask.csv"), n_targets=2)
    nds = NeuvaDataset(ds, in_size=4)
    NeuvaTrainer().train(model, nds, epochs=5, lr=0.05)
    out = capsys.readouterr().out
    losses = [float(line.split("loss:")[1]) for line in out.splitlines() if "loss:" in line]
    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_evaluate_accuracy_returns_per_head_dict():
    model = _multitask_model()
    ds = DataSet(path=os.path.join(DATA_DIR, "multitask.csv"), n_targets=2)
    nds = NeuvaDataset(ds, in_size=4)
    NeuvaTrainer().train(model, nds, epochs=5, lr=0.05)
    acc = evaluate_accuracy(model, nds)
    assert isinstance(acc, dict)
    assert set(acc.keys()) == {"classification", "regression"}
    assert 0.0 <= acc["classification"] <= 1.0


def test_multi_output_training_without_targets_raises():
    model = _multitask_model()

    class NoTargetDS:
        X = torch.randn(8, 4)
        y = None

    try:
        NeuvaTrainer().train(model, NoTargetDS(), epochs=1, lr=0.01)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_multitask_demo_example_runs_and_prints_dict(capsys):
    path = os.path.join(EXAMPLES_DIR, "multitask_demo.nva")
    tree = NeuvaParser().parse_file(path)
    NeuvaInterpreter().visit(tree)
    out = capsys.readouterr().out
    assert "Per-head accuracy" in out
    assert "classification" in out
    assert "regression" in out
