import torch

from neuva.backend.torch_backend import NeuvaModel, precision, recall, f1_score, confusion_matrix
from neuva.parser.ast_nodes import LayerStatement
from neuva.interpreter.interpreter import NeuvaInterpreter


class _FakeDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


def _perfect_model_and_data():
    """A hand-set linear model that perfectly separates two clusters."""
    layers = [LayerStatement(name="dense", args=[2, 2, "softmax"])]
    model = NeuvaModel(layers)
    with torch.no_grad():
        model.linears[0].weight.copy_(torch.tensor([[-10.0, 0.0], [10.0, 0.0]]))
        model.linears[0].bias.zero_()
    X = torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])
    return model, _FakeDataset(X, y)


def test_precision_perfect_classifier():
    model, data = _perfect_model_and_data()
    assert precision(model, data) == [1.0, 1.0]


def test_recall_perfect_classifier():
    model, data = _perfect_model_and_data()
    assert recall(model, data) == [1.0, 1.0]


def test_f1_score_perfect_classifier():
    model, data = _perfect_model_and_data()
    assert f1_score(model, data) == [1.0, 1.0]


def test_confusion_matrix_perfect_classifier(capsys):
    model, data = _perfect_model_and_data()
    cm = confusion_matrix(model, data)
    assert cm == [[2, 0], [0, 2]]
    out = capsys.readouterr().out
    assert "Confusion Matrix" in out


def test_metrics_empty_dataset_returns_empty():
    layers = [LayerStatement(name="dense", args=[2, 2, "softmax"])]
    model = NeuvaModel(layers)
    data = _FakeDataset(None, None)
    assert precision(model, data) == []
    assert recall(model, data) == []
    assert f1_score(model, data) == []


def test_metrics_builtins_registered_in_interpreter():
    interp = NeuvaInterpreter()
    for name in ("precision", "recall", "f1_score", "confusion_matrix"):
        assert callable(interp.env.get(name))
