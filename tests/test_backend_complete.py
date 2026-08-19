"""Tests for backend features added in the v1.1.0 completion pass: batch norm layer,
predict/predict_proba, export_onnx, lr_warmup, freeze/unfreeze, and continued training."""
import os
import tempfile

import torch

from neuva.parser import NeuvaParser
from neuva.interpreter.interpreter import NeuvaInterpreter
from neuva.backend.torch_backend import NeuvaModel, NeuvaTrainer, NeuvaDataset

_MODEL_SRC = (
    'model Net {\n'
    '    layer dense(4 -> 8, relu)\n'
    '    layer norm(8)\n'
    '    layer dense(8 -> 3, softmax)\n'
    '}\n'
)


def _run(source: str) -> NeuvaInterpreter:
    interp = NeuvaInterpreter()
    interp.visit(NeuvaParser().parse(source))
    return interp


def _fake_dataset(n=32, in_size=4, n_classes=3):
    X = torch.randn(n, in_size)
    y = torch.randint(0, n_classes, (n,))
    return NeuvaDataset(type("DS", (), {"X": X, "y": y})())


# ── layer norm -> BatchNorm1d ────────────────────────────────────────────────

def test_norm_layer_maps_to_batchnorm1d():
    model = NeuvaModel([
        __import__("neuva.parser.ast_nodes", fromlist=["LayerStatement"]).LayerStatement(name="dense", args=[4, 8, "relu"]),
        __import__("neuva.parser.ast_nodes", fromlist=["LayerStatement"]).LayerStatement(name="norm", args=[8]),
    ])
    assert isinstance(model.linears[1], torch.nn.BatchNorm1d)
    assert model.linears[1].num_features == 8


def test_norm_layer_trains_and_saves_loads():
    interp = _run(
        _MODEL_SRC +
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'train Net on train_data for 2 epochs, lr = 0.01, loss = crossentropy\n'
    )
    model = interp.env.get("Net")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "net.nva")
        from neuva.backend.torch_backend import save_model, load_model
        save_model(model, path)
        reloaded = load_model(path)
    assert isinstance(reloaded.linears[1], torch.nn.BatchNorm1d)


# ── predict / predict_proba ──────────────────────────────────────────────────

def test_predict_returns_class_labels():
    interp = _run(
        _MODEL_SRC +
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'train Net on train_data for 2 epochs, lr = 0.01, loss = crossentropy\n'
        'let preds = predict(Net, test_data)\n'
    )
    preds = interp.env.get("preds")
    assert isinstance(preds, list)
    assert all(p in (0, 1, 2) for p in preds)


def test_predict_statement_prints_predictions(capsys):
    _run(
        _MODEL_SRC +
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'train Net on train_data for 2 epochs, lr = 0.01, loss = crossentropy\n'
        'predict Net on test_data\n'
    )
    assert "Predictions:" in capsys.readouterr().out


def test_predict_proba_returns_probabilities_summing_to_one():
    interp = _run(
        _MODEL_SRC +
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'train Net on train_data for 2 epochs, lr = 0.01, loss = crossentropy\n'
        'let probs = predict_proba(Net, test_data)\n'
    )
    probs = interp.env.get("probs")
    assert isinstance(probs, list)
    for row in probs:
        assert abs(sum(row) - 1.0) < 1e-4


# ── export_onnx ───────────────────────────────────────────────────────────────

def test_export_onnx_writes_a_file():
    interp = _run(
        _MODEL_SRC +
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'train Net on train_data for 1 epochs, lr = 0.01, loss = crossentropy\n'
    )
    model = interp.env.get("Net")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "net.onnx")
        from neuva.backend.torch_backend import export_onnx
        export_onnx(model, path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


def test_export_onnx_rejects_multi_output_model():
    import pytest
    from neuva.backend.torch_backend import export_onnx
    from neuva.parser.ast_nodes import LayerStatement, OutputLayerStatement

    model = NeuvaModel(
        [LayerStatement(name="dense", args=[4, 8, "relu"])],
        outputs=[OutputLayerStatement(output_name="out1", layer=LayerStatement(name="dense", args=[8, 2, "softmax"]))],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            export_onnx(model, os.path.join(tmpdir, "net.onnx"))


# ── lr_warmup ─────────────────────────────────────────────────────────────────

def test_lr_warmup_trains_without_error(capsys):
    _run(
        _MODEL_SRC +
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'train Net on train_data for 5 epochs, lr = 0.05, loss = crossentropy, lr_warmup = 3\n'
    )
    out = capsys.readouterr().out
    assert "Epoch 5/5" in out


def test_lr_warmup_actually_ramps_lr():
    model = NeuvaModel([__import__("neuva.parser.ast_nodes", fromlist=["LayerStatement"]).LayerStatement(name="dense", args=[4, 3, "softmax"])])
    seen_lrs = []
    orig_step = torch.optim.Adam.step

    def spy_step(self, *a, **k):
        seen_lrs.append(self.param_groups[0]["lr"])
        return orig_step(self, *a, **k)

    torch.optim.Adam.step = spy_step
    try:
        NeuvaTrainer().train(model, _fake_dataset(), epochs=4, lr=0.1, loss_fn="crossentropy", lr_warmup=4, batch_size=32)
    finally:
        torch.optim.Adam.step = orig_step
    # warmup epoch 1 should use a smaller lr than the final target
    assert seen_lrs[0] < 0.1


# ── freeze / unfreeze ─────────────────────────────────────────────────────────

def test_freeze_sets_requires_grad_false():
    model = NeuvaModel([__import__("neuva.parser.ast_nodes", fromlist=["LayerStatement"]).LayerStatement(name="dense", args=[4, 3, "softmax"])])
    model.freeze()
    assert all(not p.requires_grad for p in model.parameters())
    model.unfreeze()
    assert all(p.requires_grad for p in model.parameters())


def test_training_frozen_model_is_skipped_not_crashed(capsys):
    interp = _run(
        _MODEL_SRC +
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'train Net on train_data for 2 epochs, lr = 0.01, loss = crossentropy\n'
        'let acc_before = accuracy(Net, test_data)\n'
        'Net.freeze()\n'
        'train Net on train_data for 2 epochs, lr = 0.01, loss = crossentropy\n'
        'let acc_after = accuracy(Net, test_data)\n'
    )
    assert interp.env.get("acc_before") == interp.env.get("acc_after")
    assert "skipping training" in capsys.readouterr().out


def test_continued_training_reuses_same_weights_not_fresh_random_init():
    interp = _run(
        _MODEL_SRC +
        'let data = load("examples/data/iris.csv")\n'
        'let train_data, test_data = data.split(0.8)\n'
        'train Net on train_data for 3 epochs, lr = 0.05, loss = crossentropy\n'
    )
    model_after_first = interp.env.get("Net")
    first_weight = model_after_first.linears[0].weight.clone()

    interp.visit(NeuvaParser().parse(
        'train Net on train_data for 1 epochs, lr = 0.05, loss = crossentropy\n'
    ))
    model_after_second = interp.env.get("Net")
    assert model_after_second is model_after_first
    # weights should have moved (continued training), not been reset to fresh random init
    assert not torch.allclose(first_weight, model_after_second.linears[0].weight)
