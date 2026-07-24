import os
import torch

from neuva.parser import NeuvaParser
from neuva.parser.ast_nodes import LayerStatement, OutputLayerStatement
from neuva.backend.torch_backend import NeuvaModel, save_model, load_model
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def _roundtrip(tmp_path, model, x):
    path = str(tmp_path / "model.nva")
    save_model(model, path)
    loaded = load_model(path)
    out1 = model(x)
    out2 = loaded(x)
    if isinstance(out1, dict):
        assert out1.keys() == out2.keys()
        for k in out1:
            assert torch.allclose(out1[k], out2[k])
    else:
        assert torch.allclose(out1, out2)
    return loaded


def test_save_load_dense_model(tmp_path):
    model = NeuvaModel([
        LayerStatement(name="dense", args=[4, 8, "relu"]),
        LayerStatement(name="dense", args=[8, 3, "softmax"]),
    ])
    loaded = _roundtrip(tmp_path, model, torch.randn(5, 4))
    assert loaded.layer_specs == model.layer_specs


def test_save_load_rnn_model(tmp_path):
    model = NeuvaModel([
        LayerStatement(name="rnn", args=[10, 16]),
        LayerStatement(name="dense", args=[16, 4, "softmax"]),
    ])
    loaded = _roundtrip(tmp_path, model, torch.randn(5, 10))
    assert loaded.linears[0].kind == "rnn"
    assert loaded.linears[0].hidden_size == 16


def test_save_load_lstm_model(tmp_path):
    model = NeuvaModel([
        LayerStatement(name="lstm", args=[10, 16, 2]),
        LayerStatement(name="dense", args=[16, 4, "softmax"]),
    ])
    loaded = _roundtrip(tmp_path, model, torch.randn(5, 10))
    assert loaded.linears[0].kind == "lstm"
    assert loaded.linears[0].num_layers == 2


def test_save_load_multi_output_model(tmp_path):
    trunk = [LayerStatement(name="dense", args=[10, 64, "relu"])]
    outputs = [
        OutputLayerStatement(output_name="classification", layer=LayerStatement(name="dense", args=[64, 3, "softmax"])),
        OutputLayerStatement(output_name="regression", layer=LayerStatement(name="dense", args=[64, 1, "linear"])),
    ]
    model = NeuvaModel(trunk, outputs=outputs)
    loaded = _roundtrip(tmp_path, model, torch.randn(5, 10))
    assert set(loaded.output_names) == {"classification", "regression"}


def test_load_legacy_checkpoint_format(tmp_path):
    model = NeuvaModel([
        LayerStatement(name="dense", args=[4, 8, "relu"]),
        LayerStatement(name="dense", args=[8, 3, "softmax"]),
    ])
    path = str(tmp_path / "legacy.nva")
    torch.save({"state_dict": model.state_dict(), "activations": model.activation_names}, path)
    loaded = load_model(path)
    x = torch.randn(3, 4)
    assert torch.allclose(model(x), loaded(x))


def test_save_load_demo_example_runs(capsys):
    path = os.path.join(EXAMPLES_DIR, "save_load_demo.nva")
    tree = NeuvaParser().parse_file(path)
    NeuvaInterpreter().visit(tree)
    out = capsys.readouterr().out
    saved_path = os.path.join(os.getcwd(), "iris_model.nva")
    if os.path.isfile(saved_path):
        os.remove(saved_path)
    assert "Accuracy before save" in out
    assert "Accuracy after load" in out


def test_save_load_demo_accuracy_matches_before_and_after(capsys):
    path = os.path.join(EXAMPLES_DIR, "save_load_demo.nva")
    tree = NeuvaParser().parse_file(path)
    NeuvaInterpreter().visit(tree)
    out = capsys.readouterr().out
    saved_path = os.path.join(os.getcwd(), "iris_model.nva")
    if os.path.isfile(saved_path):
        os.remove(saved_path)

    def _extract(label):
        for line in out.splitlines():
            if line.startswith(label):
                return line.split(":", 1)[1].strip()
        return None

    before = _extract("Accuracy before save")
    after = _extract("Accuracy after load")
    assert before is not None and after is not None
    assert before == after
