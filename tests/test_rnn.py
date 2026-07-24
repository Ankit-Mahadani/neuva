import os
import torch

from neuva.parser import NeuvaParser
from neuva.parser.ast_nodes import LayerStatement, ModelStatement
from neuva.backend.torch_backend import NeuvaModel
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_rnn_layer_parses():
    src = "model Seq {\n    layer rnn(10 -> 32)\n    layer dense(32 -> 2, softmax)\n}\n"
    tree = NeuvaParser().parse(src)
    model = tree.body[0]
    assert isinstance(model, ModelStatement)
    assert model.layers[0].name == "rnn"
    assert model.layers[0].args == [10, 32]


def test_lstm_layer_with_num_layers_parses():
    src = "model Seq {\n    layer lstm(10 -> 32, 3)\n}\n"
    tree = NeuvaParser().parse(src)
    model = tree.body[0]
    assert model.layers[0].name == "lstm"
    assert model.layers[0].args == [10, 32, 3]


def test_rnn_model_builds_and_forward():
    layers = [LayerStatement(name="rnn", args=[10, 16]), LayerStatement(name="dense", args=[16, 4, "softmax"])]
    model = NeuvaModel(layers)
    out = model(torch.randn(5, 10))
    assert out.shape == (5, 4)


def test_lstm_model_default_num_layers():
    layers = [LayerStatement(name="lstm", args=[8, 12])]
    model = NeuvaModel(layers)
    wrapper = model.linears[0]
    assert wrapper.kind == "lstm"
    assert wrapper.num_layers == 1


def test_lstm_model_explicit_num_layers():
    layers = [LayerStatement(name="lstm", args=[8, 12, 3])]
    model = NeuvaModel(layers)
    assert model.linears[0].num_layers == 3


def test_rnn_accepts_3d_sequence_input():
    layers = [LayerStatement(name="rnn", args=[6, 10]), LayerStatement(name="dense", args=[10, 2, "softmax"])]
    model = NeuvaModel(layers)
    out = model(torch.randn(4, 7, 6))  # (batch, seq_len, features)
    assert out.shape == (4, 2)


def test_rnn_demo_example_runs():
    path = os.path.join(EXAMPLES_DIR, "rnn_demo.nva")
    tree = NeuvaParser().parse_file(path)
    NeuvaInterpreter().visit(tree)
