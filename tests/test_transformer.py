import os
import torch

from neuva.parser import NeuvaParser
from neuva.parser.ast_nodes import LayerStatement, ModelStatement
from neuva.backend.torch_backend import NeuvaModel
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_embedding_layer_parses():
    src = "model M {\n    layer embedding(1000, 64)\n}\n"
    tree = NeuvaParser().parse(src)
    model = tree.body[0]
    assert isinstance(model, ModelStatement)
    assert model.layers[0].name == "embedding"
    assert model.layers[0].args == [1000, 64]


def test_attention_layer_parses():
    src = "model M {\n    layer attention(64, 4)\n}\n"
    tree = NeuvaParser().parse(src)
    model = tree.body[0]
    assert model.layers[0].name == "attention"
    assert model.layers[0].args == [64, 4]


def test_transformer_layer_parses():
    src = "model M {\n    layer transformer(64, 4, 128)\n}\n"
    tree = NeuvaParser().parse(src)
    model = tree.body[0]
    assert model.layers[0].name == "transformer"
    assert model.layers[0].args == [64, 4, 128]


def test_embedding_layer_builds_and_forwards():
    layers = [LayerStatement(name="embedding", args=[1000, 64])]
    model = NeuvaModel(layers)
    out = model(torch.randint(0, 1000, (5, 10)))
    assert out.shape == (5, 10, 64)


def test_embedding_accepts_float_indices():
    layers = [LayerStatement(name="embedding", args=[100, 16])]
    model = NeuvaModel(layers)
    out = model(torch.randint(0, 100, (3, 4)).float())
    assert out.shape == (3, 4, 16)


def test_attention_layer_builds_and_forwards():
    layers = [LayerStatement(name="embedding", args=[500, 32]), LayerStatement(name="attention", args=[32, 4])]
    model = NeuvaModel(layers)
    out = model(torch.randint(0, 500, (3, 8)))
    assert out.shape == (3, 32)  # pooled back to (batch, embed_dim)


def test_attention_accepts_2d_input():
    layers = [LayerStatement(name="attention", args=[16, 2])]
    model = NeuvaModel(layers)
    out = model(torch.randn(4, 16))
    assert out.shape == (4, 16)


def test_transformer_layer_full_pipeline():
    layers = [
        LayerStatement(name="embedding", args=[1000, 64]),
        LayerStatement(name="transformer", args=[64, 4, 128]),
        LayerStatement(name="flatten", args=[]),
        LayerStatement(name="dense", args=[64, 3, "softmax"]),
    ]
    model = NeuvaModel(layers)
    out = model(torch.randint(0, 1000, (5, 12)))
    assert out.shape == (5, 3)


def test_transformer_demo_example_runs():
    path = os.path.join(EXAMPLES_DIR, "transformer_demo.nva")
    tree = NeuvaParser().parse_file(path)
    NeuvaInterpreter().visit(tree)
