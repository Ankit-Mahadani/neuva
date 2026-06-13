import os
import pytest
from neuva.parser import NeuvaParser
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def _iris_examples():
    paths = []
    for fname in sorted(os.listdir(EXAMPLES_DIR)):
        if not fname.endswith(".nva"):
            continue
        fpath = os.path.join(EXAMPLES_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as fh:
            if "iris.csv" in fh.read():
                paths.append(pytest.param(fpath, id=fname))
    return paths


@pytest.mark.parametrize("path", _iris_examples())
def test_iris_example_parses(path):
    parser = NeuvaParser()
    tree = parser.parse_file(path)
    assert tree is not None


@pytest.mark.parametrize("fname", ["spam_classifier.nva", "house_price.nva", "control_flow_demo.nva"])
def test_new_examples_parse(fname):
    parser = NeuvaParser()
    tree = parser.parse_file(os.path.join(EXAMPLES_DIR, fname))
    assert tree is not None


def test_control_flow_demo_interprets():
    fpath = os.path.join(EXAMPLES_DIR, "control_flow_demo.nva")
    parser = NeuvaParser()
    tree = parser.parse_file(fpath)
    interp = NeuvaInterpreter()
    interp.visit(tree)
