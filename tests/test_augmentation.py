import os
import torch

from neuva.backend.data_loader import DataSet
from neuva.parser import NeuvaParser
from neuva.interpreter.interpreter import NeuvaInterpreter

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def _imbalanced_dataset():
    X = torch.randn(20, 4)
    y = torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(5, dtype=torch.long)])
    return DataSet("test", X, y)


def test_oversample_balances_classes():
    ds = _imbalanced_dataset()
    balanced = ds.oversample()
    counts = torch.bincount(balanced.y)
    assert counts[0].item() == counts[1].item() == 15
    assert len(balanced) == 30


def test_undersample_balances_classes():
    ds = _imbalanced_dataset()
    balanced = ds.undersample()
    counts = torch.bincount(balanced.y)
    assert counts[0].item() == counts[1].item() == 5
    assert len(balanced) == 10


def test_oversample_noop_for_regression_targets():
    X = torch.randn(10, 3)
    y = torch.randn(10)  # float target -> regression, not classification
    ds = DataSet("reg", X, y)
    result = ds.oversample()
    assert len(result) == len(ds)


def test_augment_tabular_uses_noise_fallback():
    X = torch.zeros(10, 4)
    y = torch.zeros(10, dtype=torch.long)
    ds = DataSet("tab", X, y)
    aug = ds.augment()
    assert aug.X.shape == X.shape
    assert not torch.equal(aug.X, X)  # noise was added


def test_augment_image_shaped_data_preserves_shape():
    X = torch.rand(6, 3, 8, 8)
    y = torch.zeros(6, dtype=torch.long)
    ds = DataSet("imgs", X, y)
    aug = ds.augment()
    assert aug.X.shape == X.shape


def test_augment_empty_dataset_is_noop():
    ds = DataSet("empty")
    assert ds.augment() is ds


def test_augmentation_demo_example_runs():
    path = os.path.join(EXAMPLES_DIR, "augmentation_demo.nva")
    tree = NeuvaParser().parse_file(path)
    NeuvaInterpreter().visit(tree)
