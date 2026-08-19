import os
import torch
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


def load_csv(path: str, n_targets: int = 1) -> Dict[str, Any]:
    """Read a CSV and return {"X": tensor, "y": tensor, "columns": list}.

    With n_targets=1 (the default), y is a 1D tensor — long dtype when the target
    column contains only integers (classification), float32 otherwise (regression).
    With n_targets>1, the last `n_targets` numeric columns become a 2D float32 y
    tensor of shape (N, n_targets), one column per target — used for multi-output
    models where each column feeds a different output head.
    """
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < n_targets + 1:
        raise ValueError(
            f"'{path}' needs at least {n_targets + 1} numeric columns (features + "
            f"{n_targets} target(s)); found {numeric.shape[1]}"
        )

    feature_cols = numeric.columns[:-n_targets].tolist()
    target_cols = numeric.columns[-n_targets:].tolist()

    X = torch.tensor(numeric[feature_cols].values, dtype=torch.float32)

    if n_targets == 1:
        target_col = target_cols[0]
        if pd.api.types.is_integer_dtype(numeric[target_col]):
            y = torch.tensor(numeric[target_col].values, dtype=torch.long)
        else:
            y = torch.tensor(numeric[target_col].values, dtype=torch.float32)
    else:
        y = torch.tensor(numeric[target_cols].values, dtype=torch.float32)

    return {"X": X, "y": y, "columns": feature_cols + target_cols}


class DataSet:
    def __init__(
        self,
        name: str = "dataset",
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        path: Optional[str] = None,
        n_targets: int = 1,
    ) -> None:
        if path is not None and os.path.isfile(path):
            data = load_csv(path, n_targets=n_targets)
            self.name = os.path.splitext(os.path.basename(path))[0]
            self.X = data["X"]
            self.y = data["y"]
            self.columns: List[str] = data["columns"]
        elif path is not None:
            raise FileNotFoundError(f"Data file not found: '{path}'")
        else:
            self.name = name
            self.X = X
            self.y = y
            self.columns = []

    def split(self, ratio: float) -> Tuple["DataSet", "DataSet"]:
        """Return (train, test) using a random row permutation; ratio is the train fraction."""
        if self.X is None:
            return DataSet(f"{self.name}_train"), DataSet(f"{self.name}_test")
        n = len(self.X)
        idx = torch.randperm(n)
        n_train = max(1, int(n * ratio))
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        train = DataSet(f"{self.name}_train", self.X[train_idx], self.y[train_idx])
        test = DataSet(f"{self.name}_test", self.X[test_idx], self.y[test_idx])
        train.columns = self.columns
        test.columns = self.columns
        return train, test

    def normalize(self) -> "DataSet":
        """Z-score normalize features (and float regression targets, if present) to
        zero mean/unit variance; a zero-variance feature is left unscaled (divided by 1)
        instead of producing NaN/inf."""
        if self.X is None:
            return self
        mean = self.X.mean(dim=0)
        std = self.X.std(dim=0)
        std[std == 0] = 1.0
        X_norm = (self.X - mean) / std
        # Normalize continuous targets (float32) for regression; leave class indices (long) unchanged.
        if self.y is not None and self.y.dtype == torch.float32:
            y_std = self.y.std()
            y_norm = (self.y - self.y.mean()) / (
                y_std if y_std.item() != 0 else torch.tensor(1.0)
            )
        else:
            y_norm = self.y
        result = DataSet(self.name, X_norm, y_norm)
        result.columns = self.columns
        return result

    def shuffle(self) -> "DataSet":
        if self.X is None:
            return self
        idx = torch.randperm(len(self.X))
        result = DataSet(self.name, self.X[idx], self.y[idx])
        result.columns = self.columns
        return result

    def augment(self) -> "DataSet":
        """Random horizontal flip + rotation for image-shaped data (dim >= 3);
        falls back to simple noise addition otherwise or if torchvision is missing."""
        if self.X is None:
            return self
        X_aug = None
        if self.X.dim() >= 3:
            try:
                import random
                import torchvision.transforms.functional as TF

                augmented = []
                for img in self.X:
                    if random.random() < 0.5:
                        img = TF.hflip(img)
                    img = TF.rotate(img, random.uniform(-15, 15))
                    augmented.append(img)
                X_aug = torch.stack(augmented)
            except ImportError:
                X_aug = None
        if X_aug is None:
            X_aug = self.X + torch.randn_like(self.X) * 0.05
        result = DataSet(self.name, X_aug, self.y)
        result.columns = self.columns
        return result

    def _class_indices(self) -> Tuple[Dict[Any, torch.Tensor], torch.Tensor]:
        """Return ({class_value: row_indices_tensor}, per-class_counts) for `self.y`."""
        classes, counts = torch.unique(self.y, return_counts=True)
        return {
            c.item(): (self.y == c).nonzero(as_tuple=True)[0] for c in classes
        }, counts

    def oversample(self) -> "DataSet":
        """Duplicate minority-class samples (with replacement) so every class matches
        the majority class's count. No-op for regression targets (non-integer y)."""
        if self.X is None or self.y is None or self.y.dtype != torch.long:
            return self
        by_class, counts = self._class_indices()
        target_count = counts.max().item()
        idx_parts = []
        for cls_idx in by_class.values():
            idx_parts.append(cls_idx)
            deficit = target_count - len(cls_idx)
            if deficit > 0:
                extra = cls_idx[torch.randint(0, len(cls_idx), (deficit,))]
                idx_parts.append(extra)
        all_idx = torch.cat(idx_parts)
        all_idx = all_idx[torch.randperm(len(all_idx))]
        result = DataSet(self.name, self.X[all_idx], self.y[all_idx])
        result.columns = self.columns
        return result

    def undersample(self) -> "DataSet":
        """Randomly drop majority-class samples so every class matches the minority
        class's count. No-op for regression targets (non-integer y)."""
        if self.X is None or self.y is None or self.y.dtype != torch.long:
            return self
        by_class, counts = self._class_indices()
        target_count = counts.min().item()
        idx_parts = []
        for cls_idx in by_class.values():
            keep = cls_idx[torch.randperm(len(cls_idx))[:target_count]]
            idx_parts.append(keep)
        all_idx = torch.cat(idx_parts)
        all_idx = all_idx[torch.randperm(len(all_idx))]
        result = DataSet(self.name, self.X[all_idx], self.y[all_idx])
        result.columns = self.columns
        return result

    def __len__(self) -> int:
        return len(self.X) if self.X is not None else 0

    def __repr__(self) -> str:
        return f"DataSet({self.name!r}, {len(self)} rows)"


def load_iris_dataset() -> DataSet:
    """The classic Iris flower dataset (150 rows, 4 features, 3 classes) via sklearn —
    bundled with scikit-learn, no download required."""
    from sklearn.datasets import load_iris

    data = load_iris()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)
    ds = DataSet(name="iris", X=X, y=y)
    ds.columns = list(data.feature_names) + ["target"]
    return ds


def load_housing_dataset() -> DataSet:
    """The California housing regression dataset (20640 rows, 8 features) via sklearn.

    Unlike load_iris/load_mnist_sample, this one is not bundled with scikit-learn — the
    first call downloads and caches it (~/scikit_learn_data), so it requires network
    access once.
    """
    from sklearn.datasets import fetch_california_housing

    try:
        data = fetch_california_housing()
    except Exception as exc:
        raise RuntimeError(
            "load_housing() needs to download the California housing dataset "
            f"(no network access available right now): {exc}"
        ) from exc
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32)
    ds = DataSet(name="housing", X=X, y=y)
    ds.columns = list(data.feature_names) + ["target"]
    return ds


def load_mnist_sample_dataset(n_rows: int = 100) -> DataSet:
    """A small (n_rows-row) handwritten-digit sample via sklearn's bundled `load_digits`
    (8x8 images, no download required) — a lightweight offline stand-in for full MNIST
    (28x28, ~70k rows, not bundled with sklearn)."""
    from sklearn.datasets import load_digits

    data = load_digits()
    n_rows = min(n_rows, len(data.data))
    X = torch.tensor(data.data[:n_rows], dtype=torch.float32)
    y = torch.tensor(data.target[:n_rows], dtype=torch.long)
    ds = DataSet(name="mnist_sample", X=X, y=y)
    ds.columns = [f"pixel{i}" for i in range(X.shape[1])] + ["target"]
    return ds
