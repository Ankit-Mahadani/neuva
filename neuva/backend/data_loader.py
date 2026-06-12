import os
import torch
import pandas as pd
from typing import Optional


def load_csv(path: str) -> dict:
    """Read a CSV and return {"X": tensor, "y": tensor, "columns": list}.

    y dtype is long when the target column contains only integers (classification),
    float32 otherwise (regression).
    """
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError(
            f"'{path}' needs at least 2 numeric columns (features + target); "
            f"found {numeric.shape[1]}"
        )

    feature_cols = numeric.columns[:-1].tolist()
    target_col = numeric.columns[-1]

    X = torch.tensor(numeric[feature_cols].values, dtype=torch.float32)

    if pd.api.types.is_integer_dtype(numeric[target_col]):
        y = torch.tensor(numeric[target_col].values, dtype=torch.long)
    else:
        y = torch.tensor(numeric[target_col].values, dtype=torch.float32)

    return {"X": X, "y": y, "columns": feature_cols + [target_col]}


class DataSet:
    def __init__(
        self,
        name: str = "dataset",
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        path: Optional[str] = None,
    ):
        if path is not None and os.path.isfile(path):
            data = load_csv(path)
            self.name = os.path.splitext(os.path.basename(path))[0]
            self.X = data["X"]
            self.y = data["y"]
            self.columns: list = data["columns"]
        else:
            self.name = name
            self.X = X
            self.y = y
            self.columns = []

    def split(self, ratio: float):
        """Return (train, test) using a random row permutation; ratio is the train fraction."""
        if self.X is None:
            raise ValueError("Cannot split an empty DataSet")
        n = len(self.X)
        idx = torch.randperm(n)
        n_train = max(1, int(n * ratio))
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        train = DataSet(f"{self.name}_train", self.X[train_idx], self.y[train_idx])
        test = DataSet(f"{self.name}_test", self.X[test_idx], self.y[test_idx])
        train.columns = self.columns
        test.columns = self.columns
        return train, test

    def normalize(self):
        if self.X is None:
            return self
        mean = self.X.mean(dim=0)
        std = self.X.std(dim=0)
        std[std == 0] = 1.0
        result = DataSet(self.name, (self.X - mean) / std, self.y)
        result.columns = self.columns
        return result

    def shuffle(self):
        if self.X is None:
            return self
        idx = torch.randperm(len(self.X))
        result = DataSet(self.name, self.X[idx], self.y[idx])
        result.columns = self.columns
        return result

    def __len__(self) -> int:
        return len(self.X) if self.X is not None else 0

    def __repr__(self) -> str:
        return f"DataSet({self.name!r}, {len(self)} rows)"
