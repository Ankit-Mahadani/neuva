import torch
import torch.nn as nn
import torch.optim as optim

from neuva.parser.ast_nodes import LayerStatement

_ACTIVATIONS = {
    "relu":     torch.relu,
    "sigmoid":  torch.sigmoid,
    "tanh":     torch.tanh,
    "softmax":  lambda x: torch.softmax(x, dim=-1),
    "linear":   lambda x: x,
}

_LOSSES = {
    "crossentropy": nn.CrossEntropyLoss,
    "mse":          nn.MSELoss,
    "mae":          nn.L1Loss,
}


class NeuvaModel(nn.Module):
    def __init__(self, layers: list):
        super().__init__()
        self.linears = nn.ModuleList()
        self.activations = []
        for layer in layers:
            in_size, out_size, act_name = layer.args[0], layer.args[1], layer.args[2]
            self.linears.append(nn.Linear(in_size, out_size))
            self.activations.append(_ACTIVATIONS.get(act_name, lambda x: x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, activate in zip(self.linears, self.activations):
            x = activate(linear(x))
        return x


class NeuvaDataset:
    """Wraps a DataSet placeholder and exposes torch tensors for training."""

    def __init__(self, dataset, in_size: int = 1, n_samples: int = 64):
        self.name = getattr(dataset, "name", "dataset")
        self.X = torch.randn(n_samples, in_size)
        self.y = torch.randn(n_samples, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuvaTrainer:
    def train(
        self,
        model: NeuvaModel,
        data: NeuvaDataset,
        epochs: int,
        lr: float = 0.001,
        loss_fn: str = "mse",
    ) -> None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = _LOSSES.get(loss_fn, nn.MSELoss)()

        out_features = model.linears[-1].out_features if model.linears else 1
        is_classification = isinstance(criterion, nn.CrossEntropyLoss)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            outputs = model(data.X)
            if is_classification:
                targets = torch.randint(0, out_features, (len(data.X),))
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}/{epochs} — loss: {loss.item():.4f}")
