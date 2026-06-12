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
        self.activation_names: list[str] = []
        for layer in layers:
            if hasattr(layer, "args"):
                lname = getattr(layer, "name", "dense")
                args = layer.args
            else:
                lname, args = "dense", list(layer)  # (in, out, act_name) tuple

            if lname == "conv":
                in_ch, out_ch, kernel = args[0], args[1], args[2]
                self.linears.append(nn.Conv2d(in_ch, out_ch, kernel))
                self.activations.append(lambda x: x)
                self.activation_names.append("linear")
            elif lname == "pool":
                self.linears.append(nn.MaxPool2d(args[0]))
                self.activations.append(lambda x: x)
                self.activation_names.append("linear")
            elif lname == "flatten":
                self.linears.append(nn.Flatten())
                self.activations.append(lambda x: x)
                self.activation_names.append("linear")
            else:  # dense / any named linear layer
                in_size, out_size, act_name = args[0], args[1], args[2]
                self.linears.append(nn.Linear(in_size, out_size))
                self.activations.append(_ACTIVATIONS.get(act_name, lambda x: x))
                self.activation_names.append(act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, activate in zip(self.linears, self.activations):
            x = activate(linear(x))
        return x


class NeuvaDataset:
    """Wraps a DataSet and exposes torch tensors for training."""

    def __init__(self, dataset, in_size: int = 1, n_samples: int = 64):
        self.name = getattr(dataset, "name", "dataset")
        if getattr(dataset, "X", None) is not None:
            self.X = dataset.X
            self.y = dataset.y
        else:
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
                if data.y is not None and data.y.numel() > 0:
                    targets = data.y.squeeze().long()
                else:
                    targets = torch.randint(0, out_features, (len(data.X),))
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}/{epochs} — loss: {loss.item():.4f}")


def evaluate_accuracy(model: NeuvaModel, dataset) -> float:
    model.eval()
    with torch.no_grad():
        outputs = model(dataset.X)
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == dataset.y).sum().item()
        total = dataset.y.size(0)
        return correct / total


def evaluate(model: NeuvaModel, dataset) -> float:
    """Return accuracy (classification) or R² (regression) on dataset."""
    X = getattr(dataset, "X", None)
    y = getattr(dataset, "y", None)
    if X is None or y is None or len(X) == 0:
        return 0.0

    model.eval()
    with torch.no_grad():
        outputs = model(X)
    model.train()

    out_features = outputs.shape[1] if outputs.dim() > 1 else 1

    if out_features > 1:
        preds = outputs.argmax(dim=1)
        labels = y.squeeze().long()
        return (preds == labels).float().mean().item()

    # regression: R² score
    y_flat = y.squeeze()
    pred_flat = outputs.squeeze()
    ss_res = ((y_flat - pred_flat) ** 2).sum()
    ss_tot = ((y_flat - y_flat.mean()) ** 2).sum()
    if ss_tot.item() == 0.0:
        return 1.0
    return max(0.0, (1.0 - ss_res / ss_tot).item())


def save_model(model: NeuvaModel, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "activations": model.activation_names}, path)
    print(f"Model saved to '{path}'")


def load_model(path: str) -> NeuvaModel:
    checkpoint = torch.load(path, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        activation_names = checkpoint.get("activations", [])
    else:
        state_dict = checkpoint
        activation_names = []

    weight_keys = sorted(k for k in state_dict if k.endswith(".weight"))
    layers = [
        (state_dict[k].shape[1], state_dict[k].shape[0], activation_names[i] if i < len(activation_names) else "linear")
        for i, k in enumerate(weight_keys)
    ]
    model = NeuvaModel(layers)
    model.load_state_dict(state_dict)
    model.eval()
    return model
