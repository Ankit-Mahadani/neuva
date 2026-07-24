import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics as _skmetrics

from neuva.parser.ast_nodes import LayerStatement, OutputLayerStatement

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ACTIVATIONS = {
    "relu":     torch.relu,
    "sigmoid":  torch.sigmoid,
    "tanh":     torch.tanh,
    "softmax":  lambda x: torch.softmax(x, dim=-1),
    "linear":   lambda x: x,
}

_LOSSES = {
    "crossentropy":        nn.CrossEntropyLoss,
    "binary_crossentropy": nn.BCELoss,
    "mse":                 nn.MSELoss,
    "mae":                 nn.L1Loss,
}

LOSS_NAMES = set(_LOSSES.keys())


# ── callable loss functions (usable as builtins and inside custom Neuva loss fns) ──

def _align_regression_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reshape/one-hot `target` so it matches `pred`'s shape for elementwise losses."""
    target = target.to(pred.dtype)
    if target.shape == pred.shape:
        return target
    if target.numel() == pred.numel():
        return target.view_as(pred)
    if pred.dim() > 1 and pred.shape[-1] > 1 and target.numel() == pred.shape[0]:
        # target looks like class indices but pred has multiple outputs — one-hot it
        return F.one_hot(target.long(), num_classes=pred.shape[-1]).to(pred.dtype)
    return target.view(-1, *([1] * (pred.dim() - 1))).expand_as(pred)


def mse_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, _align_regression_target(pred, target))


def mae_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, _align_regression_target(pred, target))


def crossentropy_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(pred, target.view(-1).long())


def binary_crossentropy_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(pred.dtype).view_as(pred)
    return F.binary_cross_entropy(pred, target)


LOSS_CALLABLES = {
    "mse": mse_fn,
    "mae": mae_fn,
    "crossentropy": crossentropy_fn,
    "binary_crossentropy": binary_crossentropy_fn,
}


class _RNNWrapper(nn.Module):
    """Wraps nn.RNN/nn.LSTM so it can chain into dense layers like any other layer.

    Accepts 2D (batch, features) input from Neuva's tabular DataSet — treated as a
    length-1 sequence — or proper 3D (batch, seq, features) input, and returns the
    final time-step's hidden state.
    """

    def __init__(self, kind: str, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.kind = kind
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        cls = nn.LSTM if kind == "lstm" else nn.RNN
        self.rnn = cls(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out_features = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        return out[:, -1, :]


class _EmbeddingWrapper(nn.Module):
    """nn.Embedding, cast-tolerant so a stray float32 index tensor doesn't crash it."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.out_features = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long())


class _AttentionWrapper(nn.Module):
    """Self-attention over nn.MultiheadAttention. Accepts (batch, embed) or
    (batch, seq, embed) input and mean-pools the sequence back to (batch, embed) so
    it composes with dense/flatten layers like every other layer in the pipeline."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_features = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.attn(x, x, x)
        return out.mean(dim=1)


class _TransformerWrapper(nn.Module):
    """A single nn.TransformerEncoderLayer, with the same auto-reshape/pool behavior
    as _AttentionWrapper so it composes with the rest of the layer pipeline."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True,
        )
        self.out_features = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.layer(x)
        return out.mean(dim=1)


class NeuvaModel(nn.Module):
    def __init__(self, layers: list, outputs: list = None):
        super().__init__()
        self.linears = nn.ModuleList()
        self.activations = []
        self.activation_names: list[str] = []
        # Serializable architecture description (name, args) per layer/output head —
        # kept alongside the built nn.Modules so save/load can reconstruct any
        # architecture (rnn/lstm/multi-output/etc), not just infer it from tensor shapes.
        self.layer_specs: list = []
        self.output_specs: list = []
        for layer in layers:
            if hasattr(layer, "args"):
                lname = getattr(layer, "name", "dense")
                args = layer.args
            else:
                lname, args = "dense", list(layer)  # (in, out, act_name) tuple
            self.layer_specs.append((lname, list(args)))

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
            elif lname == "dropout":
                p = float(args[0]) if args else 0.5
                self.linears.append(nn.Dropout(p))
                self.activations.append(lambda x: x)
                self.activation_names.append("linear")
            elif lname in ("rnn", "lstm"):
                input_size, hidden_size = args[0], args[1]
                num_layers = int(args[2]) if len(args) >= 3 else 1
                self.linears.append(_RNNWrapper(lname, input_size, hidden_size, num_layers))
                self.activations.append(lambda x: x)
                self.activation_names.append("linear")
            elif lname == "embedding":
                vocab_size, embed_dim = int(args[0]), int(args[1])
                self.linears.append(_EmbeddingWrapper(vocab_size, embed_dim))
                self.activations.append(lambda x: x)
                self.activation_names.append("linear")
            elif lname == "attention":
                embed_dim, num_heads = int(args[0]), int(args[1])
                self.linears.append(_AttentionWrapper(embed_dim, num_heads))
                self.activations.append(lambda x: x)
                self.activation_names.append("linear")
            elif lname == "transformer":
                embed_dim, num_heads, ff_dim = int(args[0]), int(args[1]), int(args[2])
                self.linears.append(_TransformerWrapper(embed_dim, num_heads, ff_dim))
                self.activations.append(lambda x: x)
                self.activation_names.append("linear")
            else:  # dense / any named linear layer
                in_size, out_size, act_name = args[0], args[1], args[2]
                self.linears.append(nn.Linear(in_size, out_size))
                self.activations.append(_ACTIVATIONS.get(act_name, lambda x: x))
                self.activation_names.append(act_name)

        self.output_names: list[str] = []
        self.output_heads = nn.ModuleDict()
        self.output_activation_names: dict = {}
        self._output_activation_fns: dict = {}
        for out in outputs or []:
            # accepts (name, LayerStatement) tuples or OutputLayerStatement objects
            if hasattr(out, "output_name"):
                oname, olayer = out.output_name, out.layer
            else:
                oname, olayer = out
            in_size, out_size, act_name = olayer.args[0], olayer.args[1], olayer.args[2]
            self.output_heads[oname] = nn.Linear(in_size, out_size)
            self.output_activation_names[oname] = act_name
            self._output_activation_fns[oname] = _ACTIVATIONS.get(act_name, lambda x: x)
            self.output_names.append(oname)
            self.output_specs.append((oname, olayer.name, [in_size, out_size, act_name]))

        self.to(DEVICE)

    def forward(self, x: torch.Tensor):
        for linear, activate in zip(self.linears, self.activations):
            x = activate(linear(x))
        if self.output_names:
            return {
                name: self._output_activation_fns[name](self.output_heads[name](x))
                for name in self.output_names
            }
        return x


class NeuvaDataset:
    """Wraps a DataSet and exposes torch tensors for training."""

    def __init__(self, dataset, in_size: int = 1, n_samples: int = 64):
        self.name = getattr(dataset, "name", "dataset")
        if getattr(dataset, "X", None) is not None:
            self.X = dataset.X.to(DEVICE)
            self.y = dataset.y.to(DEVICE)
        else:
            self.X = torch.randn(n_samples, in_size, device=DEVICE)
            self.y = torch.randn(n_samples, 1, device=DEVICE)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


_CLASSIFICATION_ACTIVATIONS = {"softmax", "sigmoid"}


def _head_kinds(model: "NeuvaModel"):
    """Split a multi-output model's heads into (regression_heads, classification_heads),
    auto-detected from each head's activation (softmax/sigmoid = classification, else regression)."""
    reg_heads, cls_heads = [], []
    for name in model.output_names:
        act = model.output_activation_names.get(name, "linear")
        (cls_heads if act in _CLASSIFICATION_ACTIVATIONS else reg_heads).append(name)
    return reg_heads, cls_heads


def _split_head_targets(model: "NeuvaModel", y_batch: torch.Tensor) -> dict:
    """Slice a (batch, n_heads) target tensor into per-head targets: regression heads
    consume columns first (as float values), classification heads consume the remaining
    columns (as class indices) — e.g. for 2 heads, y[:, 0] -> regression, y[:, 1:] -> classification."""
    if y_batch.dim() == 1:
        y_batch = y_batch.unsqueeze(1)
    reg_heads, cls_heads = _head_kinds(model)
    targets = {}
    col = 0
    for name in reg_heads:
        targets[name] = y_batch[:, col:col + 1].float()
        col += 1
    for name in cls_heads:
        targets[name] = y_batch[:, col].long()
        col += 1
    return targets


def _multi_output_loss(model: "NeuvaModel", outputs: dict, y_batch: torch.Tensor) -> torch.Tensor:
    """Sum equally-weighted per-head losses: MSE for regression heads, cross-entropy
    for classification heads."""
    targets = _split_head_targets(model, y_batch)
    reg_heads, cls_heads = _head_kinds(model)
    loss = None
    for name in reg_heads:
        head_loss = F.mse_loss(outputs[name], targets[name])
        loss = head_loss if loss is None else loss + head_loss
    for name in cls_heads:
        head_loss = F.cross_entropy(outputs[name], targets[name])
        loss = head_loss if loss is None else loss + head_loss
    return loss


class NeuvaTrainer:
    def train(
        self,
        model: NeuvaModel,
        data: NeuvaDataset,
        epochs: int,
        lr: float = 0.001,
        loss_fn="mse",
        batch_size: int = 16,
        lr_schedule: str = "none",
        early_stop: int = None,
    ) -> None:
        print(f"Training on: {DEVICE.type}")
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # loss_fn is either a builtin loss name (str) or a callable(pred, target) -> tensor
        # (a user-defined Neuva loss function, wired up by the interpreter).
        custom_loss = callable(loss_fn) and not isinstance(loss_fn, str)
        criterion = loss_fn if custom_loss else _LOSSES.get(loss_fn, nn.MSELoss)()

        scheduler = None
        if lr_schedule == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        elif lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        last_layer = model.linears[-1] if model.linears else None
        out_features = getattr(last_layer, "out_features", 1)
        is_multiclass = (not custom_loss) and isinstance(criterion, nn.CrossEntropyLoss)
        is_binary = (not custom_loss) and isinstance(criterion, nn.BCELoss)
        is_multi_output = bool(model.output_names)
        n = len(data.X)
        effective_batch = min(batch_size, n)

        best_loss = float("inf")
        patience = 0

        for epoch in range(1, epochs + 1):
            indices = torch.randperm(n, device=DEVICE)
            total_loss = 0.0
            num_batches = 0

            for start in range(0, n, effective_batch):
                batch_idx = indices[start: start + effective_batch]
                X_batch = data.X[batch_idx]
                y_batch = data.y[batch_idx] if data.y is not None else None

                optimizer.zero_grad()
                outputs = model(X_batch)

                if is_multi_output:
                    if y_batch is None:
                        raise ValueError("multi-output training requires target data (y)")
                    loss = _multi_output_loss(model, outputs, y_batch)
                elif custom_loss:
                    targets = y_batch if y_batch is not None else torch.zeros_like(outputs)
                    loss = criterion(outputs, targets)
                elif is_multiclass:
                    if y_batch is not None and y_batch.numel() > 0:
                        targets = y_batch.squeeze().long()
                    else:
                        targets = torch.randint(0, out_features, (len(X_batch),), device=DEVICE)
                    loss = criterion(outputs, targets)
                elif is_binary:
                    if y_batch is None or y_batch.numel() == 0 or y_batch.min() < 0 or y_batch.max() > 1:
                        y_batch = torch.randint(0, 2, outputs.shape, device=DEVICE).float()
                    else:
                        y_batch = y_batch.float().view_as(outputs)
                    loss = criterion(outputs, y_batch)
                else:
                    targets = y_batch.view_as(outputs) if y_batch is not None else torch.zeros_like(outputs)
                    loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}")

            if scheduler is not None:
                scheduler.step()

            if early_stop is not None:
                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stop:
                        print(f"Early stopping at epoch {epoch} (no improvement for {early_stop} epochs)")
                        break


def _evaluate_multi_output_accuracy(model: NeuvaModel, outputs: dict, y: torch.Tensor) -> dict:
    """Per-head accuracy: R² for regression heads, classification accuracy for the rest."""
    targets = _split_head_targets(model, y)
    reg_heads, cls_heads = _head_kinds(model)
    result = {}
    for name in reg_heads:
        target = targets[name]
        pred = outputs[name]
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        result[name] = 1.0 if ss_tot.item() == 0.0 else max(0.0, (1.0 - ss_res / ss_tot).item())
    for name in cls_heads:
        target = targets[name]
        pred = outputs[name]
        preds = pred.argmax(dim=1) if pred.shape[-1] > 1 else (pred.squeeze(-1) > 0.5).long()
        result[name] = (preds == target).float().mean().item()
    return result


def evaluate_accuracy(model: NeuvaModel, dataset):
    """Return accuracy as a float, or — for multi-output models — a dict of per-head accuracy."""
    X = getattr(dataset, "X", None)
    y = getattr(dataset, "y", None)

    if model.output_names:
        if X is None or y is None or len(X) == 0:
            return {name: 0.0 for name in model.output_names}
        model.eval()
        with torch.no_grad():
            outputs = model(X)
        return _evaluate_multi_output_accuracy(model, outputs, y)

    if X is None or y is None or len(X) == 0:
        in_size = model.linears[0].in_features if model.linears else 1
        out_size = model.linears[-1].out_features if model.linears else 1
        X = torch.randn(64, in_size, device=DEVICE)
        y = torch.randint(0, max(2, out_size), (64,), device=DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        if outputs.shape[-1] == 1:
            predictions = (outputs.squeeze() > 0.5).long()
        else:
            predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == y.squeeze().long()).sum().item()
    return correct / len(X)


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


def _predict_labels(model: NeuvaModel, dataset):
    """Return (predictions, labels) as plain Python int lists for classification metrics."""
    X = getattr(dataset, "X", None)
    y = getattr(dataset, "y", None)
    if X is None or y is None or len(X) == 0:
        return [], []

    model.eval()
    with torch.no_grad():
        outputs = model(X)
    model.train()

    if isinstance(outputs, dict):
        raise ValueError("Metrics are not supported on multi-output models yet")

    if outputs.dim() > 1 and outputs.shape[-1] > 1:
        preds = outputs.argmax(dim=1)
    else:
        preds = (outputs.squeeze(-1) if outputs.dim() > 1 else outputs)
        preds = (preds > 0.5).long()
    labels = y.squeeze().long()
    return preds.cpu().tolist(), labels.cpu().tolist()


def precision(model: NeuvaModel, dataset) -> list:
    """Per-class precision."""
    preds, labels = _predict_labels(model, dataset)
    if not preds:
        return []
    return _skmetrics.precision_score(labels, preds, average=None, zero_division=0).tolist()


def recall(model: NeuvaModel, dataset) -> list:
    """Per-class recall."""
    preds, labels = _predict_labels(model, dataset)
    if not preds:
        return []
    return _skmetrics.recall_score(labels, preds, average=None, zero_division=0).tolist()


def f1_score(model: NeuvaModel, dataset) -> list:
    """Per-class F1 score."""
    preds, labels = _predict_labels(model, dataset)
    if not preds:
        return []
    return _skmetrics.f1_score(labels, preds, average=None, zero_division=0).tolist()


def confusion_matrix(model: NeuvaModel, dataset) -> list:
    """Print a formatted confusion matrix and return it as a list of lists."""
    preds, labels = _predict_labels(model, dataset)
    if not preds:
        print("(empty dataset)")
        return []
    cm = _skmetrics.confusion_matrix(labels, preds)
    n = len(cm)
    col_w = max(4, len(str(cm.max())) + 1)
    header = " " * 6 + "".join(f"P{c}".rjust(col_w) for c in range(n))
    lines = ["Confusion Matrix (rows=actual, cols=predicted)", header, " " * 6 + "-" * (col_w * n)]
    for i, row in enumerate(cm):
        lines.append(f"A{i}".ljust(5) + "|" + "".join(str(v).rjust(col_w) for v in row))
    print("\n".join(lines))
    return cm.tolist()


MODEL_FORMAT_VERSION = "1.1.0"


def save_model(model: NeuvaModel, path: str) -> None:
    torch.save({
        "architecture": model.layer_specs,
        "outputs": model.output_specs,
        "state_dict": model.state_dict(),
        "version": MODEL_FORMAT_VERSION,
    }, path)
    print(f"Model saved to '{path}'")


def load_model(path: str) -> NeuvaModel:
    checkpoint = torch.load(path, weights_only=True)

    if isinstance(checkpoint, dict) and "architecture" in checkpoint:
        layers = [LayerStatement(name=name, args=list(args)) for name, args in checkpoint["architecture"]]
        outputs = [
            OutputLayerStatement(output_name=oname, layer=LayerStatement(name=lname, args=list(args)))
            for oname, lname, args in checkpoint.get("outputs", [])
        ]
        model = NeuvaModel(layers, outputs=outputs)
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Legacy format: dense-only architecture inferred from weight tensor shapes.
        state_dict = checkpoint["state_dict"]
        activation_names = checkpoint.get("activations", [])
        weight_keys = sorted(k for k in state_dict if k.endswith(".weight"))
        layers = [
            LayerStatement(
                name="dense",
                args=[state_dict[k].shape[1], state_dict[k].shape[0], activation_names[i] if i < len(activation_names) else "linear"],
            )
            for i, k in enumerate(weight_keys)
        ]
        model = NeuvaModel(layers)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"'{path}' is not a recognized Neuva model file")

    model.eval()
    return model
