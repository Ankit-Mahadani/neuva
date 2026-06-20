<div align="center">

<br/>

<pre align="center">
 ███╗   ██╗███████╗██╗   ██╗██╗   ██╗ █████╗
 ████╗  ██║██╔════╝██║   ██║██║   ██║██╔══██╗
 ██╔██╗ ██║█████╗  ██║   ██║██║   ██║███████║
 ██║╚██╗██║██╔══╝  ██║   ██║╚██╗ ██╔╝██╔══██║
 ██║ ╚████║███████╗╚██████╔╝ ╚████╔╝ ██║  ██║
 ╚═╝  ╚═══╝╚══════╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
</pre>

**A programming language built purely for Machine Learning.**

*Simple like Python. Clear like Rust. Made for ML.*

<br/>

[![Version: 1.0.0](https://img.shields.io/badge/version-1.0.0-7C3AED.svg?style=flat-square)]()
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-7C3AED.svg?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3B82F6.svg?style=flat-square)](https://python.org)
[![PyTorch Backend](https://img.shields.io/badge/Backend-PyTorch-EF4444.svg?style=flat-square)](https://pytorch.org)
[![Tests: 54 passing](https://img.shields.io/badge/Tests-54%20passing-10B981.svg?style=flat-square)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-F59E0B.svg?style=flat-square)](CONTRIBUTING.md)

<br/>

</div>

---

## What is Neuva?

Neuva (NOO-vah) is an open source programming language designed from the ground up for Machine Learning. No boilerplate. No imports. No configuration. Just describe your model and train it.

```neuva
# Train a digit classifier — the entire program
let data = load("examples/data/iris.csv")
let train_data, test_data = data.split(0.8)

model DigitNet {
    layer dense(4 -> 16, relu)
    layer dense(16 -> 3,  softmax)
}

train DigitNet on train_data for 20 epochs, lr = 0.001, loss = crossentropy
print "Accuracy:", accuracy(DigitNet, test_data)
```

That is it. No `import torch`. No `nn.Module`. No training loop. Neuva handles it all.

---

## What's New in v1.0.0

Neuva started as a single `.lark` grammar file and a stub interpreter. Over 34 days of public development — one commit per day — it grew into a complete ML language with a real PyTorch backend.

**The journey in short:**
- Days 5–9: grammar, parser, and first AST tests
- Days 10–11: working interpreter; Neuva ran its first program
- Days 12–17: real PyTorch backend; neural networks actually trained
- Days 20–21: real CSV data loading, train/test splits, 93% accuracy on Iris
- Days 22–25: VS Code extension, documentation, Rust-style error messages
- Days 27–30: `else if` chains, string concatenation, `--version` flag
- Days 31–33: interactive REPL, GPU detection, mini-batch training
- Day 34: f-strings, lists, `and`/`or`, dropout layers, model summary printing
- v1.0.0: 54 tests, 13 runnable examples, stable release

---

## Why Neuva?

Most ML code looks like this:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
```

Neuva code looks like this:

```neuva
model DigitNet {
    layer dense(784 -> 128, relu)
    layer dense(128 -> 10,  softmax)
}

train DigitNet on train_data for 20 epochs, lr = 0.001
```

Same result. A fraction of the code.

---

## Features

- **ML-first keywords** — `model`, `layer`, `train`, `predict`, `save` are built into the language
- **No boilerplate** — no imports, no class definitions, no training loops
- **PyTorch backend** — compiles to PyTorch under the hood, fast and compatible
- **GPU support** — automatically detects and uses CUDA if available
- **Mini-batch training** — shuffled batches each epoch, averaged loss reporting
- **F-string interpolation** — `print "Accuracy: {acc}%"` substitutes variables inline
- **List support** — `let scores = [85, 90, 78]`, `scores[0]`, `len(scores)`
- **Boolean logic** — `and` / `or` with short-circuit evaluation
- **Dropout layers** — `layer dropout(0.3)` maps directly to `nn.Dropout`
- **Model summary** — `print MyModel` shows layers and total parameter count
- **Interactive REPL** — `neuva shell` for live exploration with persistent state
- **Rust-style errors** — line numbers, column pointers, did-you-mean suggestions
- **Simple types** — `tensor`, `matrix`, `int`, `float`, `bool`, `string`
- **Open source** — AGPL v3 licensed, built in public, contributions welcome

---

## Install

```bash
pip install neuva-lang
```

Run a `.nva` file:

```bash
neuva my_model.nva
```

Start the interactive shell:

```bash
neuva shell
```

Check the version:

```bash
neuva --version
```

---

## Language Overview

### Variables

```neuva
let name   = "Neuva"        # string  (inferred)
let layers = 3              # int     (inferred)
let rate: float = 0.001     # float   (explicit)
let ready: bool = true      # bool    (explicit)
```

### F-String Interpolation

```neuva
let acc = 93
print "Accuracy: {acc}%"    # prints: Accuracy: 93%

let model_name = "DigitNet"
print "Training {model_name}..."
```

### Lists

```neuva
let scores = [85, 90, 78, 92]
print scores[0]             # 85
let n = len(scores)         # 4
```

### Boolean Logic

```neuva
if score > 50 and score < 100 {
    print "valid score"
}

if passed or bonus {
    print "you qualify"
}
```

### Models

```neuva
model MyNet {
    layer dense(784 -> 128, relu)
    layer dropout(0.3)
    layer dense(128 -> 10,  softmax)
}

print MyNet   # prints layer summary and parameter count
```

Supported layer types: `dense`, `conv`, `pool`, `dropout`, `flatten`

Supported activations: `relu`, `sigmoid`, `softmax`, `tanh`, `linear`

### Loading Data

```neuva
let data = load("examples/data/iris.csv")

let train_data, test_data = data.split(0.8)   # 80/20 split
data = data.normalize()                        # normalize values
data = data.shuffle()                          # shuffle rows
```

### Training

```neuva
# one-liner
train MyNet on train_data for 10 epochs, lr = 0.001

# multi-line (more readable)
train MyNet
    on    train_data
    for   50 epochs
    lr    = 0.0005
    loss  = crossentropy
```

Supported loss functions: `crossentropy`, `mse`, `mae`, `binary_crossentropy`

### Predict and Evaluate

```neuva
let acc = accuracy(MyNet, test_data)
print "Accuracy: {acc}"
```

### Save and Load

```neuva
save MyNet to "my_model.nva"
let loaded = load("my_model.nva")
```

### Functions

```neuva
fn welcome(name: string) {
    print "Hello from Neuva, {name}"
}

fn square(x: float) -> float {
    return x * x
}
```

### Control Flow

```neuva
if acc > 0.95 {
    print "Excellent model!"
} else if acc > 0.80 {
    print "Good model."
} else {
    print "Keep training."
}

for i in range(10) {
    print "Epoch {i}"
}

let count = 0
while count < 5 {
    let count = count + 1
}
```

### Interactive REPL

```
$ neuva shell
Neuva 1.0.0 — interactive shell. Type 'exit' to quit.
>>> let x = 42
>>> print "The answer is {x}"
The answer is 42
>>> exit
```

---

## Examples

| File | Description |
|---|---|
| [`examples/hello.nva`](examples/hello.nva) | Hello world |
| [`examples/digit_classifier.nva`](examples/digit_classifier.nva) | Iris classification, 93% accuracy |
| [`examples/linear_regression.nva`](examples/linear_regression.nva) | House price regression |
| [`examples/spam_classifier.nva`](examples/spam_classifier.nva) | Binary spam detection |
| [`examples/house_price.nva`](examples/house_price.nva) | Multi-layer regression |
| [`examples/cnn_classifier.nva`](examples/cnn_classifier.nva) | CNN layer definitions |
| [`examples/control_flow_demo.nva`](examples/control_flow_demo.nva) | `if`, `for`, `while`, functions |
| [`examples/elif_demo.nva`](examples/elif_demo.nva) | Chained `else if` |
| [`examples/fstring_demo.nva`](examples/fstring_demo.nva) | F-string interpolation |
| [`examples/list_demo.nva`](examples/list_demo.nva) | Lists and indexing |
| [`examples/logical_demo.nva`](examples/logical_demo.nva) | `and` / `or` operators |
| [`examples/dropout_demo.nva`](examples/dropout_demo.nva) | Dropout layer and model summary |
| [`examples/string_concat_demo.nva`](examples/string_concat_demo.nva) | String concatenation |

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| 1 — Design | ✅ Done | Language syntax and keywords |
| 2 — GitHub Setup | ✅ Done | Repo, license, structure |
| 3 — Lexer | ✅ Done | Tokenizer with full terminal set |
| 4 — Parser | ✅ Done | Lark Earley parser, full AST |
| 5 — Interpreter | ✅ Done | Tree-walking interpreter, 54 tests |
| 6 — PyTorch backend | ✅ Done | Mini-batch training, GPU support |
| 7 — PyPI release | ✅ Done | `pip install neuva-lang` |

---

## Documentation

| Document | Description |
|---|---|
| [Getting Started](docs/GETTING_STARTED.md) | Install, hello world, running `.nva` files |
| [Language Reference](docs/LANGUAGE_REFERENCE.md) | Every keyword with syntax and examples |
| [Examples](docs/EXAMPLES.md) | Line-by-line walkthrough of the iris classifier |
| [Changelog](CHANGELOG.md) | Full version history |
| [Roadmap](ROADMAP.md) | What's planned post-1.0 |

---

## Contributing

Neuva is open source and we welcome all contributors — beginners and experts alike.

```bash
git clone https://github.com/Ankit-Mahadani/neuva.git
cd neuva
pip install -e ".[dev]"
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

Look for issues labelled `good first issue` to get started: [github.com/Ankit-Mahadani/neuva/issues](https://github.com/Ankit-Mahadani/neuva/issues)

---

## Built With

- **Python 3.10+** — the interpreter is written in Python
- **PyTorch ≥ 2.0** — ML execution backend
- **Lark ≥ 1.1** — grammar and parser
- **NumPy + Pandas** — data loading and preprocessing

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This means:
- You can use, study, modify, and distribute Neuva freely
- If you use Neuva in a product or service (including over a network), you **must** release your source code under the same license
- Any modifications must also be open source

See the full [LICENSE](LICENSE) file for details.

Copyright © 2026 Ankit Mahadani and Neuva Contributors

---

<div align="center">

**Built in public. Day by day. v1.0.0.**

*Star the repo to follow the journey.*

</div>
