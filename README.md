<div align="center">

<br/>

```
 ███╗   ██╗███████╗██╗   ██╗██╗   ██╗ █████╗
 ████╗  ██║██╔════╝██║   ██║██║   ██║██╔══██╗
 ██╔██╗ ██║█████╗  ██║   ██║██║   ██║███████║
 ██║╚██╗██║██╔══╝  ██║   ██║╚██╗ ██╔╝██╔══██║
 ██║ ╚████║███████╗╚██████╔╝ ╚████╔╝ ██║  ██║
 ╚═╝  ╚═══╝╚══════╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
```

**A programming language built purely for Machine Learning.**

*Simple like Python. Clear like Rust. Made for ML.*

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-7C3AED.svg?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3B82F6.svg?style=flat-square)](https://python.org)
[![PyTorch Backend](https://img.shields.io/badge/Backend-PyTorch-EF4444.svg?style=flat-square)](https://pytorch.org)
[![Status: Building](https://img.shields.io/badge/Status-Building%20in%20Public-10B981.svg?style=flat-square)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-F59E0B.svg?style=flat-square)](CONTRIBUTING.md)

<br/>

</div>

---

## What is Neuva?

Neuva (NOO-vah) is an open source programming language designed from the ground up for Machine Learning. No boilerplate. No imports. No configuration. Just describe your model and train it.

```neuva
# Train a digit classifier — the entire program
let data = load("mnist.csv")
let train_data, test_data = data.split(0.8)

model DigitNet {
    layer dense(784 -> 128, relu)
    layer dense(128 -> 10,  softmax)
}

train DigitNet on train_data for 20 epochs, lr=0.001
print accuracy(DigitNet, test_data)
```

That is it. No `import torch`. No `nn.Module`. No training loop. Neuva handles it all.

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

train DigitNet on train_data for 20 epochs, lr=0.001
```

Same result. A fraction of the code.

---

## Features

- **ML-first keywords** — `model`, `layer`, `train`, `predict`, `save` are built into the language
- **No boilerplate** — no imports, no class definitions, no training loops
- **PyTorch backend** — compiles to PyTorch under the hood, so it is fast and compatible
- **Rust-style errors** — clear error messages with line numbers and helpful hints
- **Simple types** — `tensor`, `matrix`, `int`, `float`, `bool`, `string`
- **Beginner friendly** — if you know any programming language, you can write Neuva
- **Open source** — MIT licensed, built in public, contributions welcome

---

## Language Overview

### Variables

```neuva
let name   = "Neuva"        # string  (inferred)
let layers = 3              # int     (inferred)
let rate: float = 0.001     # float   (explicit)
let ready: bool = true      # bool    (explicit)
```

### Models

```neuva
model MyNet {
    layer dense(784 -> 128, relu)      # input 784, output 128, relu activation
    layer dense(128 -> 64,  relu)
    layer dense(64  -> 10,  softmax)   # output layer
}
```

Supported layer types: `dense`, `conv`, `pool`, `dropout`, `flatten`, `norm`

Supported activations: `relu`, `sigmoid`, `softmax`, `tanh`, `linear`, `gelu`

### Loading Data

```neuva
let data = load("dataset.csv")

let train_data, test_data = data.split(0.8)   # 80/20 split
data = data.normalize()                        # normalize values
data = data.shuffle()                          # shuffle rows
```

### Training

```neuva
# one-liner
train MyNet on train_data for 10 epochs, lr=0.001

# multi-line (more readable)
train MyNet
    on    train_data
    for   50 epochs
    lr    = 0.0005
    loss  = crossentropy
```

Supported loss functions: `crossentropy`, `mse`, `mae`, `huber`, `binary_crossentropy`

### Predict and Evaluate

```neuva
let result = predict MyNet on test_data
let acc    = accuracy(MyNet, test_data)
print "Accuracy:", acc
```

### Save and Load

```neuva
save MyNet to "my_model.nva"
let loaded = load("my_model.nva")
```

### Functions

```neuva
fn welcome(name: string) {
    print "Hello from Neuva,", name
}

fn square(x: float) -> float {
    return x * x
}
```

### Control Flow

```neuva
if acc > 0.95 {
    print "Excellent model!"
} else {
    print "Keep training."
}

for i in 0..10 {
    print i
}
```

---

## Examples

| File | Description |
|---|---|
| [`examples/hello.nva`](examples/hello.nva) | Hello world |
| [`examples/iris_classifier.nva`](examples/iris_classifier.nva) | Iris flower classification |
| [`examples/digit_classifier.nva`](examples/digit_classifier.nva) | MNIST digit recognition |
| [`examples/regression.nva`](examples/regression.nva) | House price regression |
| [`examples/full_pipeline.nva`](examples/full_pipeline.nva) | End-to-end ML pipeline |

---

## Install

> Neuva is currently in early development. The interpreter is being built in public.

Once released:

```bash
pip install neuva-lang
```

Run a `.nva` file:

```bash
neuva run my_model.nva
```

---

## Project Status

Neuva is being built live, day by day, in public.

| Phase | Status | Description |
|---|---|---|
| 1 — Design | ✅ Done | Language syntax and keywords |
| 2 — GitHub Setup | ✅ Done | Repo, license, structure |
| 3 — Lexer | 🔨 In progress | Tokenizer |
| 4 — Parser | ⏳ Upcoming | AST builder |
| 5 — Interpreter | ⏳ Upcoming | Tree walker |
| 6 — PyTorch backend | ⏳ Upcoming | ML execution |
| 7 — PyPI release | ⏳ Upcoming | `pip install neuva-lang` |

Follow the journey on [LinkedIn](https://www.linkedin.com/in/ankitmahadani/) — posting every milestone.

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

Look for issues labelled `good first issue` to get started quickly.

---

## Built With

- **Python 3.10+** — the interpreter is written in Python
- **PyTorch** — ML execution backend
- **NumPy + Pandas** — data handling

---

## Roadmap

- [ ] Lexer + Parser (current focus)
- [ ] Working interpreter for basic programs
- [ ] PyTorch backend for model training
- [ ] `pip install neuva-lang` on PyPI
- [ ] VS Code syntax highlighting extension
- [ ] GPU support
- [ ] CNN and RNN layer types
- [ ] Interactive REPL (`neuva shell`)
- [ ] Web playground

---

## License

MIT © 2026 Neuva Contributors

---

<div align="center">

**Built in public. Day by day.**

*Star the repo to follow the journey.*

</div>
