# Getting Started with Neuva

Neuva is a programming language built for Machine Learning. No imports, no boilerplate — just describe your model and train it.

## Install

```bash
pip install neuva-lang
```

Requires Python 3.10+ and PyTorch. Both are pulled in automatically.

To verify the install:

```bash
neuva --version
```

## Hello World

Create a file called `hello.nva`:

```neuva
let name = "Neuva"
print "Hello from", name
```

Run it:

```bash
neuva hello.nva
```

Output:

```
Hello from Neuva
```

## Your First ML Program

Create `classify.nva`:

```neuva
let data = load("examples/data/iris.csv")
let train_data, test_data = data.split(0.8)

model IrisNet {
    layer dense(4 -> 16, relu)
    layer dense(16 -> 3, softmax)
}

train IrisNet
    on train_data
    for 50 epochs
    lr   = 0.01
    loss = crossentropy

let acc = accuracy(IrisNet, test_data)
print "Test accuracy:", acc
save IrisNet to "iris_model.nva"
```

Run it:

```bash
neuva classify.nva
```

You will see per-epoch loss printed, then the final test accuracy, and the model saved to `iris_model.nva`.

## Running Any `.nva` File

```bash
neuva path/to/program.nva
```

Neuva looks for the file relative to the current working directory. Data paths inside the `.nva` file are also resolved from the directory where you run the command.

## Interactive Shell (REPL)

```bash
neuva shell
```

The shell keeps a persistent interpreter — variables, models, and functions defined in one line are available to every line after it.

```
$ neuva shell
Neuva 1.1.0 — interactive shell. Type :help for commands, :quit to exit.
>>> let x = 42
>>> print "The answer is {x}"
The answer is 42
>>> exit
```

**Multi-line input.** If a line ends with an unmatched `{`, the shell switches to a `... ` continuation prompt and keeps reading lines until every `{` has a matching `}`:

```
>>> fn square(x: float) -> float {
...     return x * x
... }
>>> print square(4.0)
16.0
```

**Command history.** Up/down arrows cycle through previous input, backed by `readline` on Linux/macOS and `pyreadline3` on Windows (both installed automatically with Neuva).

**Special commands** (all start with `:`):

| Command | Effect |
|---|---|
| `:help` | List every keyword and built-in function |
| `:clear` | Clear the screen |
| `:examples` | Print 3 short example programs |
| `:reset` | Wipe the interpreter environment (clear all variables/models/functions) |
| `:quit` | Exit the shell (`exit`, `quit`, and Ctrl-D also work) |

**Colorized output.** Model summaries (`print MyModel`) are shown in cyan and any output line containing the word "accuracy" is shown in green; errors are shown in red. Colors are automatically disabled when stdout isn't a real terminal (e.g. `neuva shell < script.txt`).

## Web Playground

Neuva also ships a browser-based playground at `playground/index.html`, powered by [Pyodide](https://pyodide.org) running `playground/neuva_web.py` — a small, self-contained, pure-Python reimplementation of the Neuva lexer/parser/interpreter with a simulated ML backend (no `lark`, `torch`, or `pandas` dependency, so it can run entirely client-side in WebAssembly). It supports the same language syntax as the CLI for programs that don't need real numeric training results.

To run it locally:

```bash
cd playground
python -m http.server 8000
```

Then open `http://localhost:8000/` in a browser. (Opening `index.html` directly via `file://` will not work — the page fetches `neuva_web.py` over HTTP, which browsers block for local files.)

The repo also has a GitHub Actions workflow (`.github/workflows/deploy_playground.yml`) that publishes `playground/` to GitHub Pages on every push to `main`; if Pages is enabled for this repository, the playground is also reachable online without any local setup.

## Standard Library

Neuva ships six stdlib modules as plain `.nva` source files under `neuva/stdlib/`. Bring one into scope with `import <name>`:

```neuva
import metrics

let f1 = f1_from_pr(0.8, 0.6)
print "F1:", f1
```

| Module | What it provides |
|---|---|
| `metrics` | `f1_from_pr(p, r)`, `average(values)`, `report(model, data)` (prints accuracy/precision/recall/F1 and returns accuracy) |
| `preprocessing` | `normalize_data(data)`, `shuffle_data(data)`, `one_hot(index, num_classes)`, `one_hot_batch(indices, num_classes)` |
| `visualization` | `print_loss_curve(losses)` — ASCII bar chart of a loss list |
| `datasets` | `load_iris()`, `load_housing()`, `load_mnist_sample()` — ready-to-use toy/benchmark datasets with no CSV file needed (see the Language Reference for details on each) |
| `optimizers` | `use_sgd(lr, momentum)`, `use_adam(lr, beta1, beta2)`, `use_rmsprop(lr)` — configure the optimizer for the *next* `train` statement only |
| `callbacks` | `on_epoch_end(fn)`, `on_improvement(fn)` — register a callback for the *next* `train` statement only |

Each `import` re-parses and runs the module's `.nva` source into your program's environment, so its functions become directly callable — no module-qualified names (`metrics.report(...)` doesn't exist; it's just `report(...)`).

## Development Install (from source)

```bash
git clone https://github.com/Ankit-Mahadani/neuva.git
cd neuva
pip install -e ".[dev]"
pytest tests/
```

This installs Neuva in editable mode so changes to the source take effect immediately.
