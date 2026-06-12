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

## Development Install (from source)

```bash
git clone https://github.com/Ankit-Mahadani/neuva.git
cd neuva
pip install -e ".[dev]"
pytest tests/
```

This installs Neuva in editable mode so changes to the source take effect immediately.
