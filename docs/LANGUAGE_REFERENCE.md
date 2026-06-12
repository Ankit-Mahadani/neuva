# Neuva Language Reference

Complete reference for every keyword and construct in Neuva.

---

## `let` — Variable Declaration

Declares a variable and binds it to a value. Type annotation is optional; Neuva infers the type.

**Syntax**

```neuva
let name = value
let name: type = value
let a, b = expr          # destructure a tuple (e.g. from split)
```

**Examples**

```neuva
let x = 42
let rate: float = 0.001
let ready: bool = true
let greeting = "hello"
let train_data, test_data = data.split(0.8)
```

**Types:** `int`, `float`, `bool`, `string`, `tensor`, `matrix`

---

## `print` — Output

Prints one or more values separated by spaces.

**Syntax**

```neuva
print expr
print expr, expr, ...
```

**Examples**

```neuva
print "Hello, world"
print "Accuracy:", acc
print x, y, z
```

---

## `model` — Neural Network Definition

Defines a named neural network as a stack of layers.

**Syntax**

```neuva
model Name {
    layer ...
    layer ...
}
```

**Example**

```neuva
model MyNet {
    layer dense(128 -> 64, relu)
    layer dense(64 -> 10, softmax)
}
```

---

## `layer` — Layer Types

Used inside a `model` block. Four types are supported.

### `dense` — Fully Connected Layer

Maps `in_features` inputs to `out_features` outputs, then applies an activation function.

```neuva
layer dense(in -> out, activation)
```

```neuva
layer dense(784 -> 128, relu)
layer dense(128 -> 10,  softmax)
```

**Activations:** `relu`, `sigmoid`, `tanh`, `softmax`, `linear`

### `conv` — 2D Convolution

Applies a 2D convolution with `kernel_size × kernel_size` filters.

```neuva
layer conv(in_channels -> out_channels, kernel_size)
```

```neuva
layer conv(1 -> 32, 3)    # 1 input channel, 32 output channels, 3×3 kernel
layer conv(32 -> 64, 3)
```

Maps to `nn.Conv2d(in_channels, out_channels, kernel_size)`.

### `pool` — Max Pooling

Reduces spatial dimensions using max pooling.

```neuva
layer pool(kernel_size)
```

```neuva
layer pool(2)    # 2×2 max pool, halves height and width
```

Maps to `nn.MaxPool2d(kernel_size)`.

### `flatten` — Flatten

Collapses all dimensions except the batch dimension into one. Required before a `dense` layer after `conv`/`pool`.

```neuva
layer flatten
```

Maps to `nn.Flatten()`.

---

## `train` — Train a Model

Runs the training loop for a given model on a dataset.

**Syntax (single line)**

```neuva
train Name on data for N epochs, lr = 0.001, loss = crossentropy
```

**Syntax (multi-line)**

```neuva
train Name
    on data
    for N epochs
    lr   = 0.001
    loss = crossentropy
```

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `lr` | `0.001` | Learning rate for Adam optimiser |
| `loss` | `mse` | Loss function |

**Loss functions:** `crossentropy`, `mse`, `mae`

**Example**

```neuva
train IrisNet
    on train_data
    for 50 epochs
    lr   = 0.01
    loss = crossentropy
```

---

## `save` — Save Model Weights

Saves a trained model's weights to a `.nva` file.

**Syntax**

```neuva
save Name to "path.nva"
```

**Example**

```neuva
save IrisNet to "models/iris.nva"
```

Internally calls `torch.save` with the model's state dict and activation names, enabling a full round-trip load.

---

## `load` — Load Data or Model

Loads a CSV dataset or a previously saved model, detected by file extension.

**Syntax**

```neuva
let x = load("file.csv")    # returns a DataSet
let x = load("file.nva")    # returns a trained NeuvaModel
```

**Examples**

```neuva
let data  = load("examples/data/iris.csv")
let model = load("models/iris.nva")
```

For CSV files:
- All numeric columns except the last become features (`X`).
- The last numeric column becomes the label (`y`).
- Integer labels produce a `long` tensor (classification); float labels produce `float32` (regression).

---

## `accuracy` — Evaluate Classification Accuracy

Runs the model on a dataset and returns the fraction of correctly predicted labels.

**Syntax**

```neuva
let acc = accuracy(model, dataset)
```

**Example**

```neuva
let acc = accuracy(IrisNet, test_data)
print "Accuracy:", acc
```

Returns a `float` between 0 and 1. Uses `argmax` over the output logits and compares to ground-truth labels.

---

## Data Methods

DataSet objects returned by `load` expose three chainable methods.

### `.split(ratio)`

Splits rows randomly into a train set and a test set.

```neuva
let train_data, test_data = data.split(0.8)    # 80% train, 20% test
```

### `.normalize()`

Normalises each feature column to zero mean and unit variance.

```neuva
let data = data.normalize()
```

### `.shuffle()`

Randomly permutes the rows.

```neuva
let data = data.shuffle()
```

---

## `fn` — Function Definition

Defines a reusable function. Parameters require a type annotation. Return type is optional.

**Syntax**

```neuva
fn name(param: type, ...) {
    ...
}

fn name(param: type, ...) -> return_type {
    return expr
}
```

**Examples**

```neuva
fn greet(name: string) {
    print "Hello,", name
}

fn square(x: float) -> float {
    return x * x
}

greet("world")
let result = square(3.0)
```

---

## `return` — Return a Value

Returns a value from a function.

**Syntax**

```neuva
return expr
return          # returns nothing (None)
```

---

## `if` / `else` — Conditional

**Syntax**

```neuva
if condition {
    ...
}

if condition {
    ...
} else {
    ...
}
```

**Example**

```neuva
if acc > 0.9 {
    print "Great accuracy!"
} else {
    print "Needs more training."
}
```

---

## `for` — For Loop

Iterates over a range or any iterable.

**Syntax**

```neuva
for var in iterable {
    ...
}
```

**Example**

```neuva
for i in range(5) {
    print i
}
```

---

## `while` — While Loop

Repeats while a condition is true.

**Syntax**

```neuva
while condition {
    ...
}
```

**Example**

```neuva
let x = 0
while x < 3 {
    print x
    let x = x + 1
}
```

---

## Operators

| Operator | Description |
|----------|-------------|
| `+` `-` `*` `/` `%` | Arithmetic |
| `==` `!=` `<` `<=` `>` `>=` | Comparison |
| `!` | Logical not |
| `-` (unary) | Negation |

---

## Comments

```neuva
# This is a comment
let x = 42    # inline comment
```
