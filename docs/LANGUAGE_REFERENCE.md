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

## Reassignment — `=`, `+=`, `-=`, `*=`, `/=`

Mutates a variable that has already been introduced with `let`. `let` is still required for the *first* binding — assigning to a name that was never `let`-bound is a runtime error.

Unlike `let` (which always writes into the *current* frame, creating a new binding there if needed), `=` and the compound operators walk up to whichever enclosing frame the variable was originally declared in and mutate it there. This is what lets a function body update a variable declared *outside* the function — see the Environment/scoping model in [ARCHITECTURE.md](ARCHITECTURE.md).

**Syntax**

```neuva
name = expr
name += expr    # name = name + expr
name -= expr    # name = name - expr
name *= expr    # name = name * expr
name /= expr    # name = name / expr
```

**Examples**

```neuva
let x = 10
x = 20        # plain reassignment
x += 5        # 25
x -= 3        # 22
x *= 2        # 44
x /= 4        # 11.0

let s = "a"
s += "b"      # "ab" — += concatenates strings, matching `+`

let counter = 0
fn bump() {
    counter += 1    # mutates the OUTER `counter`, not a new local one
    return counter
}
bump()    # 1
bump()    # 2
```

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

`print table(data)` renders a `DataSet`, or a list of rows, as a formatted ASCII table. `print plot(values)` renders a list of numbers as an ASCII line chart:

```neuva
print table(test_data)
print plot([0.9, 0.7, 0.5, 0.3, 0.2, 0.1])
```

`table()`/`plot()` are only meaningful as the direct argument to `print` — they wrap their argument in a marker object.

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

Used inside a `model` block.

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

### `dropout` — Dropout Regularization

Randomly zeroes a fraction `p` of activations during training, to reduce overfitting.

```neuva
layer dropout(p)
```

```neuva
layer dropout(0.3)    # drop 30% of activations
```

Maps to `nn.Dropout(p)`.

### `norm` — Batch Normalization

Normalizes activations across the batch dimension.

```neuva
layer norm(num_features)
```

```neuva
layer norm(64)    # num_features must match the previous layer's output size
```

Maps to `nn.BatchNorm1d(num_features)`.

---

**Layer dimension checking.** The type checker validates that consecutive `dense` layers agree on size: if one `dense` layer's output size doesn't match the next `dense` layer's input size, running the program reports a `layer dimension mismatch` error before training starts (skip this with `--no-check`).

**Beyond these layer types.** The backend also supports `rnn`, `lstm`, `embedding`, `attention`, and `transformer` layers for sequence/NLP-style models, plus multi-output models via `output name: dense(in -> out, activation)` heads inside a `model` block. These are real, tested, working features, but are more specialized than the layer types documented above — see `examples/rnn_demo.nva`, `examples/transformer_demo.nva`, and `examples/multitask_demo.nva` for working examples of each.

---

## Model Methods

Methods callable on a trained model — the value a variable holds after a `train` statement has run on it, or after `load("path.nva")`.

### `.freeze()`

Freezes every parameter (`requires_grad = False`) so none of them are updated by further training — used to hold a pretrained backbone fixed. There is no partial or per-layer freeze; `.freeze()` always freezes the whole model.

### `.unfreeze()`

Unfreezes every parameter, undoing `.freeze()`.

**Example — transfer learning**

```neuva
train Backbone on train_data for 20 epochs, lr = 0.01, loss = crossentropy

Backbone.freeze()
train Backbone on train_data for 5 epochs, lr = 0.01, loss = crossentropy
# training is skipped: nothing to update, weights are unchanged

Backbone.unfreeze()
train Backbone on train_data for 15 epochs, lr = 0.001, loss = crossentropy
# fine-tunes the SAME weights, continuing from where pretraining left off
```

Calling `train` on a fully-frozen model prints a message and skips training instead of crashing. Calling `train` again on an already-trained model (frozen or not) continues training its existing weights rather than rebuilding a fresh, randomly-initialized model — see the note under `train` below.

---

## `train` — Train a Model

Runs the training loop for a given model on a dataset. The epoch count (`N`) must be a literal integer; the model and data names must be bare identifiers (not arbitrary expressions).

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
| `lr` | `0.001` | Learning rate. May be a literal or any expression, e.g. a loop variable (see `examples/hyperparameter_search.nva`). |
| `loss` | `mse` | Loss function name — a built-in (`crossentropy`, `mse`, `mae`, `binary_crossentropy`) or the name of a Neuva `fn(pred, target)` you defined. Must be a bare name, not an expression. |
| `lr_schedule` | `none` | `none`, `step` (halves `lr` every 10 epochs), or `cosine` (cosine annealing over all epochs). |
| `early_stop` | off | Stop training if the loss hasn't improved for this many consecutive epochs. May be a literal or expression. |
| `lr_warmup` | off | Linearly ramp `lr` from `0` up to the target value over this many epochs, then hold steady. Pauses `lr_schedule` during the warmup window so the two don't fight over the learning rate in the same epoch. May be a literal or expression. |

**Loss functions:** `crossentropy`, `mse`, `mae`, `binary_crossentropy`, or a custom loss function:

```neuva
fn my_loss(pred, target) {
    return mse(pred, target) + 0.1
}

train Regressor on train_data for 10 epochs, lr = 0.001, loss = my_loss
```

**Example**

```neuva
train IrisNet
    on train_data
    for 50 epochs
    lr   = 0.01
    loss = crossentropy
```

**`lr_warmup` example**

```neuva
train Net on data for 30 epochs, lr = 0.01, lr_warmup = 5
# epochs 1-5:  lr ramps 0.002, 0.004, 0.006, 0.008, 0.01
# epochs 6-30: lr holds at 0.01 (or follows lr_schedule, if also set)
```

**Continuing training.** If `Name` already refers to a trained model, a further `train` statement continues training the *same* weights rather than rebuilding a fresh, randomly-initialized model from the `model` block — this is what makes `.freeze()`/`.unfreeze()` fine-tuning workflows possible. To train a brand-new model from scratch on each iteration of a loop (e.g. a hyperparameter search), re-declare the `model { ... }` block before each `train`.

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

## `predict` — Run Inference

**Statement form** — runs the model on a dataset and prints the predictions:

```neuva
predict Name on data
```

```neuva
predict IrisNet on test_data
# Predictions: [0, 2, 1, 0, ...]
```

**Function form** — the same computation, returning the list instead of printing it:

```neuva
let preds = predict(model, data)
```

Both forms return predicted class labels — argmax over multi-class output, or a 0.5-threshold over a single sigmoid/binary output — as a plain list, or (for a multi-output model) a dict of `{output_name: [predictions...]}`.

### `predict_proba(model, data)` — Raw Probabilities

Like `predict()`, but returns the model's raw forward-pass output (softmax/sigmoid probabilities, or whatever activation the output layer uses) with no argmax/threshold applied.

```neuva
let probs = predict_proba(IrisNet, test_data)
```

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

Returns a `float` between 0 and 1. Uses `argmax` over the output logits and compares to ground-truth labels. For a multi-output model, returns a dict of per-head accuracy instead (R² for regression heads, classification accuracy for the rest).

---

## Classification Metrics — `precision`, `recall`, `f1_score`, `confusion_matrix`

Each takes `(model, dataset)`:

```neuva
let p  = precision(IrisNet, test_data)     # per-class precision, a list
let r  = recall(IrisNet, test_data)        # per-class recall, a list
let f1 = f1_score(IrisNet, test_data)      # per-class F1, a list
confusion_matrix(IrisNet, test_data)       # prints a formatted matrix, returns it as a list of lists
```

Not supported on multi-output models.

---

## `export_onnx` — Export to ONNX

```neuva
export_onnx(model, "model.onnx")
```

Exports a trained model via `torch.onnx.export`. Only works for models whose first layer has a fixed input width (`dense`, `conv`, ...) — models starting with an `rnn`/`lstm`/`embedding` layer, and multi-output models, raise a clear error instead of exporting something broken.

---

## Data Methods

DataSet objects returned by `load` expose the following chainable methods.

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

### `.augment()`

Simple data augmentation: random horizontal flip + small rotation for image-shaped data (3+ dimensional `X`); falls back to adding a small amount of Gaussian noise for anything else (or if `torchvision` isn't installed).

```neuva
let data = data.augment()
```

### `.oversample()`

Duplicates minority-class rows (with replacement) so every class has as many rows as the majority class. No-op on regression targets (non-integer labels).

```neuva
let data = data.oversample()
```

### `.undersample()`

Randomly drops majority-class rows so every class has as few rows as the minority class. No-op on regression targets.

```neuva
let data = data.undersample()
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

**`range`** takes 1 to 3 arguments, matching Python's `range`:

```neuva
range(stop)                 # 0, 1, ..., stop-1
range(start, stop)          # start, start+1, ..., stop-1
range(start, stop, step)    # start, start+step, ..., while < stop
```

```neuva
for i in range(0, 10, 2) {
    print i    # 0, 2, 4, 6, 8
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
    x += 1
}
```

Since `x` is already `let`-bound, `x += 1` (or `let x = x + 1`, redeclaring it) both work here — see the Reassignment section above.

---

## `match` / `case` — Pattern-Style Branching

Compares a subject expression against a sequence of literal values and runs the statement for the first match; falls through to `default` if nothing matches (or does nothing if there's no `default`).

**Syntax**

```neuva
match subject {
    case literal1: statement
    case literal2: statement
    default: statement
}
```

**Example**

```neuva
match result {
    case 0: print "negative"
    case 1: print "positive"
    default: print "unknown"
}
```

**Limitations**

- Exactly **one statement** per `case`/`default` — there is no `{ }` block form, so a case can't hold multiple statements directly. Call a function if you need more than one:
  ```neuva
  match result {
      case 1: handle_positive()
      default: handle_unknown()
  }
  ```
- `case` values must be literal constants — `int`, `float`, `string`, `bool`, or a triple-quoted string — not variables, expressions, or negative-number literals (`case -1:` is a parse error).
- `default` is optional; a `match` with no matching case and no `default` is a no-op.

---

## List

**Syntax**

```neuva
let items = [expr, expr, ...]
let empty = []
```

**Examples**

```neuva
let scores = [85, 90, 78, 92]
print scores[0]        # 85
print len(scores)       # 4
let nested = [[1, 2], [3, 4]]
print nested[1][0]      # 3
```

Lists are plain Python lists at runtime, so `len()`, indexing (`list[i]`), and the math builtins `sum()`/`min()`/`max()`/`mean()` all work directly on them. There's no `.append()` or slice syntax; build a new list with `+` (list concatenation) instead:

```neuva
let a = [1, 2]
let b = a + [3]    # [1, 2, 3]
```

---

## Dict

**Syntax**

```neuva
let d = {key_expr: value_expr, ...}
let empty = {}
```

**Examples**

```neuva
let config = {"lr": 0.001, "epochs": 20}
let lr = config["lr"]              # 0.001
let nested = {"a": [1, 2, 3], "b": {"c": 4}}
print nested["b"]["c"]             # 4
```

A dict literal produces a plain Python `dict` at runtime, so any of Python's native dict methods work on it via a method call — `d.keys()`, `d.values()`, `d.items()`, `d.get("lr", 0.0)`, etc. — with no special Neuva syntax required. Indexing a missing key raises a runtime error; use `.get(key, default)` for a fallback instead of a bare `d[key]`.

---

## Strings

**Syntax**

```neuva
"regular string"
"""
triple-quoted
multiline string
"""
```

Regular strings are double-quoted and can't contain a literal `"` or span multiple lines. Triple-quoted strings (`"""..."""`) can span multiple lines and are otherwise identical — neither form interprets escape sequences (`\n`, `\t`, etc. are kept literally, not translated).

**Interpolation.** Any string literal substitutes `{name}` with the value of a variable named `name`:

```neuva
let acc = 93
print "Accuracy: {acc}%"    # Accuracy: 93%
```

If `name` isn't a defined variable, `{name}` is left in the output unchanged rather than raising an error.

---

## String Functions

| Function | Description |
|---|---|
| `len(s)` | Length of a string (or list) |
| `upper(s)` | Uppercase |
| `lower(s)` | Lowercase |
| `strip(s)` | Remove leading/trailing whitespace |
| `split(s, sep)` | Split into a list on `sep` (default `" "`) |
| `join(sep, list)` | Join a list into a string, `sep` between elements |
| `replace(s, old, new)` | Replace every occurrence of `old` with `new` |

```neuva
let name = "  Neuva  "
print strip(name)                 # "Neuva"
print upper(strip(name))          # "NEUVA"
print split("a,b,c", ",")         # ["a", "b", "c"]
print join("-", [1, 2, 3])        # "1-2-3"
print replace("abc", "b", "z")    # "azc"
```

---

## Math Functions

| Function | Description |
|---|---|
| `abs(x)` | Absolute value |
| `sqrt(x)` | Square root |
| `pow(x, n)` | `x` to the power `n` |
| `log(x)` | Natural log |
| `exp(x)` | `e ** x` |
| `round(x, n)` | Round `x` to `n` decimal places |
| `min(...)` | Minimum of its arguments |
| `max(...)` | Maximum of its arguments |
| `sum(list)` | Sum of a list |
| `mean(list)` | Arithmetic mean of a list |

```neuva
print sqrt(16)             # 4.0
print pow(2, 3)              # 8
print round(3.14159, 2)      # 3.14
print mean([1, 2, 3])        # 2.0
```

---

## Operators

| Operator | Description |
|----------|-------------|
| `+` `-` `*` `/` `%` | Arithmetic (`+` also concatenates strings) |
| `==` `!=` `<` `<=` `>` `>=` | Comparison |
| `and` `or` | Logical AND/OR, short-circuit evaluation |
| `!` / `not` | Logical not — `!x` and `not x` are exactly equivalent |
| `-` (unary) | Negation |
| `=` | Reassign an already-`let`-declared variable |
| `+=` `-=` `*=` `/=` | Compound assignment (see the Reassignment section above) |

---

## Comments

```neuva
# This is a comment
let x = 42    # inline comment
```
