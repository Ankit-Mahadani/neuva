# Neuva Examples

## Iris Classifier — Line by Line

The file `examples/digit_classifier.nva` trains a small neural network on the Iris dataset and achieves ~93% accuracy. Here is a complete walkthrough.

---

### The full program

```neuva
# Iris Classifier — 4 features, 3 species classes
let data = load("examples/data/iris.csv")
let train_data, test_data = data.split(0.8)

model DigitNet {
    layer dense(4 -> 16, relu)
    layer dense(16 -> 3, softmax)
}

train DigitNet
    on train_data
    for 50 epochs
    lr   = 0.01
    loss = crossentropy

let acc = accuracy(DigitNet, test_data)
print "Test accuracy:", acc
save DigitNet to "digit_model.nva"
```

---

### Line by line

**Line 1 — comment**

```neuva
# Iris Classifier — 4 features, 3 species classes
```

Comments start with `#` and are ignored by the interpreter.

---

**Line 2 — load the dataset**

```neuva
let data = load("examples/data/iris.csv")
```

`load` reads the CSV file and returns a `DataSet` object. The Iris dataset has 150 rows and 5 columns: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (features), and `species` (label, values 0/1/2).

Neuva treats the last column as the label and the rest as features. Because `species` contains integers, the label tensor uses `torch.long`, which is what cross-entropy loss expects.

---

**Line 3 — train/test split**

```neuva
let train_data, test_data = data.split(0.8)
```

`split(0.8)` randomly shuffles the 150 rows using `torch.randperm` and returns two `DataSet` objects: 80% (120 rows) for training and 20% (30 rows) for evaluation. Both sides share the same feature and label tensors, just different row slices.

The two names on the left side of `let` unpack the tuple returned by `split`.

---

**Lines 5–8 — model definition**

```neuva
model DigitNet {
    layer dense(4 -> 16, relu)
    layer dense(16 -> 3, softmax)
}
```

`model` defines a named neural network. Each `layer` line describes one transformation:

- `dense(4 -> 16, relu)` — a fully connected layer taking 4 inputs (one per feature) and producing 16 outputs, with ReLU applied element-wise. Maps to `nn.Linear(4, 16)` followed by `torch.relu`.
- `dense(16 -> 3, softmax)` — another fully connected layer narrowing from 16 to 3 outputs (one per class), with softmax converting raw scores to probabilities. Maps to `nn.Linear(16, 3)` followed by `torch.softmax`.

At this point the model is defined but not yet instantiated or trained — the weights are created when `train` runs.

---

**Lines 10–14 — training**

```neuva
train DigitNet
    on train_data
    for 50 epochs
    lr   = 0.01
    loss = crossentropy
```

This is the multi-line form of the `train` statement. It:

1. Instantiates `DigitNet` as a PyTorch `nn.Module`.
2. Wraps `train_data` to expose `.X` (shape `[120, 4]`) and `.y` (shape `[120]`, long).
3. Creates an Adam optimiser with `lr = 0.01`.
4. Uses `nn.CrossEntropyLoss` because `loss = crossentropy`.
5. Runs 50 forward + backward passes, printing the loss after each epoch.

Because the label tensor is `long` (integer class indices), it is passed directly to `CrossEntropyLoss` without any conversion.

After training, `DigitNet` in the environment is replaced with the trained `NeuvaModel` (a live PyTorch module with learned weights).

---

**Line 16 — evaluate accuracy**

```neuva
let acc = accuracy(DigitNet, test_data)
```

`accuracy` calls `evaluate_accuracy`, which:
1. Sets the model to `eval()` mode.
2. Runs `model(test_data.X)` with `torch.no_grad()`.
3. Takes `argmax` over the 3 output logits to get the predicted class for each sample.
4. Compares predictions to `test_data.y` and returns `correct / total` as a float.

On Iris with this architecture and 50 epochs, this typically returns around `0.93`.

---

**Line 17 — print result**

```neuva
print "Test accuracy:", acc
```

Prints the string and the float value side by side. Output looks like:

```
Test accuracy: 0.9333333333333333
```

---

**Line 18 — save the model**

```neuva
save DigitNet to "digit_model.nva"
```

Saves the model's weights and activation names to `digit_model.nva` using `torch.save`. The file can be loaded back in another program:

```neuva
let loaded = load("digit_model.nva")
let acc2   = accuracy(loaded, test_data)
```

`load` detects the `.nva` extension and calls `load_model`, which reconstructs the layer sizes from the saved weight shapes and restores the activation functions from the metadata. The round-trip is lossless.

---

## Other Examples

| File | What it shows |
|------|---------------|
| [`examples/hello.nva`](../examples/hello.nva) | Variables, print, basic arithmetic |
| [`examples/linear_regression.nva`](../examples/linear_regression.nva) | Regression with MSE loss |
| [`examples/cnn_classifier.nva`](../examples/cnn_classifier.nva) | `conv`, `pool`, `flatten` layer syntax |
| [`examples/digit_classifier.nva`](../examples/digit_classifier.nva) | Full iris classification pipeline |
