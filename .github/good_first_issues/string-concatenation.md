# Add string concatenation with +

**Labels:** `good first issue`, `enhancement`, `interpreter`

## Description

The `+` operator only works on numbers today. Attempting to concatenate two strings
raises a Python `TypeError` that bubbles out as an unformatted crash rather than a
friendly Neuva error. String concatenation with `+` is one of the most natural
operations in any language and is a common first thing people try.

## Current behavior

```nva
let first = "hello"
let second = " world"
let message = first + second
print message
```

Produces an unhandled `TypeError` at runtime instead of printing `hello world`.

## Desired behavior

```nva
let greeting = "Hello, " + "Neuva!"
print greeting
# Hello, Neuva!

let name = "iris"
let path = "examples/data/" + name + ".csv"
print path
# examples/data/iris.csv

let data = load("examples/data/iris.csv")
let n = 120
print "Training on " + n + " samples"
# Training on 120 samples  (int coerced to string when mixed with +)
```

## Implementation notes

- The `+` lambda in `_OPS` inside `neuva/interpreter/interpreter.py` is
  `lambda a, b: a + b`. Python already raises `TypeError` for `str + int`; the fix
  is to coerce both operands to `str` when either one is a `str`:

  ```python
  "+": lambda a, b: str(a) + str(b) if isinstance(a, str) or isinstance(b, str) else a + b,
  ```

- No grammar or AST changes are needed — `+` is already a binary operator.
- Add a `RuntimeError_` guard for any remaining type incompatibilities so the error
  message is formatted correctly rather than crashing with a raw Python exception.

## Files to change

- `neuva/interpreter/interpreter.py` (`_OPS` dict, and optionally `eval_BinaryExpr`)
