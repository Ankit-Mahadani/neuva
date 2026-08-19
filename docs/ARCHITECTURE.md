# Neuva Architecture

How a `.nva` file turns into a running program, and the design decisions that shape what Neuva can and can't express.

---

## Pipeline Overview

```
 .nva source
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. LEXER  (declarative — inside grammar.lark's terminals)    │
│    Turns raw text into tokens: NAME, NUMBER, STRING,         │
│    keywords, operators. Lark generates this from the         │
│    terminal definitions at the bottom of grammar.lark.       │
└─────────────────────────────────────────────────────────────┘
     │  token stream
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PARSER  (neuva/parser/grammar.lark, Lark Earley parser)   │
│    Applies the grammar rules to the token stream and         │
│    produces a Lark parse tree (nested Tree/Token nodes,       │
│    shaped exactly like the grammar's rule hierarchy).         │
└─────────────────────────────────────────────────────────────┘
     │  Lark parse tree
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TRANSFORMER  (neuva/parser/transformer.py)                │
│    A Lark Transformer that walks the parse tree bottom-up    │
│    and rebuilds it as typed AST dataclasses from              │
│    neuva/parser/ast_nodes.py (Program, LetStatement,          │
│    TrainStatement, MatchStatement, ...).                     │
└─────────────────────────────────────────────────────────────┘
     │  typed AST (a Program node)
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. TYPE CHECKER  (neuva/typechecker.py)                      │
│    A best-effort, non-blocking-by-default static pass over   │
│    the AST: undefined-name checks, train-option sanity       │
│    checks, layer dimension checks, for-loop shadow            │
│    warnings. Runs between parsing and interpreting unless     │
│    `--no-check` is passed; `--strict` promotes its warnings   │
│    to errors.                                                │
└─────────────────────────────────────────────────────────────┘
     │  same AST (unchanged — the checker never rewrites it)
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. INTERPRETER  (neuva/interpreter/interpreter.py)            │
│    A tree-walking interpreter. NeuvaInterpreter.visit(node)   │
│    dispatches to visit_<NodeType> for statements and          │
│    .evaluate(node) dispatches to eval_<NodeType> for          │
│    expressions, by looking up the node's Python class name.   │
└─────────────────────────────────────────────────────────────┘
     │  calls into the backend for anything ML-shaped
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. PYTORCH BACKEND  (neuva/backend/torch_backend.py,          │
│    neuva/backend/data_loader.py)                              │
│    torch_backend.py builds a real nn.Module (NeuvaModel) out  │
│    of a model's layer_specs and runs the actual training      │
│    loop (NeuvaTrainer) with a real optimizer/scheduler/loss.  │
│    data_loader.py loads CSV files and sklearn toy datasets     │
│    (iris/housing/digits) into DataSet objects (torch tensors). │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
   stdout / a trained model / a saved .nva file
```

Each stage's output is the next stage's input, and each stage lives in its own file/module — there's no cross-cutting global state. `neuva/cli.py` is the only place that wires all six stages together for a single run (`neuva file.nva`) or a REPL session (`neuva shell`).

---

## 1–2. Lexer and Parser

Neuva's real tokenizer is **not** a separate hand-written pass. Lark generates it directly from the terminal (`UPPERCASE`) definitions at the bottom of `neuva/parser/grammar.lark` — things like:

```lark
_KW_BOUNDARY: /(?![A-Za-z0-9_])/
_TRAIN: "train" _KW_BOUNDARY
NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
STRING: "\"" /[^"\\]*/ "\""
TRIPLE_STRING.2: /"""[\s\S]*?"""/
AUGOP: "+=" | "-=" | "*=" | "/="
```

Each keyword terminal is anchored with a negative-lookahead `_KW_BOUNDARY` so that, say, `print_loss_curve` lexes as one `NAME` token rather than a `PRINT` keyword followed by `_loss_curve` — this is what lets user identifiers safely start with a keyword. The leading underscore on keyword terminal names (`_TRAIN`, `_LET`, ...) is Lark convention for "match this but don't put it in the parse tree," matching how plain string literals in the grammar are already dropped by default.

**`neuva/lexer/` is legacy and unused.** There's a separate hand-written `Lexer`/`Token`/`TokenType` module under `neuva/lexer/`, left over from an earlier, non-Lark implementation. `NeuvaParser` (`neuva/parser/parser.py`) never imports it — it goes straight from source text to a Lark `Lark(...)` instance built from `grammar.lark`. Don't extend `neuva/lexer/` when adding a language feature; it has no effect on what the language actually accepts.

Parsing itself uses Lark's **Earley** parser (`parser="earley"` in `NeuvaParser.__init__`), not LALR — Earley tolerates the grammar's ambiguity more gracefully (`ambiguity="resolve"`) and doesn't require the grammar to be strictly LL/LR-shaped, at some cost to parse speed versus a table-driven parser. `propagate_positions=True` is what gives every AST node its `line`/`col` for Rust-style error messages.

---

## 3. Transformer → AST

`NeuvaTransformer` (a `lark.Transformer` subclass) defines one method per grammar rule name (`train_stmt`, `layer_dense`, `match_stmt`, `assign`, `aug_assign`, ...). Lark calls these bottom-up as it walks the parse tree, so by the time, e.g., `train_stmt` runs, its children have already been transformed into `TrainOption`/AST nodes rather than raw `Tree`/`Token` objects. The result is a `Program(body=[...])` — a tree of plain dataclasses from `neuva/parser/ast_nodes.py`, with no Lark types left anywhere in it.

A notable transformer quirk worth knowing before you touch it: the shape of a `layer` statement's arguments (not the layer's name) decides which grammar rule fires — `layer_dense` matches `NUM -> NUM, NAME` (e.g. `dense(4 -> 16, relu)`), `layer_conv` matches `NUM -> NUM, NUM` (e.g. `conv(1 -> 32, 3)`), `layer_pair` matches `NUM, NUM` (e.g. `attention(8, 2)`), and so on. The actual layer *name* (`dense`, `conv`, `norm`, `attention`, ...) is just captured as a string into `LayerStatement.name`; the interpreter and backend never see a `dense`-specific or `conv`-specific AST node, only a generic `LayerStatement(name, args)`. This is why adding a *new layer type that reuses an existing argument shape* (see CONTRIBUTING_GUIDE.md) needs no grammar or transformer change at all — only a new branch in `NeuvaModel.__init__`.

---

## 4. Type Checker

`TypeChecker.check(program)` (`neuva/typechecker.py`) is explicitly **not** a sound type system — the module docstring calls it a "best-effort" pass. Two structural reasons why:

- Neuva has no block scoping (see below), so an `if`/`for`/`while` body shares its enclosing scope rather than introducing a new one.
- Function bodies are checked against every name that exists *anywhere* in the file (not just names declared before the function), because a function's body only actually runs later, through a closure — by then, a global defined after the function is perfectly valid to reference.

Concretely, it walks the AST once, collecting a pre-pass of every top-level name (`_collect_top_names`), then does a second pass (`_check_stmt`/`_check_expr`) that reports things like undefined variables/models, `train` options with an invalid literal value (`lr <= 0`, non-positive `early_stop`/`lr_warmup`, an unrecognized `loss` name), a for-loop variable shadowing an existing name (a **warning**, not an error), and dense-to-dense layer dimension mismatches. Only literal values are checked — `train M on d for 10 epochs, lr = lr_candidate` where `lr_candidate` is a variable can't be checked statically and is skipped rather than guessed at.

The checker runs between parsing and interpreting (`neuva/cli.py::main`) unless `--no-check` is passed. With `--strict`, everything in `checker.warnings` is folded into the error list before the pass-fail decision is made, so a shadow warning becomes a hard failure. It never mutates the AST — the interpreter that runs afterward sees exactly the same tree the transformer produced.

---

## 5. Interpreter

`NeuvaInterpreter` is a straightforward tree-walker with two dispatch entry points, both using Python's `type(node).__name__` to find a handler by naming convention:

```python
def visit(self, node):        # statements — no return value expected
    method = f"visit_{type(node).__name__}"
    ...

def evaluate(self, node):     # expressions — returns a Python/torch value
    method = f"eval_{type(node).__name__}"
    ...
```

So `TrainStatement` is handled by `visit_TrainStatement`, `BinaryExpr` by `eval_BinaryExpr`, and so on. Adding a new statement or expression node type means adding a new `visit_`/`eval_` method with a matching name — nothing registers handlers explicitly.

All built-in functions (`range`, `len`, `load`, `accuracy`, `predict`, `sqrt`, `upper`, ...) are just Python callables stashed into the root `Environment` in `NeuvaInterpreter.__init__` via `self.env.set("name", callable)`. A function call (`eval_CallExpr`) looks the callee up by name, and if it's a plain Python `callable` (not a `NeuvaFunction`), calls it directly with the evaluated arguments — so builtins need no AST node of their own.

### Environment / scoping model

`Environment` (in `interpreter.py`) is deliberately **flat within a call frame** — there is exactly one `Environment` per function call, and `if`/`for`/`while`/`match` bodies all execute directly against the enclosing frame's environment rather than pushing a new one. Only a function call (`_call_function`) creates a new child `Environment`, chained to the closure's environment as its `parent`.

This single decision explains several behaviors documented in LANGUAGE_REFERENCE.md:

- **`let` inside a loop doesn't shadow per-iteration** — `for i in range(3) { let x = i }` overwrites the same `x` in the same frame each iteration, rather than creating a fresh binding each time. That's also why the type checker's for-loop shadow check is a warning, not an error: shadowing an *outer* name from inside a loop body is legal and sometimes intentional, since it's really the same frame.
- **`let` vs. `=`/`+=` are fundamentally different operations**, not just syntax sugar for each other. `Environment.set` (used by `let`) always writes into the *current* frame. `Environment.assign` (used by `=`/`+=`/`-=`/`*=`/`/=`) walks the parent chain to find whichever frame *already* owns the name and mutates it there. This is exactly why a function can only mutate an outer/global variable through `+=` (or `=`), never by repeating `let` — a second `let counter = counter + 1` inside a function body would create a brand-new local `counter`, shadow the outer one for the rest of that call, and leave the outer variable untouched after the function returns.
- **A `case`/`default` statement in a `match` runs in the same frame as the `match` itself** — there's no per-case scope, consistent with everything else in the language.

---

## 6. PyTorch Backend

`torch_backend.py` is where Neuva stops being a toy interpreter and starts doing real machine learning:

- `NeuvaModel(nn.Module)` builds one real `nn.Module` per `LayerStatement` (a `nn.Linear`, `nn.Conv2d`, `nn.Dropout`, `nn.BatchNorm1d`, or a small wrapper class like `_RNNWrapper`/`_AttentionWrapper`/`_TransformerWrapper`/`_EmbeddingWrapper` for the more exotic layer types) and chains them in `forward()`. It also stores `layer_specs`/`output_specs` — the original `(name, args)` for every layer — alongside the built modules, so `save_model`/`load_model` can reconstruct *any* architecture from a checkpoint rather than only being able to infer a dense-only stack from tensor shapes.
- `NeuvaTrainer.train(...)` is the actual training loop: mini-batches via `torch.randperm`, a real `torch.optim` optimizer (Adam by default, or SGD/RMSprop if `optimizers.nva` set a pending override), an optional `lr_scheduler` (`step`/`cosine`), `lr_warmup` ramping, early stopping, and per-epoch callback hooks — all real PyTorch, not simulated.
- `data_loader.py`'s `DataSet` wraps CSV-loaded or sklearn-loaded data as `torch.Tensor`s and implements `.split()`/`.normalize()`/`.shuffle()`/`.augment()`/`.oversample()`/`.undersample()` as methods that return new `DataSet`s.

The interpreter never touches `torch` directly — every ML-shaped statement (`visit_TrainStatement`, `visit_SaveStatement`, `visit_PredictStatement`, the `accuracy`/`predict`/`predict_proba`/`export_onnx` builtins) is a thin call into this module.

---

## The Web Playground Is a Separate Implementation

`playground/neuva_web.py` is **not** the `neuva` Python package running in the browser — it's an independent, from-scratch reimplementation of a Neuva lexer, parser, and interpreter, written to have zero dependencies beyond the Python standard library (no `lark`, `torch`, or `pandas`), so it can run under [Pyodide](https://pyodide.org) in WebAssembly. It simulates the ML backend (`model`/`train`/`save`/`predict`) in pure Python instead of calling real PyTorch, which is why programs that only exercise the language (control flow, functions, printing, data structures) behave identically to the CLI, but numeric training results are simulated rather than real.

Practically, this means: a language feature added to `grammar.lark`/`transformer.py`/`interpreter.py` does **not** automatically show up in the playground — `playground/neuva_web.py` has its own hand-written `Lexer`/`Parser`/AST/interpreter classes that need the equivalent change made independently. When adding a language feature and the playground matters to you, check whether `playground/neuva_web.py` needs a matching update; nothing enforces the two staying in sync.

---

## Where Things Live

| Concern | File |
|---|---|
| Grammar / real tokenizer | `neuva/parser/grammar.lark` |
| Parse-tree → AST | `neuva/parser/transformer.py` |
| AST node definitions | `neuva/parser/ast_nodes.py` |
| Parser entry point | `neuva/parser/parser.py` |
| Static checks | `neuva/typechecker.py` |
| Interpreter | `neuva/interpreter/interpreter.py` |
| PyTorch model/training | `neuva/backend/torch_backend.py` |
| Data loading | `neuva/backend/data_loader.py` |
| Stdlib modules (`.nva` source) | `neuva/stdlib/*.nva` |
| CLI + REPL | `neuva/cli.py` |
| Version string | `neuva/version.py` |
| Legacy, unused hand-written lexer | `neuva/lexer/` |
| Browser playground (separate reimplementation) | `playground/neuva_web.py`, `playground/index.html` |
