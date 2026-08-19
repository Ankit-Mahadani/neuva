# Contributing to Neuva — A Practical Guide

This is a "how do I actually do X" guide for the four most common kinds of change. For process (branches, PRs, code style), see [CONTRIBUTING.md](../CONTRIBUTING.md). For how the pieces fit together, see [ARCHITECTURE.md](ARCHITECTURE.md) first if you haven't read it — this guide assumes you know the pipeline (grammar → transformer → AST → type checker → interpreter → backend).

---

## (a) Adding a New Keyword, End to End

There are two shapes this usually takes: a new **operator** (reuses an existing AST node) and a new **statement** (needs a new AST node). Both worked examples below are real features from this repo — you can `git log -p` the actual commits if you want to see the literal diffs.

### Worked example 1: the `not` keyword (operator — no new AST node needed)

`not` was added as a keyword-spelled alternative to the existing `!` unary operator. Because it means exactly the same thing as `!`, most of the pipeline needed zero changes.

1. **`neuva/parser/grammar.lark`** — add the keyword terminal, anchored with `_KW_BOUNDARY` so it doesn't swallow the prefix of a longer identifier:
   ```lark
   _NOT: "not" _KW_BOUNDARY
   ```
2. **Grammar rule** — add an alternative to the `unary` rule that reuses the *same* transformer callback name (`not_`) as `!` already uses:
   ```lark
   ?unary: "-" unary -> neg
         | "!" unary -> not_
         | _NOT unary -> not_
         | call_or_attr
   ```
3. **`neuva/parser/transformer.py`** — nothing new; `not_` already existed:
   ```python
   def not_(self, items): return BinaryExpr(op="not", left=None, right=items[0])
   ```
4. **`neuva/parser/ast_nodes.py`** — nothing new; `not` reuses the existing `BinaryExpr` node with `op="not"`.
5. **`neuva/interpreter/interpreter.py`** — nothing new; `eval_BinaryExpr` already special-cases `op == "not"`:
   ```python
   if op == "not":
       return not self.evaluate(node.right)
   ```
6. **`neuva/typechecker.py`** — nothing new; `_check_expr`'s generic `BinaryExpr` branch recurses into `.left`/`.right` regardless of `op`.

**Test** (see `tests/test_language_complete.py`):
```python
def test_not_keyword_matches_bang_operator():
    interp = _run('let a = not true\nlet b = !true\n')
    assert interp.env.get("a") == interp.env.get("b") == False
```

### Worked example 2: `match`/`case`/`default` (statement — needs a new AST node)

A genuinely new statement needs a change at every stage:

1. **`neuva/parser/grammar.lark`** — keyword terminals, grammar rules, and registering the new statement as an alternative of `statement`:
   ```lark
   statement: ... | match_stmt | ...

   match_stmt: _MATCH expr "{" NEWLINE case_clause* default_clause? "}" NEWLINE
   case_clause: _CASE literal ":" statement
   default_clause: _DEFAULT ":" statement

   _MATCH: "match" _KW_BOUNDARY
   _CASE: "case" _KW_BOUNDARY
   _DEFAULT: "default" _KW_BOUNDARY
   ```
   Note `case_clause` takes a `literal`, not a full `expr` — that's a deliberate, documented limitation (see LANGUAGE_REFERENCE.md's `match` section): supporting arbitrary case expressions would need pattern-matching semantics this interpreter doesn't have.

2. **`neuva/parser/ast_nodes.py`** — a new dataclass to carry the statement's data through the rest of the pipeline:
   ```python
   @dataclass
   class MatchStatement(Node):
       subject: Any = None
       cases: List[Any] = field(default_factory=list)  # (value_node, stmt) tuples
       default: Any = None
   ```

3. **`neuva/parser/transformer.py`** — `case_clause`/`default_clause` return small tagged dicts (a common pattern in this transformer for "this needs to be assembled by its parent rule, not turned into a final AST node yet" — see `elif_clause`/`else_clause` for the same pattern), and `match_stmt` assembles them into the real node:
   ```python
   def case_clause(self, items):
       return {"__type__": "case", "value": items[0], "stmt": items[1] if len(items) > 1 else None}

   def default_clause(self, items):
       return {"__type__": "default", "stmt": items[0] if items else None}

   def match_stmt(self, items):
       clean = [i for i in items if i is not None and not isinstance(i, Token)]
       subject, cases, default = clean[0], [], None
       for item in clean[1:]:
           if item.get("__type__") == "case":
               cases.append((item["value"], item["stmt"]))
           elif item.get("__type__") == "default":
               default = item["stmt"]
       return MatchStatement(subject=subject, cases=cases, default=default)
   ```

4. **`neuva/interpreter/interpreter.py`** — a `visit_MatchStatement` method (dispatch is automatic — `visit()` looks up `visit_{type(node).__name__}` by name, no registration needed):
   ```python
   def visit_MatchStatement(self, node: MatchStatement) -> None:
       subject = self.evaluate(node.subject)
       for value_node, stmt in node.cases:
           if subject == self.evaluate(value_node):
               if stmt is not None:
                   self.visit(stmt)
               return
       if node.default is not None:
           self.visit(node.default)
   ```

5. **`neuva/typechecker.py`** — a `MatchStatement` branch in `_check_stmt` so undefined-name checking still recurses into it (easy to forget — a statement type the checker doesn't know about is just silently skipped, not an error, so a bug here shows up as *missing* diagnostics rather than a crash):
   ```python
   elif t == "MatchStatement":
       self._check_expr(stmt.subject, scope, in_function)
       for value_node, case_stmt in stmt.cases:
           self._check_expr(value_node, scope, in_function)
           if case_stmt is not None:
               self._check_stmt(case_stmt, scope, in_function)
       if stmt.default is not None:
           self._check_stmt(stmt.default, scope, in_function)
   ```

6. **Also update, if relevant:** `neuva/cli.py`'s `_REPL_KEYWORDS` tuple (so `:help` in the shell lists the new keyword) and `playground/neuva_web.py` (a fully separate reimplementation — see ARCHITECTURE.md — that does **not** automatically gain new language features).

**Tests** — see `tests/test_language_complete.py` for the full set (`test_match_case_hits_matching_case`, `test_match_case_falls_through_to_default`, `test_match_case_no_default_no_match_is_noop`, `test_match_case_string_scrutinee`).

---

## (b) Adding a New Layer Type

Layer types are the easiest extension point in the codebase, *because* the grammar doesn't know layer names at all — it only knows argument shapes (see ARCHITECTURE.md's transformer section). `layer norm(64)` and `layer pool(2)` both parse via the same `layer_pool` grammar rule (single `NUMBER` in parens); the string `"norm"` vs `"pool"` is just captured into `LayerStatement.name` and interpreted generically downstream.

**If your new layer reuses an existing argument shape** (one number, two numbers, `in -> out, activation_name`, etc. — check the `layer_stmt` alternatives in `grammar.lark`), you need **no grammar or transformer change at all**. Just add a new branch to `NeuvaModel.__init__` in `neuva/backend/torch_backend.py`, following the pattern of the existing `dropout`/`norm` branches:

```python
elif lname == "norm":
    num_features = int(args[0])
    self.linears.append(nn.BatchNorm1d(num_features))
    self.activations.append(lambda x: x)
    self.activation_names.append("linear")
```

A minimal new layer (say, a hypothetical `layer gelu_dense(...)`, reusing the existing `in -> out, activation` shape) would be:

```python
elif lname == "my_layer":
    in_size, out_size, act_name = args[0], args[1], args[2]
    self.linears.append(nn.Linear(in_size, out_size))  # or whatever nn.Module you need
    self.activations.append(_ACTIVATIONS.get(act_name, lambda x: x))
    self.activation_names.append(act_name)
```

If the layer needs custom `forward()` behavior beyond a single `nn.Module` call (like the RNN/attention/transformer/embedding layers do — see `_RNNWrapper`, `_AttentionWrapper`, `_TransformerWrapper`, `_EmbeddingWrapper` in `torch_backend.py`), write a small `nn.Module` wrapper class and append an instance of *that* instead. Every wrapper in this codebase follows the same contract: accept whatever shape `NeuvaDataset` hands it (2D tabular by default), reshape/pool as needed in `forward()`, and expose enough attributes (`in_features`/`out_features` or equivalent) that `_neuva_model_summary` in `interpreter.py` and `export_onnx`'s shape inference can introspect it.

**You don't need to touch save/load.** `NeuvaModel.layer_specs` captures `(name, args)` for every layer generically as it's built (`self.layer_specs.append((lname, list(args)))`), and `save_model`/`load_model` round-trip through that list plus `state_dict()` — so a new layer type gets working `save`/`load` automatically, as long as its `__init__` branch is deterministic from `(name, args)` alone.

**If your layer genuinely needs a new argument shape** the grammar doesn't already support (e.g. 4+ numeric args, or a mix the existing `layer_stmt` alternatives don't cover), you'll need a new `layer_stmt` alternative in `grammar.lark` plus a matching builder method in `transformer.py` (follow `layer_triple`/`layer_pair` as templates) — then the `NeuvaModel.__init__` branch as above.

---

## (c) Adding a New Built-in Function

One line in `NeuvaInterpreter.__init__` (`neuva/interpreter/interpreter.py`):

```python
self.env.set("my_builtin", lambda x, y: x + y)
```

Any plain Python `callable` works — `eval_CallExpr` calls it directly with evaluated arguments if it's `callable` and not a `NeuvaFunction`. No AST node, no grammar change, no transformer change.

Two easy-to-miss follow-ups if the type checker runs on programs that call it (which is the default — only skipped with `--no-check`):

- Add the name to `_BUILTIN_NAMES` in `neuva/typechecker.py`. This set is **not** derived from the interpreter's environment automatically — it's a separate, manually-maintained set. Forgetting this means any program calling your new builtin fails type checking with `undefined variable 'my_builtin'` even though it runs fine with `--no-check`.
- Optionally add it to `_REPL_BUILTINS` in `neuva/cli.py` so `:help` in the shell lists it.

If the builtin is genuinely stdlib-shaped (a helper composed from other Neuva-visible operations, not something needing raw Python/torch access), consider writing it as an `fn` in one of the `.nva` files under `neuva/stdlib/` instead — see `neuva/stdlib/metrics.nva` for the pattern. Reserve interpreter-level builtins for things that need direct access to Python/PyTorch internals (tensors, sklearn, file I/O) that pure Neuva code can't express — that's also why `datasets.nva`/`optimizers.nva`/`callbacks.nva` are each a thin `fn` wrapper around a hidden `__`-prefixed interpreter builtin (`__load_iris`, `__use_sgd`, `__on_epoch_end`, ...): the real logic needs Python interop, but the public name is still plain Neuva.

---

## (d) Writing Tests

Tests are plain `pytest` functions under `tests/`. The standard pattern: parse a source string, run it through the interpreter, and assert on captured stdout (`capsys`) and/or interpreter state (`interp.env.get(name)`).

`tests/test_language_complete.py` is the most direct template — it's the test file written for this exact feature-completion pass (match/case, dict, `not`, reassignment, string/math builtins, triple-quoted strings):

```python
from neuva.parser import NeuvaParser
from neuva.interpreter.interpreter import NeuvaInterpreter, RuntimeError_

def _run(source: str) -> NeuvaInterpreter:
    interp = NeuvaInterpreter()
    interp.visit(NeuvaParser().parse(source))
    return interp

def test_dict_literal_and_indexing():
    interp = _run('let config = {"lr": 0.001, "epochs": 20}\nlet lr = config["lr"]\n')
    assert interp.env.get("config") == {"lr": 0.001, "epochs": 20}
    assert interp.env.get("lr") == 0.001

def test_dict_missing_key_raises():
    import pytest
    with pytest.raises(RuntimeError_):
        _run('let d = {"a": 1}\nlet x = d["missing"]\n')

def test_not_keyword_matches_bang_operator():
    interp = _run('let a = not true\nlet b = !true\n')
    assert interp.env.get("a") == interp.env.get("b") == False
```

For output rather than state, use `capsys`:

```python
def test_match_case_hits_matching_case(capsys):
    _run(
        'let result = 1\n'
        'match result {\n'
        '    case 0: print "negative"\n'
        '    case 1: print "positive"\n'
        '}\n'
    )
    assert capsys.readouterr().out.strip() == "positive"
```

For backend/training behavior specifically, `tests/test_early_stopping.py` is a good template for driving `NeuvaModel`/`NeuvaTrainer` directly (bypassing the interpreter) when you want a controlled, deterministic setup:

```python
import torch
from neuva.parser.ast_nodes import LayerStatement
from neuva.backend.torch_backend import NeuvaModel, NeuvaTrainer

def test_early_stop_halts_before_epoch_limit(capsys):
    torch.manual_seed(0)
    layers = [LayerStatement(name="dense", args=[4, 1, "linear"])]
    model = NeuvaModel(layers)

    class FakeDS:
        X = torch.zeros(16, 4)
        y = torch.zeros(16, 1)

    NeuvaTrainer().train(model, FakeDS(), epochs=200, lr=0.05, loss_fn="mse", early_stop=3)
    out = capsys.readouterr().out
    assert "Early stopping" in out
    assert "Epoch 200/200" not in out
```

And for parser-level-only checks (no interpretation), assert directly on the AST, as in `tests/test_early_stopping.py`'s `test_early_stop_parses`:

```python
def test_early_stop_parses():
    src = "train M on data for 100 epochs, lr=0.001, loss=mse, early_stop=10\n"
    stmt = NeuvaParser().parse(src).body[0]
    opts = {o.key: o.value for o in stmt.options}
    assert opts["early_stop"].value == 10
```

If your feature has a runnable end-to-end story, add a `.nva` file under `examples/` too (see `examples/hyperparameter_search.nva`, `examples/transfer_learning.nva`). Note that `tests/test_examples.py` does **not** automatically pick up every file in `examples/` and run it — it only *parses* (not interprets) files that reference `iris.csv`, plus a fixed, hand-maintained list of filenames for a `parses`-only check. If you want your new example actually executed in CI (not just parsed), add an explicit test for it — see `tests/test_stdlib.py::test_stdlib_demo_example_runs` for the pattern (`NeuvaInterpreter().visit(NeuvaParser().parse_file(path))`).

Run the full suite from the repo root:

```bash
python -m pytest tests/ -q
```
