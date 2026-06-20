# Changelog

All notable changes to Neuva are documented here.

---

## [1.0.0] — First stable release
- **First stable milestone** — all core language features complete and tested
- F-string interpolation: `"Hello, {name}"` substitutes variables inline
- List literals and index access: `let scores = [85, 90, 78]`, `scores[0]`
- `and` / `or` keywords with correct short-circuit evaluation and proper precedence
- `dropout(p)` maps to `nn.Dropout` as a real regularization layer
- `print ModelName` shows a formatted summary with layer descriptions and parameter count
- Built-in `len()` function for lists
- CLI: `neuva shell` (interactive REPL), `--version`, `--help`
- CLI: `Error: file 'foo.nva' not found` — clean message, no Python traceback
- `neuva shell` version banner reads `pyproject.toml` dynamically
- 54 tests passing across lexer, parser, AST, interpreter, and examples
- All 13 example files run without errors

---

## [0.4.0] — Day 34
- F-string interpolation in string literals: `"Hello, {name}"`
- List literals and index access: `[1, 2, 3]`, `scores[0]`
- `and` / `or` keywords with correct short-circuit evaluation
- `dropout(p)` as a real `nn.Dropout` layer in models
- `print ModelName` shows a formatted model summary with parameter count
- `neuva shell` banner now shows current version dynamically
- CLI error for missing files: `Error: file 'foo.nva' not found`
- Built-in `len()` function for lists

## [0.4.0-pre] — Day 33
- Mini-batch training with shuffled indices (`torch.randperm`)
- Average loss reported per epoch instead of last batch loss
- `batch_size` parameter (default 16) on `NeuvaTrainer.train`

## [0.3.0] — Day 32
- Automatic GPU/CPU detection: `DEVICE = torch.device("cuda" if ... else "cpu")`
- `NeuvaModel` moves to DEVICE at construction time
- `NeuvaDataset` tensors pinned to DEVICE
- `NeuvaTrainer` prints `Training on: cpu` / `Training on: cuda`

## [0.2.0] — Day 31
- `neuva shell` — interactive REPL with persistent interpreter state
- `neuva --version` / `-v` — print version and exit
- `neuva --help` / `-h` — print usage and exit
- Version read dynamically from `pyproject.toml`

## [0.1.5] — Day 29–30
- String concatenation with `+` operator (auto-coerces to str)
- `else if` / `elif` chained branches in `if` statements

## [0.1.4] — Day 27
- Three new example programs: `spam_classifier.nva`, `house_price.nva`, `control_flow_demo.nva`
- `test_examples.py` — parametrized pytest suite, 32/32 passing

## [0.1.3] — Day 25
- Rust-style error messages with did-you-mean suggestions for undefined variables
- Line and column numbers in all parse and runtime errors

## [0.1.2] — Day 23
- VS Code syntax-highlighting extension for `.nva` files

## [0.1.1] — Day 22
- Getting-started guide, language reference, and example documentation

## [0.1.0] — Day 20–21
- Real Iris dataset loading; 93.3% test accuracy end-to-end
- CNN layer support: `conv`, `pool`, `flatten`
- Real CSV data loading, train/test split, accuracy evaluation

## [0.0.5] — Day 12–17
- Real PyTorch backend connected; Neuva trains actual neural networks
- Multi-line `train` statement, method calls, built-in functions (`load`, `accuracy`)
- All three example programs running

## [0.0.4] — Day 10–11
- Control flow: `if`/`else`, `for`, `while`
- User-defined functions with `fn`, `return`
- Interpreter working; first Neuva program runs successfully

## [0.0.3] — Day 8–9
- AST node definitions and Lark transformer
- 6 AST tests, 5 interpreter tests passing

## [0.0.2] — Day 5–7
- Lark grammar file and parser
- 8 parser tests passing

## [0.0.1] — Initial commit
- Project scaffolding, README, package registration as `neuva-lang`
