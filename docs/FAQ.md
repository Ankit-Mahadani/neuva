# FAQ

Honest answers, grounded in what's actually in this repo today (v1.1.0).

---

### Is Neuva production ready?

No, and it doesn't try to be yet. Neuva is a real language with a real PyTorch backend — training actually happens, models actually save/load, and the test suite (`python -m pytest tests/ -q`) passes — but there are hard edges you'd hit quickly in production:

- The type checker is explicitly "best-effort," not sound — it catches obvious mistakes (undefined names, non-positive `lr`, dense-layer dimension mismatches) but has no real type inference, and most of it is skippable with `--no-check`.
- Some limitations are permanent-by-design, not bugs: no bare attribute access (`obj.field` doesn't exist, only `obj.method(...)`), `match`/`case` allows exactly one statement per case, `.freeze()`/`.unfreeze()` are whole-model only (no per-layer freeze).
- No multi-GPU support, no model registry/versioning, no production serving story beyond `export_onnx()` (and that only works for dense/conv-first, single-output models).

Good fits today: learning, prototyping small models quickly, teaching the "what does training a network actually involve" story without PyTorch boilerplate. Not yet a Keras/PyTorch-Lightning replacement for production training pipelines.

---

### Can I use Neuva with my existing PyTorch models?

Not directly, and mostly one-way. There's no way to import an existing `nn.Module` into a Neuva `model` block — Neuva builds its own `nn.Module` (`NeuvaModel`) from a `model { layer ... }` definition, and that's the only way a model comes into existence.

Going the other direction is partially possible: `export_onnx(model, "path.onnx")` exports a trained Neuva model to ONNX, which most PyTorch-adjacent tooling can then consume — but only for models whose first layer has a fixed input width (`dense`/`conv`-first, single-output). RNN/embedding-first and multi-output models raise a clear error instead of exporting something broken. There's also no supported way to load raw PyTorch `state_dict` weights into a `NeuvaModel` — `load("model.nva")` only understands Neuva's own checkpoint format.

---

### How do I contribute?

```bash
git clone https://github.com/Ankit-Mahadani/neuva.git
cd neuva
pip install -e ".[dev]"
pytest tests/
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the process (fork/branch/PR conventions) and [CONTRIBUTING_GUIDE.md](CONTRIBUTING_GUIDE.md) for the practical "how do I add a keyword / layer type / builtin / test" walkthroughs with real worked examples from this codebase. Issues labeled `good first issue` are a reasonable place to start: [github.com/Ankit-Mahadani/neuva/issues](https://github.com/Ankit-Mahadani/neuva/issues). Because the project is AGPL-3.0 licensed, contributions you submit are distributed under the same license.

---

### What datasets does Neuva support?

Two ways to get data in:

- **Any CSV file**, via `load("path.csv")` — numeric columns become features, the last numeric column becomes the label (integer → classification, float → regression). No datasets are bundled with the CLI install; you bring your own CSV.
- **Three built-in toy/benchmark datasets**, via `import datasets` then `load_iris()`, `load_housing()`, or `load_mnist_sample()`:
  - `load_iris()` — the classic 150-row Iris dataset, bundled with scikit-learn, no network needed.
  - `load_housing()` — the California housing regression dataset (~20,640 rows). Not bundled — the first call downloads it via scikit-learn and caches it locally, so it needs network access once.
  - `load_mnist_sample()` — **not real MNIST.** It's scikit-learn's bundled `load_digits` dataset (8×8-pixel handwritten digits, not the real 28×28 MNIST), capped at 100 rows, used purely as an offline stand-in so you can try a "digit classification" example with no download.

---

### How does Neuva compare to Keras?

Both aim to make "describe a model, train it" as short as possible, but they're different kinds of things. Keras is a Python API on top of a mature, widely-deployed framework (TensorFlow, with a PyTorch backend option too) — huge layer catalog, callback ecosystem, distributed training, serving tooling, years of production hardening. Neuva is a small standalone *language* — its own grammar, parser, and interpreter — that compiles ML-shaped statements down to a fairly small set of PyTorch primitives. It has a handful of layer types, one training loop implementation, and no ecosystem around it yet.

Practically: reach for Keras (or plain PyTorch/Lightning) for anything you intend to actually deploy or scale. Reach for Neuva if you want to see how little code training a model can take, or if you're using it as a teaching/prototyping tool.

---

### Can Neuva run on GPU?

Yes, automatically — `neuva/backend/torch_backend.py` picks `torch.device("cuda" if torch.cuda.is_available() else "cpu")` once at import time, and every model/dataset tensor is moved to that device. `neuva my_model.nva` prints `Training on: cuda` or `Training on: cpu` at the start of every `train` statement so you can confirm which one you got. There's currently no language-level way to force CPU when a CUDA device is available, and no multi-GPU support (`DataParallel`/`DistributedDataParallel` wrapping isn't implemented).

---

### What is the web playground?

A browser-based Neuva editor at `playground/index.html`, running entirely client-side via [Pyodide](https://pyodide.org) (Python compiled to WebAssembly). Importantly, it does **not** run the real `neuva` package — `playground/neuva_web.py` is a separate, from-scratch reimplementation of the lexer/parser/interpreter with no `lark`/`torch`/`pandas` dependency, and it simulates the ML backend in pure Python rather than doing real PyTorch training. It's meant for trying out language syntax without installing anything, not for real training runs. Run it locally with `cd playground && python -m http.server 8000`, then open `http://localhost:8000/` (it needs to be served over HTTP, not opened as a `file://` URL). The repo also has a GitHub Actions workflow that can publish it to GitHub Pages on push to `main`.

---

### How do I report a bug?

Open an issue at [github.com/Ankit-Mahadani/neuva/issues](https://github.com/Ankit-Mahadani/neuva/issues) — the repo has a bug-report issue template (`.github/ISSUE_TEMPLATE/bug_report.md`). The most useful report includes: a minimal `.nva` file that reproduces it, the output of `neuva --version`, and (if it looks like an interpreter/type-checker bug rather than a training-quality issue) whether it reproduces with `--no-check`. If you can turn the repro into a failing `pytest` case, even better — see [CONTRIBUTING_GUIDE.md](CONTRIBUTING_GUIDE.md#d-writing-tests) for the test-writing pattern.

---

### Is Neuva free?

Yes — Neuva is open source under the **GNU Affero General Public License v3.0 (AGPL-3.0)** (see [LICENSE](../LICENSE)). You can use, study, modify, and redistribute it freely. The AGPL's distinguishing clause is its network-use provision: if you run a modified version of Neuva as a service that others interact with over a network, you're required to make your modified source available to those users too, not just to people you hand a binary to directly. `pip install neuva-lang` itself costs nothing.

---

### What's coming next?

Take [ROADMAP.md](../ROADMAP.md) with a grain of salt — as of this writing it still lists RNN/LSTM layers, transformer/attention layers, and the web playground as *future* work, but all three already exist in the current codebase (this is a known staleness in that file, not a hint about priority). The genuinely still-open items visible in the roadmap are multi-GPU training (`DataParallel`/`DistributedDataParallel`, gradient accumulation) and a community model registry (`neuva publish`, pulling shared architectures, a benchmark leaderboard) — but treat both as aspirational rather than scheduled. The most reliable source for "what's actually new" is [CHANGELOG.md](../CHANGELOG.md) after the fact, not the roadmap in advance.
