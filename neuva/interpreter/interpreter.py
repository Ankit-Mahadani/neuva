import difflib
import os
import re
from typing import Any, Optional
from neuva.backend.torch_backend import (
    NeuvaModel, NeuvaTrainer, NeuvaDataset, evaluate, evaluate_accuracy, save_model, load_model,
    mse_fn, mae_fn, crossentropy_fn, binary_crossentropy_fn, LOSS_NAMES,
    precision, recall, f1_score, confusion_matrix,
)
from neuva.backend.data_loader import DataSet
from neuva.parser import NeuvaParser
from neuva.parser.ast_nodes import (
    Program, LetStatement, PrintStatement, ExprStatement,
    NumberLiteral, FloatLiteral, StringLiteral, BoolLiteral,
    VarExpr, BinaryExpr, CallExpr, MethodCallExpr,
    IfStatement, ForStatement, WhileStatement, FnStatement, ReturnStatement,
    ModelStatement, TrainStatement, SaveStatement, PredictStatement,
    ListLiteral, IndexExpr, ImportStatement,
)

_STDLIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stdlib")


class TableView:
    """Marker returned by the `table()` builtin so `print` can render it as an ASCII table."""
    def __init__(self, data: Any):
        self.data = data


class PlotView:
    """Marker returned by the `plot()` builtin so `print` can render it as an ASCII line chart."""
    def __init__(self, values: Any):
        self.values = values


def _fmt_cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_table(dataset: Any, max_rows: int = 10) -> str:
    X = getattr(dataset, "X", None)
    if X is None:
        rows = dataset if isinstance(dataset, list) else []
        if not rows:
            return "(empty table)"
        headers = [f"col{i}" for i in range(len(rows[0]))]
        data_rows = [[_fmt_cell(v) for v in row] for row in rows[:max_rows]]
        total = len(rows)
    else:
        y = getattr(dataset, "y", None)
        columns = list(getattr(dataset, "columns", None) or [])
        n_features = X.shape[1] if X.dim() > 1 else 1
        headers = columns[:n_features] if len(columns) >= n_features else [f"col{i}" for i in range(n_features)]
        if y is not None:
            headers = headers + [columns[-1] if len(columns) > n_features else "target"]
        total = len(X)
        data_rows = []
        for i in range(min(max_rows, total)):
            row = [_fmt_cell(v) for v in X[i].tolist()] if X.dim() > 1 else [_fmt_cell(X[i].item())]
            if y is not None:
                row.append(_fmt_cell(y[i].item()))
            data_rows.append(row)

    widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def hline(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def fmt_row(cells):
        return "│" + "│".join(f" {c.rjust(widths[i])} " for i, c in enumerate(cells)) + "│"

    lines = [hline("┌", "┬", "┐"), fmt_row(headers), hline("├", "┼", "┤")]
    lines += [fmt_row(r) for r in data_rows]
    lines.append(hline("└", "┴", "┘"))
    if total > max_rows:
        lines.append(f"... {total - max_rows} more rows")
    return "\n".join(lines)


def render_plot(values: Any, height: int = 10) -> str:
    values = [float(v) for v in values]
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    width = len(values)
    grid = [[" "] * width for _ in range(height)]
    for x, v in enumerate(values):
        norm = (v - lo) / rng
        row = height - 1 - round(norm * (height - 1))
        grid[row][x] = "*"
    lines = ["".join(r) for r in grid]
    lines.append("─" * width)
    lines.append(f"min={lo:.4f}  max={hi:.4f}  n={len(values)}")
    return "\n".join(lines)


def find_similar_name(name: str, available_names: list) -> Optional[str]:
    matches = difflib.get_close_matches(name, available_names, n=1, cutoff=0.6)
    if matches:
        return f"Did you mean '{matches[0]}'?"
    return None


class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class NeuvaFunction:
    def __init__(self, name: str, params: list, body: list, closure: "Environment"):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure

    def __repr__(self) -> str:
        return f"<fn {self.name}>"


class RuntimeError_(Exception):
    def __init__(self, message: str, line: int = None, col: int = None, hint: str = None):
        self.raw_message = message
        self.line = line
        self.col = col
        self.hint = hint
        super().__init__(message)

    def __str__(self) -> str:
        if self.line is not None and self.col is not None:
            return f"[Line {self.line}:{self.col}] Error: {self.raw_message}"
        if self.line is not None:
            return f"[Line {self.line}] Error: {self.raw_message}"
        return f"Error: {self.raw_message}"


class Environment:
    def __init__(self, parent: Optional["Environment"] = None):
        self._store: dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str, line: int = None, col: int = None) -> Any:
        env = self
        while env is not None:
            if name in env._store:
                return env._store[name]
            env = env.parent
        hint = find_similar_name(name, self._all_names())
        raise RuntimeError_(f"undefined variable '{name}'", line=line, col=col, hint=hint)

    def _all_names(self) -> list:
        names = list(self._store.keys())
        if self.parent is not None:
            names.extend(self.parent._all_names())
        return names

    def set(self, name: str, value: Any) -> None:
        self._store[name] = value


_OPS = {
    "+":  lambda a, b: a + b,
    "-":  lambda a, b: a - b,
    "*":  lambda a, b: a * b,
    "/":  lambda a, b: a / b,
    "%":  lambda a, b: a % b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
}


class NeuvaInterpreter:
    def __init__(self):
        self.env = Environment()
        def _load(path):
            path = str(path)
            if path.endswith(".nva"):
                return load_model(path)
            try:
                return DataSet(path=path)
            except FileNotFoundError as exc:
                raise RuntimeError_(str(exc))

        self.env.set("range", range)
        self.env.set("len", len)
        self.env.set("load", _load)
        self.env.set("accuracy", lambda model, data: evaluate_accuracy(model, data))
        self.env.set("predict",  lambda model, data: "predictions")
        self.env.set("normalize", lambda: DataSet())
        self.env.set("shuffle",   lambda: DataSet())

        # Loss functions, callable directly (e.g. inside a custom `fn` loss)
        self.env.set("mse", mse_fn)
        self.env.set("mae", mae_fn)
        self.env.set("crossentropy", crossentropy_fn)
        self.env.set("binary_crossentropy", binary_crossentropy_fn)

        # Evaluation metrics
        self.env.set("precision", lambda model, data: precision(model, data))
        self.env.set("recall", lambda model, data: recall(model, data))
        self.env.set("f1_score", lambda model, data: f1_score(model, data))
        self.env.set("confusion_matrix", lambda model, data: confusion_matrix(model, data))

        # print table(...) / print plot(...)
        self.env.set("table", lambda d: TableView(d))
        self.env.set("plot", lambda v: PlotView(v))

    # ── dispatch ───────────────────────────────────────────────────────────

    def visit(self, node) -> Any:
        method = f"visit_{type(node).__name__}"
        handler = getattr(self, method, self._unhandled)
        return handler(node)

    def _unhandled(self, node) -> None:
        pass  # silently skip unimplemented node types

    def evaluate(self, node) -> Any:
        method = f"eval_{type(node).__name__}"
        handler = getattr(self, method, None)
        if handler is None:
            raise RuntimeError_(
                f"Cannot evaluate node type '{type(node).__name__}'",
                getattr(node, "line", None),
                getattr(node, "col", None),
            )
        return handler(node)

    # ── statement visitors ─────────────────────────────────────────────────

    def visit_Program(self, node: Program) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_LetStatement(self, node: LetStatement) -> None:
        value = self.evaluate(node.value)
        if len(node.names) == 1:
            self.env.set(node.names[0], value)
        else:
            unpacked = list(value)
            if len(unpacked) != len(node.names):
                raise RuntimeError_(
                    f"Cannot unpack {len(unpacked)} values into {len(node.names)} variables"
                )
            for name, val in zip(node.names, unpacked):
                self.env.set(name, val)

    def visit_ModelStatement(self, node: ModelStatement) -> None:
        self.env.set(node.name, node)

    def visit_TrainStatement(self, node: TrainStatement) -> None:
        model_node = self.env.get(node.model)
        data_obj = self.env.get(node.data)

        lr = 0.001
        loss_name = "mse"
        lr_schedule = "none"
        early_stop = None
        for opt in node.options:
            if opt.key == "lr":
                lr = float(opt.value)
            elif opt.key == "loss":
                loss_name = str(opt.value)
            elif opt.key == "lr_schedule":
                lr_schedule = str(opt.value)
            elif opt.key == "early_stop":
                early_stop = int(opt.value)

        if loss_name in LOSS_NAMES:
            loss_fn = loss_name
        else:
            fn_obj = self.env.get(loss_name, line=getattr(node, "line", None), col=getattr(node, "col", None))
            if not isinstance(fn_obj, NeuvaFunction):
                raise RuntimeError_(
                    f"'{loss_name}' is not a valid loss — expected a built-in "
                    f"({', '.join(sorted(LOSS_NAMES))}) or a function name",
                    getattr(node, "line", None), getattr(node, "col", None),
                )
            loss_fn = lambda pred, target, _fn=fn_obj: self._call_function(_fn, [pred, target])

        in_size = model_node.layers[0].args[0] if model_node.layers else (
            model_node.outputs[0].layer.args[0] if model_node.outputs else 1
        )
        neuva_model = NeuvaModel(model_node.layers, outputs=model_node.outputs)
        neuva_model.model_name = node.model
        dataset = NeuvaDataset(data_obj, in_size=in_size)
        NeuvaTrainer().train(
            neuva_model, dataset, node.epochs, lr=lr, loss_fn=loss_fn,
            lr_schedule=lr_schedule, early_stop=early_stop,
        )
        self.env.set(node.model, neuva_model)

    def visit_SaveStatement(self, node: SaveStatement) -> None:
        model = self.env.get(node.model)
        if not isinstance(model, NeuvaModel):
            raise RuntimeError_(f"'{node.model}' is not a trained model and cannot be saved")
        path = str(self.evaluate(node.path))
        save_model(model, path)

    def visit_PredictStatement(self, node: PredictStatement) -> None:
        pass

    def visit_ImportStatement(self, node: ImportStatement) -> None:
        module_path = os.path.join(_STDLIB_DIR, f"{node.module}.nva")
        if not os.path.isfile(module_path):
            raise RuntimeError_(
                f"stdlib module '{node.module}' not found",
                getattr(node, "line", None), getattr(node, "col", None),
            )
        tree = NeuvaParser().parse_file(module_path)
        for stmt in tree.body:
            self.visit(stmt)

    def visit_PrintStatement(self, node: PrintStatement) -> None:
        values = [self.evaluate(e) for e in node.exprs]
        if len(values) == 1:
            val = values[0]
            if isinstance(val, ModelStatement):
                print(self._model_stmt_summary(val))
                return
            if isinstance(val, NeuvaModel):
                print(self._neuva_model_summary(val))
                return
            if isinstance(val, TableView):
                print(render_table(val.data))
                return
            if isinstance(val, PlotView):
                print(render_plot(val.values))
                return
        print(*values)

    @staticmethod
    def _boxify(lines: list) -> str:
        width = max(len(l) for l in lines)
        top = "┌" + "─" * (width + 2) + "┐"
        bottom = "└" + "─" * (width + 2) + "┘"
        body = [f"│ {l.ljust(width)} │" for l in lines]
        return "\n".join([top] + body + [bottom])

    def _model_stmt_summary(self, model: ModelStatement) -> str:
        lines = [f"Model: {model.name}"]
        total = 0
        for layer in model.layers:
            if layer.name in ("rnn", "lstm") and len(layer.args) >= 2:
                num_layers = layer.args[2] if len(layer.args) >= 3 else 1
                lines.append(f"  {layer.name}({layer.args[0]} -> {layer.args[1]}, num_layers={num_layers})")
            elif len(layer.args) >= 3:
                lines.append(f"  {layer.name}({layer.args[0]} -> {layer.args[1]}, {layer.args[2]})")
                try:
                    total += int(float(str(layer.args[0]))) * int(float(str(layer.args[1]))) + int(float(str(layer.args[1])))
                except (ValueError, TypeError):
                    pass
            elif len(layer.args) == 1:
                lines.append(f"  {layer.name}({layer.args[0]})")
            else:
                lines.append(f"  {layer.name}")
        for out in model.outputs:
            a = out.layer.args
            lines.append(f"  output {out.output_name}: {out.layer.name}({a[0]} -> {a[1]}, {a[2]})")
        lines.append(f"Total parameters: {total}")
        return self._boxify(lines)

    def _neuva_model_summary(self, model: NeuvaModel) -> str:
        name = getattr(model, "model_name", "(trained)")
        lines = [f"Model: {name}"]
        for i, layer in enumerate(model.linears):
            act = model.activation_names[i] if i < len(model.activation_names) else "linear"
            ltype = type(layer).__name__
            if ltype == "Linear":
                lines.append(f"  dense({layer.in_features} -> {layer.out_features}, {act})")
            elif ltype == "Dropout":
                lines.append(f"  dropout({layer.p})")
            elif ltype == "MaxPool2d":
                lines.append(f"  pool({layer.kernel_size})")
            elif ltype == "Conv2d":
                lines.append(f"  conv({layer.in_channels} -> {layer.out_channels}, {layer.kernel_size})")
            elif ltype == "Flatten":
                lines.append("  flatten")
            elif ltype == "_RNNWrapper":
                lines.append(f"  {layer.kind}({layer.input_size} -> {layer.hidden_size}, num_layers={layer.num_layers})")
            else:
                lines.append(f"  {ltype.lower()}")
        for oname in getattr(model, "output_names", []):
            head = model.output_heads[oname]
            act = model.output_activation_names.get(oname, "linear")
            lines.append(f"  output {oname}: dense({head.in_features} -> {head.out_features}, {act})")
        total = sum(p.numel() for p in model.parameters())
        lines.append(f"Total parameters: {total}")
        return self._boxify(lines)

    def visit_ExprStatement(self, node: ExprStatement) -> Any:
        return self.evaluate(node.expr)

    def visit_IfStatement(self, node: IfStatement) -> None:
        if self.evaluate(node.condition):
            for stmt in node.then_body:
                self.visit(stmt)
            return
        for elif_cond, elif_body in node.elif_branches:
            if self.evaluate(elif_cond):
                for stmt in elif_body:
                    self.visit(stmt)
                return
        for stmt in node.else_branch:
            self.visit(stmt)

    def visit_ForStatement(self, node: ForStatement) -> None:
        iterable = self.evaluate(node.iterable)
        for val in iterable:
            self.env.set(node.var, val)
            for stmt in node.body:
                self.visit(stmt)

    def visit_WhileStatement(self, node: WhileStatement) -> None:
        while self.evaluate(node.condition):
            for stmt in node.body:
                self.visit(stmt)

    def visit_FnStatement(self, node: FnStatement) -> None:
        self.env.set(node.name, NeuvaFunction(
            name=node.name,
            params=node.params,
            body=node.body,
            closure=self.env,
        ))

    def visit_ReturnStatement(self, node: ReturnStatement) -> None:
        value = self.evaluate(node.value) if node.value is not None else None
        raise ReturnSignal(value)

    # ── expression evaluators ──────────────────────────────────────────────

    def eval_NumberLiteral(self, node: NumberLiteral) -> int:
        return node.value

    def eval_FloatLiteral(self, node: FloatLiteral) -> float:
        return node.value

    def eval_StringLiteral(self, node: StringLiteral) -> str:
        s = node.value
        if '{' not in s:
            return s
        def _sub(m):
            try:
                return str(self.env.get(m.group(1)))
            except RuntimeError_:
                return m.group(0)
        return re.sub(r'\{(\w+)\}', _sub, s)

    def eval_ListLiteral(self, node: ListLiteral) -> list:
        return [self.evaluate(e) for e in node.elements]

    def eval_IndexExpr(self, node: IndexExpr) -> Any:
        obj = self.evaluate(node.obj)
        idx = self.evaluate(node.index)
        try:
            return obj[idx]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError_(
                str(exc),
                getattr(node, "line", None),
                getattr(node, "col", None),
            )

    def eval_BoolLiteral(self, node: BoolLiteral) -> bool:
        return node.value

    def eval_VarExpr(self, node: VarExpr) -> Any:
        return self.env.get(
            node.name,
            line=getattr(node, "line", None),
            col=getattr(node, "col", None),
        )

    def eval_BinaryExpr(self, node: BinaryExpr) -> Any:
        op = node.op
        if op == "neg":
            return -self.evaluate(node.right)
        if op == "not":
            return not self.evaluate(node.right)
        if op == "and":
            left = self.evaluate(node.left)
            return left and self.evaluate(node.right)
        if op == "or":
            left = self.evaluate(node.left)
            return left or self.evaluate(node.right)

        fn = _OPS.get(op)
        if fn is None:
            raise RuntimeError_(
                f"Unknown operator '{op}'",
                getattr(node, "line", None),
                getattr(node, "col", None),
            )

        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        if op == "/" and right == 0:
            raise RuntimeError_(
                "Division by zero",
                getattr(node, "line", None),
                getattr(node, "col", None),
            )

        if op == "+" and (isinstance(left, str) or isinstance(right, str)):
            return str(left) + str(right)

        return fn(left, right)

    def _call_function(self, fn: "NeuvaFunction", args: list, line: int = None, col: int = None) -> Any:
        if len(args) != len(fn.params):
            raise RuntimeError_(f"'{fn.name}' expects {len(fn.params)} args, got {len(args)}", line, col)
        local_env = Environment(parent=fn.closure)
        for param, val in zip(fn.params, args):
            local_env.set(param.name, val)
        saved = self.env
        self.env = local_env
        try:
            for stmt in fn.body:
                self.visit(stmt)
            return None
        except ReturnSignal as sig:
            return sig.value
        finally:
            self.env = saved

    def eval_CallExpr(self, node: CallExpr) -> Any:
        callee = self.evaluate(node.callee)
        args = [self.evaluate(a) for a in node.args]

        if callable(callee) and not isinstance(callee, NeuvaFunction):
            return callee(*args)

        if isinstance(callee, NeuvaFunction):
            return self._call_function(callee, args, getattr(node, "line", None), getattr(node, "col", None))

        raise RuntimeError_(
            f"'{callee}' is not callable",
            getattr(node, "line", None),
            getattr(node, "col", None),
        )

    def eval_MethodCallExpr(self, node: MethodCallExpr) -> Any:
        obj = self.evaluate(node.obj)
        method = getattr(obj, node.method, None)
        if method is None:
            raise RuntimeError_(
                f"Object has no method '{node.method}'",
                getattr(node, "line", None),
                getattr(node, "col", None),
            )
        args = [self.evaluate(a) for a in node.args]
        return method(*args)
