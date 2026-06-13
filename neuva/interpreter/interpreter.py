import difflib
from typing import Any, Optional
from neuva.backend.torch_backend import NeuvaModel, NeuvaTrainer, NeuvaDataset, evaluate, evaluate_accuracy, save_model, load_model
from neuva.backend.data_loader import DataSet
from neuva.parser.ast_nodes import (
    Program, LetStatement, PrintStatement, ExprStatement,
    NumberLiteral, FloatLiteral, StringLiteral, BoolLiteral,
    VarExpr, BinaryExpr, CallExpr, MethodCallExpr,
    IfStatement, ForStatement, WhileStatement, FnStatement, ReturnStatement,
    ModelStatement, TrainStatement, SaveStatement, PredictStatement,
)


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
        self.env.set("range", range)
        self.env.set("load",     lambda path: load_model(str(path)) if str(path).endswith(".nva") else DataSet(path=str(path)))
        self.env.set("accuracy", lambda model, data: evaluate_accuracy(model, data))
        self.env.set("predict",  lambda model, data: "predictions")
        self.env.set("normalize", lambda: DataSet())
        self.env.set("shuffle",   lambda: DataSet())

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
        loss_fn = "mse"
        for opt in node.options:
            if opt.key == "lr":
                lr = float(opt.value)
            elif opt.key == "loss":
                loss_fn = str(opt.value)

        in_size = model_node.layers[0].args[0] if model_node.layers else 1
        neuva_model = NeuvaModel(model_node.layers)
        dataset = NeuvaDataset(data_obj, in_size=in_size)
        NeuvaTrainer().train(neuva_model, dataset, node.epochs, lr=lr, loss_fn=loss_fn)
        self.env.set(node.model, neuva_model)

    def visit_SaveStatement(self, node: SaveStatement) -> None:
        model = self.env.get(node.model)
        if not isinstance(model, NeuvaModel):
            raise RuntimeError_(f"'{node.model}' is not a trained model and cannot be saved")
        path = str(self.evaluate(node.path))
        save_model(model, path)

    def visit_PredictStatement(self, node: PredictStatement) -> None:
        pass

    def visit_PrintStatement(self, node: PrintStatement) -> None:
        print(*[self.evaluate(e) for e in node.exprs])

    def visit_ExprStatement(self, node: ExprStatement) -> Any:
        return self.evaluate(node.expr)

    def visit_IfStatement(self, node: IfStatement) -> None:
        branch = node.then_body if self.evaluate(node.condition) else node.else_body
        for stmt in branch:
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
        return node.value

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

        return fn(left, right)

    def eval_CallExpr(self, node: CallExpr) -> Any:
        callee = self.evaluate(node.callee)
        args = [self.evaluate(a) for a in node.args]

        if callable(callee):
            return callee(*args)

        if isinstance(callee, NeuvaFunction):
            if len(args) != len(callee.params):
                raise RuntimeError_(
                    f"'{callee.name}' expects {len(callee.params)} args, got {len(args)}",
                    getattr(node, "line", None),
                    getattr(node, "col", None),
                )
            local_env = Environment(parent=callee.closure)
            for param, val in zip(callee.params, args):
                local_env.set(param.name, val)
            saved = self.env
            self.env = local_env
            try:
                for stmt in callee.body:
                    self.visit(stmt)
                return None
            except ReturnSignal as sig:
                return sig.value
            finally:
                self.env = saved

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
