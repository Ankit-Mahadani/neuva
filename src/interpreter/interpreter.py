from typing import Any, Optional
from parser.ast_nodes import (
    Program, LetStatement, PrintStatement, ExprStatement,
    NumberLiteral, FloatLiteral, StringLiteral, BoolLiteral,
    VarExpr, BinaryExpr, CallExpr,
    IfStatement, ForStatement, WhileStatement, FnStatement, ReturnStatement,
)


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
    def __init__(self, message: str, line: int = None, col: int = None):
        self.line = line
        self.col = col
        loc = f" (line {line}, col {col})" if line is not None else ""
        super().__init__(f"RuntimeError{loc}: {message}")


class Environment:
    def __init__(self, parent: Optional["Environment"] = None):
        self._store: dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self._store:
            return self._store[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise RuntimeError_(f"Undefined variable '{name}'")

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
        self.env.set(node.name, value)

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
        return self.env.get(node.name)

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
