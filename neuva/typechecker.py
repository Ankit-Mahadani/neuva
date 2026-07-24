"""Basic static checks that run between parsing and interpreting.

This is intentionally a "best-effort" pass, not a sound type system: Neuva has no
block scoping (an `if`/`for`/`while` body shares its enclosing scope, matching the
interpreter's single flat Environment per call frame), and function bodies run later
through a shared closure environment, so a function may legitimately reference a
global defined *after* the function itself. To avoid false positives, top-level code
(including nested if/for/while bodies) is checked in strict textual order, while
function bodies are checked against every name known to exist *anywhere* in the file.
"""
import os

from neuva.backend.torch_backend import LOSS_NAMES
from neuva.interpreter.interpreter import find_similar_name
from neuva.parser import NeuvaParser

_STDLIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stdlib")

_BUILTIN_NAMES = {
    "range", "len", "load", "accuracy", "predict", "normalize", "shuffle",
    "mse", "mae", "crossentropy", "binary_crossentropy",
    "precision", "recall", "f1_score", "confusion_matrix",
    "table", "plot",
}


class TypeCheckError(Exception):
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


class TypeChecker:
    def __init__(self):
        self.errors = []
        self.model_names = set()
        self.fn_names = set()
        self.all_top_names = set(_BUILTIN_NAMES)
        self._stdlib_cache = {}

    def check(self, program) -> list:
        """Walk `program` (a Program AST node) and return a list of TypeCheckError."""
        self.errors = []
        self.model_names = set()
        self.fn_names = set()
        self.all_top_names = set(_BUILTIN_NAMES)
        self._stdlib_cache = {}

        self._collect_top_names(program.body)
        self._check_block(program.body, set(_BUILTIN_NAMES), in_function=False)
        return self.errors

    # ── module resolution ─────────────────────────────────────────────────

    def _load_stdlib_module(self, name: str):
        if name in self._stdlib_cache:
            return self._stdlib_cache[name]
        path = os.path.join(_STDLIB_DIR, f"{name}.nva")
        module_ast = None
        if os.path.isfile(path):
            try:
                module_ast = NeuvaParser().parse_file(path)
            except Exception:
                module_ast = None
        self._stdlib_cache[name] = module_ast
        return module_ast

    # ── pre-pass: collect every name defined anywhere at top level ─────────

    def _collect_top_names(self, stmts) -> None:
        for stmt in stmts:
            t = type(stmt).__name__
            if t == "LetStatement":
                self.all_top_names.update(stmt.names)
            elif t == "ModelStatement":
                self.model_names.add(stmt.name)
                self.all_top_names.add(stmt.name)
            elif t == "FnStatement":
                self.fn_names.add(stmt.name)
                self.all_top_names.add(stmt.name)
            elif t == "ForStatement":
                self.all_top_names.add(stmt.var)
                self._collect_top_names(stmt.body)
            elif t == "IfStatement":
                self._collect_top_names(stmt.then_body)
                for _, body in stmt.elif_branches:
                    self._collect_top_names(body)
                self._collect_top_names(stmt.else_branch)
            elif t == "WhileStatement":
                self._collect_top_names(stmt.body)
            elif t == "ImportStatement":
                module_ast = self._load_stdlib_module(stmt.module)
                if module_ast is not None:
                    self._collect_top_names(module_ast.body)
            # FnStatement bodies are NOT recursed into here: their params/locals stay
            # local to that call frame, not exposed at top level.

    # ── main pass ────────────────────────────────────────────────────────

    def _error(self, message: str, node, hint: str = None) -> None:
        self.errors.append(TypeCheckError(message, getattr(node, "line", None), getattr(node, "col", None), hint))

    def _check_block(self, stmts, scope: set, in_function: bool) -> None:
        for stmt in stmts:
            self._check_stmt(stmt, scope, in_function)

    def _check_stmt(self, stmt, scope: set, in_function: bool) -> None:
        t = type(stmt).__name__

        if t == "LetStatement":
            self._check_expr(stmt.value, scope, in_function)
            scope.update(stmt.names)

        elif t == "ModelStatement":
            scope.add(stmt.name)
            self._check_model_layers(stmt)

        elif t == "TrainStatement":
            self._check_model_name(stmt.model, stmt)
            self._check_name_exists(stmt.data, scope, stmt, in_function)
            for opt in stmt.options:
                if opt.key == "loss":
                    loss_name = str(opt.value)
                    if loss_name not in LOSS_NAMES and loss_name not in self.fn_names:
                        self._error(
                            f"'{loss_name}' is not a valid loss — expected a built-in "
                            f"({', '.join(sorted(LOSS_NAMES))}) or a defined function",
                            stmt,
                        )

        elif t == "PredictStatement":
            self._check_model_name(stmt.model, stmt)
            self._check_name_exists(stmt.data, scope, stmt, in_function)

        elif t == "SaveStatement":
            self._check_model_name(stmt.model, stmt)
            self._check_expr(stmt.path, scope, in_function)

        elif t == "PrintStatement":
            for e in stmt.exprs:
                self._check_expr(e, scope, in_function)

        elif t == "ExprStatement":
            self._check_expr(stmt.expr, scope, in_function)

        elif t == "ReturnStatement":
            if stmt.value is not None:
                self._check_expr(stmt.value, scope, in_function)

        elif t == "IfStatement":
            self._check_expr(stmt.condition, scope, in_function)
            self._check_block(stmt.then_body, scope, in_function)
            for cond, body in stmt.elif_branches:
                self._check_expr(cond, scope, in_function)
                self._check_block(body, scope, in_function)
            self._check_block(stmt.else_branch, scope, in_function)

        elif t == "ForStatement":
            self._check_expr(stmt.iterable, scope, in_function)
            scope.add(stmt.var)
            self._check_block(stmt.body, scope, in_function)

        elif t == "WhileStatement":
            self._check_expr(stmt.condition, scope, in_function)
            self._check_block(stmt.body, scope, in_function)

        elif t == "FnStatement":
            scope.add(stmt.name)
            fn_scope = set(self.all_top_names) | set(scope)
            for p in stmt.params:
                fn_scope.add(p.name)
            self._check_block(stmt.body, fn_scope, True)

        elif t == "ImportStatement":
            module_ast = self._load_stdlib_module(stmt.module)
            if module_ast is None:
                self._error(f"stdlib module '{stmt.module}' not found", stmt)
            else:
                self._check_block(module_ast.body, scope, in_function)

        # other/unhandled statement types are skipped silently

    def _check_model_name(self, name: str, node) -> None:
        if name in self.model_names:
            return
        if name in self.all_top_names or name in self.fn_names:
            self._error(f"'{name}' is used as a model but is not a model definition", node)
        else:
            hint = find_similar_name(name, list(self.model_names))
            self._error(f"undefined model '{name}'", node, hint=hint)

    def _check_name_exists(self, name: str, scope: set, node, in_function: bool) -> None:
        if name in scope or name in self.model_names or name in self.fn_names:
            return
        if in_function and name in self.all_top_names:
            return
        hint = find_similar_name(name, list(scope | self.all_top_names | self.model_names | self.fn_names))
        self._error(f"undefined variable '{name}'", node, hint=hint)

    def _check_model_layers(self, model_stmt) -> None:
        """Dense-to-dense adjacency check: layer N's output size must match layer N+1's input size."""
        layers = model_stmt.layers
        for i in range(len(layers) - 1):
            cur, nxt = layers[i], layers[i + 1]
            if cur.name == "dense" and nxt.name == "dense" and len(cur.args) >= 2 and len(nxt.args) >= 1:
                if cur.args[1] != nxt.args[0]:
                    self._error(
                        f"layer dimension mismatch in model '{model_stmt.name}': "
                        f"dense output size {cur.args[1]} does not match the next dense layer's input size {nxt.args[0]}",
                        nxt,
                    )

    def _check_expr(self, expr, scope: set, in_function: bool) -> None:
        if expr is None:
            return
        t = type(expr).__name__

        if t == "VarExpr":
            self._check_name_exists(expr.name, scope, expr, in_function)
        elif t == "BinaryExpr":
            self._check_expr(expr.left, scope, in_function)
            self._check_expr(expr.right, scope, in_function)
        elif t == "CallExpr":
            self._check_expr(expr.callee, scope, in_function)
            for a in expr.args:
                self._check_expr(a, scope, in_function)
        elif t == "MethodCallExpr":
            self._check_expr(expr.obj, scope, in_function)
            for a in expr.args:
                self._check_expr(a, scope, in_function)
        elif t == "ListLiteral":
            for e in expr.elements:
                self._check_expr(e, scope, in_function)
        elif t == "IndexExpr":
            self._check_expr(expr.obj, scope, in_function)
            self._check_expr(expr.index, scope, in_function)
        # literals: nothing to check
