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
from typing import Any, Optional, Union

from neuva.backend.torch_backend import LOSS_NAMES
from neuva.interpreter.interpreter import find_similar_name
from neuva.parser import NeuvaParser
from neuva.parser.ast_nodes import Program

_STDLIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stdlib")

_BUILTIN_NAMES = {
    "range",
    "len",
    "load",
    "accuracy",
    "predict",
    "predict_proba",
    "normalize",
    "shuffle",
    "export_onnx",
    "mse",
    "mae",
    "crossentropy",
    "binary_crossentropy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "table",
    "plot",
    "upper",
    "lower",
    "split",
    "join",
    "strip",
    "replace",
    "abs",
    "sqrt",
    "pow",
    "log",
    "exp",
    "round",
    "min",
    "max",
    "sum",
    "mean",
}

# Best-effort literal-AST-node -> declared-type-annotation mapping, used only for the
# best-effort return-type check below (Neuva has no real static type system).
_LITERAL_TYPE_MAP = {
    "NumberLiteral": "type_int",
    "FloatLiteral": "type_float",
    "StringLiteral": "type_string",
    "BoolLiteral": "type_bool",
}

# AST node types that are obviously never a list/dict at parse time, so indexing them
# is a clear-cut error rather than a type-tracking guess.
_NON_INDEXABLE_LITERALS = {"NumberLiteral", "FloatLiteral", "BoolLiteral"}


class TypeCheckError(Exception):
    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        col: Optional[int] = None,
        hint: Optional[str] = None,
    ) -> None:
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
    def __init__(self) -> None:
        self.errors: list[TypeCheckError] = []
        self.warnings: list[TypeCheckError] = []
        self.model_names: set = set()
        self.fn_names: set = set()
        self.all_top_names: set = set(_BUILTIN_NAMES)
        # Names `let`-bound to an unambiguous non-model literal (e.g. `let x = 5`) — used
        # to catch obvious model-name typos/misuse without flagging a `let`-bound name
        # whose value comes from a call (e.g. `load(...)`) that could return a model.
        self.non_model_names: set = set()
        self._stdlib_cache: dict = {}

    def check(self, program: Program) -> list[TypeCheckError]:
        """Walk `program` (a Program AST node) and return a list of TypeCheckError.

        Warnings (non-fatal — e.g. a for-loop variable shadowing an outer name) are
        collected separately in `self.warnings` after this call returns.
        """
        self.errors = []
        self.warnings = []
        self.model_names = set()
        self.fn_names = set()
        self.all_top_names = set(_BUILTIN_NAMES)
        self.non_model_names = set()
        self._stdlib_cache = {}

        self._collect_top_names(program.body)
        self._check_block(program.body, set(_BUILTIN_NAMES), in_function=False)
        return self.errors

    # ── module resolution ─────────────────────────────────────────────────

    def _load_stdlib_module(self, name: str) -> Optional[Program]:
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

    def _collect_top_names(self, stmts: list) -> None:
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

    def _error(self, message: str, node: Any, hint: Optional[str] = None) -> None:
        self.errors.append(
            TypeCheckError(
                message, getattr(node, "line", None), getattr(node, "col", None), hint
            )
        )

    def _warn(self, message: str, node: Any, hint: Optional[str] = None) -> None:
        self.warnings.append(
            TypeCheckError(
                message, getattr(node, "line", None), getattr(node, "col", None), hint
            )
        )

    @staticmethod
    def _literal_number(expr: Any) -> Optional[Union[int, float]]:
        """Return expr.value if it's a literal number, else None (e.g. a variable
        reference — can't be checked statically without real type inference)."""
        t = type(expr).__name__
        if t in ("NumberLiteral", "FloatLiteral"):
            return expr.value
        return None

    def _check_block(self, stmts: list, scope: set, in_function: bool) -> None:
        for stmt in stmts:
            self._check_stmt(stmt, scope, in_function)

    def _check_stmt(self, stmt: Any, scope: set, in_function: bool) -> None:
        t = type(stmt).__name__

        if t == "LetStatement":
            self._check_expr(stmt.value, scope, in_function)
            scope.update(stmt.names)
            value_type = type(stmt.value).__name__
            if len(stmt.names) == 1:
                if value_type in (
                    "NumberLiteral",
                    "FloatLiteral",
                    "StringLiteral",
                    "BoolLiteral",
                    "ListLiteral",
                    "DictLiteral",
                ):
                    self.non_model_names.add(stmt.names[0])
                else:
                    self.non_model_names.discard(stmt.names[0])

        elif t == "ModelStatement":
            scope.add(stmt.name)
            self._check_model_layers(stmt)

        elif t == "TrainStatement":
            self._check_model_name(stmt.model, scope, stmt, in_function)
            self._check_name_exists(stmt.data, scope, stmt, in_function)
            if stmt.epochs is not None and stmt.epochs <= 0:
                self._error(
                    f"train epochs must be a positive integer, got {stmt.epochs}", stmt
                )
            for opt in stmt.options:
                if opt.key == "loss":
                    loss_name = str(opt.value)
                    if loss_name not in LOSS_NAMES and loss_name not in self.fn_names:
                        self._error(
                            f"'{loss_name}' is not a valid loss — expected a built-in "
                            f"({', '.join(sorted(LOSS_NAMES))}) or a defined function",
                            stmt,
                        )
                elif opt.key == "lr":
                    self._check_expr(opt.value, scope, in_function)
                    val = self._literal_number(opt.value)
                    if val is not None and val <= 0:
                        self._error(
                            f"train lr must be a positive number, got {val}", stmt
                        )
                elif opt.key == "early_stop":
                    self._check_expr(opt.value, scope, in_function)
                    val = self._literal_number(opt.value)
                    if val is not None and (not isinstance(val, int) or val <= 0):
                        self._error(
                            f"train early_stop must be a positive integer, got {val}",
                            stmt,
                        )
                elif opt.key == "lr_warmup":
                    self._check_expr(opt.value, scope, in_function)
                    val = self._literal_number(opt.value)
                    if val is not None and (not isinstance(val, int) or val <= 0):
                        self._error(
                            f"train lr_warmup must be a positive integer, got {val}",
                            stmt,
                        )

        elif t == "PredictStatement":
            self._check_model_name(stmt.model, scope, stmt, in_function)
            self._check_name_exists(stmt.data, scope, stmt, in_function)

        elif t == "SaveStatement":
            self._check_model_name(stmt.model, scope, stmt, in_function)
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
            if stmt.var in scope:
                self._warn(
                    f"for-loop variable '{stmt.var}' shadows an existing variable in the enclosing scope",
                    stmt,
                )
            scope.add(stmt.var)
            self._check_block(stmt.body, scope, in_function)

        elif t == "WhileStatement":
            self._check_expr(stmt.condition, scope, in_function)
            self._check_block(stmt.body, scope, in_function)

        elif t == "MatchStatement":
            self._check_expr(stmt.subject, scope, in_function)
            for value_node, case_stmt in stmt.cases:
                self._check_expr(value_node, scope, in_function)
                if case_stmt is not None:
                    self._check_stmt(case_stmt, scope, in_function)
            if stmt.default is not None:
                self._check_stmt(stmt.default, scope, in_function)

        elif t == "AssignStatement":
            self._check_name_exists(stmt.name, scope, stmt, in_function)
            self._check_expr(stmt.value, scope, in_function)

        elif t == "AugAssignStatement":
            self._check_name_exists(stmt.name, scope, stmt, in_function)
            self._check_expr(stmt.value, scope, in_function)

        elif t == "FnStatement":
            scope.add(stmt.name)
            fn_scope = set(self.all_top_names) | set(scope)
            for p in stmt.params:
                fn_scope.add(p.name)
            self._check_block(stmt.body, fn_scope, True)
            self._check_return_type(stmt)

        elif t == "ImportStatement":
            module_ast = self._load_stdlib_module(stmt.module)
            if module_ast is None:
                self._error(f"stdlib module '{stmt.module}' not found", stmt)
            else:
                self._check_block(module_ast.body, scope, in_function)

        # other/unhandled statement types are skipped silently

    def _find_returns(self, body: list) -> list[Any]:
        """Recursively collect every ReturnStatement reachable from `body`, including
        through if/elif/else/for/while/match — Neuva has no block scoping, so a `return`
        nested in any of these still returns from the enclosing function."""
        found = []
        for stmt in body:
            t = type(stmt).__name__
            if t == "ReturnStatement":
                found.append(stmt)
            elif t == "IfStatement":
                found.extend(self._find_returns(stmt.then_body))
                for _, branch_body in stmt.elif_branches:
                    found.extend(self._find_returns(branch_body))
                found.extend(self._find_returns(stmt.else_branch))
            elif t in ("ForStatement", "WhileStatement"):
                found.extend(self._find_returns(stmt.body))
            elif t == "MatchStatement":
                for _, case_stmt in stmt.cases:
                    if case_stmt is not None:
                        found.extend(self._find_returns([case_stmt]))
                if stmt.default is not None:
                    found.extend(self._find_returns([stmt.default]))
        return found

    def _check_return_type(self, fn_stmt: Any) -> None:
        """Best-effort check: for `return <literal>` statements, does the literal's type
        match the function's declared return type? Non-literal return values (variables,
        calls, arithmetic, ...) can't be checked without real type inference, so they're
        skipped rather than guessed at."""
        if not fn_stmt.return_type:
            return
        for ret in self._find_returns(fn_stmt.body):
            if ret.value is None:
                continue
            value_type = type(ret.value).__name__
            declared = _LITERAL_TYPE_MAP.get(value_type)
            if declared is not None and declared != fn_stmt.return_type:
                self._error(
                    f"function '{fn_stmt.name}' declares return type "
                    f"'{fn_stmt.return_type.replace('type_', '')}' but returns a "
                    f"'{declared.replace('type_', '')}' value",
                    ret,
                )

    def _check_model_name(
        self, name: str, scope: set, node: Any, in_function: bool
    ) -> None:
        """Best-effort: a `model` block name is always fine, and a name unambiguously
        `let`-bound to a non-model literal (e.g. `let x = 5`) is always wrong — but a
        name bound to a call result (e.g. `load(...)`) could hold a model at runtime,
        so it's only checked for existing at all, not for being a model specifically."""
        if name in self.model_names:
            return
        if name in self.non_model_names:
            self._error(
                f"'{name}' is used as a model but is not a model definition", node
            )
            return
        known = (
            name in scope
            or name in self.fn_names
            or (in_function and name in self.all_top_names)
        )
        if not known:
            hint = find_similar_name(name, list(self.model_names))
            self._error(f"undefined model '{name}'", node, hint=hint)

    def _check_name_exists(
        self, name: str, scope: set, node: Any, in_function: bool
    ) -> None:
        if name in scope or name in self.model_names or name in self.fn_names:
            return
        if in_function and name in self.all_top_names:
            return
        hint = find_similar_name(
            name, list(scope | self.all_top_names | self.model_names | self.fn_names)
        )
        self._error(f"undefined variable '{name}'", node, hint=hint)

    def _check_model_layers(self, model_stmt: Any) -> None:
        """Dense-to-dense adjacency check: layer N's output size must match layer N+1's input size."""
        layers = model_stmt.layers
        for i in range(len(layers) - 1):
            cur, nxt = layers[i], layers[i + 1]
            if (
                cur.name == "dense"
                and nxt.name == "dense"
                and len(cur.args) >= 2
                and len(nxt.args) >= 1
            ):
                if cur.args[1] != nxt.args[0]:
                    self._error(
                        f"layer dimension mismatch in model '{model_stmt.name}': "
                        f"dense output size {cur.args[1]} does not match the next dense layer's input size {nxt.args[0]}",
                        nxt,
                    )

    def _check_expr(self, expr: Any, scope: set, in_function: bool) -> None:
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
        elif t == "DictLiteral":
            for k in expr.keys:
                self._check_expr(k, scope, in_function)
            for v in expr.values:
                self._check_expr(v, scope, in_function)
        elif t == "IndexExpr":
            obj_type = type(expr.obj).__name__
            if obj_type in _NON_INDEXABLE_LITERALS:
                self._error(
                    f"cannot index a '{_LITERAL_TYPE_MAP.get(obj_type, obj_type).replace('type_', '')}' "
                    "value — indexing requires a list or dict",
                    expr,
                )
            self._check_expr(expr.obj, scope, in_function)
            self._check_expr(expr.index, scope, in_function)
        # literals: nothing to check
