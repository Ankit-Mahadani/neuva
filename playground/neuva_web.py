"""Neuva-in-the-browser: a self-contained lexer, parser, and interpreter for the
Neuva language, meant to run under Pyodide with no dependency on lark, torch,
or pandas. The ML backend (model/train/save/predict) is simulated with pure
Python so every `.nva` program that doesn't need real numerical results runs
unmodified.

Entry point: run_neuva(source) -> str (captured stdout, or a formatted error).
"""

import contextlib
import difflib
import io
import math
import random
import re

# ══════════════════════════════════════════════════════════════════════════
# Errors
# ══════════════════════════════════════════════════════════════════════════


class NeuvaError(Exception):
    def __init__(self, message, line=None, col=None, hint=None):
        self.raw_message = message
        self.line = line
        self.col = col
        self.hint = hint
        super().__init__(message)


class ParseError(NeuvaError):
    pass


class RuntimeError_(NeuvaError):
    pass


def format_error(exc, source):
    source_lines = source.splitlines()
    line = exc.line
    col = exc.col
    hint = exc.hint
    raw = exc.raw_message

    if line is not None and col is not None:
        header = f"[Line {line}:{col}] Error: {raw}"
    elif line is not None:
        header = f"[Line {line}] Error: {raw}"
    else:
        header = f"Error: {raw}"

    parts = [header]
    if line is not None and 1 <= line <= len(source_lines):
        indent = "           "
        parts.append(f"{indent}{source_lines[line - 1]}")
        if col is not None:
            parts.append(indent + " " * (col - 1) + "^")
    if hint:
        parts.append(f"           {hint}")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# Lexer
# ══════════════════════════════════════════════════════════════════════════

KEYWORDS = {
    "print": "PRINT", "let": "LET", "import": "IMPORT",
    "model": "MODEL", "layer": "LAYER", "output": "OUTPUT",
    "train": "TRAIN", "on": "ON", "for": "FOR", "in": "IN", "epochs": "EPOCHS",
    "lr_schedule": "LR_SCHEDULE", "early_stop": "EARLY_STOP", "lr": "LR", "loss": "LOSS",
    "save": "SAVE", "to": "TO", "predict": "PREDICT",
    "fn": "FN", "return": "RETURN",
    "if": "IF", "else": "ELSE", "while": "WHILE",
    "and": "AND", "or": "OR",
    "int": "KW_INT", "float": "KW_FLOAT", "bool": "KW_BOOL",
    "string": "KW_STRING", "tensor": "KW_TENSOR", "matrix": "KW_MATRIX",
    "true": "BOOL", "false": "BOOL",
}

_TWO_CHAR_OPS = {"->", "==", "!=", "<=", ">="}
_ONE_CHAR_OPS = set("+-*/%=<>!(){}[],:.")


def _describe_token(tok):
    if tok.type == "NEWLINE":
        return "end of line"
    if tok.type == "EOF":
        return "end of file"
    return f"'{tok.value}'"


class Token:
    __slots__ = ("type", "value", "line", "col")

    def __init__(self, type_, value, line, col):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r})"


class Lexer:
    def __init__(self, source):
        self.src = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.n = len(source)

    def _advance(self):
        ch = self.src[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _peek(self, offset=0):
        i = self.pos + offset
        return self.src[i] if i < self.n else ""

    def tokenize(self):
        tokens = []
        while self.pos < self.n:
            ch = self._peek()

            if ch in " \t\r":
                self._advance()
                continue

            if ch == "#":
                while self.pos < self.n and self._peek() != "\n":
                    self._advance()
                continue

            if ch == "\n":
                line, col = self.line, self.col
                self._advance()
                if tokens and tokens[-1].type != "NEWLINE":
                    tokens.append(Token("NEWLINE", "\n", line, col))
                continue

            if ch == '"':
                tokens.append(self._read_string())
                continue

            if ch.isdigit():
                tokens.append(self._read_number())
                continue

            if ch.isalpha() or ch == "_":
                tokens.append(self._read_name())
                continue

            two = ch + self._peek(1)
            if two in _TWO_CHAR_OPS:
                line, col = self.line, self.col
                self._advance()
                self._advance()
                tokens.append(Token(two, two, line, col))
                continue

            if ch in _ONE_CHAR_OPS:
                line, col = self.line, self.col
                self._advance()
                tokens.append(Token(ch, ch, line, col))
                continue

            line, col = self.line, self.col
            self._advance()
            raise ParseError(f"unexpected character '{ch}'", line, col)

        if tokens and tokens[-1].type != "NEWLINE":
            tokens.append(Token("NEWLINE", "\n", self.line, self.col))
        tokens.append(Token("EOF", "", self.line, self.col))
        return tokens

    def _read_string(self):
        line, col = self.line, self.col
        self._advance()  # opening quote
        chars = []
        while self.pos < self.n and self._peek() not in ('"', "\n"):
            chars.append(self._advance())
        if self._peek() != '"':
            raise ParseError("unterminated string literal", line, col)
        self._advance()  # closing quote
        return Token("STRING", "".join(chars), line, col)

    def _read_number(self):
        line, col = self.line, self.col
        chars = []
        while self.pos < self.n and self._peek().isdigit():
            chars.append(self._advance())
        is_float = False
        if self._peek() == "." and self._peek(1) != ".":
            is_float = True
            chars.append(self._advance())
            while self.pos < self.n and self._peek().isdigit():
                chars.append(self._advance())
        text = "".join(chars)
        return Token("FLOAT" if is_float else "INT", text, line, col)

    def _read_name(self):
        line, col = self.line, self.col
        chars = []
        while self.pos < self.n and (self._peek().isalnum() or self._peek() == "_"):
            chars.append(self._advance())
        text = "".join(chars)
        kind = KEYWORDS.get(text)
        if kind:
            return Token(kind, text, line, col)
        return Token("NAME", text, line, col)


# ══════════════════════════════════════════════════════════════════════════
# AST nodes
# ══════════════════════════════════════════════════════════════════════════


class Node:
    def __init__(self, line=None, col=None):
        self.line = line
        self.col = col


class Program(Node):
    def __init__(self, body):
        super().__init__()
        self.body = body


class NumberLiteral(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value


class FloatLiteral(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value


class StringLiteral(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value


class BoolLiteral(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value


class ListLiteral(Node):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements


class VarExpr(Node):
    def __init__(self, name, line=None, col=None):
        super().__init__(line, col)
        self.name = name


class BinaryExpr(Node):
    def __init__(self, op, left, right):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right


class IndexExpr(Node):
    def __init__(self, obj, index, line=None, col=None):
        super().__init__(line, col)
        self.obj = obj
        self.index = index


class CallExpr(Node):
    def __init__(self, callee, args, line=None, col=None):
        super().__init__(line, col)
        self.callee = callee
        self.args = args


class MethodCallExpr(Node):
    def __init__(self, obj, method, args, line=None, col=None):
        super().__init__(line, col)
        self.obj = obj
        self.method = method
        self.args = args


class LetStatement(Node):
    def __init__(self, names, type_ann, value, line=None, col=None):
        super().__init__(line, col)
        self.names = names
        self.type_ann = type_ann
        self.value = value


class PrintStatement(Node):
    def __init__(self, exprs):
        super().__init__()
        self.exprs = exprs


class LayerStatement(Node):
    def __init__(self, name, args):
        super().__init__()
        self.name = name
        self.args = args


class OutputLayerStatement(Node):
    def __init__(self, output_name, layer):
        super().__init__()
        self.output_name = output_name
        self.layer = layer


class ModelStatement(Node):
    def __init__(self, name, layers, outputs):
        super().__init__()
        self.name = name
        self.layers = layers
        self.outputs = outputs


class TrainOption(Node):
    def __init__(self, key, value):
        super().__init__()
        self.key = key
        self.value = value


class TrainStatement(Node):
    def __init__(self, model, data, epochs, options, line=None, col=None):
        super().__init__(line, col)
        self.model = model
        self.data = data
        self.epochs = epochs
        self.options = options


class SaveStatement(Node):
    def __init__(self, model, path, line=None, col=None):
        super().__init__(line, col)
        self.model = model
        self.path = path


class PredictStatement(Node):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data


class ImportStatement(Node):
    def __init__(self, module, line=None, col=None):
        super().__init__(line, col)
        self.module = module


class Parameter(Node):
    def __init__(self, name, type_ann):
        super().__init__()
        self.name = name
        self.type_ann = type_ann


class FnStatement(Node):
    def __init__(self, name, params, return_type, body):
        super().__init__()
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body


class ReturnStatement(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value


class IfStatement(Node):
    def __init__(self, condition, then_body, elif_branches, else_branch):
        super().__init__()
        self.condition = condition
        self.then_body = then_body
        self.elif_branches = elif_branches
        self.else_branch = else_branch


class ForStatement(Node):
    def __init__(self, var, iterable, body):
        super().__init__()
        self.var = var
        self.iterable = iterable
        self.body = body


class WhileStatement(Node):
    def __init__(self, condition, body):
        super().__init__()
        self.condition = condition
        self.body = body


class ExprStatement(Node):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr


# ══════════════════════════════════════════════════════════════════════════
# Parser (recursive descent)
# ══════════════════════════════════════════════════════════════════════════

_TYPE_KEYWORDS = {
    "KW_INT": "type_int", "KW_FLOAT": "type_float", "KW_BOOL": "type_bool",
    "KW_STRING": "type_string", "KW_TENSOR": "type_tensor", "KW_MATRIX": "type_matrix",
}
_COMPARISON_OPS = {"==", "!=", "<", "<=", ">", ">="}


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset=0):
        i = self.pos + offset
        if i >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[i]

    def check(self, type_):
        return self.peek().type == type_

    def advance(self):
        tok = self.peek()
        if tok.type != "EOF":
            self.pos += 1
        return tok

    def expect(self, type_):
        tok = self.peek()
        if tok.type != type_:
            raise ParseError(
                f"expected '{type_}' but found {_describe_token(tok)}", tok.line, tok.col
            )
        return self.advance()

    def skip_newlines(self):
        while self.check("NEWLINE"):
            self.advance()

    # ── program ──────────────────────────────────────────────────────────

    def parse_program(self):
        self.skip_newlines()
        body = []
        while not self.check("EOF"):
            body.append(self.parse_statement())
            self.skip_newlines()
        return Program(body)

    def parse_block(self):
        self.expect("{")
        self.skip_newlines()
        stmts = []
        while not self.check("}"):
            if self.check("EOF"):
                raise ParseError("unexpected end of file, expected '}'", self.peek().line, self.peek().col)
            stmts.append(self.parse_statement())
            self.skip_newlines()
        self.expect("}")
        return stmts

    # ── statements ───────────────────────────────────────────────────────

    def parse_statement(self):
        t = self.peek().type
        if t == "PRINT":
            return self.parse_print()
        if t == "LET":
            return self.parse_let()
        if t == "MODEL":
            return self.parse_model()
        if t == "TRAIN":
            return self.parse_train()
        if t == "SAVE":
            return self.parse_save()
        if t == "PREDICT":
            return self.parse_predict()
        if t == "FN":
            return self.parse_fn()
        if t == "RETURN":
            return self.parse_return()
        if t == "IF":
            return self.parse_if()
        if t == "FOR":
            return self.parse_for()
        if t == "WHILE":
            return self.parse_while()
        if t == "IMPORT":
            return self.parse_import()
        tok = self.peek()
        return ExprStatement(self.parse_expr())

    def parse_print(self):
        self.advance()
        exprs = [self.parse_expr()]
        while self.check(","):
            self.advance()
            exprs.append(self.parse_expr())
        return PrintStatement(exprs)

    def parse_type(self):
        tok = self.advance()
        if tok.type in _TYPE_KEYWORDS:
            return _TYPE_KEYWORDS[tok.type]
        raise ParseError(f"expected a type name, found '{tok.value}'", tok.line, tok.col)

    def parse_let(self):
        tok = self.advance()
        names = [self.expect("NAME").value]
        while self.check(","):
            self.advance()
            names.append(self.expect("NAME").value)
        type_ann = None
        if self.check(":"):
            self.advance()
            type_ann = self.parse_type()
        self.expect("=")
        value = self.parse_expr()
        return LetStatement(names, type_ann, value, tok.line, tok.col)

    def parse_number_literal(self):
        tok = self.peek()
        if tok.type == "INT":
            self.advance()
            return int(tok.value)
        if tok.type == "FLOAT":
            self.advance()
            return float(tok.value)
        raise ParseError(f"expected a number, found {_describe_token(tok)}", tok.line, tok.col)

    def parse_layer_arg(self):
        tok = self.peek()
        if tok.type == "NAME" or tok.type == "BOOL":
            self.advance()
            return [tok.value]
        num = self.parse_number_literal()
        if self.check("->"):
            self.advance()
            num2 = self.parse_number_literal()
            return [num, num2]
        return [num]

    def parse_layer_call(self):
        name = self.expect("NAME").value
        args = []
        if self.check("("):
            self.advance()
            if not self.check(")"):
                args.extend(self.parse_layer_arg())
                while self.check(","):
                    self.advance()
                    args.extend(self.parse_layer_arg())
            self.expect(")")
        return name, args

    def parse_model(self):
        self.advance()
        name = self.expect("NAME").value
        self.expect("{")
        self.skip_newlines()
        layers, outputs = [], []
        while not self.check("}"):
            if self.check("LAYER"):
                self.advance()
                lname, largs = self.parse_layer_call()
                layers.append(LayerStatement(lname, largs))
            elif self.check("OUTPUT"):
                self.advance()
                out_name = self.expect("NAME").value
                self.expect(":")
                lname, largs = self.parse_layer_call()
                outputs.append(OutputLayerStatement(out_name, LayerStatement(lname, largs)))
            else:
                tok = self.peek()
                raise ParseError(
                    f"expected 'layer' or 'output' inside model body, found {_describe_token(tok)}",
                    tok.line, tok.col,
                )
            self.skip_newlines()
        self.expect("}")
        return ModelStatement(name, layers, outputs)

    def parse_train_opt(self):
        tok = self.advance()
        self.expect("=")
        if tok.type == "LR":
            return TrainOption("lr", float(self.parse_number_literal()))
        if tok.type == "LOSS":
            return TrainOption("loss", self.expect("NAME").value)
        if tok.type == "LR_SCHEDULE":
            return TrainOption("lr_schedule", self.expect("NAME").value)
        if tok.type == "EARLY_STOP":
            return TrainOption("early_stop", int(self.parse_number_literal()))
        raise ParseError(f"unknown training option '{tok.value}'", tok.line, tok.col)

    def parse_train(self):
        tok = self.advance()
        model = self.expect("NAME").value
        self.skip_newlines()
        self.expect("ON")
        data = self.expect("NAME").value
        self.skip_newlines()
        self.expect("FOR")
        epochs = int(self.parse_number_literal())
        self.expect("EPOCHS")
        options = []
        while True:
            saved = self.pos
            self.skip_newlines()
            if self.check(","):
                self.advance()
                options.append(self.parse_train_opt())
                continue
            if self.peek().type in ("LR", "LOSS", "LR_SCHEDULE", "EARLY_STOP"):
                options.append(self.parse_train_opt())
                continue
            self.pos = saved
            break
        return TrainStatement(model, data, epochs, options, tok.line, tok.col)

    def parse_save(self):
        tok = self.advance()
        model = self.expect("NAME").value
        self.expect("TO")
        path = self.parse_expr()
        return SaveStatement(model, path, tok.line, tok.col)

    def parse_predict(self):
        self.advance()
        model = self.expect("NAME").value
        self.expect("ON")
        data = self.expect("NAME").value
        return PredictStatement(model, data)

    def parse_import(self):
        tok = self.advance()
        module = self.expect("NAME").value
        return ImportStatement(module, tok.line, tok.col)

    def parse_params(self):
        params = []
        if self.check(")"):
            return params
        params.append(self.parse_param())
        while self.check(","):
            self.advance()
            params.append(self.parse_param())
        return params

    def parse_param(self):
        name = self.expect("NAME").value
        type_ann = None
        if self.check(":"):
            self.advance()
            type_ann = self.parse_type()
        return Parameter(name, type_ann)

    def parse_fn(self):
        self.advance()
        name = self.expect("NAME").value
        self.expect("(")
        params = self.parse_params()
        self.expect(")")
        return_type = None
        if self.check("->"):
            self.advance()
            return_type = self.parse_type()
        body = self.parse_block()
        return FnStatement(name, params, return_type, body)

    def parse_return(self):
        self.advance()
        if self.peek().type in ("NEWLINE", "}", "EOF"):
            return ReturnStatement(None)
        return ReturnStatement(self.parse_expr())

    def parse_if(self):
        self.advance()
        cond = self.parse_expr()
        then_body = self.parse_block()
        elif_branches = []
        else_branch = []
        while True:
            saved = self.pos
            self.skip_newlines()
            if self.check("ELSE") and self.peek(1).type == "IF":
                self.advance()
                self.advance()
                ei_cond = self.parse_expr()
                ei_body = self.parse_block()
                elif_branches.append((ei_cond, ei_body))
                continue
            if self.check("ELSE"):
                self.advance()
                else_branch = self.parse_block()
                break
            self.pos = saved
            break
        return IfStatement(cond, then_body, elif_branches, else_branch)

    def parse_for(self):
        self.advance()
        var = self.expect("NAME").value
        self.expect("IN")
        iterable = self.parse_expr()
        body = self.parse_block()
        return ForStatement(var, iterable, body)

    def parse_while(self):
        self.advance()
        cond = self.parse_expr()
        body = self.parse_block()
        return WhileStatement(cond, body)

    # ── expressions ──────────────────────────────────────────────────────

    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.check("OR"):
            self.advance()
            right = self.parse_and()
            left = BinaryExpr("or", left, right)
        return left

    def parse_and(self):
        left = self.parse_comparison()
        while self.check("AND"):
            self.advance()
            right = self.parse_comparison()
            left = BinaryExpr("and", left, right)
        return left

    def parse_comparison(self):
        left = self.parse_sum()
        if self.peek().type in _COMPARISON_OPS:
            op = self.advance().type
            right = self.parse_sum()
            return BinaryExpr(op, left, right)
        return left

    def parse_sum(self):
        left = self.parse_product()
        while self.peek().type in ("+", "-"):
            op = self.advance().type
            right = self.parse_product()
            left = BinaryExpr(op, left, right)
        return left

    def parse_product(self):
        left = self.parse_unary()
        while self.peek().type in ("*", "/", "%"):
            op = self.advance().type
            right = self.parse_unary()
            left = BinaryExpr(op, left, right)
        return left

    def parse_unary(self):
        if self.check("-"):
            self.advance()
            return BinaryExpr("neg", None, self.parse_unary())
        if self.check("!"):
            self.advance()
            return BinaryExpr("not", None, self.parse_unary())
        return self.parse_call_or_attr()

    def parse_call_or_attr(self):
        expr = self.parse_primary()
        while True:
            if self.check("."):
                dot = self.advance()
                method = self.expect("NAME").value
                self.expect("(")
                args = self.parse_arg_list()
                self.expect(")")
                expr = MethodCallExpr(expr, method, args, dot.line, dot.col)
            elif self.check("("):
                paren = self.advance()
                args = self.parse_arg_list()
                self.expect(")")
                expr = CallExpr(expr, args, paren.line, paren.col)
            elif self.check("["):
                br = self.advance()
                index = self.parse_expr()
                self.expect("]")
                expr = IndexExpr(expr, index, br.line, br.col)
            else:
                break
        return expr

    def parse_arg_list(self):
        args = []
        if self.check(")"):
            return args
        args.append(self.parse_expr())
        while self.check(","):
            self.advance()
            args.append(self.parse_expr())
        return args

    def parse_primary(self):
        tok = self.peek()
        if tok.type == "(":
            self.advance()
            expr = self.parse_expr()
            self.expect(")")
            return expr
        if tok.type == "[":
            return self.parse_list_lit()
        if tok.type == "INT":
            self.advance()
            return NumberLiteral(int(tok.value))
        if tok.type == "FLOAT":
            self.advance()
            return FloatLiteral(float(tok.value))
        if tok.type == "STRING":
            self.advance()
            return StringLiteral(tok.value)
        if tok.type == "BOOL":
            self.advance()
            return BoolLiteral(tok.value == "true")
        if tok.type == "NAME":
            self.advance()
            return VarExpr(tok.value, tok.line, tok.col)
        raise ParseError(f"unexpected token {_describe_token(tok)}", tok.line, tok.col)

    def parse_list_lit(self):
        self.advance()  # [
        elements = []
        if not self.check("]"):
            elements.append(self.parse_expr())
            while self.check(","):
                self.advance()
                elements.append(self.parse_expr())
        self.expect("]")
        return ListLiteral(elements)


# ══════════════════════════════════════════════════════════════════════════
# Fake ML backend — pure Python, no torch/pandas
# ══════════════════════════════════════════════════════════════════════════

_DATASET_PRESETS = {
    "iris": dict(n_features=4, n_rows=150, n_classes=3),
    "spam": dict(n_features=10, n_rows=300, n_classes=2),
    "digit": dict(n_features=64, n_rows=500, n_classes=10),
    "mnist": dict(n_features=784, n_rows=500, n_classes=10),
    "housing": dict(n_features=8, n_rows=400, n_classes=None),
    "house": dict(n_features=8, n_rows=400, n_classes=None),
    "price": dict(n_features=8, n_rows=400, n_classes=None),
    "multitask": dict(n_features=6, n_rows=250, n_classes=None),
    "sample": dict(n_features=4, n_rows=100, n_classes=2),
}


def _preset_for(name):
    lname = name.lower()
    for key, preset in _DATASET_PRESETS.items():
        if key in lname:
            return preset
    return dict(n_features=8, n_rows=200, n_classes=2)


class DataSet:
    def __init__(self, name="dataset", X=None, y=None, columns=None, n_classes=None):
        self.name = name
        self.X = X if X is not None else []
        self.y = y if y is not None else []
        self.columns = columns or []
        self.n_classes = n_classes

    @classmethod
    def from_path(cls, path, n_targets=1):
        base = re.split(r"[\\/]", path)[-1]
        if base.endswith(".csv"):
            base = base[:-4]
        preset = _preset_for(base)
        n_features, n_rows, n_classes = preset["n_features"], preset["n_rows"], preset["n_classes"]
        X = [[random.gauss(0, 1) for _ in range(n_features)] for _ in range(n_rows)]
        if n_classes:
            y = [random.randrange(n_classes) for _ in range(n_rows)]
        else:
            y = [round(random.gauss(0, 1) * 50 + 100, 2) for _ in range(n_rows)]
        columns = [f"feature_{i}" for i in range(n_features)]
        return cls(base, X, y, columns, n_classes)

    def split(self, ratio):
        n = len(self.X)
        idx = list(range(n))
        random.shuffle(idx)
        n_train = max(1, int(n * ratio))
        tr, te = idx[:n_train], idx[n_train:]
        train = DataSet(f"{self.name}_train", [self.X[i] for i in tr], [self.y[i] for i in tr], self.columns, self.n_classes)
        test = DataSet(f"{self.name}_test", [self.X[i] for i in te], [self.y[i] for i in te], self.columns, self.n_classes)
        return train, test

    def normalize(self):
        if not self.X:
            return self
        n_features = len(self.X[0])
        means = [sum(row[i] for row in self.X) / len(self.X) for i in range(n_features)]
        stds = []
        for i in range(n_features):
            var = sum((row[i] - means[i]) ** 2 for row in self.X) / len(self.X)
            stds.append(var ** 0.5 or 1.0)
        Xn = [[(row[i] - means[i]) / stds[i] for i in range(n_features)] for row in self.X]
        result = DataSet(self.name, Xn, list(self.y), self.columns, self.n_classes)
        return result

    def shuffle(self):
        idx = list(range(len(self.X)))
        random.shuffle(idx)
        return DataSet(self.name, [self.X[i] for i in idx], [self.y[i] for i in idx], self.columns, self.n_classes)

    def augment(self):
        Xa = [[v + random.gauss(0, 0.05) for v in row] for row in self.X]
        return DataSet(self.name, Xa, list(self.y), self.columns, self.n_classes)

    def oversample(self):
        return self

    def undersample(self):
        return self

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        return f"DataSet({self.name!r}, {len(self)} rows)"


def fake_load(path, n_targets=1):
    return DataSet.from_path(str(path), int(n_targets))


class NeuvaModel:
    def __init__(self, layers, outputs=None, name="Model"):
        self.layers = layers
        self.outputs = outputs or []
        self.model_name = name
        self.trained = False

    def __repr__(self):
        return f"<NeuvaModel {self.model_name}>"


class NeuvaTrainer:
    def train(self, model, dataset, epochs, lr=0.001, loss_fn=None, lr_schedule="none", early_stop=None):
        print("Training on: cpu")
        start = 1.05 + random.uniform(-0.05, 0.05)
        floor = random.uniform(0.25, 0.35)
        for epoch in range(1, epochs + 1):
            progress = epoch / epochs
            loss = floor + (start - floor) * ((1 - progress) ** 1.6) + random.uniform(-0.015, 0.015)
            loss = max(loss, floor - 0.03)
            print(f"Epoch {epoch}/{epochs} — loss: {loss:.4f}")
        model.trained = True


def evaluate_accuracy(model, data):
    return round(random.uniform(0.85, 0.98), 4)


def _rand_metric(lo=0.80, hi=0.95):
    return round(random.uniform(lo, hi), 4)


def confusion_matrix(model, data):
    n = max(len(data), 2)
    tp = int(n * 0.47)
    tn = int(n * 0.44)
    fp = max(n - tp - tn - 1, 0) // 2
    fn = n - tp - tn - fp
    return [[tp, fp], [fn, tn]]


def save_model(model, path):
    print(f"Model saved to '{path}'")


def load_model(path):
    layers = [LayerStatement("dense", [8, 32, "relu"]), LayerStatement("dense", [32, 1, "linear"])]
    name = re.split(r"[\\/]", str(path))[-1]
    model = NeuvaModel(layers, [], name=name)
    model.trained = True
    return model


def _flatten_pair(pred, target):
    if isinstance(pred, (list, tuple)):
        return list(zip(pred, target))
    return [(pred, target)]


def mse_fn(pred, target):
    pairs = _flatten_pair(pred, target)
    return sum((p - t) ** 2 for p, t in pairs) / len(pairs)


def mae_fn(pred, target):
    pairs = _flatten_pair(pred, target)
    return sum(abs(p - t) for p, t in pairs) / len(pairs)


def _clamp01(p):
    return min(max(p, 1e-7), 1 - 1e-7)


def binary_crossentropy_fn(pred, target):
    pairs = _flatten_pair(pred, target)
    total = 0.0
    for p, t in pairs:
        p = _clamp01(p)
        total += -(t * math.log(p) + (1 - t) * math.log(1 - p))
    return total / len(pairs)


def crossentropy_fn(pred, target):
    if isinstance(pred, (list, tuple)) and isinstance(target, int):
        p = _clamp01(pred[target]) if 0 <= target < len(pred) else _clamp01(pred[0])
        return -math.log(p)
    return binary_crossentropy_fn(pred, target)


LOSS_NAMES = {"mse", "mae", "crossentropy", "binary_crossentropy"}


class TableView:
    def __init__(self, data):
        self.data = data


class PlotView:
    def __init__(self, values):
        self.values = values


def _fmt_cell(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_table(data, max_rows=10):
    if isinstance(data, DataSet):
        n_features = len(data.X[0]) if data.X else 0
        headers = list(data.columns) if data.columns else [f"col{i}" for i in range(n_features)]
        if data.y:
            headers = headers + ["target"]
        total = len(data.X)
        rows = []
        for i in range(min(max_rows, total)):
            row = [_fmt_cell(v) for v in data.X[i]]
            if data.y:
                row.append(_fmt_cell(data.y[i]))
            rows.append(row)
    elif isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, (list, tuple)):
            headers = [f"col{i}" for i in range(len(first))]
            rows = [[_fmt_cell(v) for v in row] for row in data[:max_rows]]
        else:
            headers = ["value"]
            rows = [[_fmt_cell(v)] for v in data[:max_rows]]
        total = len(data)
    else:
        return "(empty table)"

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def hline(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def fmt_row(cells):
        return "│" + "│".join(f" {c.rjust(widths[i])} " for i, c in enumerate(cells)) + "│"

    lines = [hline("┌", "┬", "┐"), fmt_row(headers), hline("├", "┼", "┤")]
    lines += [fmt_row(r) for r in rows]
    lines.append(hline("└", "┴", "┘"))
    if total > max_rows:
        lines.append(f"... {total - max_rows} more rows")
    return "\n".join(lines)


def render_plot(values, height=10):
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


def _boxify(lines):
    width = max(len(l) for l in lines)
    top = "┌" + "─" * (width + 2) + "┐"
    bottom = "└" + "─" * (width + 2) + "┘"
    body = [f"│ {l.ljust(width)} │" for l in lines]
    return "\n".join([top] + body + [bottom])


def render_model_summary(name, layers, outputs, trained=False):
    lines = [f"Model: {name}" + (" (trained)" if trained else "")]
    total = 0
    for layer in layers:
        args = layer.args
        if layer.name in ("rnn", "lstm") and len(args) >= 2:
            num_layers = args[2] if len(args) >= 3 else 1
            lines.append(f"  {layer.name}({args[0]} -> {args[1]}, num_layers={num_layers})")
        elif layer.name == "transformer" and len(args) >= 3:
            lines.append(f"  transformer(embed_dim={args[0]}, num_heads={args[1]}, ff_dim={args[2]})")
        elif layer.name == "attention" and len(args) >= 2:
            lines.append(f"  attention(embed_dim={args[0]}, num_heads={args[1]})")
        elif layer.name == "embedding" and len(args) >= 2:
            lines.append(f"  embedding(vocab_size={args[0]}, embed_dim={args[1]})")
        elif len(args) >= 3:
            lines.append(f"  {layer.name}({args[0]} -> {args[1]}, {args[2]})")
            try:
                total += int(args[0]) * int(args[1]) + int(args[1])
            except (ValueError, TypeError):
                pass
        elif len(args) == 2:
            lines.append(f"  {layer.name}({args[0]} -> {args[1]})")
            try:
                total += int(args[0]) * int(args[1]) + int(args[1])
            except (ValueError, TypeError):
                pass
        elif len(args) == 1:
            lines.append(f"  {layer.name}({args[0]})")
        else:
            lines.append(f"  {layer.name}")
    for out in outputs:
        a = out.layer.args
        lines.append(f"  output {out.output_name}: {out.layer.name}({a[0]} -> {a[1]}, {a[2]})")
        try:
            total += int(a[0]) * int(a[1]) + int(a[1])
        except (ValueError, TypeError):
            pass
    lines.append(f"Total parameters: {total}")
    return _boxify(lines)


# ══════════════════════════════════════════════════════════════════════════
# Interpreter
# ══════════════════════════════════════════════════════════════════════════


class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


class NeuvaFunction:
    def __init__(self, name, params, body, closure):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure

    def __repr__(self):
        return f"<fn {self.name}>"


def find_similar_name(name, available_names):
    matches = difflib.get_close_matches(name, available_names, n=1, cutoff=0.6)
    if matches:
        return f"Did you mean '{matches[0]}'?"
    return None


class Environment:
    def __init__(self, parent=None):
        self._store = {}
        self.parent = parent

    def get(self, name, line=None, col=None):
        env = self
        while env is not None:
            if name in env._store:
                return env._store[name]
            env = env.parent
        hint = find_similar_name(name, self._all_names())
        raise RuntimeError_(f"undefined variable '{name}'", line=line, col=col, hint=hint)

    def _all_names(self):
        names = list(self._store.keys())
        if self.parent is not None:
            names.extend(self.parent._all_names())
        return names

    def set(self, name, value):
        self._store[name] = value


_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "%": lambda a, b: a % b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
}


class NeuvaInterpreter:
    def __init__(self):
        self.env = Environment()

        def _load(path, n_targets=1):
            return fake_load(path, n_targets)

        self.env.set("range", range)
        self.env.set("len", len)
        self.env.set("load", _load)
        self.env.set("accuracy", lambda model, data: evaluate_accuracy(model, data))
        self.env.set("predict", lambda model, data: "predictions")
        self.env.set("normalize", lambda: DataSet())
        self.env.set("shuffle", lambda: DataSet())

        self.env.set("mse", mse_fn)
        self.env.set("mae", mae_fn)
        self.env.set("crossentropy", crossentropy_fn)
        self.env.set("binary_crossentropy", binary_crossentropy_fn)

        self.env.set("precision", lambda model, data: _rand_metric(0.80, 0.95))
        self.env.set("recall", lambda model, data: _rand_metric(0.78, 0.95))
        self.env.set("f1_score", lambda model, data: _rand_metric(0.80, 0.95))
        self.env.set("confusion_matrix", lambda model, data: confusion_matrix(model, data))

        self.env.set("table", lambda d: TableView(d))
        self.env.set("plot", lambda v: PlotView(v))

    # ── dispatch ─────────────────────────────────────────────────────────

    def visit(self, node):
        method = f"visit_{type(node).__name__}"
        handler = getattr(self, method, self._unhandled)
        return handler(node)

    def _unhandled(self, node):
        pass

    def evaluate(self, node):
        method = f"eval_{type(node).__name__}"
        handler = getattr(self, method, None)
        if handler is None:
            raise RuntimeError_(f"cannot evaluate node type '{type(node).__name__}'", node.line, node.col)
        return handler(node)

    # ── statements ───────────────────────────────────────────────────────

    def visit_Program(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_LetStatement(self, node):
        value = self.evaluate(node.value)
        if len(node.names) == 1:
            self.env.set(node.names[0], value)
        else:
            unpacked = list(value)
            if len(unpacked) != len(node.names):
                raise RuntimeError_(
                    f"cannot unpack {len(unpacked)} values into {len(node.names)} variables",
                    node.line, node.col,
                )
            for name, val in zip(node.names, unpacked):
                self.env.set(name, val)

    def visit_ModelStatement(self, node):
        self.env.set(node.name, node)

    def visit_TrainStatement(self, node):
        model_node = self.env.get(node.model, node.line, node.col)
        data_obj = self.env.get(node.data, node.line, node.col)

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
            fn_obj = self.env.get(loss_name, node.line, node.col)
            if not isinstance(fn_obj, NeuvaFunction):
                raise RuntimeError_(
                    f"'{loss_name}' is not a valid loss — expected a built-in "
                    f"({', '.join(sorted(LOSS_NAMES))}) or a function name",
                    node.line, node.col,
                )
            loss_fn = lambda pred, target, _fn=fn_obj: self._call_function(_fn, [pred, target])

        if isinstance(model_node, ModelStatement):
            neuva_model = NeuvaModel(model_node.layers, model_node.outputs, name=node.model)
        else:
            neuva_model = model_node

        NeuvaTrainer().train(
            neuva_model, data_obj, node.epochs, lr=lr, loss_fn=loss_fn,
            lr_schedule=lr_schedule, early_stop=early_stop,
        )
        self.env.set(node.model, neuva_model)

    def visit_SaveStatement(self, node):
        model = self.env.get(node.model, node.line, node.col)
        if not isinstance(model, NeuvaModel):
            raise RuntimeError_(f"'{node.model}' is not a trained model and cannot be saved", node.line, node.col)
        path = str(self.evaluate(node.path))
        save_model(model, path)

    def visit_PredictStatement(self, node):
        pass

    def visit_ImportStatement(self, node):
        raise RuntimeError_(
            "stdlib imports are not available in the web playground",
            node.line, node.col,
        )

    def visit_PrintStatement(self, node):
        values = [self.evaluate(e) for e in node.exprs]
        if len(values) == 1:
            val = values[0]
            if isinstance(val, ModelStatement):
                print(render_model_summary(val.name, val.layers, val.outputs))
                return
            if isinstance(val, NeuvaModel):
                print(render_model_summary(val.model_name, val.layers, val.outputs, trained=val.trained))
                return
            if isinstance(val, TableView):
                print(render_table(val.data))
                return
            if isinstance(val, PlotView):
                print(render_plot(val.values))
                return
        print(*values)

    def visit_ExprStatement(self, node):
        return self.evaluate(node.expr)

    def visit_IfStatement(self, node):
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

    def visit_ForStatement(self, node):
        iterable = self.evaluate(node.iterable)
        for val in iterable:
            self.env.set(node.var, val)
            for stmt in node.body:
                self.visit(stmt)

    def visit_WhileStatement(self, node):
        while self.evaluate(node.condition):
            for stmt in node.body:
                self.visit(stmt)

    def visit_FnStatement(self, node):
        self.env.set(node.name, NeuvaFunction(node.name, node.params, node.body, self.env))

    def visit_ReturnStatement(self, node):
        value = self.evaluate(node.value) if node.value is not None else None
        raise ReturnSignal(value)

    # ── expressions ──────────────────────────────────────────────────────

    def eval_NumberLiteral(self, node):
        return node.value

    def eval_FloatLiteral(self, node):
        return node.value

    def eval_StringLiteral(self, node):
        s = node.value
        if "{" not in s:
            return s

        def _sub(m):
            try:
                return str(self.env.get(m.group(1)))
            except RuntimeError_:
                return m.group(0)

        return re.sub(r"\{(\w+)\}", _sub, s)

    def eval_BoolLiteral(self, node):
        return node.value

    def eval_ListLiteral(self, node):
        return [self.evaluate(e) for e in node.elements]

    def eval_IndexExpr(self, node):
        obj = self.evaluate(node.obj)
        idx = self.evaluate(node.index)
        try:
            return obj[idx]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError_(str(exc), node.line, node.col)

    def eval_VarExpr(self, node):
        return self.env.get(node.name, line=node.line, col=node.col)

    def eval_BinaryExpr(self, node):
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
            raise RuntimeError_(f"unknown operator '{op}'", node.line, node.col)

        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        if op == "/" and right == 0:
            raise RuntimeError_("division by zero", node.line, node.col)

        if op == "+" and (isinstance(left, str) or isinstance(right, str)):
            return str(left) + str(right)

        return fn(left, right)

    def _call_function(self, fn, args, line=None, col=None):
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

    def eval_CallExpr(self, node):
        callee = self.evaluate(node.callee)
        args = [self.evaluate(a) for a in node.args]

        if callable(callee) and not isinstance(callee, NeuvaFunction):
            return callee(*args)
        if isinstance(callee, NeuvaFunction):
            return self._call_function(callee, args, node.line, node.col)
        raise RuntimeError_(f"'{callee}' is not callable", node.line, node.col)

    def eval_MethodCallExpr(self, node):
        obj = self.evaluate(node.obj)
        method = getattr(obj, node.method, None)
        if method is None:
            raise RuntimeError_(f"object has no method '{node.method}'", node.line, node.col)
        args = [self.evaluate(a) for a in node.args]
        return method(*args)


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════


def run_neuva(source):
    """Run Neuva source, returning everything it printed (or a formatted error)."""
    try:
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse_program()
    except NeuvaError as exc:
        return format_error(exc, source)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            NeuvaInterpreter().visit(program)
    except NeuvaError as exc:
        return buf.getvalue() + format_error(exc, source)
    except RecursionError:
        return buf.getvalue() + "Error: recursion limit exceeded"
    except Exception as exc:  # safety net for unexpected host errors
        return buf.getvalue() + f"Error: {exc}"
    return buf.getvalue()


if __name__ == "__main__":
    _demo = 'let name = "Neuva"\nprint "Hello from", name\n'
    print(run_neuva(_demo))
