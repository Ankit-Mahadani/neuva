from lark import Transformer, Token, Tree
from .ast_nodes import (
    Program, LetStatement, PrintStatement, ModelStatement, LayerStatement,
    TrainStatement, TrainOption, SaveStatement, PredictStatement,
    FnStatement, Parameter, ReturnStatement, IfStatement, ForStatement,
    WhileStatement, ExprStatement, BinaryExpr, CallExpr, MethodCallExpr,
    VarExpr, NumberLiteral, FloatLiteral, StringLiteral, BoolLiteral,
)


def _pos(meta):
    line = getattr(meta, "line", None)
    col = getattr(meta, "column", None)
    return line, col


class NeuvaTransformer(Transformer):

    # ── top-level ──────────────────────────────────────────────────────────

    def start(self, items):
        return Program(body=[s for s in items if s is not None])

    def statement(self, items):
        # strip trailing NEWLINE tokens
        stmts = [i for i in items if not isinstance(i, Token)]
        return stmts[0] if stmts else None

    # ── statements ─────────────────────────────────────────────────────────

    def print_stmt(self, items):
        return PrintStatement(expr=items[0])

    def let_stmt(self, items):
        if len(items) == 2:
            return LetStatement(name=str(items[0]), value=items[1])
        # typed: name, type, value
        return LetStatement(name=str(items[0]), type_ann=items[1], value=items[2])

    def model_def(self, items):
        name = str(items[0])
        layers = [i for i in items[1:] if isinstance(i, LayerStatement)]
        return ModelStatement(name=name, layers=layers)

    def layer_stmt(self, items):
        name = str(items[0])
        args = list(items[1]) if len(items) > 1 and items[1] is not None else []
        return LayerStatement(name=name, args=args)

    def train_stmt(self, items):
        model = str(items[0])
        data = items[1]
        options = list(items[2]) if len(items) > 2 and items[2] is not None else []
        return TrainStatement(model=model, data=data, options=options)

    def train_options(self, items):
        return items

    def train_opt_lr(self, items):     return TrainOption(key="lr",     value=items[0])
    def train_opt_loss(self, items):   return TrainOption(key="loss",   value=items[0])
    def train_opt_epochs(self, items): return TrainOption(key="epochs", value=items[0])

    def save_stmt(self, items):
        return SaveStatement(model=str(items[0]), path=items[1])

    def predict_stmt(self, items):
        return PredictStatement(model=str(items[0]), input=items[1])

    def func_def(self, items):
        name = str(items[0])
        rest = list(items[1:])
        params = []
        return_type = None
        body = []
        for item in rest:
            if isinstance(item, list) and all(isinstance(p, Parameter) for p in item):
                params = item
            elif isinstance(item, str) and item.startswith("type_"):
                return_type = item
            elif item is not None:
                body.append(item)
        return FnStatement(name=name, params=params, return_type=return_type, body=body)

    def param_list(self, items):
        return items

    def param(self, items):
        return Parameter(name=str(items[0]), type_ann=items[1])

    def return_stmt(self, items):
        return ReturnStatement(value=items[0] if items else None)

    def if_body(self, items):
        return [i for i in items if i is not None and not isinstance(i, Token)]

    def if_stmt(self, items):
        cond = items[0]
        then_body = items[1] if len(items) > 1 else []
        else_body = items[2] if len(items) > 2 else []
        return IfStatement(condition=cond, then_body=then_body, else_body=else_body)

    def for_body(self, items):
        return [i for i in items if i is not None and not isinstance(i, Token)]

    def for_stmt(self, items):
        var = str(items[0])
        iterable = items[1]
        body = items[2] if len(items) > 2 else []
        return ForStatement(var=var, iterable=iterable, body=body)

    def while_body(self, items):
        return [i for i in items if i is not None and not isinstance(i, Token)]

    def while_stmt(self, items):
        cond = items[0]
        body = items[1] if len(items) > 1 else []
        return WhileStatement(condition=cond, body=body)

    def expr_stmt(self, items):
        return ExprStatement(expr=items[0])

    # ── expressions ────────────────────────────────────────────────────────

    def add(self, items): return BinaryExpr(op="+",  left=items[0], right=items[1])
    def sub(self, items): return BinaryExpr(op="-",  left=items[0], right=items[1])
    def mul(self, items): return BinaryExpr(op="*",  left=items[0], right=items[1])
    def div(self, items): return BinaryExpr(op="/",  left=items[0], right=items[1])
    def mod(self, items): return BinaryExpr(op="%",  left=items[0], right=items[1])
    def eq(self, items):  return BinaryExpr(op="==", left=items[0], right=items[1])
    def ne(self, items):  return BinaryExpr(op="!=", left=items[0], right=items[1])
    def lt(self, items):  return BinaryExpr(op="<",  left=items[0], right=items[1])
    def le(self, items):  return BinaryExpr(op="<=", left=items[0], right=items[1])
    def gt(self, items):  return BinaryExpr(op=">",  left=items[0], right=items[1])
    def ge(self, items):  return BinaryExpr(op=">=", left=items[0], right=items[1])
    def neg(self, items): return BinaryExpr(op="neg", left=None, right=items[0])
    def not_(self, items): return BinaryExpr(op="not", left=None, right=items[0])

    def func_call(self, items):
        callee = items[0]
        args = list(items[1]) if len(items) > 1 and items[1] is not None else []
        return CallExpr(callee=callee, args=args)

    def method_call(self, items):
        obj = items[0]
        method = str(items[1])
        args = list(items[2]) if len(items) > 2 and items[2] is not None else []
        return MethodCallExpr(obj=obj, method=method, args=args)

    def attr_access(self, items):
        # treat as VarExpr with dotted name for simplicity
        return VarExpr(name=f"{items[0]}.{items[1]}")

    def arg_list(self, items):
        return items

    def var(self, items):
        return VarExpr(name=str(items[0]))

    # ── literals ───────────────────────────────────────────────────────────

    def int_lit(self, items):   return NumberLiteral(value=int(items[0]))
    def float_lit(self, items): return FloatLiteral(value=float(items[0]))
    def string_lit(self, items):
        raw = str(items[0])
        return StringLiteral(value=raw[1:-1])  # strip quotes
    def bool_lit(self, items):
        return BoolLiteral(value=str(items[0]) == "true")

    # ── types ──────────────────────────────────────────────────────────────

    def type_int(self, _):    return "type_int"
    def type_float(self, _):  return "type_float"
    def type_bool(self, _):   return "type_bool"
    def type_string(self, _): return "type_string"
    def type_tensor(self, _): return "type_tensor"
    def type_matrix(self, _): return "type_matrix"

    # ── discard bare NEWLINEs that bubble up ───────────────────────────────

    def NEWLINE(self, token):
        return None

    def transform(self, tree):
        return super().transform(tree)
