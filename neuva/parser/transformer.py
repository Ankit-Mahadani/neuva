from lark import Transformer, Token, Tree
from .ast_nodes import (
    Program, LetStatement, PrintStatement, ModelStatement, LayerStatement,
    TrainStatement, TrainOption, SaveStatement, PredictStatement,
    FnStatement, Parameter, ReturnStatement, IfStatement, ForStatement,
    WhileStatement, ExprStatement, BinaryExpr, CallExpr, MethodCallExpr,
    VarExpr, NumberLiteral, FloatLiteral, StringLiteral, BoolLiteral,
    ListLiteral, IndexExpr,
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
        return PrintStatement(exprs=list(items))

    def let_stmt(self, items):
        names = [str(i) for i in items if isinstance(i, Token)]
        rest = [i for i in items if not isinstance(i, Token)]
        if len(rest) == 1:
            return LetStatement(names=names, value=rest[0])
        return LetStatement(names=names, type_ann=rest[0], value=rest[1])

    def model_def(self, items):
        name = str(items[0])
        layers = [i for i in items[1:] if isinstance(i, LayerStatement)]
        return ModelStatement(name=name, layers=layers)

    def layer_dense(self, items):
        # NAME NUMBER ARROW NUMBER NAME  (NEWLINE→None filtered)
        t = [i for i in items if i is not None]
        return LayerStatement(name=str(t[0]), args=[int(str(t[1])), int(str(t[3])), str(t[4])])

    def layer_conv(self, items):
        # NAME NUMBER ARROW NUMBER NUMBER
        t = [i for i in items if i is not None]
        return LayerStatement(name=str(t[0]), args=[int(str(t[1])), int(str(t[3])), int(str(t[4]))])

    def layer_pool(self, items):
        # NAME NUMBER  (number may be int like pool size or float like dropout rate)
        t = [i for i in items if i is not None]
        val = str(t[1])
        arg = float(val) if '.' in val else int(val)
        return LayerStatement(name=str(t[0]), args=[arg])

    def layer_no_args(self, items):
        # NAME
        t = [i for i in items if i is not None]
        return LayerStatement(name=str(t[0]), args=[])

    def train_body(self, items):
        clean = [i for i in items if i is not None]
        data = str(clean[0])
        epochs = int(str(clean[1]))
        options = [i for i in clean[2:] if isinstance(i, TrainOption)]
        return {"data": data, "epochs": epochs, "options": options}

    def train_stmt(self, items):
        clean = [i for i in items if i is not None]
        model = str(clean[0])
        if isinstance(clean[1], dict):
            # Multi-line: "train" NAME NEWLINE train_body
            body = clean[1]
            return TrainStatement(model=model, data=body["data"], epochs=body["epochs"], options=body["options"])
        # Single-line: "train" NAME "on" NAME "for" NUMBER "epochs" ("," train_opt)* NEWLINE
        data = str(clean[1])
        epochs = int(str(clean[2]))
        options = [i for i in clean[3:] if isinstance(i, TrainOption)]
        return TrainStatement(model=model, data=data, epochs=epochs, options=options)

    def train_opt(self, items):
        token = items[0]
        if token.type == "NAME":
            return TrainOption(key="loss", value=str(token))
        return TrainOption(key="lr", value=float(str(token)))

    def save_stmt(self, items):
        return SaveStatement(model=str(items[0]), path=items[1])

    def predict_stmt(self, items):
        return PredictStatement(model=str(items[0]), data=str(items[1]))

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

    def elif_clause(self, items):
        cond = items[0]
        body = [i for i in items[1:] if i is not None and not isinstance(i, Token)]
        return {"__type__": "elif", "cond": cond, "body": body}

    def else_clause(self, items):
        body = [i for i in items if i is not None and not isinstance(i, Token)]
        return {"__type__": "else", "body": body}

    def if_stmt(self, items):
        cond = items[0]
        then_body = []
        elif_branches = []
        else_branch = []
        for item in items[1:]:
            if item is None or isinstance(item, Token):
                continue
            if isinstance(item, dict) and item.get("__type__") == "elif":
                elif_branches.append((item["cond"], item["body"]))
            elif isinstance(item, dict) and item.get("__type__") == "else":
                else_branch = item["body"]
            else:
                then_body.append(item)
        return IfStatement(condition=cond, then_body=then_body, elif_branches=elif_branches, else_branch=else_branch)

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

    def add(self, items): return BinaryExpr(op="+",   left=items[0], right=items[1])
    def sub(self, items): return BinaryExpr(op="-",   left=items[0], right=items[1])
    def mul(self, items): return BinaryExpr(op="*",   left=items[0], right=items[1])
    def div(self, items): return BinaryExpr(op="/",   left=items[0], right=items[1])
    def mod(self, items): return BinaryExpr(op="%",   left=items[0], right=items[1])
    def eq(self, items):  return BinaryExpr(op="==",  left=items[0], right=items[1])
    def ne(self, items):  return BinaryExpr(op="!=",  left=items[0], right=items[1])
    def lt(self, items):  return BinaryExpr(op="<",   left=items[0], right=items[1])
    def le(self, items):  return BinaryExpr(op="<=",  left=items[0], right=items[1])
    def gt(self, items):  return BinaryExpr(op=">",   left=items[0], right=items[1])
    def ge(self, items):  return BinaryExpr(op=">=",  left=items[0], right=items[1])
    def and_(self, items): return BinaryExpr(op="and", left=items[0], right=items[1])
    def or_(self, items):  return BinaryExpr(op="or",  left=items[0], right=items[1])
    def neg(self, items):  return BinaryExpr(op="neg", left=None, right=items[0])
    def not_(self, items): return BinaryExpr(op="not", left=None, right=items[0])

    def list_lit(self, items):
        return ListLiteral(elements=[i for i in items if i is not None])

    def index_expr(self, items):
        return IndexExpr(obj=items[0], index=items[1])

    def func_call(self, items):
        callee = items[0]
        args = list(items[1]) if len(items) > 1 and items[1] is not None else []
        return CallExpr(callee=callee, args=args)

    def method_call(self, items):
        obj = items[0]
        method = str(items[1])
        args = list(items[2]) if len(items) > 2 and items[2] is not None else []
        return MethodCallExpr(obj=obj, method=method, args=args)

    def arg_list(self, items):
        return items

    def var(self, items):
        tok = items[0]
        return VarExpr(name=str(tok), line=getattr(tok, "line", None), col=getattr(tok, "column", None))

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
