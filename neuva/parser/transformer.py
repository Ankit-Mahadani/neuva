"""Lark tree -> AST transformer.

Each rule method receives ``items``: the list of already-transformed
children for that parse tree node (a mix of ``Token``, AST node instances,
and occasionally ``None`` for filtered-out tokens like NEWLINE). Most
methods return a single AST node; a few return plain ``dict``/``tuple``/
``list`` values used only as internal markers that get unpacked by a
parent rule method (e.g. ``elif_clause``, ``dict_entry``, ``param_list``).
"""

from typing import Any, Optional
from lark import Transformer, Token, Tree
from .ast_nodes import (
    Program,
    LetStatement,
    PrintStatement,
    ModelStatement,
    LayerStatement,
    TrainStatement,
    TrainOption,
    SaveStatement,
    PredictStatement,
    FnStatement,
    Parameter,
    ReturnStatement,
    IfStatement,
    ForStatement,
    WhileStatement,
    ExprStatement,
    BinaryExpr,
    CallExpr,
    MethodCallExpr,
    VarExpr,
    NumberLiteral,
    FloatLiteral,
    StringLiteral,
    BoolLiteral,
    ListLiteral,
    IndexExpr,
    OutputLayerStatement,
    ImportStatement,
    AssignStatement,
    AugAssignStatement,
    DictLiteral,
    MatchStatement,
)


def _pos(meta):
    line = getattr(meta, "line", None)
    col = getattr(meta, "column", None)
    return line, col


class NeuvaTransformer(Transformer):

    # ── top-level ──────────────────────────────────────────────────────────

    def start(self, items: list) -> Program:
        return Program(body=[s for s in items if s is not None])

    def statement(self, items: list) -> Optional[Any]:
        # strip trailing NEWLINE tokens
        stmts = [i for i in items if not isinstance(i, Token)]
        return stmts[0] if stmts else None

    # ── statements ─────────────────────────────────────────────────────────

    def print_stmt(self, items: list) -> PrintStatement:
        return PrintStatement(exprs=list(items))

    def let_stmt(self, items: list) -> LetStatement:
        names = [str(i) for i in items if isinstance(i, Token)]
        rest = [i for i in items if not isinstance(i, Token)]
        if len(rest) == 1:
            return LetStatement(names=names, value=rest[0])
        return LetStatement(names=names, type_ann=rest[0], value=rest[1])

    def model_def(self, items: list) -> ModelStatement:
        name = str(items[0])
        layers = [i for i in items[1:] if isinstance(i, LayerStatement)]
        outputs = [i for i in items[1:] if isinstance(i, OutputLayerStatement)]
        return ModelStatement(name=name, layers=layers, outputs=outputs)

    def layer_dense(self, items: list) -> LayerStatement:
        # NAME NUMBER ARROW NUMBER NAME  (NEWLINE→None filtered)
        t = [i for i in items if i is not None]
        return LayerStatement(
            name=str(t[0]), args=[int(str(t[1])), int(str(t[3])), str(t[4])]
        )

    def layer_conv(self, items: list) -> LayerStatement:
        # NAME NUMBER ARROW NUMBER NUMBER
        t = [i for i in items if i is not None]
        return LayerStatement(
            name=str(t[0]), args=[int(str(t[1])), int(str(t[3])), int(str(t[4]))]
        )

    def layer_seq(self, items: list) -> LayerStatement:
        # NAME NUMBER ARROW NUMBER  (rnn/lstm without explicit num_layers)
        t = [i for i in items if i is not None]
        return LayerStatement(name=str(t[0]), args=[int(str(t[1])), int(str(t[3]))])

    def layer_pool(self, items: list) -> LayerStatement:
        # NAME NUMBER  (number may be int like pool size or float like dropout rate)
        t = [i for i in items if i is not None]
        val = str(t[1])
        arg = float(val) if "." in val else int(val)
        return LayerStatement(name=str(t[0]), args=[arg])

    def layer_pair(self, items: list) -> LayerStatement:
        # NAME NUMBER "," NUMBER  (e.g. attention(embed_dim, num_heads), embedding(vocab_size, embed_dim))
        t = [i for i in items if i is not None]
        return LayerStatement(name=str(t[0]), args=[int(str(t[1])), int(str(t[2]))])

    def layer_triple(self, items: list) -> LayerStatement:
        # NAME NUMBER "," NUMBER "," NUMBER  (e.g. transformer(embed_dim, num_heads, ff_dim))
        t = [i for i in items if i is not None]
        return LayerStatement(
            name=str(t[0]), args=[int(str(t[1])), int(str(t[2])), int(str(t[3]))]
        )

    def layer_no_args(self, items: list) -> LayerStatement:
        # NAME
        t = [i for i in items if i is not None]
        return LayerStatement(name=str(t[0]), args=[])

    def output_dense(self, items: list) -> OutputLayerStatement:
        # output NAME ":" NAME "(" NUMBER ARROW NUMBER "," NAME ")"
        t = [i for i in items if i is not None]
        output_name = str(t[0])
        layer = LayerStatement(
            name=str(t[1]), args=[int(str(t[2])), int(str(t[4])), str(t[5])]
        )
        return OutputLayerStatement(output_name=output_name, layer=layer)

    def train_body(self, items: list) -> dict[str, Any]:
        clean = [i for i in items if i is not None]
        data = str(clean[0])
        epochs = int(str(clean[1]))
        options = [i for i in clean[2:] if isinstance(i, TrainOption)]
        return {"data": data, "epochs": epochs, "options": options}

    def train_stmt(self, items: list) -> TrainStatement:
        clean = [i for i in items if i is not None]
        model = str(clean[0])
        if isinstance(clean[1], dict):
            # Multi-line: "train" NAME NEWLINE train_body
            body = clean[1]
            return TrainStatement(
                model=model,
                data=body["data"],
                epochs=body["epochs"],
                options=body["options"],
            )
        # Single-line: "train" NAME "on" NAME "for" NUMBER "epochs" ("," train_opt)* NEWLINE
        data = str(clean[1])
        epochs = int(str(clean[2]))
        options = [i for i in clean[3:] if isinstance(i, TrainOption)]
        return TrainStatement(model=model, data=data, epochs=epochs, options=options)

    def opt_lr(self, items: list) -> TrainOption:
        return TrainOption(key="lr", value=items[0])

    def opt_loss(self, items: list) -> TrainOption:
        return TrainOption(key="loss", value=str(items[0]))

    def opt_lr_schedule(self, items: list) -> TrainOption:
        return TrainOption(key="lr_schedule", value=str(items[0]))

    def opt_early_stop(self, items: list) -> TrainOption:
        return TrainOption(key="early_stop", value=items[0])

    def opt_lr_warmup(self, items: list) -> TrainOption:
        return TrainOption(key="lr_warmup", value=items[0])

    def save_stmt(self, items: list) -> SaveStatement:
        return SaveStatement(model=str(items[0]), path=items[1])

    def import_stmt(self, items: list) -> ImportStatement:
        return ImportStatement(module=str(items[0]))

    def predict_stmt(self, items: list) -> PredictStatement:
        return PredictStatement(model=str(items[0]), data=str(items[1]))

    def func_def(self, items: list) -> FnStatement:
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

    def param_list(self, items: list) -> list:
        return items

    def param(self, items: list) -> Parameter:
        type_ann = items[1] if len(items) > 1 else None
        return Parameter(name=str(items[0]), type_ann=type_ann)

    def return_stmt(self, items: list) -> ReturnStatement:
        return ReturnStatement(value=items[0] if items else None)

    def elif_clause(self, items: list) -> dict[str, Any]:
        cond = items[0]
        body = [i for i in items[1:] if i is not None and not isinstance(i, Token)]
        return {"__type__": "elif", "cond": cond, "body": body}

    def else_clause(self, items: list) -> dict[str, Any]:
        body = [i for i in items if i is not None and not isinstance(i, Token)]
        return {"__type__": "else", "body": body}

    def if_stmt(self, items: list) -> IfStatement:
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
        return IfStatement(
            condition=cond,
            then_body=then_body,
            elif_branches=elif_branches,
            else_branch=else_branch,
        )

    def for_body(self, items: list) -> list:
        return [i for i in items if i is not None and not isinstance(i, Token)]

    def for_stmt(self, items: list) -> ForStatement:
        var = str(items[0])
        iterable = items[1]
        body = items[2] if len(items) > 2 else []
        return ForStatement(var=var, iterable=iterable, body=body)

    def while_body(self, items: list) -> list:
        return [i for i in items if i is not None and not isinstance(i, Token)]

    def while_stmt(self, items: list) -> WhileStatement:
        cond = items[0]
        body = items[1] if len(items) > 1 else []
        return WhileStatement(condition=cond, body=body)

    def case_clause(self, items: list) -> dict[str, Any]:
        value = items[0]
        stmt = items[1] if len(items) > 1 else None
        return {"__type__": "case", "value": value, "stmt": stmt}

    def default_clause(self, items: list) -> dict[str, Any]:
        stmt = items[0] if items else None
        return {"__type__": "default", "stmt": stmt}

    def match_stmt(self, items: list) -> MatchStatement:
        clean = [i for i in items if i is not None and not isinstance(i, Token)]
        subject = clean[0]
        cases = []
        default = None
        for item in clean[1:]:
            if isinstance(item, dict) and item.get("__type__") == "case":
                cases.append((item["value"], item["stmt"]))
            elif isinstance(item, dict) and item.get("__type__") == "default":
                default = item["stmt"]
        return MatchStatement(subject=subject, cases=cases, default=default)

    def assign(self, items: list) -> AssignStatement:
        tok = items[0]
        return AssignStatement(
            name=str(tok),
            value=items[1],
            line=getattr(tok, "line", None),
            col=getattr(tok, "column", None),
        )

    def aug_assign(self, items: list) -> AugAssignStatement:
        tok = items[0]
        return AugAssignStatement(
            name=str(tok),
            op=str(items[1])[0],
            value=items[2],
            line=getattr(tok, "line", None),
            col=getattr(tok, "column", None),
        )

    def expr_stmt(self, items: list) -> ExprStatement:
        return ExprStatement(expr=items[0])

    # ── expressions ────────────────────────────────────────────────────────

    def add(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="+", left=items[0], right=items[1])

    def sub(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="-", left=items[0], right=items[1])

    def mul(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="*", left=items[0], right=items[1])

    def div(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="/", left=items[0], right=items[1])

    def mod(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="%", left=items[0], right=items[1])

    def eq(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="==", left=items[0], right=items[1])

    def ne(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="!=", left=items[0], right=items[1])

    def lt(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="<", left=items[0], right=items[1])

    def le(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="<=", left=items[0], right=items[1])

    def gt(self, items: list) -> BinaryExpr:
        return BinaryExpr(op=">", left=items[0], right=items[1])

    def ge(self, items: list) -> BinaryExpr:
        return BinaryExpr(op=">=", left=items[0], right=items[1])

    def and_(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="and", left=items[0], right=items[1])

    def or_(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="or", left=items[0], right=items[1])

    def neg(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="neg", left=None, right=items[0])

    def not_(self, items: list) -> BinaryExpr:
        return BinaryExpr(op="not", left=None, right=items[0])

    def list_lit(self, items: list) -> ListLiteral:
        return ListLiteral(elements=[i for i in items if i is not None])

    def dict_entry(self, items: list) -> tuple:
        return (items[0], items[1])

    def dict_lit(self, items: list) -> DictLiteral:
        entries = [i for i in items if isinstance(i, tuple)]
        return DictLiteral(keys=[k for k, _ in entries], values=[v for _, v in entries])

    def index_expr(self, items: list) -> IndexExpr:
        return IndexExpr(obj=items[0], index=items[1])

    def func_call(self, items: list) -> CallExpr:
        callee = items[0]
        args = list(items[1]) if len(items) > 1 and items[1] is not None else []
        return CallExpr(callee=callee, args=args)

    def method_call(self, items: list) -> MethodCallExpr:
        obj = items[0]
        method = str(items[1])
        args = list(items[2]) if len(items) > 2 and items[2] is not None else []
        return MethodCallExpr(obj=obj, method=method, args=args)

    def arg_list(self, items: list) -> list:
        return items

    def var(self, items: list) -> VarExpr:
        tok = items[0]
        return VarExpr(
            name=str(tok),
            line=getattr(tok, "line", None),
            col=getattr(tok, "column", None),
        )

    # ── literals ───────────────────────────────────────────────────────────

    def int_lit(self, items: list) -> NumberLiteral:
        return NumberLiteral(value=int(items[0]))

    def float_lit(self, items: list) -> FloatLiteral:
        return FloatLiteral(value=float(items[0]))

    def string_lit(self, items: list) -> StringLiteral:
        raw = str(items[0])
        return StringLiteral(value=raw[1:-1])  # strip quotes

    def triple_string_lit(self, items: list) -> StringLiteral:
        raw = str(items[0])
        return StringLiteral(value=raw[3:-3])  # strip triple quotes

    def bool_lit(self, items: list) -> BoolLiteral:
        return BoolLiteral(value=str(items[0]) == "true")

    # ── types ──────────────────────────────────────────────────────────────

    def type_int(self, _: list) -> str:
        return "type_int"

    def type_float(self, _: list) -> str:
        return "type_float"

    def type_bool(self, _: list) -> str:
        return "type_bool"

    def type_string(self, _: list) -> str:
        return "type_string"

    def type_tensor(self, _: list) -> str:
        return "type_tensor"

    def type_matrix(self, _: list) -> str:
        return "type_matrix"

    # ── discard bare NEWLINEs that bubble up ───────────────────────────────

    def NEWLINE(self, token: Token) -> None:
        return None

    def transform(self, tree: Tree) -> Any:
        return super().transform(tree)
