from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class Node:
    line: Optional[int] = field(default=None, repr=False)
    col: Optional[int] = field(default=None, repr=False)


@dataclass
class NumberLiteral(Node):
    value: int = 0


@dataclass
class FloatLiteral(Node):
    value: float = 0.0


@dataclass
class StringLiteral(Node):
    value: str = ""


@dataclass
class BoolLiteral(Node):
    value: bool = False


@dataclass
class VarExpr(Node):
    name: str = ""


@dataclass
class BinaryExpr(Node):
    op: str = ""
    left: Any = None
    right: Any = None


@dataclass
class CallExpr(Node):
    callee: Any = None
    args: List[Any] = field(default_factory=list)


@dataclass
class MethodCallExpr(Node):
    obj: Any = None
    method: str = ""
    args: List[Any] = field(default_factory=list)


@dataclass
class LetStatement(Node):
    names: List[str] = field(default_factory=list)
    type_ann: Optional[str] = None
    value: Any = None


@dataclass
class PrintStatement(Node):
    exprs: List[Any] = field(default_factory=list)


@dataclass
class LayerStatement(Node):
    name: str = ""
    args: List[Any] = field(default_factory=list)


@dataclass
class ModelStatement(Node):
    name: str = ""
    layers: List[LayerStatement] = field(default_factory=list)


@dataclass
class TrainOption(Node):
    key: str = ""
    value: Any = None


@dataclass
class TrainStatement(Node):
    model: str = ""
    data: str = ""
    epochs: Any = None
    options: List[TrainOption] = field(default_factory=list)


@dataclass
class SaveStatement(Node):
    model: str = ""
    path: Any = None


@dataclass
class PredictStatement(Node):
    model: str = ""
    data: str = ""


@dataclass
class Parameter(Node):
    name: str = ""
    type_ann: str = ""


@dataclass
class ReturnStatement(Node):
    value: Any = None


@dataclass
class IfStatement(Node):
    condition: Any = None
    then_body: List[Any] = field(default_factory=list)
    elif_branches: List[Any] = field(default_factory=list)  # list of (condition, body) tuples
    else_branch: List[Any] = field(default_factory=list)


@dataclass
class ForStatement(Node):
    var: str = ""
    iterable: Any = None
    body: List[Any] = field(default_factory=list)


@dataclass
class WhileStatement(Node):
    condition: Any = None
    body: List[Any] = field(default_factory=list)


@dataclass
class FnStatement(Node):
    name: str = ""
    params: List[Parameter] = field(default_factory=list)
    return_type: Optional[str] = None
    body: List[Any] = field(default_factory=list)


@dataclass
class ExprStatement(Node):
    expr: Any = None


@dataclass
class Program(Node):
    body: List[Any] = field(default_factory=list)
