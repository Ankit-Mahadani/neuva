import os
from lark import Lark, UnexpectedInput
from .ast_nodes import Program
from .transformer import NeuvaTransformer

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")

with open(_GRAMMAR_PATH, "r") as _f:
    _GRAMMAR = _f.read()


class ParseError(Exception):
    def __init__(self, message: str, line: int = None, column: int = None):
        self.raw_message = message.splitlines()[0] if message else message
        self.line = line
        self.col = column
        self.column = column
        self.hint = None
        super().__init__(message)

    def __str__(self) -> str:
        if self.line is not None and self.col is not None:
            return f"[Line {self.line}:{self.col}] Error: {self.raw_message}"
        if self.line is not None:
            return f"[Line {self.line}] Error: {self.raw_message}"
        return f"Error: {self.raw_message}"


class NeuvaParser:
    def __init__(self) -> None:
        self._parser = Lark(
            _GRAMMAR,
            parser="earley",
            propagate_positions=True,
            ambiguity="resolve",
        )
        self._transformer = NeuvaTransformer()

    def parse(self, source: str) -> Program:
        source = source.rstrip() + "\n"
        try:
            tree = self._parser.parse(source)
            return self._transformer.transform(tree)
        except UnexpectedInput as exc:
            raise ParseError(
                str(exc), getattr(exc, "line", None), getattr(exc, "column", None)
            ) from exc

    def parse_file(self, path: str) -> Program:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        return self.parse(source)
