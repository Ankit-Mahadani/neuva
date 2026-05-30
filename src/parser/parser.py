import os
from lark import Lark, UnexpectedInput
from .transformer import NeuvaTransformer

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")

with open(_GRAMMAR_PATH, "r") as _f:
    _GRAMMAR = _f.read()


class ParseError(Exception):
    def __init__(self, message: str, line: int = None, column: int = None):
        self.line = line
        self.column = column
        loc = f" (line {line}, col {column})" if line is not None else ""
        super().__init__(f"ParseError{loc}: {message}")


class NeuvaParser:
    def __init__(self):
        self._parser = Lark(
            _GRAMMAR,
            parser="earley",
            propagate_positions=True,
            ambiguity="resolve",
        )
        self._transformer = NeuvaTransformer()

    def parse(self, source: str):
        source = source.rstrip() + "\n"
        try:
            tree = self._parser.parse(source)
            return self._transformer.transform(tree)
        except UnexpectedInput as exc:
            raise ParseError(str(exc), getattr(exc, "line", None), getattr(exc, "column", None)) from exc

    def parse_file(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        return self.parse(source)
