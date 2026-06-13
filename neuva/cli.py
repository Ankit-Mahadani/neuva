import sys

from neuva.parser import NeuvaParser
from neuva.parser.parser import ParseError
from neuva.interpreter.interpreter import NeuvaInterpreter, RuntimeError_


def format_error(exc, source: str) -> str:
    source_lines = source.splitlines()
    line = getattr(exc, "line", None)
    col = getattr(exc, "col", None)
    hint = getattr(exc, "hint", None)
    raw = getattr(exc, "raw_message", str(exc))

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


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: neuva <file.nva>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]

    if not path.endswith(".nva"):
        print(f"warning: '{path}' does not have a .nva extension", file=sys.stderr)

    try:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"error: file not found: '{path}'", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"error: could not read '{path}': {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        ast = NeuvaParser().parse(source)
    except ParseError as exc:
        print(format_error(exc, source), file=sys.stderr)
        sys.exit(1)

    try:
        NeuvaInterpreter().visit(ast)
    except RuntimeError_ as exc:
        print(format_error(exc, source), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
