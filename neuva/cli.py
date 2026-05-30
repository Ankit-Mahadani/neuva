import sys

from neuva.parser import NeuvaParser
from neuva.parser.parser import ParseError
from neuva.interpreter.interpreter import NeuvaInterpreter, RuntimeError_


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
        print(f"parse error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        NeuvaInterpreter().visit(ast)
    except RuntimeError_ as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
