import re
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from neuva.parser import NeuvaParser
from neuva.parser.parser import ParseError
from neuva.interpreter.interpreter import NeuvaInterpreter, RuntimeError_

_HELP = """\
Usage: neuva <file.nva>
       neuva shell
       neuva --version"""


def _get_version() -> str:
    # Prefer pyproject.toml so development builds always show the right version.
    try:
        toml = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', toml, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception:
        pass
    try:
        return version("neuva-lang")
    except PackageNotFoundError:
        return "0.3.0"


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


def shell() -> None:
    print(f"Neuva {_get_version()} — interactive shell. Type 'exit' to quit.")
    parser = NeuvaParser()
    interp = NeuvaInterpreter()
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        stripped = line.strip()
        if stripped in ("exit", "quit"):
            break
        if not stripped:
            continue
        try:
            ast = parser.parse(line)
            interp.visit(ast)
        except ParseError as exc:
            print(format_error(exc, line))
        except RuntimeError_ as exc:
            print(format_error(exc, line))


def main() -> None:
    if len(sys.argv) < 2:
        print(_HELP, file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]

    if arg in ("--version", "-v"):
        print(f"Neuva {_get_version()}")
        sys.exit(0)

    if arg in ("--help", "-h"):
        print(_HELP)
        sys.exit(0)

    if arg == "shell":
        shell()
        return

    path = arg

    if not path.endswith(".nva"):
        print(f"warning: '{path}' does not have a .nva extension", file=sys.stderr)

    try:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"Error: file '{path}' not found", file=sys.stderr)
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
