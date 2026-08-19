import contextlib
import io
import os
import re
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Optional, Union

from neuva.parser import NeuvaParser
from neuva.parser.parser import ParseError
from neuva.interpreter.interpreter import NeuvaInterpreter, RuntimeError_
from neuva.typechecker import TypeChecker, TypeCheckError

# The three exception types the CLI formats for display. They aren't related by
# inheritance, but all duck-type `.line`/`.col`/`.hint`/`.raw_message` (accessed via
# `getattr(..., default)` in `format_message` below), which is what actually matters here.
CheckedError = Union[ParseError, TypeCheckError, RuntimeError_]

try:
    import colorama
    from colorama import Fore, Style

    colorama.init()
    _COLOR = True
except ImportError:  # pragma: no cover - colorama is a listed dependency
    _COLOR = False

    class _NoColor:
        def __getattr__(self, _name: str) -> str:
            return ""

    Fore = Style = _NoColor()

try:
    # pyreadline3 registers itself as the `readline` module on Windows; the stdlib
    # provides it natively on Linux/macOS. Either way, importing it is enough to get
    # up/down-arrow history in `input()` — no further wiring needed.
    import readline

    _READLINE = True
except ImportError:  # pragma: no cover - platform-dependent
    readline = None
    _READLINE = False

_HELP = """\
Usage: neuva <file.nva> [--no-check] [--strict]
       neuva shell
       neuva --version

  --no-check   Skip static type checking and run the program directly.
  --strict     Treat type-checker warnings as errors (non-zero exit on any warning)."""


def _get_version() -> str:
    # neuva/version.py is the single source of truth; fall back to pyproject.toml
    # (for editable/dev checkouts where it might drift) and finally to installed
    # package metadata so a plain `pip install`-from-sdist still reports something.
    try:
        from neuva.version import __version__

        return __version__
    except Exception:
        pass
    try:
        toml = (Path(__file__).parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        m = re.search(r'^version\s*=\s*"([^"]+)"', toml, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception:
        pass
    try:
        return version("neuva-lang")
    except PackageNotFoundError:
        return "0.3.0"


def format_message(exc: CheckedError, source: str, label: str = "Error") -> str:
    source_lines = source.splitlines()
    line = getattr(exc, "line", None)
    col = getattr(exc, "col", None)
    hint = getattr(exc, "hint", None)
    raw = getattr(exc, "raw_message", str(exc))

    if line is not None and col is not None:
        header = f"[Line {line}:{col}] {label}: {raw}"
    elif line is not None:
        header = f"[Line {line}] {label}: {raw}"
    else:
        header = f"{label}: {raw}"

    parts = [header]

    if line is not None and 1 <= line <= len(source_lines):
        indent = "           "
        parts.append(f"{indent}{source_lines[line - 1]}")
        if col is not None:
            parts.append(indent + " " * (col - 1) + "^")

    if hint:
        parts.append(f"           {hint}")

    return "\n".join(parts)


def format_error(exc: CheckedError, source: str) -> str:
    return format_message(exc, source, "Error")


def format_warning(exc: CheckedError, source: str) -> str:
    return format_message(exc, source, "Warning")


_REPL_KEYWORDS = (
    "let print model layer output train on for epochs lr loss lr_schedule "
    "early_stop lr_warmup save to predict fn return if else while in match "
    "case default import and or not true false int float bool string tensor matrix"
).split()

_REPL_BUILTINS = (
    "range len load accuracy predict predict_proba export_onnx normalize shuffle "
    "mse mae crossentropy binary_crossentropy precision recall f1_score "
    "confusion_matrix table plot upper lower split join strip replace "
    "abs sqrt pow log exp round min max sum mean"
).split()

_REPL_EXAMPLES = [
    ("Hello world", 'print "Hello, Neuva!"'),
    (
        "A tiny model",
        "model Net {\n"
        "    layer dense(4 -> 8, relu)\n"
        "    layer dense(8 -> 3, softmax)\n"
        "}\n"
        "print Net",
    ),
    (
        "Loops and math",
        "for i in range(0, 5) {\n" "    print i, sqrt(i)\n" "}",
    ),
]

_REPL_COMMANDS = (":help", ":clear", ":examples", ":reset", ":quit")


def _repl_help() -> str:
    lines = [
        "Neuva REPL commands:",
        "  :help       Show this help (keywords + built-in functions)",
        "  :clear      Clear the screen",
        "  :examples   Print 3 quick example programs",
        "  :reset      Reset the interpreter environment (clear all variables)",
        "  :quit       Exit the REPL (also: exit, quit, Ctrl-D)",
        "",
        "Keywords:",
        "  " + ", ".join(sorted(_REPL_KEYWORDS)),
        "",
        "Built-in functions:",
        "  " + ", ".join(sorted(_REPL_BUILTINS)),
    ]
    return "\n".join(lines)


def _repl_examples() -> str:
    parts = []
    for title, code in _REPL_EXAMPLES:
        parts.append(f"--- {title} ---")
        parts.append(code)
        parts.append("")
    return "\n".join(parts).rstrip()


def _colorize_output(text: str) -> str:
    """Wrap model-summary boxes in cyan and accuracy lines in green; everything else
    (plain `print` output) is left uncolored."""
    if not _COLOR or not text:
        return text
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("┌"):  # "┌" — start of a boxed model summary
            block = [line]
            i += 1
            while i < len(lines) and not lines[i].startswith("└"):  # "└"
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
                i += 1
            out.append(Fore.CYAN + "\n".join(block) + Style.RESET_ALL)
        elif "accuracy" in line.lower():
            out.append(Fore.GREEN + line + Style.RESET_ALL)
            i += 1
        else:
            out.append(line)
            i += 1
    return "\n".join(out)


def _read_statement(first_line: str) -> Optional[str]:
    """Collect `first_line` plus however many `... `-prompted continuation lines are
    needed for every `{` to find its matching `}`. Returns None on Ctrl-D/Ctrl-C mid
    statement (caller should treat that as end-of-session, matching plain input())."""
    lines = [first_line]
    depth = first_line.count("{") - first_line.count("}")
    while depth > 0:
        try:
            cont = input("... ")
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        lines.append(cont)
        depth += cont.count("{") - cont.count("}")
    return "\n".join(lines)


def shell() -> None:
    print(
        f"Neuva {_get_version()} — interactive shell. Type :help for commands, :quit to exit."
    )
    parser = NeuvaParser()
    interp = NeuvaInterpreter()

    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = line.strip()
        if not stripped:
            continue

        if stripped in ("exit", "quit", ":quit", ":exit"):
            break
        if stripped == ":help":
            print(_repl_help())
            continue
        if stripped == ":clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if stripped == ":examples":
            print(_repl_examples())
            continue
        if stripped == ":reset":
            interp = NeuvaInterpreter()
            print("Interpreter environment reset.")
            continue
        if stripped.startswith(":"):
            print(
                f"Unknown command '{stripped}'. Available: {', '.join(_REPL_COMMANDS)}"
            )
            continue

        source = _read_statement(line)
        if source is None:
            break

        try:
            ast = parser.parse(source)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                interp.visit(ast)
            output = buf.getvalue()
            if output:
                print(_colorize_output(output.rstrip("\n")))
        except ParseError as exc:
            print(Fore.RED + format_error(exc, source) + Style.RESET_ALL)
        except RuntimeError_ as exc:
            print(Fore.RED + format_error(exc, source) + Style.RESET_ALL)


def main() -> None:
    argv = sys.argv[1:]
    no_check = "--no-check" in argv
    strict = "--strict" in argv
    argv = [a for a in argv if a not in ("--no-check", "--strict")]

    if not argv:
        print(_HELP, file=sys.stderr)
        sys.exit(1)

    arg = argv[0]

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

    if not no_check:
        checker = TypeChecker()
        errors = checker.check(ast)
        warnings = checker.warnings
        if strict:
            errors = errors + warnings
            warnings = []
        if errors:
            for err in errors:
                print(format_error(err, source), file=sys.stderr)
            sys.exit(1)
        for w in warnings:
            print(format_warning(w, source), file=sys.stderr)
        print(f"Type check passed: {len(warnings)} warnings, 0 errors")

    try:
        NeuvaInterpreter().visit(ast)
    except RuntimeError_ as exc:
        print(format_error(exc, source), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
