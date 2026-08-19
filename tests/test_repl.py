"""Tests for the REPL (`neuva shell`) added in the v1.1.0 completion pass: multi-line
`{`/`}` detection, `:` commands, and colorized output. Drives the REPL loop via a
monkeypatched `input()` rather than subprocess, for speed and to avoid platform console
encoding issues with the box-drawing characters in model summaries."""
import colorama
import pytest

from neuva import cli


def _feed(monkeypatch, lines):
    """Monkeypatch builtins.input to yield `lines` one at a time, then raise EOFError
    (mirroring what a real Ctrl-D / closed pipe does to input())."""
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError()

    monkeypatch.setattr("builtins.input", fake_input)


def test_read_statement_single_line():
    assert cli._read_statement('print "hi"') == 'print "hi"'


def test_read_statement_multiline_waits_for_matching_brace(monkeypatch):
    lines = iter(["    layer dense(4 -> 3, relu)", "}"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    result = cli._read_statement("model M {")
    assert result == "model M {\n    layer dense(4 -> 3, relu)\n}"


def test_read_statement_returns_none_on_eof_mid_statement(monkeypatch):
    def raise_eof(prompt=""):
        raise EOFError()
    monkeypatch.setattr("builtins.input", raise_eof)
    assert cli._read_statement("model M {") is None


def test_repl_help_lists_new_keywords_and_builtins():
    text = cli._repl_help()
    for kw in ("match", "case", "default", "not", "lr_warmup"):
        assert kw in text
    for fn in ("predict_proba", "export_onnx", "sqrt", "mean", "upper"):
        assert fn in text


def test_repl_examples_has_three_examples():
    text = cli._repl_examples()
    assert text.count("---") == 6  # 3 examples, opening + closing marker each


def test_colorize_wraps_model_box_in_cyan_and_accuracy_in_green():
    colorama.init(strip=False)
    text = "┌───┐\n│ Model: Net │\n└───┘\nAccuracy: 0.5\nplain line"
    out = cli._colorize_output(text)
    assert cli.Fore.CYAN in out
    assert cli.Fore.GREEN in out
    plain_line = out.split("\n")[-1]
    assert "plain line" in plain_line
    assert cli.Fore.CYAN not in plain_line and cli.Fore.GREEN not in plain_line


# ── full REPL loop, driven via monkeypatched input() ─────────────────────────

def test_repl_multiline_model_definition(capsys, monkeypatch):
    _feed(monkeypatch, [
        "model M {",
        "    layer dense(4 -> 3, relu)",
        "}",
        "print M",
        ":quit",
    ])
    cli.shell()
    out = capsys.readouterr().out
    assert "Model: M" in out
    assert "dense(4 -> 3, relu)" in out


def test_repl_help_command(capsys, monkeypatch):
    _feed(monkeypatch, [":help", ":quit"])
    cli.shell()
    out = capsys.readouterr().out
    assert "Neuva REPL commands" in out
    assert ":examples" in out


def test_repl_examples_command(capsys, monkeypatch):
    _feed(monkeypatch, [":examples", ":quit"])
    cli.shell()
    out = capsys.readouterr().out
    assert "Hello world" in out
    assert "A tiny model" in out
    assert "Loops and math" in out


def test_repl_reset_command_clears_variables(capsys, monkeypatch):
    _feed(monkeypatch, [
        "let x = 42",
        ":reset",
        "print x",
        ":quit",
    ])
    cli.shell()
    out = capsys.readouterr().out
    assert "Interpreter environment reset." in out
    assert "undefined variable 'x'" in out


def test_repl_quit_and_exit_aliases_both_end_session(capsys, monkeypatch):
    _feed(monkeypatch, ["print 1", "exit"])
    cli.shell()
    _feed(monkeypatch, ["print 1", "quit"])
    cli.shell()
    # both sessions terminated cleanly without raising


def test_repl_unknown_command_prints_hint(capsys, monkeypatch):
    _feed(monkeypatch, [":bogus", ":quit"])
    cli.shell()
    out = capsys.readouterr().out
    assert "Unknown command" in out


def test_repl_error_is_reported_not_fatal(capsys, monkeypatch):
    _feed(monkeypatch, ["undefined_name_here()", "print 1", ":quit"])
    cli.shell()
    out = capsys.readouterr().out
    assert "undefined variable 'undefined_name_here'" in out
    assert "1" in out  # the next statement still ran
