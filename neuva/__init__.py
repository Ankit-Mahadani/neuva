"""Neuva — A simple ML programming language."""
import sys as _sys

# Neuva prints Unicode box-drawing/chart characters (model summaries, tables, plots).
# Default Windows console codepages (e.g. cp1252) can't encode them, so force UTF-8
# on stdout/stderr with a safe fallback instead of crashing mid-run.
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

__version__ = "0.1.0"
