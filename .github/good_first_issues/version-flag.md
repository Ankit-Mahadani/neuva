# Add a `neuva --version` command

**Labels:** `good first issue`, `enhancement`, `cli`

## Description

Running `neuva --version` (or `neuva -V`) currently prints the generic usage error
because `cli.py` only checks for a `.nva` file argument. There is no way for a user to
confirm which version of Neuva is installed without digging into `pip show neuva-lang`.
This is one of the first things people try after installing a new CLI tool.

## Current behavior

```
$ neuva --version
usage: neuva <file.nva>
```

## Desired behavior

```
$ neuva --version
neuva 0.2.0

$ neuva -V
neuva 0.2.0
```

## Implementation notes

The version string is already declared in `pyproject.toml`:

```toml
[project]
version = "0.2.0"
```

Python's `importlib.metadata` (stdlib since 3.8) can read it at runtime without
hardcoding the string in two places:

```python
from importlib.metadata import version, PackageNotFoundError

def _get_version() -> str:
    try:
        return version("neuva-lang")
    except PackageNotFoundError:
        return "unknown"
```

Then in `main()`, check for the flag before the `.nva` file argument:

```python
if len(sys.argv) == 2 and sys.argv[1] in ("--version", "-V"):
    print(f"neuva {_get_version()}")
    sys.exit(0)
```

No dependencies are added — `importlib.metadata` is part of the standard library.

## Files to change

- `neuva/cli.py`
