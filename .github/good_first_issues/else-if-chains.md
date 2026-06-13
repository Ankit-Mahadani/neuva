# Add support for else-if chains

**Labels:** `good first issue`, `enhancement`, `parser`

## Description

Neuva currently supports basic `if`/`else` branching, but chaining multiple conditions
with `else if` is not possible. Writers have to nest `if` blocks inside `else` bodies,
which is awkward and adds indentation for every extra branch.

## Current behavior

```nva
let score = 85

if score >= 90 {
    print "A"
} else {
    if score >= 80 {
        print "B"
    } else {
        if score >= 70 {
            print "C"
        } else {
            print "F"
        }
    }
}
```

Attempting `else if` as a keyword sequence produces a parse error.

## Desired behavior

```nva
let score = 85

if score >= 90 {
    print "A"
} else if score >= 80 {
    print "B"
} else if score >= 70 {
    print "C"
} else {
    print "F"
}
```

## Implementation notes

- The grammar rule for `if_stmt` in `neuva/parser/grammar.lark` currently allows one
  optional `else` body. It should be extended to allow `else if expr if_body` branches
  before the final optional `else`.
- The `IfStatement` AST node in `neuva/parser/ast_nodes.py` may need an `elif_branches`
  field (list of `(condition, body)` pairs), or `else_body` can be allowed to contain a
  single nested `IfStatement` — the latter requires zero AST changes.
- `visit_IfStatement` in `neuva/interpreter/interpreter.py` must walk the chain.

## Files to change

- `neuva/parser/grammar.lark`
- `neuva/parser/transformer.py`
- `neuva/parser/ast_nodes.py` (possibly)
- `neuva/interpreter/interpreter.py`
