# Neuva Language — VS Code Extension

Syntax highlighting for `.nva` files in Visual Studio Code.

## What it highlights

- **Keywords** — `model`, `layer`, `train`, `save`, `load`, `predict`, `print`, `let`, `fn`, `return`, `if`, `else`, `for`, `while`, `on`, `to`, `epochs`, `accuracy`
- **Control flow** — `if`, `else`, `for`, `while`, `return` (distinct colour)
- **Types** — `int`, `float`, `bool`, `string`, `tensor`, `matrix`
- **Layer types & activations** — `dense`, `conv`, `pool`, `flatten`, `relu`, `sigmoid`, `softmax`, `tanh`, `linear`
- **Strings** — double-quoted, with escape sequences
- **Numbers** — integers and floats
- **Comments** — `#` to end of line
- **Arrow operator** — `->`
- **Operators** — `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `=`, `!`

## Install — manual (development)

1. Copy this folder into your VS Code extensions directory:

   **macOS / Linux**
   ```bash
   cp -r vscode-extension ~/.vscode/extensions/neuva-syntax
   ```

   **Windows**
   ```powershell
   Copy-Item -Recurse vscode-extension "$env:USERPROFILE\.vscode\extensions\neuva-syntax"
   ```

2. Restart VS Code (or run **Developer: Reload Window** from the command palette).

3. Open any `.nva` file — syntax highlighting activates automatically.

## Install — via vsce (package and install)

If you have the VS Code Extension CLI installed:

```bash
npm install -g @vscode/vsce
cd vscode-extension
vsce package          # produces neuva-syntax-0.1.0.vsix
code --install-extension neuva-syntax-0.1.0.vsix
```

## File structure

```
vscode-extension/
├── package.json                  # extension manifest
├── language-configuration.json   # comment/bracket config
├── syntaxes/
│   └── neuva.tmLanguage.json     # TextMate grammar
└── README.md                     # this file
```
