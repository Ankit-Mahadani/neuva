from neuva.parser import NeuvaParser
from neuva.parser.ast_nodes import (
    Program, LetStatement, PrintStatement, ModelStatement,
    TrainStatement, FnStatement, IfStatement,
)


def test_let_statement():
    parser = NeuvaParser()
    result = parser.parse("let x = 42")
    assert isinstance(result, Program)
    assert len(result.body) == 1
    node = result.body[0]
    assert isinstance(node, LetStatement)
    assert node.names == ["x"]


def test_model_with_layers():
    parser = NeuvaParser()
    src = (
        "model MyNet {\n"
        "    layer dense(784 -> 128, relu)\n"
        "    layer dropout(128 -> 64, tanh)\n"
        "}\n"
    )
    result = parser.parse(src)
    assert isinstance(result, Program)
    node = result.body[0]
    assert isinstance(node, ModelStatement)
    assert node.name == "MyNet"
    assert len(node.layers) == 2
    assert node.layers[0].name == "dense"
    assert node.layers[1].name == "dropout"


def test_train_statement():
    parser = NeuvaParser()
    result = parser.parse("train MyNet on data for 10 epochs, lr = 0.01, loss = crossentropy")
    assert isinstance(result, Program)
    node = result.body[0]
    assert isinstance(node, TrainStatement)
    assert node.model == "MyNet"
    assert node.epochs == 10
    assert len(node.options) == 2
    assert node.options[0].key == "lr"
    assert node.options[1].key == "loss"


def test_print_statement():
    parser = NeuvaParser()
    result = parser.parse('print("hello")')
    assert isinstance(result, Program)
    node = result.body[0]
    assert isinstance(node, PrintStatement)


def test_function_definition():
    parser = NeuvaParser()
    src = (
        "fn add(a: int, b: int) -> int {\n"
        "    return a + b\n"
        "}\n"
    )
    result = parser.parse(src)
    assert isinstance(result, Program)
    node = result.body[0]
    assert isinstance(node, FnStatement)
    assert node.name == "add"
    assert len(node.params) == 2
    assert node.params[0].name == "a"
    assert node.params[1].name == "b"


def test_if_statement():
    parser = NeuvaParser()
    src = (
        "if x > 0 {\n"
        '    print("positive")\n'
        "} else {\n"
        '    print("non-positive")\n'
        "}\n"
    )
    result = parser.parse(src)
    assert isinstance(result, Program)
    node = result.body[0]
    assert isinstance(node, IfStatement)
    assert node.condition is not None
    assert len(node.then_body) == 1
    assert len(node.else_branch) == 1


if __name__ == "__main__":
    tests = [
        test_let_statement,
        test_model_with_layers,
        test_train_statement,
        test_print_statement,
        test_function_definition,
        test_if_statement,
    ]
    for t in tests:
        t()
    print("All 6 AST tests passed!")
