from neuva.parser import NeuvaParser


def test_print_statement():
    parser = NeuvaParser()
    tree = parser.parse('print("hello")')
    assert tree is not None


def test_let_statement():
    parser = NeuvaParser()
    tree = parser.parse("let x = 42")
    assert tree is not None


def test_typed_let_statement():
    parser = NeuvaParser()
    tree = parser.parse("let x: int = 10")
    assert tree is not None


def test_model_with_layers():
    parser = NeuvaParser()
    src = (
        "model MyNet {\n"
        "    layer dense(784 -> 128, relu)\n"
        "    layer dropout(128 -> 64, tanh)\n"
        "}\n"
    )
    tree = parser.parse(src)
    assert tree is not None


def test_train_statement():
    parser = NeuvaParser()
    tree = parser.parse("train MyNet on data for 10 epochs, lr = 0.01, loss = crossentropy")
    assert tree is not None


def test_save_statement():
    parser = NeuvaParser()
    tree = parser.parse('save MyNet to "model.bin"')
    assert tree is not None


def test_function_definition():
    parser = NeuvaParser()
    src = (
        "fn add(a: int, b: int) -> int {\n"
        "    return a + b\n"
        "}\n"
    )
    tree = parser.parse(src)
    assert tree is not None


def test_if_statement():
    parser = NeuvaParser()
    src = (
        "if x > 0 {\n"
        '    print("positive")\n'
        "} else {\n"
        '    print("non-positive")\n'
        "}\n"
    )
    tree = parser.parse(src)
    assert tree is not None


def test_list_literal():
    parser = NeuvaParser()
    tree = parser.parse("let scores = [85, 90, 78, 92]")
    assert tree is not None


def test_empty_list():
    parser = NeuvaParser()
    tree = parser.parse("let empty = []")
    assert tree is not None


def test_list_index():
    parser = NeuvaParser()
    tree = parser.parse("let x = scores[0]")
    assert tree is not None


def test_and_operator():
    parser = NeuvaParser()
    src = "if x > 0 and x < 100 {\n    print \"valid\"\n}\n"
    tree = parser.parse(src)
    assert tree is not None


def test_or_operator():
    parser = NeuvaParser()
    src = "if x > 100 or x < 0 {\n    print \"out of range\"\n}\n"
    tree = parser.parse(src)
    assert tree is not None


def test_dropout_layer():
    parser = NeuvaParser()
    src = "model Net {\n    layer dense(10 -> 8, relu)\n    layer dropout(0.3)\n    layer dense(8 -> 2, softmax)\n}\n"
    tree = parser.parse(src)
    assert tree is not None


if __name__ == "__main__":
    tests = [
        test_print_statement,
        test_let_statement,
        test_typed_let_statement,
        test_model_with_layers,
        test_train_statement,
        test_save_statement,
        test_function_definition,
        test_if_statement,
    ]
    for t in tests:
        t()
    print("All 8 parser tests passed!")
