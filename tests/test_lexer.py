import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.lexer import Lexer
from src.lexer.token import TokenType

def tok(code): return Lexer(code).tokenize()
def types(code): return [t.type for t in tok(code) if t.type not in (TokenType.NEWLINE, TokenType.EOF)]

def test_let():
    assert TokenType.LET in types("let x = 42")
    assert TokenType.INT in types("let x = 42")

def test_string():
    t = tok('"Hello Neuva"')[0]
    assert t.type == TokenType.STRING and t.value == "Hello Neuva"

def test_float():
    t = tok("3.14")[0]
    assert t.type == TokenType.FLOAT and t.value == 3.14

def test_keywords():
    ts = types("model layer train load predict save")
    for kw in [TokenType.MODEL,TokenType.LAYER,TokenType.TRAIN,
               TokenType.LOAD,TokenType.PREDICT,TokenType.SAVE]:
        assert kw in ts

def test_arrow():
    assert TokenType.ARROW in types("784 -> 128")

def test_comment():
    ts = [t.type for t in tok("# comment\nlet x = 1")]
    assert TokenType.COMMENT not in ts
    assert TokenType.LET in ts

def test_model_block():
    code = 'model MyNet {\n    layer dense(784 -> 128, relu)\n}'
    ts = [t.type for t in tok(code)]
    assert TokenType.MODEL in ts
    assert TokenType.LAYER in ts
    assert TokenType.LBRACE in ts
    assert TokenType.RBRACE in ts

if __name__ == "__main__":
    test_let(); test_string(); test_float()
    test_keywords(); test_arrow(); test_comment(); test_model_block()
    print("All 7 tests passed!")
