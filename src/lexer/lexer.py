"""Neuva Lexer — converts source code into tokens."""
from .token import Token, TokenType

KEYWORD_MAP = {
    "model":TokenType.MODEL,"layer":TokenType.LAYER,"train":TokenType.TRAIN,
    "load":TokenType.LOAD,"predict":TokenType.PREDICT,"save":TokenType.SAVE,
    "print":TokenType.PRINT,"let":TokenType.LET,"fn":TokenType.FN,
    "return":TokenType.RETURN,"if":TokenType.IF,"else":TokenType.ELSE,
    "for":TokenType.FOR,"while":TokenType.WHILE,"on":TokenType.ON,
    "to":TokenType.TO,"epochs":TokenType.EPOCHS,"lr":TokenType.LR,
    "loss":TokenType.LOSS,"true":TokenType.TRUE,"false":TokenType.FALSE,
    "int":TokenType.TYPE_INT,"float":TokenType.TYPE_FLOAT,
    "bool":TokenType.TYPE_BOOL,"string":TokenType.TYPE_STRING,
    "tensor":TokenType.TYPE_TENSOR,"matrix":TokenType.TYPE_MATRIX,
}

class LexerError(Exception):
    def __init__(self, msg, line, col):
        super().__init__(f"[Line {line}:{col}] LexerError: {msg}")

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens = []

    def peek(self, offset=0):
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else None

    def advance(self):
        ch = self.source[self.pos]; self.pos += 1
        if ch == "\n": self.line += 1; self.col = 1
        else: self.col += 1
        return ch

    def match(self, expected):
        if self.peek() == expected: self.advance(); return True
        return False

    def skip_whitespace(self):
        while self.peek() in (" ", "\t", "\r"): self.advance()

    def read_string(self):
        line, col = self.line, self.col
        self.advance()
        result = []
        while self.peek() not in ('"', None): result.append(self.advance())
        if self.peek() is None: raise LexerError("Unterminated string", line, col)
        self.advance()
        return Token(TokenType.STRING, "".join(result), line, col)

    def read_number(self):
        line, col = self.line, self.col
        result = []; is_float = False
        while self.peek() and (self.peek().isdigit() or self.peek() == "."):
            if self.peek() == ".":
                if is_float: break
                is_float = True
            result.append(self.advance())
        raw = "".join(result)
        return Token(TokenType.FLOAT if is_float else TokenType.INT,
                     float(raw) if is_float else int(raw), line, col)

    def read_identifier(self):
        line, col = self.line, self.col
        result = []
        while self.peek() and (self.peek().isalnum() or self.peek() == "_"):
            result.append(self.advance())
        word = "".join(result)
        ttype = KEYWORD_MAP.get(word, TokenType.IDENTIFIER)
        return Token(ttype, word, line, col)

    def tokenize(self) -> list:
        SYM = {"+":TokenType.PLUS,"-":TokenType.MINUS,"*":TokenType.STAR,
               "/":TokenType.SLASH,"(":TokenType.LPAREN,")":TokenType.RPAREN,
               "{":TokenType.LBRACE,"}":TokenType.RBRACE,",":TokenType.COMMA,
               ":":TokenType.COLON,".":TokenType.DOT}
        while self.pos < len(self.source):
            self.skip_whitespace()
            ch = self.peek()
            if ch is None: break
            elif ch == "\n":
                self.tokens.append(Token(TokenType.NEWLINE,"\n",self.line,self.col))
                self.advance()
            elif ch == "#":
                while self.peek() and self.peek() != "\n": self.advance()
            elif ch == '"': self.tokens.append(self.read_string())
            elif ch.isdigit(): self.tokens.append(self.read_number())
            elif ch.isalpha() or ch == "_": self.tokens.append(self.read_identifier())
            elif ch == "-" and self.peek(1) == ">":
                line,col=self.line,self.col; self.advance(); self.advance()
                self.tokens.append(Token(TokenType.ARROW,"->",line,col))
            elif ch == "=":
                line,col=self.line,self.col; self.advance()
                if self.match("="): self.tokens.append(Token(TokenType.EQ,"==",line,col))
                else: self.tokens.append(Token(TokenType.EQUALS,"=",line,col))
            elif ch in SYM:
                line,col=self.line,self.col; self.advance()
                self.tokens.append(Token(SYM[ch],ch,line,col))
            else:
                raise LexerError(f"Unexpected character '{ch}'",self.line,self.col)
        self.tokens.append(Token(TokenType.EOF,None,self.line,self.col))
        return self.tokens
