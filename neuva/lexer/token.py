from enum import Enum, auto
from dataclasses import dataclass

class TokenType(Enum):
    INT=auto(); FLOAT=auto(); STRING=auto(); BOOL=auto(); IDENTIFIER=auto()
    MODEL=auto(); LAYER=auto(); TRAIN=auto(); LOAD=auto(); PREDICT=auto()
    SAVE=auto(); PRINT=auto(); LET=auto(); FN=auto(); RETURN=auto()
    IF=auto(); ELSE=auto(); FOR=auto(); WHILE=auto(); ON=auto(); TO=auto()
    EPOCHS=auto(); LR=auto(); LOSS=auto(); TRUE=auto(); FALSE=auto()
    TYPE_INT=auto(); TYPE_FLOAT=auto(); TYPE_BOOL=auto()
    TYPE_STRING=auto(); TYPE_TENSOR=auto(); TYPE_MATRIX=auto()
    PLUS=auto(); MINUS=auto(); STAR=auto(); SLASH=auto()
    ARROW=auto(); EQUALS=auto(); EQ=auto(); NEQ=auto()
    LT=auto(); GT=auto(); LTE=auto(); GTE=auto()
    LPAREN=auto(); RPAREN=auto(); LBRACE=auto(); RBRACE=auto()
    COMMA=auto(); COLON=auto(); DOT=auto()
    NEWLINE=auto(); EOF=auto(); COMMENT=auto()

@dataclass
class Token:
    type: TokenType
    value: object
    line: int
    col: int
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"
