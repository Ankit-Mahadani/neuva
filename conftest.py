import sys
import os

# Allow `from src.lexer import ...` (used by test_lexer.py)
sys.path.insert(0, os.path.dirname(__file__))
# Allow `from parser import ...` and `from interpreter import ...`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
