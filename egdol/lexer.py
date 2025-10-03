"""Lexer for egdol mini-DSL.

Token types:
- IDENT: identifiers and names
- LPAREN, RPAREN, COMMA, DOT, COLON
- ARROW => (for rules)
- QUERY ?
- VAR: capitalized identifiers (variables)
"""
from dataclasses import dataclass
import re
from typing import List, Tuple


@dataclass(frozen=True)
class Token:
    type: str
    value: str
    pos: int
    line: int
    col: int


class LexerError(Exception):
    pass


class Lexer:
    token_specification: List[Tuple[str, str]] = [
    ("RANGE", r'\.\.'),
    ("NEQ", r'\\=='),
    ("SEMI", r';'),
    ("ARROW", r'=>'),
    ("QUERY", r'\?'),
    ("COLON", r':'),
    ("CUT", r'!'),
    ("DOT", r'\.'),
    ("COMMA", r','),
    ("LPAREN", r'\('),
    ("RPAREN", r'\)'),
    ("LBRACK", r'\['),
    ("RBRACK", r'\]'),
    ("LBRACE", r'\{'),
    ("RBRACE", r'\}'),
    ("BAR", r'\|'),
    ("STRING", r'"([^"\\]|\\.)*"'),
    ("NUMBER", r'\d+'),
    ("PLUS", r'\+'),
    ("MINUS", r'-'),
    ("STAR", r'\*'),
    ("SLASH", r'/'),
    ("LE", r'<='),
    ("GE", r'>='),
    ("LT", r'<'),
    ("GT", r'>'),
    ("EQ", r'='),
    ("IDENT", r'[a-z_][a-zA-Z0-9_]*'),
    ("VAR", r'[A-Z][a-zA-Z0-9_]*'),
        ("SKIP", r'[ \t]+'),
        ("NEWLINE", r'\n'),
        ("MISMATCH", r'.'),
    ]

    def __init__(self, text: str):
        # strip full-line comments starting with # or % and inline comments
        # keep original text for token positions by replacing comment content with spaces
        import re as _re
        # remove occurrences of # or % to end-of-line
        self.text = _re.sub(r'(?m)(#|%).*$', '', text)
        parts = []
        for name, pattern in self.token_specification:
            parts.append(f"(?P<{name}>{pattern})")
        self.regex = re.compile('|'.join(parts))

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        line_start = 0
        line_no = 1
        for mo in self.regex.finditer(self.text):
            kind = mo.lastgroup
            value = mo.group()
            pos = mo.start()
            # compute line/col
            # update line_no and line_start when seeing newlines
            if kind == 'NEWLINE':
                line_no += 1
                line_start = mo.end()
                continue
            elif kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                col = pos - line_start + 1
                raise LexerError(f'Unexpected character {value!r} at line {line_no} col {col}')
            else:
                col = pos - line_start + 1
                tokens.append(Token(kind, value, pos, line_no, col))
        return tokens


if __name__ == '__main__':
    sample = "fact: human(socrates).\n? human(socrates).\n"
    print(Lexer(sample).tokenize())
