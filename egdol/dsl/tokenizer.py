"""
DSL Tokenizer for Egdol Interactive Assistant.
Tokenizes English-like DSL syntax into tokens for parsing.
"""

import re
from typing import List, Optional, Tuple
from enum import Enum, auto


class TokenType(Enum):
    # Keywords
    IS = auto()
    ARE = auto()
    A = auto()
    AN = auto()
    THE = auto()
    WHO = auto()
    WHAT = auto()
    WHERE = auto()
    WHEN = auto()
    WHY = auto()
    HOW = auto()
    DOES = auto()
    DO = auto()
    HAVE = auto()
    HAS = auto()
    CAN = auto()
    WILL = auto()
    SHOW = auto()
    ME = auto()
    ALL = auto()
    SOME = auto()
    NO = auto()
    NOT = auto()
    IF = auto()
    THEN = auto()
    AND = auto()
    OR = auto()
    BUT = auto()
    SO = auto()
    BECAUSE = auto()
    
    # Pronouns
    HE = auto()
    SHE = auto()
    IT = auto()
    THEY = auto()
    HIM = auto()
    HER = auto()
    THEM = auto()
    HIS = auto()
    HERS = auto()
    THEIR = auto()
    
    # Command keywords
    FACTS = auto()
    RULES = auto()
    RESET = auto()
    SAVE = auto()
    LOAD = auto()
    TRACE = auto()
    HELP = auto()
    QUIT = auto()
    EXIT = auto()
    
    # Literals
    IDENTIFIER = auto()
    PROPER_NOUN = auto()
    COMMON_NOUN = auto()
    ADJECTIVE = auto()
    NUMBER = auto()
    STRING = auto()
    VARIABLE = auto()
    
    # Punctuation
    PERIOD = auto()
    QUESTION = auto()
    EXCLAMATION = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    
    # Special
    EOF = auto()
    NEWLINE = auto()
    WHITESPACE = auto()


class Token:
    def __init__(self, type_: TokenType, value: str, line: int = 1, column: int = 1):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column
        
    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}', {self.line}:{self.column})"
        
    def __eq__(self, other):
        if isinstance(other, TokenType):
            return self.type == other
        return isinstance(other, Token) and self.type == other.type and self.value == other.value


class DSLTokenizer:
    """Tokenizer for the Egdol DSL."""
    
    def __init__(self):
        # Define keyword mappings
        self.keywords = {
            'is': TokenType.IS,
            'are': TokenType.ARE,
            'a': TokenType.A,
            'an': TokenType.AN,
            'the': TokenType.THE,
            'who': TokenType.WHO,
            'what': TokenType.WHAT,
            'where': TokenType.WHERE,
            'when': TokenType.WHEN,
            'why': TokenType.WHY,
            'how': TokenType.HOW,
            'does': TokenType.DOES,
            'do': TokenType.DO,
            'have': TokenType.HAVE,
            'has': TokenType.HAS,
            'can': TokenType.CAN,
            'will': TokenType.WILL,
            'show': TokenType.SHOW,
            'me': TokenType.ME,
            'all': TokenType.ALL,
            'some': TokenType.SOME,
            'no': TokenType.NO,
            'not': TokenType.NOT,
            'if': TokenType.IF,
            'then': TokenType.THEN,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'but': TokenType.BUT,
            'so': TokenType.SO,
            'because': TokenType.BECAUSE,
            
            # Pronouns
            'he': TokenType.HE,
            'she': TokenType.SHE,
            'it': TokenType.IT,
            'they': TokenType.THEY,
            'him': TokenType.HIM,
            'her': TokenType.HER,
            'them': TokenType.THEM,
            'his': TokenType.HIS,
            'hers': TokenType.HERS,
            'their': TokenType.THEIR,
        }
        
        # Command keywords (prefixed with :)
        self.commands = {
            'facts': TokenType.FACTS,
            'rules': TokenType.RULES,
            'reset': TokenType.RESET,
            'save': TokenType.SAVE,
            'load': TokenType.LOAD,
            'trace': TokenType.TRACE,
            'help': TokenType.HELP,
            'quit': TokenType.QUIT,
            'exit': TokenType.EXIT,
        }
        
        # Regex patterns
        self.patterns = [
            (r'\d+', TokenType.NUMBER),
            (r'"[^"]*"', TokenType.STRING),
            (r"'[^']*'", TokenType.STRING),
            (r'[A-Z][a-z]+', TokenType.PROPER_NOUN),  # Proper nouns (capitalized)
            (r'[A-Z]', TokenType.VARIABLE),  # Single uppercase letters are variables
            (r'[a-z]+', TokenType.IDENTIFIER),  # Lowercase identifiers
            (r'\.', TokenType.PERIOD),
            (r'\?', TokenType.QUESTION),
            (r'!', TokenType.EXCLAMATION),
            (r',', TokenType.COMMA),
            (r':', TokenType.COLON),
            (r';', TokenType.SEMICOLON),
            (r'\n', TokenType.NEWLINE),
            (r'\s+', TokenType.WHITESPACE),
        ]
        
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize input text into a list of tokens."""
        tokens = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_tokens = self._tokenize_line(line, line_num)
            tokens.extend(line_tokens)
            
        # Add EOF token
        tokens.append(Token(TokenType.EOF, '', len(lines), 1))
        
        return tokens
        
    def _tokenize_line(self, line: str, line_num: int) -> List[Token]:
        """Tokenize a single line."""
        tokens = []
        pos = 0
        
        while pos < len(line):
            # Skip whitespace
            if line[pos].isspace():
                pos += 1
                continue
                
            # Try to match patterns
            matched = False
            for pattern, token_type in self.patterns:
                match = re.match(pattern, line[pos:])
                if match:
                    value = match.group(0)
                    
                    # Handle special cases
                    if token_type == TokenType.IDENTIFIER:
                        # Check if it's a keyword
                        if value.lower() in self.keywords:
                            token_type = self.keywords[value.lower()]
                        # Check if it's a command (after colon)
                        elif value.lower() in self.commands and tokens and tokens[-1].type == TokenType.COLON:
                            token_type = self.commands[value.lower()]
                    
                    # Skip whitespace tokens
                    if token_type != TokenType.WHITESPACE:
                        token = Token(token_type, value, line_num, pos + 1)
                        tokens.append(token)
                    
                    pos += len(value)
                    matched = True
                    break
                    
            if not matched:
                # Unknown character
                raise SyntaxError(f"Unknown character '{line[pos]}' at line {line_num}, column {pos + 1}")
                
        return tokens
        
    def is_proper_noun(self, word: str) -> bool:
        """Check if a word is a proper noun (capitalized)."""
        return word and word[0].isupper() and word[1:].islower()
        
    def is_variable(self, word: str) -> bool:
        """Check if a word is a variable (single uppercase letter)."""
        return len(word) == 1 and word.isupper()
        
    def is_common_noun(self, word: str) -> bool:
        """Check if a word is a common noun (lowercase)."""
        return word and word.islower() and word not in self.keywords


class TokenStream:
    """Wrapper for token list with position tracking."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        
    def peek(self, offset: int = 0) -> Optional[Token]:
        """Peek at token at current position + offset."""
        index = self.position + offset
        if 0 <= index < len(self.tokens):
            return self.tokens[index]
        return None
        
    def consume(self) -> Optional[Token]:
        """Consume and return current token, advance position."""
        if self.position < len(self.tokens):
            token = self.tokens[self.position]
            self.position += 1
            return token
        return None
        
    def expect(self, expected_type: TokenType) -> Token:
        """Expect a specific token type, raise error if not found."""
        token = self.consume()
        if not token or token.type != expected_type:
            expected = expected_type.name
            actual = token.type.name if token else "EOF"
            raise SyntaxError(f"Expected {expected}, got {actual} at line {token.line if token else 'EOF'}")
        return token
        
    def match(self, token_type: TokenType) -> bool:
        """Check if current token matches type without consuming."""
        token = self.peek()
        return token and token.type == token_type
        
    def is_at_end(self) -> bool:
        """Check if we're at the end of the token stream."""
        return self.position >= len(self.tokens) or self.peek().type == TokenType.EOF
        
    def __len__(self):
        return len(self.tokens)
        
    def __getitem__(self, index):
        return self.tokens[index]
