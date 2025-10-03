"""
DSL 2.0 Lexer
Advanced tokenizer for natural language DSL with formal parsing.
"""

import re
from typing import List, Optional, Tuple
from enum import Enum, auto


class TokenType(Enum):
    """Token types for DSL 2.0."""
    # Keywords
    IF = auto()
    THEN = auto()
    UNLESS = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IS = auto()
    HAS = auto()
    IMPLIES = auto()
    CAUSES = auto()
    RELATED_TO = auto()
    PART_OF = auto()
    
    # Operators
    EQUALS = auto()
    NOT_EQUALS = auto()
    GREATER_THAN = auto()
    LESS_THAN = auto()
    GREATER_EQUAL = auto()
    LESS_EQUAL = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    
    # Logical operators
    AND_OP = auto()
    OR_OP = auto()
    NOT_OP = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMICOLON = auto()
    PERIOD = auto()
    QUESTION = auto()
    EXCLAMATION = auto()
    COLON = auto()
    
    # Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    
    # Special
    VARIABLE = auto()
    WILDCARD = auto()
    EOF = auto()
    NEWLINE = auto()
    WHITESPACE = auto()


class Token:
    """A token in the DSL 2.0 lexer."""
    
    def __init__(self, type_: TokenType, value: str, line: int = 1, column: int = 1):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column
        
    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}', {self.line}:{self.column})"
        
    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return (self.type == other.type and 
                self.value == other.value and 
                self.line == other.line and 
                self.column == other.column)


class DSL2Lexer:
    """Advanced lexer for DSL 2.0."""
    
    def __init__(self):
        self.keywords = {
            'if': TokenType.IF,
            'then': TokenType.THEN,
            'unless': TokenType.UNLESS,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            'is': TokenType.IS,
            'has': TokenType.HAS,
            'implies': TokenType.IMPLIES,
            'causes': TokenType.CAUSES,
            'related_to': TokenType.RELATED_TO,
            'part_of': TokenType.PART_OF,
            'true': TokenType.BOOLEAN,
            'false': TokenType.BOOLEAN
        }
        
        self.operators = {
            '==': TokenType.EQUALS,
            '!=': TokenType.NOT_EQUALS,
            '>': TokenType.GREATER_THAN,
            '<': TokenType.LESS_THAN,
            '>=': TokenType.GREATER_EQUAL,
            '<=': TokenType.LESS_EQUAL,
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '%': TokenType.MODULO,
            '**': TokenType.POWER,
            '&&': TokenType.AND_OP,
            '||': TokenType.OR_OP,
            '!': TokenType.NOT_OP
        }
        
        self.delimiters = {
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            ',': TokenType.COMMA,
            ';': TokenType.SEMICOLON,
            '.': TokenType.PERIOD,
            '?': TokenType.QUESTION,
            '!': TokenType.EXCLAMATION,
            ':': TokenType.COLON
        }
        
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize the input text."""
        tokens = []
        position = 0
        line = 1
        column = 1
        
        while position < len(text):
            char = text[position]
            
            # Handle whitespace
            if char.isspace():
                if char == '\n':
                    line += 1
                    column = 1
                else:
                    column += 1
                position += 1
                continue
                
            # Handle comments
            if char == '#' and position + 1 < len(text) and text[position + 1] == '#':
                # Single line comment
                while position < len(text) and text[position] != '\n':
                    position += 1
                continue
                
            # Handle multi-line comments
            if char == '/' and position + 1 < len(text) and text[position + 1] == '*':
                position += 2
                while position < len(text) - 1:
                    if text[position] == '*' and text[position + 1] == '/':
                        position += 2
                        break
                    position += 1
                continue
                
            # Handle strings
            if char in ['"', "'"]:
                quote = char
                position += 1
                column += 1
                string_value = ""
                
                while position < len(text):
                    if text[position] == quote:
                        position += 1
                        column += 1
                        break
                    elif text[position] == '\\' and position + 1 < len(text):
                        # Handle escape sequences
                        position += 1
                        next_char = text[position]
                        if next_char == 'n':
                            string_value += '\n'
                        elif next_char == 't':
                            string_value += '\t'
                        elif next_char == 'r':
                            string_value += '\r'
                        elif next_char == '\\':
                            string_value += '\\'
                        elif next_char == quote:
                            string_value += quote
                        else:
                            string_value += next_char
                        position += 1
                        column += 1
                    else:
                        string_value += text[position]
                        position += 1
                        column += 1
                        
                tokens.append(Token(TokenType.STRING, string_value, line, column))
                continue
                
            # Handle numbers
            if char.isdigit() or char == '.':
                number_value = ""
                while position < len(text) and (text[position].isdigit() or text[position] == '.'):
                    number_value += text[position]
                    position += 1
                    column += 1
                    
                # Determine if it's an integer or float
                if '.' in number_value:
                    tokens.append(Token(TokenType.NUMBER, number_value, line, column))
                else:
                    tokens.append(Token(TokenType.NUMBER, number_value, line, column))
                continue
                
            # Handle operators (check for multi-character operators first)
            if position + 1 < len(text):
                two_char = text[position:position + 2]
                if two_char in self.operators:
                    tokens.append(Token(self.operators[two_char], two_char, line, column))
                    position += 2
                    column += 2
                    continue
                    
            # Handle single character operators
            if char in self.operators:
                tokens.append(Token(self.operators[char], char, line, column))
                position += 1
                column += 1
                continue
                
            # Handle delimiters
            if char in self.delimiters:
                tokens.append(Token(self.delimiters[char], char, line, column))
                position += 1
                column += 1
                continue
                
            # Handle identifiers and keywords
            if char.isalpha() or char == '_':
                identifier = ""
                while position < len(text) and (text[position].isalnum() or text[position] == '_'):
                    identifier += text[position]
                    position += 1
                    column += 1
                    
                # Check if it's a keyword
                if identifier.lower() in self.keywords:
                    token_type = self.keywords[identifier.lower()]
                    tokens.append(Token(token_type, identifier, line, column))
                else:
                    # Check if it's a variable (starts with uppercase)
                    if identifier[0].isupper():
                        tokens.append(Token(TokenType.VARIABLE, identifier, line, column))
                    else:
                        tokens.append(Token(TokenType.IDENTIFIER, identifier, line, column))
                continue
                
            # Handle wildcard
            if char == '_':
                tokens.append(Token(TokenType.WILDCARD, '_', line, column))
                position += 1
                column += 1
                continue
                
            # Unknown character
            tokens.append(Token(TokenType.IDENTIFIER, char, line, column))
            position += 1
            column += 1
            
        # Add EOF token
        tokens.append(Token(TokenType.EOF, '', line, column))
        return tokens
        
    def is_keyword(self, word: str) -> bool:
        """Check if a word is a keyword."""
        return word.lower() in self.keywords
        
    def get_keyword_token_type(self, word: str) -> Optional[TokenType]:
        """Get token type for a keyword."""
        return self.keywords.get(word.lower())
        
    def is_operator(self, op: str) -> bool:
        """Check if a string is an operator."""
        return op in self.operators
        
    def get_operator_token_type(self, op: str) -> Optional[TokenType]:
        """Get token type for an operator."""
        return self.operators.get(op)
        
    def is_delimiter(self, char: str) -> bool:
        """Check if a character is a delimiter."""
        return char in self.delimiters
        
    def get_delimiter_token_type(self, char: str) -> Optional[TokenType]:
        """Get token type for a delimiter."""
        return self.delimiters.get(char)
