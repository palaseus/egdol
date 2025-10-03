"""
DSL 2.0 Parser
Formal recursive descent parser for nested logical statements and mathematical expressions.
"""

from typing import List, Optional, Dict, Any
from .lexer import DSL2Lexer, Token, TokenType
from .ast import (
    DSL2AST, Node, Expression, Statement, Rule, Fact, Query,
    Literal, Variable, BinaryExpression, UnaryExpression, FunctionCall,
    ConditionalStatement, Block
)


class ParseError(Exception):
    """Parser error exception."""
    
    def __init__(self, message: str, line: int = 1, column: int = 1):
        super().__init__(message)
        self.line = line
        self.column = column


class DSL2Parser:
    """Formal recursive descent parser for DSL 2.0."""
    
    def __init__(self):
        self.lexer = DSL2Lexer()
        self.tokens: List[Token] = []
        self.current_token: Optional[Token] = None
        self.position = 0
        
    def parse(self, text: str) -> DSL2AST:
        """Parse the input text into an AST."""
        self.tokens = self.lexer.tokenize(text)
        self.position = 0
        self.current_token = self.tokens[0] if self.tokens else None
        
        statements = []
        
        while self.current_token and self.current_token.type != TokenType.EOF:
            statement = self.parse_statement()
            if statement:
                statements.append(statement)
            else:
                # If no statement was parsed, advance to avoid infinite loop
                self.advance()
                
        return DSL2AST(statements)
        
    def advance(self):
        """Advance to the next token."""
        if self.position < len(self.tokens) - 1:
            self.position += 1
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
            
    def peek(self, offset: int = 1) -> Optional[Token]:
        """Peek at a token ahead."""
        peek_position = self.position + offset
        if peek_position < len(self.tokens):
            return self.tokens[peek_position]
        return None
        
    def expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type."""
        if not self.current_token or self.current_token.type != token_type:
            expected = token_type.name
            actual = self.current_token.type.name if self.current_token else "EOF"
            raise ParseError(f"Expected {expected}, got {actual}", 
                           self.current_token.line if self.current_token else 1,
                           self.current_token.column if self.current_token else 1)
                           
        token = self.current_token
        self.advance()
        return token
        
    def parse_statement(self) -> Optional[Statement]:
        """Parse a statement."""
        if not self.current_token:
            return None
            
        # Handle different statement types
        if self.current_token.type == TokenType.IF:
            return self.parse_conditional_statement()
        elif self.current_token.type in [TokenType.IDENTIFIER, TokenType.VARIABLE]:
            # Could be a fact or rule
            return self.parse_fact_or_rule()
        elif self.current_token.type == TokenType.QUESTION:
            return self.parse_query()
        else:
            # Try to parse as expression
            try:
                expr = self.parse_expression()
                if expr:
                    return self.parse_fact_from_expression(expr)
            except ParseError:
                pass
                
        return None
        
    def parse_conditional_statement(self) -> ConditionalStatement:
        """Parse if-then-else statement."""
        line = self.current_token.line
        column = self.current_token.column
        
        # Parse if condition
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        
        # Parse then branch
        self.expect(TokenType.THEN)
        then_branch = []
        
        while (self.current_token and 
               self.current_token.type not in [TokenType.ELSE, TokenType.END, TokenType.EOF]):
            stmt = self.parse_statement()
            if stmt:
                then_branch.append(stmt)
                
        # Parse else branch (optional)
        else_branch = []
        if self.current_token and self.current_token.type == TokenType.ELSE:
            self.advance()
            while (self.current_token and 
                   self.current_token.type not in [TokenType.END, TokenType.EOF]):
                stmt = self.parse_statement()
                if stmt:
                    else_branch.append(stmt)
                    
        return ConditionalStatement(condition, then_branch, else_branch, line, column)
        
    def parse_fact_or_rule(self) -> Statement:
        """Parse a fact or rule statement."""
        line = self.current_token.line
        column = self.current_token.column
        
        # Parse subject (just the first identifier/variable)
        if self.current_token and self.current_token.type in [TokenType.IDENTIFIER, TokenType.VARIABLE]:
            subject = self.parse_primary()
        else:
            subject = self.parse_expression()
        
        # Skip "is" token if present
        if self.current_token and self.current_token.type == TokenType.IS:
            self.advance()
        
        # Check for rule indicators
        if (self.current_token and 
            self.current_token.type in [TokenType.IF, TokenType.IMPLIES]):
            return self.parse_rule(subject, line, column)
        else:
            return self.parse_fact(subject, line, column)
            
    def parse_fact(self, subject: Expression, line: int, column: int) -> Fact:
        """Parse a fact statement."""
        # Parse predicate
        if self.current_token and self.current_token.type != TokenType.EOF:
            # For simple facts like "Alice is human", parse the next identifier
            if self.current_token.type in [TokenType.IDENTIFIER, TokenType.VARIABLE]:
                predicate = self.parse_primary()
            else:
                predicate = self.parse_expression()
        else:
            # If no predicate, create a default one
            predicate = Literal("true", "boolean", line, column)
        return Fact(subject, predicate, line, column)
        
    def parse_rule(self, subject: Expression, line: int, column: int) -> Rule:
        """Parse a rule statement."""
        conditions = [subject]
        actions = []
        
        # Parse conditions
        if self.current_token and self.current_token.type == TokenType.IF:
            self.advance()
            while (self.current_token and 
                   self.current_token.type not in [TokenType.THEN, TokenType.IMPLIES]):
                cond = self.parse_expression()
                if cond:
                    conditions.append(cond)
                if self.current_token and self.current_token.type == TokenType.AND:
                    self.advance()
                    
        # Parse actions
        if self.current_token and self.current_token.type in [TokenType.THEN, TokenType.IMPLIES]:
            self.advance()
            while (self.current_token and 
                   self.current_token.type not in [TokenType.END, TokenType.EOF]):
                action = self.parse_expression()
                if action:
                    actions.append(action)
                if self.current_token and self.current_token.type == TokenType.AND:
                    self.advance()
                    
        return Rule(conditions, actions, line, column)
        
    def parse_query(self) -> Query:
        """Parse a query statement."""
        line = self.current_token.line
        column = self.current_token.column
        
        self.expect(TokenType.QUESTION)
        expression = self.parse_expression()
        return Query(expression, line, column)
        
    def parse_fact_from_expression(self, expr: Expression) -> Fact:
        """Parse a fact from an expression."""
        # This is a simplified fact parsing
        # In a real implementation, you'd need more sophisticated logic
        if isinstance(expr, BinaryExpression) and expr.operator == 'is':
            return Fact(expr.left, expr.right, expr.line, expr.column)
        else:
            # Create a simple fact
            return Fact(expr, Literal("true", "boolean", expr.line, expr.column), 
                       expr.line, expr.column)
        
    def parse_expression(self) -> Expression:
        """Parse an expression."""
        return self.parse_logical_or()
        
    def parse_logical_or(self) -> Expression:
        """Parse logical OR expression."""
        left = self.parse_logical_and()
        
        while (self.current_token and 
               self.current_token.type == TokenType.OR):
            operator = self.current_token.value
            self.advance()
            right = self.parse_logical_and()
            left = BinaryExpression(left, operator, right, left.line, left.column)
            
        return left
        
    def parse_logical_and(self) -> Expression:
        """Parse logical AND expression."""
        left = self.parse_equality()
        
        while (self.current_token and 
               self.current_token.type == TokenType.AND):
            operator = self.current_token.value
            self.advance()
            right = self.parse_equality()
            left = BinaryExpression(left, operator, right, left.line, left.column)
            
        return left
        
    def parse_equality(self) -> Expression:
        """Parse equality expression."""
        left = self.parse_relational()
        
        while (self.current_token and 
               self.current_token.type in [TokenType.EQUALS, TokenType.NOT_EQUALS]):
            operator = self.current_token.value
            self.advance()
            right = self.parse_relational()
            left = BinaryExpression(left, operator, right, left.line, left.column)
            
        return left
        
    def parse_relational(self) -> Expression:
        """Parse relational expression."""
        left = self.parse_addition()
        
        while (self.current_token and 
               self.current_token.type in [TokenType.GREATER_THAN, TokenType.LESS_THAN,
                                          TokenType.GREATER_EQUAL, TokenType.LESS_EQUAL]):
            operator = self.current_token.value
            self.advance()
            right = self.parse_addition()
            left = BinaryExpression(left, operator, right, left.line, left.column)
            
        return left
        
    def parse_addition(self) -> Expression:
        """Parse addition expression."""
        left = self.parse_multiplication()
        
        while (self.current_token and 
               self.current_token.type in [TokenType.PLUS, TokenType.MINUS]):
            operator = self.current_token.value
            self.advance()
            right = self.parse_multiplication()
            left = BinaryExpression(left, operator, right, left.line, left.column)
            
        return left
        
    def parse_multiplication(self) -> Expression:
        """Parse multiplication expression."""
        left = self.parse_unary()
        
        while (self.current_token and 
               self.current_token.type in [TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO]):
            operator = self.current_token.value
            self.advance()
            right = self.parse_unary()
            left = BinaryExpression(left, operator, right, left.line, left.column)
            
        return left
        
    def parse_unary(self) -> Expression:
        """Parse unary expression."""
        if (self.current_token and 
            self.current_token.type in [TokenType.NOT, TokenType.MINUS, TokenType.PLUS]):
            operator = self.current_token.value
            self.advance()
            operand = self.parse_unary()
            return UnaryExpression(operator, operand, operand.line, operand.column)
            
        return self.parse_primary()
        
    def parse_primary(self) -> Expression:
        """Parse primary expression."""
        if not self.current_token or self.current_token.type == TokenType.EOF:
            raise ParseError("Unexpected end of input")
            
        token = self.current_token
        
        if token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
            
        elif token.type == TokenType.IDENTIFIER:
            self.advance()
            # Check if it's a function call
            if (self.current_token and 
                self.current_token.type == TokenType.LPAREN):
                return self.parse_function_call(token.value, token.line, token.column)
            else:
                return Variable(token.value, token.line, token.column)
                
        elif token.type == TokenType.VARIABLE:
            self.advance()
            return Variable(token.value, token.line, token.column)
            
        elif token.type == TokenType.STRING:
            self.advance()
            return Literal(token.value, "string", token.line, token.column)
            
        elif token.type == TokenType.NUMBER:
            self.advance()
            return Literal(token.value, "number", token.line, token.column)
            
        elif token.type == TokenType.BOOLEAN:
            self.advance()
            return Literal(token.value == "true", "boolean", token.line, token.column)
            
        elif token.type == TokenType.WILDCARD:
            self.advance()
            return Variable("_", token.line, token.column)
            
        else:
            if token.type == TokenType.EOF:
                raise ParseError("Unexpected end of input", token.line, token.column)
            else:
                raise ParseError(f"Unexpected token: {token.value}", token.line, token.column)
            
    def parse_function_call(self, name: str, line: int, column: int) -> FunctionCall:
        """Parse function call."""
        self.expect(TokenType.LPAREN)
        arguments = []
        
        if (self.current_token and 
            self.current_token.type != TokenType.RPAREN):
            arguments.append(self.parse_expression())
            
            while (self.current_token and 
                   self.current_token.type == TokenType.COMMA):
                self.advance()
                arguments.append(self.parse_expression())
                
        self.expect(TokenType.RPAREN)
        return FunctionCall(name, arguments, line, column)
