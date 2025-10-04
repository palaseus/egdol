"""
DSL Parser for Egdol Interactive Assistant.
Parses tokenized DSL into Abstract Syntax Tree (AST).
"""

from typing import List, Optional, Union, Dict, Any
from .tokenizer import TokenStream, TokenType, Token
from .ast import *


class DSLParser:
    """Recursive descent parser for the Egdol DSL."""
    
    def __init__(self):
        self.stream: Optional[TokenStream] = None
        self.context_memory: List[Dict[str, Any]] = []
        
    def parse(self, tokens: List[Token]) -> Program:
        """Parse tokens into an AST."""
        self.stream = TokenStream(tokens)
        statements = []
        
        while not self.stream.is_at_end():
            if self.stream.match(TokenType.NEWLINE):
                self.stream.consume()
                continue
                
            statement = self._parse_statement()
            if statement:
                statements.append(statement)
                
        return Program(statements)
        
    def _parse_statement(self) -> Optional[Statement]:
        """Parse a single statement."""
        if self.stream.is_at_end():
            return None
            
        # Command statements (start with :)
        if self.stream.match(TokenType.COLON):
            return self._parse_command()
            
        # Query statements (start with question words)
        if self.stream.match(TokenType.WHO) or self.stream.match(TokenType.WHAT) or \
           self.stream.match(TokenType.WHERE) or self.stream.match(TokenType.WHEN) or \
           self.stream.match(TokenType.WHY) or self.stream.match(TokenType.HOW) or \
           self.stream.match(TokenType.DOES) or self.stream.match(TokenType.DO) or \
           self.stream.match(TokenType.SHOW):
            return self._parse_query()
            
        # Rule statements (start with if/when)
        if self.stream.match(TokenType.IF) or self.stream.match(TokenType.WHEN):
            return self._parse_rule()
            
        # Fact statements (everything else)
        return self._parse_fact()
        
    def _parse_command(self) -> CommandStatement:
        """Parse a command statement (:command args)."""
        self.stream.expect(TokenType.COLON)
        
        command_token = self.stream.consume()
        if not command_token:
            raise SyntaxError("Expected command name after ':'")
            
        command_name = command_token.value.lower()
        args = []
        
        # Parse arguments
        while not self.stream.is_at_end() and not self.stream.match(TokenType.NEWLINE):
            if self.stream.match(TokenType.STRING):
                token = self.stream.consume()
                args.append(token.value.strip('"'))
            elif self.stream.match(TokenType.IDENTIFIER) or self.stream.match(TokenType.PROPER_NOUN):
                token = self.stream.consume()
                args.append(token.value)
            elif self.stream.match(TokenType.NUMBER):
                token = self.stream.consume()
                args.append(int(token.value))
            else:
                break
                
        return CommandStatement(command_name, args)
        
    def _parse_query(self) -> QueryStatement:
        """Parse a query statement."""
        query_type = self.stream.consume()
        
        if query_type.type == TokenType.WHO:
            return self._parse_who_query()
        elif query_type.type == TokenType.WHAT:
            return self._parse_what_query()
        elif query_type.type == TokenType.DOES or query_type.type == TokenType.DO:
            return self._parse_does_query()
        elif query_type.type == TokenType.SHOW:
            return self._parse_show_query()
        else:
            # Generic query
            return self._parse_generic_query()
            
    def _parse_who_query(self) -> WhoQuery:
        """Parse 'who is X' query."""
        if self.stream.match(TokenType.IS):
            self.stream.consume()
            
        # Handle 'is a' construction
        if self.stream.match(TokenType.A) or self.stream.match(TokenType.AN):
            self.stream.consume()
            
        predicate = self._parse_predicate()
        
        # Handle question mark at the end
        if self.stream.match(TokenType.QUESTION):
            self.stream.consume()
            
        return WhoQuery(predicate)
        
    def _parse_what_query(self) -> WhatQuery:
        """Parse 'what is X' query."""
        if self.stream.match(TokenType.IS):
            self.stream.consume()
            
        subject = self._parse_subject()
        
        # Handle question mark at the end
        if self.stream.match(TokenType.QUESTION):
            self.stream.consume()
            
        return WhatQuery(subject)
        
    def _parse_does_query(self) -> DoesQuery:
        """Parse 'does X have Y' query."""
        subject = self._parse_subject()
        
        if self.stream.match(TokenType.HAVE) or self.stream.match(TokenType.HAS):
            self.stream.consume()
            property_ = self._parse_predicate()
            
            # Handle compound predicates like "age 25"
            while (self.stream.peek() and 
                   self.stream.peek().type in [TokenType.NUMBER, TokenType.IDENTIFIER] and
                   not self.stream.match(TokenType.QUESTION)):
                next_token = self.stream.consume()
                if hasattr(property_, 'noun'):
                    property_.noun = f"{property_.noun} {next_token.value}"
                elif hasattr(property_, 'number'):
                    property_.number = f"{property_.number} {next_token.value}"
                elif hasattr(property_, 'compound'):
                    property_.compound = f"{property_.compound} {next_token.value}"
            
            # Handle question mark at the end
            if self.stream.match(TokenType.QUESTION):
                self.stream.consume()
                
            return DoesQuery(subject, property_)
        else:
            # 'does X Y' form
            predicate = self._parse_predicate()
            
            # Handle question mark at the end
            if self.stream.match(TokenType.QUESTION):
                self.stream.consume()
                
            return DoesQuery(subject, predicate)
            
    def _parse_show_query(self) -> ShowQuery:
        """Parse 'show me all X' query."""
        if self.stream.match(TokenType.ME):
            self.stream.consume()
        if self.stream.match(TokenType.ALL):
            self.stream.consume()
            
        predicate = self._parse_predicate()
        
        # Handle period at the end
        if self.stream.match(TokenType.PERIOD):
            self.stream.consume()
            
        return ShowQuery(predicate)
        
    def _parse_generic_query(self) -> GenericQuery:
        """Parse generic query."""
        subject = self._parse_subject()
        
        if self.stream.match(TokenType.IS):
            self.stream.consume()
            predicate = self._parse_predicate()
            return IsQuery(subject, predicate)
        else:
            return GenericQuery(subject)
            
    def _parse_rule(self) -> RuleStatement:
        """Parse rule statement (if X then Y)."""
        if self.stream.match(TokenType.IF):
            self.stream.consume()
            condition = self._parse_condition()
            self.stream.expect(TokenType.THEN)
            conclusion = self._parse_fact()
            return IfThenRule(condition, conclusion)
        elif self.stream.match(TokenType.WHEN):
            self.stream.consume()
            condition = self._parse_condition()
            self.stream.expect(TokenType.THEN)
            conclusion = self._parse_fact()
            return WhenThenRule(condition, conclusion)
        else:
            raise SyntaxError("Expected 'if' or 'when' for rule")
            
    def _parse_fact(self) -> FactStatement:
        """Parse fact statement."""
        # Parse subject first
        subject = self._parse_subject()
        
        if self.stream.match(TokenType.IS):
            self.stream.consume()
            
            # Handle 'is not' construction
            if self.stream.match(TokenType.NOT):
                self.stream.consume()
                # Handle 'is not a' construction
                if self.stream.match(TokenType.A) or self.stream.match(TokenType.AN):
                    self.stream.consume()
                predicate = self._parse_predicate()
                
                # Handle period at the end
                if self.stream.match(TokenType.PERIOD):
                    self.stream.consume()
                    
                return NotFact(subject, predicate)
            
            # Handle 'is a' construction
            elif self.stream.match(TokenType.A) or self.stream.match(TokenType.AN):
                self.stream.consume()
                predicate = self._parse_predicate()
                
                # Handle period at the end
                if self.stream.match(TokenType.PERIOD):
                    self.stream.consume()
                    
                return IsFact(subject, predicate)
            else:
                predicate = self._parse_predicate()
                
                # Handle period at the end
                if self.stream.match(TokenType.PERIOD):
                    self.stream.consume()
                    
                return IsFact(subject, predicate)
            
        elif self.stream.match(TokenType.HAVE) or self.stream.match(TokenType.HAS):
            self.stream.consume()
            property_ = self._parse_predicate()
            
            # Handle period at the end
            if self.stream.match(TokenType.PERIOD):
                self.stream.consume()
                
            return HasFact(subject, property_)
            
        else:
            # Direct predicate
            predicate = self._parse_predicate()
            
            # Handle period at the end
            if self.stream.match(TokenType.PERIOD):
                self.stream.consume()
                
            return IsFact(subject, predicate)
            
    def _parse_condition(self) -> Condition:
        """Parse condition (for rules)."""
        if self.stream.match(TokenType.NOT):
            self.stream.consume()
            fact = self._parse_fact()
            return NotCondition(fact)
        else:
            fact = self._parse_fact()
            
            # Check for conjunction/disjunction
            if self.stream.match(TokenType.AND):
                self.stream.consume()
                other = self._parse_condition()
                return AndCondition(fact, other)
            elif self.stream.match(TokenType.OR):
                self.stream.consume()
                other = self._parse_condition()
                return OrCondition(fact, other)
            else:
                return SimpleCondition(fact)
                
    def _parse_subject(self) -> Subject:
        """Parse subject (proper noun, variable, or pronoun)."""
        token = self.stream.consume()
        
        if token.type == TokenType.PROPER_NOUN:
            # Store in context memory
            self._update_context('subject', token.value)
            return ProperNounSubject(token.value)
            
        elif token.type == TokenType.VARIABLE:
            return VariableSubject(token.value)
            
        elif token.type in [TokenType.HE, TokenType.SHE, TokenType.IT, TokenType.THEY,
                           TokenType.HIM, TokenType.HER, TokenType.THEM]:
            # Resolve pronoun from context
            resolved = self._resolve_pronoun(token.value)
            return PronounSubject(token.value, resolved)
            
        else:
            raise SyntaxError(f"Expected subject, got {token.value}")
            
    def _parse_predicate(self) -> Predicate:
        """Parse predicate (noun, adjective, number, or compound)."""
        token = self.stream.consume()
        
        if token.type == TokenType.IDENTIFIER:
            # Handle compound predicates like "age 30"
            predicate_parts = [token.value]
            
            # Collect additional words that might be part of the predicate
            while (self.stream.peek() and 
                   self.stream.peek().type in [TokenType.NUMBER, TokenType.IDENTIFIER, TokenType.ADJECTIVE, TokenType.VARIABLE] and
                   not self.stream.match(TokenType.PERIOD) and
                   not self.stream.match(TokenType.QUESTION)):
                next_token = self.stream.consume()
                predicate_parts.append(next_token.value)
            
            if len(predicate_parts) > 1:
                return CompoundPredicate(" ".join(predicate_parts))
            else:
                return CommonNounPredicate(token.value)
        elif token.type == TokenType.PROPER_NOUN:
            return ProperNounPredicate(token.value)
        elif token.type == TokenType.ADJECTIVE:
            return AdjectivePredicate(token.value)
        elif token.type == TokenType.NUMBER:
            # Handle compound predicates like "25 years old"
            predicate_parts = [token.value]
            
            # Collect additional words that might be part of the predicate
            while (self.stream.peek() and 
                   self.stream.peek().type in [TokenType.IDENTIFIER, TokenType.ADJECTIVE] and
                   not self.stream.match(TokenType.PERIOD) and
                   not self.stream.match(TokenType.QUESTION)):
                next_token = self.stream.consume()
                predicate_parts.append(next_token.value)
            
            return CompoundPredicate(" ".join(predicate_parts))
        else:
            raise SyntaxError(f"Expected predicate, got {token.value}")
            
    def _update_context(self, key: str, value: str):
        """Update context memory."""
        if not self.context_memory:
            self.context_memory.append({})
        self.context_memory[-1][key] = value
        
    def _resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve pronoun from context memory."""
        if not self.context_memory:
            return None
            
        # Look through context stack (most recent first)
        for context in reversed(self.context_memory):
            if 'subject' in context:
                return context['subject']
                
        return None


class ParseError(Exception):
    """Exception raised during parsing."""
    pass
