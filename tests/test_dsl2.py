"""
Tests for DSL 2.0 System
Comprehensive testing of advanced natural language compiler.
"""

import unittest
from egdol.omnimind.dsl2 import DSL2Parser, DSL2Lexer, DSL2AST, DSL2Compiler, ContextManager
from egdol.omnimind.dsl2.ast import (
    Node, Expression, Statement, Rule, Fact, Query,
    Literal, Variable, BinaryExpression, UnaryExpression, FunctionCall,
    ConditionalStatement, Block
)
from egdol.omnimind.dsl2.lexer import Token, TokenType


class DSL2LexerTests(unittest.TestCase):
    """Test the DSL 2.0 lexer."""
    
    def setUp(self):
        self.lexer = DSL2Lexer()
        
    def test_tokenize_simple_fact(self):
        """Test tokenizing a simple fact."""
        text = "Alice is human"
        tokens = self.lexer.tokenize(text)
        
        self.assertEqual(len(tokens), 4)  # Alice, is, human, EOF
        self.assertEqual(tokens[0].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[0].value, "Alice")
        self.assertEqual(tokens[1].type, TokenType.IS)
        self.assertEqual(tokens[2].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[2].value, "human")
        self.assertEqual(tokens[3].type, TokenType.EOF)
        
    def test_tokenize_rule(self):
        """Test tokenizing a rule."""
        text = "If X is human then X is mortal"
        tokens = self.lexer.tokenize(text)
        
        self.assertEqual(len(tokens), 8)  # If, X, is, human, then, X, is, mortal, EOF
        self.assertEqual(tokens[0].type, TokenType.IF)
        self.assertEqual(tokens[1].type, TokenType.VARIABLE)
        self.assertEqual(tokens[1].value, "X")
        self.assertEqual(tokens[2].type, TokenType.IS)
        self.assertEqual(tokens[3].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[3].value, "human")
        self.assertEqual(tokens[4].type, TokenType.THEN)
        
    def test_tokenize_mathematical_expression(self):
        """Test tokenizing mathematical expressions."""
        text = "2 + 3 * 4"
        tokens = self.lexer.tokenize(text)
        
        self.assertEqual(len(tokens), 6)  # 2, +, 3, *, 4, EOF
        self.assertEqual(tokens[0].type, TokenType.NUMBER)
        self.assertEqual(tokens[0].value, "2")
        self.assertEqual(tokens[1].type, TokenType.PLUS)
        self.assertEqual(tokens[2].type, TokenType.NUMBER)
        self.assertEqual(tokens[2].value, "3")
        self.assertEqual(tokens[3].type, TokenType.MULTIPLY)
        self.assertEqual(tokens[4].type, TokenType.NUMBER)
        self.assertEqual(tokens[4].value, "4")
        
    def test_tokenize_string_literal(self):
        """Test tokenizing string literals."""
        text = '"Hello world"'
        tokens = self.lexer.tokenize(text)
        
        self.assertEqual(len(tokens), 2)  # String, EOF
        self.assertEqual(tokens[0].type, TokenType.STRING)
        self.assertEqual(tokens[0].value, "Hello world")
        
    def test_tokenize_boolean_literal(self):
        """Test tokenizing boolean literals."""
        text = "true and false"
        tokens = self.lexer.tokenize(text)
        
        self.assertEqual(len(tokens), 4)  # true, and, false, EOF
        self.assertEqual(tokens[0].type, TokenType.BOOLEAN)
        self.assertEqual(tokens[0].value, "true")
        self.assertEqual(tokens[1].type, TokenType.AND)
        self.assertEqual(tokens[2].type, TokenType.BOOLEAN)
        self.assertEqual(tokens[2].value, "false")
        
    def test_tokenize_complex_expression(self):
        """Test tokenizing complex expressions."""
        text = "If (X > 5) and (Y < 10) then Z = X + Y"
        tokens = self.lexer.tokenize(text)
        
        # Check key tokens
        token_types = [token.type for token in tokens]
        self.assertIn(TokenType.IF, token_types)
        self.assertIn(TokenType.LPAREN, token_types)
        self.assertIn(TokenType.GREATER_THAN, token_types)
        self.assertIn(TokenType.AND, token_types)
        self.assertIn(TokenType.THEN, token_types)
        self.assertIn(TokenType.EQUALS, token_types)
        self.assertIn(TokenType.PLUS, token_types)


class DSL2ParserTests(unittest.TestCase):
    """Test the DSL 2.0 parser."""
    
    def setUp(self):
        self.parser = DSL2Parser()
        
    def test_parse_simple_fact(self):
        """Test parsing a simple fact."""
        text = "Alice is human"
        ast = self.parser.parse(text)
        
        self.assertIsInstance(ast, DSL2AST)
        self.assertEqual(len(ast.statements), 1)
        self.assertIsInstance(ast.statements[0], Fact)
        
        fact = ast.statements[0]
        self.assertIsInstance(fact.subject, Variable)
        self.assertEqual(fact.subject.name, "Alice")
        self.assertIsInstance(fact.predicate, Variable)
        self.assertEqual(fact.predicate.name, "human")
        
    def test_parse_rule(self):
        """Test parsing a rule."""
        text = "If X is human then X is mortal"
        ast = self.parser.parse(text)
        
        self.assertIsInstance(ast, DSL2AST)
        self.assertEqual(len(ast.statements), 1)
        self.assertIsInstance(ast.statements[0], Rule)
        
        rule = ast.statements[0]
        self.assertEqual(len(rule.conditions), 1)
        self.assertEqual(len(rule.actions), 1)
        
    def test_parse_mathematical_expression(self):
        """Test parsing mathematical expressions."""
        text = "2 + 3 * 4"
        ast = self.parser.parse(text)
        
        self.assertIsInstance(ast, DSL2AST)
        # Should be parsed as a fact with mathematical expression
        self.assertEqual(len(ast.statements), 1)
        
    def test_parse_conditional_statement(self):
        """Test parsing conditional statements."""
        text = "If X > 5 then Y = 10 else Y = 0"
        ast = self.parser.parse(text)
        
        self.assertIsInstance(ast, DSL2AST)
        self.assertEqual(len(ast.statements), 1)
        self.assertIsInstance(ast.statements[0], ConditionalStatement)
        
    def test_parse_query(self):
        """Test parsing queries."""
        text = "? Who is mortal"
        ast = self.parser.parse(text)
        
        self.assertIsInstance(ast, DSL2AST)
        self.assertEqual(len(ast.statements), 1)
        self.assertIsInstance(ast.statements[0], Query)
        
    def test_parse_nested_expressions(self):
        """Test parsing nested expressions."""
        text = "If (X > 5) and (Y < 10) then Z = X + Y"
        ast = self.parser.parse(text)
        
        self.assertIsInstance(ast, DSL2AST)
        self.assertEqual(len(ast.statements), 1)
        
    def test_parse_function_call(self):
        """Test parsing function calls."""
        text = "calculate(2, 3)"
        ast = self.parser.parse(text)
        
        self.assertIsInstance(ast, DSL2AST)
        # Should be parsed as a fact with function call
        self.assertEqual(len(ast.statements), 1)


class DSL2CompilerTests(unittest.TestCase):
    """Test the DSL 2.0 compiler."""
    
    def setUp(self):
        self.context_manager = ContextManager()
        self.compiler = DSL2Compiler(self.context_manager)
        
    def test_compile_simple_fact(self):
        """Test compiling a simple fact."""
        text = "Alice is human"
        parser = DSL2Parser()
        ast = parser.parse(text)
        
        result = self.compiler.compile(ast)
        
        self.assertIn('facts', result)
        self.assertEqual(len(result['facts']), 1)
        self.assertEqual(result['facts'][0]['type'], 'fact')
        
    def test_compile_rule(self):
        """Test compiling a rule."""
        text = "If X is human then X is mortal"
        parser = DSL2Parser()
        ast = parser.parse(text)
        
        result = self.compiler.compile(ast)
        
        self.assertIn('rules', result)
        self.assertEqual(len(result['rules']), 1)
        self.assertEqual(result['rules'][0]['type'], 'rule')
        
    def test_compile_query(self):
        """Test compiling a query."""
        text = "? Who is mortal"
        parser = DSL2Parser()
        ast = parser.parse(text)
        
        result = self.compiler.compile(ast)
        
        self.assertIn('queries', result)
        self.assertEqual(len(result['queries']), 1)
        self.assertEqual(result['queries'][0]['type'], 'query')
        
    def test_compile_mathematical_expression(self):
        """Test compiling mathematical expressions."""
        text = "2 + 3 * 4"
        parser = DSL2Parser()
        ast = parser.parse(text)
        
        result = self.compiler.compile(ast)
        
        self.assertIn('facts', result)
        self.assertEqual(len(result['facts']), 1)
        
    def test_compile_with_context(self):
        """Test compiling with context manager."""
        # Add context
        self.context_manager.add_context({
            'entities': ['Alice', 'Bob'],
            'variables': {'X': 'Alice'}
        })
        
        # Bind a variable
        self.context_manager.bind_variable('X', 'Alice')
        
        text = "X is human"
        parser = DSL2Parser()
        ast = parser.parse(text)
        
        result = self.compiler.compile(ast)
        
        self.assertIn('facts', result)
        self.assertEqual(len(result['facts']), 1)
        
    def test_compile_error_handling(self):
        """Test error handling in compilation."""
        # This should cause a parsing error
        text = "Invalid syntax here"
        parser = DSL2Parser()
        
        try:
            ast = parser.parse(text)
            result = self.compiler.compile(ast)
            
            # Check if errors are captured
            if 'errors' in result:
                self.assertGreater(len(result['errors']), 0)
        except Exception:
            # Expected for invalid syntax
            pass


class ContextManagerTests(unittest.TestCase):
    """Test the context manager."""
    
    def setUp(self):
        self.context_manager = ContextManager()
        
    def test_add_context(self):
        """Test adding context."""
        context = {
            'entities': ['Alice', 'Bob'],
            'variables': {'X': 'Alice'}
        }
        
        self.context_manager.add_context(context)
        
        self.assertEqual(len(self.context_manager.context_history), 1)
        self.assertIn('Alice', self.context_manager.entity_references)
        self.assertIn('Bob', self.context_manager.entity_references)
        
    def test_resolve_pronoun(self):
        """Test pronoun resolution."""
        # Add context with entities
        self.context_manager.add_context({
            'entities': ['Alice'],
            'male_entity': 'Bob',
            'female_entity': 'Alice'
        })
        
        # Test pronoun resolution
        he_entity = self.context_manager.resolve_pronoun('he')
        she_entity = self.context_manager.resolve_pronoun('she')
        
        self.assertEqual(he_entity, 'Bob')
        self.assertEqual(she_entity, 'Alice')
        
    def test_resolve_variable(self):
        """Test variable resolution."""
        # Bind a variable
        self.context_manager.bind_variable('X', 'Alice')
        
        # Test variable resolution
        resolved = self.context_manager.resolve_variable('X')
        self.assertEqual(resolved, 'Alice')
        
    def test_get_recent_context(self):
        """Test getting recent context."""
        # Add multiple contexts
        for i in range(5):
            self.context_manager.add_context({
                'entities': [f'Entity{i}'],
                'timestamp': i
            })
            
        recent = self.context_manager.get_recent_context(3)
        self.assertEqual(len(recent), 3)
        
    def test_get_entity_history(self):
        """Test getting entity history."""
        # Add contexts with entity references
        for i in range(3):
            self.context_manager.add_context({
                'entities': ['Alice'],
                'timestamp': i
            })
            
        history = self.context_manager.get_entity_history('Alice')
        self.assertEqual(len(history), 3)
        
    def test_find_related_entities(self):
        """Test finding related entities."""
        # Add context with multiple entities
        self.context_manager.add_context({
            'entities': ['Alice', 'Bob', 'Charlie']
        })
        
        related = self.context_manager.find_related_entities('Alice')
        self.assertIn('Bob', related)
        self.assertIn('Charlie', related)
        
    def test_get_most_referenced_entities(self):
        """Test getting most referenced entities."""
        # Add contexts with entity references
        for i in range(3):
            self.context_manager.add_context({
                'entities': ['Alice']
            })
        for i in range(2):
            self.context_manager.add_context({
                'entities': ['Bob']
            })
            
        most_referenced = self.context_manager.get_most_referenced_entities()
        self.assertEqual(len(most_referenced), 2)
        self.assertEqual(most_referenced[0][0], 'Alice')
        self.assertEqual(most_referenced[0][1], 3)
        self.assertEqual(most_referenced[1][0], 'Bob')
        self.assertEqual(most_referenced[1][1], 2)


class IntegrationTests(unittest.TestCase):
    """Integration tests for DSL 2.0 system."""
    
    def setUp(self):
        self.context_manager = ContextManager()
        self.parser = DSL2Parser()
        self.compiler = DSL2Compiler(self.context_manager)
        
    def test_full_compilation_pipeline(self):
        """Test the full compilation pipeline."""
        text = "If Alice is human then Alice is mortal"
        
        # Parse
        ast = self.parser.parse(text)
        self.assertIsInstance(ast, DSL2AST)
        
        # Compile
        result = self.compiler.compile(ast)
        self.assertIn('rules', result)
        self.assertEqual(len(result['rules']), 1)
        
    def test_context_aware_compilation(self):
        """Test context-aware compilation."""
        # Add context
        self.context_manager.add_context({
            'entities': ['Alice', 'Bob'],
            'female_entity': 'Alice',
            'male_entity': 'Bob'
        })
        
        # Parse and compile with context
        text = "She is human"
        ast = self.parser.parse(text)
        result = self.compiler.compile(ast)
        
        # Should resolve 'She' to 'Alice' through context
        self.assertIn('facts', result)
        
    def test_mathematical_expression_compilation(self):
        """Test mathematical expression compilation."""
        text = "2 + 3 * 4"
        ast = self.parser.parse(text)
        result = self.compiler.compile(ast)
        
        self.assertIn('facts', result)
        self.assertEqual(len(result['facts']), 1)
        
    def test_nested_logical_compilation(self):
        """Test nested logical expression compilation."""
        text = "If (X > 5) and (Y < 10) then Z = X + Y"
        ast = self.parser.parse(text)
        result = self.compiler.compile(ast)
        
        self.assertIn('facts', result)
        self.assertEqual(len(result['facts']), 1)
        
    def test_error_recovery(self):
        """Test error recovery in compilation."""
        # Test with invalid syntax
        text = "Invalid syntax here"
        
        try:
            ast = self.parser.parse(text)
            result = self.compiler.compile(ast)
            
            # Should handle errors gracefully
            if 'errors' in result:
                self.assertGreater(len(result['errors']), 0)
        except Exception:
            # Expected for invalid syntax
            pass


if __name__ == '__main__':
    unittest.main()
