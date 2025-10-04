"""
Tests for the Egdol DSL Interactive Assistant.
"""

import unittest
from egdol.dsl import DSLTokenizer, DSLParser, DSLExecutor
from egdol.dsl.ast import CommandStatement
from egdol.rules_engine import RulesEngine


class DSLTests(unittest.TestCase):
    """Test the DSL functionality."""
    
    def setUp(self):
        self.engine = RulesEngine()
        self.executor = DSLExecutor(self.engine)
        
    def test_tokenizer_basic(self):
        """Test basic tokenization."""
        tokenizer = DSLTokenizer()
        tokens = tokenizer.tokenize("Alice is a human.")
        
        # Should have tokens for each word
        self.assertGreater(len(tokens), 5)
        
        # Check for specific tokens
        token_types = [token.type.name for token in tokens]
        self.assertIn('PROPER_NOUN', token_types)
        self.assertIn('IS', token_types)
        self.assertIn('A', token_types)
        self.assertIn('IDENTIFIER', token_types)
        
    def test_parser_basic_fact(self):
        """Test parsing basic fact."""
        tokenizer = DSLTokenizer()
        parser = DSLParser()
        
        tokens = tokenizer.tokenize("Alice is a human.")
        ast = parser.parse(tokens)
        
        self.assertIsNotNone(ast)
        self.assertEqual(len(ast.statements), 1)
        
    def test_translator_basic_fact(self):
        """Test translating basic fact to Egdol."""
        result = self.executor.execute("Alice is a human.")
        
        self.assertIn('facts', result)
        self.assertEqual(len(result['facts']), 1)
        
        # Check that fact was added to engine
        stats = self.engine.stats()
        self.assertGreater(stats['num_facts'], 0)
        
    def test_translator_basic_rule(self):
        """Test translating basic rule to Egdol."""
        # First add a fact
        self.executor.execute("Alice is a human.")
        
        # Then add a rule
        result = self.executor.execute("If X is a human then X is mortal.")
        
        self.assertIn('rules', result)
        self.assertEqual(len(result['rules']), 1)
        
        # Check that rule was added to engine
        stats = self.engine.stats()
        self.assertGreater(stats['num_rules'], 0)
        
    def test_translator_basic_query(self):
        """Test translating basic query to Egdol."""
        # Add facts and rules
        self.executor.execute("Alice is a human.")
        self.executor.execute("If X is a human then X is mortal.")
        
        # Query
        result = self.executor.execute("Who is mortal?")
        
        self.assertIn('queries', result)
        self.assertEqual(len(result['queries']), 1)
        
        # Should find Alice as mortal
        query_result = result['queries'][0]
        self.assertIsNotNone(query_result['results'])
        
    def test_command_parsing(self):
        """Test parsing commands."""
        tokenizer = DSLTokenizer()
        parser = DSLParser()
        
        tokens = tokenizer.tokenize(":facts")
        ast = parser.parse(tokens)
        
        self.assertIsNotNone(ast)
        self.assertEqual(len(ast.statements), 1)
        self.assertIsInstance(ast.statements[0], CommandStatement)
        
    def test_pronoun_resolution(self):
        """Test pronoun resolution in context."""
        # Add fact with proper noun
        self.executor.execute("Alice is a human.")
        
        # Use pronoun
        result = self.executor.execute("She is mortal.")
        
        # Should work with context
        self.assertIn('facts', result)
        
    def test_complex_rule(self):
        """Test complex rule with conditions."""
        # Add facts
        self.executor.execute("Alice is a human.")
        self.executor.execute("Bob is a human.")
        
        # Add complex rule
        result = self.executor.execute("If X is a human and X is not a robot then X is mortal.")
        
        self.assertIn('rules', result)
        self.assertEqual(len(result['rules']), 1)
        
    def test_query_variations(self):
        """Test different query types."""
        # Add facts
        self.executor.execute("Alice is a human.")
        self.executor.execute("Alice is 25 years old.")
        
        # Test different query types
        queries = [
            "Who is a human?",
            "What is Alice?",
            "Does Alice have age 25?",
            "Show me all humans."
        ]
        
        for query in queries:
            result = self.executor.execute(query)
            self.assertIn('queries', result)
            self.assertEqual(len(result['queries']), 1)
            
    def test_plugin_system(self):
        """Test plugin system integration."""
        from egdol.plugins import PluginManager
        from egdol.plugins.arithmetic import ArithmeticPlugin
        
        # Create plugin manager
        plugin_manager = PluginManager(self.engine)
        
        # Register arithmetic plugin
        arithmetic_plugin = ArithmeticPlugin()
        plugin_manager.register_plugin(arithmetic_plugin)
        
        # Test plugin functionality
        self.assertTrue(plugin_manager.evaluate_predicate("equals", [5, 5]))
        self.assertFalse(plugin_manager.evaluate_predicate("equals", [5, 6]))
        
        result = plugin_manager.evaluate_function("add", [2, 3])
        self.assertEqual(result, 5.0)


class DSLIntegrationTests(unittest.TestCase):
    """Integration tests for the DSL system."""
    
    def setUp(self):
        self.engine = RulesEngine()
        self.executor = DSLExecutor(self.engine)
        
    def test_complete_scenario(self):
        """Test a complete scenario with facts, rules, and queries."""
        # Add facts
        self.executor.execute("Alice is a human.")
        self.executor.execute("Bob is a human.")
        self.executor.execute("Alice is 25 years old.")
        self.executor.execute("Bob has age 30.")
        
        # Add rules
        self.executor.execute("If X is a human then X is mortal.")
        self.executor.execute("If X has age Y and Y is greater than 20 then X is adult.")
        
        # Query
        result = self.executor.execute("Who is mortal?")
        
        # Should find both Alice and Bob
        query_result = result['queries'][0]
        self.assertGreater(len(query_result['results']), 0)
        
    def test_session_persistence(self):
        """Test session persistence with save/load."""
        # Add some facts
        self.executor.execute("Alice is a human.")
        self.executor.execute("Alice is 25 years old.")
        
        # Check stats
        stats = self.engine.stats()
        self.assertGreater(stats['num_facts'], 0)
        
        # Reset and verify
        self.engine = RulesEngine()
        self.executor = DSLExecutor(self.engine)
        
        stats = self.engine.stats()
        self.assertEqual(stats['num_facts'], 0)


if __name__ == '__main__':
    unittest.main()
