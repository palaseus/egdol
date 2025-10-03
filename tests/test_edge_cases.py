"""
Comprehensive edge case testing for egdol.
Tests boundary conditions, error cases, and unusual scenarios.
"""

import unittest
import time
from egdol.rules_engine import RulesEngine
from egdol.interpreter import Interpreter, MaxDepthExceededError
from egdol.parser import Term, Variable, Constant, Rule, Fact, ParseError


class EdgeCaseTests(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        self.engine = RulesEngine()
        self.interp = Interpreter(self.engine)

    def test_empty_engine_queries(self):
        """Queries on empty engine should return empty results."""
        results = list(self.interp.query(Term('test', [Variable('X')])))
        self.assertEqual(len(results), 0)

    def test_self_referential_facts(self):
        """Self-referential facts should be handled correctly."""
        # Add self-referential fact
        self.engine.add_fact(Term('self', [Term('self', [Constant('1')])]))
        
        # Query should not crash
        results = list(self.interp.query(Term('self', [Variable('X')])))
        self.assertEqual(len(results), 1)
        self.assertIn('X', results[0])

    def test_circular_rules(self):
        """Circular rules should be handled with depth limits."""
        # Add circular rule: p(X) :- p(X)
        self.engine.add_rule(Rule(
            Term('p', [Variable('X')]),
            [Term('p', [Variable('X')])]
        ))
        
        # Query should hit depth limit
        with self.assertRaises(MaxDepthExceededError):
            list(self.interp.query(Term('p', [Constant('1')])))

    def test_very_deep_recursion(self):
        """Very deep recursion should be handled gracefully."""
        # Create deep recursion
        for i in range(1, 11):
            self.engine.add_fact(Term('parent', [Constant(str(i)), Constant(str(i+1))]))
        
        # Add recursive rule
        self.engine.add_rule(Rule(
            Term('ancestor', [Variable('X'), Variable('Y')]),
            [Term('parent', [Variable('X'), Variable('Y')])]
        ))
        
        self.engine.add_rule(Rule(
            Term('ancestor', [Variable('X'), Variable('Z')]),
            [Term('parent', [Variable('X'), Variable('Y')]),
             Term('ancestor', [Variable('Y'), Variable('Z')])]
        ))
        
        # Set low depth limit
        self.interp.max_depth = 5
        
        # Query should hit depth limit
        with self.assertRaises(MaxDepthExceededError):
            list(self.interp.query(Term('ancestor', [Constant('1'), Variable('Y')])))

    def test_large_term_structures(self):
        """Large term structures should be handled efficiently."""
        # Create very large term
        large_args = [Constant(str(i)) for i in range(100)]
        large_term = Term('large', large_args)
        
        self.engine.add_fact(large_term)
        
        # Query should work
        results = list(self.interp.query(large_term))
        self.assertEqual(len(results), 1)

    def test_very_long_variable_names(self):
        """Very long variable names should be handled."""
        long_var = 'X' * 1000
        fact = Term('test', [Variable(long_var)])
        
        self.engine.add_fact(fact)
        
        # Query with different variable name should work
        query = Term('test', [Variable('Y')])
        results = list(self.interp.query(query))
        self.assertEqual(len(results), 1)

    def test_unicode_handling(self):
        """Unicode characters should be handled correctly."""
        unicode_fact = Term('unicode', [Constant('测试'), Constant('🚀')])
        
        self.engine.add_fact(unicode_fact)
        
        # Query should work
        results = list(self.interp.query(unicode_fact))
        self.assertEqual(len(results), 1)

    def test_special_characters_in_atoms(self):
        """Special characters in atom names should be handled."""
        special_fact = Term('atom-with-dashes', [Constant('value')])
        
        self.engine.add_fact(special_fact)
        
        # Query should work
        results = list(self.interp.query(special_fact))
        self.assertEqual(len(results), 1)

    def test_numeric_atoms(self):
        """Numeric atom names should be handled."""
        numeric_fact = Term('123', [Constant('test')])
        
        self.engine.add_fact(numeric_fact)
        
        # Query should work
        results = list(self.interp.query(numeric_fact))
        self.assertEqual(len(results), 1)

    def test_empty_lists(self):
        """Empty lists should be handled correctly."""
        empty_list = Term('.', [])
        
        self.engine.add_fact(Term('empty', [empty_list]))
        
        # Query should work
        results = list(self.interp.query(Term('empty', [Variable('X')])))
        self.assertEqual(len(results), 1)

    def test_nested_structures(self):
        """Deeply nested structures should be handled."""
        # Create deeply nested term
        nested = Constant('base')
        for i in range(10):
            nested = Term('nest', [nested])
        
        self.engine.add_fact(Term('deep', [nested]))
        
        # Query should work
        results = list(self.interp.query(Term('deep', [Variable('X')])))
        self.assertEqual(len(results), 1)

    def test_duplicate_facts(self):
        """Duplicate facts should be handled correctly."""
        fact = Term('test', [Constant('1')])
        
        # Add same fact multiple times
        for _ in range(5):
            self.engine.add_fact(fact)
        
        # Query should return results (may be multiple due to duplicates)
        results = list(self.interp.query(fact))
        self.assertGreaterEqual(len(results), 1)

    def test_duplicate_rules(self):
        """Duplicate rules should be handled correctly."""
        rule = Rule(
            Term('test', [Variable('X')]),
            [Term('base', [Variable('X')])]
        )
        
        # Add same rule multiple times
        for _ in range(3):
            self.engine.add_rule(rule)
        
        # Add base fact
        self.engine.add_fact(Term('base', [Constant('1')]))
        
        # Query should work
        results = list(self.interp.query(Term('test', [Variable('X')])))
        self.assertGreaterEqual(len(results), 1)

    def test_mixed_case_variables(self):
        """Mixed case variables should be handled."""
        mixed_var = 'MixedCase'
        fact = Term('test', [Variable(mixed_var)])
        
        self.engine.add_fact(fact)
        
        # Query with different variable name should work
        query = Term('test', [Variable('X')])
        results = list(self.interp.query(query))
        self.assertEqual(len(results), 1)

    def test_very_small_numbers(self):
        """Very small numbers should be handled."""
        small_num = 1e-10
        fact = Term('small', [Constant(str(small_num))])
        
        self.engine.add_fact(fact)
        
        # Query should work
        results = list(self.interp.query(fact))
        self.assertEqual(len(results), 1)

    def test_very_large_numbers(self):
        """Very large numbers should be handled."""
        large_num = 1e10
        fact = Term('large', [Constant(str(large_num))])
        
        self.engine.add_fact(fact)
        
        # Query should work
        results = list(self.interp.query(fact))
        self.assertEqual(len(results), 1)

    def test_negative_numbers(self):
        """Negative numbers should be handled."""
        neg_num = -42
        fact = Term('negative', [Constant(str(neg_num))])
        
        self.engine.add_fact(fact)
        
        # Query should work
        results = list(self.interp.query(fact))
        self.assertEqual(len(results), 1)

    def test_floating_point_precision(self):
        """Floating point precision should be handled."""
        pi = 3.141592653589793
        fact = Term('pi', [Constant(str(pi))])
        
        self.engine.add_fact(fact)
        
        # Query should work
        results = list(self.interp.query(fact))
        self.assertEqual(len(results), 1)

    def test_boolean_values(self):
        """Boolean values should be handled."""
        true_fact = Term('bool', [Constant('True')])
        false_fact = Term('bool', [Constant('False')])
        
        self.engine.add_fact(true_fact)
        self.engine.add_fact(false_fact)
        
        # Query should work
        results = list(self.interp.query(Term('bool', [Variable('X')])))
        self.assertEqual(len(results), 2)

    def test_none_values(self):
        """None values should be handled."""
        none_fact = Term('none', [Constant('None')])
        
        self.engine.add_fact(none_fact)
        
        # Query should work
        results = list(self.interp.query(none_fact))
        self.assertEqual(len(results), 1)

    def test_empty_strings(self):
        """Empty strings should be handled."""
        empty_fact = Term('empty', [Constant('')])
        
        self.engine.add_fact(empty_fact)
        
        # Query should work
        results = list(self.interp.query(empty_fact))
        self.assertEqual(len(results), 1)

    def test_whitespace_strings(self):
        """Whitespace strings should be handled."""
        whitespace_fact = Term('space', [Constant('   ')])
        
        self.engine.add_fact(whitespace_fact)
        
        # Query should work
        results = list(self.interp.query(whitespace_fact))
        self.assertEqual(len(results), 1)

    def test_special_strings(self):
        """Special strings should be handled."""
        special_strings = ['\n', '\t', '\r', '\\', '"', "'"]
        
        for i, s in enumerate(special_strings):
            fact = Term('special', [Constant(s), Constant(str(i))])
            self.engine.add_fact(fact)
        
        # Query should work
        results = list(self.interp.query(Term('special', [Variable('X'), Variable('Y')])))
        self.assertEqual(len(results), len(special_strings))

    def test_very_long_strings(self):
        """Very long strings should be handled."""
        long_string = 'a' * 10000
        fact = Term('long', [Constant(long_string)])
        
        self.engine.add_fact(fact)
        
        # Query should work
        results = list(self.interp.query(fact))
        self.assertEqual(len(results), 1)

    def test_concurrent_operations(self):
        """Concurrent-like operations should not cause issues."""
        # Add facts rapidly
        for i in range(100):
            self.engine.add_fact(Term('rapid', [Constant(str(i))]))
        
        # Query while adding more facts
        results = []
        for i in range(100, 200):
            self.engine.add_fact(Term('rapid', [Constant(str(i))]))
            if i % 10 == 0:  # Query every 10 additions
                results.extend(list(self.interp.query(Term('rapid', [Variable('X')]))))
        
        # Should have some results
        self.assertGreater(len(results), 0)

    def test_memory_pressure(self):
        """Memory pressure should be handled gracefully."""
        # Add many facts to create memory pressure
        for i in range(10000):
            self.engine.add_fact(Term('pressure', [Constant(str(i)), Constant(str(i*2))]))
        
        # Query should still work
        results = list(self.interp.query(Term('pressure', [Variable('X'), Variable('Y')])))
        self.assertEqual(len(results), 10000)

    def test_timeout_handling(self):
        """Timeout handling should work correctly."""
        # Set very short timeout
        self.interp.timeout_seconds = 0.001  # 1ms
        
        # Add fact that might take time to process
        self.engine.add_fact(Term('timeout', [Constant('1')]))
        
        # Query should complete quickly or timeout gracefully
        start_time = time.perf_counter()
        try:
            results = list(self.interp.query(Term('timeout', [Variable('X')])))
            end_time = time.perf_counter()
            # Should complete quickly
            self.assertLess(end_time - start_time, 0.1)
        except Exception as e:
            # Timeout or other exceptions are acceptable
            self.assertIn(type(e).__name__, ['TimeoutError', 'MaxDepthExceededError'])

    def test_engine_reset(self):
        """Engine reset should work correctly."""
        # Add some facts
        self.engine.add_fact(Term('test', [Constant('1')]))
        
        # Query should work
        results = list(self.interp.query(Term('test', [Variable('X')])))
        self.assertEqual(len(results), 1)
        
        # Reset engine
        self.engine = RulesEngine()
        self.interp = Interpreter(self.engine)
        
        # Query should return empty
        results = list(self.interp.query(Term('test', [Variable('X')])))
        self.assertEqual(len(results), 0)

    def test_interpreter_configuration_edge_cases(self):
        """Interpreter configuration edge cases should be handled."""
        # Test extreme values
        self.interp.max_depth = 0
        self.interp.trace_level = -1
        
        # Should still work
        self.engine.add_fact(Term('test', [Constant('1')]))
        results = list(self.interp.query(Term('test', [Variable('X')])))
        self.assertEqual(len(results), 1)

    def test_rule_body_edge_cases(self):
        """Rule body edge cases should be handled."""
        # Empty rule body
        empty_rule = Rule(Term('empty', [Variable('X')]), [])
        self.engine.add_rule(empty_rule)
        
        # Should work (always succeeds)
        results = list(self.interp.query(Term('empty', [Constant('1')])))
        self.assertEqual(len(results), 1)
        
        # Rule with many body terms
        many_body = [Term('test', [Constant(str(i))]) for i in range(10)]
        many_rule = Rule(Term('many', [Variable('X')]), many_body)
        self.engine.add_rule(many_rule)
        
        # Add facts for all body terms
        for i in range(10):
            self.engine.add_fact(Term('test', [Constant(str(i))]))
        
        # Should work
        results = list(self.interp.query(Term('many', [Constant('1')])))
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()
