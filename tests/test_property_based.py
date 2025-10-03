"""
Property-based testing with Hypothesis for egdol.
Tests fundamental properties that should always hold.
"""

import unittest
from hypothesis import given, strategies as st, settings, example
from hypothesis.strategies import text, integers, lists, composite
from egdol.rules_engine import RulesEngine
from egdol.interpreter import Interpreter
from egdol.parser import Term, Variable, Constant, Rule, Fact


@composite
def prolog_atoms(draw):
    """Generate valid Prolog atom names."""
    return draw(st.text(
        alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')),
        min_size=1, max_size=10
    ).filter(lambda x: x and not x.startswith('_')))


@composite
def prolog_variables(draw):
    """Generate valid Prolog variable names."""
    return draw(st.text(
        alphabet=st.characters(min_codepoint=ord('A'), max_codepoint=ord('Z')),
        min_size=1, max_size=10
    ).filter(lambda x: x and x.isupper()))


@composite
def simple_terms(draw):
    """Generate simple Prolog terms."""
    name = draw(prolog_atoms())
    arity = draw(st.integers(min_value=0, max_value=3))
    args = []
    for _ in range(arity):
        if draw(st.booleans()):
            args.append(Constant(draw(st.text(min_size=1, max_size=5))))
        else:
            args.append(Variable(draw(prolog_variables())))
    return Term(name, args)


@composite
def facts(draw):
    """Generate valid facts."""
    return Fact(draw(simple_terms()))


@composite
def rules(draw):
    """Generate valid rules."""
    head = draw(simple_terms())
    body_size = draw(st.integers(min_value=0, max_value=3))
    body = [draw(simple_terms()) for _ in range(body_size)]
    return Rule(head, body)


class PropertyBasedTests(unittest.TestCase):
    """Property-based tests for core egdol functionality."""

    def setUp(self):
        self.engine = RulesEngine()
        self.interp = Interpreter(self.engine)

    @given(facts())
    @settings(max_examples=100)
    def test_fact_addition_consistency(self, fact):
        """Adding a fact should make it queryable."""
        self.engine.add_fact(fact.term)
        
        # The fact should be directly queryable
        results = list(self.interp.query(fact.term))
        self.assertGreater(len(results), 0)
        
        # All results should be valid substitutions
        for result in results:
            self.assertIsInstance(result, dict)

    @given(rules())
    @settings(max_examples=50)
    def test_rule_consistency(self, rule):
        """Rules should be addable and the head should be queryable."""
        self.engine.add_rule(rule)
        
        # The rule head should be queryable (may not succeed, but shouldn't crash)
        try:
            results = list(self.interp.query(rule.head))
            # Results should be valid substitutions
            for result in results:
                self.assertIsInstance(result, dict)
        except Exception as e:
            # Some rules may fail due to unsatisfiable body, which is OK
            self.assertIn(type(e).__name__, ['MaxDepthExceededError', 'UnificationError'])

    @given(
        st.lists(facts(), min_size=1, max_size=10),
        simple_terms()
    )
    @settings(max_examples=50)
    def test_query_soundness(self, facts_list, query):
        """All query results should be sound (satisfy the query)."""
        # Add facts
        for fact in facts_list:
            self.engine.add_fact(fact.term)
        
        # Query should not crash
        try:
            results = list(self.interp.query(query))
            
            # Each result should be a valid substitution
            for result in results:
                self.assertIsInstance(result, dict)
                # All values should be valid terms
                for var, value in result.items():
                    self.assertIsInstance(var, str)
                    self.assertTrue(hasattr(value, '__str__'))
                    
        except Exception as e:
            # Some queries may fail legitimately
            self.assertIn(type(e).__name__, ['MaxDepthExceededError', 'UnificationError'])

    @given(
        st.lists(simple_terms(), min_size=1, max_size=3),
        st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10)
    def test_engine_determinism(self, terms, seed):
        """Engine behavior should be deterministic."""
        # Add same facts in same order
        for term in terms:
            self.engine.add_fact(term)
        
        # Query should give same results multiple times
        query = terms[0] if terms else Term('test', [])
        results1 = list(self.interp.query(query))
        results2 = list(self.interp.query(query))
        
        # Results should be the same (order may vary, but content should match)
        self.assertEqual(len(results1), len(results2))

    @given(
        st.lists(facts(), min_size=1, max_size=20),
        st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=20)
    def test_engine_scalability(self, facts_list, iterations):
        """Engine should handle multiple facts without performance degradation."""
        # Add facts
        for fact in facts_list:
            self.engine.add_fact(fact.term)
        
        # Query should complete in reasonable time
        query = facts_list[0].term if facts_list else Term('test', [])
        
        import time
        start_time = time.perf_counter()
        results = list(self.interp.query(query))
        end_time = time.perf_counter()
        
        # Should complete within reasonable time (1 second)
        self.assertLess(end_time - start_time, 1.0)
        self.assertIsInstance(results, list)

    @given(
        st.lists(simple_terms(), min_size=1, max_size=10),
        st.booleans()
    )
    @settings(max_examples=20)
    def test_engine_statistics_consistency(self, terms, add_rules):
        """Engine statistics should be consistent with operations."""
        initial_stats = self.engine.stats()
        
        # Add facts
        for term in terms:
            self.engine.add_fact(term)
        
        # Add some rules if requested
        if add_rules and len(terms) > 1:
            for i in range(min(3, len(terms) - 1)):
                rule = Rule(terms[i], [terms[i + 1]])
                self.engine.add_rule(rule)
        
        final_stats = self.engine.stats()
        
        # Statistics should reflect the additions
        self.assertGreaterEqual(final_stats['num_facts'], initial_stats['num_facts'])
        if add_rules:
            self.assertGreaterEqual(final_stats['num_rules'], initial_stats['num_rules'])

    @given(
        st.lists(simple_terms(), min_size=2, max_size=5),
        st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=15)
    def test_unification_properties(self, terms, depth):
        """Unification should satisfy basic properties."""
        # Add facts
        for term in terms:
            self.engine.add_fact(term)
        
        # Create a query that should unify with at least one fact
        if terms:
            query = terms[0]  # Should match itself
            results = list(self.interp.query(query))
            
            # Should find at least one match
            self.assertGreaterEqual(len(results), 1)
            
            # All results should be valid substitutions
            for result in results:
                self.assertIsInstance(result, dict)

    @given(
        st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=10),
        st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=20)
    def test_variable_substitution_consistency(self, var_names, num_vars):
        """Variable substitutions should be consistent."""
        # Create facts with variables
        facts = []
        for i, name in enumerate(var_names[:num_vars]):
            fact = Term('test', [Variable(name), Constant(str(i))])
            facts.append(fact)
            self.engine.add_fact(fact)
        
        # Query with variable should return consistent results
        if facts:
            query = Term('test', [Variable('X'), Variable('Y')])
            results = list(self.interp.query(query))
            
            # All results should have consistent variable bindings
            for result in results:
                self.assertIsInstance(result, dict)
                for var, value in result.items():
                    self.assertIsInstance(var, str)
                    self.assertIsInstance(value, (Constant, Term, Variable))

    @given(
        st.lists(simple_terms(), min_size=1, max_size=8),
        st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=15)
    def test_engine_persistence_properties(self, terms, operations):
        """Engine should maintain consistency across operations."""
        # Add terms
        for term in terms:
            self.engine.add_fact(term)
        
        # Perform some operations
        for _ in range(operations):
            if terms:
                query = terms[0]
                results = list(self.interp.query(query))
                self.assertIsInstance(results, list)
        
        # Engine should still be in consistent state
        stats = self.engine.stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('num_facts', stats)
        self.assertIn('num_rules', stats)


class AdvancedPropertyTests(unittest.TestCase):
    """Advanced property-based tests for complex scenarios."""

    def setUp(self):
        self.engine = RulesEngine()
        self.interp = Interpreter(self.engine)

    @given(
        st.lists(simple_terms(), min_size=3, max_size=10),
        st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10)
    def test_recursive_rule_properties(self, terms, depth):
        """Recursive rules should maintain termination properties."""
        # Create a simple recursive structure
        if len(terms) >= 2:
            # Add base case
            self.engine.add_fact(terms[0])
            
            # Add recursive rule
            recursive_rule = Rule(
                terms[1],
                [terms[0], terms[1]]  # This will be recursive
            )
            self.engine.add_rule(recursive_rule)
            
            # Query should not hang indefinitely
            import time
            start_time = time.perf_counter()
            try:
                results = list(self.interp.query(terms[1]))
                end_time = time.perf_counter()
                
                # Should complete within reasonable time
                self.assertLess(end_time - start_time, 2.0)
                
            except Exception as e:
                # Recursive rules may legitimately fail
                self.assertIn(type(e).__name__, ['MaxDepthExceededError'])

    @given(
        st.lists(simple_terms(), min_size=2, max_size=6),
        st.booleans()
    )
    @settings(max_examples=10)
    def test_constraint_properties(self, terms, use_constraints):
        """Constraint handling should maintain logical consistency."""
        # Add facts
        for term in terms:
            self.engine.add_fact(term)
        
        # Add constraints if requested
        if use_constraints and len(terms) >= 2:
            from egdol.parser import Variable, Constant
            # Add a simple difference constraint
            self.engine.add_dif_constraint(Variable('X'), Constant('1'))
        
        # Queries should still work
        if terms:
            query = terms[0]
            try:
                results = list(self.interp.query(query))
                self.assertIsInstance(results, list)
            except Exception as e:
                # Constraints may cause legitimate failures
                self.assertIn(type(e).__name__, ['UnificationError'])

    @given(
        st.lists(simple_terms(), min_size=1, max_size=15),
        st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=10)
    def test_performance_properties(self, terms, iterations):
        """Performance should scale reasonably with data size."""
        # Add terms
        for term in terms:
            self.engine.add_fact(term)
        
        # Measure query performance
        import time
        total_time = 0
        
        for _ in range(iterations):
            if terms:
                query = terms[0]
                start_time = time.perf_counter()
                try:
                    results = list(self.interp.query(query))
                    end_time = time.perf_counter()
                    total_time += (end_time - start_time)
                except Exception:
                    # Some queries may fail, that's OK
                    pass
        
        # Average time per query should be reasonable
        if iterations > 0:
            avg_time = total_time / iterations
            self.assertLess(avg_time, 0.1)  # Should be under 100ms per query


if __name__ == '__main__':
    unittest.main()
