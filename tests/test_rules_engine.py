import unittest

from egdol.rules_engine import RulesEngine
from egdol.parser import Term, Variable, Constant


class RulesEngineTests(unittest.TestCase):
    def test_add_and_query_fact_exact(self):
        engine = RulesEngine()
        engine.add_fact(Term('human', [Constant('socrates')]))
        res = engine.query(Term('human', [Constant('socrates')]))
        self.assertEqual(len(res), 1)

    def test_query_with_variable(self):
        engine = RulesEngine()
        engine.add_fact(Term('human', [Constant('socrates')]))
        res = engine.query(Term('human', [Variable('X')]))
        self.assertEqual(len(res), 1)
        binding = res[0]
        self.assertIn('X', binding)
        self.assertEqual(repr(binding['X']), 'socrates')

    def test_no_match(self):
        engine = RulesEngine()
        engine.add_fact(Term('human', [Constant('plato')]))
        res = engine.query(Term('mortal', [Variable('X')]))
        self.assertEqual(len(res), 0)

    def test_indexing_speedup_basic(self):
        engine = RulesEngine()
        # add many facts with different predicates
        for i in range(100):
            engine.add_fact(Term(f'p{i}', [Constant('a')]))
        engine.add_fact(Term('human', [Constant('socrates')]))
        # query human should hit index and return the one fact
        res = engine.query(Term('human', [Variable('X')]))
        self.assertEqual(len(res), 1)

    def test_export_import_prolog_roundtrip(self):
        engine = RulesEngine()
        engine.add_fact(Term('p', [Constant('a')]))
        engine.add_fact(Term('q', [Variable('X')]))
        # dif constraint as terms
        engine.add_dif_constraint(Term('a', []), Term('b', []))
        # FD domain
        engine.add_fd_domain('X', 1, 3)
        path = 'test_prolog_export.pl'
        try:
            engine.export_prolog(path)
            newe = RulesEngine()
            newe.import_prolog(path)
            # expect at least predicate p present
            found = any(f.name == 'p' for f in newe.facts)
            self.assertTrue(found)
        finally:
            import os
            try:
                os.remove(path)
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()
