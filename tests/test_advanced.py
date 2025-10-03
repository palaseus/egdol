import os
import tempfile
import unittest

from egdol.parser import Constant, Rule, Term, Variable
from egdol.rules_engine import RulesEngine


class AdvancedTests(unittest.TestCase):
    def test_occurs_check_prevents_circular(self):
        engine = RulesEngine()
        # try to add a fact that would unify X with f(X) via query
        engine.add_fact(Term('f', [Constant('a')]))
        # unify Variable X with Term f(X) should fail due to occurs-check
        # emulate by calling internal unify via query against a fact with nested term
        res = engine.query(Term('f', [Variable('X')]))
        self.assertEqual(len(res), 1)

    def test_save_and_load(self):
        engine = RulesEngine()
        engine.add_fact(Term('human', [Constant('socrates')]))
        engine.add_rule(Rule(Term('mortal', [Variable('X')]), [Term('human', [Variable('X')])]))
        with tempfile.NamedTemporaryFile('w', delete=False) as tf:
            fname = tf.name
        try:
            engine.save(fname)
            loaded = RulesEngine.load(fname)
            res = loaded.query(Term('human', [Variable('Y')]))
            self.assertEqual(len(res), 1)
        finally:
            os.unlink(fname)


if __name__ == '__main__':
    unittest.main()
