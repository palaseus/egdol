import unittest

from egdol.interpreter import Interpreter
from egdol.parser import Term
from egdol.rules_engine import RulesEngine


class RegisterBuiltinTests(unittest.TestCase):
    def test_register_custom_builtin(self):
        engine = RulesEngine()
        interp = Interpreter(engine)

        def always_yes(term, subst):
            yield subst

        interp.register_builtin('always', always_yes)
        q = Term('always', [])
        res = interp.query(q)
        self.assertEqual(len(res), 1)


if __name__ == '__main__':
    unittest.main()
