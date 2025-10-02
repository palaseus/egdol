import unittest

from edgol.interpreter import Interpreter
from edgol.rules_engine import RulesEngine
from edgol.parser import Term, Constant


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
