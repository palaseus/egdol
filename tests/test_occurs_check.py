import unittest

from egdol.interpreter import Interpreter, UnificationError
from egdol.parser import Constant, Term, Variable
from egdol.rules_engine import RulesEngine


class OccursCheckTests(unittest.TestCase):
    def test_simple_occurs_check_returns_no_solution(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        # default behavior: occurs_check enabled, but raise_on_occurs False -> unification fails (no solutions)
        goal = Term('=', [Variable('X'), Term('f', [Variable('X')])])
        res = interp.query(goal)
        # expect no solutions
        self.assertEqual(len(res), 0)

    def test_simple_occurs_check_raises_when_configured(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        interp.raise_on_occurs = True
        goal = Term('=', [Variable('X'), Term('f', [Variable('X')])])
        with self.assertRaises(UnificationError):
            interp.query(goal)

    def test_list_occurs_check(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        # X = [X] represented as list Term '.'(Head, Tail) with Constant '[]'
        # build [X] as Term('.', [X, Constant('[]')])
        x = Variable('X')
        lst = Term('.', [x, Constant('[]')])
        goal = Term('=', [x, lst])
        res = interp.query(goal)
        # binding X = [X] should be rejected by occurs-check
        self.assertEqual(len(res), 0)


if __name__ == '__main__':
    unittest.main()
