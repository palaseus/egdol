import unittest

from egdol.interpreter import Interpreter
from egdol.rules_engine import RulesEngine
from egdol.parser import Term, Constant, Variable


class AppendGenerativeTests(unittest.TestCase):
    def test_append_generates_splits(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        # query: append(X,Y,[1,2,3]). -- represented as Term
        lst = Term('.', [Constant('1'), Term('.', [Constant('2'), Term('.', [Constant('3'), Constant('[]')])])])
        q = Term('append', [Variable('X'), Variable('Y'), lst])
        results = interp.query(q)
        # expect 4 splits: 0..3
        self.assertEqual(len(results), 4)


if __name__ == '__main__':
    unittest.main()
