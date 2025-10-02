import unittest

from edgol.rules_engine import RulesEngine
from edgol.interpreter import Interpreter
from edgol.parser import Term, Variable, Constant, Rule


class BuiltinsTests(unittest.TestCase):
    def test_is_builtin(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        # is(X, 2+3)
        res = interp.query(Term('is', [Variable('X'), Term('+', [Constant('2'), Constant('3')])]))
        self.assertEqual(len(res), 1)
        self.assertEqual(str(res[0]['X']), '5')

    def test_comparison_builtin(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        res = interp.query(Term('<', [Term('+', [Constant('1'), Constant('1')]), Constant('3')]))
        self.assertEqual(len(res), 1)

    def test_eq_builtin(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        # =/2 should unify
        res = interp.query(Term('=', [Variable('X'), Constant('7')]))
        self.assertEqual(len(res), 1)
        self.assertEqual(str(res[0]['X']), '7')

    def test_member_and_append(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        # list [1,2,3] parsed as terms: build via parser by parsing string
        parser_list = Term('.', [Constant('1'), Term('.', [Constant('2'), Term('.', [Constant('3'), Constant('[]')])])])
        # member(X, [1,2,3]) should yield X=1,2,3
        res = interp.query(Term('member', [Variable('X'), parser_list]))
        vals = {str(r['X']) for r in res}
        self.assertEqual(vals, {'1', '2', '3'})

    def test_atom_concat_and_io(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        res = interp.query(Term('atom_concat', [Constant('hello'), Constant('world'), Variable('R')]))
        self.assertEqual(len(res), 1)
        self.assertEqual(str(res[0]['R']), 'helloworld')
        # write and nl should succeed (side effects printed)
        res2 = interp.query(Term('write', [Constant('test')]))
        self.assertEqual(len(res2), 1)
        res3 = interp.query(Term('nl', []))
        self.assertEqual(len(res3), 1)


if __name__ == '__main__':
    unittest.main()
