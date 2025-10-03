import unittest

from egdol.interpreter import Interpreter
from egdol.parser import Constant, Rule, Term, Variable
from egdol.rules_engine import RulesEngine


class TablingTests(unittest.TestCase):
    def test_tabling_reduces_prove_calls(self):
        engine = RulesEngine()
        # define edges for a small DAG
        edges = [(1,2),(1,3),(2,4),(3,4),(4,5)]
        for a,b in edges:
            engine.add_fact(Term('edge', [Constant(str(a)), Constant(str(b))]))
        # path(X,Y) :- edge(X,Y).
        engine.add_rule(Rule(Term('path', [Variable('X'), Variable('Y')]), [Term('edge', [Variable('X'), Variable('Y')])]))
        # path(X,Z) :- edge(X,Y), path(Y,Z).
        engine.add_rule(Rule(Term('path', [Variable('X'), Variable('Z')]), [Term('edge', [Variable('X'), Variable('Y')]), Term('path', [Variable('Y'), Variable('Z')])]))

        interp = Interpreter(engine)
        # run without tabling
        interp.tabling = False
        interp.prove_count = 0
        res_no = interp.query(Term('path', [Constant('1'), Variable('Z')]))
        calls_no = interp.prove_count

        # run with tabling
        interp2 = Interpreter(engine)
        interp2.tabling = True
        interp2.prove_count = 0
        res_yes = interp2.query(Term('path', [Constant('1'), Variable('Z')]))
        calls_yes = interp2.prove_count

        # Both should find same number of solutions (node 2,3,4,5 reachable)
        self.assertEqual(len(res_no), len(res_yes))
        # With tabling we expect fewer prove_count calls (or equal)
        self.assertLessEqual(calls_yes, calls_no)


if __name__ == '__main__':
    unittest.main()
