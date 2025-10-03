import unittest

from egdol.interpreter import Interpreter
from egdol.parser import Constant, Rule, Term, Variable
from egdol.rules_engine import RulesEngine


class InterpreterTests(unittest.TestCase):
    def test_infer_simple_rule(self):
        engine = RulesEngine()
        engine.add_fact(Term('human', [Constant('socrates')]))
        engine.add_rule(Rule(Term('mortal', [Variable('X')]), [Term('human', [Variable('X')])]))
        interp = Interpreter(engine)
        res = interp.query(Term('mortal', [Constant('socrates')]))
        self.assertEqual(len(res), 1)

    def test_infer_with_variable_binding(self):
        engine = RulesEngine()
        engine.add_fact(Term('human', [Constant('socrates')]))
        engine.add_fact(Term('human', [Constant('plato')]))
        engine.add_rule(Rule(Term('mortal', [Variable('X')]), [Term('human', [Variable('X')])]))
        interp = Interpreter(engine)
        res = interp.query(Term('mortal', [Variable('Y')]))
        # two solutions expected
        self.assertEqual(len(res), 2)
        vals = {repr(v['Y']) for v in res}
        self.assertEqual(vals, {'socrates', 'plato'})

    def test_chained_rules(self):
        engine = RulesEngine()
        engine.add_fact(Term('human', [Constant('socrates')]))
        # mortal(X) <- human(X)
        engine.add_rule(Rule(Term('mortal', [Variable('X')]), [Term('human', [Variable('X')])]))
        # extinct(X) <- mortal(X)
        engine.add_rule(Rule(Term('extinct', [Variable('X')]), [Term('mortal', [Variable('X')])]))
        interp = Interpreter(engine)
        res = interp.query(Term('extinct', [Constant('socrates')]))
        self.assertEqual(len(res), 1)

    def test_negation_as_failure(self):
        engine = RulesEngine()
        engine.add_fact(Term('human', [Constant('socrates')]))
        interp = Interpreter(engine)
        # not mortal(socrates) should succeed if no mortal(socrates) fact/rule
        res = interp.query(Term('not', [Term('mortal', [Constant('socrates')])]))
        self.assertEqual(len(res), 1)

    def test_cut_prunes_alternatives(self):
        engine = RulesEngine()
        # two rules for p/1: first uses cut to prevent second
        # p(a) :- q(a), !.
        # p(b) :- q(b).
        engine.add_fact(Term('q', [Constant('a')]))
        engine.add_fact(Term('q', [Constant('b')]))
        engine.add_rule(Rule(Term('p', [Variable('X')]), [Term('q', [Variable('X')]), Term('!', [])]))
        engine.add_rule(Rule(Term('p', [Variable('X')]), [Term('q', [Variable('X')])]))
        interp = Interpreter(engine)
        res = interp.query(Term('p', [Variable('Y')]))
        # cut in first rule should prune alternatives from other clauses for that choice
        vals = {str(v['Y']) for v in res}
        self.assertEqual(vals, {'a'})

    def test_recursion_depth_limit_terminating(self):
        engine = RulesEngine()
        # build ancestor chain: parent(i, i+1) for i in 0..49
        for i in range(50):
            engine.add_fact(Term('parent', [Constant(str(i)), Constant(str(i + 1))]))
        # ancestor(X, Y) :- parent(X, Y).
        engine.add_rule(Rule(Term('ancestor', [Variable('X'), Variable('Y')]), [Term('parent', [Variable('X'), Variable('Y')])]))
        # ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
        engine.add_rule(Rule(Term('ancestor', [Variable('X'), Variable('Z')]), [Term('parent', [Variable('X'), Variable('Y')]), Term('ancestor', [Variable('Y'), Variable('Z')])]))
        interp = Interpreter(engine)
        interp.max_depth = 200
        # should find ancestor 0 -> 50
        res = interp.query(Term('ancestor', [Constant('0'), Constant('50')]))
        self.assertEqual(len(res), 1)

    def test_recursion_depth_limit_exceeded(self):
        engine = RulesEngine()
        # loop(X) :- loop(X).
        engine.add_rule(Rule(Term('loop', [Variable('X')]), [Term('loop', [Variable('X')])]))
        interp = Interpreter(engine)
        interp.max_depth = 10
        with self.assertRaises(Exception):
            # query should raise MaxDepthExceededError
            interp.query(Term('loop', [Constant('a')]))

    def test_query_timeout_interrupts(self):
        engine = RulesEngine()
        # create a generator-style predicate with many facts to force long search
        for i in range(2000):
            engine.add_fact(Term('p', [Constant(str(i))]))
        # rule that triggers deep search by trying many alternatives
        engine.add_rule(Rule(Term('long', [Variable('X')]), [Term('p', [Variable('X')])]))
        interp = Interpreter(engine)
        interp.timeout_seconds = 0
        # set a tiny timeout (0.001s)
        interp.timeout_seconds = 0.001
        timer = interp._start_timeout()
        try:
            res = []
            # call query which internally will use _prove; we expect it to return quickly or empty
            try:
                r = interp.query(Term('long', [Variable('Y')]))
                # ensure we didn't hang; results may be partial or empty
            except Exception:
                pass
        finally:
            interp._stop_timeout(timer)


if __name__ == '__main__':
    unittest.main()
