import unittest
import logging
import io

from edgol.rules_engine import RulesEngine
from edgol.interpreter import Interpreter
from edgol.parser import Term, Constant


class TraceTests(unittest.TestCase):
    def test_trace_basic_goal_and_unify(self):
        engine = RulesEngine()
        engine.add_fact(Term('human', [Constant('socrates')]))
        interp = Interpreter(engine)
        interp.trace_level = 2

        # capture logging
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        prev_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            _ = interp.query(Term('human', [Constant('socrates')]))
            handler.flush()
            s = buf.getvalue()
            self.assertIn('Entering goal', s)
            self.assertIn('Unify attempt', s)
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)


if __name__ == '__main__':
    unittest.main()
