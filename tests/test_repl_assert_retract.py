import unittest

from egdol.main import run_session
from egdol.rules_engine import RulesEngine


class ReplAssertRetractTests(unittest.TestCase):
    def test_assert_and_retract(self):
        engine = RulesEngine()
        # assert fact
        _, results = run_session('fact: human(bob).', engine)
        self.assertEqual(len(engine.facts), 1)
        # retract it
        engine.retract(engine.facts[0])
        self.assertEqual(len(engine.facts), 0)


if __name__ == '__main__':
    unittest.main()
