import unittest
import tempfile
import os

from edgol.main import run_session, load_file
from edgol.rules_engine import RulesEngine


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
