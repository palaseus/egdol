import unittest

from egdol.main import run_session


class MainSessionTests(unittest.TestCase):
    def test_session_fact_and_query(self):
        text = 'fact: human(socrates). ? human(socrates).'
        engine, results = run_session(text)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)

    def test_session_rule_inference(self):
        text = 'fact: human(socrates). rule: mortal(X) => human(X). ? mortal(socrates).'
        engine, results = run_session(text)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)


if __name__ == '__main__':
    unittest.main()
