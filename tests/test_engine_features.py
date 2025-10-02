import unittest
import os
from edgol.rules_engine import RulesEngine
from edgol.parser import Term, Constant

class EngineFeatureTests(unittest.TestCase):
    def test_macro_and_export_json_and_check(self):
        eng = RulesEngine()
        eng.add_macro('q', ['X'], 'fact: p(X).')
        expanded = eng.expand_macros('q(1)')
        self.assertIn('p(1)', expanded)
        issues = eng.check_constraints()
        self.assertEqual(issues, [])
        path = 'tmp_export.json'
        try:
            eng.export_json(path)
            self.assertTrue(os.path.exists(path))
            with open(path, 'r', encoding='utf-8') as fh:
                data = fh.read()
            self.assertIn('facts', data)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

if __name__ == '__main__':
    unittest.main()
