import os
import tempfile
import unittest

from egdol.interpreter import Interpreter
from egdol.main import run_session
from egdol.rules_engine import RulesEngine


class PersistenceTests(unittest.TestCase):
    def test_settings_persist(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        interp.max_depth = 42
        interp.trace_level = 3
        # run empty session to attach settings
        engine, _, stats = run_session('', engine)
        # save to temp file
        with tempfile.NamedTemporaryFile('w', delete=False) as tf:
            fname = tf.name
        try:
            engine.save(fname)
            loaded = RulesEngine.load(fname)
            self.assertIn('settings', getattr(loaded, '__dict__', {}))
            s = loaded._settings
            self.assertEqual(s.get('max_depth'), 42)
            self.assertEqual(s.get('trace_level'), 3)
        finally:
            os.unlink(fname)


if __name__ == '__main__':
    unittest.main()
