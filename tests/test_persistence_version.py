import json
import os
import tempfile
import unittest

from egdol.rules_engine import RulesEngine


class PersistenceVersionTests(unittest.TestCase):
    def test_migrate_legacy_save(self):
        # create legacy data (no version)
        data = {
            'facts': [{'term': {'name': 'a', 'args': []}}],
            'rules': [],
            'settings': {'max_depth': 5}
        }
        with tempfile.NamedTemporaryFile('w', delete=False) as tf:
            json.dump(data, tf)
            fname = tf.name
        try:
            loaded = RulesEngine.load(fname)
            self.assertIn('_settings', getattr(loaded, '__dict__', {}))
            # save back and verify version inserted
            loaded.save(fname)
            with open(fname, 'r') as fh:
                data2 = json.load(fh)
            self.assertEqual(data2.get('version'), 1)
        finally:
            os.unlink(fname)


if __name__ == '__main__':
    unittest.main()
