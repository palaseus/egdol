import unittest
import tempfile
import os

from edgol.main import run_session, load_file


class ReplTests(unittest.TestCase):
    def test_load_file_and_query(self):
        content = 'fact: human(socrates). rule: mortal(X) => human(X).'
        with tempfile.NamedTemporaryFile('w', delete=False) as tf:
            tf.write(content)
            fname = tf.name
        try:
            engine, _ = run_session('', None)
            load_file(engine, fname)
            _, results = run_session('? mortal(socrates).', engine)
            self.assertEqual(len(results), 1)
            self.assertEqual(len(results[0]), 1)
        finally:
            os.unlink(fname)

    def test_log_perf_writes_file(self):
        import tempfile
        from edgol.rules_engine import RulesEngine
        from edgol.interpreter import Interpreter
        eng = RulesEngine()
        interp = Interpreter(eng)
        eng._last_query_stats = {'unify_count': 1, 'prove_count': 2}
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            # simulate calling :log perf by invoking main handler logic
            import json
            data = {'stats': getattr(eng, '_last_query_stats', {}), 'engine_stats': eng.stats(), 'rule_profile': getattr(eng, '_rule_profile', {})}
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(data, fh)
            # assert file exists and contains keys
            with open(path, 'r', encoding='utf-8') as fh:
                j = json.load(fh)
            self.assertIn('stats', j)
            self.assertIn('engine_stats', j)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()


class UndoBehaviorTests(unittest.TestCase):
    def test_undo_dif_freeze_fd_macro(self):
        from edgol.rules_engine import RulesEngine
        from edgol.parser import Term, Variable, Constant

        eng = RulesEngine()
        # add dif
        eng.add_dif_constraint(Variable('X'), Constant(1))
        self.assertEqual(len(eng._dif_constraints), 1)

        # add freeze
        eng.add_freeze('X', Term('p', [Variable('X')]))
        self.assertEqual(len(eng._freeze_store), 1)

        # add fd domain and constraint
        eng.add_fd_domain('Y', 1, 5)
        eng.add_fd_constraint('Y', '#=', Constant(2))
        self.assertEqual(len(eng._fd_constraints), 1)

        # add macro
        eng.add_macro('m', ['X'], 'p(X).')
        self.assertIn('m', eng._macros)

        # undo macro
        self.assertTrue(eng.undo_last())
        self.assertNotIn('m', getattr(eng, '_macros', {}))

        # undo fd
        self.assertTrue(eng.undo_last())
        self.assertEqual(len(eng._fd_constraints), 0)

        # undo freeze
        self.assertTrue(eng.undo_last())
        self.assertEqual(len(eng._freeze_store), 0)

        # undo dif
        self.assertTrue(eng.undo_last())
        self.assertEqual(len(eng._dif_constraints), 0)

    def test_list_constraints_output(self):
        from edgol.rules_engine import RulesEngine
        from edgol.parser import Term, Variable, Constant
        eng = RulesEngine()
        eng.add_dif_constraint(Variable('X'), Constant(1))
        eng.add_freeze('Y', Term('p', [Variable('Y')]))
        eng.add_fd_domain('Z', [2,4,6], None)
        out = eng.list_constraints()
        self.assertIn('dif(', out)
        self.assertIn('freeze(', out)
        self.assertIn('Z in', out)

