import unittest
from edgol.rules_engine import RulesEngine
from edgol.parser import Term, Constant

class IndexingTests(unittest.TestCase):
    def test_set_trie_index_mode(self):
        eng = RulesEngine()
        # add some facts
        for i in range(100):
            eng.add_fact(Term('p', [Constant(i)]))
        # default index mode
        s = eng.stats()
        self.assertEqual(s['index_mode'], 'hash')
        # switch to trie
        eng.set_index_mode('trie')
        s2 = eng.stats()
        self.assertEqual(s2['index_mode'], 'trie')
        # query should find 100 facts via trie lookup
        from edgol.parser import Variable
        key = Term('p', [Variable('X')])
        results = eng.query(key)
        # results are list of bindings (empty for ground facts) so expect 100 matches
        self.assertEqual(len(results), 100)

if __name__ == '__main__':
    unittest.main()
