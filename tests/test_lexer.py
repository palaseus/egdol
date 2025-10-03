import unittest

from egdol.lexer import Lexer, Token, LexerError


class LexerTests(unittest.TestCase):
    def test_simple_fact_tokens(self):
        src = 'fact: human(socrates).'
        tokens = Lexer(src).tokenize()
        types = [t.type for t in tokens]
        values = [t.value for t in tokens]
        self.assertIn('IDENT', types)
        self.assertIn('COLON', types)
        self.assertIn('LPAREN', types)
        self.assertIn('RPAREN', types)
        self.assertIn('DOT', types)
        self.assertIn('IDENT', types)
        # check sequence roughly
        self.assertEqual(values[0], 'fact')
        self.assertEqual(values[1], ':')

    def test_rule_arrow_and_query(self):
        src = 'rule: mortal(X) => human(X). ? mortal(socrates).'
        tokens = Lexer(src).tokenize()
        types = [t.type for t in tokens]
        self.assertIn('ARROW', types)
        self.assertIn('QUERY', types)

    def test_invalid_char_raises(self):
        src = 'fact: human(@).'
        with self.assertRaises(LexerError):
            Lexer(src).tokenize()


if __name__ == '__main__':
    unittest.main()
