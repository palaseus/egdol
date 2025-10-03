import unittest

from egdol.lexer import Lexer
from egdol.parser import Parser, Fact, Rule, Query, Term, Variable, Constant, ParseError


class ParserTests(unittest.TestCase):
    def parse_nodes(self, src):
        toks = Lexer(src).tokenize()
        return Parser(toks).parse()

    def test_parse_fact(self):
        nodes = self.parse_nodes('fact: human(socrates).')
        self.assertEqual(len(nodes), 1)
        self.assertIsInstance(nodes[0], Fact)
        self.assertEqual(repr(nodes[0].term), 'human(socrates)')

    def test_parse_rule(self):
        nodes = self.parse_nodes('rule: mortal(X) => human(X).')
        self.assertEqual(len(nodes), 1)
        self.assertIsInstance(nodes[0], Rule)
        rule = nodes[0]
        self.assertEqual(repr(rule.head), 'mortal(X)')
        self.assertEqual(len(rule.body), 1)
        self.assertEqual(repr(rule.body[0]), 'human(X)')

    def test_parse_query(self):
        nodes = self.parse_nodes('? mortal(socrates).')
        self.assertEqual(len(nodes), 1)
        self.assertIsInstance(nodes[0], Query)
        self.assertEqual(repr(nodes[0].term), 'mortal(socrates)')

    def test_invalid_syntax_raises(self):
        with self.assertRaises(ParseError):
            self.parse_nodes('fact human(socrates).')

    def test_nested_term_arg(self):
        nodes = self.parse_nodes('fact: parent(father(john), mary).')
        self.assertEqual(len(nodes), 1)
        self.assertEqual(repr(nodes[0].term), 'parent(father(john), mary)')

    def test_macro_validate_valid_and_invalid(self):
        from egdol.rules_engine import RulesEngine
        eng = RulesEngine()
        # valid macro: params declared and used
        eng.add_macro('q', ['X'], 'fact: p(X).')
        res = eng.validate_macro('q')
        self.assertIsNone(res)
        # invalid macro: uses Y but not declared
        eng.add_macro('bad', ['X'], 'fact: p(X, Y).')
        res2 = eng.validate_macro('bad')
        self.assertIsNotNone(res2)
        self.assertIn('unbound variables', res2)

    def test_macro_ast_multi_statement_and_hygienic(self):
        from egdol.rules_engine import RulesEngine
        from egdol.parser import Term, Variable, Constant
        eng = RulesEngine()
        # macro with two statements: a fact and a rule using X
        eng.add_macro('m', ['X'], 'fact: p(X). rule: q(X) => p(X).')
        # validate
        v = eng.validate_macro('m')
        self.assertIsNone(v)
        # expansion: when invoked with argument 'a', ensure produced text contains two statements
        out = eng.expand_macros('m(a)')
        self.assertIn('p(a).', out)
        self.assertIn('q(a) :- p(a).', out)


if __name__ == '__main__':
    unittest.main()
