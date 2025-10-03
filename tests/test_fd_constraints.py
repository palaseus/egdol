import unittest

from egdol.interpreter import Interpreter
from egdol.lexer import Lexer
from egdol.parser import Constant, Parser, Term, Variable
from egdol.rules_engine import RulesEngine


class FDTests(unittest.TestCase):
    def test_parse_in_range(self):
        src = "X in 1..5."
        toks = Lexer(src).tokenize()
        p = Parser(toks)
        # parse_domain_if_present isn't exposed in parse; use parse_expression flow
        node = p.parse_domain_if_present()
        self.assertIsNotNone(node)
        self.assertEqual(node.name, 'in_range')
        self.assertIsInstance(node.args[0], Variable)
        self.assertIsInstance(node.args[1], Constant)
        self.assertEqual(node.args[1].value, 1)
        self.assertEqual(node.args[2].value, 5)

    def test_fd_builtin_basic(self):
        eng = RulesEngine()
        interp = Interpreter(eng)
        # assert domain via builtin in script: X in 1..5 and X #= 3
        # emulate proving of builtins
        term_domain = Term('in_range', [Variable('X'), Constant(1), Constant(5)])
        # add to engine domain
        eng.add_fd_domain('X', 1, 5)
        # add fd constraint X #= 3
        eng.add_fd_constraint('X', '#=', Constant(3))
        ok = eng.check_fd_consistency()
        self.assertTrue(ok)
        self.assertEqual(eng._fd_domains['X'], {3})
        # compatibility tuple
        self.assertEqual(eng.get_fd_range('X'), (3, 3))

    def test_fd_propagation_arith(self):
        eng = RulesEngine()
        interp = Interpreter(eng)
        # Y in 1..5, X #= Y+1 -> X in 2..6 and intersects with any existing
        eng.add_fd_domain('Y', 1, 5)
        eng.add_fd_constraint('X', '#=', Term('+', [Variable('Y'), Constant(1)]))
        ok = eng.check_fd_consistency()
        self.assertTrue(ok)
        # X domain should be 2..6
        self.assertEqual(eng._fd_domains.get('X'), set(range(2, 7)))
        # compatibility tuple should be (2,6)
        self.assertEqual(eng.get_fd_range('X'), (2, 6))

if __name__ == '__main__':
    unittest.main()
