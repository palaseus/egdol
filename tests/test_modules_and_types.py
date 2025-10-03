import unittest

from egdol.interpreter import Interpreter
from egdol.lexer import Lexer
from egdol.parser import Constant, Parser, Term
from egdol.rules_engine import RulesEngine


class ModuleAndTypeTests(unittest.TestCase):
    def test_module_declaration_and_qualified_query(self):
        txt = 'module: m1. fact: human(alice).'
        tokens = Lexer(txt).tokenize()
        nodes = Parser(tokens).parse()
        engine = RulesEngine()
        interp = Interpreter(engine)
        # apply module declaration: Parser returns Module nodes with .name
        for n in nodes:
            if hasattr(n, 'name') and getattr(n, 'name', None) == 'm1':
                engine.current_module = n.name
            if hasattr(n, 'term') and isinstance(n.term, Term):
                engine.add_fact(n.term)
        # query qualified
        q = Term('m1:human', [Constant('alice')])
        res = interp.query(q)
        self.assertEqual(len(res), 1)

    def test_type_checking_atom_length(self):
        engine = RulesEngine()
        interp = Interpreter(engine)
        interp.type_checking = True
        # atom_length should accept constant
        from egdol.parser import Constant, Term, Variable
        q = Term('atom_length', [Constant('hello'), Variable('N')])
        res = interp.query(q)
        self.assertEqual(len(res), 1)
        # atom_length on list should raise TypeError when type_checking enabled
        lst = Term('.', [Constant('a'), Constant('[]')])
        q2 = Term('atom_length', [lst, Variable('N')])
        with self.assertRaises(TypeError):
            # we will call the builtin directly via query path which now should raise
            interp.query(q2)


if __name__ == '__main__':
    unittest.main()
