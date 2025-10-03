from egdol.lexer import Lexer
from egdol.parser import Parser
from egdol.rules_engine import RulesEngine
from egdol.interpreter import Interpreter


def test_comments_are_ignored():
    text = '# this is a comment\nfact: p(a).\n% another comment\n? p(a).\n'
    tokens = Lexer(text).tokenize()
    nodes = Parser(tokens).parse()
    assert any(getattr(n, 'term', None) is not None for n in nodes)


def test_dif_constraint():
    eng = RulesEngine()
    interp = Interpreter(eng)
    # assert two facts
    eng.add_fact(Parser(Lexer('fact: a(1).').tokenize()).parse()[0].term)
    # dif(X,1) should hold for X not yet bound
    res = list(interp._builtin_dif(Parser(Lexer('dif(X,1).').tokenize()).parse()[0].term, {}))
    assert res


def test_freeze_delays_goal():
    eng = RulesEngine()
    interp = Interpreter(eng)
    # freeze(X, goal) should record freeze
    term = Parser(Lexer('freeze(X, p(X)).').tokenize()).parse()[0].term
    res = list(interp._builtin_freeze(term, {}))
    assert res
