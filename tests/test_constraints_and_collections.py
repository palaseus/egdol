from egdol.interpreter import Interpreter
from egdol.lexer import Lexer
from egdol.parser import Parser
from egdol.rules_engine import RulesEngine


def test_dif_rejects_early():
    eng = RulesEngine()
    interp = Interpreter(eng)
    # register dif constraint between X and 1
    term = Parser(Lexer('dif(X,1).').tokenize()).parse()[0].term
    out = list(interp._builtin_dif(term, {}))
    assert out
    # now try to bind X to 1 via unify through a fake goal
    s = interp._extend_subst({}, 'X', Parser(Lexer('1').tokenize()).parse()[0].term)
    # binding X to 1 should be rejected because dif(X,1) exists
    assert s is None


def test_freeze_wakes_on_bind():
    eng = RulesEngine()
    interp = Interpreter(eng)
    # freeze X until p(X) is provable; add rule p(1) later
    term = Parser(Lexer('freeze(X, p(X)).').tokenize()).parse()[0].term
    assert list(interp._builtin_freeze(term, {}))
    # now bind X to 1 and ensure freeze tries to prove p(1) and fails (no p(1))
    s = interp._extend_subst({}, 'X', Parser(Lexer('1').tokenize()).parse()[0].term)
    # since p(1) not in DB, binding should be rejected (freeze enforces goal)
    assert s is None
    # add p(1) and try again
    eng.add_fact(Parser(Lexer('fact: p(1).').tokenize()).parse()[0].term)
    s2 = interp._extend_subst({}, 'X', Parser(Lexer('1').tokenize()).parse()[0].term)
    assert s2 is not None


def test_bagof_collects():
    eng = RulesEngine()
    interp = Interpreter(eng)
    eng.add_fact(Parser(Lexer('fact: q(a).').tokenize()).parse()[0].term)
    eng.add_fact(Parser(Lexer('fact: q(b).').tokenize()).parse()[0].term)
    term = Parser(Lexer('bagof(X, q(X), L).').tokenize()).parse()[0].term
    outs = list(interp._builtin_bagof(term, {}))
    assert outs
