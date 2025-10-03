from egdol.main import run_session, format_profile_summary
from egdol.rules_engine import RulesEngine
from egdol.interpreter import Interpreter


def test_format_profile_summary_contains_expected_keys():
    eng = RulesEngine()
    # set up a rule and fact, then query to generate profile data
    txt = 'fact: q(a). rule: p(X) => q(X). ? p(a).'
    eng, results = run_session(txt, eng)
    interp = Interpreter(eng)
    s = format_profile_summary(eng, interp)
    assert 'Interpreter counters:' in s
    assert 'Per-rule profile:' in s
    # should mention p/1 since we ran it
    assert 'p/1' in s


def test_format_profile_summary_top_n():
    eng = RulesEngine()
    txt = 'fact: q(a). rule: p(X) => q(X). rule: r(X) => q(X). ? p(a).'
    eng, results = run_session(txt, eng)
    interp = Interpreter(eng)
    out_all = format_profile_summary(eng, interp)
    out_top1 = format_profile_summary(eng, interp, top_n=1)
    # top_n output should be no longer than full (simple check)
    assert len(out_top1) <= len(out_all)
    assert 'Interpreter counters:' in out_top1
