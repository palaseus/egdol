import io
from contextlib import redirect_stdout
from edgol.main import run_session, repl
from edgol.rules_engine import RulesEngine


def test_pretty_facts_and_rules_and_stats(capsys):
    eng = RulesEngine()
    run_session('fact: m:fa(x). rule: m:hr(a) => b(a).', eng)
    f = io.StringIO()
    # use repl-like printing helpers by invoking run_session and then printing :stats
    _, results = run_session('? fa(x).', eng)
    assert results is not None
    s = eng.stats()
    assert s['num_facts'] >= 1
 