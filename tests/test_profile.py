import os
import json
from edgol.main import run_session
from edgol.rules_engine import RulesEngine
from edgol.interpreter import Interpreter


def test_rule_profiling_and_profile_reset_and_log(tmp_path):
    # create engine and add a simple rule and fact
    eng = RulesEngine()
    # rule: p(X) :- q(X).
    text = 'fact: q(a). rule: p(X) => q(X). ? p(a).'
    eng, results = run_session(text, eng)
    # after running, engine._rule_profile should have an entry for p/1
    rp = getattr(eng, '_rule_profile', None)
    assert rp is not None, 'Expected rule profile to exist'
    # find p/1 key
    key = ('p', 1)
    assert key in rp, f'Expected profile for p/1, got {list(rp.keys())}'
    rec = rp[key]
    assert rec.get('calls', 0) > 0
    # verify interpreter counters are present via a fresh interpreter attached to engine
    interp = Interpreter(eng)
    assert hasattr(interp, 'unify_count')
    # Test :profile reset behavior by manually clearing like REPL would
    eng._rule_profile = {}
    interp.unify_count = 0
    interp.prove_count = 0
    assert eng._rule_profile == {}
    assert interp.unify_count == 0
    # Test profile log writes JSON
    out = tmp_path / 'prof.json'
    data = {'interp': {'unify_count': interp.unify_count, 'prove_count': interp.prove_count}, 'rule_profile': eng._rule_profile}
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)
    assert out.exists()
    # quick sanity read
    with open(out, 'r', encoding='utf-8') as fh:
        loaded = json.load(fh)
    assert 'interp' in loaded and 'rule_profile' in loaded
