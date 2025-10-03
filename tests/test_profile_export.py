import json

from egdol.interpreter import Interpreter
from egdol.main import analyze_profile, export_profile_json, format_profile_summary
from egdol.rules_engine import RulesEngine


def make_engine_with_profile():
    engine = RulesEngine()
    # synthetic profile for testing
    engine._rule_profile = {
        ('slow', 2): {'calls': 10, 'total_time': 0.5, 'avg_time': 0.05, 'total_unify': 100, 'total_unify_time': 0.02, 'avg_unify': 0.0002},
        ('hot', 3): {'calls': 12000, 'total_time': 1.2, 'avg_time': 0.0001, 'total_unify': 50000, 'total_unify_time': 10.0, 'avg_unify': 0.0002},
        ('ok', 1): {'calls': 30, 'total_time': 0.03, 'avg_time': 0.001, 'total_unify': 300, 'total_unify_time': 0.03, 'avg_unify': 0.0001},
        ('weird', 1): {'calls': 50, 'total_time': 0.2, 'avg_time': 0.004, 'total_unify': 50, 'total_unify_time': 0.5, 'avg_unify': 0.01},
    }
    interp = Interpreter(engine)
    interp.unify_count = 100000
    interp.prove_count = 5000
    return engine, interp


def test_format_profile_structured():
    engine, interp = make_engine_with_profile()
    text, structured = format_profile_summary(engine, interp, structured=True)
    assert 'Interpreter counters:' in text
    assert 'Per-rule profile:' in text
    assert 'slow/2' in text or any(r['name'] == 'slow' and r['arity'] == 2 for r in structured['rules'])
    assert structured['interp']['unify_count'] == 100000
    assert structured['interp']['prove_count'] == 5000
    # ensure required fields
    for r in structured['rules']:
        assert 'name' in r and 'arity' in r and 'calls' in r and 'avg_time' in r and 'avg_unify_time' in r


def test_export_profile_json(tmp_path):
    engine, interp = make_engine_with_profile()
    p = tmp_path / 'prof.json'
    path = export_profile_json(engine, interp, str(p))
    assert path == str(p)
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    assert 'interp' in data and 'rules' in data
    assert data['interp']['unify_count'] == 100000


def test_analyze_profile_flags():
    engine, interp = make_engine_with_profile()
    text, analysis = analyze_profile(engine, interp, top_n=3, json_out=False)
    # analysis should flag slow rule and possibly weird unify
    assert 'Top slowest rules:' in text
    # expect 'slow/2' in top slowest
    assert any(t['name'] == 'slow' and t['arity'] == 2 for t in analysis['top_slowest'])
    # expect suspect_unify includes 'weird' because its avg_unify (0.01) is large
    assert any(s['name'] == 'weird' for s in analysis['suspect_unify'])
