
from egdol.rules_engine import RulesEngine


def test_export_import_list_disjunction_roundtrip(tmp_path):
    engine = RulesEngine()
    src = """
    p([a,b,c]).
    q(X) :- p([X|Ys]) ; r(X).
    s(X) :- X \\== a.
    """
    f = tmp_path / "in.pl"
    f.write_text(src)

    # import
    engine.import_prolog(f.as_posix())

    out = tmp_path / "out.pl"
    engine.export_prolog(out.as_posix())

    out_txt = out.read_text()

    # Ensure that lists are exported using bracket notation and disjunctions
    assert '[a, b, c]' in out_txt
    assert ' ; ' in out_txt or ';' in out_txt
    assert '\\==' in out_txt or '\\==' in out_txt
