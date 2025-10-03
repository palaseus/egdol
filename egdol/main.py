"""CLI entrypoint and session runner for egdol"""
import argparse
import logging
import os
from typing import List

from .interpreter import Interpreter, MaxDepthExceededError
from .lexer import Lexer
from .parser import Fact, Parser, Query, Rule
from .rules_engine import RulesEngine

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')


def run_session(text: str, engine: RulesEngine = None):
    """Parse and execute statements in `text` against a RulesEngine.

    Returns the engine used and a list of query results. Each query result is
    the raw list of binding dicts returned by Interpreter.query.
    """
    engine_provided = engine is not None
    if engine is None:
        engine = RulesEngine()
    # prefer an existing interpreter attached to engine (tests may have created one)
    last_interp = getattr(engine, '_last_interp', None)
    if last_interp is None:
        interp = Interpreter(engine)
        interp._internal = True
    else:
        interp = last_interp
    # apply settings if engine contains saved settings
    if hasattr(engine, '_settings'):
        s = engine._settings
        if 'max_depth' in s:
            interp.max_depth = s['max_depth']
        if 'trace_level' in s:
            interp.trace_level = s['trace_level']
    # expand macros if engine provides them
    if hasattr(engine, 'expand_macros'):
        try:
            text = engine.expand_macros(text)
        except Exception:
            pass
    tokens = Lexer(text).tokenize()
    nodes = Parser(tokens).parse()
    results: List[List[dict]] = []
    for node in nodes:
        if isinstance(node, Fact):
            engine.add_fact(node.term)
        elif isinstance(node, Rule):
            engine.add_rule(node)
        elif isinstance(node, Query):
            try:
                res = interp.query(node.term)
            except Exception:
                raise
            results.append(res)
    # update engine settings for persistence under 'settings' (older tests expect this key)
    engine._settings = {'max_depth': interp.max_depth, 'trace_level': interp.trace_level}
    engine.settings = engine._settings
    stats = {'unify_count': interp.unify_count, 'prove_count': interp.prove_count}
    # store last query stats on engine for REPL :stats display
    try:
        engine._last_query_stats = stats
    except Exception:
        pass
    # return shape: if caller provided an engine and that interpreter was
    # constructed externally (no _internal flag), include stats (3-tuple).
    if engine_provided and not getattr(interp, '_internal', False):
        return engine, results, stats
    return engine, results


def format_profile_summary(engine: RulesEngine, interp: Interpreter, top_n: int = None, sort_by: str = 'avg_time', structured: bool = False):
    """Return a formatted profiling summary string for the given engine and interpreter.

    If `structured` is True, return a tuple (text_summary, structured_dict).
    structured_dict contains:
      { 'interp': {unify_count, prove_count}, 'rules': [ {name, arity, calls, avg_time, avg_unify_time}, ... ] }

    top_n: if provided, only include the top N rules sorted by `sort_by` ("avg_time" or "avg_unify").
    """
    interp_stats = {'unify_count': getattr(interp, 'unify_count', 0), 'prove_count': getattr(interp, 'prove_count', 0)}
    rp = getattr(engine, '_rule_profile', {}) or {}

    # build structured dict
    structured_data = {'interp': interp_stats, 'rules': []}
    for (name, ar), rec in rp.items():
        structured_data['rules'].append({
            'name': name,
            'arity': ar,
            'calls': rec.get('calls', 0),
            'avg_time': rec.get('avg_time', 0.0),
            'avg_unify_time': rec.get('avg_unify') if rec.get('avg_unify') is not None else None,
        })

    # sorting
    if sort_by == 'avg_unify':
        keyfn = lambda r: (r.get('avg_unify_time') or 0.0)
    else:
        keyfn = lambda r: (r.get('avg_time') or 0.0)
    items_sorted = sorted(structured_data['rules'], key=keyfn, reverse=True)
    if top_n is not None:
        items_sorted = items_sorted[:int(top_n)]

    # text summary
    lines = []
    lines.append('Interpreter counters:')
    lines.append(f"  unify_count = {interp_stats['unify_count']}")
    lines.append(f"  prove_count = {interp_stats['prove_count']}")
    lines.append('Per-rule profile:')
    if not items_sorted:
        lines.append('  No rule profile data')
    else:
        for rec in items_sorted:
            calls = rec.get('calls', 0)
            avg_time = rec.get('avg_time', 0.0)
            avg_unify = rec.get('avg_unify_time', None)
            avg_unify_str = (f"{avg_unify:.6f}s" if avg_unify is not None else 'N/A')
            lines.append(f"  {rec.get('name')}/{rec.get('arity')}: calls={calls}, avg_time={avg_time:.6f}s, avg_unify_time={avg_unify_str}")

    if structured:
        return '\n'.join(lines), structured_data
    return '\n'.join(lines)


def export_profile_json(engine: RulesEngine, interp: Interpreter, path: str):
    """Write structured profile JSON to path (pretty-printed).
    Returns the path on success.
    """
    import json
    _, structured = format_profile_summary(engine, interp, structured=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(structured, fh, indent=2)
    return path


def analyze_profile(engine: RulesEngine, interp: Interpreter, top_n: int = 3, json_out: bool = False):
    """Analyze structured profile data and return (text, structured_analysis) if json_out False,
    else return structured_analysis (and text will be JSON string).

    structured_analysis format:
      {'top_slowest': [ {name, arity, avg_time, calls}, ... ], 'suspect_unify': [ {name, arity, avg_unify_time, calls, score}, ... ] }
    """
    import statistics
    _, structured = format_profile_summary(engine, interp, structured=True)
    rules = structured.get('rules', [])
    analysis = {'top_slowest': [], 'suspect_unify': []}
    if not rules:
        text = 'No profiling data available.'
        return (text, analysis) if not json_out else analysis

    # top N slowest by avg_time
    by_time = sorted(rules, key=lambda r: r.get('avg_time') or 0.0, reverse=True)
    for r in by_time[:top_n]:
        analysis['top_slowest'].append({'name': r['name'], 'arity': r['arity'], 'avg_time': r['avg_time'], 'calls': r['calls']})

    # analyze avg_unify anomalies: compute global mean of avg_unify_time for rules that have it
    unify_times = [r.get('avg_unify_time') for r in rules if r.get('avg_unify_time') is not None]
    mean_unify = statistics.mean(unify_times) if unify_times else None
    mean_calls = statistics.mean([r.get('calls', 0) for r in rules]) if rules else 0
    if mean_unify is not None:
        for r in rules:
            au = r.get('avg_unify_time')
            calls = r.get('calls', 0)
            if au is None:
                continue
            # flag if avg_unify is > 2x mean OR if avg_unify > mean and calls is significantly above average
            if au > 2 * mean_unify or (au > mean_unify and calls > 2 * mean_calls):
                analysis['suspect_unify'].append({'name': r['name'], 'arity': r['arity'], 'avg_unify_time': au, 'calls': calls, 'factor': au / mean_unify})

    # build human-readable text
    lines = []
    if analysis['top_slowest']:
        lines.append('Top slowest rules:')
        for t in analysis['top_slowest']:
            lines.append(f"⚠ Rule {t['name']}/{t['arity']} is slow (avg_time={t['avg_time']:.6f}s, calls={t['calls']})")
    if analysis['suspect_unify']:
        lines.append('\nRules with unusually high avg_unify_time relative to global mean:')
        for s in analysis['suspect_unify']:
            lines.append(f"⚡ Rule {s['name']}/{s['arity']} avg_unify_time={s['avg_unify_time']:.6f}s, calls={s['calls']}, factor={s['factor']:.2f}x")

    text = '\n'.join(lines) if lines else 'No significant performance issues detected.'
    return (text, analysis) if not json_out else analysis


# load_file helper: load facts/rules from a file into the given engine.
def load_file(engine: RulesEngine, path: str):
    """Load facts/rules from a file into the given engine.

    Returns the tuple (engine, results) from running the file contents.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    rv = run_session(text, engine)
    # run_session may return (engine, results) or (engine, results, stats).
    if isinstance(rv, tuple) and len(rv) == 3:
        return rv[0], rv[1]
    return rv


def repl():
    print("egdol REPL - type 'exit' to quit")
    engine = RulesEngine()
    interp = Interpreter(engine)
    # setup readline autocompletion for commands and known predicate names
    try:
        if 'readline' in globals() and readline is not None:
            commands = [':load', ':save', ':dump', ':run', ':assert', ':retract', ':facts', ':rules', ':stats', ':undo', ':benchmark', ':trace', ':profile', ':help']
            def completer(text, state):
                opts = [c for c in commands + list(interp.builtins.keys()) if c.startswith(text)]
                if state < len(opts):
                    return opts[state]
                return None
            readline.set_completer(completer)
            readline.parse_and_bind('tab: complete')
    except Exception:
        pass
    buffer = ''
    history = []
    profile_mode = False
    # undo stack: list of callables to undo last actions (max 20)
    undo_stack = []
    undo_limit = 20
    while True:
        try:
            line = input('egdol> ')
        except EOFError:
            break
        if not line:
            continue
        cmd = line.strip()
        if cmd in (':quit', ':exit', 'exit', 'quit'):
            break

        if cmd.startswith(':load ') or cmd.startswith(':l '):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print('Usage: :load <filename>')
                continue
            path = parts[1].strip()
            try:
                load_file(engine, path)
                print(f'Loaded {path}')
            except Exception as e:
                print('Error loading file:', e)
            continue

        if cmd.startswith(':export '):
            parts = cmd.split(None, 3)
            if len(parts) >= 3 and parts[1] == 'dot':
                path = parts[2]
                try:
                    engine.export_to_dot(path)
                    print(f'Exported DOT to {path}')
                except Exception as e:
                    print('Export failed:', e)
                continue
            if len(parts) >= 3 and parts[1] == 'script':
                path = parts[2]
                try:
                    # write facts/rules as script with comments
                    with open(path, 'w', encoding='utf-8') as fh:
                        for f in engine.facts:
                            fh.write(f'fact: {f}.\n')
                        for r in engine.rules:
                            fh.write(f'rule: {r.head} => {", ".join(map(str, r.body))}.\n')
                    print(f'Exported script to {path}')
                except Exception as e:
                    print('Export failed:', e)
                continue
            if len(parts) >= 3 and parts[1] == 'json':
                path = parts[2]
                try:
                    engine.export_json(path)
                    print(f'Exported JSON to {path}')
                except Exception as e:
                    print('Export failed:', e)
                continue
            if len(parts) >= 3 and parts[1] == 'prolog':
                path = parts[2]
                try:
                    with open(path, 'w', encoding='utf-8') as fh:
                        for f in engine.facts:
                            fh.write(f'{f}.\n')
                        for r in engine.rules:
                            fh.write(f'{r.head} :- {", ".join(map(str, r.body))}.\n')
                    print(f'Exported Prolog to {path}')
                except Exception as e:
                    print('Export failed:', e)
                continue
            print('Usage: :export dot <file> | :export script <file>')
            continue

        if cmd.startswith(':module '):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print('Usage: :module <name>')
                continue
            m = parts[1].strip()
            engine.current_module = m
            print(f'Current module set to {m}')
            continue

        if cmd.startswith(':run '):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print('Usage: :run <file>')
                continue
            path = parts[1].strip()
            try:
                # run script and print results
                if hasattr(interp, 'run_script'):
                    try:
                        with open(path, 'r', encoding='utf-8') as fh:
                            txt = fh.read()
                        if hasattr(engine, 'expand_macros'):
                            txt = engine.expand_macros(txt)
                        run_session(txt, engine)
                    except Exception:
                        interp.run_script(path)
                else:
                    # fallback: load file and run session
                    load_file(engine, path)
                print(f'Ran {path}')
            except Exception as e:
                print('Error running file:', e)
            continue

        if cmd.startswith(':save '):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print('Usage: :save <filename>')
                continue
            path = parts[1].strip()
            try:
                # persist current interpreter settings if available
                engine._settings = {'max_depth': getattr(interp, 'max_depth', None), 'trace_level': getattr(interp, 'trace_level', None)}
                engine.save(path)
                print(f'Saved to {path}')
            except Exception as e:
                print('Error saving file:', e)
            continue

        if cmd.startswith(':dump '):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print('Usage: :dump <filename>')
                continue
            path = parts[1].strip()
            try:
                engine.dump(path)
                print(f'Dumped to {path}')
            except Exception as e:
                print('Error dumping file:', e)
            continue

        if cmd == ':facts':
            for f in engine.facts:
                print(str(f))
            continue

        if cmd == ':constraints':
            try:
                out = engine.list_constraints()
                print(out)
            except Exception as e:
                print('Failed to list constraints:', e)
            continue

        if cmd.startswith(':assert '):
            # quick assert: accept a term string like 'fact: human(bob).' or 'rule: ...'
            stmt = cmd[len(':assert '):].strip()
            try:
                tokens = Lexer(stmt).tokenize()
                nodes = Parser(tokens).parse()
                for n in nodes:
                    if hasattr(n, 'term') and isinstance(n, type(n)):
                        # Fact or Query or Rule
                        if isinstance(n, type(n)) and hasattr(n, 'term'):
                            engine.add_fact(n.term)
                    if hasattr(n, 'head'):
                        engine.add_rule(n)
                print('Asserted')
            except Exception as e:
                print('Assert failed:', e)
            continue

        if cmd.startswith(':retract '):
            term_str = cmd[len(':retract '):].strip()
            try:
                tokens = Lexer(term_str).tokenize()
                nodes = Parser(tokens).parse()
                if not nodes:
                    print('Nothing to retract')
                    continue
                node = nodes[0]
                # use string form to match
                if hasattr(node, 'term'):
                    term = node.term
                elif hasattr(node, 'head'):
                    term = node.head
                else:
                    print('Unsupported retract target')
                    continue
                matches = engine.find_matches(term)
                if not matches:
                    print('No matching items')
                    continue
                print(f'{len(matches)} match(es) preview:')
                for i, m in enumerate(matches, start=1):
                    print(f'{i}: {m}')
                yn = input('Retract these? (y/n): ')
                if yn.lower() in ('y', 'yes'):
                    removed = engine.retract(term)
                    # push undo action
                    def _undo(removed_items):
                        for it in removed_items:
                            if isinstance(it, Rule):
                                engine.add_rule(it)
                            else:
                                engine.add_fact(it)
                    undo_stack.append(lambda removed=removed: _undo(removed))
                    if len(undo_stack) > undo_limit:
                        undo_stack.pop(0)
                    print(f'Retracted {len(removed)} items')
                else:
                    print('Aborted retract')
            except Exception as e:
                print('Retract failed:', e)
            continue

        if cmd == ':undo':
            # first try undo stack actions
            if undo_stack:
                action = undo_stack.pop()
                try:
                    action()
                    print('Undo successful')
                except Exception as e:
                    print('Undo failed:', e)
                continue
            # fall back to engine-level undo (constraints/macros)
            if hasattr(engine, 'undo_last'):
                ok = engine.undo_last()
                if ok:
                    print('Engine undo successful')
                else:
                    print('Nothing to undo')
                continue
            print('Nothing to undo')
            continue

        if cmd.startswith(':benchmark '):
            parts = cmd.split(None, 2)
            if len(parts) < 3:
                print('Usage: :benchmark <query> <N>')
                continue
            query_str = parts[1]
            try:
                runs = int(parts[2])
            except Exception:
                print('Invalid N')
                continue
            import time
            t0 = time.perf_counter()
            for i in range(runs):
                run_session(query_str, engine)
            t1 = time.perf_counter()
            print(f'Ran {runs} iterations in {t1-t0:.6f}s, avg {((t1-t0)/runs):.6f}s')
            continue
            continue
        if cmd == ':undo':
            if not undo_stack:
                print('Nothing to undo')
                continue
            action = undo_stack.pop()
            try:
                action()
                print('Undo successful')
            except Exception as e:
                print('Undo failed:', e)
            continue
        if cmd.startswith(':benchmark '):
            parts = cmd.split(None, 2)
            if len(parts) < 3:
                print('Usage: :benchmark <query> <N>')
                continue
            query_str = parts[1]
            try:
                runs = int(parts[2])
            except Exception:
                print('Invalid N')
                continue
            import time
            t0 = time.perf_counter()
            for i in range(runs):
                run_session(query_str, engine)
            t1 = time.perf_counter()
            print(f'Ran {runs} iterations in {t1-t0:.6f}s, avg {((t1-t0)/runs):.6f}s')
            continue
        if cmd == ':rules':
            for r in engine.rules:
                head = str(r.head)
                body = ', '.join(map(str, r.body))
                print(f"{head} :- {body}")
            continue
        if cmd.startswith(':import '):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print('Usage: :import <file>')
                continue
            path = parts[1]
            try:
                load_file(engine, path)
                print(f'Imported {path}')
            except Exception as e:
                print('Import failed:', e)
            continue
        if cmd.startswith(':optimize'):
            parts = cmd.split()
            if len(parts) == 1:
                print('Optimization mode: current index sizes:', engine.stats())
                continue
            mode = parts[1]
            print(f'Set optimization mode to {mode} (no-op placeholder)')
            continue
        if cmd.startswith(':bind '):
            # simple mapping: ':bind Ctrl-R :history'
            parts = cmd.split(None, 2)
            if len(parts) < 3:
                print('Usage: :bind <Key> <Command>')
                continue
            key, mapped = parts[1], parts[2]
            # store in engine for session
            try:
                engine._bindings = getattr(engine, '_bindings', {})
                engine._bindings[key] = mapped
                print(f'Bound {key} to {mapped}')
            except Exception as e:
                print('Bind failed:', e)
            continue
        if cmd == ':help' or cmd == ':h':
            print('Commands: :load <file>, :save <file>, :dump <file>, :set max_depth N, :get max_depth, :trace, :profile on|off, :history')
            print('Commands: :load <file>, :quit, :help')
            continue
        if cmd.startswith(':trace'):
            parts = cmd.split()
            if len(parts) == 1:
                print(f'Current trace level: {getattr(interp, "trace_level", 0)}')
                continue
            arg = parts[1]
            if arg == 'on':
                interp.trace_level = 1
            elif arg == 'off':
                interp.trace_level = 0
            else:
                try:
                    interp.trace_level = int(arg)
                except Exception:
                    print('Usage: :trace on|off|<level>')
                    continue
            print(f'Set trace level to {interp.trace_level}')
            # set logging level
            if interp.trace_level >= 2:
                logging.getLogger().setLevel(logging.INFO)
            else:
                logging.getLogger().setLevel(logging.WARNING)
            continue
        if cmd == ':check':
            issues = engine.check_constraints()
            if not issues:
                print('No constraint issues')
            else:
                for it in issues:
                    print('Issue:', it)
            # print FD domains if present
            if hasattr(engine, '_fd_domains'):
                print('FD domains:')
                for v, dom in engine._fd_domains.items():
                    print(f'  {v}: {sorted(list(dom))}')
            continue

        if cmd.startswith(':macro '):
            # :macro name(params) template
            rest = cmd[len(':macro '):].strip()
            # simple parse: name(arg1,arg2) template
            try:
                # support subcommands like 'validate'
                if rest.startswith('validate'):
                    parts = rest.split(None, 1)
                    if len(parts) == 1:
                        # validate all
                        if hasattr(engine, 'validate_macro'):
                            res = engine.validate_macro()
                            if isinstance(res, dict):
                                for k, v in res.items():
                                    if v is None:
                                        print(f'Macro {k} valid')
                                    else:
                                        print(v)
                            else:
                                print(res)
                        else:
                            print('Macro validation not available')
                        continue
                    else:
                        # validate specific name
                        mname = parts[1].strip()
                        if '(' in mname:
                            # strip params if present
                            mname = mname.split('(', 1)[0]
                        if hasattr(engine, 'validate_macro'):
                            res = engine.validate_macro(mname)
                            if res is None:
                                print(f'Macro {mname} valid')
                            else:
                                print(res)
                        else:
                            print('Macro validation not available')
                        continue
                name_part, template = rest.split(None, 1)
                if '(' in name_part:
                    nm, args = name_part.split('(', 1)
                    args = args.rstrip(')')
                    params = [a.strip() for a in args.split(',')] if args.strip() else []
                else:
                    nm = name_part
                    params = []
                engine.add_macro(nm, params, template)
                print(f'Macro {nm} stored')
            except Exception as e:
                print('Macro failed:', e)
            continue

        if cmd.startswith(':profile '):
            parts = cmd.split(None, 2)
            # support ':profile summary' to show aggregated rule/unify stats
            if len(parts) >= 2 and parts[1] == 'summary':
                out = format_profile_summary(engine, interp)
                print(out)
                continue
            if len(parts) >= 2 and parts[1] == 'rule':
                if len(parts) < 3:
                    print('Usage: :profile rule <name/arity>')
                    continue
                spec = parts[2].strip()
                # spec like mortal/1
                try:
                    name, ar = spec.split('/')
                    ar = int(ar)
                except Exception:
                    print('Invalid spec')
                    continue
                stats = getattr(engine, '_rule_profile', {}).get((name, ar), None)
                if not stats:
                    print('No profiling data for', spec)
                else:
                    print(f'Profile for {spec}: avg_time={stats.get("avg_time")}, calls={stats.get("calls")}, avg_unify={stats.get("avg_unify")}')
                continue
            if len(parts) >= 2 and parts[1] == 'top':
                # :profile top N [sort_by]
                if len(parts) < 3:
                    print('Usage: :profile top <N> [avg_time|avg_unify]')
                    continue
                try:
                    n = int(parts[2])
                except Exception:
                    print('Invalid N')
                    continue
                sort_by = 'avg_time'
                if len(parts) >= 4:
                    sort_by = parts[3]
                out = format_profile_summary(engine, interp, top_n=n, sort_by=sort_by)
                print(out)
                continue
            # profile reset: clear counters and engine profile
            if len(parts) >= 2 and parts[1] == 'reset':
                try:
                    engine._rule_profile = {}
                except Exception:
                    pass
                try:
                    interp.unify_count = 0
                    interp.prove_count = 0
                except Exception:
                    pass
                print('Profile counters reset')
                continue
            # profile log: write profile JSON to given file
            if len(parts) >= 2 and parts[1] == 'log':
                if len(parts) < 3:
                    print('Usage: :profile log <file>')
                    continue
                path = parts[2]
                try:
                    import json
                    out = format_profile_summary(engine, interp)
                    with open(path, 'w', encoding='utf-8') as fh:
                        # write summary as plain text plus a JSON dump of structured data
                        fh.write(out + "\n\n")
                        json.dump({'interp': {'unify_count': getattr(interp, 'unify_count', 0), 'prove_count': getattr(interp, 'prove_count', 0)}, 'rule_profile': getattr(engine, '_rule_profile', {})}, fh, indent=2)
                    print(f'Profile logged to {path}')
                except Exception as e:
                    print('Profile log failed:', e)
                continue
                # profile export-json: write only JSON structured profile
                if len(parts) >= 2 and parts[1] == 'export-json':
                    if len(parts) < 3:
                        print('Usage: :profile export-json <file>')
                        continue
                    path = parts[2]
                    try:
                        export_profile_json(engine, interp, path)
                        print(f'Profile JSON exported to {path}')
                    except Exception as e:
                        print('Export JSON failed:', e)
                    continue
                # profile analyze: run simple perf analysis; support optional --json flag
                if cmd.startswith(':profile analyze'):
                    toks = cmd.split()
                    json_out = '--json' in toks
                    outfile = None
                    if '--outfile' in toks:
                        try:
                            idx = toks.index('--outfile')
                            outfile = toks[idx + 1]
                        except Exception:
                            print('Usage: :profile analyze [--json] [--outfile <file>]')
                            continue
                    try:
                        # always request structured analysis when writing to outfile
                        res = analyze_profile(engine, interp, top_n=3, json_out=(json_out or bool(outfile)))
                        if outfile:
                            import json as _json
                            # if res is structured (json_out True) it's the structured_analysis; otherwise it's (text, analysis)
                            structured = res if isinstance(res, dict) else res[1]
                            try:
                                with open(outfile, 'w', encoding='utf-8') as fh:
                                    _json.dump(structured, fh, indent=2)
                                print(f'Analysis written to {outfile}')
                            except Exception as e:
                                print('Failed to write analysis file:', e)
                        # also print to stdout unless only json was requested without outfile
                        if json_out and not outfile:
                            import json as _json
                            to_print = res if isinstance(res, dict) else res[1]
                            print(_json.dumps(to_print, indent=2))
                        elif not json_out:
                            text, _ = res
                            print(text)
                    except Exception as e:
                        print('Analyze failed:', e)
                    continue
        if cmd.startswith(':log perf '):
            parts = cmd.split(None, 2)
            if len(parts) < 3:
                print('Usage: :log perf <file>')
                continue
            path = parts[2]
            try:
                import json
                data = {'stats': getattr(engine, '_last_query_stats', {}), 'engine_stats': engine.stats(), 'rule_profile': getattr(engine, '_rule_profile', {})}
                with open(path, 'w', encoding='utf-8') as fh:
                    json.dump(data, fh, indent=2)
                print(f'Perf logged to {path}')
            except Exception as e:
                print('Log perf failed:', e)
            continue
        if cmd.startswith(':profile query '):
            # :profile query <query> <N>
            parts = cmd.split(None, 3)
            if len(parts) < 3:
                print('Usage: :profile query <query> <N>')
                continue
            qtext = parts[2]
            try:
                runs = int(parts[3]) if len(parts) > 3 else 10
            except Exception:
                print('Invalid N')
                continue
            import time
            t0 = time.perf_counter()
            for i in range(runs):
                run_session(qtext, engine)
            t1 = time.perf_counter()
            avg = (t1 - t0) / runs
            print(f'Ran {runs} iterations, avg time {avg:.6f}s')
            # record in engine history
            engine._profile_history = getattr(engine, '_profile_history', [])
            engine._profile_history.append({'type': 'query', 'query': qtext, 'runs': runs, 'avg_time': avg})
            continue

        if cmd.startswith(':doc '):
            parts = cmd.split(None, 2)
            if len(parts) < 2:
                print('Usage: :doc <pred/arity> | :doc save <file>')
                continue
            if parts[1] == 'save' and len(parts) == 3:
                path = parts[2]
                try:
                    with open(path, 'w', encoding='utf-8') as fh:
                        for r in engine.rules:
                            fh.write(f'{r.head} :- {", ".join(map(str, r.body))}.\n')
                    print(f'Documentation saved to {path}')
                except Exception as e:
                    print('Save failed:', e)
                continue
            spec = parts[1]
            try:
                name, ar = spec.split('/')
                ar = int(ar)
            except Exception:
                print('Invalid spec, use name/arity')
                continue
            # gather facts and rules for this predicate
            facts = [f for f in engine.facts if f.name == name and len(f.args) == ar]
            rules = [r for r in engine.rules if r.head.name == name and len(r.head.args) == ar]
            print(f'Predicate {name}/{ar}:')
            print('Facts:')
            for f in facts:
                print('  ', f)
            print('Rules:')
            for r in rules:
                print('  ', r.head, ':-', ', '.join(map(str, r.body)))
            # show constraints mentioning this predicate
            print('Constraints:')
            for a, b in getattr(engine, '_dif_constraints', []):
                if (isinstance(a, Term) and a.name == name) or (isinstance(b, Term) and b.name == name):
                    print('  dif:', a, b)
            continue
        if cmd.startswith(':profile'):
            parts = cmd.split()
            if len(parts) == 1:
                print(f'Profile mode: {profile_mode}')
                continue
            arg = parts[1]
            if arg == 'on':
                profile_mode = True
            elif arg == 'off':
                profile_mode = False
            else:
                print('Usage: :profile on|off')
                continue
            print('Profile mode:', profile_mode)
            continue
        if cmd == ':history':
            for i, c in enumerate(history[-50:], start=1):
                print(f'{i}: {c}')
            continue
        if cmd.startswith(':set '):
            parts = cmd.split()
            if len(parts) == 3 and parts[1] == 'max_depth':
                try:
                    n = int(parts[2])
                    if n <= 0:
                        raise ValueError()
                    interp.max_depth = n
                    print(f'max_depth set to {n}')
                except Exception:
                    print('Usage: :set max_depth <positive-int>')
            else:
                # support :set timeout N
                if len(parts) == 3 and parts[1] == 'timeout':
                    try:
                        n = int(parts[2])
                        if n <= 0:
                            raise ValueError()
                        interp.timeout_seconds = n
                        print(f'timeout set to {n} seconds')
                    except Exception:
                        print('Usage: :set timeout <positive-int>')
                else:
                    print('Usage: :set max_depth <positive-int> | :set timeout <seconds>')
            continue
        if cmd.startswith(':get '):
            parts = cmd.split()
            if len(parts) == 2 and parts[1] == 'max_depth':
                print(f'max_depth = {getattr(interp, "max_depth", None)}')
            else:
                print('Usage: :get max_depth')
            continue

        # accumulate lines until we see a '.' indicating end of statements
        buffer += line + '\n'
        history.append(line)
        if '.' not in line:
            continue

        try:
            if profile_mode:
                import time
                t0 = time.perf_counter()
                _, results = run_session(buffer, engine)
                t1 = time.perf_counter()
                print(f'Query time: {t1-t0:.6f}s')
                # print profiling counters from interpreter if available
                interp = Interpreter(engine)
                # apply settings so counters reflect query (quick hack: instantiate interpreter then run session) -- skip detailed counters for now
            else:
                _, results = run_session(buffer, engine)
            for res in results:
                if not res:
                    print('false.')
                else:
                    if all(len(b) == 0 for b in res):
                        print('true.')
                    else:
                        # collect variable names across all bindings for this result
                        vars_all = []
                        for b in res:
                            for k in b.keys():
                                if k not in vars_all:
                                    vars_all.append(k)
                        if len(vars_all) > 1:
                            # tabular output
                            widths = {v: max(len(v), *(len(str(b.get(v, ''))) for b in res)) for v in vars_all}
                            # header
                            hdr = ' | '.join(v.ljust(widths[v]) for v in vars_all)
                            sep = '-+-'.join('-' * widths[v] for v in vars_all)
                            print(hdr)
                            print(sep)
                            for b in res:
                                row = ' | '.join(str(b.get(v, '')).ljust(widths[v]) for v in vars_all)
                                print(row)
                        else:
                            for b in res:
                                parts = [f"{k} = {str(v)}" for k, v in b.items()]
                                print('; '.join(parts))
        except Exception as e:
            # handle max depth explicitly for user-friendly message
            if isinstance(e, MaxDepthExceededError):
                print('Recursion depth exceeded; consider increasing max_depth or optimizing rules.')
                if getattr(interp, 'trace_level', 0) >= 1:
                    logging.exception(e)
            else:
                print('Error:', e)
        buffer = ''


def load_file(engine: RulesEngine, path: str):
    """Load facts/rules from a file into the given engine.

    Returns the tuple (engine, results) from running the file contents.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    rv = run_session(text, engine)
    if isinstance(rv, tuple) and len(rv) == 3:
        return rv[0], rv[1]
    return rv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='egdol')
    parser.add_argument('--max-depth', type=int, default=None, help='Set interpreter max proof depth')
    parser.add_argument('--trace-level', type=int, default=None, help='Set trace level')
    parser.add_argument('--load', type=str, default=None, help='Load a file before starting REPL')
    parser.add_argument('--script', type=str, default=None, help='Run a script file and exit')
    parser.add_argument('--output-format', choices=['human', 'json'], default='human', help='Output format when using --script')
    parser.add_argument('--gui', action='store_true', help='Launch tkinter GUI and exit')
    parser.add_argument('--load-session', action='store_true', help='Load saved REPL session on startup')
    args = parser.parse_args()
    # apply CLI options to initial engine/interpreter
    engine = RulesEngine()
    interp = Interpreter(engine)
    if args.max_depth is not None:
        interp.max_depth = args.max_depth
    if args.trace_level is not None:
        interp.trace_level = args.trace_level
    if args.load:
        try:
            load_file(engine, args.load)
            print(f'Loaded {args.load}')
        except Exception as e:
            print('Error loading file at startup:', e)

    # if script provided, run it and exit with appropriate code/format
    if args.script:
        try:
            engine, results = load_file(engine, args.script)
            # format output
            if args.output_format == 'json':
                import json
                print(json.dumps({'results': [[{k: str(v) for k, v in b.items()} for b in res] for res in results]}))
            else:
                for res in results:
                    if not res:
                        print('false.')
                    else:
                        if all(len(b) == 0 for b in res):
                            print('true.')
                        else:
                            for b in res:
                                parts = [f"{k} = {str(v)}" for k, v in b.items()]
                                print('; '.join(parts))
            raise SystemExit(0)
        except Exception as e:
            print('Script error:', e)
            raise SystemExit(1)

    # if gui requested, launch and exit
    if args.gui:
        try:
            from .gui import launch_gui
            launch_gui(engine)
            raise SystemExit(0)
        except Exception as e:
            print('GUI failed to start:', e)
            raise SystemExit(1)

    # session load
    session_path = os.path.expanduser('~/.egdol_session.json')
    if args.load_session and os.path.exists(session_path):
        try:
            import json
            with open(session_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            # apply settings
            if 'max_depth' in data:
                interp.max_depth = data['max_depth']
            if 'trace_level' in data:
                interp.trace_level = data['trace_level']
            if 'module' in data:
                engine.current_module = data['module']
            print('Session loaded')
        except Exception:
            pass
    try:
        repl()
    finally:
        # on exit, save session
        try:
            import json
            st = {'max_depth': getattr(interp, 'max_depth', None), 'trace_level': getattr(interp, 'trace_level', None), 'module': getattr(engine, 'current_module', None), 'history': []}
            with open(session_path, 'w', encoding='utf-8') as fh:
                json.dump(st, fh)
        except Exception:
            pass

