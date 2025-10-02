--------
-----
-----------
-------------
-------
-----
-------------------
----------------
---------------
---------------
---------------
----------------------
edgol — a lightweight local rule-based assistant
===============================================

Overview
--------
edgol is a compact, Prolog-style rule and fact engine implemented in pure Python
with zero external dependencies. It's intended as a fast, offline playground for
experimenting with logic programming, rapid prototyping of rule systems, and
local reasoning tasks.

Core files
----------
- `edgol/lexer.py`      — tokenization of the mini-DSL
- `edgol/parser.py`     — AST and parser for facts, rules, and queries
- `edgol/rules_engine.py` — storage, indexing, and constraint support
- `edgol/interpreter.py` — unification and depth-first proof search (inference)
- `edgol/main.py`       — REPL and session runner (including `:load`)
- `tests/`              — unit tests for lexer/parser/engine/interpreter

Quick start
-----------
Start the REPL:

```bash
python3 -m edgol.main
```

Examples (statements end with `.`):

- Add a fact:

  fact: human(socrates).

- Add a rule:

  rule: mortal(X) => human(X).

- Query:

  ? mortal(socrates).

REPL commands (selected)
------------------------
- `:load <file>` — load facts/rules from a file
- `:quit` / `:exit` — exit the REPL
- `:help` — show help
- `:facts` — list loaded facts
- `:rules` — list loaded rules
- `:save <file>` — save DB to JSON
- `:trace on|off|<level>` — tracing (0=off, 1=goals, 2=unify, 3=full)

Testing
-------
Run the unit tests:

```bash
python3 -m unittest discover -v
```

Notes
-----
edgol is intentionally small and educational. The interpreter implements a
standardizing-apart unifier and depth-first search without an occurs-check and
without many Prolog optimizations. It's a reliable base for experimentation and
local tooling.

Built-in predicates (high level)
--------------------------------
- `is/2` — arithmetic evaluation (e.g., `is(X, 2+3).` binds `X` to `5`).
- `=/2` — explicit unification (e.g., `=(X, 7)` binds `X` to `7`).
- `<, >, <=, >=` — numeric comparisons (works over integer expressions).

Control features
----------------
- Cut (`!`) — prunes choice points. Example (max/3):

  rule: max(X, Y, Z) => X > Y, !, Z = X.
  rule: max(X, Y, Z) => Z = Y.

  The cut prevents the second rule from running if the first succeeds — a
  simple if-then-else pattern.

- Negation-as-failure `not/1` — succeeds if the subgoal fails. Example:

  fact: male(john).
  fact: married(alice).
  rule: bachelor(X) => male(X), not married(X).

  Query: `? bachelor(john).` succeeds if `married(john)` is not found.

Extending edgol
---------------
Add custom builtins by subclassing `Interpreter` and registering handlers:

```py
class MyInterp(Interpreter):
    def __init__(self, engine):
        super().__init__(engine)
        self.builtins['my_pred'] = self.my_builtin

    def my_builtin(self, term, subst):
        # yield subst dictionaries on success
        if ...:
            yield subst
```

REPL trace mode
---------------
- `:trace on` or `:trace 1` — basic goal tracing
- `:trace 2` — include unification attempts
- `:trace 3` — full backtracking trace

Profiling & Analysis (built-in, production-ready)
-----------------------------------------------
edgol includes a lightweight, non-intrusive profiling subsystem built for
observability, CI integration, and local optimization. Profiling inspects
already-collected counters — it does not change interpreter semantics.

Profiling flow
--------------
A tiny visual of the profiling pipeline (how data flows from REPL to CI/dashboards):

```
REPL / run_session
    │
    ▼
  Interpreter counters (unify_count, prove_count)
    │
    ▼
  Per-rule aggregates (engine._rule_profile)
    │
    ▼
  Structured JSON export (:profile export-json) ---> analysis (analyze_profile)
    │                                                  │
    └───────────────> CI / Dashboard / Visualizer <────┘
```

Optional (Mermaid) flowchart for markdown renderers that support it:

```mermaid
flowchart LR
  A[REPL / run_session] --> B[Interpreter counters]
  B --> C[Per-rule aggregates]
  C --> D[Export JSON (:profile export-json)]
  C --> E[Analyze (:profile analyze)]
  D --> F[Dashboard / CI]
  E --> F
```

REPL profiling commands
~~~~~~~~~~~~~~~~~~~~~~~
- `:profile summary`
  - Print a human-readable summary containing interpreter counters and
    per-rule aggregated metrics.

- `:profile top N [avg_time|avg_unify]`
  - Show the top `N` rules sorted by average rule execution time (default)
    or average per-unify time.

- `:profile log <file>`
  - Write a plain-text summary followed by a JSON dump (legacy combined
    format) useful for quick inspection.

- `:profile reset`
  - Clear collected profile counters and per-rule aggregates.

- `:profile export-json <file>`
  - Export the structured profile JSON only (pretty-printed). Ideal for
    dashboards and external tooling.

- `:profile analyze [--json] [--outfile <file>]`
  - Run a static analysis over the profile data. By default prints a short
    human-readable report highlighting:
    - Top slowest rules (by average rule time)
    - Rules with unusually high per-unify cost relative to others
  - Use `--json` to print the structured analysis JSON.
  - Use `--outfile <file>` to save analysis JSON to disk (CI-friendly).

Structured profile JSON (schema)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`export-json` writes a JSON object with this shape:

```json
{
  "interp": {
    "unify_count": 12345,
    "prove_count": 678
  },
  "rules": [
    {
      "name": "pred_name",
      "arity": 2,
      "calls": 10,
      "avg_time": 0.012345,         // seconds
      "avg_unify_time": 0.000123    // seconds or null
    }
  ]
}
```

Programmatic API
~~~~~~~~~~~~~~~
Use these helpers from `edgol.main` when building tools or CI steps:

- `format_profile_summary(engine, interp, structured=True)` — returns
  `(text_summary, structured_dict)`
- `export_profile_json(engine, interp, path)` — writes structured JSON to
  `path` and returns the path
- `analyze_profile(engine, interp, top_n=3, json_out=False)` — returns
  `(text, structured_analysis)` or `structured_analysis` when `json_out=True`

Example workflow
----------------
1. Run a query or test run in the REPL (or via `run_session`) to collect
   profile data.
2. Inspect a human-readable summary:

```text
:profile summary
```

3. Find hotspots (top 5 rules):

```text
:profile top 5
```

4. Analyze and save results for CI or dashboards:

```text
:profile analyze --json --outfile /tmp/analysis.json
:profile export-json /tmp/profile.json
```

5. Feed `/tmp/profile.json` to a dashboard or post-process it in CI for
   regression checks.

Why this profiler?
-------------------
- Non-intrusive: it reads counters and aggregates already produced by the
  interpreter; it doesn't change proof behavior.
- Test-friendly: easy to export JSON and analyze in CI for performance
  regressions.
- Extensible: the structured schema is small and can be extended (schema
  versioning recommended for long-term stability).

Next steps / Extending edgol
---------------------------
If you want to expand edgol beyond the core engine, here are high-ROI
ideas:

- Add custom builtins for domain-specific reasoning (see `Interpreter` API).
- Enhance profiling exports with histograms or time-series snapshots.
- Add CI checks that fail when analysis flags new regressions.
- Build a small visualizer that reads `export-json` output and shows per-rule
  latencies and call counts.

Contributing
------------
PRs and issues welcome. Please include tests for behavior changes and keep
profiling outputs stable if you rely on them in CI.

License
-------
MIT License.

