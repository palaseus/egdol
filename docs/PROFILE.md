Profile subsystem (Phase 1)
==========================

This document describes the runtime profiling and analysis features added to edgol.
It covers REPL commands, the structured JSON schema for exports, and the analysis output.

REPL commands
-------------
- :profile summary
  - Print a human-readable summary of interpreter counters and per-rule metrics.

- :profile export-json <file>
  - Write the structured profile JSON to <file> (pretty-printed). The file contains interpreter counters and an array of per-rule metrics.

- :profile log <file>
  - Write a plain-text summary followed by a JSON dump (legacy combined format).

- :profile top N [avg_time|avg_unify]
  - Print top N rules by average rule time (default) or average unify time.

- :profile reset
  - Clear engine._rule_profile and reset interpreter counters (unify_count, prove_count).

- :profile analyze [--json] [--outfile <file>]
  - Run a simple static analysis on the collected profile data.
  - By default prints a short human-readable analysis: top slowest rules and suspect rules with unusually high per-unify cost.
  - Add --json to print structured analysis JSON instead of text.
  - Add --outfile <file> to save the analysis JSON to disk (useful for CI or tooling).

JSON schema (structured profile)
--------------------------------
The structured profile JSON (written by :profile export-json) has this shape:

{
  "interp": {
    "unify_count": <int>,
    "prove_count": <int>
  },
  "rules": [
    {
      "name": "pred_name",
      "arity": <int>,
      "calls": <int>,
      "avg_time": <float seconds>,
      "avg_unify_time": <float seconds | null>
    },
    ...
  ]
}

Notes and extension points
--------------------------
- The analyzer is intentionally lightweight: it uses simple heuristics (top-N and avg_unify comparisons to mean) to identify candidates for optimization.
- The JSON schema is extensible: future fields can include total_unify, total_unify_time, per-call histograms, or timestamps.
- The analysis is offline and read-only: it inspects collected profile data and does not alter interpreter execution.

Example usage (REPL)
--------------------
:profile export-json /tmp/profile.json
:profile analyze --json
:profile analyze --outfile /tmp/analysis.json

If you'd like a README section added to the repository README.md instead, I can merge this content there instead.