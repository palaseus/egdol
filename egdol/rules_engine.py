"""Core logic for storing facts and rules and simple unification queries.

This module provides a minimal unifier to match query terms against stored facts.
Returns a list of bindings (dict from variable name to matched Term/Constant).
"""
from typing import List, Dict, Optional
from .parser import Term, Variable, Constant, Rule, ParseError, Fact, Query, Parser
from .lexer import Lexer


class RulesEngine:
    def __init__(self):
        self.facts: List[Term] = []
        self.rules: List[Rule] = []
        # simple index mapping predicate name to list of facts/rules
        # improved index by (name, arity) -> list
        self._fact_index = {}
        self._rule_index = {}
        self._fact_index_by_arity = {}
        self._rule_counter = 0
        # current module name (None means global)
        self.current_module = None
        # constraint store: list of inequality constraints (dif) and delayed goals (freeze)
        # dif constraints stored as tuples (TermA, TermB)
        self._dif_constraints = []
        # freeze store: list of tuples (VariableName, GoalTerm)
        self._freeze_store = []
        # dependency maps for quick lookup: varname -> list of constraint indices or goals
        self._dif_deps = {}  # varname -> set of indices into _dif_constraints
        self._freeze_deps = {}  # varname -> list of goals
        # finite-domain store: varname -> set of possible values (explicit domain)
        self._fd_domains = {}  # varname -> set of values
        # FD constraints list as tuples (varname, op, expr_term) e.g. ('X', '#=', Term('+', [Variable('Y'), Constant(1)]))
        self._fd_constraints = []
        # AC-3 queue
        self._fd_queue = []
        # indexing mode: 'hash' (default) or 'trie'
        self._index_mode = 'hash'
        # simple trie structure: root dict, children by char, values stored at terminal (name, arity) -> list
        self._trie_root = {}
        # action history for undo (list of tuples (type, payload))
        self._action_history = []
        # simple call-answer table for tabling/memoization: key -> list of answer dicts
        # key is (pred_name, arity, canonical_str); answers are list of dicts varname->ground_term
        self._table = {}

    def add_fd_domain(self, varname: str, low, high):
        # accept either an explicit iterable of values or a range low..high inclusive
        if hasattr(low, '__iter__') and not isinstance(low, (str, bytes)):
            self._fd_domains[varname] = set(low)
            return
        # numeric range
        vals = set(range(int(low), int(high) + 1))
        self._fd_domains[varname] = vals

    def get_fd_range(self, varname: str):
        """Compatibility accessor: return (min, max) tuple for var's domain.

        Returns None if var has no domain. If domain contains non-numeric
        values, returns (min, max) based on Python ordering where possible.
        """
        dom = self._fd_domains.get(varname)
        if dom is None:
            return None
        if not dom:
            return (None, None)
        try:
            nums = [int(x) for x in dom]
            return (min(nums), max(nums))
        except Exception:
            # fallback to min/max of domain elements
            try:
                mn = min(dom)
                mx = max(dom)
                return (mn, mx)
            except Exception:
                return None

    def add_fd_constraint(self, varname: str, op: str, expr_term: Term):
        idx = len(self._fd_constraints)
        self._fd_constraints.append((varname, op, expr_term))
        # enqueue for AC-3 propagation
        self._fd_queue.append(idx)
        # register deps for quick lookup
        vars_in_expr = set([varname]) | self._vars_in_term(expr_term)
        for v in vars_in_expr:
            self._dif_deps.setdefault(v, set()).add(('fd', idx))
        # record for undo
        self._action_history.append(('fd', idx))
        return idx

    def check_fd_consistency(self):
        # AC-3 style propagation over explicit domains
        # _fd_domains: var -> set(values)
        from collections import deque
        queue = deque(self._fd_queue)
        # Clear internal queue after taking snapshot
        self._fd_queue = []

        def eval_expr_values(expr):
            # compute possible values for expression given current domains
            if isinstance(expr, Constant) and isinstance(expr.value, (int, float)):
                return {int(expr.value)}
            if isinstance(expr, Variable):
                return set(self._fd_domains.get(expr.name, set()))
            if isinstance(expr, Term):
                if expr.name == '+':
                    a = eval_expr_values(expr.args[0])
                    b = eval_expr_values(expr.args[1])
                    return set(x + y for x in a for y in b)
                if expr.name == '-':
                    a = eval_expr_values(expr.args[0])
                    b = eval_expr_values(expr.args[1])
                    return set(x - y for x in a for y in b)
            return set()

        while queue:
            idx = queue.popleft()
            if idx >= len(self._fd_constraints):
                continue
            vname, op, expr = self._fd_constraints[idx]
            # domain for expression
            expr_vals = eval_expr_values(expr)
            cur = set(self._fd_domains.get(vname, set()))
            if op == '#=':
                # restrict vname to values where v == expr_vals
                new_domain = cur.intersection(expr_vals) if cur else set(expr_vals)
            elif op == '#<':
                new_domain = set(x for x in cur if any(x < y for y in expr_vals)) if cur else set()
            elif op == '#>':
                new_domain = set(x for x in cur if any(x > y for y in expr_vals)) if cur else set()
            else:
                new_domain = cur

            if not new_domain:
                # empty domain -> inconsistency
                return False
            if new_domain != cur:
                self._fd_domains[vname] = new_domain
                # enqueue other constraints that mention vname
                for dep in list(self._dif_deps.get(vname, [])):
                    # dep may be an integer index (dif) or a ('fd', idx) tuple
                    if isinstance(dep, tuple) and dep[0] == 'fd':
                        queue.append(dep[1])
        return True

    def _rebuild_dependency_maps(self):
        """Rebuild dependency maps (_dif_deps, _freeze_deps) and any indexes
        that reference list indices. Call after removals to keep indexes
        consistent.
        """
        # Rebuild dif deps and freeze deps from current constraint lists
        self._dif_deps = {}
        for idx, (A, B) in enumerate(getattr(self, '_dif_constraints', [])):
            vars_a = self._vars_in_term(A)
            vars_b = self._vars_in_term(B)
            for v in vars_a.union(vars_b):
                self._dif_deps.setdefault(v, set()).add(idx)

        self._freeze_deps = {}
        for idx, (vname, goal) in enumerate(getattr(self, '_freeze_store', [])):
            self._freeze_deps.setdefault(vname, []).append(goal)

        # Rebuild FD deps to include ('fd', idx) entries for expressions
        for idx, (vname, op, expr) in enumerate(getattr(self, '_fd_constraints', [])):
            vars_in_expr = set([vname]) | self._vars_in_term(expr)
            for v in vars_in_expr:
                self._dif_deps.setdefault(v, set()).add(('fd', idx))


    def check_constraints(self):
        """Validate overall constraint store: FD consistency and dif contradictions.
        Returns a list of issues (empty if ok).
        """
        issues = []
        # check FD
        try:
            ok = self.check_fd_consistency()
            if not ok:
                issues.append('FD constraints unsatisfiable')
        except Exception as e:
            issues.append(f'FD check error: {e}')
        # check dif constraints: if any dif between two identical constants -> issue
        for (A, B) in getattr(self, '_dif_constraints', []):
            if isinstance(A, Constant) and isinstance(B, Constant) and A.value == B.value:
                issues.append(f'dif contradiction: {A} vs {B}')
        return issues

    def add_macro(self, name: str, params: list, template: str):
        """Add a macro. Template may be a source string containing one or more statements
        or a pre-parsed list of AST nodes. Macros are stored as (params, nodes, src).
        """
        self._macros = getattr(self, '_macros', {})
        # parse template if it's a string
        nodes = None
        src = None
        if isinstance(template, str):
            src = template
            try:
                toks = Lexer(template).tokenize()
                nodes = Parser(toks).parse()
            except Exception:
                # store raw string to allow later validation
                nodes = None
        else:
            # assume already AST nodes list
            nodes = template
            src = None
        self._macros[name] = (params, nodes, src)
        # record action for undo
        self._action_history.append(('macro', name))

    def validate_macro(self, name: str = None, params: list = None, template: str = None):
        """Validate a macro template.

        If template (and params) are provided, validate that the template
        parses and that all variable names referenced in the template are
        declared in params. Returns None on success, or an error string on
        failure.

        If only name is provided (template is None), validate the stored
        macro with that name. If name is None, validate all stored macros
        and return a dict of name->error (or None)."""
        # helper to validate one macro
        def _validate_one(mname, mparams, mtemplate):
            if not isinstance(mparams, list):
                return f'Macro {mname}: params must be a list'
            # mtemplate may be AST nodes (list) or a source string
            nodes = None
            if isinstance(mtemplate, str):
                # parse
                txt = mtemplate
                if '.' not in txt.strip():
                    txt = txt.strip() + '.'
                try:
                    toks = Lexer(txt).tokenize()
                    nodes = Parser(toks).parse()
                except ParseError as pe:
                    return f'Macro {mname}: parse error: {pe}'
                except Exception as e:
                    return f'Macro {mname}: parse error: {e}'
            elif isinstance(mtemplate, list):
                nodes = mtemplate
            else:
                return f'Macro {mname}: template must be source string or AST nodes'

            # collect variable names used in parsed nodes
            used_vars = set()
            def _collect_from_term(t):
                if isinstance(t, Variable):
                    used_vars.add(t.name)
                elif isinstance(t, Term):
                    for a in t.args:
                        _collect_from_term(a)

            for n in nodes:
                if isinstance(n, Fact):
                    _collect_from_term(n.term)
                elif isinstance(n, Query):
                    _collect_from_term(n.term)
                elif isinstance(n, Rule):
                    _collect_from_term(n.head)
                    for b in n.body:
                        _collect_from_term(b)

            # determine unbound vars (used but not declared in params)
            declared = set(mparams or [])
            unbound = sorted([v for v in used_vars if v not in declared])
            if unbound:
                return f'Macro {mname}: unbound variables: {", ".join(unbound)}'
            return None

        # If explicit template provided, validate that
        if template is not None:
            return _validate_one(name or '<inline>', params or [], template)

        # If name provided, validate stored macro
        macros = getattr(self, '_macros', {})
        if name is not None:
            if name not in macros:
                return f'Unknown macro: {name}'
            mparams, mnodes, msrc = macros[name]
            # prefer stored nodes, else src
            return _validate_one(name, mparams, mnodes if mnodes is not None else msrc)

        # validate all macros
        results = {}
        for mname, (mparams, mnodes, msrc) in macros.items():
            results[mname] = _validate_one(mname, mparams, mnodes if mnodes is not None else msrc)
        return results

    def expand_macros(self, text: str) -> str:
        """AST-based macro expansion. For each macro invocation name(arg1,arg2) in
        the input text we parse the argument expressions into AST nodes and then
        clone the stored macro AST nodes, substituting parameter Variables with
        the provided AST, and hygienically renaming other Variables to avoid
        name clashes.
        """
        import re
        macros = getattr(self, '_macros', {})
        if not macros:
            return text
        out = text

        def ast_to_text(nodes):
            # convert AST nodes back to string statements
            parts = []
            for n in nodes:
                # format Fact, Rule, Query nodes in Prolog-like syntax
                if hasattr(n, 'term'):
                    parts.append(str(n.term) + '.')
                elif hasattr(n, 'head') and hasattr(n, 'body'):
                    parts.append(f"{str(n.head)} :- {', '.join(map(str, n.body))}.")
                elif hasattr(n, 'term'):
                    parts.append(str(n.term) + '.')
                else:
                    parts.append(str(n) + '.')
            return '\n'.join(parts)

        def deep_copy_and_subst(node, param_map, rename_map, fresh_prefix):
            # node is Term, Variable, or Constant
            from copy import deepcopy
            if isinstance(node, Variable):
                # parameter substitution: if parameter name matches, substitute AST
                if node.name in param_map:
                    return deepcopy(param_map[node.name])
                # otherwise hygienic rename
                if node.name in rename_map:
                    return Variable(rename_map[node.name])
                newname = f"{node.name}__m{fresh_prefix}"
                rename_map[node.name] = newname
                return Variable(newname)
            if isinstance(node, Term):
                return Term(node.name, [deep_copy_and_subst(a, param_map, rename_map, fresh_prefix) for a in node.args])
            # Constant
            return deepcopy(node)

        # find macro invocations using regex: name(args)
        for name, (params, nodes, src) in list(macros.items()):
            if nodes is None and src is None:
                # nothing to expand (invalid macro) -> skip
                continue
            pattern = re.compile(rf"\b{name}\s*\(([^)]*)\)")

            def repl(m):
                argstr = m.group(1)
                args_text = [a.strip() for a in argstr.split(',')] if argstr.strip() else []
                # parse each argument into AST (term or constant/var)
                param_asts = {}
                for i, p in enumerate(params):
                    if i < len(args_text):
                        a_txt = args_text[i]
                        try:
                            toks = Lexer(a_txt).tokenize()
                            parsed = Parser(toks).parse()
                            # parsed may be a list of nodes; take first node's term if Fact/Query
                            if parsed:
                                node = parsed[0]
                                if hasattr(node, 'term'):
                                    param_asts[p] = node.term
                                elif hasattr(node, 'head'):
                                    # rule as arg not expected; fall back to Constant text
                                    param_asts[p] = Constant(a_txt)
                                else:
                                    param_asts[p] = node
                            else:
                                param_asts[p] = Constant(a_txt)
                        except Exception:
                            param_asts[p] = Constant(a_txt)
                    else:
                        param_asts[p] = Variable(p)

                # fresh prefix for renaming
                self._macro_counter = getattr(self, '_macro_counter', 0) + 1
                fresh_prefix = self._macro_counter

                # build expanded AST nodes by cloning and substituting
                expanded_nodes = []
                source_nodes = nodes if nodes is not None else []
                for n in source_nodes:
                    rename_map = {}
                    # if node is Fact/Rule/Query, we need to transform the contained Term(s)
                    if hasattr(n, 'term'):
                        newterm = deep_copy_and_subst(n.term, param_asts, rename_map, fresh_prefix)
                        expanded_nodes.append(type(n)(newterm))
                    elif hasattr(n, 'head'):
                        newhead = deep_copy_and_subst(n.head, param_asts, rename_map, fresh_prefix)
                        newbody = [deep_copy_and_subst(b, param_asts, rename_map, fresh_prefix) for b in n.body]
                        expanded_nodes.append(type(n)(newhead, newbody))
                    else:
                        # Module or other node types: attempt to deepcopy
                        from copy import deepcopy
                        expanded_nodes.append(deepcopy(n))

                return ast_to_text(expanded_nodes)

            out = pattern.sub(repl, out)
        return out

    def list_macros(self):
        """Return a dict of macro name -> (params, src or repr of AST)."""
        res = {}
        for k, v in getattr(self, '_macros', {}).items():
            params, nodes, src = v
            if src:
                res[k] = (params, src)
            else:
                # render AST nodes
                if nodes:
                    res[k] = (params, '; '.join(str(n) for n in nodes))
                else:
                    res[k] = (params, None)
        return res

    def export_json(self, path: str):
        import json
        def serialize_term(t):
            if isinstance(t, Constant):
                return {'const': t.value}
            if isinstance(t, Term):
                return {'term': {'name': t.name, 'args': [serialize_term(a) for a in t.args]}}
            # Variable
            return {'var': t.name}

        data = {
            'facts': [serialize_term(f) for f in self.facts],
            'rules': [
                {'head': serialize_term(r.head), 'body': [serialize_term(b) for b in r.body], 'id': getattr(r, '_id', None)}
                for r in self.rules
            ],
            'dif': [[serialize_term(a), serialize_term(b)] for (a, b) in getattr(self, '_dif_constraints', [])],
            'fd': [{'var': v, 'op': op, 'expr': serialize_term(expr)} for (v, op, expr) in getattr(self, '_fd_constraints', [])],
            'macros': {k: (v[0], v[2] if v[2] is not None else ("; ".join(str(n) for n in v[1]) if v[1] else None)) for k, v in getattr(self, '_macros', {}).items()},
        }
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        return path

    def export_prolog(self, path: str):
        """Export facts, rules, dif, FD domains, and optionally macros to a Prolog-like file.

        Renders list Terms of the form .(A, .(B, [])) as [A,B]. Disjunctions represented
        as Term(';', [A,B]) are written in 'A ; B' form on the right-hand side of rules.
        """
        def term_to_prolog(t):
            # Constants
            if isinstance(t, Constant):
                return str(t.value)
            if isinstance(t, Term):
                # list detection: nested '.' terms
                if t.name == '.' and len(t.args) == 2:
                    # collect elements
                    elems = []
                    cur = t
                    tail = None
                    while isinstance(cur, Term) and cur.name == '.' and len(cur.args) == 2:
                        elems.append(term_to_prolog(cur.args[0]))
                        cur = cur.args[1]
                    tail = cur
                    if isinstance(tail, Constant) and tail.value == '[]':
                        return '[' + ', '.join(elems) + ']'
                    else:
                        return '[' + ', '.join(elems) + ' | ' + term_to_prolog(tail) + ']'
                # disjunction
                if t.name == ';' and len(t.args) == 2:
                    return f"{term_to_prolog(t.args[0])} ; {term_to_prolog(t.args[1])}"
                # inequality/dif mapping: internal 'dif' terms are printed as '\\=='
                if t.name == 'dif' and len(t.args) == 2:
                    return f"{term_to_prolog(t.args[0])} \\== {term_to_prolog(t.args[1])}"
                # general term
                return f"{t.name}({', '.join(term_to_prolog(a) for a in t.args)})" if t.args else t.name
            return str(t)

        with open(path, 'w', encoding='utf-8') as fh:
            # facts
            for f in self.facts:
                fh.write(f"{term_to_prolog(f)}.")
                fh.write("\n")
            # rules
            for r in self.rules:
                rhs = None
                # if body contains a single disjunction Term, render accordingly
                if len(r.body) == 1 and isinstance(r.body[0], Term) and r.body[0].name == ';':
                    rhs = term_to_prolog(r.body[0])
                else:
                    rhs = ', '.join(term_to_prolog(b) for b in r.body)
                fh.write(f"{term_to_prolog(r.head)} :- {rhs}.")
                fh.write("\n")
            # dif constraints
            for a, b in getattr(self, '_dif_constraints', []):
                fh.write(f"{term_to_prolog(a)} \\== {term_to_prolog(b)}.")
                fh.write("\n")
            # FD domains
            for v, dom in getattr(self, '_fd_domains', {}).items():
                try:
                    if dom:
                        mn = min(dom)
                        mx = max(dom)
                        try:
                            rng = set(range(int(mn), int(mx) + 1))
                        except Exception:
                            rng = None
                        if rng is not None and set(dom) == rng:
                            fh.write(f"{v} in {mn}..{mx}.")
                            fh.write("\n")
                        else:
                            fh.write(f"{v} in {{{', '.join(map(str, sorted(dom)))}}}.")
                            fh.write("\n")
                except Exception:
                    fh.write(f"{v} in {{{', '.join(map(str, sorted(dom)))}}}.")
                    fh.write("\n")
            # macros as comments
            for mname, (mparams, mnodes, msrc) in getattr(self, '_macros', {}).items():
                if msrc:
                    fh.write(f"% macro {mname}({', '.join(mparams)}): {msrc}\n")
                elif mnodes:
                    fh.write(f"% macro {mname}({', '.join(mparams)}): {'; '.join(str(n) for n in mnodes)}\n")
                else:
                    fh.write(f"% macro {mname}({', '.join(mparams)}): <invalid>\n")
        return path

    def import_prolog(self, path: str):
        r"""Import a very small subset of Prolog-like syntax: facts, rules, dif (\==), and in/2 domains.

        This parser uses the existing Lexer/Parser where possible but also
        accepts bare Prolog terms like 'p(a).' or 'p(X) :- q(X).'"""
        if not path:
            raise FileNotFoundError(path)
        from .lexer import Lexer
        from .parser import Parser
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
    # Preprocess: convert basic Prolog lines into egdol statements usable by Parser.
        # We'll split carefully preserving '..' ranges and '|' in lists. We support:
        #  - rules: head :- body. -> rule: head => body.
        #  - disjunctions: A ; B -> rendered in body as Term(';', [A,B]) when parsed
        #  - dif: A \== B. -> fact: dif(A, B).
        #  - lists: [a,b|c] will be preserved so Parser.builds Term('.', ...)
        import re
        out_lines = []
        # mask '..' and '|'
        placeholder_range = '__RANGE__'
        placeholder_bar = '__BAR__'
        masked = text.replace('..', placeholder_range).replace('|', placeholder_bar)
        for raw in masked.split('.'):
            s = raw.strip()
            if not s:
                continue
            s = s.replace(placeholder_range, '..').replace(placeholder_bar, '|')
            # strip Prolog comments starting with '%' (rest of line)
            s = re.sub(r'%.*$', '', s).strip()
            if not s:
                continue
            # rule
            if ':-' in s:
                head, body = s.split(':-', 1)
                # body may contain ';' disjunction; leave as-is and let Parser parse ';'
                out_lines.append(f'rule: {head.strip()} => {body.strip()}.')
                continue
            # dif
            if '\\==' in s:
                a, b = s.split('\\==', 1)
                out_lines.append(f'fact: dif({a.strip()}, {b.strip()}).')
                continue
            # domain patterns: VAR in low..high or VAR in {a,b}
            m = re.match(r"^([A-Z][a-zA-Z0-9_]*)\s+in\s+(\d+)\.\.(\d+)$", s)
            if m:
                var, lo, hi = m.group(1), m.group(2), m.group(3)
                out_lines.append(f'fact: in_range({var}, {lo}, {hi}).')
                continue
            m2 = re.match(r"^([A-Z][a-zA-Z0-9_]*)\s+in\s*\{([^}]*)\}$", s)
            if m2:
                var = m2.group(1)
                items = m2.group(2).strip()
                out_lines.append(f'fact: in_set({var}, {items}).')
                continue
            # default: treat as fact (may already look like 'fact: p(a)')
            # if the string already begins with a labeled statement, keep it
            if s.startswith('fact:') or s.startswith('rule:'):
                out_lines.append(s.strip() + '.')
            else:
                out_lines.append(f'fact: {s.strip()}.')
        new_text = '\n'.join(out_lines)
        toks = Lexer(new_text).tokenize()
        nodes = Parser(toks).parse()
        for n in nodes:
            if hasattr(n, 'term'):
                self.add_fact(n.term)
            elif hasattr(n, 'head'):
                self.add_rule(n)
        return self

    def undo_last(self):
        if not self._action_history:
            return False
        typ, payload = self._action_history.pop()
        if typ == 'dif':
            idx = payload
            # remove the constraint entry and rebuild deps
            try:
                # remove by index if still present
                if 0 <= idx < len(self._dif_constraints):
                    del self._dif_constraints[idx]
                self._rebuild_dependency_maps()
                return True
            except Exception:
                return False
        if typ == 'fd':
            idx = payload
            try:
                if 0 <= idx < len(self._fd_constraints):
                    del self._fd_constraints[idx]
                # also remove any queued items referencing this index
                self._fd_queue = [i for i in self._fd_queue if i != idx]
                self._rebuild_dependency_maps()
                return True
            except Exception:
                return False
        if typ == 'freeze':
            idx = payload
            try:
                if 0 <= idx < len(self._freeze_store):
                    del self._freeze_store[idx]
                self._rebuild_dependency_maps()
                return True
            except Exception:
                return False
        if typ == 'macro':
            name = payload
            try:
                if hasattr(self, '_macros') and name in self._macros:
                    del self._macros[name]
                return True
            except Exception:
                return False
        return False
        if not path:
            raise FileNotFoundError(path)
        from .lexer import Lexer
        from .parser import Parser
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
        # Preprocess: map '\\==' to 'dif' syntax the parser understands
        text = text.replace('\\==', 'dif')
        # Keep 'in' as used by parser's domain handling
        toks = Lexer(text).tokenize()
        nodes = Parser(toks).parse()
        for n in nodes:
            if hasattr(n, 'term'):
                self.add_fact(n.term)
            elif hasattr(n, 'head'):
                self.add_rule(n)
        return self
        if typ == 'dif':
            idx = payload
            # remove the constraint entry and rebuild deps
            try:
                # remove by index if still present
                if 0 <= idx < len(self._dif_constraints):
                    del self._dif_constraints[idx]
                self._rebuild_dependency_maps()
                return True
            except Exception:
                return False
        if typ == 'fd':
            idx = payload
            try:
                if 0 <= idx < len(self._fd_constraints):
                    del self._fd_constraints[idx]
                # also remove any queued items referencing this index
                self._fd_queue = [i for i in self._fd_queue if i != idx]
                self._rebuild_dependency_maps()
                return True
            except Exception:
                return False
        if typ == 'freeze':
            idx = payload
            try:
                if 0 <= idx < len(self._freeze_store):
                    del self._freeze_store[idx]
                self._rebuild_dependency_maps()
                return True
            except Exception:
                return False
        if typ == 'macro':
            name = payload
            try:
                if hasattr(self, '_macros') and name in self._macros:
                    del self._macros[name]
                return True
            except Exception:
                return False
        return False

    def add_fact(self, fact: Term):
        """Add a fact term to the DB."""
        # store with module qualification if a current module is set and the
        # fact is not already qualified.
        if self.current_module and ':' not in fact.name:
            fact = Term(f"{self.current_module}:{fact.name}", fact.args)
        self.facts.append(fact)
        self._fact_index.setdefault(fact.name, []).append(fact)
        key = (fact.name, len(fact.args))
        self._fact_index_by_arity.setdefault(key, []).append(fact)
        if self._index_mode == 'trie':
            # insert into trie keyed by name and arity
            node = self._trie_root
            keystr = f"{fact.name}/{len(fact.args)}"
            for ch in keystr:
                node = node.setdefault(ch, {})
            node.setdefault('_vals', []).append(fact)
        # invalidate memoization table on dynamic update
        try:
            self._table.clear()
        except Exception:
            self._table = {}

    def add_rule(self, rule: Rule):
        """Store a rule (not used for inference yet)."""
        # assign a unique id to the rule for tracing
        try:
            rule._id = self._rule_counter
        except Exception:
            pass
        self._rule_counter += 1
        # qualify rule head name with current module if appropriate
        if self.current_module and ':' not in rule.head.name:
            rule = Rule(Term(f"{self.current_module}:{rule.head.name}", rule.head.args), rule.body)
        self.rules.append(rule)
        self._rule_index.setdefault(rule.head.name, []).append(rule)
        if self._index_mode == 'trie':
            node = self._trie_root
            keystr = f"{rule.head.name}/{len(rule.head.args)}"
            for ch in keystr:
                node = node.setdefault(ch, {})
            node.setdefault('_vals', []).append(rule)
        # invalidate memoization table on dynamic update
        try:
            self._table.clear()
        except Exception:
            self._table = {}

    def query(self, q: Term) -> List[Dict[str, object]]:
        """Query stored facts for matches against term `q`.

        Returns a list of bindings (possibly empty). Each binding is a dict
        mapping variable name to the bound Term/Constant.
        """
        results: List[Dict[str, object]] = []
        # iterate only over facts with matching predicate name when possible
        # respect module qualification: if q.name contains 'mod:pred', match only those
        target_name = q.name
        if self.current_module and ':' not in target_name:
            # prefer facts in current module by prefixing names when indexing
            # but also allow global matches; build candidates accordingly
            key = (f"{self.current_module}:{target_name}", len(q.args) if isinstance(q, Term) else 0)
            fact_candidates = self._fact_index_by_arity.get(key, []) + self._fact_index_by_arity.get((target_name, len(q.args)), [])
        else:
            key = (target_name, len(q.args) if isinstance(q, Term) else 0)
            if self._index_mode == 'trie':
                # lookup in trie
                keystr = f"{target_name}/{len(q.args)}"
                node = self._trie_root
                for ch in keystr:
                    if ch not in node:
                        node = None
                        break
                    node = node[ch]
                if node is not None and '_vals' in node:
                    fact_candidates = node['_vals']
                else:
                    fact_candidates = self._fact_index_by_arity.get(key, self._fact_index.get(target_name, self.facts))
            else:
                fact_candidates = self._fact_index_by_arity.get(key, self._fact_index.get(target_name, self.facts))
        for fact in fact_candidates:
            env = {}
            unified = self._unify(q, fact, env)
            if unified is not None:
                # check dif constraints against this unified env
                if self._check_dif_constraints(unified):
                    results.append(unified)
        return results

    # constraint API
    def _vars_in_term(self, term: Term):
        # return set of variable names appearing in term
        res = set()
        def walk(x):
            from .parser import Variable as PVar
            if isinstance(x, PVar):
                res.add(x.name)
            elif isinstance(x, Term):
                for a in x.args:
                    walk(a)
        walk(term)
        return res

    def add_dif_constraint(self, A: Term, B: Term):
        idx = len(self._dif_constraints)
        self._dif_constraints.append((A, B))
        # record action for undo
        self._action_history.append(('dif', idx))
        # register deps
        vars_a = self._vars_in_term(A)
        vars_b = self._vars_in_term(B)
        for v in vars_a.union(vars_b):
            self._dif_deps.setdefault(v, set()).add(idx)
        return idx

    def add_freeze(self, varname: str, goal: Term):
        idx = len(self._freeze_store)
        self._freeze_store.append((varname, goal))
        self._freeze_deps.setdefault(varname, []).append(goal)
        self._action_history.append(('freeze', idx))
        return idx

    def on_binding(self, varname: str, subst: Dict[str, object]):
        """Called when varname is bound; returns (violated: bool, to_prove: list[Term]).
        violated=True indicates a dif constraint violation under subst; to_prove is list of freeze goals to attempt.
        """
        # check dif constraints dependent on varname
        violated = False
        for idx in list(self._dif_deps.get(varname, [])):
            A, B = self._dif_constraints[idx]
            # attempt to check if A and B are concretely equal under subst
            def concrete_equal(a, b):
                a2 = a
                b2 = b
                # apply simple substitution using names in subst where available
                from .parser import Variable as PVar, Constant as PConst, Term as PTerm
                def apply_simple(x):
                    if isinstance(x, PVar) and x.name in subst:
                        return subst[x.name]
                    if isinstance(x, PTerm):
                        return PTerm(x.name, [apply_simple(z) for z in x.args])
                    return x
                try:
                    a2 = apply_simple(a)
                    b2 = apply_simple(b)
                except Exception:
                    return False
                if isinstance(a2, PConst) and isinstance(b2, PConst):
                    return a2.value == b2.value
                if isinstance(a2, PTerm) and isinstance(b2, PTerm):
                    if a2.name != b2.name or len(a2.args) != len(b2.args):
                        return False
                    for xa, xb in zip(a2.args, b2.args):
                        if not concrete_equal(xa, xb):
                            return False
                    return True
                return False
            try:
                if concrete_equal(A, B):
                    violated = True
                    break
            except Exception:
                continue

        # collect freeze goals to attempt
        to_prove = list(self._freeze_deps.get(varname, []))
        # remove one-shot freeze deps (will be retried only once when var binds)
        if varname in self._freeze_deps:
            try:
                del self._freeze_deps[varname]
            except Exception:
                pass
        return violated, to_prove

    def _check_dif_constraints(self, env: Dict[str, object]) -> bool:
        # Ensure no dif constraint is violated under env
        for a, b, c_env in getattr(self, '_dif_constraints', []):
            # naive check: if both sides bound in env, ensure they don't unify equal
            av = env.get(a, a)
            bv = env.get(b, b)
            if isinstance(av, Constant) and isinstance(bv, Constant) and av.value == bv.value:
                return False
        return True

    def stats(self):
        return {
            'num_facts': len(self.facts),
            'num_rules': len(self.rules),
            'fact_index_size': len(self._fact_index),
            'rule_index_size': len(self._rule_index),
            'index_mode': self._index_mode,
        }

    def set_index_mode(self, mode: str):
        if mode not in ('hash', 'trie'):
            raise ValueError('Unknown index mode')
        self._index_mode = mode
        # rebuild trie if switching to trie
        if mode == 'trie':
            self._trie_root = {}
            for f in self.facts:
                node = self._trie_root
                keystr = f"{f.name}/{len(f.args)}"
                for ch in keystr:
                    node = node.setdefault(ch, {})
                node.setdefault('_vals', []).append(f)
            for r in self.rules:
                node = self._trie_root
                keystr = f"{r.head.name}/{len(r.head.args)}"
                for ch in keystr:
                    node = node.setdefault(ch, {})
                node.setdefault('_vals', []).append(r)

    def find_matches(self, term: Term) -> List[Term]:
        """Return list of facts/rules that match term string-wise for previewing retracts."""
        matches = []
        for f in self.facts:
            if str(f) == str(term) or (':' in str(f) and str(f).endswith(str(term))):
                matches.append(f)
        for r in self.rules:
            if str(r.head) == str(term) or (':' in str(r.head) and str(r.head).endswith(str(term))):
                matches.append(r)
        return matches

    def _unify(self, a, b, env: Dict[str, object]) -> Optional[Dict[str, object]]:
        """Attempt to unify a and b with initial environment env.

        Returns a new environment dict on success, or None on failure.
        """
        # Work with a copy to avoid mutating caller's dict
        env = dict(env)

        def unify(x, y, e: Dict[str, object]) -> Optional[Dict[str, object]]:
            # Variable cases
            if isinstance(x, Variable):
                return unify_var(x, y, e)
            if isinstance(y, Variable):
                return unify_var(y, x, e)

            # Constant vs Constant
            if isinstance(x, Constant) and isinstance(y, Constant):
                return e if x.value == y.value else None

            # Term vs Term
            if isinstance(x, Term) and isinstance(y, Term):
                if x.name != y.name or len(x.args) != len(y.args):
                    return None
                for xa, ya in zip(x.args, y.args):
                    e = unify(xa, ya, e)
                    if e is None:
                        return None
                return e

            # Constant vs Term (allow matching constant to zero-arg term)
            if isinstance(x, Constant) and isinstance(y, Term):
                if not y.args and x.value == y.name:
                    return e
                return None
            if isinstance(y, Constant) and isinstance(x, Term):
                if not x.args and y.value == x.name:
                    return e
                return None

            return None

        def occurs_check(varname: str, term, e: Dict[str, object]) -> bool:
            # return True if varname occurs in term (considering env bindings)
            def walk(x):
                x = e.get(x.name, x) if isinstance(x, Variable) and x.name in e else x
                if isinstance(x, Variable):
                    return x.name == varname
                if isinstance(x, Term):
                    for a in x.args:
                        if walk(a):
                            return True
                return False
            return walk(term)

        def unify_var(var: Variable, val, e: Dict[str, object]) -> Optional[Dict[str, object]]:
            # If var already bound, unify bound value with val
            if var.name in e:
                return unify(e[var.name], val, e)
            # If val is a variable already bound, unify var with that binding
            if isinstance(val, Variable) and val.name in e:
                return unify(var, e[val.name], e)
            # Occurs check: do not bind var to a term that contains var
            if isinstance(val, Term) and occurs_check(var.name, val, e):
                return None
            e2 = dict(e)
            e2[var.name] = val
            return e2

        return unify(a, b, env)

    # Persistence helpers
    def save(self, path: str):
        import json

        def serialize_term(t):
            if isinstance(t, Constant):
                return {'const': t.value}
            if isinstance(t, Term):
                return {'term': {'name': t.name, 'args': [serialize_term(a) for a in t.args]}}
            # Variable in facts/rules will be serialized as var
            return {'var': t.name}

        data = {
            'version': 1,
            'facts': [serialize_term(f) for f in self.facts],
            'rules': [
                {'head': serialize_term(r.head), 'body': [serialize_term(b) for b in r.body], 'id': getattr(r, '_id', None)}
                for r in self.rules
            ],
            'settings': getattr(self, '_settings', {})
        }
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path: str) -> 'RulesEngine':
        import json

        def deserialize(obj):
            if 'const' in obj:
                return Constant(obj['const'])
            if 'var' in obj:
                from .parser import Variable as PVar
                return PVar(obj['var'])
            if 'term' in obj:
                t = obj['term']
                return Term(t['name'], [deserialize(a) for a in t['args']])
            raise ValueError('Unknown term format')

        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        # migration: accept legacy saves without 'version' as v0
        version = data.get('version', 0)
        engine = cls()
        for f in data.get('facts', []):
            engine.add_fact(deserialize(f))
        for r in data.get('rules', []):
            head = deserialize(r['head'])
            body = [deserialize(b) for b in r['body']]
            rule = Rule(head, body)
            # restore id when present
            if isinstance(r, dict) and 'id' in r and r['id'] is not None:
                try:
                    rule._id = r['id']
                except Exception:
                    pass
            engine.add_rule(rule)
        # load settings if present
        if 'settings' in data:
            engine._settings = data['settings']
            engine.settings = engine._settings
        # if this is legacy (version 0) migrate by rewriting save as v1 on next save
        engine._loaded_version = version
        return engine

    def dump(self, path: str):
        # human-readable dump in Prolog-like syntax
        with open(path, 'w', encoding='utf-8') as fh:
            for f in self.facts:
                fh.write(f'fact: {f}.\n')
            for r in self.rules:
                fh.write(f'rule: {r.head} => {", ".join(map(str, r.body))}.\n')
        # ensure settings are present for compatibility
        if not hasattr(self, '_settings'):
            self._settings = {}
        self.settings = self._settings
        return self

    def retract(self, term: Term) -> List[Term]:
        """Remove facts/rules matching `term`. Returns removed items."""
        removed = []
        # remove facts equal by string representation
        remaining = []
        for f in self.facts:
            if str(f) == str(term):
                removed.append(f)
            else:
                remaining.append(f)
        self.facts = remaining
        # remove rules whose head matches
        r_remaining = []
        for r in self.rules:
            if str(r.head) == str(term):
                removed.append(r)
            else:
                r_remaining.append(r)
        self.rules = r_remaining
        # rebuild indexes simply
        self._fact_index = {}
        self._fact_index_by_arity = {}
        for f in self.facts:
            self._fact_index.setdefault(f.name, []).append(f)
            self._fact_index_by_arity.setdefault((f.name, len(f.args)), []).append(f)
        self._rule_index = {}
        for r in self.rules:
            self._rule_index.setdefault(r.head.name, []).append(r)
        # invalidate memoization table on dynamic update
        try:
            self._table.clear()
        except Exception:
            self._table = {}
        return removed

    def export_to_dot(self, path: str):
        """Export a Graphviz DOT of rule dependencies: body predicates -> head predicate."""
        nodes = set()
        edges = []
        for r in self.rules:
            head = r.head.name
            nodes.add(head)
            for b in r.body:
                if isinstance(b, Term):
                    nodes.add(b.name)
                    edges.append((b.name, head))
        # include dif constraints as dashed edges between involved predicate nodes (if any)
        for a, b in getattr(self, '_dif_constraints', []):
            # only include predicate-level representation if terms
            if isinstance(a, Term) and isinstance(b, Term):
                nodes.add(a.name)
                nodes.add(b.name)
                edges.append((a.name, b.name, 'dif'))
        # include FD constraints as node annotations
        fd_nodes = []
        for v, op, expr in getattr(self, '_fd_constraints', []):
            fd_nodes.append((v, op, str(expr)))
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('digraph egdol {\n')
            for n in nodes:
                fh.write(f'  "{n}";\n')
            for e in edges:
                if len(e) == 2:
                    a, b = e
                    fh.write(f'  "{a}" -> "{b}";\n')
                else:
                    a, b, et = e
                    if et == 'dif':
                        fh.write(f'  "{a}" -> "{b}" [style=dashed, color=red, label="dif"];\n')
            fh.write('}\n')
        # append fd annotations as comments
        if fd_nodes:
            with open(path, 'a', encoding='utf-8') as fh:
                fh.write('\n// FD constraints\n')
                for v, op, ex in fd_nodes:
                    fh.write(f'// {v} {op} {ex}\n')
        return path

    def list_constraints(self) -> str:
        """Return a formatted string listing active constraints: dif, freeze, and FD.

        Format is a simple table with columns: Type | Expression | Details
        """
        lines = []
        lines.append(f"{'Type':<8} | {'Expression':<40} | Details")
        lines.append('-' * 80)
        # dif constraints
        for a, b in getattr(self, '_dif_constraints', []):
            try:
                expr = f"dif({a}, {b})"
            except Exception:
                expr = f"dif({a}, {b})"
            lines.append(f"{'dif':<8} | {expr:<40} | -")
        # freeze store
        for vname, goal in getattr(self, '_freeze_store', []):
            expr = f"freeze({vname}, {goal})"
            lines.append(f"{'freeze':<8} | {expr:<40} | -")
        # FD domains
        for v, dom in getattr(self, '_fd_domains', {}).items():
            lines.append(f"{'fd-dom':<8} | {v + ' in ' + str(sorted(dom)):<40} | size={len(dom)}")
        # FD constraints
        for v, op, expr in getattr(self, '_fd_constraints', []):
            try:
                lines.append(f"{'fd-c':<8} | {v + ' ' + op + ' ' + str(expr):<40} | -")
            except Exception:
                lines.append(f"{'fd-c':<8} | {v + ' ' + op:<40} | {expr}")
        if len(lines) == 2:
            return 'No constraints.'
        return '\n'.join(lines)

