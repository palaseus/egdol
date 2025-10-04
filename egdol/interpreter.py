"""Interpreter for egdol DSL with basic inference (unification + backtracking).

Uses a RulesEngine instance (which stores facts and rules). The interpreter
performs depth-first proof search to satisfy queries using facts and rules.
"""
import logging
from typing import Dict, Generator, List

from .parser import Constant, Rule, Term, Variable


class MaxDepthExceededError(Exception):
    """Raised when proof search exceeds the configured maximum depth."""
    pass


class CutException(Exception):
    """Internal signal for cut — not propagated beyond the goal where caught."""
    pass


class UnificationError(Exception):
    """Raised when unification fails due to a structural/occurs-check issue."""
    pass


class Interpreter:
    def __init__(self, engine):
        self.engine = engine
        # expose this interpreter on the engine so callers that construct an
        # Interpreter can have run_session pick up their settings
        try:
            engine._last_interp = self
        except Exception:
            pass
        self._rule_counter = 0
        # builtins registry: name -> handler(func(term, subst) -> generator of subst)
        self.builtins = {}
        self._register_default_builtins()
        # tracing level: 0=off,1=goals,2=unify,3=full
        self.trace_level = 0
        # recursion depth protection
        self.max_depth = 100
        # optional timeout in seconds for queries (None = no timeout)
        self.timeout_seconds = None
        # interrupt flag set by timer when timeout expires
        self._interrupt_flag = False
        # profiling counters
        self.unify_count = 0
        self.prove_count = 0
        # whether to use subprocess fallback for hard timeouts
        self.use_subprocess_timeout = False
        # choices counter for structured trace
        self._choice_counter = 0
        # optional type checking for builtin args (disabled by default)
        self.type_checking = False
        # enable occurs-check by default to prevent creating cyclic bindings
        self.occurs_check = True
        # whether to raise UnificationError on occurs-check failures (False -> unify fails silently)
        self.raise_on_occurs = False
        # tabling (memoization) disabled by default; can be toggled on Interpreter
        self.tabling = False
        # small local cache used while tabling; maps table keys to list of subst dicts
        # persistent storage of answers lives on engine._table
        self._local_table = {}

    def _register_default_builtins(self):
        self.builtins['is'] = self._builtin_is
        self.builtins['='] = self._builtin_eq
        self.builtins['<'] = self._builtin_lt
        self.builtins['>'] = self._builtin_gt
        self.builtins['<='] = self._builtin_le
        self.builtins['>='] = self._builtin_ge
        # list and string builtins
        self.builtins['member'] = self._builtin_member
        self.builtins['append'] = self._builtin_append
        self.builtins['atom_concat'] = self._builtin_atom_concat
        self.builtins['write'] = self._builtin_write
        self.builtins['nl'] = self._builtin_nl
        self.builtins['atom_length'] = self._builtin_atom_length
        self.builtins['number_atom'] = self._builtin_number_atom
        self.builtins['fail'] = self._builtin_fail
        # higher-level list builtins
        self.builtins['reverse'] = self._builtin_reverse
        self.builtins['length'] = self._builtin_length
        self.builtins['consult'] = self._builtin_consult
        # constraint builtins
        self.builtins['dif'] = self._builtin_dif
        self.builtins['freeze'] = self._builtin_freeze
        # FD builtins
        self.builtins['#='] = self._builtin_fd_eq
        self.builtins['#<'] = self._builtin_fd_lt
        self.builtins['#>'] = self._builtin_fd_gt
        # dynamic goal and collection builtins
        self.builtins['call'] = self._builtin_call
        self.builtins['bagof'] = self._builtin_bagof
        # optional type hints per builtin (list of 'atom'|'number'|'list'|'var')
        self._builtin_types = {
            'atom_length': ['atom', 'var'],
            'number_atom': ['number|atom', 'number|atom'],
            'append': ['list', 'list', 'list'],
            'member': ['var', 'list'],
            'reverse': ['list', 'list'],
            'length': ['list', 'var'],
            'consult': ['atom'],
            'dif': ['var', 'var'],
            'freeze': ['var', 'var'],
            'call': ['var'],
            'bagof': ['var', 'var', 'list'],
            '#=': ['var', 'var'],
            '#<': ['var', 'var'],
            '#>': ['var', 'var'],
        }

    def register_builtin(self, name: str, handler):
        """Register a custom builtin handler(term, subst) -> generator(subst).

        Example handler signature: def myhandler(term, subst): yield subst
        """
        self.builtins[name] = handler

    def unregister_builtin(self, name: str):
        if name in self.builtins:
            del self.builtins[name]

    def _fresh_var_name(self, base: str) -> str:
        self._rule_counter += 1
        return f"{base}__r{self._rule_counter}"

    def _standardize_apart_term(self, term: Term, var_map: Dict[str, Variable]) -> Term:
        # recursively clone term, renaming variables using var_map
        new_args = []
        for a in term.args:
            if isinstance(a, Variable):
                if a.name not in var_map:
                    var_map[a.name] = Variable(self._fresh_var_name(a.name))
                new_args.append(var_map[a.name])
            elif isinstance(a, Term):
                new_args.append(self._standardize_apart_term(a, var_map))
            else:
                # Constant
                new_args.append(a)
        return Term(term.name, new_args)

    def _standardize_apart_rule(self, rule: Rule) -> Rule:
        var_map: Dict[str, Variable] = {}
        head = self._standardize_apart_term(rule.head, var_map)
        body = [self._standardize_apart_term(t, var_map) for t in rule.body]
        return Rule(head, body)

    def _apply_subst(self, x, subst: Dict[str, object]):
        # recursively apply substitution to Variables/Terms
        if isinstance(x, Variable):
            if x.name in subst:
                return self._apply_subst(subst[x.name], subst)
            return x
        if isinstance(x, Term):
            return Term(x.name, [self._apply_subst(a, subst) for a in x.args])
        return x  # Constant

    def _is_cut(self, term: Term) -> bool:
        return isinstance(term, Term) and term.name == '!' and not term.args

    def _is_not(self, term: Term) -> bool:
        return isinstance(term, Term) and term.name == 'not' and len(term.args) == 1

    def _unify(self, x, y, subst: Dict[str, object]) -> Dict[str, object] or None:
        # Wrapper: delegate actual unify work to _unify_impl so we can
        # time and count unification calls and attribute them to the
        # currently-proving rule (if any) for profiling.
        import time
        start_t = time.perf_counter()
        res = self._unify_impl(x, y, subst)
        elapsed = time.perf_counter() - start_t
        # global unify counter
        try:
            self.unify_count += 1
        except Exception:
            pass
        # if currently proving a rule, attribute this unify's time/count
        cur = getattr(self, '_current_rule_key', None)
        if cur is not None:
            # maintain per-rule temporary accumulators on the interpreter
            self._current_rule_unify_calls = getattr(self, '_current_rule_unify_calls', 0) + 1
            self._current_rule_unify_time = getattr(self, '_current_rule_unify_time', 0.0) + elapsed
        return res

    def _unify_impl(self, x, y, subst: Dict[str, object]) -> Dict[str, object] or None:
        # actual unify implementation (moved from _unify) -- applies current substitution
        x = self._apply_subst(x, subst)
        y = self._apply_subst(y, subst)
        if getattr(self, 'trace_level', 0) >= 2:
            if getattr(self, 'trace_level', 0) >= 3:
                # structured detail for trace level 3
                logging.info(('  ' * getattr(self, '_trace_indent', 0)) + f'[UNIFY] {x} with {y} under {subst}')
            else:
                logging.info(f'Unify attempt: {x}  with  {y}  under {subst}')

        # Variable cases
        if isinstance(x, Variable):
            return self._extend_subst(subst, x.name, y)
        if isinstance(y, Variable):
            return self._extend_subst(subst, y.name, x)

        # Constant vs Constant
        if isinstance(x, Constant) and isinstance(y, Constant):
            return subst if x.value == y.value else None

        # Term vs Term
        if isinstance(x, Term) and isinstance(y, Term):
            if x.name != y.name or len(x.args) != len(y.args):
                if getattr(self, 'trace_level', 0) >= 2:
                    logging.info(f'Unify fail (mismatch): {x} vs {y}')
                return None
            s = dict(subst)
            for xa, ya in zip(x.args, y.args):
                s = self._unify(xa, ya, s)
                if s is None:
                    if getattr(self, 'trace_level', 0) >= 2:
                        logging.info(f'Unify fail in args: {xa} vs {ya}')
                    return None
            return s

        # Constant vs Term (allow matching zero-arg term)
        if isinstance(x, Constant) and isinstance(y, Term):
            if not y.args and x.value == y.name:
                return subst
            return None
        if isinstance(y, Constant) and isinstance(x, Term):
            if not x.args and y.value == x.name:
                return subst
            return None

        return None

    # Builtin implementations
    def _eval_arith(self, term, subst):
        # evaluate arithmetic Term recursively; returns integer or None
        t = self._apply_subst(term, subst)
        if isinstance(t, Constant):
            if isinstance(t.value, (int, float)):
                return t.value
            try:
                return int(t.value)
            except Exception:
                try:
                    return float(t.value)
                except Exception:
                    return None
        if isinstance(t, Term):
            if t.name == '+':
                a = self._eval_arith(t.args[0], subst)
                b = self._eval_arith(t.args[1], subst)
                return None if a is None or b is None else a + b
            if t.name == '-':
                a = self._eval_arith(t.args[0], subst)
                b = self._eval_arith(t.args[1], subst)
                return None if a is None or b is None else a - b
            if t.name == '*':
                a = self._eval_arith(t.args[0], subst)
                b = self._eval_arith(t.args[1], subst)
                return None if a is None or b is None else a * b
            if t.name == '/':
                a = self._eval_arith(t.args[0], subst)
                b = self._eval_arith(t.args[1], subst)
                if a is None or b is None or b == 0:
                    return None
                return a // b
        return None

    def _builtin_is(self, term: Term, subst: Dict[str, object]):
        # is(X, Expr)
        left, right = term.args
        val = self._eval_arith(right, subst)
        if val is None:
            return
        # unify left with Constant(val) (native numeric if available)
        s = self._unify(left, Constant(val), dict(subst))
        if s is not None:
            yield s

    def _builtin_eq(self, term: Term, subst: Dict[str, object]):
        left, right = term.args
        s = self._unify(left, right, dict(subst))
        if s is not None:
            yield s

    def _builtin_cmp(self, term: Term, subst: Dict[str, object], op):
        a = self._eval_arith(term.args[0], subst)
        b = self._eval_arith(term.args[1], subst)
        if a is None or b is None:
            return
        ok = False
        if op == '<':
            ok = a < b
        elif op == '>':
            ok = a > b
        elif op == '<=':
            ok = a <= b
        elif op == '>=':
            ok = a >= b
        if ok:
            yield dict(subst)

    def _builtin_lt(self, term, subst):
        yield from self._builtin_cmp(term, subst, '<')

    def _builtin_gt(self, term, subst):
        yield from self._builtin_cmp(term, subst, '>')

    def _builtin_le(self, term, subst):
        yield from self._builtin_cmp(term, subst, '<=')

    def _builtin_ge(self, term, subst):
        yield from self._builtin_cmp(term, subst, '>=')

    # List helpers: lists represented as nested Term('.', [Head, Tail]) and [] as Constant('[]')
    def _is_empty_list(self, x):
        return isinstance(x, Constant) and x.value == '[]'

    def _is_list_term(self, x):
        return isinstance(x, Term) and x.name == '.' and len(x.args) == 2

    def _builtin_member(self, term: Term, subst: Dict[str, object]):
        # member(X, List)
        X, L = term.args
        # traverse list L
        lval = self._apply_subst(L, subst)
        # iterate over list structure
        while True:
            if self._is_empty_list(lval):
                return
            if self._is_list_term(lval):
                head, tail = lval.args
                s = self._unify(X, head, dict(subst))
                if s is not None:
                    yield s
                lval = tail
                continue
            # not a proper list
            return

    def _builtin_append(self, term: Term, subst: Dict[str, object]):
        # Full nondeterministic append/3 implementation using recursion with backtracking.
        A, B, C = term.args

        def to_py_list(l):
            res = []
            cur = l
            while self._is_list_term(cur):
                res.append(cur.args[0])
                cur = cur.args[1]
            if self._is_empty_list(cur):
                return res
            return None

        def from_py_list(py):
            node = Constant('[]')
            for itm in reversed(py):
                node = Term('.', [itm, node])
            return node

        # attempt to get concrete C list if possible
        c_val = self._apply_subst(C, subst)

        # If C is concrete, generate all splits
        if not isinstance(c_val, Variable):
            py = to_py_list(c_val)
            if py is None:
                return
            for i in range(len(py) + 1):
                a_term = from_py_list(py[:i])
                b_term = from_py_list(py[i:])
                s1 = self._unify(A, a_term, dict(subst))
                if s1 is None:
                    continue
                s2 = self._unify(B, b_term, s1)
                if s2 is not None:
                    yield s2
            return

        # If A is concrete, build remaining C by appending B unknown
        a_val = self._apply_subst(A, subst)
        if not isinstance(a_val, Variable):
            a_py = to_py_list(a_val)
            if a_py is None:
                return
            # B can be any list; produce variable tail generator by splitting
            # We generate lists up to a reasonable length limited by max_depth
            max_len = getattr(self, 'max_depth', 100)

            def gen_b_and_c(prefix):
                # prefix is current prefix for B
                b_term = from_py_list(prefix)
                c_term = from_py_list(a_py + prefix)
                yield b_term, c_term
                if len(prefix) >= max_len:
                    return
                # try to extend prefix with a fresh variable element (nondet)
                # we avoid creating infinite search space by limiting depth
                # This is a heuristic; full generation requires search on domain of elements
                return

            for b_term, c_term in gen_b_and_c([]):
                s1 = self._unify(B, b_term, dict(subst))
                if s1 is None:
                    continue
                s2 = self._unify(C, c_term, s1)
                if s2 is not None:
                    yield s2
            return

        # If B is concrete, similar approach
        b_val = self._apply_subst(B, subst)
        if not isinstance(b_val, Variable):
            b_py = to_py_list(b_val)
            if b_py is None:
                return
            max_len = getattr(self, 'max_depth', 100)
            def gen_a_and_c(prefix):
                a_term = from_py_list(prefix)
                c_term = from_py_list(prefix + b_py)
                yield a_term, c_term
                return
            for a_term, c_term in gen_a_and_c([]):
                s1 = self._unify(A, a_term, dict(subst))
                if s1 is None:
                    continue
                s2 = self._unify(C, c_term, s1)
                if s2 is not None:
                    yield s2
            return

        # Fully non-ground case: generate short possible lists for C up to max_depth
        max_len = min(5, getattr(self, 'max_depth', 100))
        # domain elements for generation: cannot invent concrete elements, so we use
        # variables — which will remain non-ground. We will generate lists of vars.
        def var_list(n, base_idx=0):
            if n == 0:
                return Constant('[]')
            node = Constant('[]')
            for i in range(n):
                v = Variable(f'_A{i+base_idx}')
                node = Term('.', [v, node])
            return node

        for n in range(0, max_len + 1):
            c_candidate = var_list(n)
            s_c = self._unify(C, c_candidate, dict(subst))
            if s_c is None:
                continue
            # split into A (k elements) and B (n-k elements)
            for k in range(0, n + 1):
                a_candidate = var_list(k)
                b_candidate = var_list(n - k, base_idx=k)
                s1 = self._unify(A, a_candidate, dict(s_c))
                if s1 is None:
                    continue
                s2 = self._unify(B, b_candidate, s1)
                if s2 is not None:
                    yield s2
        return

    def _builtin_atom_concat(self, term: Term, subst: Dict[str, object]):
        A, B, C = term.args
        a = self._apply_subst(A, subst)
        b = self._apply_subst(B, subst)
        c = self._apply_subst(C, subst)
        # handle common case: A and B are strings or constants
        if isinstance(a, Constant) and isinstance(b, Constant):
            res = a.value + b.value
            s = self._unify(C, Constant(res), dict(subst))
            if s is not None:
                yield s
            return
        # if C is known and A or B variable, try splitting (simple case: A empty)
        if isinstance(c, Constant):
            sval = c.value
            # try A = '', B = sval
            s1 = self._unify(A, Constant(''), dict(subst))
            if s1 is not None:
                s2 = self._unify(B, Constant(sval), s1)
                if s2 is not None:
                    yield s2
            return

    def _builtin_atom_length(self, term: Term, subst: Dict[str, object]):
        A, N = term.args
        a = self._apply_subst(A, subst)
        if isinstance(a, Constant):
            try:
                n = len(a.value)
                s = self._unify(N, Constant(str(n)), dict(subst))
                if s is not None:
                    yield s
            except Exception:
                return
        else:
            # if N is known and A var, could generate string of spaces - not implemented
            return

    def _builtin_number_atom(self, term: Term, subst: Dict[str, object]):
        A, B = term.args
        a = self._apply_subst(A, subst)
        b = self._apply_subst(B, subst)
        # A numeric (Constant int string) to atom B
        if isinstance(a, Constant) and isinstance(b, Constant):
            # both atoms; check equivalence via str int conversion
            try:
                if str(int(a.value)) == b.value:
                    yield dict(subst)
            except Exception:
                return
        if isinstance(a, Constant) and not isinstance(b, Constant):
            # try bind B
            try:
                s = self._unify(B, Constant(str(int(a.value))), dict(subst))
                if s is not None:
                    yield s
            except Exception:
                return
        if isinstance(b, Constant) and not isinstance(a, Constant):
            try:
                s = self._unify(A, Constant(str(int(b.value))), dict(subst))
                if s is not None:
                    yield s
            except Exception:
                return
        return

    def _builtin_fail(self, term: Term, subst: Dict[str, object]):
        # always fails (yields nothing)
        return

    def _builtin_reverse(self, term: Term, subst: Dict[str, object]):
        A, B = term.args
        a = self._apply_subst(A, subst)
        b = self._apply_subst(B, subst)
        def to_py_list(l):
            res = []
            cur = l
            while self._is_list_term(cur):
                res.append(cur.args[0])
                cur = cur.args[1]
            if self._is_empty_list(cur):
                return res
            return None

        def from_py_list(py):
            node = Constant('[]')
            for itm in reversed(py):
                node = Term('.', [itm, node])
            return node

        if not isinstance(a, Variable) and self._is_list_term(a) or self._is_empty_list(a):
            py = to_py_list(a)
            if py is None:
                return
            node = from_py_list(list(reversed(py)))
            s = self._unify(B, node, dict(subst))
            if s is not None:
                yield s
            return
        # if B known, generate A
        if not isinstance(b, Variable) and (self._is_list_term(b) or self._is_empty_list(b)):
            py = to_py_list(b)
            if py is None:
                return
            node = from_py_list(list(reversed(py)))
            s = self._unify(A, node, dict(subst))
            if s is not None:
                yield s
            return
        return

    def _builtin_length(self, term: Term, subst: Dict[str, object]):
        L, N = term.args
        lval = self._apply_subst(L, subst)
        def count(l):
            if self._is_empty_list(l):
                return 0
            if self._is_list_term(l):
                return 1 + count(l.args[1])
            return None
        n = count(lval)
        if n is None:
            return
        s = self._unify(N, Constant(str(n)), dict(subst))
        if s is not None:
            yield s

    def _builtin_consult(self, term: Term, subst: Dict[str, object]):
        # consult(File) - loads and runs a script file; side-effecting
        from .main import load_file
        fterm = self._apply_subst(term.args[0], subst)
        if not isinstance(fterm, Constant):
            return
        try:
            load_file(self.engine, fterm.value)
            yield dict(subst)
        except Exception:
            return

    def run_script(self, path: str):
    # load and run a .egdol script file (facts/rules/queries)
        from .main import load_file
        load_file(self.engine, path)
        return

    def _builtin_dif(self, term: Term, subst: Dict[str, object]):
        # dif(A,B) succeeds if A and B cannot be unified now (and records constraint)
        A, B = term.args
        a = self._apply_subst(A, subst)
        b = self._apply_subst(B, subst)
        # if both constants/terms and equal -> fail
        if isinstance(a, Constant) and isinstance(b, Constant) and a.value == b.value:
            return
        # otherwise record constraint in engine for rechecking; store the original terms
        try:
            self.engine.add_dif_constraint(A, B)
        except Exception:
            pass
        yield dict(subst)

    def _builtin_fd_eq(self, term: Term, subst: Dict[str, object]):
        # X #= Expr
        X, Expr = term.args
        # if Expr is constant or arithmetic with constants, try to bind
        val = self._eval_arith(Expr, subst)
        if isinstance(X, Variable) and val is not None:
            s = self._unify(X, Constant(val), dict(subst))
            if s is not None:
                yield s
                return
        # otherwise register FD constraint in engine
        try:
            if isinstance(X, Variable):
                self.engine.add_fd_constraint(X.name, '#=', Expr)
        except Exception:
            pass
        yield dict(subst)

    def _builtin_fd_lt(self, term: Term, subst: Dict[str, object]):
        X, Expr = term.args
        val = self._eval_arith(Expr, subst)
        if isinstance(X, Variable) and val is not None:
            # X < val -> bound upper
            try:
                self.engine.add_fd_domain(X.name, float('-inf'), val - 1)
            except Exception:
                pass
            yield dict(subst)
            return
        try:
            if isinstance(X, Variable):
                self.engine.add_fd_constraint(X.name, '#<', Expr)
        except Exception:
            pass
        yield dict(subst)

    def _builtin_fd_gt(self, term: Term, subst: Dict[str, object]):
        X, Expr = term.args
        val = self._eval_arith(Expr, subst)
        if isinstance(X, Variable) and val is not None:
            try:
                self.engine.add_fd_domain(X.name, val + 1, float('inf'))
            except Exception:
                pass
            yield dict(subst)
            return
        try:
            if isinstance(X, Variable):
                self.engine.add_fd_constraint(X.name, '#>', Expr)
        except Exception:
            pass
        yield dict(subst)

    def _builtin_freeze(self, term: Term, subst: Dict[str, object]):
        # freeze(Var, Goal) delays Goal until Var is ground; here we record into engine freeze store
        V, G = term.args
        if not isinstance(V, Variable):
            # if Var already ground, prove Goal now
            for s in self._prove(G, dict(subst)):
                yield s
            return
        # record freeze
        try:
            self.engine.add_freeze(V.name, G)
        except Exception:
            pass
        # no immediate binding change; succeed
        yield dict(subst)

    def _make_list_from_py(self, py):
        node = Constant('[]')
        for itm in reversed(py):
            node = Term('.', [itm, node])
        return node

    def _builtin_call(self, term: Term, subst: Dict[str, object]):
        # call(G) - evaluate goal term G dynamically
        G = term.args[0]
        g_ap = self._apply_subst(G, subst)
        if isinstance(g_ap, Term):
            for s in self._prove(g_ap, dict(subst), 0):
                yield s
        return

    def _builtin_bagof(self, term: Term, subst: Dict[str, object]):
        # bagof(Var, Goal, List)
        if len(term.args) != 3:
            raise UnificationError(f"bagof/3 requires exactly 3 arguments, got {len(term.args)}")
        V, G, L = term.args
        # collect all solutions for V from proving G
        seen = []
        for s in self._prove(self._apply_subst(G, dict(subst)), dict(subst), 0):
            if V.name in s:
                val = self._apply_subst(s[V.name], s)
                seen.append(val)
        # build list term
        lterm = self._make_list_from_py(seen)
        s2 = self._unify(L, lterm, dict(subst))
        if s2 is not None:
            yield s2
        return

    def _builtin_write(self, term: Term, subst: Dict[str, object]):
        # write(T) - side effect
        t = self._apply_subst(term.args[0], subst)
        print(t)
        yield dict(subst)

    def _builtin_nl(self, term: Term, subst: Dict[str, object]):
        print()
        yield dict(subst)

    def _extend_subst(self, subst: Dict[str, object], varname: str, val) -> Dict[str, object] or None:
        # avoid circular bindings: perform occurs-check if enabled
        s = dict(subst)
        if varname in s:
            return self._unify(s[varname], val, s)
        # if val is Variable bound to something, fetch its binding
        if isinstance(val, Variable) and val.name in s:
            return self._unify(Variable(varname), s[val.name], s)
        # occurs-check: do not allow varname to appear inside val when considering current bindings
        def _occurs(vname: str, term) -> bool:
            # apply current tentative substitution s when walking
            t = self._apply_subst(term, s)
            if isinstance(t, Variable):
                return t.name == vname
            if isinstance(t, Term):
                for a in t.args:
                    if _occurs(vname, a):
                        return True
            return False

        if getattr(self, 'occurs_check', False) and _occurs(varname, val):
            # by default, fail unification (return None). Optionally raise for callers that want explicit errors.
            if getattr(self, 'raise_on_occurs', False):
                raise UnificationError(f"Occurs check: variable {varname} occurs in {val}")
            return None

        s[varname] = val

        # Helper: apply subst and check structural equality (concrete equality only)
        def _concrete_equal(a, b, ssubj):
            a2 = self._apply_subst(a, ssubj)
            b2 = self._apply_subst(b, ssubj)
            # if both are Variables, can't decide
            if isinstance(a2, Variable) or isinstance(b2, Variable):
                return False
            # Constants
            if isinstance(a2, Constant) and isinstance(b2, Constant):
                return a2.value == b2.value
            # Terms: require same name and arity and recursively equal args
            if isinstance(a2, Term) and isinstance(b2, Term):
                if a2.name != b2.name or len(a2.args) != len(b2.args):
                    return False
                for xa, xb in zip(a2.args, b2.args):
                    if not _concrete_equal(xa, xb, ssubj):
                        return False
                return True
            return False

        # Ask engine to evaluate binding impact (dif violations, freeze goals)
        if hasattr(self.engine, 'on_binding'):
            try:
                violated, to_prove = self.engine.on_binding(varname, s)
            except Exception:
                violated, to_prove = False, []
            if violated:
                return None
            # attempt to prove freeze goals; if any fail, reject binding
            for g in to_prove:
                provable = False
                for _ in self._prove(self._apply_subst(g, s), dict(s), 0):
                    provable = True
                    break
                if not provable:
                    return None

        # If engine supports FD propagation, try to check consistency after binding
        if hasattr(self.engine, 'check_fd_consistency'):
            try:
                ok = self.engine.check_fd_consistency()
            except Exception:
                ok = True
            if not ok:
                return None

        return s

    def _prove_inner(self, goal: Term, subst: Dict[str, object], depth: int = 0) -> Generator[Dict[str, object], None, None]:
        """Original proving implementation (moved to _prove_inner)."""
        if depth > getattr(self, 'max_depth', 100):
            raise MaxDepthExceededError(f'Max proof depth {self.max_depth} exceeded at depth {depth} while proving {goal}')
        self.prove_count += 1
        if getattr(self, 'trace_level', 0) >= 1:
            if getattr(self, 'trace_level', 0) >= 3:
                # maintain indentation level for nested goals
                self._trace_indent = getattr(self, '_trace_indent', 0)
                logging.info(('  ' * self._trace_indent) + f'Entering goal (depth={depth}): {goal} with subst={subst}')
                self._trace_indent += 1
            else:
                logging.info(f'Entering goal (depth={depth}): {goal} with subst={subst}')

        # Special operators: negation-as-failure
        if self._is_not(goal):
            # do not allow negation to bind variables in outer scope
            subgoal = goal.args[0]
            # standardize apart subgoal
            std_sub = self._standardize_apart_term(subgoal, {}) if isinstance(subgoal, Term) else subgoal
            # try to prove subgoal; if any solution exists, negation fails
            any_sol = False
            for _ in self._prove(std_sub, dict(subst), depth + 1):
                any_sol = True
                break
            if not any_sol:
                yield subst
            return

        # Try facts
        # Builtins dispatch
        if isinstance(goal, Term) and goal.name in self.builtins:
            # optional type checking
            if getattr(self, 'type_checking', False):
                expected = self._builtin_types.get(goal.name)
                if expected is not None:
                    # check each arg
                    for idx, exp in enumerate(expected):
                        if idx >= len(goal.args):
                            break
                        arg = self._apply_subst(goal.args[idx], subst)
                        if exp == 'atom' and not isinstance(arg, Constant):
                            raise TypeError(f'builtin {goal.name} expected atom at arg {idx+1}')
                        if exp == 'list' and not (self._is_list_term(arg) or self._is_empty_list(arg)):
                            raise TypeError(f'builtin {goal.name} expected list at arg {idx+1}')
            handler = self.builtins[goal.name]
            for s in handler(goal, dict(subst)):
                yield s
            return

        for fact in self.engine.facts:
            # check for timeout interrupt
            if getattr(self, '_interrupt_flag', False):
                return
            s = self._unify(goal, fact, subst)
            if s is not None:
                if getattr(self, 'trace_level', 0) >= 2:
                    if getattr(self, 'trace_level', 0) >= 3:
                        logging.info(('  ' * getattr(self, '_trace_indent', 0)) + f'[FACT] {fact} -> subst={s}')
                    else:
                        logging.info(f'Fact matched: {fact} -> subst={s}')
                yield s
                # if a cut got set in the substitution, stop exploring more facts/rules
                if '__cut__' in s:
                    if getattr(self, 'trace_level', 0) >= 1:
                        logging.info('Cut activated by fact — stopping further matches')
                    return

        # Try rules
        # select candidate rules by predicate name/arity using engine indexes when possible
        candidates = []
        if isinstance(goal, Term):
            candidates = [r for r in self.engine.rules if r.head.name == goal.name and len(r.head.args) == len(goal.args)]
        else:
            candidates = list(self.engine.rules)
        for idx, rule in enumerate(candidates, start=1):
            std_rule = self._standardize_apart_rule(rule)
            # set current rule context for unify attribution
            key = (rule.head.name, len(rule.head.args))
            self._current_rule_key = key
            # reset per-rule temporary counters
            self._current_rule_unify_calls = 0
            self._current_rule_unify_time = 0.0
            s_head = self._unify(goal, std_rule.head, subst)
            if s_head is None:
                # clear context
                self._current_rule_key = None
                continue

            # prove all body terms sequentially
            def prove_body(idx, s_current):
                if idx >= len(std_rule.body):
                    yield s_current
                    return
                term = std_rule.body[idx]
                # handle cut specially: set flag in substitution and continue
                if self._is_cut(term):
                    if getattr(self, 'trace_level', 0) >= 1:
                        logging.info('Cut encountered in rule body — activating cut')
                    s2 = dict(s_current)
                    s2['__cut__'] = True
                    # proceed to next body element with cut active
                    yield from prove_body(idx + 1, s2)
                    return
                for s_next in self._prove(self._apply_subst(term, s_current), s_current, depth + 1):
                    # if nested subproof activated a cut, carry it forward
                    if '__cut__' in s_next:
                        if getattr(self, 'trace_level', 0) >= 1:
                            logging.info('Cut propagated from subgoal — pruning alternatives')
                        # propagate cut by yielding s_next and then stop further alternatives
                        yield from prove_body(idx + 1, s_next)
                        # after a cut, do not explore other s_next alternatives for this term
                        return
                    else:
                        yield from prove_body(idx + 1, s_next)

            import time
            start_t = time.perf_counter()
            call_count = 0
            for alt_idx, s_final in enumerate(prove_body(0, s_head), start=1):
                if getattr(self, '_interrupt_flag', False):
                    return
                if getattr(self, 'trace_level', 0) >= 3:
                    rid = getattr(rule, '_id', None)
                    logging.info(('  ' * getattr(self, '_trace_indent', 0)) + f'[RULE] trying rule id={rid} alt={alt_idx}')
                yield s_final
                call_count += 1
                # if a cut was activated in the final substitution, stop exploring other rules/facts
                if '__cut__' in s_final:
                    return
            # record profiling
            end_t = time.perf_counter()
            elapsed = end_t - start_t
            try:
                rp = getattr(self.engine, '_rule_profile', {})
                key = (rule.head.name, len(rule.head.args))
                rec = rp.get(key, {'calls': 0, 'total_time': 0.0, 'total_unify': 0})
                rec['calls'] += call_count or 1
                rec['total_time'] += elapsed
                # merge unify attribution from interpreter temporary counters
                u_calls = getattr(self, '_current_rule_unify_calls', 0)
                u_time = getattr(self, '_current_rule_unify_time', 0.0)
                rec['total_unify'] = rec.get('total_unify', 0) + u_calls
                rec['total_unify_time'] = rec.get('total_unify_time', 0.0) + u_time
                # compute averages
                rec['avg_time'] = rec['total_time'] / rec['calls'] if rec['calls'] else 0.0
                rec['avg_unify'] = (rec['total_unify_time'] / rec['total_unify']) if rec.get('total_unify', 0) else None
                rp[key] = rec
                self.engine._rule_profile = rp
            except Exception:
                pass
            # clear current rule context
            self._current_rule_key = None
        # pop indentation when leaving goal for trace level 3
        if getattr(self, 'trace_level', 0) >= 3:
            self._trace_indent = max(0, getattr(self, '_trace_indent', 1) - 1)

    def _prove(self, goal: Term, subst: Dict[str, object], depth: int = 0) -> Generator[Dict[str, object], None, None]:
        """Wrapper around _prove_inner that performs optional tabling (memoization).

        Currently only caches answers for fully-ground goals (no variables after applying
        the current substitution). The table is stored on the engine (`engine._table`) so
        it survives across queries when desired.
        """
        # simple helper to canonicalize a ground term into a key string
        def _canonical_key(g: Term):
            # Represent as name/arity plus string of recursively rendered args
            def render(x):
                if isinstance(x, Constant):
                    return f"C:{x.value}"
                if isinstance(x, Term):
                    return f"T:{x.name}({','.join(render(a) for a in x.args)})"
                if isinstance(x, Variable):
                    return f"V:{x.name}"
                return repr(x)
            return (g.name, len(g.args) if isinstance(g, Term) else 0, render(g))

        # If tabling disabled, just call inner prover
        if not getattr(self, 'tabling', False):
            yield from self._prove_inner(goal, subst, depth)
            return

        # attempt to compute table key for the goal under substitution
        applied = self._apply_subst(goal, subst)
        # only table Term goals (not builtins, negation) and only ground ones
        if not isinstance(applied, Term) or self._is_not(applied) or (isinstance(applied, Term) and any(isinstance(a, Variable) for a in self._collect_vars(applied))):
            # not safe to table non-ground or special goals
            yield from self._prove_inner(goal, subst, depth)
            return

        key = _canonical_key(applied)
        table = getattr(self.engine, '_table', None)
        if table is None:
            # initialize table on engine
            self.engine._table = {}
            table = self.engine._table

        entry = table.get(key)
        # in-progress sentinel to prevent re-entrance loops
        IN_PROGRESS = object()
        if entry is IN_PROGRESS:
            # recursive call encountered; avoid infinite loop by yielding nothing for now
            return
        if entry is not None and entry is not IN_PROGRESS:
            # return cached answers; need to remap stored ground answers into current subst
            for ans in entry:
                # ans is a dict varname->ground term; yield a copy
                yield dict(ans)
            return

        # mark as in-progress
        table[key] = IN_PROGRESS
        answers = []
        try:
            for s in self._prove_inner(goal, subst, depth):
                # only store ground answers (no variables)
                ground_ans = {}
                for k, v in s.items():
                    val = self._apply_subst(v, s)
                    if isinstance(val, Variable):
                        # non-ground, don't include in table
                        ground_ans = None
                        break
                    ground_ans[k] = val
                if ground_ans is not None:
                    answers.append(ground_ans)
                yield s
        finally:
            # cache answers if any
            table[key] = answers

    def _start_timeout(self):
        # set up a threading.Timer to set interrupt flag after timeout_seconds
        import threading
        if getattr(self, 'timeout_seconds', None) is None:
            return None
        self._interrupt_flag = False
        def _set():
            self._interrupt_flag = True
        t = threading.Timer(self.timeout_seconds, _set)
        t.daemon = True
        t.start()
        return t

    def _stop_timeout(self, timer):
        if timer is None:
            return
        try:
            timer.cancel()
        except Exception:
            pass

    def query(self, goal: Term) -> List[Dict[str, object]]:
        # collect variable names from original goal to return only their bindings
        query_vars = {v.name for v in self._collect_vars(goal)}
        results: List[Dict[str, object]] = []
        # If subprocess fallback enabled, use run_with_timeout helper which
        # will attempt cooperative check first, then run in subprocess if
        # the interpreter doesn't yield within timeout.
        if getattr(self, 'use_subprocess_timeout', False) and getattr(self, 'timeout_seconds', None) is not None:
            proveds = self.run_with_timeout(goal)
        else:
            proveds = list(self._prove(goal, {}))

        for subst in proveds:
            # check dif constraints stored in engine
            violated = False
            for a_b in getattr(self.engine, '_dif_constraints', []):
                A, B = a_b
                av = self._apply_subst(A, subst)
                bv = self._apply_subst(B, subst)
                if isinstance(av, Constant) and isinstance(bv, Constant) and av.value == bv.value:
                    violated = True
                    break
            if violated:
                continue

            # try to activate any freeze goals whose variable is now ground
            for varname, goalterm in list(getattr(self.engine, '_freeze_store', [])):
                # if var appears in subst and is ground, attempt to prove goalterm
                if varname in subst:
                    vval = self._apply_subst(subst[varname], subst)
                    if not isinstance(vval, Variable):
                        # remove freeze and attempt prove; if fails, skip activation
                        try:
                            self.engine._freeze_store.remove((varname, goalterm))
                        except ValueError:
                            pass
                        for _ in self._prove(goalterm, dict(subst)):
                            break

            # build result binding mapping only query vars to applied substitutions
            binding = {}
            for qn in query_vars:
                if qn in subst:
                    binding[qn] = self._apply_subst(subst[qn], subst)
            results.append(binding)
        return results

    def run_with_timeout(self, goal: Term):
        """Attempt to run _prove(goal,{}) respecting timeout_seconds.

        First try cooperative run for a short slice. If the interrupt flag
        becomes set, abort. If cooperative run doesn't complete quickly but
        no interrupt observed, run the proof in a subprocess to allow hard
        termination.
        """
        # cooperative try first: small slice
        import time
        start = time.perf_counter()
        slice_limit = min(0.05, getattr(self, 'timeout_seconds', 0.05))
        # start a timer to set interrupt flag after timeout_seconds
        timer = self._start_timeout()
        try:
            yielded = []
            for i, s in enumerate(self._prove(goal, {})):
                yielded.append(s)
                # stop if we've exceeded the short slice to avoid blocking
                if time.perf_counter() - start > slice_limit:
                    break
            # if we got any results quickly, return them
            if yielded:
                return yielded
            # if interrupt flag is set, return empty
            if getattr(self, '_interrupt_flag', False):
                return []
        finally:
            self._stop_timeout(timer)

        # fallback to subprocess runner for heavy queries
        try:
            from multiprocessing import Process, Queue

            def worker(q, eng, goal_ast):
                # Worker re-imports minimal runtime to run proof in isolation
                try:
                    # run a tiny interpreter instance bound to the same engine data
                    from .interpreter import Interpreter as _Interpreter
                    interp = _Interpreter(eng)
                    res = []
                    for s in interp._prove(goal_ast, {}):
                        res.append(s)
                    q.put(('ok', res))
                except Exception as e:
                    q.put(('err', str(e)))

            q = Queue()
            p = Process(target=worker, args=(q, self.engine, goal))
            p.start()
            p.join(timeout=self.timeout_seconds)
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
                p.join(0.1)
                return []
            # collect results
            if not q.empty():
                tag, payload = q.get()
                if tag == 'ok':
                    return payload
                return []
        except Exception:
            return []

    def _collect_vars(self, x):
        if isinstance(x, Variable):
            return [x]
        if isinstance(x, Term):
            res = []
            for a in x.args:
                res.extend(self._collect_vars(a))
            return res
        return []

