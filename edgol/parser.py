
"""Parser for egdol mini-DSL.

Produces simple AST nodes:
- Term(name, args)
- Fact(term)
- Rule(head, body)
- Query(term)

Basic grammar supported:
 statement  := 'fact' ':' term '.'
              | 'rule' ':' term '=>' term_list '.'
              | '?' term '.'
 term       := IDENT ['(' arg_list ')']
 arg_list   := arg (',' arg)*
 arg        := term | VAR | IDENT

"""
from dataclasses import dataclass
from typing import List, Union


class ParseError(Exception):
    def __init__(self, message, token=None):
        self.token = token
        if token is not None and hasattr(token, 'line'):
            super().__init__(f"{message} (line {token.line}, col {token.col})")
        else:
            super().__init__(message)


@dataclass
class Term:
    name: str
    args: List[Union['Term', 'Variable', 'Constant']]

    def __repr__(self):
        if self.args:
            return f"{self.name}({', '.join(map(repr, self.args))})"
        return f"{self.name}"

    def __str__(self):
        if self.args:
            return f"{self.name}({', '.join(map(str, self.args))})"
        return self.name


@dataclass
class Variable:
    name: str

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


@dataclass
class Constant:
    value: Union[str, int, float]

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        # always return a string representation
        return str(self.value)


@dataclass
class Fact:
    term: Term


@dataclass
class Rule:
    head: Term
    body: List[Term]


@dataclass
class Query:
    term: Term


@dataclass
class Module:
    name: str


class Parser:
    def __init__(self, tokens: List):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self):
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def expect(self, ttype: str, value: str = None):
        tok = self.peek()
        if tok is None:
            raise ParseError(f"Expected {ttype} but got EOF")
        if tok.type != ttype:
            raise ParseError(f"Expected {ttype} at token", tok)
        if value is not None and tok.value != value:
            raise ParseError(f"Expected {value!r} at token", tok)
        return self.advance()

    def accept(self, ttype: str, value: str = None):
        tok = self.peek()
        if tok and tok.type == ttype and (value is None or tok.value == value):
            return self.advance()
        return None

    def parse(self) -> List[Union[Fact, Rule, Query]]:
        nodes: List[Union[Fact, Rule, Query]] = []
        while self.peek() is not None:
            nodes.append(self.parse_statement())
        return nodes

    def parse_statement(self):
        tok = self.peek()
        # module declaration: module: name.
        if tok.type == 'IDENT' and tok.value == 'module':
            self.advance()
            self.expect('COLON')
            name_tok = self.expect('IDENT')
            self.expect('DOT')
            return Module(name_tok.value)
        if tok.type == 'IDENT' and tok.value == 'fact':
            self.advance()
            self.expect('COLON')
            term = self.parse_term()
            self.expect('DOT')
            return Fact(term)
        if tok.type == 'IDENT' and tok.value == 'rule':
            self.advance()
            self.expect('COLON')
            head = self.parse_term()
            self.expect('ARROW')
            body = self.parse_term_list()
            self.expect('DOT')
            return Rule(head, body)
        if tok.type == 'QUERY':
            self.advance()
            term = self.parse_term()
            self.expect('DOT')
            return Query(term)
        # allow bare Prolog-style statements like: p(a). or 1.
        if tok.type in ('IDENT', 'NUMBER', 'STRING', 'VAR', 'LPAREN', 'LBRACK'):
            # parse a general expression/term
            term = self.parse_expression()
            # accept either a trailing DOT or EOF
            nxt = self.peek()
            if nxt is not None and nxt.type == 'DOT':
                self.expect('DOT')
            return Fact(term)
        raise ParseError(f"Unexpected token {tok.type}({tok.value})", tok)

    def parse_term_list(self) -> List[Term]:
        # terms in a body may include disjunctions, so parse each element
        # with parse_disjunction which understands ';'
        terms = [self.parse_disjunction()]
        while self.accept('COMMA'):
            terms.append(self.parse_disjunction())
        return terms

    def parse_term(self) -> Term:
        tok = self.peek()
        if tok is None:
            raise ParseError('Unexpected EOF while parsing term')
        # support cut symbol token '!' as a special zero-arg term named '!'
        if tok.type == 'CUT':
            self.advance()
            return Term('!', [])

        if tok.type != 'IDENT':
            raise ParseError(f'Expected IDENT for term name at {tok.pos}, got {tok.type}')
        name = self.advance().value
        # support qualified predicate names like module:pred
        if self.accept('COLON'):
            # next must be IDENT
            mod_name = name
            nxt = self.expect('IDENT')
            name = f"{mod_name}:{nxt.value}"
        args: List[Union[Term, Variable, Constant]] = []
        if self.accept('LPAREN'):
            # parse args until RPAREN
            if self.accept('RPAREN'):
                return Term(name, [])
            while True:
                arg = self.parse_arg()
                args.append(arg)
                if self.accept('COMMA'):
                    continue
                self.expect('RPAREN')
                break
        return Term(name, args)

    def parse_arg(self) -> Union[Term, Variable, Constant]:
        tok = self.peek()
        if tok is None:
            raise ParseError('Unexpected EOF in arguments')
        if tok.type == 'VAR':
            return Variable(self.advance().value)
        # Allow numbers and expressions with basic infix operators
        return self.parse_expression()

    def parse_expression(self):
        # Parse simple expressions with precedence: *,/ then +,- then comparisons
        node = self.parse_term_or_atom()
        # handle + and -
        while True:
            tok = self.peek()
            if tok and tok.type in ('PLUS', 'MINUS'):
                op = self.advance().type
                right = self.parse_term_or_atom()
                name = '+' if op == 'PLUS' else '-'
                node = Term(name, [node, right])
                continue
            break
        # comparisons
        tok = self.peek()
        if tok and tok.type in ('LT', 'GT', 'LE', 'GE', 'EQ', 'NEQ'):
            opmap = {'LT': '<', 'GT': '>', 'LE': '<=', 'GE': '>=', 'EQ': '=', 'NEQ': 'dif'}
            op = self.advance().type
            right = self.parse_term_or_atom()
            opname = opmap[op]
            # map NEQ to a 'dif' term for compatibility with engine
            if opname == 'dif':
                return Term('dif', [node, right])
            return Term(opname, [node, right])
        return node

    def parse_disjunction(self):
        # parse left side (which may be a term/expression) and optionally ';' right side
        left = self.parse_expression()
        if self.accept('SEMI'):
            right = self.parse_disjunction()
            return Term(';', [left, right])
        return left

    # New syntax: domain specification like VAR 'in' NUMBER '..' NUMBER
    # We'll detect a pattern VAR IDENT('in') NUMBER RANGE NUMBER
    def parse_domain_if_present(self):
        # lookahead to detect VAR IDENT 'in' NUMBER RANGE NUMBER
        if self.pos + 4 < len(self.tokens):
            t0 = self.tokens[self.pos]
            t1 = self.tokens[self.pos + 1]
            t2 = self.tokens[self.pos + 2]
            t3 = self.tokens[self.pos + 3]
            t4 = self.tokens[self.pos + 4]
            # detect X in 1..5
            if t0.type == 'VAR' and t1.type == 'IDENT' and t1.value == 'in' and t2.type == 'NUMBER' and t3.type == 'RANGE' and t4.type == 'NUMBER':
                # consume VAR, 'in', NUMBER, '..', NUMBER and produce Term('in_range', [Variable, Constant, Constant])
                var = Variable(self.advance().value)
                self.advance()  # consume 'in'
                low = self.expect('NUMBER').value
                self.expect('RANGE')
                high = self.expect('NUMBER').value
                # convert numbers
                try:
                    lnum = int(low) if '.' not in low else float(low)
                except Exception:
                    lnum = int(low)
                try:
                    hnum = int(high) if '.' not in high else float(high)
                except Exception:
                    hnum = int(high)
                return Term('in_range', [var, Constant(lnum), Constant(hnum)])
        # detect X in {1,2,3}
        if self.pos + 3 < len(self.tokens):
            t0 = self.tokens[self.pos]
            t1 = self.tokens[self.pos + 1]
            t2 = self.tokens[self.pos + 2]
            if t0.type == 'VAR' and t1.type == 'IDENT' and t1.value == 'in' and t2.type == 'LBRACE':
                # consume VAR and 'in'
                var = Variable(self.advance().value)
                self.advance()
                # expect LBRACE
                self.expect('LBRACE')
                items = []
                while True:
                    tok = self.peek()
                    if tok is None:
                        raise ParseError('Unterminated set domain')
                    if tok.type == 'NUMBER':
                        num = self.advance().value
                        try:
                            v = int(num) if '.' not in num else float(num)
                        except Exception:
                            v = int(num)
                        items.append(Constant(v))
                        if self.accept('COMMA'):
                            continue
                        self.expect('RBRACE')
                        break
                    else:
                        raise ParseError('Unexpected token in set domain')
                return Term('in_set', [var] + items)
        return None

    def parse_term_or_atom(self):
        tok = self.peek()
        if tok is None:
            raise ParseError('Unexpected EOF in expression')
        if tok.type == 'NUMBER':
            value = self.advance().value
            # convert numeric token to int or float
            try:
                if '.' in value:
                    num = float(value)
                else:
                    num = int(value)
            except Exception:
                num = int(value)
            return Constant(num)
        if tok.type == 'STRING':
            value = self.advance().value
            # strip surrounding quotes and unescape simple escapes
            inner = value[1:-1].encode('utf-8').decode('unicode_escape')
            return Constant(inner)
        if tok.type == 'VAR':
            return Variable(self.advance().value)
        if tok.type == 'IDENT':
            # nested term or constant
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == 'LPAREN':
                return self.parse_term()
            return Constant(self.advance().value)
        if tok.type == 'LPAREN':
            self.advance()
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr
        if tok.type == 'LBRACK':
            # list parsing
            self.advance()
            if self.accept('RBRACK'):
                return Constant('[]')
            # parse first element
            head = self.parse_expression()
            items = [head]
            while self.accept('COMMA'):
                items.append(self.parse_expression())
            tail = None
            if self.accept('BAR'):
                tail = self.parse_expression()
                self.expect('RBRACK')
            else:
                self.expect('RBRACK')
                # build proper tail as []
                tail = Constant('[]')
            # build nested '.' terms
            node = tail
            for itm in reversed(items):
                node = Term('.', [itm, node])
            return node
        raise ParseError(f'Unexpected token {tok.type}({tok.value}) in expression at {tok.pos}')

