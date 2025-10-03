# egdol API Reference

## Core Classes

### RulesEngine

The main engine for managing facts, rules, and constraints.

```python
from egdol import RulesEngine

engine = RulesEngine()
```

#### Methods

- `add_fact(fact: Term)` - Add a fact to the knowledge base
- `add_rule(rule: Rule)` - Add a rule to the knowledge base
- `query(goal: Term) -> List[Dict]` - Query the knowledge base
- `stats() -> Dict` - Get engine statistics
- `export_prolog(path: str)` - Export to Prolog format
- `import_prolog(path: str)` - Import from Prolog format

### Interpreter

The inference engine that executes queries.

```python
from egdol import Interpreter

interp = Interpreter(engine)
```

#### Methods

- `query(goal: Term) -> Generator[Dict]` - Execute a query
- `prove(goal: Term, subst: Dict) -> Generator[Dict]` - Prove a goal with substitution

#### Configuration

- `max_depth: int` - Maximum proof depth (default: 100)
- `trace_level: int` - Trace verbosity (0-3)
- `occurs_check: bool` - Enable occurs check for unification
- `timeout_seconds: float` - Query timeout

### Parser Classes

#### Term

Represents a Prolog term.

```python
from egdol.parser import Term, Variable, Constant

# Atoms
atom = Constant('hello')

# Variables  
var = Variable('X')

# Complex terms
term = Term('parent', [Variable('X'), Constant('john')])
```

#### Rule

Represents a Prolog rule.

```python
from egdol.parser import Rule

rule = Rule(
    head=Term('ancestor', [Variable('X'), Variable('Z')]),
    body=[Term('parent', [Variable('X'), Variable('Y')]), 
          Term('ancestor', [Variable('Y'), Variable('Z')])]
)
```

## Built-in Predicates

### Arithmetic

- `is/2` - Arithmetic evaluation
- `</2`, `>/2`, `=</2`, `>=/2` - Numeric comparisons
- `=:=/2`, `=\=/2` - Arithmetic equality/inequality

### List Operations

- `member/2` - List membership
- `append/3` - List concatenation
- `length/2` - List length

### String Operations

- `atom_concat/3` - String concatenation
- `atom_length/2` - String length

### Control

- `!/0` - Cut operator
- `fail/0` - Always fails
- `true/0` - Always succeeds

## Constraint Programming

### Difference Constraints

```python
engine.add_dif_constraint(Variable('X'), Constant('1'))
```

### Finite Domain Constraints

```python
engine.add_fd_domain('X', 1, 10)  # X in 1..10
engine.add_fd_constraint('X', '#=', Term('+', [Variable('Y'), Constant(1)]))
```

## Performance Profiling

### Enable Profiling

```python
interp.profile = True
```

### Get Profile Data

```python
from egdol.main import format_profile_summary

text, structured = format_profile_summary(engine, interp)
print(text)
```

### Export Profile

```python
from egdol.main import export_profile_json

with open('profile.json', 'w') as f:
    export_profile_json(engine, interp, f)
```

## Error Handling

### Exceptions

- `ParseError` - Syntax errors in input
- `MaxDepthExceededError` - Recursion depth exceeded
- `UnificationError` - Unification failures

### Example

```python
from egdol.interpreter import MaxDepthExceededError

try:
    results = list(interp.query(goal))
except MaxDepthExceededError:
    print("Query exceeded maximum depth")
```
