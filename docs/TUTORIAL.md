# egdol Tutorial

## Getting Started

### Installation

```bash
pip install egdol
```

### Basic Usage

```python
from egdol import RulesEngine, Interpreter
from egdol.parser import Term, Variable, Constant

# Create engine and interpreter
engine = RulesEngine()
interp = Interpreter(engine)

# Add facts
engine.add_fact(Term('parent', [Constant('john'), Constant('mary')]))
engine.add_fact(Term('parent', [Constant('mary'), Constant('bob')]))

# Add rules
from egdol.parser import Rule
engine.add_rule(Rule(
    Term('ancestor', [Variable('X'), Variable('Y')]),
    [Term('parent', [Variable('X'), Variable('Y')])]
))

engine.add_rule(Rule(
    Term('ancestor', [Variable('X'), Variable('Z')]),
    [Term('parent', [Variable('X'), Variable('Y')]), 
     Term('ancestor', [Variable('Y'), Variable('Z')])]
))

# Query
results = list(interp.query(Term('ancestor', [Constant('john'), Variable('X')])))
print(results)  # [{'X': 'mary'}, {'X': 'bob'}]
```

## Interactive REPL

### Starting the REPL

```bash
python -m egdol.main
```

### REPL Commands

- `:facts` - Show all facts
- `:rules` - Show all rules  
- `:stats` - Show statistics
- `:profile on/off` - Enable/disable profiling
- `:trace` - Toggle tracing
- `:help` - Show help

### Example Session

```
egdol> fact: parent(john, mary).
egdol> fact: parent(mary, bob).
egdol> rule: ancestor(X,Y) :- parent(X,Y).
egdol> rule: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z).
egdol> ? ancestor(john, X).
X = mary
X = bob
egdol> :stats
Facts: 2, Rules: 2, Queries: 1
```

## Advanced Features

### Constraint Programming

#### Difference Constraints

```python
# X != Y
engine.add_dif_constraint(Variable('X'), Variable('Y'))
```

#### Finite Domain Constraints

```python
# X in 1..10
engine.add_fd_domain('X', 1, 10)

# X = Y + 1
engine.add_fd_constraint('X', '#=', Term('+', [Variable('Y'), Constant(1)]))
```

### Performance Profiling

```python
# Enable profiling
interp.profile = True

# Run queries
results = list(interp.query(goal))

# Get profile data
from egdol.main import format_profile_summary
text, structured = format_profile_summary(engine, interp)
print(text)
```

### Custom Built-ins

```python
def my_builtin(interp, goal, subst):
    # Custom logic here
    return [subst]  # Return list of substitutions

interp.builtins['my_predicate'] = my_builtin
```

## File I/O

### Loading from File

```python
# Load Prolog file
engine.import_prolog('knowledge.pl')

# Or use REPL command
# :load knowledge.pl
```

### Saving to File

```python
# Export to Prolog
engine.export_prolog('output.pl')

# Or use REPL command  
# :save output.pl
```

## GUI Interface

### Starting GUI

```bash
python -m egdol.main --gui
```

The GUI provides:
- Tree view of facts and rules
- Query interface
- Results table
- Trace/log panel
- Statistics display

## Performance Tips

### Optimization Strategies

1. **Indexing**: Use specific predicate names for better indexing
2. **Cut Operator**: Use `!` to prune search space
3. **Ordering**: Put more selective goals first
4. **Profiling**: Use built-in profiler to identify bottlenecks

### Example Optimized Rule

```prolog
# Good: selective goal first
ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z).

# Better: add cut if appropriate
ancestor(X,Z) :- parent(X,Y), !, ancestor(Y,Z).
```

## Common Patterns

### List Processing

```python
# member/2
engine.add_rule(Rule(
    Term('member', [Variable('X'), Term('.', [Variable('X'), Variable('_')])]),
    []
))

engine.add_rule(Rule(
    Term('member', [Variable('X'), Term('.', [Variable('_'), Variable('T')])]),
    [Term('member', [Variable('X'), Variable('T')])]
))
```

### Arithmetic

```python
# length/2
engine.add_rule(Rule(
    Term('length', [Constant('[]'), Constant('0')]),
    []
))

engine.add_rule(Rule(
    Term('length', [Term('.', [Variable('_'), Variable('T')]), Variable('N')]),
    [Term('length', [Variable('T'), Term('-', [Variable('N'), Constant('1')])])]
))
```

## Troubleshooting

### Common Issues

1. **Infinite Loops**: Use `max_depth` setting
2. **Slow Queries**: Enable profiling to identify bottlenecks
3. **Memory Usage**: Monitor with `:stats` command
4. **Unification Errors**: Check variable naming and term structure

### Debugging

```python
# Enable tracing
interp.trace_level = 2

# Set depth limit
interp.max_depth = 50

# Run query with debugging
results = list(interp.query(goal))
```
