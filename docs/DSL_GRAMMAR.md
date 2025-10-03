# Egdol DSL Grammar Specification

## Overview
The Egdol DSL provides a natural English-like syntax for defining facts, rules, and queries that compile down to the core Egdol engine.

## Grammar (EBNF)

```
program = statement*

statement = fact_statement | rule_statement | query_statement | command_statement

fact_statement = subject "is" predicate
                | subject "is" "a" predicate
                | subject "has" property
                | subject "is" adjective

rule_statement = "if" condition "then" conclusion
                | conclusion "if" condition
                | "when" condition "then" conclusion

query_statement = "who" "is" predicate
                | "what" "is" subject
                | "is" subject predicate
                | "does" subject "have" property
                | "show" "me" "all" predicate

command_statement = ":" command_name [argument*]

subject = proper_noun | variable | pronoun
predicate = common_noun | adjective
property = common_noun
conclusion = fact_statement
condition = fact_statement | conjunction | disjunction

conjunction = condition "and" condition
disjunction = condition "or" condition

proper_noun = [A-Z][a-z]+
common_noun = [a-z]+
variable = [A-Z]
pronoun = "he" | "she" | "it" | "they" | "him" | "her" | "them"
adjective = [a-z]+

command_name = "facts" | "rules" | "reset" | "save" | "load" | "trace" | "help"
argument = string | number | identifier
```

## Examples

### Facts
```
Person Alice is a human.
Human X is mortal.
Alice has age 25.
Alice is smart.
```

### Rules
```
If X is a human then X is mortal.
X is mortal if X is a human.
When X is a bird then X can fly.
```

### Queries
```
Who is mortal?
What is Alice?
Is Alice mortal?
Does Alice have age 25?
Show me all humans.
```

### Commands
```
:facts
:rules
:reset
:save session.egdol
:load session.egdol
:trace who is mortal?
:help
```

## Context Memory

The DSL supports context memory through pronouns and variables:

```
Person Bob is a human.
He is mortal.
What about him?
```

The system maintains a context stack and resolves pronouns to the most recent subject.

## Advanced Features

### Negation
```
Alice is not a robot.
If X is not a bird then X cannot fly.
```

### Quantifiers
```
All humans are mortal.
Some birds can fly.
No robot is human.
```

### Temporal Logic
```
Alice was born in 1990.
Alice will be 35 in 2025.
```

### Arithmetic
```
Alice is 25 years old.
If X is Y years old then X was born in (2024 - Y).
```
