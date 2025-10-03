"""
Abstract Syntax Tree (AST) classes for the Egdol DSL.
Represents the parsed structure of DSL statements.
"""

from typing import List, Optional, Union, Any
from abc import ABC, abstractmethod


class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor for traversal."""
        pass


class Program(ASTNode):
    """Root node representing a complete program."""
    
    def __init__(self, statements: List['Statement']):
        self.statements = statements
        
    def accept(self, visitor):
        return visitor.visit_program(self)


class Statement(ASTNode):
    """Base class for all statements."""
    pass


class FactStatement(Statement):
    """Base class for fact statements."""
    pass


class IsFact(FactStatement):
    """Represents 'X is Y' facts."""
    
    def __init__(self, subject: 'Subject', predicate: 'Predicate'):
        self.subject = subject
        self.predicate = predicate
        
    def accept(self, visitor):
        return visitor.visit_is_fact(self)


class HasFact(FactStatement):
    """Represents 'X has Y' facts."""
    
    def __init__(self, subject: 'Subject', property_: 'Predicate'):
        self.subject = subject
        self.property = property_
        
    def accept(self, visitor):
        return visitor.visit_has_fact(self)


class RuleStatement(Statement):
    """Base class for rule statements."""
    pass


class IfThenRule(RuleStatement):
    """Represents 'if X then Y' rules."""
    
    def __init__(self, condition: 'Condition', conclusion: FactStatement):
        self.condition = condition
        self.conclusion = conclusion
        
    def accept(self, visitor):
        return visitor.visit_if_then_rule(self)


class WhenThenRule(RuleStatement):
    """Represents 'when X then Y' rules."""
    
    def __init__(self, condition: 'Condition', conclusion: FactStatement):
        self.condition = condition
        self.conclusion = conclusion
        
    def accept(self, visitor):
        return visitor.visit_when_then_rule(self)


class QueryStatement(Statement):
    """Base class for query statements."""
    pass


class WhoQuery(QueryStatement):
    """Represents 'who is X' queries."""
    
    def __init__(self, predicate: 'Predicate'):
        self.predicate = predicate
        
    def accept(self, visitor):
        return visitor.visit_who_query(self)


class WhatQuery(QueryStatement):
    """Represents 'what is X' queries."""
    
    def __init__(self, subject: 'Subject'):
        self.subject = subject
        
    def accept(self, visitor):
        return visitor.visit_what_query(self)


class DoesQuery(QueryStatement):
    """Represents 'does X have Y' queries."""
    
    def __init__(self, subject: 'Subject', property_: 'Predicate'):
        self.subject = subject
        self.property = property_
        
    def accept(self, visitor):
        return visitor.visit_does_query(self)


class ShowQuery(QueryStatement):
    """Represents 'show me all X' queries."""
    
    def __init__(self, predicate: 'Predicate'):
        self.predicate = predicate
        
    def accept(self, visitor):
        return visitor.visit_show_query(self)


class IsQuery(QueryStatement):
    """Represents 'is X Y' queries."""
    
    def __init__(self, subject: 'Subject', predicate: 'Predicate'):
        self.subject = subject
        self.predicate = predicate
        
    def accept(self, visitor):
        return visitor.visit_is_query(self)


class GenericQuery(QueryStatement):
    """Represents generic queries."""
    
    def __init__(self, subject: 'Subject'):
        self.subject = subject
        
    def accept(self, visitor):
        return visitor.visit_generic_query(self)


class CommandStatement(Statement):
    """Represents command statements (:command)."""
    
    def __init__(self, command: str, args: List[Any]):
        self.command = command
        self.args = args
        
    def accept(self, visitor):
        return visitor.visit_command(self)


class Subject(ASTNode):
    """Base class for subjects."""
    pass


class ProperNounSubject(Subject):
    """Represents proper noun subjects (e.g., 'Alice')."""
    
    def __init__(self, name: str):
        self.name = name
        
    def accept(self, visitor):
        return visitor.visit_proper_noun_subject(self)


class VariableSubject(Subject):
    """Represents variable subjects (e.g., 'X')."""
    
    def __init__(self, name: str):
        self.name = name
        
    def accept(self, visitor):
        return visitor.visit_variable_subject(self)


class PronounSubject(Subject):
    """Represents pronoun subjects (e.g., 'he', 'she')."""
    
    def __init__(self, pronoun: str, resolved: Optional[str] = None):
        self.pronoun = pronoun
        self.resolved = resolved
        
    def accept(self, visitor):
        return visitor.visit_pronoun_subject(self)


class Predicate(ASTNode):
    """Base class for predicates."""
    pass


class CommonNounPredicate(Predicate):
    """Represents common noun predicates (e.g., 'human')."""
    
    def __init__(self, noun: str):
        self.noun = noun
        
    def accept(self, visitor):
        return visitor.visit_common_noun_predicate(self)


class ProperNounPredicate(Predicate):
    """Represents proper noun predicates (e.g., 'Alice')."""
    
    def __init__(self, name: str):
        self.name = name
        
    def accept(self, visitor):
        return visitor.visit_proper_noun_predicate(self)


class AdjectivePredicate(Predicate):
    """Represents adjective predicates (e.g., 'smart')."""
    
    def __init__(self, adjective: str):
        self.adjective = adjective
        
    def accept(self, visitor):
        return visitor.visit_adjective_predicate(self)


class Condition(ASTNode):
    """Base class for conditions."""
    pass


class SimpleCondition(Condition):
    """Represents simple conditions."""
    
    def __init__(self, fact: FactStatement):
        self.fact = fact
        
    def accept(self, visitor):
        return visitor.visit_simple_condition(self)


class NotCondition(Condition):
    """Represents negated conditions."""
    
    def __init__(self, fact: FactStatement):
        self.fact = fact
        
    def accept(self, visitor):
        return visitor.visit_not_condition(self)


class AndCondition(Condition):
    """Represents conjunction of conditions."""
    
    def __init__(self, left: Condition, right: Condition):
        self.left = left
        self.right = right
        
    def accept(self, visitor):
        return visitor.visit_and_condition(self)


class OrCondition(Condition):
    """Represents disjunction of conditions."""
    
    def __init__(self, left: Condition, right: Condition):
        self.left = left
        self.right = right
        
    def accept(self, visitor):
        return visitor.visit_or_condition(self)


class ASTVisitor(ABC):
    """Base class for AST visitors."""
    
    def visit_program(self, node: Program):
        return self.visit_statements(node.statements)
        
    def visit_statements(self, statements: List[Statement]):
        return [stmt.accept(self) for stmt in statements]
        
    def visit_is_fact(self, node: IsFact):
        return {
            'type': 'is_fact',
            'subject': node.subject.accept(self),
            'predicate': node.predicate.accept(self)
        }
        
    def visit_has_fact(self, node: HasFact):
        return {
            'type': 'has_fact',
            'subject': node.subject.accept(self),
            'property': node.property.accept(self)
        }
        
    def visit_if_then_rule(self, node: IfThenRule):
        return {
            'type': 'if_then_rule',
            'condition': node.condition.accept(self),
            'conclusion': node.conclusion.accept(self)
        }
        
    def visit_when_then_rule(self, node: WhenThenRule):
        return {
            'type': 'when_then_rule',
            'condition': node.condition.accept(self),
            'conclusion': node.conclusion.accept(self)
        }
        
    def visit_who_query(self, node: WhoQuery):
        return {
            'type': 'who_query',
            'predicate': node.predicate.accept(self)
        }
        
    def visit_what_query(self, node: WhatQuery):
        return {
            'type': 'what_query',
            'subject': node.subject.accept(self)
        }
        
    def visit_does_query(self, node: DoesQuery):
        return {
            'type': 'does_query',
            'subject': node.subject.accept(self),
            'property': node.property.accept(self)
        }
        
    def visit_show_query(self, node: ShowQuery):
        return {
            'type': 'show_query',
            'predicate': node.predicate.accept(self)
        }
        
    def visit_is_query(self, node: IsQuery):
        return {
            'type': 'is_query',
            'subject': node.subject.accept(self),
            'predicate': node.predicate.accept(self)
        }
        
    def visit_generic_query(self, node: GenericQuery):
        return {
            'type': 'generic_query',
            'subject': node.subject.accept(self)
        }
        
    def visit_command(self, node: CommandStatement):
        return {
            'type': 'command',
            'command': node.command,
            'args': node.args
        }
        
    def visit_proper_noun_subject(self, node: ProperNounSubject):
        return {'type': 'proper_noun', 'name': node.name}
        
    def visit_variable_subject(self, node: VariableSubject):
        return {'type': 'variable', 'name': node.name}
        
    def visit_pronoun_subject(self, node: PronounSubject):
        return {
            'type': 'pronoun',
            'pronoun': node.pronoun,
            'resolved': node.resolved
        }
        
    def visit_common_noun_predicate(self, node: CommonNounPredicate):
        return {'type': 'common_noun', 'noun': node.noun}
        
    def visit_proper_noun_predicate(self, node: ProperNounPredicate):
        return {'type': 'proper_noun', 'name': node.name}
        
    def visit_adjective_predicate(self, node: AdjectivePredicate):
        return {'type': 'adjective', 'adjective': node.adjective}
        
    def visit_simple_condition(self, node: SimpleCondition):
        return {
            'type': 'simple_condition',
            'fact': node.fact.accept(self)
        }
        
    def visit_not_condition(self, node: NotCondition):
        return {
            'type': 'not_condition',
            'fact': node.fact.accept(self)
        }
        
    def visit_and_condition(self, node: AndCondition):
        return {
            'type': 'and_condition',
            'left': node.left.accept(self),
            'right': node.right.accept(self)
        }
        
    def visit_or_condition(self, node: OrCondition):
        return {
            'type': 'or_condition',
            'left': node.left.accept(self),
            'right': node.right.accept(self)
        }
