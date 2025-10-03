"""
DSL to Egdol Translator.
Converts AST nodes to Egdol facts, rules, and queries.
"""

from typing import List, Dict, Any, Optional, Union
from .ast import *
from ..rules_engine import RulesEngine
from ..parser import Term, Variable, Constant, Rule, Fact
from ..interpreter import Interpreter


class DSLTranslator:
    """Translates DSL AST to Egdol internal representation."""
    
    def __init__(self, engine: RulesEngine):
        self.engine = engine
        self.interpreter = Interpreter(engine)
        self.context_memory: List[Dict[str, Any]] = []
        
    def translate_program(self, program: Program) -> Dict[str, Any]:
        """Translate a complete program."""
        results = {
            'facts': [],
            'rules': [],
            'queries': [],
            'commands': []
        }
        
        for statement in program.statements:
            result = self._translate_statement(statement)
            if result:
                if 'type' in result:
                    if result['type'] == 'fact':
                        results['facts'].append(result)
                    elif result['type'] == 'rule':
                        results['rules'].append(result)
                    elif result['type'] == 'query':
                        results['queries'].append(result)
                    elif result['type'] == 'command':
                        results['commands'].append(result)
                        
        return results
        
    def _translate_statement(self, statement: Statement) -> Optional[Dict[str, Any]]:
        """Translate a single statement."""
        if isinstance(statement, FactStatement):
            return self._translate_fact(statement)
        elif isinstance(statement, RuleStatement):
            return self._translate_rule(statement)
        elif isinstance(statement, QueryStatement):
            return self._translate_query(statement)
        elif isinstance(statement, CommandStatement):
            return self._translate_command(statement)
        else:
            return None
            
    def _translate_fact(self, fact: FactStatement) -> Dict[str, Any]:
        """Translate fact statement to Egdol fact."""
        if isinstance(fact, IsFact):
            subject_term = self._translate_subject(fact.subject)
            predicate_term = self._translate_predicate(fact.predicate)
            
            # Create fact: predicate(subject)
            egdol_fact = Fact(Term(predicate_term.name, [subject_term]))
            self.engine.add_fact(egdol_fact.term)
            
            return {
                'type': 'fact',
                'fact': egdol_fact,
                'description': f"{fact.subject} is {fact.predicate}"
            }
            
        elif isinstance(fact, HasFact):
            subject_term = self._translate_subject(fact.subject)
            property_term = self._translate_predicate(fact.property)
            
            # Create fact: has(subject, property)
            egdol_fact = Fact(Term('has', [subject_term, property_term]))
            self.engine.add_fact(egdol_fact.term)
            
            return {
                'type': 'fact',
                'fact': egdol_fact,
                'description': f"{fact.subject} has {fact.property}"
            }
            
    def _translate_rule(self, rule: RuleStatement) -> Dict[str, Any]:
        """Translate rule statement to Egdol rule."""
        if isinstance(rule, IfThenRule):
            head = self._translate_fact_to_term(rule.conclusion)
            body = self._translate_condition_to_terms(rule.condition)
            
            egdol_rule = Rule(head, body)
            self.engine.add_rule(egdol_rule)
            
            return {
                'type': 'rule',
                'rule': egdol_rule,
                'description': f"if {rule.condition} then {rule.conclusion}"
            }
            
        elif isinstance(rule, WhenThenRule):
            head = self._translate_fact_to_term(rule.conclusion)
            body = self._translate_condition_to_terms(rule.condition)
            
            egdol_rule = Rule(head, body)
            self.engine.add_rule(egdol_rule)
            
            return {
                'type': 'rule',
                'rule': egdol_rule,
                'description': f"when {rule.condition} then {rule.conclusion}"
            }
            
    def _translate_query(self, query: QueryStatement) -> Dict[str, Any]:
        """Translate query statement to Egdol query."""
        if isinstance(query, WhoQuery):
            # Who is X -> find all subjects that are X
            predicate_term = self._translate_predicate(query.predicate)
            goal = Term(predicate_term.name, [Variable('X')])
            
            results = list(self.interpreter.query(goal))
            
            return {
                'type': 'query',
                'query': goal,
                'results': results,
                'description': f"who is {query.predicate}"
            }
            
        elif isinstance(query, WhatQuery):
            # What is X -> find all predicates of X
            subject_term = self._translate_subject(query.subject)
            goal = Term('is', [subject_term, Variable('Y')])
            
            results = list(self.interpreter.query(goal))
            
            return {
                'type': 'query',
                'query': goal,
                'results': results,
                'description': f"what is {query.subject}"
            }
            
        elif isinstance(query, DoesQuery):
            # Does X have Y -> check if has(X, Y) is true
            subject_term = self._translate_subject(query.subject)
            property_term = self._translate_predicate(query.property)
            goal = Term('has', [subject_term, property_term])
            
            results = list(self.interpreter.query(goal))
            
            return {
                'type': 'query',
                'query': goal,
                'results': results,
                'description': f"does {query.subject} have {query.property}"
            }
            
        elif isinstance(query, ShowQuery):
            # Show me all X -> find all instances of X
            predicate_term = self._translate_predicate(query.predicate)
            goal = Term(predicate_term.name, [Variable('X')])
            
            results = list(self.interpreter.query(goal))
            
            return {
                'type': 'query',
                'query': goal,
                'results': results,
                'description': f"show me all {query.predicate}"
            }
            
        elif isinstance(query, IsQuery):
            # Is X Y -> check if X is Y
            subject_term = self._translate_subject(query.subject)
            predicate_term = self._translate_predicate(query.predicate)
            goal = Term(predicate_term.name, [subject_term])
            
            results = list(self.interpreter.query(goal))
            
            return {
                'type': 'query',
                'query': goal,
                'results': results,
                'description': f"is {query.subject} {query.predicate}"
            }
            
        else:
            # Generic query
            subject_term = self._translate_subject(query.subject)
            goal = Term('is', [subject_term, Variable('Y')])
            
            results = list(self.interpreter.query(goal))
            
            return {
                'type': 'query',
                'query': goal,
                'results': results,
                'description': f"query about {query.subject}"
            }
            
    def _translate_command(self, command: CommandStatement) -> Dict[str, Any]:
        """Translate command statement."""
        return {
            'type': 'command',
            'command': command.command,
            'args': command.args,
            'description': f":{command.command} {' '.join(map(str, command.args))}"
        }
        
    def _translate_subject(self, subject: Subject) -> Term:
        """Translate subject to Egdol term."""
        if isinstance(subject, ProperNounSubject):
            return Constant(subject.name)
        elif isinstance(subject, VariableSubject):
            return Variable(subject.name)
        elif isinstance(subject, PronounSubject):
            # Resolve pronoun
            resolved = subject.resolved or self._resolve_pronoun(subject.pronoun)
            if resolved:
                return Constant(resolved)
            else:
                return Variable('X')  # Default variable
        else:
            return Variable('X')
            
    def _translate_predicate(self, predicate: Predicate) -> Term:
        """Translate predicate to Egdol term."""
        if isinstance(predicate, CommonNounPredicate):
            return Term(predicate.noun, [])
        elif isinstance(predicate, ProperNounPredicate):
            return Constant(predicate.name)
        elif isinstance(predicate, AdjectivePredicate):
            return Term(predicate.adjective, [])
        else:
            return Term('unknown', [])
            
    def _translate_fact_to_term(self, fact: FactStatement) -> Term:
        """Translate fact to Egdol term."""
        if isinstance(fact, IsFact):
            subject_term = self._translate_subject(fact.subject)
            predicate_term = self._translate_predicate(fact.predicate)
            return Term(predicate_term.name, [subject_term])
        elif isinstance(fact, HasFact):
            subject_term = self._translate_subject(fact.subject)
            property_term = self._translate_predicate(fact.property)
            return Term('has', [subject_term, property_term])
        else:
            return Term('unknown', [])
            
    def _translate_condition_to_terms(self, condition: Condition) -> List[Term]:
        """Translate condition to list of Egdol terms."""
        if isinstance(condition, SimpleCondition):
            fact_term = self._translate_fact_to_term(condition.fact)
            return [fact_term]
        elif isinstance(condition, NotCondition):
            # Negation as failure - create a special term
            fact_term = self._translate_fact_to_term(condition.fact)
            return [Term('not', [fact_term])]
        elif isinstance(condition, AndCondition):
            left_terms = self._translate_condition_to_terms(condition.left)
            right_terms = self._translate_condition_to_terms(condition.right)
            return left_terms + right_terms
        elif isinstance(condition, OrCondition):
            # For OR conditions, we need to create separate rules
            # This is a simplification - in practice, you'd need more complex handling
            left_terms = self._translate_condition_to_terms(condition.left)
            return left_terms  # Simplified - just take left side
        else:
            return []
            
    def _resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve pronoun from context memory."""
        if not self.context_memory:
            return None
            
        # Look through context stack (most recent first)
        for context in reversed(self.context_memory):
            if 'subject' in context:
                return context['subject']
                
        return None
        
    def _update_context(self, key: str, value: str):
        """Update context memory."""
        if not self.context_memory:
            self.context_memory.append({})
        self.context_memory[-1][key] = value


class DSLExecutor:
    """Executes DSL statements and returns results."""
    
    def __init__(self, engine: RulesEngine):
        self.engine = engine
        self.translator = DSLTranslator(engine)
        
    def execute(self, dsl_text: str) -> Dict[str, Any]:
        """Execute DSL text and return results."""
        from .tokenizer import DSLTokenizer
        from .parser import DSLParser
        
        # Tokenize
        tokenizer = DSLTokenizer()
        tokens = tokenizer.tokenize(dsl_text)
        
        # Parse
        parser = DSLParser()
        ast = parser.parse(tokens)
        
        # Translate and execute
        results = self.translator.translate_program(ast)
        
        return results
        
    def execute_fact(self, fact_text: str) -> Dict[str, Any]:
        """Execute a single fact statement."""
        return self.execute(fact_text)
        
    def execute_rule(self, rule_text: str) -> Dict[str, Any]:
        """Execute a single rule statement."""
        return self.execute(rule_text)
        
    def execute_query(self, query_text: str) -> Dict[str, Any]:
        """Execute a single query statement."""
        return self.execute(query_text)
        
    def execute_command(self, command_text: str) -> Dict[str, Any]:
        """Execute a single command."""
        return self.execute(command_text)
