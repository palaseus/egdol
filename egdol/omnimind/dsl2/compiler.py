"""
DSL 2.0 Compiler
Compiles DSL 2.0 AST to Egdol internal representation.
"""

from typing import Dict, Any, List, Optional, Union
from .ast import (
    DSL2AST, Node, Expression, Statement, Rule, Fact, Query,
    Literal, Variable, BinaryExpression, UnaryExpression, FunctionCall,
    ConditionalStatement, Block
)
# Import Egdol types
from ...lexer import Token as EgdolToken
from ...parser import Fact as EgdolFact, Rule as EgdolRule
from ...interpreter import Variable as EgdolVariable


class DSL2Compiler:
    """Compiles DSL 2.0 AST to Egdol internal representation."""
    
    def __init__(self, context_manager=None):
        self.context_manager = context_manager
        self.compilation_results: List[Dict[str, Any]] = []
        
    def compile(self, ast: DSL2AST) -> Dict[str, Any]:
        """Compile AST to Egdol representation."""
        results = {
            'facts': [],
            'rules': [],
            'queries': [],
            'errors': [],
            'warnings': []
        }
        
        for statement in ast.statements:
            try:
                compiled = self.compile_statement(statement)
                if compiled:
                    if isinstance(statement, Fact):
                        results['facts'].append(compiled)
                    elif isinstance(statement, Rule):
                        results['rules'].append(compiled)
                    elif isinstance(statement, Query):
                        results['queries'].append(compiled)
            except Exception as e:
                results['errors'].append({
                    'statement': str(statement),
                    'error': str(e),
                    'line': getattr(statement, 'line', 1),
                    'column': getattr(statement, 'column', 1)
                })
                
        self.compilation_results.append(results)
        return results
        
    def compile_statement(self, statement: Statement) -> Optional[Dict[str, Any]]:
        """Compile a single statement."""
        if isinstance(statement, Fact):
            return self.compile_fact(statement)
        elif isinstance(statement, Rule):
            return self.compile_rule(statement)
        elif isinstance(statement, Query):
            return self.compile_query(statement)
        elif isinstance(statement, ConditionalStatement):
            return self.compile_conditional_statement(statement)
        elif isinstance(statement, Block):
            return self.compile_block(statement)
        else:
            return None
            
    def compile_fact(self, fact: Fact) -> Dict[str, Any]:
        """Compile a fact statement."""
        subject = self.compile_expression(fact.subject)
        predicate = self.compile_expression(fact.predicate)
        
        # Create Egdol fact
        egdol_fact = EgdolFact(subject, predicate)
        
        return {
            'type': 'fact',
            'egdol_fact': egdol_fact,
            'subject': subject,
            'predicate': predicate,
            'line': fact.line,
            'column': fact.column
        }
        
    def compile_rule(self, rule: Rule) -> Dict[str, Any]:
        """Compile a rule statement."""
        conditions = []
        actions = []
        
        for condition in rule.conditions:
            compiled_condition = self.compile_expression(condition)
            conditions.append(compiled_condition)
            
        for action in rule.actions:
            compiled_action = self.compile_expression(action)
            actions.append(compiled_action)
            
        # Create Egdol rule
        egdol_rule = EgdolRule(conditions, actions)
        
        return {
            'type': 'rule',
            'egdol_rule': egdol_rule,
            'conditions': conditions,
            'actions': actions,
            'line': rule.line,
            'column': rule.column
        }
        
    def compile_query(self, query: Query) -> Dict[str, Any]:
        """Compile a query statement."""
        expression = self.compile_expression(query.expression)
        
        return {
            'type': 'query',
            'expression': expression,
            'line': query.line,
            'column': query.column
        }
        
    def compile_conditional_statement(self, conditional: ConditionalStatement) -> Dict[str, Any]:
        """Compile a conditional statement."""
        condition = self.compile_expression(conditional.condition)
        
        then_compiled = []
        for stmt in conditional.then_branch:
            compiled = self.compile_statement(stmt)
            if compiled:
                then_compiled.append(compiled)
                
        else_compiled = []
        for stmt in conditional.else_branch:
            compiled = self.compile_statement(stmt)
            if compiled:
                else_compiled.append(compiled)
                
        return {
            'type': 'conditional',
            'condition': condition,
            'then_branch': then_compiled,
            'else_branch': else_compiled,
            'line': conditional.line,
            'column': conditional.column
        }
        
    def compile_block(self, block: Block) -> Dict[str, Any]:
        """Compile a block statement."""
        compiled_statements = []
        
        for stmt in block.statements:
            compiled = self.compile_statement(stmt)
            if compiled:
                compiled_statements.append(compiled)
                
        return {
            'type': 'block',
            'statements': compiled_statements,
            'line': block.line,
            'column': block.column
        }
        
    def compile_expression(self, expression: Expression) -> Any:
        """Compile an expression."""
        if isinstance(expression, Literal):
            return self.compile_literal(expression)
        elif isinstance(expression, Variable):
            return self.compile_variable(expression)
        elif isinstance(expression, BinaryExpression):
            return self.compile_binary_expression(expression)
        elif isinstance(expression, UnaryExpression):
            return self.compile_unary_expression(expression)
        elif isinstance(expression, FunctionCall):
            return self.compile_function_call(expression)
        else:
            return None
            
    def compile_literal(self, literal: Literal) -> Any:
        """Compile a literal expression."""
        if literal.literal_type == "string":
            return literal.value
        elif literal.literal_type == "number":
            try:
                if '.' in literal.value:
                    return float(literal.value)
                else:
                    return int(literal.value)
            except ValueError:
                return literal.value
        elif literal.literal_type == "boolean":
            return literal.value
        else:
            return literal.value
            
    def compile_variable(self, variable: Variable) -> EgdolVariable:
        """Compile a variable expression."""
        # Check context for variable resolution
        if self.context_manager:
            resolved = self.context_manager.resolve_variable(variable.name)
            if resolved is not None:
                return EgdolVariable(str(resolved))
                
        return EgdolVariable(variable.name)
        
    def compile_binary_expression(self, binary: BinaryExpression) -> Dict[str, Any]:
        """Compile a binary expression."""
        left = self.compile_expression(binary.left)
        right = self.compile_expression(binary.right)
        
        return {
            'type': 'binary_expression',
            'left': left,
            'operator': binary.operator,
            'right': right,
            'line': binary.line,
            'column': binary.column
        }
        
    def compile_unary_expression(self, unary: UnaryExpression) -> Dict[str, Any]:
        """Compile a unary expression."""
        operand = self.compile_expression(unary.operand)
        
        return {
            'type': 'unary_expression',
            'operator': unary.operator,
            'operand': operand,
            'line': unary.line,
            'column': unary.column
        }
        
    def compile_function_call(self, func_call: FunctionCall) -> Dict[str, Any]:
        """Compile a function call expression."""
        arguments = []
        for arg in func_call.arguments:
            compiled_arg = self.compile_expression(arg)
            arguments.append(compiled_arg)
            
        return {
            'type': 'function_call',
            'name': func_call.name,
            'arguments': arguments,
            'line': func_call.line,
            'column': func_call.column
        }
        
    def get_compilation_summary(self) -> Dict[str, Any]:
        """Get summary of compilation results."""
        if not self.compilation_results:
            return {'total_compilations': 0}
            
        total_facts = sum(len(result['facts']) for result in self.compilation_results)
        total_rules = sum(len(result['rules']) for result in self.compilation_results)
        total_queries = sum(len(result['queries']) for result in self.compilation_results)
        total_errors = sum(len(result['errors']) for result in self.compilation_results)
        total_warnings = sum(len(result['warnings']) for result in self.compilation_results)
        
        return {
            'total_compilations': len(self.compilation_results),
            'total_facts': total_facts,
            'total_rules': total_rules,
            'total_queries': total_queries,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'success_rate': (total_facts + total_rules + total_queries) / 
                          (total_facts + total_rules + total_queries + total_errors) if 
                          (total_facts + total_rules + total_queries + total_errors) > 0 else 0
        }
        
    def clear_compilation_history(self):
        """Clear compilation history."""
        self.compilation_results.clear()
