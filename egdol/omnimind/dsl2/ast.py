"""
DSL 2.0 Abstract Syntax Tree
Advanced AST for nested logical statements and mathematical expressions.
"""

from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from enum import Enum, auto


class NodeType(Enum):
    """Types of AST nodes."""
    EXPRESSION = auto()
    STATEMENT = auto()
    RULE = auto()
    FACT = auto()
    QUERY = auto()
    CONDITION = auto()
    ACTION = auto()
    LITERAL = auto()
    VARIABLE = auto()
    OPERATOR = auto()
    FUNCTION = auto()


class Node(ABC):
    """Base class for all AST nodes."""
    
    def __init__(self, node_type: NodeType, line: int = 1, column: int = 1):
        self.node_type = node_type
        self.line = line
        self.column = column
        
    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor for traversal."""
        pass
        
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        pass
        
    @abstractmethod
    def __repr__(self):
        pass


class Expression(Node):
    """Base class for expressions."""
    
    def __init__(self, node_type: NodeType = NodeType.EXPRESSION, line: int = 1, column: int = 1):
        super().__init__(node_type, line, column)


class Literal(Expression):
    """Literal value (string, number, boolean)."""
    
    def __init__(self, value: Any, literal_type: str, line: int = 1, column: int = 1):
        super().__init__(NodeType.LITERAL, line, column)
        self.value = value
        self.literal_type = literal_type
        
    def accept(self, visitor):
        return visitor.visit_literal(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'literal',
            'value': self.value,
            'literal_type': self.literal_type,
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"Literal({self.value}, {self.literal_type})"


class Variable(Expression):
    """Variable reference."""
    
    def __init__(self, name: str, line: int = 1, column: int = 1):
        super().__init__(NodeType.VARIABLE, line, column)
        self.name = name
        
    def accept(self, visitor):
        return visitor.visit_variable(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'variable',
            'name': self.name,
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"Variable({self.name})"


class BinaryExpression(Expression):
    """Binary operation expression."""
    
    def __init__(self, left: Expression, operator: str, right: Expression, 
                 line: int = 1, column: int = 1):
        super().__init__(NodeType.EXPRESSION, line, column)
        self.left = left
        self.operator = operator
        self.right = right
        
    def accept(self, visitor):
        return visitor.visit_binary_expression(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'binary_expression',
            'left': self.left.to_dict(),
            'operator': self.operator,
            'right': self.right.to_dict(),
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"BinaryExpression({self.left}, {self.operator}, {self.right})"


class UnaryExpression(Expression):
    """Unary operation expression."""
    
    def __init__(self, operator: str, operand: Expression, line: int = 1, column: int = 1):
        super().__init__(NodeType.EXPRESSION, line, column)
        self.operator = operator
        self.operand = operand
        
    def accept(self, visitor):
        return visitor.visit_unary_expression(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'unary_expression',
            'operator': self.operator,
            'operand': self.operand.to_dict(),
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"UnaryExpression({self.operator}, {self.operand})"


class FunctionCall(Expression):
    """Function call expression."""
    
    def __init__(self, name: str, arguments: List[Expression], line: int = 1, column: int = 1):
        super().__init__(NodeType.FUNCTION, line, column)
        self.name = name
        self.arguments = arguments
        
    def accept(self, visitor):
        return visitor.visit_function_call(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'function_call',
            'name': self.name,
            'arguments': [arg.to_dict() for arg in self.arguments],
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"FunctionCall({self.name}, {self.arguments})"


class Statement(Node):
    """Base class for statements."""
    
    def __init__(self, node_type: NodeType = NodeType.STATEMENT, line: int = 1, column: int = 1):
        super().__init__(node_type, line, column)


class Fact(Statement):
    """Fact statement."""
    
    def __init__(self, subject: Expression, predicate: Expression, 
                 line: int = 1, column: int = 1):
        super().__init__(NodeType.FACT, line, column)
        self.subject = subject
        self.predicate = predicate
        
    def accept(self, visitor):
        return visitor.visit_fact(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'fact',
            'subject': self.subject.to_dict(),
            'predicate': self.predicate.to_dict(),
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"Fact({self.subject}, {self.predicate})"


class Rule(Statement):
    """Rule statement with conditions and actions."""
    
    def __init__(self, conditions: List[Expression], actions: List[Expression],
                 line: int = 1, column: int = 1):
        super().__init__(NodeType.RULE, line, column)
        self.conditions = conditions
        self.actions = actions
        
    def accept(self, visitor):
        return visitor.visit_rule(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'rule',
            'conditions': [cond.to_dict() for cond in self.conditions],
            'actions': [action.to_dict() for action in self.actions],
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"Rule({self.conditions}, {self.actions})"


class Query(Statement):
    """Query statement."""
    
    def __init__(self, expression: Expression, line: int = 1, column: int = 1):
        super().__init__(NodeType.QUERY, line, column)
        self.expression = expression
        
    def accept(self, visitor):
        return visitor.visit_query(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'query',
            'expression': self.expression.to_dict(),
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"Query({self.expression})"


class ConditionalStatement(Statement):
    """Conditional statement (if-then-else)."""
    
    def __init__(self, condition: Expression, then_branch: List[Statement],
                 else_branch: Optional[List[Statement]] = None,
                 line: int = 1, column: int = 1):
        super().__init__(NodeType.STATEMENT, line, column)
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch or []
        
    def accept(self, visitor):
        return visitor.visit_conditional_statement(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'conditional_statement',
            'condition': self.condition.to_dict(),
            'then_branch': [stmt.to_dict() for stmt in self.then_branch],
            'else_branch': [stmt.to_dict() for stmt in self.else_branch],
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"ConditionalStatement({self.condition}, {self.then_branch}, {self.else_branch})"


class Block(Statement):
    """Block of statements."""
    
    def __init__(self, statements: List[Statement], line: int = 1, column: int = 1):
        super().__init__(NodeType.STATEMENT, line, column)
        self.statements = statements
        
    def accept(self, visitor):
        return visitor.visit_block(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'block',
            'statements': [stmt.to_dict() for stmt in self.statements],
            'line': self.line,
            'column': self.column
        }
        
    def __repr__(self):
        return f"Block({self.statements})"


class DSL2AST:
    """DSL 2.0 Abstract Syntax Tree."""
    
    def __init__(self, statements: List[Statement]):
        self.statements = statements
        
    def accept(self, visitor):
        return visitor.visit_ast(self)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'ast',
            'statements': [stmt.to_dict() for stmt in self.statements]
        }
        
    def __repr__(self):
        return f"DSL2AST({self.statements})"
