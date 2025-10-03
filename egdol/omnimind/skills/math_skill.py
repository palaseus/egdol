"""
Math Skill for OmniMind
Handles mathematical calculations and expressions.
"""

import re
import ast
import operator
from typing import Dict, Any, Optional
from .base import BaseSkill


class MathSkill(BaseSkill):
    """Handles mathematical calculations."""
    
    def __init__(self):
        super().__init__()
        self.description = "Handles mathematical calculations and expressions"
        self.capabilities = [
            "arithmetic operations",
            "algebraic expressions", 
            "mathematical functions",
            "unit conversions"
        ]
        
        # Safe operators for eval
        self.safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        """Check if this is a math-related input."""
        user_input_lower = user_input.lower()
        
        # Math keywords
        math_keywords = [
            'calculate', 'compute', 'solve', 'math', 'arithmetic',
            'add', 'subtract', 'multiply', 'divide', 'plus', 'minus',
            'times', 'equals', 'sum', 'total', 'average', 'mean'
        ]
        
        # Math operators
        math_operators = ['+', '-', '*', '/', '=', '^', '**', '%']
        
        # Check for math keywords
        if any(keyword in user_input_lower for keyword in math_keywords):
            return True
            
        # Check for math operators
        if any(op in user_input for op in math_operators):
            return True
            
        # Check for numbers and basic math patterns
        if re.search(r'\d+[\+\-\*\/\=]\d+', user_input):
            return True
            
        return False
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mathematical input."""
        try:
            # Extract mathematical expression
            expression = self._extract_expression(user_input)
            if not expression:
                return {
                    'content': "I couldn't find a mathematical expression to evaluate.",
                    'reasoning': ['No mathematical expression detected'],
                    'metadata': {'skill': 'math', 'success': False}
                }
                
            # Evaluate expression safely
            result = self._safe_eval(expression)
            
            if result is not None:
                content = f"{expression} = {result}"
                reasoning = [f"Evaluated expression: {expression}", f"Result: {result}"]
            else:
                content = f"I couldn't evaluate the expression: {expression}"
                reasoning = [f"Failed to evaluate: {expression}"]
                
            return {
                'content': content,
                'reasoning': reasoning,
                'metadata': {'skill': 'math', 'expression': expression, 'result': result}
            }
            
        except Exception as e:
            return {
                'content': f"I encountered an error with the mathematical expression: {str(e)}",
                'reasoning': [f"Error: {str(e)}"],
                'metadata': {'skill': 'math', 'error': str(e)}
            }
            
    def _extract_expression(self, user_input: str) -> Optional[str]:
        """Extract mathematical expression from user input."""
        # Remove common words
        cleaned = re.sub(r'\b(calculate|compute|solve|what is|equals?)\b', '', user_input, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        
        # Look for mathematical patterns
        patterns = [
            r'[\d\+\-\*\/\^\(\)\.\s]+',  # Basic math expression
            r'\d+\s*[\+\-\*\/]\s*\d+',   # Simple arithmetic
            r'[\d\s\+\-\*\/\^\(\)\.]+'   # Extended expression
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                return match.group().strip()
                
        return None
        
    def _safe_eval(self, expression: str):
        """Safely evaluate a mathematical expression."""
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            
            # Check if it's safe (only contains numbers and operators)
            if self._is_safe_expression(tree):
                return eval(expression, {"__builtins__": {}}, self.safe_operators)
            else:
                return None
                
        except Exception:
            return None
            
    def _is_safe_expression(self, tree):
        """Check if an expression is safe to evaluate."""
        for node in ast.walk(tree):
            # Only allow specific node types
            if isinstance(node, (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare)):
                continue
            elif isinstance(node, ast.Name):
                # No variable names allowed
                return False
            elif isinstance(node, ast.Call):
                # No function calls allowed
                return False
            else:
                return False
        return True
        
    def _calculate_basic(self, expression: str):
        """Calculate basic arithmetic expressions."""
        try:
            # Replace common symbols
            expression = expression.replace('^', '**')
            expression = expression.replace('×', '*')
            expression = expression.replace('÷', '/')
            
            # Evaluate safely
            result = eval(expression, {"__builtins__": {}}, {})
            return result
        except Exception:
            return None
