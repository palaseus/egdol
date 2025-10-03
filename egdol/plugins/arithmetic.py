"""
Arithmetic plugin for Egdol DSL.
Provides arithmetic operations and comparisons.
"""

from typing import List, Any
from . import Plugin


class ArithmeticPlugin(Plugin):
    """Plugin providing arithmetic operations."""
    
    def __init__(self):
        super().__init__("arithmetic")
        self._register_predicates()
        self._register_functions()
        
    def _register_predicates(self):
        """Register arithmetic predicates."""
        self.register_predicate("equals", self._equals)
        self.register_predicate("greater_than", self._greater_than)
        self.register_predicate("less_than", self._less_than)
        self.register_predicate("greater_equal", self._greater_equal)
        self.register_predicate("less_equal", self._less_equal)
        self.register_predicate("not_equals", self._not_equals)
        
    def _register_functions(self):
        """Register arithmetic functions."""
        self.register_function("add", self._add)
        self.register_function("subtract", self._subtract)
        self.register_function("multiply", self._multiply)
        self.register_function("divide", self._divide)
        self.register_function("modulo", self._modulo)
        self.register_function("power", self._power)
        self.register_function("abs", self._abs)
        self.register_function("min", self._min)
        self.register_function("max", self._max)
        
    def _equals(self, a: Any, b: Any) -> bool:
        """Check if two values are equal."""
        try:
            return float(a) == float(b)
        except (ValueError, TypeError):
            return str(a) == str(b)
            
    def _greater_than(self, a: Any, b: Any) -> bool:
        """Check if a > b."""
        try:
            return float(a) > float(b)
        except (ValueError, TypeError):
            return False
            
    def _less_than(self, a: Any, b: Any) -> bool:
        """Check if a < b."""
        try:
            return float(a) < float(b)
        except (ValueError, TypeError):
            return False
            
    def _greater_equal(self, a: Any, b: Any) -> bool:
        """Check if a >= b."""
        try:
            return float(a) >= float(b)
        except (ValueError, TypeError):
            return False
            
    def _less_equal(self, a: Any, b: Any) -> bool:
        """Check if a <= b."""
        try:
            return float(a) <= float(b)
        except (ValueError, TypeError):
            return False
            
    def _not_equals(self, a: Any, b: Any) -> bool:
        """Check if two values are not equal."""
        return not self._equals(a, b)
        
    def _add(self, *args: Any) -> float:
        """Add numbers."""
        try:
            return sum(float(arg) for arg in args)
        except (ValueError, TypeError):
            return 0.0
            
    def _subtract(self, a: Any, b: Any) -> float:
        """Subtract b from a."""
        try:
            return float(a) - float(b)
        except (ValueError, TypeError):
            return 0.0
            
    def _multiply(self, *args: Any) -> float:
        """Multiply numbers."""
        try:
            result = 1.0
            for arg in args:
                result *= float(arg)
            return result
        except (ValueError, TypeError):
            return 0.0
            
    def _divide(self, a: Any, b: Any) -> float:
        """Divide a by b."""
        try:
            return float(a) / float(b)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
            
    def _modulo(self, a: Any, b: Any) -> float:
        """Modulo operation."""
        try:
            return float(a) % float(b)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
            
    def _power(self, a: Any, b: Any) -> float:
        """Power operation."""
        try:
            return float(a) ** float(b)
        except (ValueError, TypeError):
            return 0.0
            
    def _abs(self, a: Any) -> float:
        """Absolute value."""
        try:
            return abs(float(a))
        except (ValueError, TypeError):
            return 0.0
            
    def _min(self, *args: Any) -> float:
        """Minimum value."""
        try:
            return min(float(arg) for arg in args)
        except (ValueError, TypeError):
            return 0.0
            
    def _max(self, *args: Any) -> float:
        """Maximum value."""
        try:
            return max(float(arg) for arg in args)
        except (ValueError, TypeError):
            return 0.0
