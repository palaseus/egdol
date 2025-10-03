"""
Simplified query optimizer for egdol.
Basic optimization without complex type dependencies.
"""

from typing import Dict, List, Set, Any
from .parser import Term, Variable
from .rules_engine import RulesEngine


class SimpleQueryOptimizer:
    """Simplified query optimizer."""
    
    def __init__(self, engine: RulesEngine):
        self.engine = engine
        
    def optimize_query(self, goal: Term):
        """Optimize a query (simplified implementation)."""
        return SimpleOptimizedQuery(goal, self.engine)
        
    def get_optimization_suggestions(self, goal: Term) -> List[str]:
        """Get optimization suggestions for a query."""
        suggestions = []
        
        # Check if goal has variables
        has_variables = self._has_variables(goal)
        if has_variables:
            suggestions.append("Consider adding constraints to reduce search space")
            
        # Check if goal is complex
        complexity = self._calculate_complexity(goal)
        if complexity > 5:
            suggestions.append("Consider breaking down complex queries into simpler parts")
            
        return suggestions
        
    def _has_variables(self, goal: Term) -> bool:
        """Check if goal has variables."""
        for arg in goal.args:
            if isinstance(arg, Variable):
                return True
            elif isinstance(arg, Term):
                if self._has_variables(arg):
                    return True
        return False
        
    def _calculate_complexity(self, goal: Term) -> int:
        """Calculate complexity of goal."""
        complexity = 1
        for arg in goal.args:
            if isinstance(arg, Term):
                complexity += self._calculate_complexity(arg)
        return complexity


class SimpleOptimizedQuery:
    """Simplified optimized query."""
    
    def __init__(self, goal: Term, engine: RulesEngine):
        self.goal = goal
        self.engine = engine
        
    def execute(self) -> List[Dict[str, Any]]:
        """Execute optimized query."""
        from .interpreter import Interpreter
        interp = Interpreter(self.engine)
        return list(interp.query(self.goal))
