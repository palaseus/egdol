"""
Plugin system for Egdol DSL.
Allows extending the DSL with custom predicates and built-ins.
"""

from typing import Dict, Any, Callable, List
from ..rules_engine import RulesEngine
from ..parser import Term, Variable, Constant


class Plugin:
    """Base class for DSL plugins."""
    
    def __init__(self, name: str):
        self.name = name
        self.predicates: Dict[str, Callable] = {}
        self.functions: Dict[str, Callable] = {}
        
    def register_predicate(self, name: str, func: Callable):
        """Register a custom predicate."""
        self.predicates[name] = func
        
    def register_function(self, name: str, func: Callable):
        """Register a custom function."""
        self.functions[name] = func
        
    def evaluate_predicate(self, name: str, args: List[Any]) -> bool:
        """Evaluate a custom predicate."""
        if name in self.predicates:
            return self.predicates[name](*args)
        return False
        
    def evaluate_function(self, name: str, args: List[Any]) -> Any:
        """Evaluate a custom function."""
        if name in self.functions:
            return self.functions[name](*args)
        return None


class PluginManager:
    """Manages plugins for the DSL."""
    
    def __init__(self, engine: RulesEngine):
        self.engine = engine
        self.plugins: Dict[str, Plugin] = {}
        
    def register_plugin(self, plugin: Plugin):
        """Register a plugin."""
        self.plugins[plugin.name] = plugin
        
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self.plugins.get(name)
        
    def evaluate_predicate(self, name: str, args: List[Any]) -> bool:
        """Evaluate a predicate across all plugins."""
        for plugin in self.plugins.values():
            if name in plugin.predicates:
                return plugin.evaluate_predicate(name, args)
        return False
        
    def evaluate_function(self, name: str, args: List[Any]) -> Any:
        """Evaluate a function across all plugins."""
        for plugin in self.plugins.values():
            if name in plugin.functions:
                return plugin.evaluate_function(name, args)
        return None
