"""
Actions for Egdol.
Provides action system for autonomous behaviors.
"""

import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto


class ActionType(Enum):
    """Types of actions."""
    MEMORY = auto()
    REASONING = auto()
    COMMUNICATION = auto()
    MODIFICATION = auto()
    NOTIFICATION = auto()


@dataclass
class Action:
    """An action that can be executed."""
    name: str
    action_type: ActionType
    function: Callable
    parameters: Dict[str, Any] = None
    enabled: bool = True
    last_executed: float = None
    execution_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.metadata is None:
            self.metadata = {}
        if self.last_executed is None:
            self.last_executed = 0.0
            
    def execute(self, **kwargs) -> bool:
        """Execute the action."""
        try:
            # Merge parameters with kwargs
            params = {**self.parameters, **kwargs}
            result = self.function(**params)
            self.last_executed = time.time()
            self.execution_count += 1
            return result
        except Exception as e:
            print(f"Error executing action '{self.name}': {e}")
            return False


class ActionManager:
    """Manages actions and their execution."""
    
    def __init__(self):
        self.actions: Dict[str, Action] = {}
        
    def add_action(self, name: str, action_type: ActionType, function: Callable,
                  parameters: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> bool:
        """Add an action."""
        if name in self.actions:
            return False
            
        action = Action(
            name=name,
            action_type=action_type,
            function=function,
            parameters=parameters or {},
            metadata=metadata or {}
        )
        
        self.actions[name] = action
        return True
        
    def remove_action(self, name: str) -> bool:
        """Remove an action."""
        if name in self.actions:
            del self.actions[name]
            return True
        return False
        
    def execute_action(self, name: str, **kwargs) -> bool:
        """Execute an action."""
        if name not in self.actions:
            return False
            
        action = self.actions[name]
        if not action.enabled:
            return False
            
        return action.execute(**kwargs)
        
    def get_action_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific action."""
        if name not in self.actions:
            return None
            
        action = self.actions[name]
        return {
            'name': action.name,
            'type': action.action_type.name,
            'enabled': action.enabled,
            'last_executed': action.last_executed,
            'execution_count': action.execution_count,
            'metadata': action.metadata
        }
        
    def get_all_actions(self) -> List[Dict[str, Any]]:
        """Get status of all actions."""
        return [self.get_action_status(name) for name in self.actions.keys()]
        
    def get_action_stats(self) -> Dict[str, Any]:
        """Get action statistics."""
        total_actions = len(self.actions)
        enabled_actions = len([a for a in self.actions.values() if a.enabled])
        
        # Count by type
        type_counts = {}
        for action in self.actions.values():
            action_type = action.action_type.name
            type_counts[action_type] = type_counts.get(action_type, 0) + 1
            
        # Total executions
        total_executions = sum(a.execution_count for a in self.actions.values())
        
        return {
            'total_actions': total_actions,
            'enabled_actions': enabled_actions,
            'type_distribution': type_counts,
            'total_executions': total_executions
        }
        
    def create_memory_action(self, name: str, function: Callable,
                           parameters: Dict[str, Any] = None) -> bool:
        """Create a memory-related action."""
        return self.add_action(
            name=name,
            action_type=ActionType.MEMORY,
            function=function,
            parameters=parameters,
            metadata={'category': 'memory'}
        )
        
    def create_reasoning_action(self, name: str, function: Callable,
                              parameters: Dict[str, Any] = None) -> bool:
        """Create a reasoning-related action."""
        return self.add_action(
            name=name,
            action_type=ActionType.REASONING,
            function=function,
            parameters=parameters,
            metadata={'category': 'reasoning'}
        )
        
    def create_communication_action(self, name: str, function: Callable,
                                  parameters: Dict[str, Any] = None) -> bool:
        """Create a communication-related action."""
        return self.add_action(
            name=name,
            action_type=ActionType.COMMUNICATION,
            function=function,
            parameters=parameters,
            metadata={'category': 'communication'}
        )
        
    def create_modification_action(self, name: str, function: Callable,
                                 parameters: Dict[str, Any] = None) -> bool:
        """Create a modification-related action."""
        return self.add_action(
            name=name,
            action_type=ActionType.MODIFICATION,
            function=function,
            parameters=parameters,
            metadata={'category': 'modification'}
        )
        
    def create_notification_action(self, name: str, function: Callable,
                                parameters: Dict[str, Any] = None) -> bool:
        """Create a notification-related action."""
        return self.add_action(
            name=name,
            action_type=ActionType.NOTIFICATION,
            function=function,
            parameters=parameters,
            metadata={'category': 'notification'}
        )
