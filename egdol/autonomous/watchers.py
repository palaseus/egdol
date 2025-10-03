"""
Watchers for Egdol.
Provides production rule system with condition-action pairs.
"""

import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto


class WatcherStatus(Enum):
    """Status of a watcher."""
    ACTIVE = auto()
    INACTIVE = auto()
    TRIGGERED = auto()
    ERROR = auto()


@dataclass
class Watcher:
    """A watcher that monitors conditions and triggers actions."""
    name: str
    condition: Callable
    action: Callable
    priority: int = 0
    enabled: bool = True
    last_triggered: float = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_triggered is None:
            self.last_triggered = 0.0
            
    def check_condition(self) -> bool:
        """Check if the condition is met."""
        try:
            return self.condition()
        except Exception as e:
            print(f"Error checking condition for watcher '{self.name}': {e}")
            return False
            
    def trigger_action(self) -> bool:
        """Trigger the action."""
        try:
            self.action()
            self.last_triggered = time.time()
            self.trigger_count += 1
            return True
        except Exception as e:
            print(f"Error triggering action for watcher '{self.name}': {e}")
            return False


class WatcherManager:
    """Manages watchers and their execution."""
    
    def __init__(self):
        self.watchers: Dict[str, Watcher] = {}
        self.running = False
        self.check_interval = 1.0  # Check every second
        
    def add_watcher(self, name: str, condition: Callable, action: Callable,
                   priority: int = 0, metadata: Dict[str, Any] = None) -> bool:
        """Add a watcher."""
        if name in self.watchers:
            return False
            
        watcher = Watcher(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.watchers[name] = watcher
        return True
        
    def remove_watcher(self, name: str) -> bool:
        """Remove a watcher."""
        if name in self.watchers:
            del self.watchers[name]
            return True
        return False
        
    def enable_watcher(self, name: str) -> bool:
        """Enable a watcher."""
        if name in self.watchers:
            self.watchers[name].enabled = True
            return True
        return False
        
    def disable_watcher(self, name: str) -> bool:
        """Disable a watcher."""
        if name in self.watchers:
            self.watchers[name].enabled = False
            return True
        return False
        
    def check_all_watchers(self) -> List[str]:
        """Check all watchers and return triggered ones."""
        triggered = []
        
        # Sort by priority (higher priority first)
        sorted_watchers = sorted(
            self.watchers.values(),
            key=lambda w: w.priority,
            reverse=True
        )
        
        for watcher in sorted_watchers:
            if not watcher.enabled:
                continue
                
            if watcher.check_condition():
                if watcher.trigger_action():
                    triggered.append(watcher.name)
                    
        return triggered
        
    def get_watcher_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific watcher."""
        if name not in self.watchers:
            return None
            
        watcher = self.watchers[name]
        return {
            'name': watcher.name,
            'enabled': watcher.enabled,
            'priority': watcher.priority,
            'last_triggered': watcher.last_triggered,
            'trigger_count': watcher.trigger_count,
            'metadata': watcher.metadata
        }
        
    def get_all_watchers(self) -> List[Dict[str, Any]]:
        """Get status of all watchers."""
        return [self.get_watcher_status(name) for name in self.watchers.keys()]
        
    def get_watcher_stats(self) -> Dict[str, Any]:
        """Get watcher statistics."""
        total_watchers = len(self.watchers)
        enabled_watchers = len([w for w in self.watchers.values() if w.enabled])
        
        # Count by priority
        priority_counts = {}
        for watcher in self.watchers.values():
            priority = watcher.priority
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
        # Total triggers
        total_triggers = sum(w.trigger_count for w in self.watchers.values())
        
        return {
            'total_watchers': total_watchers,
            'enabled_watchers': enabled_watchers,
            'priority_distribution': priority_counts,
            'total_triggers': total_triggers
        }
        
    def create_fact_watcher(self, name: str, predicate: str, action: Callable,
                          priority: int = 0) -> bool:
        """Create a watcher that triggers when a fact is added."""
        def condition():
            # This would need to be implemented based on the memory store
            # For now, it's a placeholder
            return False
            
        return self.add_watcher(name, condition, action, priority, {
            'type': 'fact_watcher',
            'predicate': predicate
        })
        
    def create_rule_watcher(self, name: str, rule_pattern: str, action: Callable,
                          priority: int = 0) -> bool:
        """Create a watcher that triggers when a rule is added."""
        def condition():
            # This would need to be implemented based on the memory store
            # For now, it's a placeholder
            return False
            
        return self.add_watcher(name, condition, action, priority, {
            'type': 'rule_watcher',
            'rule_pattern': rule_pattern
        })
        
    def create_time_watcher(self, name: str, interval: float, action: Callable,
                          priority: int = 0) -> bool:
        """Create a watcher that triggers at regular intervals."""
        last_triggered = time.time()
        
        def condition():
            nonlocal last_triggered
            current_time = time.time()
            if current_time - last_triggered >= interval:
                last_triggered = current_time
                return True
            return False
            
        return self.add_watcher(name, condition, action, priority, {
            'type': 'time_watcher',
            'interval': interval
        })
        
    def create_confidence_watcher(self, name: str, threshold: float, action: Callable,
                                priority: int = 0) -> bool:
        """Create a watcher that triggers when confidence drops below threshold."""
        def condition():
            # This would need to be implemented based on the memory store
            # For now, it's a placeholder
            return False
            
        return self.add_watcher(name, condition, action, priority, {
            'type': 'confidence_watcher',
            'threshold': threshold
        })
