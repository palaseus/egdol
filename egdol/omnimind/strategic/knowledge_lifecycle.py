"""
Knowledge Lifecycle Manager for OmniMind
Manages knowledge lifecycle including detection, updates, merging, and retirement.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque
import statistics


class KnowledgeState(Enum):
    """States of knowledge in the lifecycle."""
    ACTIVE = auto()
    OBSOLETE = auto()
    CONFLICTING = auto()
    REDUNDANT = auto()
    OUTDATED = auto()
    UNUSED = auto()


class LifecycleAction(Enum):
    """Actions to take on knowledge."""
    UPDATE = auto()
    MERGE = auto()
    RETIRE = auto()
    CONFLICT_RESOLVE = auto()
    OPTIMIZE = auto()
    ARCHIVE = auto()


@dataclass
class KnowledgeItem:
    """A knowledge item in the lifecycle."""
    id: str
    content: str
    knowledge_type: str
    state: KnowledgeState
    created_at: float
    last_accessed: float
    usage_count: int = 0
    confidence_score: float = 1.0
    dependencies: List[str] = None
    conflicts: List[str] = None
    lifecycle_actions: List[LifecycleAction] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.conflicts is None:
            self.conflicts = []
        if self.lifecycle_actions is None:
            self.lifecycle_actions = []
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert item to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'knowledge_type': self.knowledge_type,
            'state': self.state.name,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'usage_count': self.usage_count,
            'confidence_score': self.confidence_score,
            'dependencies': self.dependencies,
            'conflicts': self.conflicts,
            'lifecycle_actions': [a.name for a in self.lifecycle_actions]
        }


class KnowledgeLifecycleManager:
    """Manages knowledge lifecycle including detection, updates, merging, and retirement."""
    
    def __init__(self, network, learning_system):
        self.network = network
        self.learning_system = learning_system
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.lifecycle_history: List[Dict[str, Any]] = []
        self.knowledge_patterns: Dict[str, Any] = {}
        self.retirement_criteria: Dict[str, float] = {
            'usage_threshold': 0.1,
            'age_threshold': 86400,  # 24 hours
            'confidence_threshold': 0.3
        }
        
    def analyze_knowledge_lifecycle(self) -> List[KnowledgeItem]:
        """Analyze knowledge lifecycle and identify items needing attention."""
        items = []
        
        # Analyze knowledge usage patterns
        usage_analysis = self._analyze_knowledge_usage()
        items.extend(usage_analysis)
        
        # Analyze knowledge conflicts
        conflict_analysis = self._analyze_knowledge_conflicts()
        items.extend(conflict_analysis)
        
        # Analyze redundant knowledge
        redundancy_analysis = self._analyze_knowledge_redundancy()
        items.extend(redundancy_analysis)
        
        # Analyze outdated knowledge
        outdated_analysis = self._analyze_outdated_knowledge()
        items.extend(outdated_analysis)
        
        # Store knowledge items
        for item in items:
            self.knowledge_items[item.id] = item
            
        # Log lifecycle analysis
        self._log_lifecycle_event('lifecycle_analyzed', {
            'total_items': len(items),
            'states': [item.state.name for item in items]
        })
        
        return items
        
    def _analyze_knowledge_usage(self) -> List[KnowledgeItem]:
        """Analyze knowledge usage patterns."""
        items = []
        
        # Analyze agent knowledge usage
        for agent in self.network.agents.values():
            if hasattr(agent, 'knowledge_graph'):
                for knowledge_id, knowledge_data in agent.knowledge_graph.items():
                    usage_count = knowledge_data.get('usage_count', 0)
                    last_accessed = knowledge_data.get('last_accessed', time.time())
                    age = time.time() - last_accessed
                    
                    # Check if knowledge is unused
                    if usage_count < self.retirement_criteria['usage_threshold']:
                        item = KnowledgeItem(
                            id=knowledge_id,
                            content=str(knowledge_data.get('content', '')),
                            knowledge_type=knowledge_data.get('type', 'unknown'),
                            state=KnowledgeState.UNUSED,
                            created_at=knowledge_data.get('created_at', time.time()),
                            last_accessed=last_accessed,
                            usage_count=usage_count,
                            lifecycle_actions=[LifecycleAction.RETIRE]
                        )
                        items.append(item)
                        
        return items
        
    def _analyze_knowledge_conflicts(self) -> List[KnowledgeItem]:
        """Analyze knowledge conflicts."""
        items = []
        
        # This would need to be implemented based on actual knowledge conflict detection
        # For now, return empty list
        return items
        
    def _analyze_knowledge_redundancy(self) -> List[KnowledgeItem]:
        """Analyze redundant knowledge."""
        items = []
        
        # Analyze for redundant knowledge across agents
        all_knowledge = {}
        for agent in self.network.agents.values():
            if hasattr(agent, 'knowledge_graph'):
                for knowledge_id, knowledge_data in agent.knowledge_graph.items():
                    content = str(knowledge_data.get('content', ''))
                    if content in all_knowledge:
                        # Found redundant knowledge
                        item = KnowledgeItem(
                            id=knowledge_id,
                            content=content,
                            knowledge_type=knowledge_data.get('type', 'unknown'),
                            state=KnowledgeState.REDUNDANT,
                            created_at=knowledge_data.get('created_at', time.time()),
                            last_accessed=knowledge_data.get('last_accessed', time.time()),
                            usage_count=knowledge_data.get('usage_count', 0),
                            lifecycle_actions=[LifecycleAction.MERGE]
                        )
                        items.append(item)
                    else:
                        all_knowledge[content] = knowledge_id
                        
        return items
        
    def _analyze_outdated_knowledge(self) -> List[KnowledgeItem]:
        """Analyze outdated knowledge."""
        items = []
        
        # Analyze knowledge age
        for agent in self.network.agents.values():
            if hasattr(agent, 'knowledge_graph'):
                for knowledge_id, knowledge_data in agent.knowledge_graph.items():
                    created_at = knowledge_data.get('created_at', time.time())
                    age = time.time() - created_at
                    
                    if age > self.retirement_criteria['age_threshold']:
                        item = KnowledgeItem(
                            id=knowledge_id,
                            content=str(knowledge_data.get('content', '')),
                            knowledge_type=knowledge_data.get('type', 'unknown'),
                            state=KnowledgeState.OUTDATED,
                            created_at=created_at,
                            last_accessed=knowledge_data.get('last_accessed', time.time()),
                            usage_count=knowledge_data.get('usage_count', 0),
                            lifecycle_actions=[LifecycleAction.UPDATE, LifecycleAction.ARCHIVE]
                        )
                        items.append(item)
                        
        return items
        
    def apply_lifecycle_action(self, item_id: str, action: LifecycleAction) -> bool:
        """Apply a lifecycle action to a knowledge item."""
        if item_id not in self.knowledge_items:
            return False
            
        item = self.knowledge_items[item_id]
        
        # Apply action
        success = self._execute_lifecycle_action(item, action)
        
        if success:
            # Update item state
            if action == LifecycleAction.UPDATE:
                item.state = KnowledgeState.ACTIVE
            elif action == LifecycleAction.MERGE:
                item.state = KnowledgeState.ACTIVE
            elif action == LifecycleAction.RETIRE:
                item.state = KnowledgeState.OBSOLETE
            elif action == LifecycleAction.ARCHIVE:
                item.state = KnowledgeState.OBSOLETE
                
            # Log action
            self._log_lifecycle_event('action_applied', {
                'item_id': item_id,
                'action': action.name,
                'success': success
            })
            
        return success
        
    def _execute_lifecycle_action(self, item: KnowledgeItem, action: LifecycleAction) -> bool:
        """Execute a specific lifecycle action."""
        if action == LifecycleAction.UPDATE:
            return self._update_knowledge(item)
        elif action == LifecycleAction.MERGE:
            return self._merge_knowledge(item)
        elif action == LifecycleAction.RETIRE:
            return self._retire_knowledge(item)
        elif action == LifecycleAction.CONFLICT_RESOLVE:
            return self._resolve_conflict(item)
        elif action == LifecycleAction.OPTIMIZE:
            return self._optimize_knowledge(item)
        elif action == LifecycleAction.ARCHIVE:
            return self._archive_knowledge(item)
        else:
            return False
            
    def _update_knowledge(self, item: KnowledgeItem) -> bool:
        """Update knowledge item."""
        # Implement knowledge update logic
        item.last_accessed = time.time()
        item.usage_count += 1
        return True
        
    def _merge_knowledge(self, item: KnowledgeItem) -> bool:
        """Merge knowledge item with similar items."""
        # Implement knowledge merging logic
        return True
        
    def _retire_knowledge(self, item: KnowledgeItem) -> bool:
        """Retire knowledge item."""
        # Implement knowledge retirement logic
        return True
        
    def _resolve_conflict(self, item: KnowledgeItem) -> bool:
        """Resolve knowledge conflict."""
        # Implement conflict resolution logic
        return True
        
    def _optimize_knowledge(self, item: KnowledgeItem) -> bool:
        """Optimize knowledge item."""
        # Implement knowledge optimization logic
        return True
        
    def _archive_knowledge(self, item: KnowledgeItem) -> bool:
        """Archive knowledge item."""
        # Implement knowledge archiving logic
        return True
        
    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get knowledge lifecycle statistics."""
        total_items = len(self.knowledge_items)
        active_items = sum(1 for item in self.knowledge_items.values() 
                         if item.state == KnowledgeState.ACTIVE)
        obsolete_items = sum(1 for item in self.knowledge_items.values() 
                           if item.state == KnowledgeState.OBSOLETE)
        conflicting_items = sum(1 for item in self.knowledge_items.values() 
                              if item.state == KnowledgeState.CONFLICTING)
        redundant_items = sum(1 for item in self.knowledge_items.values() 
                            if item.state == KnowledgeState.REDUNDANT)
        
        # Calculate average usage and confidence
        usage_counts = [item.usage_count for item in self.knowledge_items.values()]
        confidence_scores = [item.confidence_score for item in self.knowledge_items.values()]
        
        average_usage = statistics.mean(usage_counts) if usage_counts else 0
        average_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        # Calculate state distribution
        state_distribution = defaultdict(int)
        for item in self.knowledge_items.values():
            state_distribution[item.state.name] += 1
            
        return {
            'total_items': total_items,
            'active_items': active_items,
            'obsolete_items': obsolete_items,
            'conflicting_items': conflicting_items,
            'redundant_items': redundant_items,
            'average_usage': average_usage,
            'average_confidence': average_confidence,
            'state_distribution': dict(state_distribution)
        }
        
    def _log_lifecycle_event(self, event_type: str, data: Dict[str, Any]):
        """Log a lifecycle event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.lifecycle_history.append(event)
        
    def get_lifecycle_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get lifecycle history."""
        return list(self.lifecycle_history[-limit:])
