"""
Context Manager for DSL 2.0
Manages persistent memory context for NLU and DSL translation.
"""

import time
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, deque


class ContextManager:
    """Manages context for DSL 2.0 parsing and translation."""
    
    def __init__(self, max_context_size: int = 1000):
        self.max_context_size = max_context_size
        self.context_history: deque = deque(maxlen=max_context_size)
        self.entity_references: Dict[str, List[str]] = defaultdict(list)
        self.variable_bindings: Dict[str, Any] = {}
        self.session_variables: Dict[str, Any] = {}
        self.conversation_stack: List[Dict[str, Any]] = []
        
    def add_context(self, context: Dict[str, Any]):
        """Add context to the history."""
        context['timestamp'] = time.time()
        self.context_history.append(context)
        
        # Extract entity references
        if 'entities' in context:
            for entity in context['entities']:
                self.entity_references[entity].append(context['timestamp'])
                
    def get_recent_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent context entries."""
        return list(self.context_history)[-limit:]
        
    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve pronoun to entity reference."""
        pronoun_map = {
            'he': 'male_entity',
            'she': 'female_entity',
            'it': 'neutral_entity',
            'they': 'plural_entity',
            'this': 'current_entity',
            'that': 'previous_entity'
        }
        
        if pronoun.lower() in pronoun_map:
            entity_type = pronoun_map[pronoun.lower()]
            # Find most recent entity of this type
            for context in reversed(self.context_history):
                if entity_type in context:
                    return context[entity_type]
                    
        return None
        
    def resolve_variable(self, variable_name: str) -> Optional[Any]:
        """Resolve variable to its value."""
        # Check session variables first
        if variable_name in self.session_variables:
            return self.session_variables[variable_name]
            
        # Check variable bindings
        if variable_name in self.variable_bindings:
            return self.variable_bindings[variable_name]
            
        # Check recent context for entity references
        if variable_name in self.entity_references:
            recent_refs = self.entity_references[variable_name]
            if recent_refs:
                # Return most recent reference
                return recent_refs[-1]
                
        return None
        
    def bind_variable(self, variable_name: str, value: Any):
        """Bind a variable to a value."""
        self.variable_bindings[variable_name] = value
        
    def bind_session_variable(self, variable_name: str, value: Any):
        """Bind a session variable."""
        self.session_variables[variable_name] = value
        
    def get_entity_history(self, entity: str) -> List[Dict[str, Any]]:
        """Get history of references to an entity."""
        history = []
        for context in self.context_history:
            if 'entities' in context and entity in context['entities']:
                history.append(context)
        return history
        
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context."""
        return {
            'recent_context': self.get_recent_context(),
            'entity_references': dict(self.entity_references),
            'variable_bindings': self.variable_bindings.copy(),
            'session_variables': self.session_variables.copy(),
            'conversation_depth': len(self.conversation_stack)
        }
        
    def push_conversation(self, conversation: Dict[str, Any]):
        """Push a conversation onto the stack."""
        self.conversation_stack.append(conversation)
        
    def pop_conversation(self) -> Optional[Dict[str, Any]]:
        """Pop a conversation from the stack."""
        if self.conversation_stack:
            return self.conversation_stack.pop()
        return None
        
    def get_current_conversation(self) -> Optional[Dict[str, Any]]:
        """Get current conversation."""
        if self.conversation_stack:
            return self.conversation_stack[-1]
        return None
        
    def clear_context(self):
        """Clear all context."""
        self.context_history.clear()
        self.entity_references.clear()
        self.variable_bindings.clear()
        self.session_variables.clear()
        self.conversation_stack.clear()
        
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context."""
        return {
            'total_context_entries': len(self.context_history),
            'unique_entities': len(self.entity_references),
            'variable_bindings_count': len(self.variable_bindings),
            'session_variables_count': len(self.session_variables),
            'conversation_stack_depth': len(self.conversation_stack),
            'oldest_context': self.context_history[0]['timestamp'] if self.context_history else None,
            'newest_context': self.context_history[-1]['timestamp'] if self.context_history else None
        }
        
    def find_related_entities(self, entity: str) -> List[str]:
        """Find entities related to the given entity."""
        related = set()
        
        for context in self.context_history:
            if 'entities' in context and entity in context['entities']:
                # Find other entities in the same context
                for other_entity in context['entities']:
                    if other_entity != entity:
                        related.add(other_entity)
                        
        return list(related)
        
    def get_entity_frequency(self, entity: str) -> int:
        """Get frequency of entity references."""
        return len(self.entity_references.get(entity, []))
        
    def get_most_referenced_entities(self, limit: int = 10) -> List[tuple]:
        """Get most frequently referenced entities."""
        entity_freq = [(entity, len(refs)) for entity, refs in self.entity_references.items()]
        entity_freq.sort(key=lambda x: x[1], reverse=True)
        return entity_freq[:limit]
        
    def suggest_pronoun_resolution(self, pronoun: str) -> List[str]:
        """Suggest possible resolutions for a pronoun."""
        suggestions = []
        
        # Get recent entities
        recent_entities = set()
        for context in list(self.context_history)[-5:]:  # Last 5 contexts
            if 'entities' in context:
                recent_entities.update(context['entities'])
                
        # Filter by pronoun type
        if pronoun.lower() in ['he', 'him', 'his']:
            # Look for male entities
            for entity in recent_entities:
                if self._is_male_entity(entity):
                    suggestions.append(entity)
        elif pronoun.lower() in ['she', 'her', 'hers']:
            # Look for female entities
            for entity in recent_entities:
                if self._is_female_entity(entity):
                    suggestions.append(entity)
        elif pronoun.lower() in ['it', 'its']:
            # Look for neutral entities
            for entity in recent_entities:
                if self._is_neutral_entity(entity):
                    suggestions.append(entity)
        else:
            # Return all recent entities
            suggestions = list(recent_entities)
            
        return suggestions[:5]  # Limit to 5 suggestions
        
    def _is_male_entity(self, entity: str) -> bool:
        """Check if entity is likely male."""
        male_indicators = ['man', 'boy', 'male', 'he', 'him', 'his']
        return any(indicator in entity.lower() for indicator in male_indicators)
        
    def _is_female_entity(self, entity: str) -> bool:
        """Check if entity is likely female."""
        female_indicators = ['woman', 'girl', 'female', 'she', 'her', 'hers']
        return any(indicator in entity.lower() for indicator in female_indicators)
        
    def _is_neutral_entity(self, entity: str) -> bool:
        """Check if entity is likely neutral."""
        neutral_indicators = ['thing', 'object', 'item', 'it', 'its']
        return any(indicator in entity.lower() for indicator in neutral_indicators)
