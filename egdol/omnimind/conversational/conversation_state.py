"""
Conversation State Management
Manages conversational context, history, and state transitions.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class ConversationPhase(Enum):
    """Phases of conversation flow."""
    GREETING = auto()
    EXPLORATION = auto()
    DEEP_DIVE = auto()
    REFLECTION = auto()
    CLOSURE = auto()


class ContextType(Enum):
    """Types of conversational context."""
    GENERAL = auto()
    CIVILIZATIONAL = auto()
    STRATEGIC = auto()
    PHILOSOPHICAL = auto()
    TECHNICAL = auto()
    CREATIVE = auto()
    ANALYTICAL = auto()


@dataclass
class ConversationContext:
    """Represents the current conversational context."""
    context_type: ContextType
    domain: str
    complexity_level: float  # 0.0 to 1.0
    emotional_tone: str
    focus_areas: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    turn_id: str
    timestamp: datetime
    user_input: str
    system_response: str
    intent: str
    personality_used: str
    reasoning_trace: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    context_shift: bool = False
    meta_insights: List[str] = field(default_factory=list)


@dataclass
class ConversationState:
    """Manages the complete state of a conversation session."""
    session_id: str
    created_at: datetime
    current_phase: ConversationPhase
    active_personality: str
    context: ConversationContext
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    personality_history: List[Tuple[str, datetime]] = field(default_factory=list)
    context_transitions: List[Tuple[ContextType, datetime]] = field(default_factory=list)
    meta_rules_applied: List[str] = field(default_factory=list)
    civilizational_insights: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"conv_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn to the history."""
        self.conversation_history.append(turn)
        
        # Track personality changes
        if self.personality_history and self.personality_history[-1][0] != turn.personality_used:
            self.personality_history.append((turn.personality_used, turn.timestamp))
        elif not self.personality_history:
            self.personality_history.append((turn.personality_used, turn.timestamp))
    
    def switch_personality(self, new_personality: str) -> None:
        """Switch to a new personality."""
        if self.active_personality != new_personality:
            self.active_personality = new_personality
            self.personality_history.append((new_personality, datetime.now()))
    
    def update_context(self, new_context: ConversationContext) -> None:
        """Update the conversation context."""
        if self.context.context_type != new_context.context_type:
            self.context_transitions.append((new_context.context_type, datetime.now()))
        self.context = new_context
    
    def get_recent_context(self, turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        return self.conversation_history[-turns:] if self.conversation_history else []
    
    def get_personality_usage(self) -> Dict[str, int]:
        """Get usage statistics for personalities."""
        usage = {}
        for personality, _ in self.personality_history:
            usage[personality] = usage.get(personality, 0) + 1
        return usage
    
    def get_context_evolution(self) -> List[Tuple[ContextType, datetime]]:
        """Get the evolution of context types over time."""
        return self.context_transitions.copy()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        return {
            'session_id': self.session_id,
            'duration': (datetime.now() - self.created_at).total_seconds(),
            'total_turns': len(self.conversation_history),
            'current_phase': self.current_phase.name,
            'active_personality': self.active_personality,
            'context_type': self.context.context_type.name,
            'personality_usage': self.get_personality_usage(),
            'context_evolution': len(self.context_transitions),
            'meta_rules_applied': len(self.meta_rules_applied),
            'civilizational_insights': len(self.civilizational_insights)
        }
    
    def should_switch_personality(self, user_input: str, available_personalities: List[str]) -> Optional[str]:
        """Determine if personality should be switched based on input."""
        # Simple heuristics for personality switching
        input_lower = user_input.lower()
        
        # Strategic/military context
        if any(word in input_lower for word in ['strategy', 'war', 'battle', 'tactics', 'military']):
            if 'strategos' in [p.lower() for p in available_personalities]:
                return 'strategos'
        
        # Historical/philosophical context
        if any(word in input_lower for word in ['history', 'philosophy', 'ancient', 'past', 'tradition']):
            if 'archivist' in [p.lower() for p in available_personalities]:
                return 'archivist'
        
        # Legal/meta-rule context
        if any(word in input_lower for word in ['law', 'rule', 'meta', 'principle', 'governance']):
            if 'lawmaker' in [p.lower() for p in available_personalities]:
                return 'lawmaker'
        
        # Universal/comparative context
        if any(word in input_lower for word in ['universe', 'compare', 'universal', 'cosmic', 'reality']):
            if 'oracle' in [p.lower() for p in available_personalities]:
                return 'oracle'
        
        return None
    
    def evolve_phase(self, user_input: str) -> ConversationPhase:
        """Evolve conversation phase based on input patterns."""
        input_lower = user_input.lower()
        
        # Exploration phase (check first to avoid conflicts)
        if any(word in input_lower for word in ['explore', 'discover', 'learn', 'understand']):
            self.current_phase = ConversationPhase.EXPLORATION
            return ConversationPhase.EXPLORATION
        
        # Greeting phase
        if (input_lower.startswith('hello') or input_lower.startswith('hi') or 
            input_lower.startswith('greetings') or input_lower.startswith('start')):
            self.current_phase = ConversationPhase.GREETING
            return ConversationPhase.GREETING
        
        # Deep dive phase
        if any(word in input_lower for word in ['analyze', 'deep', 'complex', 'detailed']):
            self.current_phase = ConversationPhase.DEEP_DIVE
            return ConversationPhase.DEEP_DIVE
        
        # Reflection phase
        if any(word in input_lower for word in ['reflect', 'think', 'consider', 'ponder']):
            self.current_phase = ConversationPhase.REFLECTION
            return ConversationPhase.REFLECTION
        
        # Closure phase
        if any(word in input_lower for word in ['goodbye', 'end', 'finish', 'conclude']):
            self.current_phase = ConversationPhase.CLOSURE
            return ConversationPhase.CLOSURE
        
        # Default to current phase or exploration
        if self.current_phase is None:
            self.current_phase = ConversationPhase.EXPLORATION
        return self.current_phase
