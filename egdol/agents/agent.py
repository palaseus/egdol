"""
Agent and AgentProfile classes for Egdol.
Represents individual reasoning agents with their own knowledge bases.
"""

import time
import os
from typing import Dict, Any, List, Optional, Set
from ..rules_engine import RulesEngine
from ..interpreter import Interpreter
from ..memory import MemoryStore
from ..dsl.simple_dsl import SimpleDSL
from .communication import Message, MessageBus


class AgentProfile:
    """Profile for an agent with its characteristics and preferences."""
    
    def __init__(self, name: str, description: str = "", 
                 personality: Dict[str, Any] = None, 
                 expertise: List[str] = None):
        self.name = name
        self.description = description
        self.personality = personality or {}
        self.expertise = expertise or []
        self.created_at = time.time()
        self.last_active = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'personality': self.personality,
            'expertise': self.expertise,
            'created_at': self.created_at,
            'last_active': self.last_active
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentProfile':
        """Create profile from dictionary."""
        profile = cls(
            name=data['name'],
            description=data.get('description', ''),
            personality=data.get('personality', {}),
            expertise=data.get('expertise', [])
        )
        profile.created_at = data.get('created_at', time.time())
        profile.last_active = data.get('last_active', time.time())
        return profile


class Agent:
    """Individual reasoning agent with its own knowledge base."""
    
    def __init__(self, profile: AgentProfile, data_dir: str = "agents_data"):
        self.profile = profile
        self.data_dir = data_dir
        self.agent_dir = os.path.join(data_dir, profile.name)
        
        # Create agent directory
        os.makedirs(self.agent_dir, exist_ok=True)
        
        # Initialize components
        self.engine = RulesEngine()
        self.interpreter = Interpreter(self.engine)
        self.memory_store = MemoryStore(os.path.join(self.agent_dir, "memory.db"))
        self.dsl = SimpleDSL(self.engine)
        
        # Communication
        self.message_bus = MessageBus()
        self.message_history: List[Message] = []
        
        # Agent state
        self.is_active = True
        self.thinking_mode = "normal"  # normal, deep, creative
        self.confidence_threshold = 0.5
        
    def think(self, input_text: str) -> Dict[str, Any]:
        """Process input and generate response."""
        self.profile.last_active = time.time()
        
        # Store input in memory
        self.memory_store.store(
            content=input_text,
            item_type='input',
            source='user',
            confidence=1.0,
            metadata={'thinking_mode': self.thinking_mode}
        )
        
        # Process with DSL
        result = self.dsl.execute(input_text)
        
        # Store result in memory
        if result.get('type') in ['fact', 'rule', 'query']:
            self.memory_store.store(
                content=result,
                item_type='response',
                source='agent',
                confidence=0.8,
                metadata={'thinking_mode': self.thinking_mode}
            )
            
        return result
        
    def remember(self, content: str, importance: float = 0.5) -> int:
        """Explicitly remember something."""
        return self.memory_store.store(
            content=content,
            item_type='memory',
            source='explicit',
            confidence=importance,
            metadata={'explicit_memory': True}
        )
        
    def forget(self, pattern: str = None, item_type: str = None) -> int:
        """Forget memories matching criteria."""
        return self.memory_store.forget(pattern=pattern, item_type=item_type)
        
    def introspect(self) -> Dict[str, Any]:
        """Introspect the agent's current state."""
        stats = self.memory_store.get_stats()
        
        return {
            'name': self.profile.name,
            'description': self.profile.description,
            'is_active': self.is_active,
            'thinking_mode': self.thinking_mode,
            'confidence_threshold': self.confidence_threshold,
            'memory_stats': stats,
            'message_count': len(self.message_history),
            'expertise': self.profile.expertise,
            'personality': self.profile.personality
        }
        
    def set_thinking_mode(self, mode: str):
        """Set the thinking mode."""
        valid_modes = ["normal", "deep", "creative"]
        if mode in valid_modes:
            self.thinking_mode = mode
            
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for responses."""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            
    def send_message(self, recipient: str, content: str, message_type: str = "text") -> Message:
        """Send a message to another agent."""
        message = Message(
            sender=self.profile.name,
            recipient=recipient,
            content=content,
            message_type=message_type,
            timestamp=time.time()
        )
        
        self.message_bus.send(message)
        self.message_history.append(message)
        
        return message
        
    def receive_message(self, message: Message):
        """Receive a message from another agent."""
        self.message_history.append(message)
        
        # Process the message
        if message.message_type == "query":
            response = self.think(message.content)
            # Send response back
            self.send_message(
                recipient=message.sender,
                content=str(response),
                message_type="response"
            )
        elif message.message_type == "fact":
            # Store the fact
            self.remember(f"Received from {message.sender}: {message.content}")
            
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's memory."""
        memories = self.memory_store.search(limit=100)
        
        # Categorize memories
        by_type = {}
        for memory in memories:
            item_type = memory.item_type
            if item_type not in by_type:
                by_type[item_type] = []
            by_type[item_type].append(memory)
            
        # Get recent memories
        recent_memories = [m for m in memories if m.timestamp > time.time() - 86400]
        
        return {
            'total_memories': len(memories),
            'by_type': {t: len(mems) for t, mems in by_type.items()},
            'recent_memories': len(recent_memories),
            'memory_health': self._assess_memory_health(memories)
        }
        
    def _assess_memory_health(self, memories: List) -> Dict[str, Any]:
        """Assess the health of the agent's memory."""
        if not memories:
            return {'status': 'empty', 'score': 0.0}
            
        # Calculate health metrics
        total_memories = len(memories)
        high_confidence = len([m for m in memories if m.confidence > 0.7])
        recent_memories = len([m for m in memories if m.timestamp > time.time() - 86400])
        
        # Calculate health score
        confidence_score = high_confidence / total_memories if total_memories > 0 else 0
        recency_score = min(recent_memories / 10.0, 1.0)  # Cap at 1.0
        health_score = (confidence_score + recency_score) / 2
        
        if health_score > 0.8:
            status = 'excellent'
        elif health_score > 0.6:
            status = 'good'
        elif health_score > 0.4:
            status = 'fair'
        else:
            status = 'poor'
            
        return {
            'status': status,
            'score': health_score,
            'high_confidence_ratio': confidence_score,
            'recent_activity': recency_score
        }
        
    def export_knowledge(self, file_path: str) -> bool:
        """Export agent's knowledge to a file."""
        try:
            return self.memory_store.export_memories(file_path)
        except Exception:
            return False
            
    def import_knowledge(self, file_path: str) -> int:
        """Import knowledge from a file."""
        try:
            return self.memory_store.import_memories(file_path)
        except Exception:
            return 0
            
    def save_state(self) -> bool:
        """Save agent state to disk."""
        try:
            state_file = os.path.join(self.agent_dir, "state.json")
            import json
            
            state = {
                'profile': self.profile.to_dict(),
                'thinking_mode': self.thinking_mode,
                'confidence_threshold': self.confidence_threshold,
                'is_active': self.is_active,
                'last_saved': time.time()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception:
            return False
            
    def load_state(self) -> bool:
        """Load agent state from disk."""
        try:
            state_file = os.path.join(self.agent_dir, "state.json")
            import json
            
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            self.thinking_mode = state.get('thinking_mode', 'normal')
            self.confidence_threshold = state.get('confidence_threshold', 0.5)
            self.is_active = state.get('is_active', True)
            
            return True
        except Exception:
            return False
