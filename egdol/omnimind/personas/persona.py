"""
Persona System for OmniMind
Manages specialized personas with different skills and knowledge.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum, auto


class PersonaType(Enum):
    """Types of personas."""
    GENERAL = auto()
    LEGAL = auto()
    CODING = auto()
    HISTORICAL = auto()
    STRATEGIC = auto()
    TECHNICAL = auto()
    CREATIVE = auto()


@dataclass
class Persona:
    """A persona with specialized skills and knowledge."""
    id: str
    name: str
    description: str
    persona_type: PersonaType
    skills: List[str]
    knowledge_base: Dict[str, Any]
    response_style: Dict[str, Any]
    memory_context: Dict[str, Any]
    created_at: float = None
    last_used: float = None
    usage_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_used is None:
            self.last_used = time.time()
            
    def use(self):
        """Mark persona as used."""
        self.last_used = time.time()
        self.usage_count += 1
        
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> bool:
        """Check if persona can handle a query."""
        context = context or {}
        
        # Check if any skills match the query
        query_lower = query.lower()
        for skill in self.skills:
            if skill.lower() in query_lower:
                return True
                
        # Check knowledge base for relevant terms
        for key, value in self.knowledge_base.items():
            if key.lower() in query_lower:
                return True
                
        return False
        
    def get_relevance_score(self, query: str, context: Dict[str, Any] = None) -> float:
        """Get relevance score for a query."""
        context = context or {}
        score = 0.0
        
        query_lower = query.lower()
        
        # Score based on skills
        for skill in self.skills:
            if skill.lower() in query_lower:
                score += 1.0
                
        # Score based on knowledge base
        for key, value in self.knowledge_base.items():
            if key.lower() in query_lower:
                score += 0.5
                
        # Score based on persona type and query content
        if self.persona_type == PersonaType.LEGAL:
            if any(word in query_lower for word in ['legal', 'contract', 'law', 'court', 'lawyer']):
                score += 2.0
        elif self.persona_type == PersonaType.CODING:
            if any(word in query_lower for word in ['code', 'programming', 'debug', 'bug', 'software', 'development']):
                score += 2.0
        elif self.persona_type == PersonaType.HISTORICAL:
            if any(word in query_lower for word in ['history', 'historical', 'past', 'ancient', 'medieval']):
                score += 2.0
        elif self.persona_type == PersonaType.STRATEGIC:
            if any(word in query_lower for word in ['strategy', 'strategic', 'plan', 'planning', 'analysis']):
                score += 2.0
        elif self.persona_type == PersonaType.GENERAL:
            score += 0.1  # General personas have lower base score
        else:
            score += 0.2  # Other specialized personas get higher base score
            
        return score
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'persona_type': self.persona_type.name,
            'skills': self.skills,
            'knowledge_base': self.knowledge_base,
            'response_style': self.response_style,
            'memory_context': self.memory_context,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'usage_count': self.usage_count
        }


class PersonaManager:
    """Manages personas and persona switching."""
    
    def __init__(self):
        self.personas: Dict[str, Persona] = {}
        self.active_persona: Optional[Persona] = None
        self.persona_history: List[Dict[str, Any]] = []
        self.switch_history: List[Dict[str, Any]] = []
        
    def create_persona(self, name: str, description: str, persona_type: PersonaType,
                      skills: List[str], knowledge_base: Dict[str, Any] = None,
                      response_style: Dict[str, Any] = None) -> Persona:
        """Create a new persona."""
        persona_id = str(uuid.uuid4())
        
        persona = Persona(
            id=persona_id,
            name=name,
            description=description,
            persona_type=persona_type,
            skills=skills,
            knowledge_base=knowledge_base or {},
            response_style=response_style or {},
            memory_context={}
        )
        
        self.personas[persona_id] = persona
        
        # Record creation
        self.persona_history.append({
            'action': 'create',
            'persona_id': persona_id,
            'name': name,
            'timestamp': time.time()
        })
        
        return persona
        
    def switch_persona(self, persona_id: str) -> bool:
        """Switch to a different persona."""
        if persona_id not in self.personas:
            return False
            
        old_persona = self.active_persona
        new_persona = self.personas[persona_id]
        
        # Record switch
        self.switch_history.append({
            'from_persona': old_persona.id if old_persona else None,
            'to_persona': persona_id,
            'timestamp': time.time()
        })
        
        # Switch persona
        self.active_persona = new_persona
        new_persona.use()
        
        return True
        
    def get_best_persona(self, query: str, context: Dict[str, Any] = None) -> Optional[Persona]:
        """Get the best persona for a query."""
        if not self.personas:
            return None
            
        best_persona = None
        best_score = 0.0
        
        for persona in self.personas.values():
            score = persona.get_relevance_score(query, context)
            if score > best_score:
                best_score = score
                best_persona = persona
                
        return best_persona
        
    def auto_switch_persona(self, query: str, context: Dict[str, Any] = None) -> bool:
        """Automatically switch to the best persona for a query."""
        best_persona = self.get_best_persona(query, context)
        
        if best_persona and best_persona != self.active_persona:
            return self.switch_persona(best_persona.id)
            
        return False
        
    def get_active_persona(self) -> Optional[Persona]:
        """Get the currently active persona."""
        return self.active_persona
        
    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """Get a persona by ID."""
        return self.personas.get(persona_id)
        
    def get_persona_by_name(self, name: str) -> Optional[Persona]:
        """Get a persona by name."""
        for persona in self.personas.values():
            if persona.name.lower() == name.lower():
                return persona
        return None
        
    def list_personas(self) -> List[Dict[str, Any]]:
        """List all personas."""
        return [persona.to_dict() for persona in self.personas.values()]
        
    def delete_persona(self, persona_id: str) -> bool:
        """Delete a persona."""
        if persona_id not in self.personas:
            return False
            
        # Don't delete if it's the active persona
        if self.active_persona and self.active_persona.id == persona_id:
            return False
            
        del self.personas[persona_id]
        
        # Record deletion
        self.persona_history.append({
            'action': 'delete',
            'persona_id': persona_id,
            'timestamp': time.time()
        })
        
        return True
        
    def update_persona(self, persona_id: str, updates: Dict[str, Any]) -> bool:
        """Update a persona."""
        if persona_id not in self.personas:
            return False
            
        persona = self.personas[persona_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(persona, key):
                setattr(persona, key, value)
                
        # Record update
        self.persona_history.append({
            'action': 'update',
            'persona_id': persona_id,
            'updates': updates,
            'timestamp': time.time()
        })
        
        return True
        
    def get_persona_stats(self) -> Dict[str, Any]:
        """Get persona statistics."""
        total_personas = len(self.personas)
        active_persona = self.active_persona.name if self.active_persona else None
        
        # Calculate usage statistics
        usage_stats = {}
        for persona in self.personas.values():
            usage_stats[persona.name] = {
                'usage_count': persona.usage_count,
                'last_used': persona.last_used,
                'created_at': persona.created_at
            }
            
        return {
            'total_personas': total_personas,
            'active_persona': active_persona,
            'usage_stats': usage_stats,
            'switch_count': len(self.switch_history),
            'creation_count': len(self.persona_history)
        }
        
    def get_persona_history(self) -> List[Dict[str, Any]]:
        """Get persona history."""
        return self.persona_history.copy()
        
    def get_switch_history(self) -> List[Dict[str, Any]]:
        """Get persona switch history."""
        return self.switch_history.copy()
        
    def clear_history(self):
        """Clear persona history."""
        self.persona_history.clear()
        self.switch_history.clear()
