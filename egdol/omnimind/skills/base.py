"""
Base Skill class for OmniMind
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseSkill(ABC):
    """Base class for all OmniMind skills."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = "A skill for OmniMind"
        self.capabilities = []
        
    @abstractmethod
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        """Check if this skill can handle the input."""
        pass
        
    @abstractmethod
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the user input and return response."""
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about this skill."""
        return {
            'name': self.name,
            'description': self.description,
            'capabilities': self.capabilities
        }


class SkillManager:
    """Manages all skills."""
    
    def __init__(self):
        self.skills: Dict[str, BaseSkill] = {}
        
    def register_skill(self, name: str, skill: BaseSkill):
        """Register a skill."""
        self.skills[name] = skill
        
    def unregister_skill(self, name: str) -> bool:
        """Unregister a skill."""
        if name in self.skills:
            del self.skills[name]
            return True
        return False
        
    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Get a skill by name."""
        return self.skills.get(name)
        
    def get_skills(self) -> Dict[str, BaseSkill]:
        """Get all skills."""
        return self.skills.copy()
        
    def get_skill_names(self) -> List[str]:
        """Get all skill names."""
        return list(self.skills.keys())
