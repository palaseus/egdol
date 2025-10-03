"""
Skill Router for OmniMind
Routes user input to appropriate skill handlers.
"""

import importlib
import os
from typing import Dict, Any, List, Optional, Callable
from .skills import SkillManager, BaseSkill


class SkillRouter:
    """Routes user input to appropriate skill handlers."""
    
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = skills_dir
        self.skill_manager = SkillManager()
        self.loaded_skills: Dict[str, BaseSkill] = {}
        
        # Load default skills
        self._load_default_skills()
        
    def _load_default_skills(self):
        """Load default skills."""
        # Math skill
        from .skills.math_skill import MathSkill
        self.skill_manager.register_skill('math', MathSkill())
        
        # Logic skill
        from .skills.logic_skill import LogicSkill
        self.skill_manager.register_skill('logic', LogicSkill())
        
        # General skill
        from .skills.general_skill import GeneralSkill
        self.skill_manager.register_skill('general', GeneralSkill())
        
        # Code skill
        from .skills.code_skill import CodeSkill
        self.skill_manager.register_skill('code', CodeSkill())
        
        # File skill
        from .skills.file_skill import FileSkill
        self.skill_manager.register_skill('file', FileSkill())
        
    def route(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Route user input to appropriate skill."""
        # Get available skills
        skills = self.skill_manager.get_skills()
        
        # Try each skill to see if it can handle the input
        for skill_name, skill in skills.items():
            if skill.can_handle(user_input, intent, context):
                try:
                    response = skill.handle(user_input, intent, context)
                    return {
                        'handled': True,
                        'response': response['content'],
                        'skill': skill_name,
                        'reasoning': response.get('reasoning', []),
                        'metadata': response.get('metadata', {})
                    }
                except Exception as e:
                    # Skill failed, try next one
                    continue
                    
        # No skill could handle the input
        return {
            'handled': False,
            'response': None,
            'skill': None,
            'reasoning': [],
            'metadata': {}
        }
        
    def get_loaded_skills(self) -> List[str]:
        """Get list of loaded skill names."""
        return list(self.skill_manager.get_skills().keys())
        
    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific skill."""
        skills = self.skill_manager.get_skills()
        if skill_name not in skills:
            return None
            
        skill = skills[skill_name]
        return {
            'name': skill_name,
            'description': getattr(skill, 'description', 'No description'),
            'capabilities': getattr(skill, 'capabilities', []),
            'can_handle': skill.can_handle.__doc__ or 'No documentation'
        }
        
    def get_all_skill_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all skills."""
        skills = self.skill_manager.get_skills()
        return {name: self.get_skill_info(name) for name in skills.keys()}
        
    def reload_skills(self):
        """Reload all skills."""
        self.skill_manager = SkillManager()
        self._load_default_skills()
        
    def add_skill(self, skill_name: str, skill: BaseSkill):
        """Add a new skill."""
        self.skill_manager.register_skill(skill_name, skill)
        
    def remove_skill(self, skill_name: str) -> bool:
        """Remove a skill."""
        return self.skill_manager.unregister_skill(skill_name)
        
    def get_skill_stats(self) -> Dict[str, Any]:
        """Get statistics about skill usage."""
        skills = self.skill_manager.get_skills()
        return {
            'total_skills': len(skills),
            'skill_names': list(skills.keys()),
            'skill_types': [type(skill).__name__ for skill in skills.values()]
        }
