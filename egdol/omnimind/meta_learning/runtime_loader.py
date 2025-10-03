"""
Runtime Loader for OmniMind Meta-Learning
Dynamically loads and integrates generated skills at runtime.
"""

import os
import sys
import importlib
import time
from typing import Dict, Any, List, Optional
from ..skills.base import BaseSkill, SkillManager


class RuntimeLoader:
    """Loads and manages dynamically generated skills at runtime."""
    
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = skills_dir
        self.loaded_skills: Dict[str, BaseSkill] = {}
        self.skill_manager = SkillManager()
        self.load_history: List[Dict[str, Any]] = []
        
        # Ensure skills directory is in Python path
        if skills_dir not in sys.path:
            sys.path.insert(0, skills_dir)
            
    def discover_skills(self) -> List[str]:
        """Discover available skill files in the skills directory."""
        skill_files = []
        
        if not os.path.exists(self.skills_dir):
            return skill_files
            
        for filename in os.listdir(self.skills_dir):
            if filename.endswith('_skill.py') and not filename.startswith('__'):
                skill_files.append(os.path.join(self.skills_dir, filename))
                
        return skill_files
        
    def load_skill(self, skill_file: str) -> Optional[Dict[str, Any]]:
        """Load a single skill from file."""
        try:
            # Extract skill name from filename
            skill_name = os.path.splitext(os.path.basename(skill_file))[0].replace('_skill', '')
            
            # Import the module
            module_name = os.path.splitext(os.path.basename(skill_file))[0]
            module = importlib.import_module(module_name)
            
            # Find the skill class
            skill_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseSkill) and 
                    attr != BaseSkill):
                    skill_class = attr
                    break
                    
            if not skill_class:
                return None
                
            # Instantiate the skill
            skill_instance = skill_class()
            
            # Register with skill manager
            self.skill_manager.register_skill(skill_name, skill_instance)
            self.loaded_skills[skill_name] = skill_instance
            
            # Record load history
            load_record = {
                'skill_name': skill_name,
                'file': skill_file,
                'loaded_at': time.time(),
                'success': True
            }
            self.load_history.append(load_record)
            
            return {
                'skill_name': skill_name,
                'file': skill_file,
                'instance': skill_instance,
                'success': True
            }
            
        except Exception as e:
            # Record failed load
            load_record = {
                'skill_name': skill_name if 'skill_name' in locals() else 'unknown',
                'file': skill_file,
                'loaded_at': time.time(),
                'success': False,
                'error': str(e)
            }
            self.load_history.append(load_record)
            
            return {
                'skill_name': skill_name if 'skill_name' in locals() else 'unknown',
                'file': skill_file,
                'success': False,
                'error': str(e)
            }
            
    def load_all_skills(self) -> Dict[str, Any]:
        """Load all available skills."""
        skill_files = self.discover_skills()
        results = {
            'total_files': len(skill_files),
            'loaded_skills': [],
            'failed_skills': [],
            'success_count': 0,
            'failure_count': 0
        }
        
        for skill_file in skill_files:
            result = self.load_skill(skill_file)
            if result['success']:
                results['loaded_skills'].append(result)
                results['success_count'] += 1
            else:
                results['failed_skills'].append(result)
                results['failure_count'] += 1
                
        return results
        
    def reload_skill(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Reload a specific skill."""
        if skill_name in self.loaded_skills:
            # Remove from manager
            self.skill_manager.unregister_skill(skill_name)
            del self.loaded_skills[skill_name]
            
        # Find skill file
        skill_file = os.path.join(self.skills_dir, f"{skill_name}_skill.py")
        if not os.path.exists(skill_file):
            return None
            
        # Reload the skill
        return self.load_skill(skill_file)
        
    def unload_skill(self, skill_name: str) -> bool:
        """Unload a specific skill."""
        if skill_name in self.loaded_skills:
            # Remove from manager
            self.skill_manager.unregister_skill(skill_name)
            del self.loaded_skills[skill_name]
            
            # Record unload
            unload_record = {
                'skill_name': skill_name,
                'unloaded_at': time.time(),
                'action': 'unload'
            }
            self.load_history.append(unload_record)
            
            return True
        return False
        
    def get_loaded_skills(self) -> Dict[str, BaseSkill]:
        """Get all currently loaded skills."""
        return self.loaded_skills.copy()
        
    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded skill."""
        if skill_name not in self.loaded_skills:
            return None
            
        skill = self.loaded_skills[skill_name]
        return {
            'name': skill.name,
            'description': skill.description,
            'capabilities': skill.capabilities,
            'type': type(skill).__name__
        }
        
    def get_load_history(self) -> List[Dict[str, Any]]:
        """Get history of skill loading operations."""
        return self.load_history.copy()
        
    def get_skill_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded skills."""
        return {
            'total_loaded': len(self.loaded_skills),
            'skill_names': list(self.loaded_skills.keys()),
            'load_history_count': len(self.load_history),
            'recent_loads': len([h for h in self.load_history 
                               if h.get('loaded_at', 0) > time.time() - 3600])
        }
        
    def announce_new_skill(self, skill_name: str) -> str:
        """Generate announcement message for new skill."""
        skill_info = self.get_skill_info(skill_name)
        if not skill_info:
            return f"Failed to load skill: {skill_name}"
            
        announcement = f"""
🧠 New Skill Loaded: {skill_name}
📝 Description: {skill_info['description']}
🔧 Capabilities: {', '.join(skill_info['capabilities'])}
✅ Status: Ready for use
        """.strip()
        
        return announcement
        
    def test_skill_integration(self, skill_name: str) -> Dict[str, Any]:
        """Test if a skill integrates properly with the system."""
        if skill_name not in self.loaded_skills:
            return {'success': False, 'error': 'Skill not loaded'}
            
        skill = self.loaded_skills[skill_name]
        test_results = {
            'skill_name': skill_name,
            'can_handle_test': False,
            'handle_test': False,
            'integration_test': False,
            'errors': []
        }
        
        try:
            # Test can_handle method
            test_input = "test input"
            can_handle_result = skill.can_handle(test_input, 'general', {})
            test_results['can_handle_test'] = isinstance(can_handle_result, bool)
            
            # Test handle method
            handle_result = skill.handle(test_input, 'general', {})
            test_results['handle_test'] = isinstance(handle_result, dict)
            
            # Test integration with skill manager
            manager_skills = self.skill_manager.get_skills()
            test_results['integration_test'] = skill_name in manager_skills
            
        except Exception as e:
            test_results['errors'].append(str(e))
            
        test_results['success'] = all([
            test_results['can_handle_test'],
            test_results['handle_test'],
            test_results['integration_test']
        ])
        
        return test_results
