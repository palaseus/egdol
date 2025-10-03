"""
Skill Generator for OmniMind Meta-Learning
Dynamically generates new skills based on conversation patterns and user instructions.
"""

import os
import re
import time
import ast
from typing import Dict, Any, List, Optional, Tuple
from ..skills.base import BaseSkill


class SkillGenerator:
    """Generates new skills dynamically based on patterns and instructions."""
    
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = skills_dir
        self.generated_skills: Dict[str, Dict[str, Any]] = {}
        self.skill_templates = self._load_skill_templates()
        
        # Ensure skills directory exists
        os.makedirs(skills_dir, exist_ok=True)
        
    def _load_skill_templates(self) -> Dict[str, str]:
        """Load skill templates for different types of skills."""
        return {
            'data_processing': '''
class {skill_name}Skill(BaseSkill):
    """Auto-generated skill for {description}."""
    
    def __init__(self):
        super().__init__()
        self.name = "{skill_name}"
        self.description = "{description}"
        self.capabilities = {capabilities}
        
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        """Check if this skill can handle the input."""
        user_input_lower = user_input.lower()
        
        # Pattern matching for {skill_name}
        patterns = {patterns}
        return any(pattern in user_input_lower for pattern in patterns)
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the input."""
        try:
            # {skill_name} processing logic
            {processing_logic}
            
            return {{
                'content': result,
                'reasoning': ['Processed with {skill_name} skill'],
                'metadata': {{'skill': '{skill_name}', 'type': 'generated'}}
            }}
        except Exception as e:
            return {{
                'content': f"I encountered an error with {skill_name}: {{str(e)}}",
                'reasoning': [f"Error: {{str(e)}}"],
                'metadata': {{'skill': '{skill_name}', 'error': str(e)}}
            }}
''',
            'analysis': '''
class {skill_name}Skill(BaseSkill):
    """Auto-generated analysis skill for {description}."""
    
    def __init__(self):
        super().__init__()
        self.name = "{skill_name}"
        self.description = "{description}"
        self.capabilities = {capabilities}
        
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        """Check if this skill can handle the input."""
        user_input_lower = user_input.lower()
        
        # Analysis patterns
        patterns = {patterns}
        return any(pattern in user_input_lower for pattern in patterns)
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the input."""
        try:
            # Analysis logic
            {analysis_logic}
            
            return {{
                'content': result,
                'reasoning': ['Analyzed with {skill_name} skill'],
                'metadata': {{'skill': '{skill_name}', 'type': 'generated'}}
            }}
        except Exception as e:
            return {{
                'content': f"Analysis error: {{str(e)}}",
                'reasoning': [f"Error: {{str(e)}}"],
                'metadata': {{'skill': '{skill_name}', 'error': str(e)}}
            }}
''',
            'conversion': '''
class {skill_name}Skill(BaseSkill):
    """Auto-generated conversion skill for {description}."""
    
    def __init__(self):
        super().__init__()
        self.name = "{skill_name}"
        self.description = "{description}"
        self.capabilities = {capabilities}
        
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        """Check if this skill can handle the input."""
        user_input_lower = user_input.lower()
        
        # Conversion patterns
        patterns = {patterns}
        return any(pattern in user_input_lower for pattern in patterns)
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the input."""
        try:
            # Conversion logic
            {conversion_logic}
            
            return {{
                'content': result,
                'reasoning': ['Converted with {skill_name} skill'],
                'metadata': {{'skill': '{skill_name}', 'type': 'generated'}}
            }}
        except Exception as e:
            return {{
                'content': f"Conversion error: {{str(e)}}",
                'reasoning': [f"Error: {{str(e)}}"],
                'metadata': {{'skill': '{skill_name}', 'error': str(e)}}
            }}
'''
        }
        
    def analyze_conversation_patterns(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze conversation patterns to identify potential new skills."""
        patterns = []
        
        # Analyze user inputs for repeated patterns
        user_inputs = [msg['content'] for msg in conversation_history if msg.get('type') == 'user']
        
        # Look for repeated phrases or patterns
        phrase_counts = {}
        for input_text in user_inputs:
            words = input_text.lower().split()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                
        # Find frequently repeated patterns
        for phrase, count in phrase_counts.items():
            if count >= 3:  # Threshold for pattern recognition
                patterns.append({
                    'pattern': phrase,
                    'frequency': count,
                    'type': 'repeated_phrase',
                    'confidence': min(count / 10.0, 1.0)
                })
                
        # Look for skill-related keywords
        skill_keywords = {
            'convert': 'conversion',
            'analyze': 'analysis',
            'process': 'data_processing',
            'calculate': 'calculation',
            'generate': 'generation',
            'format': 'formatting',
            'parse': 'parsing',
            'validate': 'validation'
        }
        
        for keyword, skill_type in skill_keywords.items():
            count = sum(1 for input_text in user_inputs if keyword in input_text.lower())
            if count >= 2:
                patterns.append({
                    'pattern': keyword,
                    'frequency': count,
                    'type': skill_type,
                    'confidence': min(count / 5.0, 1.0)
                })
                
        return patterns
        
    def generate_skill_from_instruction(self, instruction: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a skill from explicit user instruction."""
        instruction_lower = instruction.lower()
        
        # Extract skill requirements
        skill_name = self._extract_skill_name(instruction)
        description = self._extract_description(instruction)
        capabilities = self._extract_capabilities(instruction)
        patterns = self._extract_patterns(instruction)
        
        # Determine skill type
        skill_type = self._determine_skill_type(instruction)
        
        # Generate skill code
        skill_code = self._generate_skill_code(
            skill_name, description, capabilities, patterns, skill_type
        )
        
        # Save skill file
        skill_file = os.path.join(self.skills_dir, f"{skill_name}_skill.py")
        try:
            with open(skill_file, 'w') as f:
                f.write(skill_code)
                
            # Record generation
            self.generated_skills[skill_name] = {
                'file': skill_file,
                'description': description,
                'capabilities': capabilities,
                'generated_at': time.time(),
                'type': skill_type
            }
            
            return {
                'skill_name': skill_name,
                'file': skill_file,
                'description': description,
                'capabilities': capabilities,
                'type': skill_type,
                'success': True
            }
            
        except Exception as e:
            return None
            
    def _extract_skill_name(self, instruction: str) -> str:
        """Extract skill name from instruction."""
        # Look for "teach yourself" or "create skill" patterns
        if 'teach yourself' in instruction.lower():
            # Extract what to teach
            match = re.search(r'teach yourself (?:how to )?([^.]+)', instruction.lower())
            if match:
                skill_name = match.group(1).strip().replace(' ', '_')
                return skill_name
                
        # Look for "create" or "add" patterns
        if 'create' in instruction.lower() or 'add' in instruction.lower():
            match = re.search(r'(?:create|add) (?:a )?([^.]+)', instruction.lower())
            if match:
                skill_name = match.group(1).strip().replace(' ', '_')
                return skill_name
                
        # Default to generic name
        return f"generated_skill_{int(time.time())}"
        
    def _extract_description(self, instruction: str) -> str:
        """Extract description from instruction."""
        # Use the instruction as description, cleaned up
        description = instruction.strip()
        if description.endswith('.'):
            description = description[:-1]
        return description
        
    def _extract_capabilities(self, instruction: str) -> List[str]:
        """Extract capabilities from instruction."""
        capabilities = []
        
        if 'convert' in instruction.lower():
            capabilities.append('conversion')
        if 'analyze' in instruction.lower():
            capabilities.append('analysis')
        if 'process' in instruction.lower():
            capabilities.append('data_processing')
        if 'calculate' in instruction.lower():
            capabilities.append('calculation')
        if 'generate' in instruction.lower():
            capabilities.append('generation')
        if 'format' in instruction.lower():
            capabilities.append('formatting')
            
        # Default capabilities
        if not capabilities:
            capabilities = ['general_processing']
            
        return capabilities
        
    def _extract_patterns(self, instruction: str) -> List[str]:
        """Extract patterns from instruction."""
        patterns = []
        
        # Extract keywords from instruction
        words = instruction.lower().split()
        for word in words:
            if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'that', 'this']:
                patterns.append(word)
                
        # Add common patterns based on instruction type
        if 'convert' in instruction.lower():
            patterns.extend(['convert', 'change', 'transform'])
        if 'analyze' in instruction.lower():
            patterns.extend(['analyze', 'examine', 'review'])
        if 'calculate' in instruction.lower():
            patterns.extend(['calculate', 'compute', 'solve'])
            
        return patterns[:5]  # Limit to 5 patterns
        
    def _determine_skill_type(self, instruction: str) -> str:
        """Determine skill type from instruction."""
        instruction_lower = instruction.lower()
        
        if 'convert' in instruction_lower:
            return 'conversion'
        elif 'analyze' in instruction_lower:
            return 'analysis'
        elif 'process' in instruction_lower:
            return 'data_processing'
        else:
            return 'data_processing'  # Default
            
    def _generate_skill_code(self, skill_name: str, description: str, 
                            capabilities: List[str], patterns: List[str], 
                            skill_type: str) -> str:
        """Generate skill code using templates."""
        template = self.skill_templates.get(skill_type, self.skill_templates['data_processing'])
        
        # Generate processing logic based on skill type
        if skill_type == 'conversion':
            processing_logic = '''
            # Extract conversion parameters
            input_text = user_input.strip()
            
            # Simple conversion logic (placeholder)
            if 'binary' in input_text.lower() and 'hex' in input_text.lower():
                result = "Binary to hex conversion: " + input_text
            elif 'hex' in input_text.lower() and 'binary' in input_text.lower():
                result = "Hex to binary conversion: " + input_text
            else:
                result = f"Conversion result for: {input_text}"
            '''
        elif skill_type == 'analysis':
            processing_logic = '''
            # Extract analysis parameters
            input_text = user_input.strip()
            
            # Simple analysis logic (placeholder)
            word_count = len(input_text.split())
            char_count = len(input_text)
            result = f"Analysis: {word_count} words, {char_count} characters"
            '''
        else:
            processing_logic = '''
            # Extract processing parameters
            input_text = user_input.strip()
            
            # Simple processing logic (placeholder)
            result = f"Processed: {input_text}"
            '''
            
        # Fill template
        skill_code = template.format(
            skill_name=skill_name,
            description=description,
            capabilities=capabilities,
            patterns=patterns,
            processing_logic=processing_logic,
            analysis_logic=processing_logic,
            conversion_logic=processing_logic
        )
        
        # Add imports
        imports = '''
"""
Auto-generated skill for OmniMind Meta-Learning
Generated at: {timestamp}
"""

from typing import Dict, Any
from ..base import BaseSkill
'''.format(timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
        
        return imports + skill_code
        
    def get_generated_skills(self) -> Dict[str, Dict[str, Any]]:
        """Get all generated skills."""
        return self.generated_skills.copy()
        
    def remove_skill(self, skill_name: str) -> bool:
        """Remove a generated skill."""
        if skill_name in self.generated_skills:
            skill_info = self.generated_skills[skill_name]
            try:
                # Remove file
                if os.path.exists(skill_info['file']):
                    os.remove(skill_info['file'])
                    
                # Remove from registry
                del self.generated_skills[skill_name]
                return True
            except Exception:
                return False
        return False
