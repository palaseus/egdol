"""
Skill Validator for OmniMind Meta-Learning
Validates generated skills before activation.
"""

import ast
import importlib
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from ..skills.base import BaseSkill


class SkillValidator:
    """Validates generated skills for correctness and safety."""
    
    def __init__(self):
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        
    def validate_skill(self, skill_file: str, skill_name: str) -> Dict[str, Any]:
        """Validate a generated skill file."""
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'syntax_valid': False,
            'imports_valid': False,
            'interface_valid': False,
            'safety_valid': False
        }
        
        try:
            # Check if file exists
            if not os.path.exists(skill_file):
                validation_result['errors'].append(f"Skill file not found: {skill_file}")
                return validation_result
                
            # Read and parse the file
            with open(skill_file, 'r') as f:
                skill_code = f.read()
                
            # 1. Syntax validation
            syntax_valid, syntax_errors = self._validate_syntax(skill_code)
            validation_result['syntax_valid'] = syntax_valid
            if not syntax_valid:
                validation_result['errors'].extend(syntax_errors)
                
            # 2. Import validation
            imports_valid, import_errors = self._validate_imports(skill_code)
            validation_result['imports_valid'] = imports_valid
            if not imports_valid:
                validation_result['errors'].extend(import_errors)
                
            # 3. Interface validation
            interface_valid, interface_errors = self._validate_interface(skill_code, skill_name)
            validation_result['interface_valid'] = interface_valid
            if not interface_valid:
                validation_result['errors'].extend(interface_errors)
                
            # 4. Safety validation
            safety_valid, safety_errors = self._validate_safety(skill_code)
            validation_result['safety_valid'] = safety_valid
            if not safety_valid:
                validation_result['errors'].extend(safety_errors)
                
            # 5. Overall validation
            validation_result['valid'] = all([
                syntax_valid, imports_valid, interface_valid, safety_valid
            ])
            
            # Store result
            self.validation_results[skill_name] = validation_result
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            
        return validation_result
        
    def _validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax."""
        errors = []
        
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Parse error: {e}")
            return False, errors
            
    def _validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """Validate imports are safe and available."""
        errors = []
        
        try:
            tree = ast.parse(code)
            
            # Check for dangerous imports
            dangerous_imports = [
                'os.system', 'subprocess', 'eval', 'exec', 'compile',
                '__import__', 'globals', 'locals', 'vars'
            ]
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['os', 'subprocess', 'sys']:
                            errors.append(f"Dangerous import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in ['os', 'subprocess', 'sys']:
                        errors.append(f"Dangerous import from: {node.module}")
                        
            # Check for required imports (relaxed validation)
            has_base_skill = False
            has_typing = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and 'base' in node.module and any(alias.name == 'BaseSkill' for alias in node.names):
                        has_base_skill = True
                    elif node.module == 'typing':
                        has_typing = True
                        
            # Only warn about missing imports, don't fail validation
            if not has_base_skill:
                errors.append("Warning: Missing BaseSkill import")
            if not has_typing:
                errors.append("Warning: Missing typing import")
                
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Import validation error: {e}"]
            
    def _validate_interface(self, code: str, skill_name: str) -> Tuple[bool, List[str]]:
        """Validate that the skill implements the required interface."""
        errors = []
        
        try:
            tree = ast.parse(code)
            
            # Find the skill class
            skill_class = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and skill_name.lower() in node.name.lower():
                    skill_class = node
                    break
                    
            if not skill_class:
                errors.append(f"Skill class not found: {skill_name}")
                return False, errors
                
            # Check for required methods
            required_methods = ['can_handle', 'handle']
            method_names = [node.name for node in skill_class.body if isinstance(node, ast.FunctionDef)]
            
            for method in required_methods:
                if method not in method_names:
                    errors.append(f"Missing required method: {method}")
                    
            # Check for inheritance from BaseSkill
            has_base_skill = False
            for base in skill_class.bases:
                if isinstance(base, ast.Name) and base.id == 'BaseSkill':
                    has_base_skill = True
                    break
                elif isinstance(base, ast.Attribute) and base.attr == 'BaseSkill':
                    has_base_skill = True
                    break
                    
            if not has_base_skill:
                errors.append("Skill class must inherit from BaseSkill")
                
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Interface validation error: {e}"]
            
    def _validate_safety(self, code: str) -> Tuple[bool, List[str]]:
        """Validate that the code is safe to execute."""
        errors = []
        
        try:
            tree = ast.parse(code)
            
            # Check for dangerous operations
            dangerous_operations = [
                'eval', 'exec', 'compile', '__import__',
                'open', 'file', 'input', 'raw_input',
                'system', 'popen', 'spawn', 'fork', 'execv'
            ]
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in dangerous_operations:
                            errors.append(f"Dangerous operation: {node.func.id}")
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr in dangerous_operations:
                            errors.append(f"Dangerous operation: {node.func.attr}")
                            
            # Check for file system operations
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['remove', 'rmdir', 'unlink', 'chmod', 'chown']:
                            errors.append(f"File system operation: {node.func.attr}")
                            
            # Check for network operations
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['connect', 'send', 'recv', 'request']:
                            errors.append(f"Network operation: {node.func.attr}")
                            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Safety validation error: {e}"]
            
    def test_skill_execution(self, skill_file: str, skill_name: str) -> Dict[str, Any]:
        """Test if the skill can be imported and instantiated."""
        test_result = {
            'import_success': False,
            'instantiation_success': False,
            'method_tests': {},
            'errors': []
        }
        
        try:
            # Add the skills directory to Python path
            skills_dir = os.path.dirname(skill_file)
            if skills_dir not in sys.path:
                sys.path.insert(0, skills_dir)
                
            # Import the skill module
            module_name = os.path.splitext(os.path.basename(skill_file))[0]
            module = importlib.import_module(module_name)
            
            test_result['import_success'] = True
            
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
                test_result['errors'].append("Skill class not found")
                return test_result
                
            # Instantiate the skill
            skill_instance = skill_class()
            test_result['instantiation_success'] = True
            
            # Test methods
            test_cases = [
                {
                    'method': 'can_handle',
                    'args': ('test input', 'general', {}),
                    'expected_type': bool
                },
                {
                    'method': 'handle',
                    'args': ('test input', 'general', {}),
                    'expected_type': dict
                }
            ]
            
            for test_case in test_cases:
                method_name = test_case['method']
                args = test_case['args']
                expected_type = test_case['expected_type']
                
                try:
                    method = getattr(skill_instance, method_name)
                    result = method(*args)
                    
                    if isinstance(result, expected_type):
                        test_result['method_tests'][method_name] = 'PASS'
                    else:
                        test_result['method_tests'][method_name] = f'FAIL: Expected {expected_type}, got {type(result)}'
                        
                except Exception as e:
                    test_result['method_tests'][method_name] = f'ERROR: {str(e)}'
                    test_result['errors'].append(f"{method_name} test failed: {str(e)}")
                    
        except Exception as e:
            test_result['errors'].append(f"Execution test error: {str(e)}")
            
        return test_result
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {'total_skills': 0, 'valid_skills': 0, 'invalid_skills': 0, 'validation_rate': 0}
            
        total_skills = len(self.validation_results)
        valid_skills = sum(1 for result in self.validation_results.values() if result['valid'])
        invalid_skills = total_skills - valid_skills
        
        return {
            'total_skills': total_skills,
            'valid_skills': valid_skills,
            'invalid_skills': invalid_skills,
            'validation_rate': valid_skills / total_skills if total_skills > 0 else 0
        }
