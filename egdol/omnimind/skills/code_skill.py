"""
Code Skill for OmniMind
Handles code analysis, generation, and execution.
"""

import re
import ast
import subprocess
import tempfile
import os
from typing import Dict, Any, List, Optional
from .base import BaseSkill


class CodeSkill(BaseSkill):
    """Handles code-related tasks."""
    
    def __init__(self):
        super().__init__()
        self.description = "Handles code analysis, generation, and execution"
        self.capabilities = [
            "code analysis",
            "code generation",
            "syntax checking",
            "code explanation",
            "simple execution"
        ]
        
        # Supported languages
        self.supported_languages = ['python', 'javascript', 'java', 'cpp', 'c', 'html', 'css']
        
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        """Check if this is a code-related input."""
        user_input_lower = user_input.lower()
        
        # Code keywords
        code_keywords = [
            'code', 'program', 'function', 'class', 'method', 'variable',
            'write', 'create', 'generate', 'implement', 'debug', 'fix',
            'syntax', 'error', 'compile', 'run', 'execute'
        ]
        
        # Language keywords
        language_keywords = [
            'python', 'javascript', 'java', 'cpp', 'c++', 'html', 'css',
            'def ', 'class ', 'function ', 'var ', 'let ', 'const '
        ]
        
        # Code patterns
        code_patterns = [
            r'def\s+\w+\s*\(',  # Python function
            r'class\s+\w+',     # Class definition
            r'function\s+\w+',  # JavaScript function
            r'<[^>]+>',         # HTML tags
            r'import\s+\w+',    # Import statements
        ]
        
        # Check for code keywords
        if any(keyword in user_input_lower for keyword in code_keywords):
            return True
            
        # Check for language keywords
        if any(keyword in user_input_lower for keyword in language_keywords):
            return True
            
        # Check for code patterns
        if any(re.search(pattern, user_input) for pattern in code_patterns):
            return True
            
        return False
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code-related input."""
        try:
            # Detect code type
            code_type = self._detect_code_type(user_input)
            
            if code_type == 'analysis':
                return self._analyze_code(user_input)
            elif code_type == 'generation':
                return self._generate_code(user_input)
            elif code_type == 'execution':
                return self._execute_code(user_input)
            elif code_type == 'explanation':
                return self._explain_code(user_input)
            else:
                return self._general_code_response(user_input)
                
        except Exception as e:
            return {
                'content': f"I encountered an error with the code: {str(e)}",
                'reasoning': [f"Error: {str(e)}"],
                'metadata': {'skill': 'code', 'error': str(e)}
            }
            
    def _detect_code_type(self, user_input: str) -> str:
        """Detect the type of code request."""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['analyze', 'check', 'review', 'examine']):
            return 'analysis'
        elif any(word in user_input_lower for word in ['write', 'create', 'generate', 'implement']):
            return 'generation'
        elif any(word in user_input_lower for word in ['run', 'execute', 'compile']):
            return 'execution'
        elif any(word in user_input_lower for word in ['explain', 'what does', 'how does']):
            return 'explanation'
        else:
            return 'general'
            
    def _analyze_code(self, user_input: str) -> Dict[str, Any]:
        """Analyze code for issues."""
        # Extract code from input
        code = self._extract_code(user_input)
        if not code:
            return {
                'content': "I couldn't find any code to analyze. Please provide code in your message.",
                'reasoning': ['No code detected'],
                'metadata': {'skill': 'code', 'type': 'analysis'}
            }
            
        # Basic syntax analysis
        issues = self._check_syntax(code)
        
        if issues:
            content = f"Code analysis found {len(issues)} issue(s):\n" + "\n".join(f"- {issue}" for issue in issues)
        else:
            content = "Code analysis: No syntax errors found. The code appears to be syntactically correct."
            
        return {
            'content': content,
            'reasoning': ['Analyzed code syntax'],
            'metadata': {'skill': 'code', 'type': 'analysis', 'issues': len(issues)}
        }
        
    def _generate_code(self, user_input: str) -> Dict[str, Any]:
        """Generate code based on requirements."""
        # Extract requirements
        requirements = self._extract_requirements(user_input)
        
        # Generate simple code examples
        if 'python' in user_input.lower() or 'def ' in user_input:
            code = self._generate_python_code(requirements)
        elif 'javascript' in user_input.lower() or 'function ' in user_input:
            code = self._generate_javascript_code(requirements)
        else:
            code = self._generate_generic_code(requirements)
            
        return {
            'content': f"Here's some code that might help:\n\n```\n{code}\n```",
            'reasoning': ['Generated code based on requirements'],
            'metadata': {'skill': 'code', 'type': 'generation', 'language': 'python'}
        }
        
    def _execute_code(self, user_input: str) -> Dict[str, Any]:
        """Execute code safely."""
        code = self._extract_code(user_input)
        if not code:
            return {
                'content': "I couldn't find any code to execute. Please provide executable code.",
                'reasoning': ['No code detected'],
                'metadata': {'skill': 'code', 'type': 'execution'}
            }
            
        # Only execute Python code for safety
        if self._is_python_code(code):
            result = self._safe_execute_python(code)
            return {
                'content': f"Code execution result:\n{result}",
                'reasoning': ['Executed Python code safely'],
                'metadata': {'skill': 'code', 'type': 'execution', 'result': result}
            }
        else:
            return {
                'content': "I can only execute Python code safely. For other languages, please use appropriate compilers/interpreters.",
                'reasoning': ['Unsupported language for execution'],
                'metadata': {'skill': 'code', 'type': 'execution'}
            }
            
    def _explain_code(self, user_input: str) -> Dict[str, Any]:
        """Explain code functionality."""
        code = self._extract_code(user_input)
        if not code:
            return {
                'content': "I couldn't find any code to explain. Please provide code in your message.",
                'reasoning': ['No code detected'],
                'metadata': {'skill': 'code', 'type': 'explanation'}
            }
            
        explanation = self._generate_code_explanation(code)
        
        return {
            'content': f"Code explanation:\n{explanation}",
            'reasoning': ['Explained code functionality'],
            'metadata': {'skill': 'code', 'type': 'explanation'}
        }
        
    def _general_code_response(self, user_input: str) -> Dict[str, Any]:
        """General code response."""
        return {
            'content': "I can help you with code analysis, generation, and execution. What specific coding task would you like help with?",
            'reasoning': ['General code response'],
            'metadata': {'skill': 'code', 'type': 'general'}
        }
        
    def _extract_code(self, user_input: str) -> Optional[str]:
        """Extract code from user input."""
        # Look for code blocks
        code_block_match = re.search(r'```(?:python|javascript|java|cpp|c\+\+|html|css)?\n(.*?)\n```', user_input, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
            
        # Look for inline code
        inline_code_match = re.search(r'`([^`]+)`', user_input)
        if inline_code_match:
            return inline_code_match.group(1).strip()
            
        # Look for code patterns
        if any(pattern in user_input for pattern in ['def ', 'class ', 'function ', 'import ']):
            return user_input.strip()
            
        return None
        
    def _extract_requirements(self, user_input: str) -> str:
        """Extract requirements from user input."""
        # Remove common words
        cleaned = re.sub(r'\b(write|create|generate|implement|code|program)\b', '', user_input, flags=re.IGNORECASE)
        return cleaned.strip()
        
    def _check_syntax(self, code: str) -> List[str]:
        """Check code syntax for issues."""
        issues = []
        
        try:
            # Try to parse as Python
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except Exception as e:
            issues.append(f"Parse error: {e}")
            
        return issues
        
    def _is_python_code(self, code: str) -> bool:
        """Check if code is Python."""
        python_keywords = ['def ', 'class ', 'import ', 'from ', 'if __name__']
        return any(keyword in code for keyword in python_keywords)
        
    def _safe_execute_python(self, code: str) -> str:
        """Safely execute Python code."""
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                }
            }
            
            # Execute in a restricted environment
            exec(code, safe_globals)
            return "Code executed successfully"
            
        except Exception as e:
            return f"Execution error: {str(e)}"
            
    def _generate_python_code(self, requirements: str) -> str:
        """Generate Python code based on requirements."""
        if 'function' in requirements.lower() or 'def' in requirements.lower():
            return """def example_function():
    \"\"\"Example function\"\"\"
    return "Hello, World!"

# Usage
result = example_function()
print(result)"""
        else:
            return """# Example Python code
print("Hello, World!")

# Variables
name = "OmniMind"
age = 1

# Simple calculation
result = age * 2
print(f"Result: {result}")"""
            
    def _generate_javascript_code(self, requirements: str) -> str:
        """Generate JavaScript code based on requirements."""
        return """// Example JavaScript code
function exampleFunction() {
    return "Hello, World!";
}

// Usage
const result = exampleFunction();
console.log(result);

// Variables
const name = "OmniMind";
let age = 1;

// Simple calculation
const result2 = age * 2;
console.log(`Result: ${result2}`);"""
        
    def _generate_generic_code(self, requirements: str) -> str:
        """Generate generic code."""
        return """# Generic code example
# This is a simple example

def main():
    print("Hello, World!")
    
if __name__ == "__main__":
    main()"""
        
    def _generate_code_explanation(self, code: str) -> str:
        """Generate explanation for code."""
        lines = code.split('\n')
        explanation = []
        
        for i, line in enumerate(lines, 1):
            if line.strip():
                if line.strip().startswith('def '):
                    explanation.append(f"Line {i}: Defines a function")
                elif line.strip().startswith('class '):
                    explanation.append(f"Line {i}: Defines a class")
                elif line.strip().startswith('import '):
                    explanation.append(f"Line {i}: Imports a module")
                elif line.strip().startswith('if '):
                    explanation.append(f"Line {i}: Conditional statement")
                elif line.strip().startswith('for '):
                    explanation.append(f"Line {i}: Loop statement")
                elif line.strip().startswith('print('):
                    explanation.append(f"Line {i}: Prints output")
                else:
                    explanation.append(f"Line {i}: {line.strip()}")
                    
        return "\n".join(explanation)
