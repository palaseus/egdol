"""
Logic Skill for OmniMind
Handles logical reasoning and forwards complex queries to Egdol.
"""

import re
from typing import Dict, Any, Optional, List
from .base import BaseSkill


class LogicSkill(BaseSkill):
    """Handles logical reasoning and Egdol integration."""
    
    def __init__(self):
        super().__init__()
        self.description = "Handles logical reasoning and forwards to Egdol"
        self.capabilities = [
            "logical reasoning",
            "fact checking",
            "rule evaluation",
            "inference",
            "explanation"
        ]
        
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        """Check if this is a logic-related input."""
        user_input_lower = user_input.lower()
        
        # Logic keywords
        logic_keywords = [
            'if', 'then', 'because', 'therefore', 'since', 'given that',
            'logical', 'reasoning', 'inference', 'deduce', 'conclude',
            'premise', 'conclusion', 'argument', 'proof'
        ]
        
        # Question patterns
        question_patterns = [
            r'why\s+',
            r'how\s+do\s+you\s+know',
            r'what\s+is\s+the\s+reason',
            r'explain\s+why',
            r'prove\s+that'
        ]
        
        # Check for logic keywords
        if any(keyword in user_input_lower for keyword in logic_keywords):
            return True
            
        # Check for question patterns
        if any(re.search(pattern, user_input_lower) for pattern in question_patterns):
            return True
            
        # Check for rule patterns
        if 'if' in user_input_lower and 'then' in user_input_lower:
            return True
            
        return False
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle logical reasoning input."""
        try:
            # Get Egdol engine from context
            engine = context.get('engine')
            if not engine:
                return {
                    'content': "I need access to the reasoning engine to handle logical queries.",
                    'reasoning': ['No reasoning engine available'],
                    'metadata': {'skill': 'logic', 'success': False}
                }
                
            # Try to process with Egdol
            from ..core import OmniMind
            dsl = SimpleDSL(engine)
            
            # Convert to DSL if possible
            dsl_query = self._convert_to_dsl(user_input)
            if dsl_query:
                result = dsl.execute(dsl_query)
                
                if result.get('type') == 'query' and result.get('results'):
                    content = self._format_query_results(result['results'])
                    reasoning = [f"Egdol query: {dsl_query}", f"Found {len(result['results'])} results"]
                elif result.get('type') == 'fact':
                    content = f"I've learned: {result['description']}"
                    reasoning = [f"Added fact: {result['description']}"]
                elif result.get('type') == 'rule':
                    content = f"I've added the rule: {result['description']}"
                    reasoning = [f"Added rule: {result['description']}"]
                else:
                    content = "I couldn't process that logical statement. Could you rephrase it?"
                    reasoning = [f"Failed to process: {dsl_query}"]
            else:
                # Fallback to general logical analysis
                content = self._analyze_logical_structure(user_input)
                reasoning = ["Analyzed logical structure"]
                
            return {
                'content': content,
                'reasoning': reasoning,
                'metadata': {'skill': 'logic', 'dsl_query': dsl_query}
            }
            
        except Exception as e:
            return {
                'content': f"I encountered an error with the logical reasoning: {str(e)}",
                'reasoning': [f"Error: {str(e)}"],
                'metadata': {'skill': 'logic', 'error': str(e)}
            }
            
    def _convert_to_dsl(self, user_input: str) -> Optional[str]:
        """Convert user input to DSL format."""
        user_input_lower = user_input.lower().strip()
        
        # Question patterns
        if user_input_lower.startswith('who is'):
            subject = user_input_lower.replace('who is', '').strip().rstrip('?')
            return f"who is {subject}?"
        elif user_input_lower.startswith('what is'):
            subject = user_input_lower.replace('what is', '').strip().rstrip('?')
            return f"what is {subject}?"
        elif user_input_lower.startswith('is '):
            parts = user_input_lower.replace('is ', '').strip().rstrip('?').split()
            if len(parts) >= 2:
                subject = parts[0]
                predicate = ' '.join(parts[1:])
                return f"is {subject} {predicate}?"
                
        # Fact patterns
        if ' is ' in user_input_lower and not '?' in user_input:
            return user_input  # Pass through as-is
            
        # Rule patterns
        if 'if' in user_input_lower and 'then' in user_input_lower:
            return user_input  # Pass through as-is
            
        return None
        
    def _format_query_results(self, results: List[Dict[str, Any]]) -> str:
        """Format query results for display."""
        if not results:
            return "No results found."
            
        formatted = []
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                parts = []
                for var, value in result.items():
                    parts.append(f"{var}={value}")
                formatted.append(f"{i}. {', '.join(parts)}")
            else:
                formatted.append(f"{i}. {result}")
                
        return "Based on my knowledge:\n" + "\n".join(formatted)
        
    def _analyze_logical_structure(self, user_input: str) -> str:
        """Analyze the logical structure of the input."""
        user_input_lower = user_input.lower()
        
        # Check for logical operators
        if 'and' in user_input_lower:
            return "This appears to be a conjunction (AND statement)."
        elif 'or' in user_input_lower:
            return "This appears to be a disjunction (OR statement)."
        elif 'if' in user_input_lower and 'then' in user_input_lower:
            return "This appears to be a conditional statement (IF-THEN)."
        elif 'not' in user_input_lower:
            return "This appears to be a negation (NOT statement)."
        elif 'because' in user_input_lower or 'since' in user_input_lower:
            return "This appears to be a causal statement."
        else:
            return "This appears to be a simple statement."
