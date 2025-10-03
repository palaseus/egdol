"""
NLU to DSL Translator for OmniMind
Converts natural language input into Egdol DSL queries and assertions.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from ..dsl.simple_dsl import SimpleDSL


class NLUTranslator:
    """Translates natural language to Egdol DSL."""
    
    def __init__(self):
        self.patterns = {
            'factual_questions': [
                r'who is (.+)\?',
                r'what is (.+)\?',
                r'where is (.+)\?',
                r'when is (.+)\?',
                r'how is (.+)\?'
            ],
            'yes_no_questions': [
                r'is (.+) (.+)\?',
                r'does (.+) (.+)\?',
                r'can (.+) (.+)\?',
                r'will (.+) (.+)\?'
            ],
            'fact_assertions': [
                r'(.+) is (.+)',
                r'(.+) are (.+)',
                r'(.+) has (.+)',
                r'(.+) have (.+)'
            ],
            'rule_definitions': [
                r'if (.+) then (.+)',
                r'when (.+) then (.+)',
                r'(.+) if (.+)'
            ],
            'commands': [
                r'tell me about (.+)',
                r'explain (.+)',
                r'what do you know about (.+)',
                r'show me (.+)'
            ]
        }
        
    def translate(self, user_input: str) -> Optional[str]:
        """Translate natural language input to DSL."""
        user_input = user_input.strip()
        
        # Try each pattern category
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.match(pattern, user_input, re.IGNORECASE)
                if match:
                    return self._translate_match(category, pattern, match, user_input)
                    
        # Fallback to fuzzy matching
        return self._fuzzy_translate(user_input)
        
    def _translate_match(self, category: str, pattern: str, match: re.Match, user_input: str) -> str:
        """Translate a matched pattern to DSL."""
        groups = match.groups()
        
        if category == 'factual_questions':
            if pattern.startswith(r'who is'):
                return f"who is {groups[0]}?"
            elif pattern.startswith(r'what is'):
                return f"what is {groups[0]}?"
            elif pattern.startswith(r'where is'):
                return f"where is {groups[0]}?"
            elif pattern.startswith(r'when is'):
                return f"when is {groups[0]}?"
            elif pattern.startswith(r'how is'):
                return f"how is {groups[0]}?"
                
        elif category == 'yes_no_questions':
            if pattern.startswith(r'is '):
                return f"is {groups[0]} {groups[1]}?"
            elif pattern.startswith(r'does '):
                return f"does {groups[0]} {groups[1]}?"
            elif pattern.startswith(r'can '):
                return f"can {groups[0]} {groups[1]}?"
            elif pattern.startswith(r'will '):
                return f"will {groups[0]} {groups[1]}?"
                
        elif category == 'fact_assertions':
            if pattern.startswith(r'(.+) is '):
                return f"{groups[0]} is {groups[1]}"
            elif pattern.startswith(r'(.+) are '):
                return f"{groups[0]} are {groups[1]}"
            elif pattern.startswith(r'(.+) has '):
                return f"{groups[0]} has {groups[1]}"
            elif pattern.startswith(r'(.+) have '):
                return f"{groups[0]} have {groups[1]}"
                
        elif category == 'rule_definitions':
            if pattern.startswith(r'if '):
                return f"if {groups[0]} then {groups[1]}"
            elif pattern.startswith(r'when '):
                return f"if {groups[0]} then {groups[1]}"
            elif pattern.endswith(r' if '):
                return f"if {groups[1]} then {groups[0]}"
                
        elif category == 'commands':
            if pattern.startswith(r'tell me about'):
                return f"what is {groups[0]}?"
            elif pattern.startswith(r'explain'):
                return f"what is {groups[0]}?"
            elif pattern.startswith(r'what do you know about'):
                return f"what is {groups[0]}?"
            elif pattern.startswith(r'show me'):
                return f"show me {groups[0]}"
                
        return user_input
        
    def _fuzzy_translate(self, user_input: str) -> Optional[str]:
        """Fuzzy translation for unknown patterns."""
        user_input_lower = user_input.lower()
        
        # Look for question words
        if any(word in user_input_lower for word in ['who', 'what', 'where', 'when', 'why', 'how']):
            if '?' in user_input:
                return user_input  # Pass through as-is
            else:
                return user_input + '?'
                
        # Look for assertion patterns
        if any(word in user_input_lower for word in ['is', 'are', 'has', 'have']):
            return user_input
            
        # Look for rule patterns
        if 'if' in user_input_lower and 'then' in user_input_lower:
            return user_input
            
        # Default: treat as question
        if not user_input.endswith('?'):
            return user_input + '?'
            
        return user_input
        
    def extract_entities(self, user_input: str) -> List[Tuple[str, str]]:
        """Extract entities from user input."""
        entities = []
        
        # Extract proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', user_input)
        for noun in proper_nouns:
            entities.append(('proper_noun', noun))
            
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', user_input)
        for number in numbers:
            entities.append(('number', number))
            
        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', user_input)
        for quote in quoted:
            entities.append(('quoted_string', quote))
            
        return entities
        
    def detect_intent(self, user_input: str) -> str:
        """Detect user intent from input."""
        user_input_lower = user_input.lower().strip()
        
        # Question patterns
        if any(word in user_input_lower for word in ['who', 'what', 'where', 'when', 'why', 'how']):
            if '?' in user_input:
                return 'question'
                
        # Command patterns
        if user_input_lower.startswith('tell me') or user_input_lower.startswith('explain'):
            return 'explanation_request'
            
        # Fact patterns
        if any(word in user_input_lower for word in ['is', 'are', 'has', 'have']) and not '?' in user_input:
            return 'fact_assertion'
            
        # Rule patterns
        if 'if' in user_input_lower and 'then' in user_input_lower:
            return 'rule_definition'
            
        # Default
        return 'general'
        
    def get_translation_confidence(self, user_input: str, dsl_output: str) -> float:
        """Get confidence score for translation."""
        # Simple confidence based on pattern matching
        if any(pattern in user_input.lower() for pattern in ['who is', 'what is', 'is ', 'if ']):
            return 0.9
        elif '?' in user_input:
            return 0.7
        else:
            return 0.5
            
    def validate_dsl(self, dsl_output: str) -> bool:
        """Validate that DSL output is syntactically correct."""
        try:
            # Basic validation - check for common DSL patterns
            if dsl_output.endswith('?'):
                # Question format
                return True
            elif ' is ' in dsl_output or ' are ' in dsl_output:
                # Fact format
                return True
            elif 'if ' in dsl_output and ' then ' in dsl_output:
                # Rule format
                return True
            else:
                return False
        except Exception:
            return False
