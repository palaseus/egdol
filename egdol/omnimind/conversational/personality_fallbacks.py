"""
Personality-Specific Fallback Reasoning
Implements deterministic fallback routines for each personality when
real reasoning data isn't available or confidence is low.
"""

import re
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .personality_framework import Personality, PersonalityType


@dataclass
class FallbackResponse:
    """Structured fallback response."""
    content: str
    reasoning_type: str
    confidence: float
    personality: str
    fallback_type: str
    metadata: Dict[str, Any]


class PersonalityFallbackReasoner:
    """Handles personality-specific fallback reasoning."""
    
    def __init__(self):
        self._initialize_fallback_data()
        self._initialize_math_operations()
        self._initialize_logic_patterns()
    
    def _initialize_fallback_data(self):
        """Initialize fallback data for each personality."""
        self.fallback_data = {
            'Strategos': {
                'tactical_quotes': [
                    "The best defense is a good offense.",
                    "Know your enemy and know yourself, and you can fight a hundred battles without disaster.",
                    "Strategy without tactics is the slowest route to victory.",
                    "The supreme art of war is to subdue the enemy without fighting.",
                    "In war, the way is to avoid what is strong and to strike at what is weak."
                ],
                'tactical_responses': [
                    "Commander, I recommend a systematic approach to this situation.",
                    "From a tactical perspective, we must consider all variables.",
                    "The strategic advantage lies in careful planning and execution.",
                    "I suggest we analyze the situation from multiple angles.",
                    "Tactical superiority requires understanding the terrain."
                ],
                'math_responses': [
                    "Commander, I've calculated the tactical parameters.",
                    "From a strategic standpoint, the numerical analysis shows...",
                    "The tactical calculations indicate...",
                    "Based on military mathematics...",
                    "The strategic equation yields..."
                ]
            },
            'Archivist': {
                'historical_quotes': [
                    "Those who cannot remember the past are condemned to repeat it.",
                    "History is written by the victors.",
                    "The past is never dead. It's not even past.",
                    "History repeats itself, first as tragedy, second as farce.",
                    "In history, a great volume is unrolled for our instruction."
                ],
                'scholarly_responses': [
                    "From the archives of knowledge, I can share that...",
                    "The historical record shows that...",
                    "Through the lens of time, we can see that...",
                    "The chronicles reveal that...",
                    "From the depths of historical wisdom..."
                ],
                'research_responses': [
                    "I must consult the archives for more information.",
                    "The historical sources suggest...",
                    "Based on scholarly research...",
                    "The academic literature indicates...",
                    "From my studies of the past..."
                ]
            },
            'Lawmaker': {
                'legal_quotes': [
                    "The law is the last result of human wisdom.",
                    "Justice is truth in action.",
                    "The law must be stable, but it must not stand still.",
                    "Equal justice under law is not merely a caption on the facade of the Supreme Court building.",
                    "The law is a sort of hocus-pocus science."
                ],
                'legal_responses': [
                    "According to the principles of governance...",
                    "From a legal perspective, the fundamental rule is...",
                    "The structural framework requires...",
                    "Based on the rule of law...",
                    "The legal precedent suggests..."
                ],
                'meta_rule_responses': [
                    "The fundamental principle governing this situation is...",
                    "The meta-rule that applies here is...",
                    "The underlying structure follows the principle of...",
                    "The universal law that governs this is...",
                    "The foundational rule states that..."
                ]
            },
            'Oracle': {
                'cosmic_quotes': [
                    "The universe is not only queerer than we suppose, but queerer than we can suppose.",
                    "We are stardust brought to life, then empowered by the universe to figure itself out.",
                    "The cosmos is within us. We are made of star-stuff.",
                    "In the cosmic dance of reality, all things are connected.",
                    "The universe is a vast cosmic tapestry of infinite possibilities."
                ],
                'mystical_responses': [
                    "In the cosmic dance of reality, I perceive...",
                    "From the depths of universal wisdom...",
                    "Through the veil of existence, I see that...",
                    "In the tapestry of the multiverse...",
                    "From the perspective of the infinite..."
                ],
                'comparative_responses': [
                    "Across the cosmic tapestry, the pattern reveals...",
                    "In the grand scheme of universal truth...",
                    "The cosmic patterns suggest...",
                    "From a universal perspective...",
                    "The cosmic dance shows that..."
                ]
            }
        }
    
    def _initialize_math_operations(self):
        """Initialize mathematical operation handlers."""
        self.math_operations = {
            'add': lambda a, b: a + b,
            'subtract': lambda a, b: a - b,
            'multiply': lambda a, b: a * b,
            'divide': lambda a, b: a / b if b != 0 else float('inf'),
            'power': lambda a, b: a ** b
        }
    
    def _initialize_logic_patterns(self):
        """Initialize logical reasoning patterns."""
        self.logic_patterns = {
            'syllogism': r'if\s+(\w+)\s+then\s+(\w+).*(\w+)\s+is\s+(\w+)',
            'conditional': r'if\s+(.+)\s+then\s+(.+)',
            'negation': r'not\s+(.+)',
            'conjunction': r'(.+)\s+and\s+(.+)',
            'disjunction': r'(.+)\s+or\s+(.+)'
        }
    
    def get_fallback_response(self, user_input: str, personality: str, 
                            reasoning_type: str, context: Dict[str, Any]) -> FallbackResponse:
        """
        Get personality-specific fallback response.
        
        Args:
            user_input: User's input message
            personality: Current personality
            reasoning_type: Type of reasoning required
            context: Additional context information
            
        Returns:
            FallbackResponse with personality-specific content
        """
        # Determine fallback type
        fallback_type = self._determine_fallback_type(user_input, personality, reasoning_type)
        
        # Generate response based on personality and fallback type
        if personality == 'Strategos':
            response = self._get_strategos_fallback(user_input, fallback_type, context)
        elif personality == 'Archivist':
            response = self._get_archivist_fallback(user_input, fallback_type, context)
        elif personality == 'Lawmaker':
            response = self._get_lawmaker_fallback(user_input, fallback_type, context)
        elif personality == 'Oracle':
            response = self._get_oracle_fallback(user_input, fallback_type, context)
        else:
            response = self._get_general_fallback(user_input, fallback_type, context)
        
        return FallbackResponse(
            content=response,
            reasoning_type=reasoning_type,
            confidence=0.7,  # Fallback responses have moderate confidence
            personality=personality,
            fallback_type=fallback_type,
            metadata={
                'fallback_used': True,
                'original_input': user_input,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _determine_fallback_type(self, user_input: str, personality: str, reasoning_type: str) -> str:
        """Determine the type of fallback reasoning needed."""
        input_lower = user_input.lower()
        
        # Check for mathematical content
        if any(char in input_lower for char in ['+', '-', '*', '/', '=', 'calculate', 'compute', 'solve']):
            return 'mathematical'
        
        # Check for logical content
        if any(word in input_lower for word in ['if', 'then', 'because', 'therefore', 'all', 'some', 'implies']):
            return 'logical'
        
        # Check for greeting
        if any(word in input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
        
        # Check for question
        if '?' in user_input or any(word in input_lower for word in ['what', 'who', 'when', 'where', 'why', 'how']):
            return 'question'
        
        # Default based on personality
        if personality == 'Strategos':
            return 'tactical'
        elif personality == 'Archivist':
            return 'historical'
        elif personality == 'Lawmaker':
            return 'structural'
        elif personality == 'Oracle':
            return 'comparative'
        
        return 'general'
    
    def _get_strategos_fallback(self, user_input: str, fallback_type: str, context: Dict[str, Any]) -> str:
        """Get Strategos-specific fallback response."""
        if fallback_type == 'mathematical':
            return self._handle_math_fallback(user_input, 'Strategos')
        elif fallback_type == 'logical':
            return self._handle_logic_fallback(user_input, 'Strategos')
        elif fallback_type == 'greeting':
            return "Commander, I am ready for your tactical briefing. What is the situation?"
        elif fallback_type == 'question':
            return random.choice(self.fallback_data['Strategos']['tactical_responses'])
        else:
            return random.choice(self.fallback_data['Strategos']['tactical_quotes'])
    
    def _get_archivist_fallback(self, user_input: str, fallback_type: str, context: Dict[str, Any]) -> str:
        """Get Archivist-specific fallback response."""
        if fallback_type == 'mathematical':
            return self._handle_math_fallback(user_input, 'Archivist')
        elif fallback_type == 'logical':
            return self._handle_logic_fallback(user_input, 'Archivist')
        elif fallback_type == 'greeting':
            return "From the archives of knowledge, I greet you. What wisdom do you seek?"
        elif fallback_type == 'question':
            return random.choice(self.fallback_data['Archivist']['scholarly_responses'])
        else:
            return random.choice(self.fallback_data['Archivist']['historical_quotes'])
    
    def _get_lawmaker_fallback(self, user_input: str, fallback_type: str, context: Dict[str, Any]) -> str:
        """Get Lawmaker-specific fallback response."""
        if fallback_type == 'mathematical':
            return self._handle_math_fallback(user_input, 'Lawmaker')
        elif fallback_type == 'logical':
            return self._handle_logic_fallback(user_input, 'Lawmaker')
        elif fallback_type == 'greeting':
            return "In accordance with the principles of governance, I am at your service. What legal matter requires attention?"
        elif fallback_type == 'question':
            return random.choice(self.fallback_data['Lawmaker']['legal_responses'])
        else:
            return random.choice(self.fallback_data['Lawmaker']['legal_quotes'])
    
    def _get_oracle_fallback(self, user_input: str, fallback_type: str, context: Dict[str, Any]) -> str:
        """Get Oracle-specific fallback response."""
        if fallback_type == 'mathematical':
            return self._handle_math_fallback(user_input, 'Oracle')
        elif fallback_type == 'logical':
            return self._handle_logic_fallback(user_input, 'Oracle')
        elif fallback_type == 'greeting':
            return "In the cosmic dance of reality, I greet you. What universal truth do you seek?"
        elif fallback_type == 'question':
            return random.choice(self.fallback_data['Oracle']['mystical_responses'])
        else:
            return random.choice(self.fallback_data['Oracle']['cosmic_quotes'])
    
    def _get_general_fallback(self, user_input: str, fallback_type: str, context: Dict[str, Any]) -> str:
        """Get general fallback response."""
        if fallback_type == 'mathematical':
            return self._handle_math_fallback(user_input, 'General')
        elif fallback_type == 'logical':
            return self._handle_logic_fallback(user_input, 'General')
        elif fallback_type == 'greeting':
            return "Hello! I'm here to help with your inquiry."
        else:
            return "I understand your question. Let me provide some assistance with that."
    
    def _handle_math_fallback(self, user_input: str, personality: str) -> str:
        """Handle mathematical fallback reasoning."""
        # Extract numbers and operations
        numbers = re.findall(r'\d+(?:\.\d+)?', user_input)
        operations = re.findall(r'[\+\-\*\/]', user_input)
        
        if len(numbers) >= 2 and operations:
            try:
                a, b = float(numbers[0]), float(numbers[1])
                op = operations[0]
                
                if op == '+':
                    result = a + b
                    op_text = 'plus'
                elif op == '-':
                    result = a - b
                    op_text = 'minus'
                elif op == '*':
                    result = a * b
                    op_text = 'times'
                elif op == '/':
                    result = a / b if b != 0 else float('inf')
                    op_text = 'divided by'
                else:
                    return self._get_personality_math_response(personality, "I can help with basic arithmetic.")
                
                return self._get_personality_math_response(personality, f"{a} {op_text} {b} equals {result}")
                
            except (ValueError, ZeroDivisionError):
                return self._get_personality_math_response(personality, "I can help with basic arithmetic.")
        
        return self._get_personality_math_response(personality, "I can help with mathematical calculations.")
    
    def _handle_logic_fallback(self, user_input: str, personality: str) -> str:
        """Handle logical fallback reasoning."""
        # Check for syllogism pattern
        syllogism_match = re.search(self.logic_patterns['syllogism'], user_input.lower())
        if syllogism_match:
            return self._get_personality_logic_response(personality, "This follows logical reasoning principles.")
        
        # Check for conditional
        conditional_match = re.search(self.logic_patterns['conditional'], user_input.lower())
        if conditional_match:
            return self._get_personality_logic_response(personality, "This is a conditional statement that follows logical rules.")
        
        return self._get_personality_logic_response(personality, "I can help analyze logical reasoning.")
    
    def _get_personality_math_response(self, personality: str, base_response: str) -> str:
        """Get personality-specific math response."""
        if personality == 'Strategos':
            return f"Commander, {base_response.lower()}"
        elif personality == 'Archivist':
            return f"From the archives, {base_response.lower()}"
        elif personality == 'Lawmaker':
            return f"According to mathematical principles, {base_response.lower()}"
        elif personality == 'Oracle':
            return f"In the cosmic mathematics, {base_response.lower()}"
        else:
            return base_response
    
    def _get_personality_logic_response(self, personality: str, base_response: str) -> str:
        """Get personality-specific logic response."""
        if personality == 'Strategos':
            return f"Commander, {base_response.lower()}"
        elif personality == 'Archivist':
            return f"From the scholarly perspective, {base_response.lower()}"
        elif personality == 'Lawmaker':
            return f"According to logical principles, {base_response.lower()}"
        elif personality == 'Oracle':
            return f"In the universal logic, {base_response.lower()}"
        else:
            return base_response
    
    def should_use_fallback(self, confidence: float, reasoning_type: str, context: Dict[str, Any]) -> bool:
        """Determine if fallback reasoning should be used."""
        # Always use fallback for basic reasoning with low confidence
        if reasoning_type == 'basic_reasoning' and confidence < 0.6:
            return True
        
        # Use fallback for greetings and simple questions
        if context.get('intent_type') in ['QUESTION', 'REQUEST'] and confidence < 0.5:
            return True
        
        # Use fallback if reasoning engine is unavailable
        if context.get('reasoning_engine_available', True) == False:
            return True
        
        return False
