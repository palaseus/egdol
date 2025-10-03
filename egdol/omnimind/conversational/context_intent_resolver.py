"""
Context-Intent Reasoning Stabilization Layer
Pre-processes user inputs, detects intent type, ensures required fields exist,
and initializes default reasoning contexts if missing.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .intent_parser import IntentParser, Intent, IntentType
from .personality_framework import Personality, PersonalityType


@dataclass
class ReasoningContext:
    """Normalized reasoning context with all required fields."""
    reasoning_type: str
    intent_type: str
    personality: str
    domain: str
    complexity_level: float
    confidence: float
    entities: List[str] = field(default_factory=list)
    context_clues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reasoning engine."""
        return {
            'reasoning_type': self.reasoning_type,
            'intent_type': self.intent_type,
            'personality': self.personality,
            'domain': self.domain,
            'complexity_level': self.complexity_level,
            'confidence': self.confidence,
            'entities': self.entities,
            'context_clues': self.context_clues,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class ContextIntentResolver:
    """Resolves intent and context, ensuring all required fields are present."""
    
    def __init__(self):
        self.intent_parser = IntentParser()
        self._initialize_fallback_patterns()
        self._initialize_domain_mappings()
    
    def _initialize_fallback_patterns(self):
        """Initialize patterns for fallback reasoning."""
        self.fallback_patterns = {
            'greeting': [
                r'\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b',
                r'\b(how are you|how do you do|what\'s up)\b'
            ],
            'math': [
                r'\b(calculate|compute|solve|what is|what\'s)\b.*\b(\d+|\+|\-|\*|\/|\=)',
                r'\b(\d+)\s*[\+\-\*\/]\s*(\d+)',
                r'\b(plus|minus|times|divided by|equals)\b'
            ],
            'logic': [
                r'\b(if|then|because|therefore|since|given that)\b',
                r'\b(all|some|none|every|each)\b',
                r'\b(implies|follows|concludes)\b'
            ],
            'question': [
                r'\b(what|who|when|where|why|how|which|can|could|would|should)\b',
                r'\?'
            ],
            'command': [
                r'\b(do|make|create|build|generate|show|tell|explain)\b'
            ]
        }
    
    def _initialize_domain_mappings(self):
        """Initialize domain mappings for reasoning types."""
        self.domain_mappings = {
            'mathematical': 'basic_reasoning',
            'logical': 'basic_reasoning',
            'philosophical': 'civilizational_reasoning',
            'historical': 'civilizational_reasoning',
            'strategic': 'strategic_reasoning',
            'governance': 'meta_rule_reasoning',
            'cosmic': 'universe_reasoning',
            'general': 'basic_reasoning'
        }
    
    def resolve_intent_and_context(self, user_input: str, personality: str) -> ReasoningContext:
        """
        Resolve intent and context, ensuring all required fields are present.
        
        Args:
            user_input: User's input message
            personality: Current active personality
            
        Returns:
            ReasoningContext with all required fields normalized
        """
        # Parse intent
        intent = self.intent_parser.parse(user_input)
        
        # Determine reasoning type based on intent and personality
        reasoning_type = self._determine_reasoning_type(intent, personality)
        
        # Determine domain
        domain = self._determine_domain(intent, personality)
        
        # Ensure confidence is valid
        confidence = max(0.1, min(1.0, intent.confidence))
        
        # Create normalized context
        context = ReasoningContext(
            reasoning_type=reasoning_type,
            intent_type=intent.intent_type.name,
            personality=personality,
            domain=domain,
            complexity_level=intent.complexity_level,
            confidence=confidence,
            entities=intent.entities,
            context_clues=intent.context_clues,
            metadata={
                'original_input': user_input,
                'personality_hint': intent.personality_hint,
                'urgency': intent.urgency,
                'emotional_tone': intent.emotional_tone,
                'fallback_required': confidence < 0.5
            }
        )
        
        return context
    
    def _determine_reasoning_type(self, intent: Intent, personality: str) -> str:
        """Determine reasoning type based on intent and personality."""
        # Check for specific reasoning types based on intent
        if intent.intent_type == IntentType.CIVILIZATIONAL_QUERY:
            return 'civilizational_reasoning'
        elif intent.intent_type == IntentType.STRATEGIC_ANALYSIS:
            return 'strategic_reasoning'
        elif intent.intent_type == IntentType.META_RULE_DISCOVERY:
            return 'meta_rule_reasoning'
        elif intent.intent_type == IntentType.UNIVERSE_COMPARISON:
            return 'universe_reasoning'
        
        # Check for basic reasoning patterns
        input_lower = intent.context_clues[0].lower() if intent.context_clues else ""
        
        if any(re.search(pattern, input_lower) for pattern in self.fallback_patterns['math']):
            return 'basic_reasoning'
        elif any(re.search(pattern, input_lower) for pattern in self.fallback_patterns['logic']):
            return 'basic_reasoning'
        elif any(re.search(pattern, input_lower) for pattern in self.fallback_patterns['greeting']):
            return 'basic_reasoning'
        
        # Personality-specific reasoning types
        if personality == 'Strategos':
            return 'strategic_reasoning'
        elif personality == 'Archivist':
            return 'civilizational_reasoning'
        elif personality == 'Lawmaker':
            return 'meta_rule_reasoning'
        elif personality == 'Oracle':
            return 'universe_reasoning'
        
        # Default fallback
        return 'basic_reasoning'
    
    def _determine_domain(self, intent: Intent, personality: str) -> str:
        """Determine domain based on intent and personality."""
        # Use intent domain if available
        if intent.domain:
            return intent.domain
        
        # Personality-specific domains
        if personality == 'Strategos':
            return 'strategic'
        elif personality == 'Archivist':
            return 'historical'
        elif personality == 'Lawmaker':
            return 'governance'
        elif personality == 'Oracle':
            return 'cosmic'
        
        # Check for domain clues in context
        if intent.context_clues:
            for clue in intent.context_clues:
                if 'math' in clue.lower() or 'calculate' in clue.lower():
                    return 'mathematical'
                elif 'logic' in clue.lower() or 'reason' in clue.lower():
                    return 'logical'
                elif 'history' in clue.lower() or 'ancient' in clue.lower():
                    return 'historical'
                elif 'strategy' in clue.lower() or 'tactical' in clue.lower():
                    return 'strategic'
        
        # Default domain
        return 'general'
    
    def normalize_reasoning_input(self, context: ReasoningContext, user_input: str) -> Dict[str, Any]:
        """
        Normalize reasoning input to standard structure.
        
        Args:
            context: Resolved reasoning context
            user_input: Original user input
            
        Returns:
            Normalized input dictionary for reasoning engine
        """
        return {
            'user_input': user_input,
            'reasoning_type': context.reasoning_type,
            'intent_type': context.intent_type,
            'personality': context.personality,
            'domain': context.domain,
            'complexity_level': context.complexity_level,
            'confidence': context.confidence,
            'entities': context.entities,
            'context_clues': context.context_clues,
            'metadata': context.metadata,
            'timestamp': context.timestamp.isoformat(),
            'requires_fallback': context.metadata.get('fallback_required', False)
        }
    
    def detect_fallback_requirement(self, context: ReasoningContext) -> bool:
        """Detect if fallback reasoning is required."""
        # Low confidence requires fallback
        if context.confidence < 0.5:
            return True
        
        # Missing critical entities
        if not context.entities and context.intent_type in ['QUESTION', 'COMMAND']:
            return True
        
        # Simple greetings don't need complex reasoning
        if context.reasoning_type == 'basic_reasoning' and context.intent_type == 'QUESTION':
            return True
        
        return False
    
    def get_fallback_reasoning_type(self, context: ReasoningContext) -> str:
        """Get appropriate fallback reasoning type."""
        if context.personality == 'Strategos':
            return 'tactical_fallback'
        elif context.personality == 'Archivist':
            return 'historical_fallback'
        elif context.personality == 'Lawmaker':
            return 'structural_fallback'
        elif context.personality == 'Oracle':
            return 'comparative_fallback'
        else:
            return 'general_fallback'
    
    def validate_context(self, context: ReasoningContext) -> Tuple[bool, List[str]]:
        """
        Validate that context has all required fields.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if not context.reasoning_type:
            errors.append("Missing reasoning_type")
        if not context.intent_type:
            errors.append("Missing intent_type")
        if not context.personality:
            errors.append("Missing personality")
        if not context.domain:
            errors.append("Missing domain")
        
        # Check numeric fields
        if not isinstance(context.complexity_level, (int, float)) or not (0 <= context.complexity_level <= 1):
            errors.append("Invalid complexity_level")
        if not isinstance(context.confidence, (int, float)) or not (0 <= context.confidence <= 1):
            errors.append("Invalid confidence")
        
        # Check collections
        if not isinstance(context.entities, list):
            errors.append("Invalid entities")
        if not isinstance(context.context_clues, list):
            errors.append("Invalid context_clues")
        if not isinstance(context.metadata, dict):
            errors.append("Invalid metadata")
        
        return len(errors) == 0, errors
    
    def create_default_context(self, user_input: str, personality: str) -> ReasoningContext:
        """Create a default context when parsing fails."""
        return ReasoningContext(
            reasoning_type='basic_reasoning',
            intent_type='REQUEST',
            personality=personality,
            domain='general',
            complexity_level=0.5,
            confidence=0.3,
            entities=[],
            context_clues=[],
            metadata={
                'original_input': user_input,
                'fallback_required': True,
                'default_context': True
            }
        )
