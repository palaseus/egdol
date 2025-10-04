"""
Reasoning Engine Input Normalization Layer
Ensures all inputs to the reasoning engine are properly normalized
with non-null context, intent, reasoning_type, personality, and metadata.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .context_intent_resolver import ReasoningContext
from .personality_fallbacks import PersonalityFallbackReasoner, FallbackResponse


@dataclass
class NormalizedReasoningInput:
    """Normalized input for reasoning engine."""
    user_input: str
    reasoning_type: str
    intent_type: str
    personality: str
    domain: str
    complexity_level: float
    confidence: float
    entities: List[str] = field(default_factory=list)
    context_clues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    requires_fallback: bool = False
    fallback_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reasoning engine."""
        return {
            'user_input': self.user_input,
            'reasoning_type': self.reasoning_type,
            'intent_type': self.intent_type,
            'personality': self.personality,
            'domain': self.domain,
            'complexity_level': self.complexity_level,
            'confidence': self.confidence,
            'entities': self.entities,
            'context_clues': self.context_clues,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'requires_fallback': self.requires_fallback,
            'fallback_type': self.fallback_type
        }


class ReasoningNormalizer:
    """Normalizes inputs before passing to reasoning engine."""
    
    def __init__(self):
        self.fallback_reasoner = PersonalityFallbackReasoner()
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default values for normalization."""
        self.default_reasoning_types = {
            'basic_reasoning': 'basic',
            'civilizational_reasoning': 'civilizational',
            'strategic_reasoning': 'strategic',
            'meta_rule_reasoning': 'meta_rule',
            'universe_reasoning': 'universe'
        }
        
        self.default_domains = {
            'Strategos': 'strategic',
            'Archivist': 'historical',
            'Lawmaker': 'governance',
            'Oracle': 'cosmic'
        }
    
    def normalize_input(self, user_input: str, personality: str, 
                       context: Optional[ReasoningContext] = None) -> NormalizedReasoningInput:
        """
        Normalize input for reasoning engine.
        
        Args:
            user_input: User's input message
            personality: Current personality
            context: Optional pre-resolved context
            
        Returns:
            NormalizedReasoningInput ready for reasoning engine
        """
        # If context is provided, use it; otherwise create default
        if context is None:
            context = self._create_default_context(user_input, personality)
        
        # Determine if fallback is required
        requires_fallback = self._determine_fallback_requirement(context)
        fallback_type = self._get_fallback_type(context) if requires_fallback else None
        
        # Create normalized input
        normalized = NormalizedReasoningInput(
            user_input=user_input,
            reasoning_type=context.reasoning_type,
            intent_type=context.intent_type,
            personality=personality,
            domain=context.domain,
            complexity_level=context.complexity_level,
            confidence=context.confidence,
            entities=context.entities,
            context_clues=context.context_clues,
            metadata=context.metadata,
            requires_fallback=requires_fallback,
            fallback_type=fallback_type
        )
        
        # Validate normalization
        is_valid, errors = self._validate_normalized_input(normalized)
        if not is_valid:
            # Fix common issues
            normalized = self._fix_normalization_issues(normalized, errors)
        
        return normalized
    
    def _create_default_context(self, user_input: str, personality: str) -> ReasoningContext:
        """Create default context when none is provided."""
        from .context_intent_resolver import ReasoningContext
        
        return ReasoningContext(
            reasoning_type='basic_reasoning',
            intent_type='REQUEST',
            personality=personality,
            domain=self.default_domains.get(personality, 'general'),
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
    
    def _determine_fallback_requirement(self, context: ReasoningContext) -> bool:
        """Determine if fallback reasoning is required."""
        # Only use fallback for very low confidence
        if context.confidence < 0.2:
            return True
        
        # Only use fallback for very basic reasoning with simple inputs
        if context.reasoning_type == 'basic_reasoning' and context.intent_type in ['GREETING', 'CLOSURE']:
            return True
        
        # Only use fallback if explicitly required
        if context.metadata.get('fallback_required', False):
            return True
        
        return False
    
    def _get_fallback_type(self, context: ReasoningContext) -> str:
        """Get appropriate fallback type."""
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
    
    def _validate_normalized_input(self, normalized: NormalizedReasoningInput) -> Tuple[bool, List[str]]:
        """Validate normalized input."""
        errors = []
        
        # Check required fields
        if not normalized.user_input:
            errors.append("Missing user_input")
        if not normalized.reasoning_type:
            errors.append("Missing reasoning_type")
        if not normalized.intent_type:
            errors.append("Missing intent_type")
        if not normalized.personality:
            errors.append("Missing personality")
        if not normalized.domain:
            errors.append("Missing domain")
        
        # Check numeric fields
        if not isinstance(normalized.complexity_level, (int, float)) or not (0 <= normalized.complexity_level <= 1):
            errors.append("Invalid complexity_level")
        if not isinstance(normalized.confidence, (int, float)) or not (0 <= normalized.confidence <= 1):
            errors.append("Invalid confidence")
        
        # Check collections
        if not isinstance(normalized.entities, list):
            errors.append("Invalid entities")
        if not isinstance(normalized.context_clues, list):
            errors.append("Invalid context_clues")
        if not isinstance(normalized.metadata, dict):
            errors.append("Invalid metadata")
        
        return len(errors) == 0, errors
    
    def _fix_normalization_issues(self, normalized: NormalizedReasoningInput, errors: List[str]) -> NormalizedReasoningInput:
        """Fix common normalization issues."""
        # Fix missing fields
        if "Missing user_input" in errors:
            normalized.user_input = "Hello"
        if "Missing reasoning_type" in errors:
            normalized.reasoning_type = 'basic_reasoning'
        if "Missing intent_type" in errors:
            normalized.intent_type = 'REQUEST'
        if "Missing personality" in errors:
            normalized.personality = 'Strategos'
        if "Missing domain" in errors:
            normalized.domain = self.default_domains.get(normalized.personality, 'general')
        
        # Fix invalid numeric fields
        if "Invalid complexity_level" in errors:
            normalized.complexity_level = 0.5
        if "Invalid confidence" in errors:
            normalized.confidence = 0.5
        
        # Fix invalid collections
        if "Invalid entities" in errors:
            normalized.entities = []
        if "Invalid context_clues" in errors:
            normalized.context_clues = []
        if "Invalid metadata" in errors:
            normalized.metadata = {}
        
        return normalized
    
    def get_fallback_response(self, normalized: NormalizedReasoningInput) -> FallbackResponse:
        """Get fallback response for normalized input."""
        return self.fallback_reasoner.get_fallback_response(
            user_input=normalized.user_input,
            personality=normalized.personality,
            reasoning_type=normalized.reasoning_type,
            context=normalized.metadata
        )
    
    def should_use_fallback(self, normalized: NormalizedReasoningInput) -> bool:
        """Determine if fallback should be used instead of reasoning engine."""
        return self.fallback_reasoner.should_use_fallback(
            confidence=normalized.confidence,
            reasoning_type=normalized.reasoning_type,
            context=normalized.metadata
        )
    
    def create_reasoning_engine_input(self, normalized: NormalizedReasoningInput) -> Dict[str, Any]:
        """Create properly formatted input for reasoning engine."""
        return {
            'query': normalized.user_input,
            'reasoning_type': normalized.reasoning_type,
            'intent_type': normalized.intent_type,
            'personality': normalized.personality,
            'domain': normalized.domain,
            'complexity_level': normalized.complexity_level,
            'confidence': normalized.confidence,
            'entities': normalized.entities,
            'context_clues': normalized.context_clues,
            'metadata': normalized.metadata,
            'timestamp': normalized.timestamp
        }
    
    def create_fallback_input(self, normalized: NormalizedReasoningInput) -> Dict[str, Any]:
        """Create input for fallback reasoning."""
        return {
            'user_input': normalized.user_input,
            'personality': normalized.personality,
            'reasoning_type': normalized.reasoning_type,
            'fallback_type': normalized.fallback_type,
            'context': normalized.metadata
        }
    
    def log_normalization(self, normalized: NormalizedReasoningInput, stage: str):
        """Log normalization process for debugging."""
        print(f"[NORMALIZATION {stage}] Input: {normalized.user_input[:50]}...")
        print(f"[NORMALIZATION {stage}] Reasoning: {normalized.reasoning_type}")
        print(f"[NORMALIZATION {stage}] Personality: {normalized.personality}")
        print(f"[NORMALIZATION {stage}] Fallback: {normalized.requires_fallback}")
        print(f"[NORMALIZATION {stage}] Confidence: {normalized.confidence}")
        print()
