"""
Reflection Mode for Automatic Retry with Fallback Reasoning
Implements introspection and retry mechanisms when responses fail.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .context_intent_resolver import ContextIntentResolver, ReasoningContext
from .reasoning_normalizer import ReasoningNormalizer, NormalizedReasoningInput
from .personality_fallbacks import PersonalityFallbackReasoner, FallbackResponse


class ReflectionType(Enum):
    """Types of reflection modes."""
    NONE = auto()
    BASIC = auto()
    DEEP = auto()
    ADAPTIVE = auto()


@dataclass
class ReflectionResult:
    """Result of reflection process."""
    success: bool
    response: str
    reflection_type: ReflectionType
    attempts_made: int
    errors_encountered: List[str]
    fallback_used: bool
    confidence_improvement: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReflectionInsight:
    """Insight gained from reflection."""
    insight_type: str
    description: str
    confidence: float
    applicable_to: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReflectionMode:
    """Handles reflection and retry mechanisms."""
    
    def __init__(self, max_reflection_attempts: int = 3, reflection_timeout: float = 5.0):
        self.max_reflection_attempts = max_reflection_attempts
        self.reflection_timeout = reflection_timeout
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.context_resolver = ContextIntentResolver()
        self.reasoning_normalizer = ReasoningNormalizer()
        self.fallback_reasoner = PersonalityFallbackReasoner()
        
        # Reflection insights storage
        self.reflection_insights: List[ReflectionInsight] = []
        self.error_patterns: Dict[str, int] = {}
        self.success_patterns: Dict[str, int] = {}
        
        # Reflection strategies
        self._initialize_reflection_strategies()
    
    def _initialize_reflection_strategies(self):
        """Initialize reflection strategies."""
        self.reflection_strategies = {
            'context_analysis': self._analyze_context_failure,
            'personality_adaptation': self._adapt_personality_response,
            'reasoning_reroute': self._reroute_reasoning_path,
            'fallback_escalation': self._escalate_fallback_reasoning,
            'input_simplification': self._simplify_input,
            'confidence_boost': self._boost_confidence
        }
    
    def reflect_and_retry(self, user_input: str, personality: str, 
                         original_error: str, context: Optional[ReasoningContext] = None) -> ReflectionResult:
        """
        Reflect on failure and retry with improved approach.
        
        Args:
            user_input: Original user input
            personality: Current personality
            original_error: Error that caused reflection
            context: Optional pre-resolved context
            
        Returns:
            ReflectionResult with retry outcome
        """
        start_time = time.time()
        attempts_made = 0
        errors_encountered = [original_error]
        fallback_used = False
        confidence_improvement = 0.0
        
        self.logger.info(f"Starting reflection for input: {user_input[:50]}...")
        
        # Analyze the failure
        reflection_insights = self._analyze_failure(user_input, personality, original_error, context)
        
        # Try different reflection strategies
        for attempt in range(self.max_reflection_attempts):
            if time.time() - start_time > self.reflection_timeout:
                self.logger.warning("Reflection timeout reached")
                break
            
            attempts_made += 1
            self.logger.info(f"Reflection attempt {attempt + 1}/{self.max_reflection_attempts}")
            
            try:
                # Select reflection strategy
                strategy = self._select_reflection_strategy(attempt, reflection_insights)
                
                # Apply strategy
                result = self._apply_reflection_strategy(
                    user_input, personality, strategy, reflection_insights
                )
                
                if result['success']:
                    confidence_improvement = result.get('confidence_improvement', 0.0)
                    fallback_used = result.get('fallback_used', False)
                    
                    # Store successful pattern
                    self._record_success_pattern(user_input, personality, strategy)
                    
                    return ReflectionResult(
                        success=True,
                        response=result['response'],
                        reflection_type=ReflectionType.ADAPTIVE,
                        attempts_made=attempts_made,
                        errors_encountered=errors_encountered,
                        fallback_used=fallback_used,
                        confidence_improvement=confidence_improvement,
                        metadata={
                            'strategy_used': strategy,
                            'reflection_insights': [i.description for i in reflection_insights],
                            'processing_time': time.time() - start_time
                        }
                    )
                
            except Exception as e:
                error_msg = str(e)
                errors_encountered.append(error_msg)
                self.logger.error(f"Reflection attempt {attempt + 1} failed: {error_msg}")
                
                # Record error pattern
                self._record_error_pattern(user_input, personality, error_msg)
        
        # All attempts failed, use final fallback
        self.logger.warning("All reflection attempts failed, using final fallback")
        final_fallback = self._final_fallback_response(user_input, personality)
        
        return ReflectionResult(
            success=False,
            response=final_fallback,
            reflection_type=ReflectionType.BASIC,
            attempts_made=attempts_made,
            errors_encountered=errors_encountered,
            fallback_used=True,
            confidence_improvement=0.0,
            metadata={
                'final_fallback_used': True,
                'processing_time': time.time() - start_time
            }
        )
    
    def _analyze_failure(self, user_input: str, personality: str, 
                        error: str, context: Optional[ReasoningContext]) -> List[ReflectionInsight]:
        """Analyze the failure to gain insights."""
        insights = []
        
        # Analyze error type
        if "NoneType" in error:
            insights.append(ReflectionInsight(
                insight_type="null_reference",
                description="Null reference error detected",
                confidence=0.8,
                applicable_to=["context_resolution", "reasoning_normalization"]
            ))
        
        if "attribute" in error and "get" in error:
            insights.append(ReflectionInsight(
                insight_type="missing_attribute",
                description="Missing attribute error detected",
                confidence=0.9,
                applicable_to=["context_resolution", "reasoning_engine"]
            ))
        
        if "reasoning_type" in error:
            insights.append(ReflectionInsight(
                insight_type="reasoning_type_missing",
                description="Reasoning type not properly set",
                confidence=0.7,
                applicable_to=["reasoning_normalization"]
            ))
        
        # Analyze input characteristics
        if len(user_input.strip()) == 0:
            insights.append(ReflectionInsight(
                insight_type="empty_input",
                description="Empty input detected",
                confidence=1.0,
                applicable_to=["input_validation"]
            ))
        
        if len(user_input) > 1000:
            insights.append(ReflectionInsight(
                insight_type="long_input",
                description="Very long input detected",
                confidence=0.6,
                applicable_to=["input_simplification"]
            ))
        
        # Analyze personality-specific issues
        if personality == "Strategos" and "tactical" not in user_input.lower():
            insights.append(ReflectionInsight(
                insight_type="personality_mismatch",
                description="Input doesn't match personality expectations",
                confidence=0.5,
                applicable_to=["personality_adaptation"]
            ))
        
        return insights
    
    def _select_reflection_strategy(self, attempt: int, insights: List[ReflectionInsight]) -> str:
        """Select appropriate reflection strategy based on attempt and insights."""
        if attempt == 0:
            # First attempt: try context analysis
            return "context_analysis"
        elif attempt == 1:
            # Second attempt: try personality adaptation
            return "personality_adaptation"
        else:
            # Final attempt: use fallback escalation
            return "fallback_escalation"
    
    def _apply_reflection_strategy(self, user_input: str, personality: str, 
                                 strategy: str, insights: List[ReflectionInsight]) -> Dict[str, Any]:
        """Apply selected reflection strategy."""
        if strategy == "context_analysis":
            return self._analyze_context_failure(user_input, personality, insights)
        elif strategy == "personality_adaptation":
            return self._adapt_personality_response(user_input, personality, insights)
        elif strategy == "reasoning_reroute":
            return self._reroute_reasoning_path(user_input, personality, insights)
        elif strategy == "fallback_escalation":
            return self._escalate_fallback_reasoning(user_input, personality, insights)
        elif strategy == "input_simplification":
            return self._simplify_input(user_input, personality, insights)
        elif strategy == "confidence_boost":
            return self._boost_confidence(user_input, personality, insights)
        else:
            return {"success": False, "error": "Unknown strategy"}
    
    def _analyze_context_failure(self, user_input: str, personality: str, 
                               insights: List[ReflectionInsight]) -> Dict[str, Any]:
        """Analyze and fix context resolution failure."""
        try:
            # Re-resolve context with error handling
            context = self.context_resolver.resolve_intent_and_context(user_input, personality)
            
            # Validate context
            is_valid, errors = self.context_resolver.validate_context(context)
            if not is_valid:
                # Create default context
                context = self.context_resolver.create_default_context(user_input, personality)
            
            # Normalize input
            normalized = self.reasoning_normalizer.normalize_input(user_input, personality, context)
            
            # Get fallback response
            fallback_response = self.fallback_reasoner.get_fallback_response(
                user_input, personality, normalized.reasoning_type, normalized.metadata
            )
            
            return {
                "success": True,
                "response": fallback_response.content,
                "fallback_used": True,
                "confidence_improvement": 0.2
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _adapt_personality_response(self, user_input: str, personality: str, 
                                  insights: List[ReflectionInsight]) -> Dict[str, Any]:
        """Adapt personality response to better match input."""
        try:
            # Get personality-specific fallback
            fallback_response = self.fallback_reasoner.get_fallback_response(
                user_input, personality, "basic_reasoning", {}
            )
            
            # Apply personality-specific styling
            response = self._apply_personality_styling(fallback_response.content, personality)
            
            return {
                "success": True,
                "response": response,
                "fallback_used": True,
                "confidence_improvement": 0.3
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _reroute_reasoning_path(self, user_input: str, personality: str, 
                              insights: List[ReflectionInsight]) -> Dict[str, Any]:
        """Reroute to different reasoning path."""
        try:
            # Try different reasoning type
            if personality == "Strategos":
                reasoning_type = "strategic_reasoning"
            elif personality == "Archivist":
                reasoning_type = "civilizational_reasoning"
            elif personality == "Lawmaker":
                reasoning_type = "meta_rule_reasoning"
            elif personality == "Oracle":
                reasoning_type = "universe_reasoning"
            else:
                reasoning_type = "basic_reasoning"
            
            # Get fallback response with different reasoning type
            fallback_response = self.fallback_reasoner.get_fallback_response(
                user_input, personality, reasoning_type, {}
            )
            
            return {
                "success": True,
                "response": fallback_response.content,
                "fallback_used": True,
                "confidence_improvement": 0.1
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _escalate_fallback_reasoning(self, user_input: str, personality: str, 
                                   insights: List[ReflectionInsight]) -> Dict[str, Any]:
        """Escalate to more aggressive fallback reasoning."""
        try:
            # Use more aggressive fallback
            fallback_response = self.fallback_reasoner.get_fallback_response(
                user_input, personality, "basic_reasoning", {}
            )
            
            # Apply additional styling
            response = self._apply_personality_styling(fallback_response.content, personality)
            
            return {
                "success": True,
                "response": response,
                "fallback_used": True,
                "confidence_improvement": 0.4
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _simplify_input(self, user_input: str, personality: str, 
                       insights: List[ReflectionInsight]) -> Dict[str, Any]:
        """Simplify input and retry."""
        try:
            # Simplify input
            simplified_input = self._simplify_user_input(user_input)
            
            # Get fallback response for simplified input
            fallback_response = self.fallback_reasoner.get_fallback_response(
                simplified_input, personality, "basic_reasoning", {}
            )
            
            return {
                "success": True,
                "response": fallback_response.content,
                "fallback_used": True,
                "confidence_improvement": 0.2
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _boost_confidence(self, user_input: str, personality: str, 
                        insights: List[ReflectionInsight]) -> Dict[str, Any]:
        """Boost confidence and retry."""
        try:
            # Get fallback response
            fallback_response = self.fallback_reasoner.get_fallback_response(
                user_input, personality, "basic_reasoning", {}
            )
            
            # Apply confidence boost
            response = self._apply_confidence_boost(fallback_response.content, personality)
            
            return {
                "success": True,
                "response": response,
                "fallback_used": True,
                "confidence_improvement": 0.5
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _apply_personality_styling(self, response: str, personality: str) -> str:
        """Apply personality-specific styling to response."""
        if personality == "Strategos":
            if not response.startswith("Commander"):
                response = f"Commander, {response.lower()}"
        elif personality == "Archivist":
            if not response.startswith("From the archives"):
                response = f"From the archives of knowledge, {response.lower()}"
        elif personality == "Lawmaker":
            if not response.startswith("According to"):
                response = f"According to the principles of governance, {response.lower()}"
        elif personality == "Oracle":
            if not response.startswith("In the cosmic"):
                response = f"In the cosmic dance of reality, {response.lower()}"
        
        return response
    
    def _simplify_user_input(self, user_input: str) -> str:
        """Simplify user input for better processing."""
        # Remove extra whitespace
        simplified = " ".join(user_input.split())
        
        # Truncate if too long
        if len(simplified) > 200:
            simplified = simplified[:200] + "..."
        
        # Extract key words
        words = simplified.split()
        if len(words) > 10:
            # Keep first and last few words
            simplified = " ".join(words[:5] + words[-5:])
        
        return simplified
    
    def _apply_confidence_boost(self, response: str, personality: str) -> str:
        """Apply confidence boost to response."""
        confidence_phrases = {
            "Strategos": "I am confident in this tactical assessment.",
            "Archivist": "I am certain of this historical analysis.",
            "Lawmaker": "I am confident in this legal interpretation.",
            "Oracle": "I am certain of this cosmic insight."
        }
        
        phrase = confidence_phrases.get(personality, "I am confident in this response.")
        return f"{response} {phrase}"
    
    def _final_fallback_response(self, user_input: str, personality: str) -> str:
        """Generate final fallback response when all else fails."""
        fallback_responses = {
            "Strategos": "Commander, I encountered a tactical error. I recommend alternative approaches.",
            "Archivist": "From the archives, I must report an error in my analysis. Let me consult other sources.",
            "Lawmaker": "According to the principles of governance, I must report an error. Let me review the legal framework.",
            "Oracle": "In the cosmic tapestry, I perceive an error in my analysis. Let me consult the universal patterns."
        }
        
        return fallback_responses.get(personality, "I encountered an error. Let me try a different approach.")
    
    def _record_success_pattern(self, user_input: str, personality: str, strategy: str):
        """Record successful pattern for future use."""
        pattern_key = f"{personality}_{strategy}"
        self.success_patterns[pattern_key] = self.success_patterns.get(pattern_key, 0) + 1
    
    def _record_error_pattern(self, user_input: str, personality: str, error: str):
        """Record error pattern for analysis."""
        error_key = f"{personality}_{error[:50]}"
        self.error_patterns[error_key] = self.error_patterns.get(error_key, 0) + 1
    
    def get_reflection_insights(self) -> List[ReflectionInsight]:
        """Get accumulated reflection insights."""
        return self.reflection_insights
    
    def get_success_patterns(self) -> Dict[str, int]:
        """Get success patterns."""
        return self.success_patterns
    
    def get_error_patterns(self) -> Dict[str, int]:
        """Get error patterns."""
        return self.error_patterns
    
    def clear_insights(self):
        """Clear accumulated insights."""
        self.reflection_insights.clear()
        self.success_patterns.clear()
        self.error_patterns.clear()
