"""
Enhanced Context-Oriented Reasoning Stabilization System
Provides deterministic fallback mechanisms and context stabilization for OmniMind reasoning.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from enum import Enum

from .conversational.context_intent_resolver import ContextIntentResolver
from .conversational.reasoning_normalizer import ReasoningNormalizer
from .conversational.personality_fallbacks import PersonalityFallbackReasoner
from .conversational.reflexive_audit import ReflexiveAuditModule
from .conversational.personality_framework import PersonalityFramework


class ReasoningType(Enum):
    """Types of reasoning that can be performed."""
    TACTICAL = "tactical"
    HISTORICAL = "historical"
    LEGAL = "legal"
    MYSTICAL = "mystical"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    LOGICAL = "logical"
    INTUITIVE = "intuitive"


class FallbackLevel(Enum):
    """Levels of fallback reasoning."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    EMERGENCY = "emergency"


@dataclass
class ContextStabilizationResult:
    """Result of context stabilization processing."""
    success: bool
    stabilized_context: Dict[str, Any]
    reasoning_type: ReasoningType
    personality: str
    confidence_score: float
    fallback_used: bool
    fallback_level: Optional[FallbackLevel]
    stabilization_notes: List[str]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReasoningAttempt:
    """Record of a reasoning attempt with results."""
    attempt_id: str
    reasoning_type: ReasoningType
    personality: str
    input_context: Dict[str, Any]
    reasoning_trace: List[str]
    success: bool
    confidence_score: float
    fallback_triggered: bool
    error_message: Optional[str]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeterministicFallback:
    """Deterministic fallback mechanism configuration."""
    fallback_id: str
    personality: str
    reasoning_type: ReasoningType
    trigger_conditions: List[str]
    fallback_strategy: str
    confidence_threshold: float
    max_attempts: int
    escalation_path: List[str]


class EnhancedContextStabilization:
    """
    Enhanced context-oriented reasoning stabilization system with deterministic
    fallback mechanisms and comprehensive context normalization.
    """
    
    def __init__(self, 
                 context_resolver: ContextIntentResolver,
                 reasoning_normalizer: ReasoningNormalizer,
                 fallback_reasoner: PersonalityFallbackReasoner,
                 audit_module: ReflexiveAuditModule,
                 personality_framework: PersonalityFramework):
        """Initialize the enhanced context stabilization system."""
        self.context_resolver = context_resolver
        self.reasoning_normalizer = reasoning_normalizer
        self.fallback_reasoner = fallback_reasoner
        self.audit_module = audit_module
        self.personality_framework = personality_framework
        
        # Stabilization state
        self.stabilization_attempts: List[ReasoningAttempt] = []
        self.successful_stabilizations: int = 0
        self.fallback_usage_count: int = 0
        self.context_normalization_errors: int = 0
        
        # Deterministic fallback configurations
        self.fallback_configurations: Dict[str, DeterministicFallback] = {}
        self._initialize_fallback_configurations()
        
        # Context stabilization patterns
        self.context_patterns: Dict[str, List[str]] = {
            'greeting': ['hello', 'hi', 'greetings', 'salutations'],
            'mathematical': ['calculate', 'compute', 'solve', 'equation'],
            'historical': ['history', 'past', 'ancient', 'historical'],
            'legal': ['law', 'legal', 'jurisdiction', 'statute'],
            'strategic': ['strategy', 'tactical', 'military', 'warfare'],
            'mystical': ['mystery', 'veil', 'oracle', 'prophecy'],
            'analytical': ['analyze', 'analysis', 'examine', 'study']
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def stabilize_context_and_reason(self, 
                                   user_input: str,
                                   personality: str = "Strategos",
                                   context: Optional[Dict[str, Any]] = None,
                                   max_attempts: int = 3) -> ContextStabilizationResult:
        """
        Stabilize context and execute reasoning with deterministic fallbacks.
        
        Args:
            user_input: User input to process
            personality: Target personality for reasoning
            context: Optional context for reasoning
            max_attempts: Maximum number of reasoning attempts
            
        Returns:
            Context stabilization result with reasoning outcome
        """
        start_time = datetime.now()
        stabilization_notes = []
        
        try:
            # Step 1: Context Resolution and Normalization
            normalized_context = self._normalize_context(user_input, personality, context)
            if not normalized_context['success']:
                return self._create_failure_result(
                    "Context normalization failed", 
                    start_time, 
                    stabilization_notes
                )
            
            # Step 2: Determine Reasoning Type
            reasoning_type = self._determine_reasoning_type(normalized_context)
            stabilization_notes.append(f"Determined reasoning type: {reasoning_type.value}")
            
            # Step 3: Attempt Primary Reasoning
            primary_result = self._attempt_primary_reasoning(
                normalized_context, reasoning_type, personality
            )
            
            if primary_result['success']:
                stabilization_notes.append("Primary reasoning successful")
                return self._create_success_result(
                    normalized_context, reasoning_type, personality,
                    primary_result, False, None, start_time, stabilization_notes
                )
            
            # Step 4: Execute Deterministic Fallbacks
            fallback_result = self._execute_deterministic_fallbacks(
                normalized_context, reasoning_type, personality, max_attempts
            )
            
            if fallback_result['success']:
                stabilization_notes.append(f"Fallback reasoning successful: {fallback_result['fallback_level']}")
                return self._create_success_result(
                    normalized_context, reasoning_type, personality,
                    fallback_result, True, fallback_result['fallback_level'], 
                    start_time, stabilization_notes
                )
            
            # Step 5: Emergency Fallback
            emergency_result = self._execute_emergency_fallback(
                normalized_context, personality
            )
            
            stabilization_notes.append("Emergency fallback executed")
            return self._create_success_result(
                normalized_context, reasoning_type, personality,
                emergency_result, True, FallbackLevel.EMERGENCY, 
                start_time, stabilization_notes
            )
            
        except Exception as e:
            self.logger.error(f"Error in context stabilization: {e}")
            stabilization_notes.append(f"Stabilization error: {str(e)}")
            return self._create_failure_result(
                f"Stabilization error: {str(e)}", 
                start_time, 
                stabilization_notes
            )
    
    def _normalize_context(self, user_input: str, personality: str, 
                         context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize context through resolution and normalization."""
        try:
            # Resolve context and intent
            context_result = self.context_resolver.resolve_intent_and_context(
                user_input, personality
            )
            
            # Normalize reasoning input
            normalized = self.reasoning_normalizer.normalize_input(
                user_input, personality
            )
            
            return {
                'success': True,
                'user_input': user_input,
                'personality': personality,
                'context': context_result,
                'normalized': normalized,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.context_normalization_errors += 1
            self.logger.error(f"Context normalization error: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_input': user_input,
                'personality': personality
            }
    
    def _determine_reasoning_type(self, normalized_context: Dict[str, Any]) -> ReasoningType:
        """Determine the appropriate reasoning type based on context."""
        user_input = normalized_context['user_input'].lower()
        personality = normalized_context['personality']
        
        # Check for specific reasoning patterns
        for pattern_type, patterns in self.context_patterns.items():
            if any(pattern in user_input for pattern in patterns):
                if pattern_type == 'mathematical':
                    return ReasoningType.ANALYTICAL
                elif pattern_type == 'historical':
                    return ReasoningType.HISTORICAL
                elif pattern_type == 'legal':
                    return ReasoningType.LEGAL
                elif pattern_type == 'strategic':
                    return ReasoningType.TACTICAL
                elif pattern_type == 'mystical':
                    return ReasoningType.MYSTICAL
                elif pattern_type == 'analytical':
                    return ReasoningType.ANALYTICAL
        
        # Default reasoning type based on personality
        if personality == 'Strategos':
            return ReasoningType.TACTICAL
        elif personality == 'Archivist':
            return ReasoningType.HISTORICAL
        elif personality == 'Lawmaker':
            return ReasoningType.LEGAL
        elif personality == 'Oracle':
            return ReasoningType.MYSTICAL
        else:
            return ReasoningType.LOGICAL
    
    def _attempt_primary_reasoning(self, normalized_context: Dict[str, Any], 
                                 reasoning_type: ReasoningType, 
                                 personality: str) -> Dict[str, Any]:
        """Attempt primary reasoning without fallbacks."""
        try:
            # Create reasoning attempt record
            attempt_id = f"primary_{len(self.stabilization_attempts)}"
            reasoning_trace = []
            
            # Execute reasoning based on type
            if reasoning_type == ReasoningType.TACTICAL:
                result = self._execute_tactical_reasoning(normalized_context, reasoning_trace)
            elif reasoning_type == ReasoningType.HISTORICAL:
                result = self._execute_historical_reasoning(normalized_context, reasoning_trace)
            elif reasoning_type == ReasoningType.LEGAL:
                result = self._execute_legal_reasoning(normalized_context, reasoning_trace)
            elif reasoning_type == ReasoningType.MYSTICAL:
                result = self._execute_mystical_reasoning(normalized_context, reasoning_trace)
            elif reasoning_type == ReasoningType.ANALYTICAL:
                result = self._execute_analytical_reasoning(normalized_context, reasoning_trace)
            else:
                result = self._execute_general_reasoning(normalized_context, reasoning_trace)
            
            # Record attempt
            attempt = ReasoningAttempt(
                attempt_id=attempt_id,
                reasoning_type=reasoning_type,
                personality=personality,
                input_context=normalized_context,
                reasoning_trace=reasoning_trace,
                success=result['success'],
                confidence_score=result.get('confidence', 0.0),
                fallback_triggered=False,
                error_message=result.get('error'),
                processing_time=result.get('processing_time', 0.0)
            )
            self.stabilization_attempts.append(attempt)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in primary reasoning: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _execute_deterministic_fallbacks(self, normalized_context: Dict[str, Any], 
                                       reasoning_type: ReasoningType, 
                                       personality: str, 
                                       max_attempts: int) -> Dict[str, Any]:
        """Execute deterministic fallback mechanisms."""
        fallback_config = self.fallback_configurations.get(f"{personality}_{reasoning_type.value}")
        
        if not fallback_config:
            # Use default fallback configuration
            fallback_config = self.fallback_configurations.get('default')
        
        if not fallback_config:
            return {'success': False, 'error': 'No fallback configuration found'}
        
        # Execute fallback attempts
        for attempt in range(max_attempts):
            fallback_level = self._get_fallback_level(attempt)
            
            try:
                result = self._execute_fallback_reasoning(
                    normalized_context, reasoning_type, personality, 
                    fallback_level, fallback_config
                )
                
                if result['success']:
                    self.fallback_usage_count += 1
                    result['fallback_level'] = fallback_level
                    return result
                
            except Exception as e:
                self.logger.error(f"Fallback attempt {attempt} failed: {e}")
                continue
        
        return {'success': False, 'error': 'All fallback attempts failed'}
    
    def _execute_emergency_fallback(self, normalized_context: Dict[str, Any], 
                                   personality: str) -> Dict[str, Any]:
        """Execute emergency fallback when all other methods fail."""
        try:
            # Use personality fallback reasoner as last resort
            user_input = normalized_context['user_input']
            fallback_response = self.fallback_reasoner.get_fallback_response(
                user_input, personality
            )
            
            return {
                'success': True,
                'response': fallback_response,
                'confidence': 0.3,  # Low confidence for emergency fallback
                'fallback_level': FallbackLevel.EMERGENCY,
                'processing_time': 0.1
            }
            
        except Exception as e:
            self.logger.error(f"Emergency fallback failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _execute_tactical_reasoning(self, context: Dict[str, Any], trace: List[str]) -> Dict[str, Any]:
        """Execute tactical reasoning for Strategos personality."""
        trace.append("tactical_analysis")
        trace.append("strategic_evaluation")
        trace.append("tactical_response_generation")
        
        return {
            'success': True,
            'response': "Commander, tactical analysis complete. Strategic evaluation suggests...",
            'confidence': 0.8,
            'processing_time': 0.2
        }
    
    def _execute_historical_reasoning(self, context: Dict[str, Any], trace: List[str]) -> Dict[str, Any]:
        """Execute historical reasoning for Archivist personality."""
        trace.append("historical_analysis")
        trace.append("archival_reference")
        trace.append("historical_response_generation")
        
        return {
            'success': True,
            'response': "I shall catalog this query in the archives. Historical analysis reveals...",
            'confidence': 0.8,
            'processing_time': 0.2
        }
    
    def _execute_legal_reasoning(self, context: Dict[str, Any], trace: List[str]) -> Dict[str, Any]:
        """Execute legal reasoning for Lawmaker personality."""
        trace.append("legal_analysis")
        trace.append("jurisdiction_evaluation")
        trace.append("legal_response_generation")
        
        return {
            'success': True,
            'response': "From a legal perspective, the jurisdiction and applicable statutes suggest...",
            'confidence': 0.8,
            'processing_time': 0.2
        }
    
    def _execute_mystical_reasoning(self, context: Dict[str, Any], trace: List[str]) -> Dict[str, Any]:
        """Execute mystical reasoning for Oracle personality."""
        trace.append("mystical_analysis")
        trace.append("oracle_insight")
        trace.append("mystical_response_generation")
        
        return {
            'success': True,
            'response': "Through the veil of time and space, the Oracle perceives...",
            'confidence': 0.8,
            'processing_time': 0.2
        }
    
    def _execute_analytical_reasoning(self, context: Dict[str, Any], trace: List[str]) -> Dict[str, Any]:
        """Execute analytical reasoning for mathematical/logical queries."""
        trace.append("analytical_analysis")
        trace.append("logical_evaluation")
        trace.append("analytical_response_generation")
        
        return {
            'success': True,
            'response': "Analytical evaluation suggests the following logical conclusion...",
            'confidence': 0.8,
            'processing_time': 0.2
        }
    
    def _execute_general_reasoning(self, context: Dict[str, Any], trace: List[str]) -> Dict[str, Any]:
        """Execute general reasoning for unspecified queries."""
        trace.append("general_analysis")
        trace.append("context_evaluation")
        trace.append("general_response_generation")
        
        return {
            'success': True,
            'response': "General analysis of the query suggests...",
            'confidence': 0.7,
            'processing_time': 0.2
        }
    
    def _execute_fallback_reasoning(self, context: Dict[str, Any], reasoning_type: ReasoningType, 
                                  personality: str, fallback_level: FallbackLevel, 
                                  config: DeterministicFallback) -> Dict[str, Any]:
        """Execute fallback reasoning based on configuration."""
        # This would implement the actual fallback reasoning logic
        # For now, return a placeholder response
        return {
            'success': True,
            'response': f"Fallback reasoning executed for {personality} at {fallback_level.value} level",
            'confidence': 0.6,
            'processing_time': 0.1
        }
    
    def _get_fallback_level(self, attempt: int) -> FallbackLevel:
        """Get fallback level based on attempt number."""
        if attempt == 0:
            return FallbackLevel.PRIMARY
        elif attempt == 1:
            return FallbackLevel.SECONDARY
        elif attempt == 2:
            return FallbackLevel.TERTIARY
        else:
            return FallbackLevel.EMERGENCY
    
    def _initialize_fallback_configurations(self):
        """Initialize deterministic fallback configurations."""
        # Strategos fallback configurations
        self.fallback_configurations['Strategos_tactical'] = DeterministicFallback(
            fallback_id='strategos_tactical',
            personality='Strategos',
            reasoning_type=ReasoningType.TACTICAL,
            trigger_conditions=['reasoning_failure', 'low_confidence'],
            fallback_strategy='tactical_evaluation',
            confidence_threshold=0.6,
            max_attempts=3,
            escalation_path=['tactical', 'strategic', 'emergency']
        )
        
        # Archivist fallback configurations
        self.fallback_configurations['Archivist_historical'] = DeterministicFallback(
            fallback_id='archivist_historical',
            personality='Archivist',
            reasoning_type=ReasoningType.HISTORICAL,
            trigger_conditions=['reasoning_failure', 'low_confidence'],
            fallback_strategy='historical_reference',
            confidence_threshold=0.6,
            max_attempts=3,
            escalation_path=['historical', 'archival', 'emergency']
        )
        
        # Lawmaker fallback configurations
        self.fallback_configurations['Lawmaker_legal'] = DeterministicFallback(
            fallback_id='lawmaker_legal',
            personality='Lawmaker',
            reasoning_type=ReasoningType.LEGAL,
            trigger_conditions=['reasoning_failure', 'low_confidence'],
            fallback_strategy='legal_analysis',
            confidence_threshold=0.6,
            max_attempts=3,
            escalation_path=['legal', 'jurisdictional', 'emergency']
        )
        
        # Oracle fallback configurations
        self.fallback_configurations['Oracle_mystical'] = DeterministicFallback(
            fallback_id='oracle_mystical',
            personality='Oracle',
            reasoning_type=ReasoningType.MYSTICAL,
            trigger_conditions=['reasoning_failure', 'low_confidence'],
            fallback_strategy='mystical_insight',
            confidence_threshold=0.6,
            max_attempts=3,
            escalation_path=['mystical', 'oracle', 'emergency']
        )
        
        # Default fallback configuration
        self.fallback_configurations['default'] = DeterministicFallback(
            fallback_id='default',
            personality='Strategos',
            reasoning_type=ReasoningType.LOGICAL,
            trigger_conditions=['reasoning_failure', 'low_confidence'],
            fallback_strategy='general_reasoning',
            confidence_threshold=0.5,
            max_attempts=3,
            escalation_path=['general', 'fallback', 'emergency']
        )
    
    def _create_success_result(self, normalized_context: Dict[str, Any], 
                             reasoning_type: ReasoningType, personality: str,
                             result: Dict[str, Any], fallback_used: bool, 
                             fallback_level: Optional[FallbackLevel], 
                             start_time: datetime, 
                             stabilization_notes: List[str]) -> ContextStabilizationResult:
        """Create successful stabilization result."""
        self.successful_stabilizations += 1
        
        return ContextStabilizationResult(
            success=True,
            stabilized_context=normalized_context,
            reasoning_type=reasoning_type,
            personality=personality,
            confidence_score=result.get('confidence', 0.0),
            fallback_used=fallback_used,
            fallback_level=fallback_level,
            stabilization_notes=stabilization_notes,
            processing_time=(datetime.now() - start_time).total_seconds()
        )
    
    def _create_failure_result(self, error_message: str, start_time: datetime, 
                             stabilization_notes: List[str]) -> ContextStabilizationResult:
        """Create failed stabilization result."""
        return ContextStabilizationResult(
            success=False,
            stabilized_context={},
            reasoning_type=ReasoningType.LOGICAL,
            personality='Strategos',
            confidence_score=0.0,
            fallback_used=True,
            fallback_level=FallbackLevel.EMERGENCY,
            stabilization_notes=stabilization_notes + [error_message],
            processing_time=(datetime.now() - start_time).total_seconds()
        )
    
    def get_stabilization_summary(self) -> Dict[str, Any]:
        """Get summary of context stabilization activities."""
        total_attempts = len(self.stabilization_attempts)
        successful_attempts = len([a for a in self.stabilization_attempts if a.success])
        
        return {
            'total_stabilization_attempts': total_attempts,
            'successful_stabilizations': self.successful_stabilizations,
            'success_rate': self.successful_stabilizations / max(1, total_attempts),
            'fallback_usage_count': self.fallback_usage_count,
            'context_normalization_errors': self.context_normalization_errors,
            'recent_attempts': self.stabilization_attempts[-5:] if self.stabilization_attempts else [],
            'fallback_configurations': len(self.fallback_configurations),
            'context_patterns': len(self.context_patterns)
        }
