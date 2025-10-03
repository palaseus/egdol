"""
OmniMind Civilizational Feedback + Meta-Rule Evolution System
Complete integration of conversation, simulation, and meta-rule evolution.
"""

from .civilizational_feedback import (
    CivilizationalFeedbackEngine,
    CivilizationalInsight,
    MetaRuleCandidate,
    CivilizationalFeedbackResult
)

from .meta_rule_discovery import (
    MetaRuleDiscoveryEngine,
    PatternMatch,
    MetaRuleTemplate,
    MetaRuleValidation
)

from .enhanced_context_stabilization import (
    EnhancedContextStabilization,
    ContextStabilizationResult,
    ReasoningType,
    FallbackLevel
)

from .reflection_mode_plus_plus import (
    ReflectionModePlusPlus,
    ReflectionAnalysis,
    HeuristicUpdate,
    ReflectionResultPlusPlus,
    ReflectionTrigger,
    HeuristicAdjustment
)

from .execution_pipeline import (
    OmniMindExecutionPipeline,
    StructuredResponse,
    PipelineStage,
    PipelineStatus,
    PipelineExecution
)

# Import conversational components
from .conversational import (
    ReflexiveAuditModule,
    MetaLearningEngine,
    PersonalityEvolutionEngine,
    ReflectionModePlus,
    ContextIntentResolver,
    ReasoningNormalizer,
    PersonalityFallbackReasoner,
    PersonalityFramework
)

__all__ = [
    # Civilizational Feedback
    'CivilizationalFeedbackEngine',
    'CivilizationalInsight',
    'MetaRuleCandidate',
    'CivilizationalFeedbackResult',
    
    # Meta-Rule Discovery
    'MetaRuleDiscoveryEngine',
    'PatternMatch',
    'MetaRuleTemplate',
    'MetaRuleValidation',
    
    # Enhanced Context Stabilization
    'EnhancedContextStabilization',
    'ContextStabilizationResult',
    'ReasoningType',
    'FallbackLevel',
    
    # Reflection Mode Plus Plus
    'ReflectionModePlusPlus',
    'ReflectionAnalysis',
    'HeuristicUpdate',
    'ReflectionResultPlusPlus',
    'ReflectionTrigger',
    'HeuristicAdjustment',
    
    # Execution Pipeline
    'OmniMindExecutionPipeline',
    'StructuredResponse',
    'PipelineStage',
    'PipelineStatus',
    'PipelineExecution',
    
    # Conversational Components
    'ReflexiveAuditModule',
    'MetaLearningEngine',
    'PersonalityEvolutionEngine',
    'ReflectionModePlus',
    'ContextIntentResolver',
    'ReasoningNormalizer',
    'PersonalityFallbackReasoner',
    'PersonalityFramework'
]