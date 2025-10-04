"""
OmniMind Conversational Personality Layer
Provides personality-driven conversational interface to OmniMind's civilizational intelligence.
"""

from .interface import ConversationalInterface
from .personality_framework import PersonalityFramework, Personality, PersonalityType
from .reasoning_engine import ConversationalReasoningEngine
from .conversation_state import ConversationState, ConversationContext, ConversationPhase, ContextType
from .intent_parser import IntentParser, Intent, IntentType
from .response_generator import ResponseGenerator, ResponseStyle
from .context_intent_resolver import ContextIntentResolver, ReasoningContext
from .reasoning_normalizer import ReasoningNormalizer, NormalizedReasoningInput
from .personality_fallbacks import PersonalityFallbackReasoner, FallbackResponse
from .reflection_mode import ReflectionMode, ReflectionResult, ReflectionType, ReflectionInsight
from .reflexive_audit import ReflexiveAuditModule, AuditResult, PersonalityPerformance
from .meta_learning_engine import MetaLearningEngine, PersonalityProfile, LearningMetrics, LearningInsight, HeuristicUpdate
from .personality_evolution import PersonalityEvolutionEngine, PersonalityEvolutionState
from .reflection_mode_plus import ReflectionModePlus, ReflectionInsightPlus

__all__ = [
    'ConversationalInterface',
    'PersonalityFramework', 'Personality', 'PersonalityType',
    'ConversationalReasoningEngine',
    'ConversationState', 'ConversationContext', 'ConversationPhase', 'ContextType',
    'IntentParser', 'Intent', 'IntentType',
    'ResponseGenerator', 'ResponseStyle',
    'ContextIntentResolver', 'ReasoningContext',
    'ReasoningNormalizer', 'NormalizedReasoningInput',
    'PersonalityFallbackReasoner', 'FallbackResponse',
    'ReflectionMode', 'ReflectionResult', 'ReflectionType', 'ReflectionInsight',
    'ReflexiveAuditModule', 'AuditResult', 'PersonalityPerformance',
    'MetaLearningEngine', 'PersonalityProfile', 'LearningMetrics', 'LearningInsight', 'HeuristicUpdate',
    'PersonalityEvolutionEngine', 'PersonalityEvolutionState',
    'ReflectionModePlus', 'ReflectionInsightPlus'
]
