"""
Transcendence Layer for OmniMind Civilization Intelligence
Enables autonomous civilization simulation and strategic civilization intelligence.
"""

from .core_structures import (
    Civilization, AgentCluster, EnvironmentState, TemporalState,
    CivilizationIntelligenceCore, GovernanceModel, AgentType, CivilizationArchetype
)

from .civilization_architect import CivilizationArchitect

from .temporal_evolution_engine import TemporalEvolutionEngine

from .macro_pattern_detector import (
    MacroPatternDetector, MacroPattern, PatternType, PatternSignificance
)

from .strategic_orchestrator import (
    StrategicCivilizationalOrchestrator, StrategicDomain, PolicyArchetype,
    InteractionType, StrategicInteraction, StrategicSimulation
)

from .experimentation_system import (
    CivilizationExperimentationSystem, ExperimentType, ExperimentStatus,
    ExperimentScenario, ExperimentResult, ExperimentBatch
)

from .pattern_codification_engine import (
    PatternCodificationEngine, PatternBlueprint, BlueprintType, CodificationStatus,
    StrategicDoctrine
)

from .civilizational_genetic_archive import (
    CivilizationalGeneticArchive, CivilizationDNA, LineageType, ArchiveStatus,
    LineageTree
)

from .strategic_feedback_loop import (
    StrategicFeedbackLoop, FeedbackApplication, FeedbackPipeline, FeedbackType, FeedbackStatus
)

from .reflexive_introspection_layer import (
    ReflexiveIntrospectionLayer, IntrospectionAnalysis, IntrospectionType, IntrospectionStatus,
    MetaRule, MetaRuleType, SystemInsight
)

from .multi_universe_orchestration import (
    MultiUniverseOrchestrator, UniverseParameters, UniverseState, UniverseType, UniverseStatus,
    CrossUniverseAnalysis
)

__all__ = [
    # Core structures
    'Civilization', 'AgentCluster', 'EnvironmentState', 'TemporalState',
    'CivilizationIntelligenceCore', 'GovernanceModel', 'AgentType', 'CivilizationArchetype',
    
    # Civilization architect
    'CivilizationArchitect',
    
    # Temporal evolution engine
    'TemporalEvolutionEngine',
    
    # Macro-pattern detector
    'MacroPatternDetector', 'MacroPattern', 'PatternType', 'PatternSignificance',
    
    # Strategic orchestrator
    'StrategicCivilizationalOrchestrator', 'StrategicDomain', 'PolicyArchetype',
    'InteractionType', 'StrategicInteraction', 'StrategicSimulation',
    
    # Experimentation system
    'CivilizationExperimentationSystem', 'ExperimentType', 'ExperimentStatus',
    'ExperimentScenario', 'ExperimentResult', 'ExperimentBatch',
    
    # Pattern codification engine
    'PatternCodificationEngine', 'PatternBlueprint', 'BlueprintType', 'CodificationStatus',
    'StrategicDoctrine',
    
    # Civilizational genetic archive
    'CivilizationalGeneticArchive', 'CivilizationDNA', 'LineageType', 'ArchiveStatus',
    'LineageTree',
    
    # Strategic feedback loop
    'StrategicFeedbackLoop', 'FeedbackApplication', 'FeedbackPipeline', 'FeedbackType', 'FeedbackStatus',
    
    # Reflexive introspection layer
    'ReflexiveIntrospectionLayer', 'IntrospectionAnalysis', 'IntrospectionType', 'IntrospectionStatus',
    'MetaRule', 'MetaRuleType', 'SystemInsight',
    
    # Multi-universe orchestration
    'MultiUniverseOrchestrator', 'UniverseParameters', 'UniverseState', 'UniverseType', 'UniverseStatus',
    'CrossUniverseAnalysis'
]