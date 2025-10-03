"""
Experimental Intelligence System for OmniMind
Enables autonomous research, hypothesis generation, and creative intelligence.
"""

from .hypothesis_generator import HypothesisGenerator, Hypothesis, HypothesisType, HypothesisStatus
from .experiment_executor import ExperimentExecutor, Experiment, ExperimentType, ExperimentStatus
from .result_analyzer import ResultAnalyzer, ExperimentResult, ResultType, ConfidenceLevel
from .creative_synthesizer import CreativeSynthesizer, CreativeOutput, SynthesisType, InnovationLevel
from .autonomous_research import AutonomousResearcher, ResearchProject, ResearchPhase, ResearchStatus
from .knowledge_expansion import KnowledgeExpander, KnowledgeItem, ExpansionStrategy, DiscoveryType, IntegrationStatus
from .experimental_coordinator import ExperimentalCoordinator, ResearchCycle, CycleStatus

__all__ = [
    'HypothesisGenerator', 'Hypothesis', 'HypothesisType', 'HypothesisStatus',
    'ExperimentExecutor', 'Experiment', 'ExperimentType', 'ExperimentStatus',
    'ResultAnalyzer', 'ExperimentResult', 'ResultType', 'ConfidenceLevel',
    'CreativeSynthesizer', 'CreativeOutput', 'SynthesisType', 'InnovationLevel',
    'AutonomousResearcher', 'ResearchProject', 'ResearchPhase', 'ResearchStatus',
    'KnowledgeExpander', 'KnowledgeItem', 'ExpansionStrategy', 'DiscoveryType', 'IntegrationStatus',
    'ExperimentalCoordinator', 'ResearchCycle', 'CycleStatus'
]
