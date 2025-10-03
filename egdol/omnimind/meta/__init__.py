"""
Meta-Intelligence & Self-Evolution System for OmniMind
Enables autonomous meta-cognitive reasoning and system evolution.
"""

from .architecture_inventor import ArchitectureInventor, ArchitectureProposal, ArchitectureType, InnovationLevel
from .skill_policy_innovator import SkillPolicyInnovator, InnovationProposal, InnovationType, PolicyType
from .self_upgrader import SelfUpgrader, UpgradePlan, UpgradeStatus, RollbackStatus
from .evaluation_engine import EvaluationEngine, EvaluationResult, MetricType, EvaluationStatus
from .evolution_simulator import EvolutionSimulator, EvolutionaryPathway, SimulationOutcome, SimulationOutcomeType, EvolutionaryStage
from .meta_coordinator import MetaCoordinator, MetaCycle, MetaCycleStatus

__all__ = [
    'ArchitectureInventor', 'ArchitectureProposal', 'ArchitectureType', 'InnovationLevel',
    'SkillPolicyInnovator', 'InnovationProposal', 'InnovationType', 'PolicyType',
    'SelfUpgrader', 'UpgradePlan', 'UpgradeStatus', 'RollbackStatus',
    'EvaluationEngine', 'EvaluationResult', 'MetricType', 'EvaluationStatus',
    'EvolutionSimulator', 'EvolutionaryPathway', 'SimulationOutcome', 'SimulationOutcomeType', 'EvolutionaryStage',
    'MetaCoordinator', 'MetaCycle', 'MetaCycleStatus'
]
