"""
Next-Generation Autonomous Research & Knowledge Expansion System
Enables fully self-directing intelligence with autonomous research capabilities.
"""

from .research_project_generator import (
    ResearchProjectGenerator, 
    ResearchProject, 
    ProjectPhase, 
    ProjectStatus,
    ResearchDomain,
    ComplexityLevel,
    InnovationType
)
from .autonomous_experimenter import (
    AutonomousExperimenter,
    Experiment,
    ExperimentType,
    ExperimentStatus,
    ResourceManager,
    ConstraintManager
)
from .discovery_analyzer import (
    DiscoveryAnalyzer,
    Discovery,
    DiscoveryType,
    NoveltyLevel,
    SignificanceLevel,
    ValidationResult
)
from .knowledge_integrator import (
    KnowledgeIntegrator,
    KnowledgeItem,
    IntegrationStrategy,
    IntegrationStatus,
    CompatibilityCheck
)
from .safety_rollback_controller import (
    SafetyRollbackController,
    RollbackPlan,
    SafetyCheck,
    DeterministicOperation,
    SafetyLevel
)
from .networked_collaboration import (
    NetworkedResearchCollaboration,
    ResearchAgent,
    CollaborationProtocol,
    CollaborationStatus,
    Collaboration,
    KnowledgeFusion,
    CrossDomainInsight
)
from .auto_fix_workflow import (
    AutoFixWorkflow,
    ErrorDetector,
    PatchGenerator,
    ValidationEngine,
    RollbackTrigger
)
from .performance_regression import (
    PerformanceRegressionMonitor,
    BenchmarkSuite,
    RegressionDetector,
    PerformanceMetrics,
    OptimizationSuggestion
)

__all__ = [
    # Research Project Generator
    'ResearchProjectGenerator', 'ResearchProject', 'ProjectPhase', 'ProjectStatus',
    'ResearchDomain', 'ComplexityLevel', 'InnovationType',
    
    # Autonomous Experimenter
    'AutonomousExperimenter', 'Experiment', 'ExperimentType', 'ExperimentStatus',
    'ResourceManager', 'ConstraintManager',
    
    # Discovery Analyzer
    'DiscoveryAnalyzer', 'Discovery', 'DiscoveryType', 'NoveltyLevel',
    'SignificanceLevel', 'ValidationResult',
    
    # Knowledge Integrator
    'KnowledgeIntegrator', 'KnowledgeItem', 'IntegrationStrategy',
    'IntegrationStatus', 'CompatibilityCheck',
    
    # Safety Rollback Controller
    'SafetyRollbackController', 'RollbackPlan', 'SafetyCheck',
    'DeterministicOperation', 'SafetyLevel',
    
    # Networked Collaboration
    'NetworkedResearchCollaboration', 'ResearchAgent', 'CollaborationProtocol',
    'CollaborationStatus', 'Collaboration', 'KnowledgeFusion', 'CrossDomainInsight',
    
    # Auto-Fix Workflow
    'AutoFixWorkflow', 'ErrorDetector', 'PatchGenerator', 'ValidationEngine', 'RollbackTrigger',
    
    # Performance Regression
    'PerformanceRegressionMonitor', 'BenchmarkSuite', 'RegressionDetector',
    'PerformanceMetrics', 'OptimizationSuggestion'
]
