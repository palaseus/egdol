"""
Strategic Feedback Loop for OmniMind Civilization Intelligence Layer
Feeds top-performing civilization blueprints back into OmniMind ProgenyGenerator.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from .core_structures import (
    Civilization, AgentCluster, EnvironmentState, TemporalState,
    CivilizationIntelligenceCore, GovernanceModel, AgentType, CivilizationArchetype
)
from .pattern_codification_engine import PatternBlueprint, BlueprintType, StrategicDoctrine
from .civilizational_genetic_archive import CivilizationDNA, LineageType, ArchiveStatus


class FeedbackType(Enum):
    """Types of strategic feedback."""
    BLUEPRINT_INTEGRATION = auto()
    DOCTRINE_APPLICATION = auto()
    DNA_MUTATION = auto()
    LINEAGE_EVOLUTION = auto()
    PERFORMANCE_OPTIMIZATION = auto()


class FeedbackStatus(Enum):
    """Status of feedback application."""
    PENDING = auto()
    APPLYING = auto()
    APPLIED = auto()
    FAILED = auto()
    REVERTED = auto()


@dataclass
class FeedbackApplication:
    """Record of a feedback application."""
    id: str
    feedback_type: FeedbackType
    source_blueprint_id: Optional[str]
    source_dna_id: Optional[str]
    target_civilization_id: str
    
    # Application details
    application_timestamp: datetime
    status: FeedbackStatus
    success_metrics: Dict[str, float] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    
    # Performance tracking
    pre_application_metrics: Dict[str, float] = field(default_factory=dict)
    post_application_metrics: Dict[str, float] = field(default_factory=dict)
    performance_delta: Dict[str, float] = field(default_factory=dict)
    
    # Rollback information
    rollback_data: Dict[str, Any] = field(default_factory=dict)
    can_rollback: bool = True


@dataclass
class FeedbackPipeline:
    """Pipeline for processing strategic feedback."""
    id: str
    name: str
    pipeline_type: FeedbackType
    
    # Pipeline configuration
    source_criteria: Dict[str, Any]
    target_criteria: Dict[str, Any]
    application_rules: Dict[str, Any]
    performance_thresholds: Dict[str, float]
    
    # Pipeline state
    is_active: bool = True
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    # Performance tracking
    average_improvement: float = 0.0
    best_improvement: float = 0.0
    worst_improvement: float = 0.0


class StrategicFeedbackLoop:
    """Strategic feedback loop for civilization intelligence evolution."""
    
    def __init__(self, core: CivilizationIntelligenceCore, 
                 pattern_codification_engine, genetic_archive):
        """Initialize the strategic feedback loop."""
        self.core = core
        self.pattern_codification_engine = pattern_codification_engine
        self.genetic_archive = genetic_archive
        
        # Feedback storage
        self.feedback_applications: Dict[str, FeedbackApplication] = {}
        self.feedback_pipelines: Dict[str, FeedbackPipeline] = {}
        
        # Feedback parameters
        self.feedback_parameters = {
            'min_blueprint_performance': 0.7,
            'min_dna_fitness': 0.6,
            'feedback_application_timeout': 30.0,  # seconds
            'rollback_threshold': 0.1,  # Performance degradation threshold
            'max_feedback_applications_per_civilization': 10,
            'feedback_evaluation_period': 7  # days
        }
        
        # Feedback statistics
        self.feedback_stats = {
            'total_applications': 0,
            'successful_applications': 0,
            'failed_applications': 0,
            'rolled_back_applications': 0,
            'average_improvement': 0.0,
            'best_improvement': 0.0,
            'pipeline_executions': 0,
            'pipeline_successes': 0,
            'pipeline_failures': 0
        }
        
        # Performance tracking
        self.performance_tracking = {
            'feedback_application_times': [],
            'performance_improvements': [],
            'rollback_rates': [],
            'pipeline_success_rates': []
        }
        
        # Initialize default pipelines
        self._initialize_default_pipelines()
    
    def _initialize_default_pipelines(self):
        """Initialize default feedback pipelines."""
        # Blueprint integration pipeline
        blueprint_pipeline = FeedbackPipeline(
            id=str(uuid.uuid4()),
            name="Blueprint Integration Pipeline",
            pipeline_type=FeedbackType.BLUEPRINT_INTEGRATION,
            source_criteria={
                'min_validation_score': 0.7,
                'min_success_rate': 0.6,
                'blueprint_types': [BlueprintType.GOVERNANCE_MODEL, BlueprintType.RESOURCE_STRATEGY]
            },
            target_criteria={
                'min_population': 50,
                'compatible_archetypes': [CivilizationArchetype.COOPERATIVE, CivilizationArchetype.HIERARCHICAL]
            },
            application_rules={
                'max_applications_per_civilization': 3,
                'application_interval': 24,  # hours
                'performance_monitoring_period': 7  # days
            },
            performance_thresholds={
                'min_improvement': 0.05,
                'max_degradation': 0.1,
                'stability_threshold': 0.6
            }
        )
        self.feedback_pipelines[blueprint_pipeline.id] = blueprint_pipeline
        
        # DNA mutation pipeline
        dna_pipeline = FeedbackPipeline(
            id=str(uuid.uuid4()),
            name="DNA Mutation Pipeline",
            pipeline_type=FeedbackType.DNA_MUTATION,
            source_criteria={
                'min_fitness_score': 0.8,
                'lineage_types': [LineageType.TECHNOLOGICAL_PATHWAY, LineageType.STRATEGIC_DOCTRINE]
            },
            target_criteria={
                'min_population': 100,
                'innovation_capacity': 0.5
            },
            application_rules={
                'max_applications_per_civilization': 2,
                'application_interval': 48,  # hours
                'mutation_stability_period': 14  # days
            },
            performance_thresholds={
                'min_improvement': 0.1,
                'max_degradation': 0.15,
                'adaptability_threshold': 0.7
            }
        )
        self.feedback_pipelines[dna_pipeline.id] = dna_pipeline
    
    def process_strategic_feedback(self, civilization_id: str) -> List[str]:
        """Process strategic feedback for a civilization."""
        civilization = self.core.get_civilization(civilization_id)
        if not civilization:
            return []
        
        applied_feedbacks = []
        
        # Process each active pipeline
        for pipeline_id, pipeline in self.feedback_pipelines.items():
            if not pipeline.is_active:
                continue
            
            # Check if civilization meets target criteria
            if not self._evaluate_target_criteria(civilization, pipeline.target_criteria):
                continue
            
            # Find suitable feedback sources
            feedback_sources = self._find_feedback_sources(pipeline.source_criteria)
            if not feedback_sources:
                continue
            
            # Apply feedback
            for source in feedback_sources:
                try:
                    feedback_id = self._apply_feedback(
                        civilization_id, source, pipeline
                    )
                    if feedback_id:
                        applied_feedbacks.append(feedback_id)
                        
                        # Update pipeline statistics
                        pipeline.execution_count += 1
                        self.feedback_stats['pipeline_executions'] += 1
                        
                except Exception as e:
                    print(f"Error applying feedback: {e}")
                    pipeline.failure_count += 1
                    self.feedback_stats['pipeline_failures'] += 1
        
        return applied_feedbacks
    
    def _evaluate_target_criteria(self, civilization: Civilization, 
                                criteria: Dict[str, Any]) -> bool:
        """Evaluate if a civilization meets target criteria."""
        # Check population criteria
        if 'min_population' in criteria:
            if civilization.total_population < criteria['min_population']:
                return False
        
        # Check archetype compatibility
        if 'compatible_archetypes' in criteria:
            if civilization.archetype.name not in criteria['compatible_archetypes']:
                return False
        
        # Check performance criteria
        if 'innovation_capacity' in criteria:
            if civilization.innovation_capacity < criteria['innovation_capacity']:
                return False
        
        if 'stability' in criteria:
            if civilization.stability < criteria['stability']:
                return False
        
        if 'cooperation_level' in criteria:
            if civilization.cooperation_level < criteria['cooperation_level']:
                return False
        
        return True
    
    def _find_feedback_sources(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find suitable feedback sources based on criteria."""
        sources = []
        
        # Find blueprints
        if 'min_validation_score' in criteria:
            blueprints = self.pattern_codification_engine.get_high_performing_blueprints(
                criteria['min_validation_score']
            )
            for blueprint in blueprints:
                if self._evaluate_blueprint_criteria(blueprint, criteria):
                    sources.append({
                        'type': 'blueprint',
                        'id': blueprint.id,
                        'data': blueprint
                    })
        
        # Find DNA sequences
        if 'min_fitness_score' in criteria:
            dna_sequences = self.genetic_archive.search_dna_by_fitness(
                criteria['min_fitness_score']
            )
            for dna in dna_sequences:
                if self._evaluate_dna_criteria(dna, criteria):
                    sources.append({
                        'type': 'dna',
                        'id': dna.id,
                        'data': dna
                    })
        
        return sources
    
    def _evaluate_blueprint_criteria(self, blueprint: PatternBlueprint, 
                                   criteria: Dict[str, Any]) -> bool:
        """Evaluate if a blueprint meets criteria."""
        # Check validation score
        if 'min_validation_score' in criteria:
            if blueprint.validation_score < criteria['min_validation_score']:
                return False
        
        # Check success rate
        if 'min_success_rate' in criteria:
            if blueprint.success_rate < criteria['min_success_rate']:
                return False
        
        # Check blueprint types
        if 'blueprint_types' in criteria:
            if blueprint.blueprint_type not in criteria['blueprint_types']:
                return False
        
        return True
    
    def _evaluate_dna_criteria(self, dna: CivilizationDNA, 
                             criteria: Dict[str, Any]) -> bool:
        """Evaluate if a DNA sequence meets criteria."""
        # Check fitness score
        if 'min_fitness_score' in criteria:
            if dna.fitness_score < criteria['min_fitness_score']:
                return False
        
        # Check lineage types
        if 'lineage_types' in criteria:
            if dna.lineage_type not in criteria['lineage_types']:
                return False
        
        # Check status
        if dna.status != ArchiveStatus.ACTIVE:
            return False
        
        return True
    
    def _apply_feedback(self, civilization_id: str, source: Dict[str, Any], 
                       pipeline: FeedbackPipeline) -> Optional[str]:
        """Apply feedback to a civilization."""
        try:
            # Create feedback application record
            application = FeedbackApplication(
                id=str(uuid.uuid4()),
                feedback_type=pipeline.pipeline_type,
                source_blueprint_id=source['id'] if source['type'] == 'blueprint' else None,
                source_dna_id=source['id'] if source['type'] == 'dna' else None,
                target_civilization_id=civilization_id,
                application_timestamp=datetime.now(),
                status=FeedbackStatus.APPLYING
            )
            
            # Record pre-application metrics
            civilization = self.core.get_civilization(civilization_id)
            application.pre_application_metrics = self._get_civilization_metrics(civilization)
            
            # Create rollback data
            application.rollback_data = self._create_rollback_data(civilization)
            
            # Apply feedback based on type
            if source['type'] == 'blueprint':
                success = self._apply_blueprint_feedback(civilization, source['data'], application)
            elif source['type'] == 'dna':
                success = self._apply_dna_feedback(civilization, source['data'], application)
            else:
                success = False
            
            # Update application status
            if success:
                application.status = FeedbackStatus.APPLIED
                application.success_metrics = self._get_civilization_metrics(civilization)
                application.post_application_metrics = application.success_metrics
                
                # Calculate performance delta
                application.performance_delta = self._calculate_performance_delta(
                    application.pre_application_metrics, application.post_application_metrics
                )
                
                # Update statistics
                self.feedback_stats['successful_applications'] += 1
                pipeline.success_count += 1
                
            else:
                application.status = FeedbackStatus.FAILED
                self.feedback_stats['failed_applications'] += 1
                pipeline.failure_count += 1
            
            # Store application
            self.feedback_applications[application.id] = application
            self.feedback_stats['total_applications'] += 1
            
            return application.id if success else None
            
        except Exception as e:
            print(f"Error applying feedback: {e}")
            return None
    
    def _apply_blueprint_feedback(self, civilization: Civilization, 
                                blueprint: PatternBlueprint, 
                                application: FeedbackApplication) -> bool:
        """Apply blueprint feedback to a civilization."""
        try:
            # Apply blueprint based on type
            if blueprint.blueprint_type == BlueprintType.GOVERNANCE_MODEL:
                return self._apply_governance_blueprint(civilization, blueprint)
            elif blueprint.blueprint_type == BlueprintType.RESOURCE_STRATEGY:
                return self._apply_resource_blueprint(civilization, blueprint)
            elif blueprint.blueprint_type == BlueprintType.COMMUNICATION_PROTOCOL:
                return self._apply_communication_blueprint(civilization, blueprint)
            elif blueprint.blueprint_type == BlueprintType.INNOVATION_PATHWAY:
                return self._apply_innovation_blueprint(civilization, blueprint)
            elif blueprint.blueprint_type == BlueprintType.CULTURAL_MEME:
                return self._apply_cultural_blueprint(civilization, blueprint)
            else:
                return False
                
        except Exception as e:
            print(f"Error applying blueprint feedback: {e}")
            return False
    
    def _apply_governance_blueprint(self, civilization: Civilization, 
                                   blueprint: PatternBlueprint) -> bool:
        """Apply governance blueprint to civilization."""
        try:
            # Extract governance requirements
            requirements = blueprint.implementation_requirements
            
            # Update governance model
            if 'governance_model' in requirements:
                civilization.governance_model = requirements['governance_model']
            
            # Update decision making efficiency
            if 'decision_making_efficiency' in requirements:
                civilization.decision_making_efficiency = requirements['decision_making_efficiency']
            
            # Update hierarchy levels
            if 'hierarchy_levels' in requirements:
                civilization.governance_model.hierarchy_levels = requirements['hierarchy_levels']
            
            return True
            
        except Exception as e:
            print(f"Error applying governance blueprint: {e}")
            return False
    
    def _apply_resource_blueprint(self, civilization: Civilization, 
                                blueprint: PatternBlueprint) -> bool:
        """Apply resource blueprint to civilization."""
        try:
            # Extract resource requirements
            requirements = blueprint.implementation_requirements
            
            # Update resource efficiency
            if 'resource_efficiency' in requirements:
                civilization.resource_efficiency.update(requirements['resource_efficiency'])
            
            # Update resource allocation strategy
            if 'resource_allocation_strategy' in requirements:
                civilization.resource_allocation_strategy = requirements['resource_allocation_strategy']
            
            # Update resource management approach
            if 'resource_management_approach' in requirements:
                civilization.resource_management_approach = requirements['resource_management_approach']
            
            return True
            
        except Exception as e:
            print(f"Error applying resource blueprint: {e}")
            return False
    
    def _apply_communication_blueprint(self, civilization: Civilization, 
                                      blueprint: PatternBlueprint) -> bool:
        """Apply communication blueprint to civilization."""
        try:
            # Extract communication requirements
            requirements = blueprint.implementation_requirements
            
            # Update communication efficiency
            if 'communication_efficiency' in requirements:
                civilization.communication_efficiency = requirements['communication_efficiency']
            
            # Update network topology
            if 'network_topology' in requirements:
                civilization.communication_network_topology = requirements['network_topology']
            
            # Update communication protocols
            if 'communication_protocols' in requirements:
                civilization.communication_protocols = requirements['communication_protocols']
            
            return True
            
        except Exception as e:
            print(f"Error applying communication blueprint: {e}")
            return False
    
    def _apply_innovation_blueprint(self, civilization: Civilization, 
                                   blueprint: PatternBlueprint) -> bool:
        """Apply innovation blueprint to civilization."""
        try:
            # Extract innovation requirements
            requirements = blueprint.implementation_requirements
            
            # Update innovation capacity
            if 'innovation_capacity' in requirements:
                civilization.innovation_capacity = requirements['innovation_capacity']
            
            # Update innovation rate
            if 'innovation_rate' in requirements:
                civilization.innovation_rate = requirements['innovation_rate']
            
            # Update innovation focus areas
            if 'innovation_focus_areas' in requirements:
                civilization.innovation_focus_areas = requirements['innovation_focus_areas']
            
            return True
            
        except Exception as e:
            print(f"Error applying innovation blueprint: {e}")
            return False
    
    def _apply_cultural_blueprint(self, civilization: Civilization, 
                                blueprint: PatternBlueprint) -> bool:
        """Apply cultural blueprint to civilization."""
        try:
            # Extract cultural requirements
            requirements = blueprint.implementation_requirements
            
            # Update cooperation level
            if 'cooperation_level' in requirements:
                civilization.cooperation_level = requirements['cooperation_level']
            
            # Update cultural values
            if 'cultural_values' in requirements:
                civilization.cultural_values = requirements['cultural_values']
            
            # Update social norms
            if 'social_norms' in requirements:
                civilization.social_norms = requirements['social_norms']
            
            return True
            
        except Exception as e:
            print(f"Error applying cultural blueprint: {e}")
            return False
    
    def _apply_dna_feedback(self, civilization: Civilization, 
                          dna: CivilizationDNA, 
                          application: FeedbackApplication) -> bool:
        """Apply DNA feedback to a civilization."""
        try:
            # Apply DNA mutations based on lineage type
            if dna.lineage_type == LineageType.GOVERNANCE_EVOLUTION:
                return self._apply_governance_dna(civilization, dna)
            elif dna.lineage_type == LineageType.TECHNOLOGICAL_PATHWAY:
                return self._apply_technology_dna(civilization, dna)
            elif dna.lineage_type == LineageType.CULTURAL_TRADITION:
                return self._apply_cultural_dna(civilization, dna)
            elif dna.lineage_type == LineageType.STRATEGIC_DOCTRINE:
                return self._apply_strategic_dna(civilization, dna)
            else:
                return False
                
        except Exception as e:
            print(f"Error applying DNA feedback: {e}")
            return False
    
    def _apply_governance_dna(self, civilization: Civilization, dna: CivilizationDNA) -> bool:
        """Apply governance DNA to civilization."""
        try:
            # Extract governance genome
            governance_genome = dna.governance_genome
            
            # Apply governance characteristics
            if 'decision_efficiency' in governance_genome:
                civilization.decision_making_efficiency = governance_genome['decision_efficiency']
            
            if 'hierarchy_levels' in governance_genome:
                civilization.governance_model.hierarchy_levels = governance_genome['hierarchy_levels']
            
            if 'decision_making_process' in governance_genome:
                civilization.governance_model.decision_making_process = governance_genome['decision_making_process']
            
            return True
            
        except Exception as e:
            print(f"Error applying governance DNA: {e}")
            return False
    
    def _apply_technology_dna(self, civilization: Civilization, dna: CivilizationDNA) -> bool:
        """Apply technology DNA to civilization."""
        try:
            # Extract innovation genome
            innovation_genome = dna.innovation_genome
            
            # Apply innovation characteristics
            if 'innovation_capacity' in innovation_genome:
                civilization.innovation_capacity = innovation_genome['innovation_capacity']
            
            if 'innovation_rate' in innovation_genome:
                civilization.innovation_rate = innovation_genome['innovation_rate']
            
            if 'innovation_focus_areas' in innovation_genome:
                civilization.innovation_focus_areas = innovation_genome['innovation_focus_areas']
            
            return True
            
        except Exception as e:
            print(f"Error applying technology DNA: {e}")
            return False
    
    def _apply_cultural_dna(self, civilization: Civilization, dna: CivilizationDNA) -> bool:
        """Apply cultural DNA to civilization."""
        try:
            # Extract cultural genome
            cultural_genome = dna.cultural_genome
            
            # Apply cultural characteristics
            if 'cooperation_level' in cultural_genome:
                civilization.cooperation_level = cultural_genome['cooperation_level']
            
            if 'cultural_values' in cultural_genome:
                civilization.cultural_values = cultural_genome['cultural_values']
            
            if 'social_norms' in cultural_genome:
                civilization.social_norms = cultural_genome['social_norms']
            
            return True
            
        except Exception as e:
            print(f"Error applying cultural DNA: {e}")
            return False
    
    def _apply_strategic_dna(self, civilization: Civilization, dna: CivilizationDNA) -> bool:
        """Apply strategic DNA to civilization."""
        try:
            # Apply strategic characteristics from all genomes
            governance_genome = dna.governance_genome
            resource_genome = dna.resource_genome
            communication_genome = dna.communication_genome
            innovation_genome = dna.innovation_genome
            cultural_genome = dna.cultural_genome
            
            # Apply strategic characteristics
            if 'decision_efficiency' in governance_genome:
                civilization.decision_making_efficiency = governance_genome['decision_efficiency']
            
            if 'resource_efficiency' in resource_genome:
                civilization.resource_efficiency.update(resource_genome['resource_efficiency'])
            
            if 'communication_efficiency' in communication_genome:
                civilization.communication_efficiency = communication_genome['communication_efficiency']
            
            if 'innovation_capacity' in innovation_genome:
                civilization.innovation_capacity = innovation_genome['innovation_capacity']
            
            if 'cooperation_level' in cultural_genome:
                civilization.cooperation_level = cultural_genome['cooperation_level']
            
            return True
            
        except Exception as e:
            print(f"Error applying strategic DNA: {e}")
            return False
    
    def _get_civilization_metrics(self, civilization: Civilization) -> Dict[str, float]:
        """Get current civilization metrics."""
        return {
            'stability': civilization.stability,
            'complexity': civilization.complexity,
            'adaptability': civilization.adaptability,
            'resilience': civilization.resilience,
            'innovation_capacity': civilization.innovation_capacity,
            'cooperation_level': civilization.cooperation_level,
            'decision_making_efficiency': civilization.decision_making_efficiency,
            'communication_efficiency': civilization.communication_efficiency
        }
    
    def _create_rollback_data(self, civilization: Civilization) -> Dict[str, Any]:
        """Create rollback data for a civilization."""
        return {
            'governance_model': civilization.governance_model.__dict__.copy(),
            'resource_efficiency': civilization.resource_efficiency.copy(),
            'communication_efficiency': civilization.communication_efficiency,
            'innovation_capacity': civilization.innovation_capacity,
            'cooperation_level': civilization.cooperation_level,
            'decision_making_efficiency': civilization.decision_making_efficiency
        }
    
    def _calculate_performance_delta(self, pre_metrics: Dict[str, float], 
                                  post_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance delta between pre and post application."""
        delta = {}
        for key in pre_metrics:
            if key in post_metrics:
                delta[key] = post_metrics[key] - pre_metrics[key]
        return delta
    
    def rollback_feedback(self, application_id: str) -> bool:
        """Rollback a feedback application."""
        application = self.feedback_applications.get(application_id)
        if not application or not application.can_rollback:
            return False
        
        try:
            civilization = self.core.get_civilization(application.target_civilization_id)
            if not civilization:
                return False
            
            # Restore civilization state
            rollback_data = application.rollback_data
            
            # Restore governance model
            if 'governance_model' in rollback_data:
                civilization.governance_model.__dict__.update(rollback_data['governance_model'])
            
            # Restore resource efficiency
            if 'resource_efficiency' in rollback_data:
                civilization.resource_efficiency = rollback_data['resource_efficiency']
            
            # Restore other attributes
            for attr, value in rollback_data.items():
                if hasattr(civilization, attr):
                    setattr(civilization, attr, value)
            
            # Update application status
            application.status = FeedbackStatus.REVERTED
            
            # Update statistics
            self.feedback_stats['rolled_back_applications'] += 1
            
            return True
            
        except Exception as e:
            print(f"Error rolling back feedback: {e}")
            return False
    
    def evaluate_feedback_performance(self, application_id: str) -> Dict[str, Any]:
        """Evaluate the performance of a feedback application."""
        application = self.feedback_applications.get(application_id)
        if not application:
            return {}
        
        # Calculate performance metrics
        performance_metrics = {
            'application_id': application_id,
            'feedback_type': application.feedback_type.name,
            'status': application.status.name,
            'performance_delta': application.performance_delta,
            'overall_improvement': sum(application.performance_delta.values()) / len(application.performance_delta) if application.performance_delta else 0.0,
            'success_metrics': application.success_metrics,
            'application_timestamp': application.application_timestamp.isoformat()
        }
        
        return performance_metrics
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback loop statistics."""
        return {
            'feedback_stats': self.feedback_stats.copy(),
            'total_pipelines': len(self.feedback_pipelines),
            'active_pipelines': len([p for p in self.feedback_pipelines.values() if p.is_active]),
            'total_applications': len(self.feedback_applications),
            'successful_applications': len([a for a in self.feedback_applications.values() if a.status == FeedbackStatus.APPLIED]),
            'failed_applications': len([a for a in self.feedback_applications.values() if a.status == FeedbackStatus.FAILED]),
            'rolled_back_applications': len([a for a in self.feedback_applications.values() if a.status == FeedbackStatus.REVERTED]),
            'average_improvement': self._calculate_average_improvement(),
            'best_improvement': self._calculate_best_improvement(),
            'worst_improvement': self._calculate_worst_improvement()
        }
    
    def _calculate_average_improvement(self) -> float:
        """Calculate average improvement across all applications."""
        improvements = []
        for application in self.feedback_applications.values():
            if application.performance_delta:
                overall_improvement = sum(application.performance_delta.values()) / len(application.performance_delta)
                improvements.append(overall_improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _calculate_best_improvement(self) -> float:
        """Calculate best improvement across all applications."""
        improvements = []
        for application in self.feedback_applications.values():
            if application.performance_delta:
                overall_improvement = sum(application.performance_delta.values()) / len(application.performance_delta)
                improvements.append(overall_improvement)
        
        return max(improvements) if improvements else 0.0
    
    def _calculate_worst_improvement(self) -> float:
        """Calculate worst improvement across all applications."""
        improvements = []
        for application in self.feedback_applications.values():
            if application.performance_delta:
                overall_improvement = sum(application.performance_delta.values()) / len(application.performance_delta)
                improvements.append(overall_improvement)
        
        return min(improvements) if improvements else 0.0
