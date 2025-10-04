"""
Civilization Experimentation & Meta-Evaluation System for OmniMind
Runs controlled civilization-scale experiments with deterministic rollback.
"""

import uuid
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import time
import json

from .core_structures import Civilization, AgentCluster, CivilizationIntelligenceCore
from ...utils.pretty_printing import print_experiment_results, pp


class ExperimentType(Enum):
    """Types of civilization experiments."""
    GOVERNANCE_EXPERIMENT = auto()
    TECHNOLOGY_EVOLUTION = auto()
    CULTURAL_MEME_PROPAGATION = auto()
    KNOWLEDGE_NETWORK_TOPOLOGY = auto()
    RESOURCE_SCARCITY = auto()
    COOPERATION_VS_COMPETITION = auto()
    INNOVATION_RATE = auto()
    ADAPTABILITY_STRESS = auto()
    RESILIENCE_TESTING = auto()
    EMERGENT_BEHAVIOR = auto()


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class ExperimentScenario:
    """Represents an experiment scenario."""
    id: str
    name: str
    description: str
    experiment_type: ExperimentType
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Results of a civilization experiment."""
    experiment_id: str
    scenario_id: str
    civilization_id: str
    
    # Experiment data
    start_time: datetime
    end_time: datetime
    duration: int  # in ticks
    
    # Results
    success: bool
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    behavioral_changes: Dict[str, Any] = field(default_factory=dict)
    emergent_patterns: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    
    # Data for analysis
    time_series_data: Dict[str, List[float]] = field(default_factory=dict)
    snapshot_data: Dict[str, Any] = field(default_factory=dict)
    
    # Meta-analysis
    significance_score: float = 0.0
    reproducibility_score: float = 0.0
    generalizability_score: float = 0.0


@dataclass
class ExperimentBatch:
    """A batch of related experiments."""
    id: str
    name: str
    description: str
    scenario: ExperimentScenario
    civilization_ids: List[str]
    parameter_variations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Batch state
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Results
    results: List[ExperimentResult] = field(default_factory=list)
    batch_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Meta-evaluation
    statistical_significance: float = 0.0
    effect_size: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


class CivilizationExperimentationSystem:
    """Manages civilization-scale experiments and meta-evaluation."""
    
    def __init__(self, core: CivilizationIntelligenceCore):
        """Initialize the experimentation system."""
        self.core = core
        self.active_experiments: Dict[str, ExperimentBatch] = {}
        self.experiment_history: List[ExperimentResult] = []
        self.scenario_templates: Dict[ExperimentType, ExperimentScenario] = {}
        
        # Experimentation parameters
        self.experimentation_parameters = {
            'max_concurrent_experiments': 5,
            'default_experiment_duration': 1000,
            'snapshot_interval': 10,
            'rollback_checkpoint_interval': 50,
            'statistical_significance_threshold': 0.05,
            'minimum_sample_size': 10
        }
        
        # Meta-evaluation metrics
        self.meta_evaluation_metrics = {
            'total_experiments': 0,
            'successful_experiments': 0,
            'failed_experiments': 0,
            'average_significance': 0.0,
            'reproducibility_rate': 0.0,
            'generalizability_score': 0.0
        }
        
        # Threading for parallel experiments
        self.experiment_threads: Dict[str, threading.Thread] = {}
        self.experiment_lock = threading.Lock()
        
        # Initialize scenario templates
        self._initialize_scenario_templates()
    
    def _initialize_scenario_templates(self):
        """Initialize experiment scenario templates."""
        # Governance experiment template
        self.scenario_templates[ExperimentType.GOVERNANCE_EXPERIMENT] = ExperimentScenario(
            id="gov_template",
            name="Governance Model Experiment",
            description="Test different governance models under various conditions",
            experiment_type=ExperimentType.GOVERNANCE_EXPERIMENT,
            parameters={
                'governance_models': ['democratic', 'autocratic', 'consensus', 'meritocratic'],
                'stress_conditions': ['resource_scarcity', 'external_threat', 'internal_conflict'],
                'performance_metrics': ['stability', 'efficiency', 'adaptability', 'resilience']
            },
            constraints={
                'minimum_population': 50,
                'maximum_duration': 2000,
                'resource_limits': {'energy': 1000, 'materials': 500}
            },
            success_criteria=['stability > 0.7', 'efficiency > 0.6', 'adaptability > 0.5'],
            failure_conditions=['stability < 0.3', 'population < 10', 'resource_exhaustion']
        )
        
        # Technology evolution template
        self.scenario_templates[ExperimentType.TECHNOLOGY_EVOLUTION] = ExperimentScenario(
            id="tech_template",
            name="Technology Evolution Experiment",
            description="Study technology evolution under different conditions",
            experiment_type=ExperimentType.TECHNOLOGY_EVOLUTION,
            parameters={
                'innovation_rates': [0.1, 0.3, 0.5, 0.7],
                'knowledge_diffusion_rates': [0.2, 0.4, 0.6, 0.8],
                'specialization_levels': ['low', 'medium', 'high']
            },
            constraints={
                'minimum_knowledge_base': 10,
                'maximum_innovation_rate': 1.0,
                'technology_diversity_required': True
            },
            success_criteria=['innovation_capacity > 0.8', 'knowledge_growth > 0.5'],
            failure_conditions=['innovation_capacity < 0.2', 'knowledge_stagnation']
        )
        
        # Cultural meme propagation template
        self.scenario_templates[ExperimentType.CULTURAL_MEME_PROPAGATION] = ExperimentScenario(
            id="meme_template",
            name="Cultural Meme Propagation Experiment",
            description="Study how cultural memes spread and evolve",
            experiment_type=ExperimentType.CULTURAL_MEME_PROPAGATION,
            parameters={
                'meme_types': ['values', 'beliefs', 'practices', 'traditions'],
                'transmission_rates': [0.1, 0.3, 0.5, 0.7],
                'mutation_rates': [0.05, 0.1, 0.2, 0.3]
            },
            constraints={
                'minimum_cultural_diversity': 0.3,
                'maximum_meme_mutation_rate': 0.5,
                'cultural_stability_required': True
            },
            success_criteria=['cultural_vitality > 0.6', 'meme_persistence > 0.5'],
            failure_conditions=['cultural_stagnation', 'meme_extinction']
        )
        
        # Knowledge network topology template
        self.scenario_templates[ExperimentType.KNOWLEDGE_NETWORK_TOPOLOGY] = ExperimentScenario(
            id="knowledge_template",
            name="Knowledge Network Topology Experiment",
            description="Study knowledge network formation and evolution",
            experiment_type=ExperimentType.KNOWLEDGE_NETWORK_TOPOLOGY,
            parameters={
                'network_densities': [0.1, 0.3, 0.5, 0.7],
                'connection_patterns': ['random', 'scale_free', 'small_world', 'hierarchical'],
                'knowledge_transfer_rates': [0.1, 0.3, 0.5, 0.7]
            },
            constraints={
                'minimum_network_size': 20,
                'maximum_connection_density': 0.8,
                'knowledge_persistence_required': True
            },
            success_criteria=['network_efficiency > 0.6', 'knowledge_retention > 0.5'],
            failure_conditions=['network_fragmentation', 'knowledge_loss']
        )
    
    def create_experiment_batch(self, name: str, description: str,
                              experiment_type: ExperimentType,
                              civilization_ids: List[str],
                              parameter_variations: Optional[List[Dict[str, Any]]] = None,
                              custom_scenario: Optional[ExperimentScenario] = None) -> ExperimentBatch:
        """Create a new experiment batch."""
        batch_id = str(uuid.uuid4())
        
        # Use custom scenario or template
        if custom_scenario:
            scenario = custom_scenario
        else:
            scenario = self.scenario_templates.get(experiment_type)
            if not scenario:
                raise ValueError(f"No template found for experiment type: {experiment_type}")
        
        # Generate parameter variations if not provided
        if parameter_variations is None:
            parameter_variations = self._generate_parameter_variations(scenario)
        
        batch = ExperimentBatch(
            id=batch_id,
            name=name,
            description=description,
            scenario=scenario,
            civilization_ids=civilization_ids,
            parameter_variations=parameter_variations
        )
        
        self.active_experiments[batch_id] = batch
        return batch
    
    def _generate_parameter_variations(self, scenario: ExperimentScenario) -> List[Dict[str, Any]]:
        """Generate parameter variations for an experiment."""
        variations = []
        
        # Generate variations based on scenario parameters
        for param_name, param_values in scenario.parameters.items():
            if isinstance(param_values, list):
                for value in param_values:
                    variation = {param_name: value}
                    variations.append(variation)
        
        # If no variations generated, create default variation
        if not variations:
            variations = [{}]
        
        return variations
    
    def start_experiment_batch(self, batch_id: str) -> bool:
        """Start an experiment batch."""
        if batch_id not in self.active_experiments:
            return False
        
        batch = self.active_experiments[batch_id]
        batch.status = ExperimentStatus.RUNNING
        batch.start_time = datetime.now()
        
        # Start experiment thread
        thread = threading.Thread(
            target=self._run_experiment_batch,
            args=(batch_id,),
            daemon=True
        )
        self.experiment_threads[batch_id] = thread
        thread.start()
        
        return True
    
    def _run_experiment_batch(self, batch_id: str):
        """Run an experiment batch."""
        batch = self.active_experiments[batch_id]
        
        try:
            # Run experiments for each civilization and parameter variation
            for civ_id in batch.civilization_ids:
                for variation in batch.parameter_variations:
                    # Create experiment result
                    result = self._run_single_experiment(
                        batch.scenario, civ_id, variation, batch_id
                    )
                    
                    if result:
                        batch.results.append(result)
                        self.experiment_history.append(result)
            
            # Complete batch
            batch.status = ExperimentStatus.COMPLETED
            batch.end_time = datetime.now()
            
            # Perform meta-evaluation
            self._perform_meta_evaluation(batch)
            
        except Exception as e:
            print(f"Experiment batch error: {e}")
            batch.status = ExperimentStatus.FAILED
            batch.end_time = datetime.now()
    
    def _run_single_experiment(self, scenario: ExperimentScenario, civilization_id: str,
                             parameters: Dict[str, Any], batch_id: str) -> Optional[ExperimentResult]:
        """Run a single experiment."""
        civilization = self.core.get_civilization(civilization_id)
        if not civilization:
            return None
        
        # Create experiment result
        result = ExperimentResult(
            experiment_id=str(uuid.uuid4()),
            scenario_id=scenario.id,
            civilization_id=civilization_id,
            start_time=datetime.now(),
            end_time=datetime.now(),  # Will be updated later
            duration=0,  # Will be updated later
            success=False  # Will be updated later
        )
        
        # Create civilization snapshot for rollback
        snapshot = self._create_civilization_snapshot(civilization)
        
        try:
            # Apply experiment parameters
            self._apply_experiment_parameters(civilization, parameters)
            
            # Run experiment based on type
            if scenario.experiment_type == ExperimentType.GOVERNANCE_EXPERIMENT:
                self._run_governance_experiment(civilization, scenario, result)
            elif scenario.experiment_type == ExperimentType.TECHNOLOGY_EVOLUTION:
                self._run_technology_experiment(civilization, scenario, result)
            elif scenario.experiment_type == ExperimentType.CULTURAL_MEME_PROPAGATION:
                self._run_cultural_experiment(civilization, scenario, result)
            else:
                self._run_generic_experiment(civilization, scenario, result)
            
            # Evaluate experiment success
            result.success = self._evaluate_experiment_success(civilization, scenario)
            
            # Record final state
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.snapshot_data = self._create_civilization_snapshot(civilization)
            
            # Calculate meta-evaluation scores
            result.significance_score = self._calculate_significance_score(result)
            result.reproducibility_score = self._calculate_reproducibility_score(result)
            result.generalizability_score = self._calculate_generalizability_score(result)
            
        except Exception as e:
            print(f"Experiment error: {e}")
            result.success = False
            result.failure_reasons.append(str(e))
            result.end_time = datetime.now()
        
        finally:
            # Restore civilization state
            self._restore_civilization_snapshot(civilization, snapshot)
        
        return result
    
    def _run_governance_experiment(self, civilization: Civilization, scenario: ExperimentScenario,
                                 result: ExperimentResult):
        """Run a governance experiment."""
        # Apply governance model changes
        if 'governance_models' in scenario.parameters:
            governance_model = random.choice(scenario.parameters['governance_models'])
            civilization.governance_model = governance_model
        
        # Apply stress conditions
        if 'stress_conditions' in scenario.parameters:
            stress_condition = random.choice(scenario.parameters['stress_conditions'])
            self._apply_stress_condition(civilization, stress_condition)
        
        # Monitor performance metrics
        self._monitor_experiment_metrics(civilization, scenario, result)
    
    def _run_technology_experiment(self, civilization: Civilization, scenario: ExperimentScenario,
                                result: ExperimentResult):
        """Run a technology evolution experiment."""
        # Apply innovation rate changes
        if 'innovation_rates' in scenario.parameters:
            innovation_rate = random.choice(scenario.parameters['innovation_rates'])
            civilization.innovation_capacity = innovation_rate
        
        # Apply knowledge diffusion changes
        if 'knowledge_diffusion_rates' in scenario.parameters:
            diffusion_rate = random.choice(scenario.parameters['knowledge_diffusion_rates'])
            civilization.knowledge_diffusion_rate = diffusion_rate
        
        # Monitor technology evolution
        self._monitor_experiment_metrics(civilization, scenario, result)
    
    def _run_cultural_experiment(self, civilization: Civilization, scenario: ExperimentScenario,
                              result: ExperimentResult):
        """Run a cultural meme propagation experiment."""
        # Apply meme transmission rates
        if 'transmission_rates' in scenario.parameters:
            transmission_rate = random.choice(scenario.parameters['transmission_rates'])
            civilization.cooperation_level = transmission_rate
        
        # Apply mutation rates
        if 'mutation_rates' in scenario.parameters:
            mutation_rate = random.choice(scenario.parameters['mutation_rates'])
            civilization.adaptability = mutation_rate
        
        # Monitor cultural evolution
        self._monitor_experiment_metrics(civilization, scenario, result)
    
    def _run_generic_experiment(self, civilization: Civilization, scenario: ExperimentScenario,
                              result: ExperimentResult):
        """Run a generic experiment."""
        # Apply generic parameter changes
        for param_name, param_value in scenario.parameters.items():
            if hasattr(civilization, param_name):
                setattr(civilization, param_name, param_value)
        
        # Monitor generic metrics
        self._monitor_experiment_metrics(civilization, scenario, result)
    
    def _apply_stress_condition(self, civilization: Civilization, stress_condition: str):
        """Apply a stress condition to a civilization."""
        if stress_condition == 'resource_scarcity':
            # Reduce resource pools
            for resource in civilization.resource_pools:
                civilization.resource_pools[resource] *= 0.5
        elif stress_condition == 'external_threat':
            # Reduce stability
            civilization.stability *= 0.7
        elif stress_condition == 'internal_conflict':
            # Reduce cooperation
            civilization.cooperation_level *= 0.6
    
    def _monitor_experiment_metrics(self, civilization: Civilization, scenario: ExperimentScenario,
                                  result: ExperimentResult):
        """Monitor experiment metrics over time."""
        # Record time series data
        time_series_data = {
            'stability': [civilization.stability],
            'complexity': [civilization.complexity],
            'innovation_capacity': [civilization.innovation_capacity],
            'cooperation_level': [civilization.cooperation_level],
            'adaptability': [civilization.adaptability],
            'resilience': [civilization.resilience]
        }
        
        result.time_series_data = time_series_data
        
        # Record performance metrics
        result.performance_metrics = {
            'stability': civilization.stability,
            'complexity': civilization.complexity,
            'innovation_capacity': civilization.innovation_capacity,
            'cooperation_level': civilization.cooperation_level,
            'adaptability': civilization.adaptability,
            'resilience': civilization.resilience
        }
    
    def _evaluate_experiment_success(self, civilization: Civilization, scenario: ExperimentScenario) -> bool:
        """Evaluate if an experiment was successful."""
        # Check success criteria
        for criterion in scenario.success_criteria:
            if not self._evaluate_criterion(civilization, criterion):
                return False
        
        # Check failure conditions
        for condition in scenario.failure_conditions:
            if self._evaluate_criterion(civilization, condition):
                return False
        
        return True
    
    def _evaluate_criterion(self, civilization: Civilization, criterion: str) -> bool:
        """Evaluate a single criterion."""
        try:
            # Simple criterion evaluation (can be extended)
            if 'stability >' in criterion:
                threshold = float(criterion.split('>')[1].strip())
                return civilization.stability > threshold
            elif 'stability <' in criterion:
                threshold = float(criterion.split('<')[1].strip())
                return civilization.stability < threshold
            elif 'innovation_capacity >' in criterion:
                threshold = float(criterion.split('>')[1].strip())
                return civilization.innovation_capacity > threshold
            elif 'cooperation_level >' in criterion:
                threshold = float(criterion.split('>')[1].strip())
                return civilization.cooperation_level > threshold
            else:
                return True  # Default to true for unknown criteria
        except:
            return False
    
    def _create_civilization_snapshot(self, civilization: Civilization) -> Dict[str, Any]:
        """Create a snapshot of civilization state for rollback."""
        return {
            'id': civilization.id,
            'name': civilization.name,
            'governance_model': civilization.governance_model,
            'stability': civilization.stability,
            'complexity': civilization.complexity,
            'innovation_capacity': civilization.innovation_capacity,
            'cooperation_level': civilization.cooperation_level,
            'adaptability': civilization.adaptability,
            'resilience': civilization.resilience,
            'resource_pools': civilization.resource_pools.copy(),
            'knowledge_base': civilization.knowledge_base.copy(),
            'cultural_traits': civilization.cultural_traits.copy(),
            'strategic_capabilities': civilization.strategic_capabilities.copy()
        }
    
    def _restore_civilization_snapshot(self, civilization: Civilization, snapshot: Dict[str, Any]):
        """Restore civilization state from snapshot."""
        civilization.governance_model = snapshot['governance_model']
        civilization.stability = snapshot['stability']
        civilization.complexity = snapshot['complexity']
        civilization.innovation_capacity = snapshot['innovation_capacity']
        civilization.cooperation_level = snapshot['cooperation_level']
        civilization.adaptability = snapshot['adaptability']
        civilization.resilience = snapshot['resilience']
        civilization.resource_pools = snapshot['resource_pools'].copy()
        civilization.knowledge_base = snapshot['knowledge_base'].copy()
        civilization.cultural_traits = snapshot['cultural_traits'].copy()
        civilization.strategic_capabilities = snapshot['strategic_capabilities'].copy()
    
    def _apply_experiment_parameters(self, civilization: Civilization, parameters: Dict[str, Any]):
        """Apply experiment parameters to civilization."""
        for param_name, param_value in parameters.items():
            if hasattr(civilization, param_name):
                setattr(civilization, param_name, param_value)
    
    def _calculate_significance_score(self, result: ExperimentResult) -> float:
        """Calculate statistical significance score for experiment result."""
        # Simple significance calculation based on performance metrics
        if not result.performance_metrics:
            return 0.0
        
        # Calculate variance in performance metrics
        values = list(result.performance_metrics.values())
        if len(values) < 2:
            return 0.0
        
        variance = np.var(values)
        mean_value = np.mean(values)
        
        # Significance based on variance and mean
        significance = min(1.0, variance / mean_value if mean_value > 0 else 0.0)
        return significance
    
    def _calculate_reproducibility_score(self, result: ExperimentResult) -> float:
        """Calculate reproducibility score for experiment result."""
        # Simple reproducibility calculation
        # In a real implementation, this would compare with historical results
        return random.uniform(0.6, 0.9)  # Placeholder
    
    def _calculate_generalizability_score(self, result: ExperimentResult) -> float:
        """Calculate generalizability score for experiment result."""
        # Simple generalizability calculation
        # In a real implementation, this would analyze cross-civilization applicability
        return random.uniform(0.5, 0.8)  # Placeholder
    
    def _perform_meta_evaluation(self, batch: ExperimentBatch):
        """Perform meta-evaluation of experiment batch."""
        if not batch.results:
            return
        
        # Calculate batch metrics
        success_rate = sum(1 for r in batch.results if r.success) / len(batch.results)
        average_significance = np.mean([r.significance_score for r in batch.results])
        average_reproducibility = np.mean([r.reproducibility_score for r in batch.results])
        average_generalizability = np.mean([r.generalizability_score for r in batch.results])
        
        batch.batch_metrics = {
            'success_rate': success_rate,
            'average_significance': average_significance,
            'average_reproducibility': average_reproducibility,
            'average_generalizability': average_generalizability,
            'total_experiments': len(batch.results)
        }
        
        # Calculate statistical significance
        if len(batch.results) >= self.experimentation_parameters['minimum_sample_size']:
            batch.statistical_significance = self._calculate_statistical_significance(batch.results)
        
        # Update meta-evaluation metrics
        self.meta_evaluation_metrics['total_experiments'] += len(batch.results)
        self.meta_evaluation_metrics['successful_experiments'] += sum(1 for r in batch.results if r.success)
        self.meta_evaluation_metrics['failed_experiments'] += sum(1 for r in batch.results if not r.success)
        self.meta_evaluation_metrics['average_significance'] = average_significance
        self.meta_evaluation_metrics['reproducibility_rate'] = average_reproducibility
        self.meta_evaluation_metrics['generalizability_score'] = average_generalizability
    
    def _calculate_statistical_significance(self, results: List[ExperimentResult]) -> float:
        """Calculate statistical significance of experiment results."""
        # Simple statistical significance calculation
        # In a real implementation, this would use proper statistical tests
        if len(results) < 2:
            return 0.0
        
        # Calculate variance in significance scores
        significance_scores = [r.significance_score for r in results]
        variance = np.var(significance_scores)
        mean_significance = np.mean(significance_scores)
        
        # Statistical significance based on variance and mean
        if mean_significance > 0:
            significance = min(1.0, variance / mean_significance)
        else:
            significance = 0.0
        
        return significance
    
    def get_experiment_batch(self, batch_id: str) -> Optional[ExperimentBatch]:
        """Get an experiment batch by ID."""
        return self.active_experiments.get(batch_id)
    
    def get_experiment_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment results for a batch."""
        batch = self.get_experiment_batch(batch_id)
        if not batch:
            return None
        
        return {
            'batch': batch,
            'results': batch.results,
            'batch_metrics': batch.batch_metrics,
            'statistical_significance': batch.statistical_significance
        }
    
    def get_meta_evaluation_metrics(self) -> Dict[str, Any]:
        """Get meta-evaluation metrics."""
        return self.meta_evaluation_metrics.copy()
    
    def get_experiment_history(self) -> List[ExperimentResult]:
        """Get experiment history."""
        return self.experiment_history.copy()
    
    def stop_experiment_batch(self, batch_id: str) -> bool:
        """Stop an experiment batch."""
        if batch_id in self.active_experiments:
            batch = self.active_experiments[batch_id]
            batch.status = ExperimentStatus.CANCELLED
            batch.end_time = datetime.now()
            return True
        return False
