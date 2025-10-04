"""
Civilization Experimentation System for Transcendence Layer
Runs controlled civilization-scale experiments.
"""

import uuid
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time
from collections import defaultdict, Counter
import statistics


class ExperimentType(Enum):
    """Types of civilization experiments."""
    GOVERNANCE = auto()
    TECHNOLOGY_EVOLUTION = auto()
    CULTURAL_MEME = auto()
    KNOWLEDGE_NETWORK = auto()
    RESOURCE_ALLOCATION = auto()
    POPULATION_DYNAMICS = auto()
    ENVIRONMENTAL_ADAPTATION = auto()
    STRATEGIC_CONFLICT = auto()


@dataclass
class ExperimentResult:
    """Represents the result of a civilization experiment."""
    id: str
    experiment_id: str
    experiment_type: ExperimentType
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GovernanceExperiment:
    """Represents a governance experiment."""
    id: str
    name: str
    description: str
    governance_type: str  # e.g., 'democratic', 'hierarchical', 'consensus'
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: List[str] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    duration: int = 100
    status: str = 'pending'


@dataclass
class TechnologyEvolutionExperiment:
    """Represents a technology evolution experiment."""
    id: str
    name: str
    description: str
    technology_focus: str  # e.g., 'AI', 'biotech', 'energy', 'communication'
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: List[str] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    duration: int = 100
    status: str = 'pending'


@dataclass
class CulturalMemeExperiment:
    """Represents a cultural meme experiment."""
    id: str
    name: str
    description: str
    meme_type: str  # e.g., 'belief', 'practice', 'value', 'tradition'
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: List[str] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    duration: int = 100
    status: str = 'pending'


@dataclass
class KnowledgeNetworkExperiment:
    """Represents a knowledge network experiment."""
    id: str
    name: str
    description: str
    network_topology: str  # e.g., 'centralized', 'distributed', 'hierarchical'
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: List[str] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    duration: int = 100
    status: str = 'pending'


class CivilizationExperimentationSystem:
    """Runs controlled civilization-scale experiments."""
    
    def __init__(self, civilization_architect, temporal_evolution_engine, network, memory_manager, knowledge_graph):
        self.civilization_architect = civilization_architect
        self.temporal_evolution_engine = temporal_evolution_engine
        self.network = network
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph
        
        # Experiment management
        self.active_experiments: Dict[str, Any] = {}
        self.experiment_results: Dict[str, ExperimentResult] = {}
        self.experiment_history: List[Dict[str, Any]] = []
        
        # Experiment parameters
        self.experiment_parameters: Dict[str, Any] = {
            'default_duration': 100,
            'time_step': 1,
            'evaluation_interval': 10,
            'success_threshold': 0.7,
            'stability_threshold': 0.6,
            'innovation_threshold': 0.5,
            'cooperation_threshold': 0.5
        }
        
        # Experiment tracking
        self.experiment_metrics: Dict[str, Any] = {
            'total_experiments': 0,
            'successful_experiments': 0,
            'failed_experiments': 0,
            'experiment_types': {},
            'average_success_rate': 0.0,
            'average_duration': 0.0
        }
        
        # Threading for parallel experiments
        self.experiment_threads: Dict[str, threading.Thread] = {}
        self.experiment_lock = threading.Lock()
        
    def create_governance_experiment(self, 
                                   name: str,
                                   description: str,
                                   governance_type: str,
                                   civilization_ids: List[str],
                                   parameters: Optional[Dict[str, Any]] = None,
                                   duration: int = 100) -> GovernanceExperiment:
        """Create a governance experiment."""
        experiment_id = str(uuid.uuid4())
        
        if parameters is None:
            parameters = {
                'participation_level': random.uniform(0.3, 0.9),
                'decision_speed': random.uniform(0.2, 0.8),
                'consensus_threshold': random.uniform(0.5, 0.9),
                'hierarchy_level': random.uniform(0.1, 0.9)
            }
        
        experiment = GovernanceExperiment(
            id=experiment_id,
            name=name,
            description=description,
            governance_type=governance_type,
            parameters=parameters,
            expected_outcomes=[
                'improved_governance_effectiveness',
                'increased_participation',
                'better_decision_quality',
                'enhanced_stability'
            ],
            success_criteria={
                'governance_effectiveness': 0.7,
                'participation_level': 0.6,
                'decision_quality': 0.7,
                'stability': 0.6
            },
            duration=duration
        )
        
        self.active_experiments[experiment_id] = {
            'experiment': experiment,
            'civilization_ids': civilization_ids,
            'start_time': self.temporal_evolution_engine.current_tick,
            'current_time': 0,
            'status': 'pending'
        }
        
        return experiment
    
    def create_technology_evolution_experiment(self, 
                                             name: str,
                                             description: str,
                                             technology_focus: str,
                                             civilization_ids: List[str],
                                             parameters: Optional[Dict[str, Any]] = None,
                                             duration: int = 100) -> TechnologyEvolutionExperiment:
        """Create a technology evolution experiment."""
        experiment_id = str(uuid.uuid4())
        
        if parameters is None:
            parameters = {
                'innovation_rate': random.uniform(0.1, 0.5),
                'adoption_rate': random.uniform(0.2, 0.8),
                'cross_domain_synthesis': random.uniform(0.1, 0.6),
                'technological_advancement': random.uniform(0.2, 0.8)
            }
        
        experiment = TechnologyEvolutionExperiment(
            id=experiment_id,
            name=name,
            description=description,
            technology_focus=technology_focus,
            parameters=parameters,
            expected_outcomes=[
                'increased_innovation',
                'faster_technology_adoption',
                'cross_domain_breakthroughs',
                'technological_advancement'
            ],
            success_criteria={
                'innovation_rate': 0.6,
                'adoption_rate': 0.7,
                'cross_domain_synthesis': 0.5,
                'technological_advancement': 0.7
            },
            duration=duration
        )
        
        self.active_experiments[experiment_id] = {
            'experiment': experiment,
            'civilization_ids': civilization_ids,
            'start_time': self.temporal_evolution_engine.current_tick,
            'current_time': 0,
            'status': 'pending'
        }
        
        return experiment
    
    def create_cultural_meme_experiment(self, 
                                      name: str,
                                      description: str,
                                      meme_type: str,
                                      civilization_ids: List[str],
                                      parameters: Optional[Dict[str, Any]] = None,
                                      duration: int = 100) -> CulturalMemeExperiment:
        """Create a cultural meme experiment."""
        experiment_id = str(uuid.uuid4())
        
        if parameters is None:
            parameters = {
                'meme_strength': random.uniform(0.3, 0.9),
                'transmission_rate': random.uniform(0.2, 0.8),
                'mutation_rate': random.uniform(0.1, 0.5),
                'cultural_impact': random.uniform(0.2, 0.8)
            }
        
        experiment = CulturalMemeExperiment(
            id=experiment_id,
            name=name,
            description=description,
            meme_type=meme_type,
            parameters=parameters,
            expected_outcomes=[
                'meme_propagation',
                'cultural_change',
                'social_cohesion',
                'behavioral_adaptation'
            ],
            success_criteria={
                'meme_propagation': 0.6,
                'cultural_change': 0.5,
                'social_cohesion': 0.6,
                'behavioral_adaptation': 0.5
            },
            duration=duration
        )
        
        self.active_experiments[experiment_id] = {
            'experiment': experiment,
            'civilization_ids': civilization_ids,
            'start_time': self.temporal_evolution_engine.current_tick,
            'current_time': 0,
            'status': 'pending'
        }
        
        return experiment
    
    def create_knowledge_network_experiment(self, 
                                          name: str,
                                          description: str,
                                          network_topology: str,
                                          civilization_ids: List[str],
                                          parameters: Optional[Dict[str, Any]] = None,
                                          duration: int = 100) -> KnowledgeNetworkExperiment:
        """Create a knowledge network experiment."""
        experiment_id = str(uuid.uuid4())
        
        if parameters is None:
            parameters = {
                'network_density': random.uniform(0.3, 0.9),
                'information_flow': random.uniform(0.2, 0.8),
                'knowledge_diffusion': random.uniform(0.3, 0.8),
                'network_resilience': random.uniform(0.4, 0.9)
            }
        
        experiment = KnowledgeNetworkExperiment(
            id=experiment_id,
            name=name,
            description=description,
            network_topology=network_topology,
            parameters=parameters,
            expected_outcomes=[
                'improved_knowledge_diffusion',
                'enhanced_information_flow',
                'increased_network_resilience',
                'better_collaboration'
            ],
            success_criteria={
                'knowledge_diffusion': 0.7,
                'information_flow': 0.6,
                'network_resilience': 0.7,
                'collaboration': 0.6
            },
            duration=duration
        )
        
        self.active_experiments[experiment_id] = {
            'experiment': experiment,
            'civilization_ids': civilization_ids,
            'start_time': self.temporal_evolution_engine.current_tick,
            'current_time': 0,
            'status': 'pending'
        }
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        try:
            if experiment_id not in self.active_experiments:
                return False
            
            experiment_data = self.active_experiments[experiment_id]
            experiment_data['status'] = 'running'
            experiment_data['start_time'] = self.temporal_evolution_engine.current_tick
            
            # Start experiment thread
            thread = threading.Thread(
                target=self._run_experiment,
                args=(experiment_id,),
                daemon=True
            )
            self.experiment_threads[experiment_id] = thread
            thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting experiment {experiment_id}: {e}")
            return False
    
    def _run_experiment(self, experiment_id: str):
        """Run an experiment."""
        try:
            experiment_data = self.active_experiments[experiment_id]
            experiment = experiment_data['experiment']
            civilization_ids = experiment_data['civilization_ids']
            duration = experiment.duration
            
            while (experiment_data['current_time'] < duration and 
                   experiment_data['status'] == 'running' and 
                   experiment_id in self.active_experiments):
                
                with self.experiment_lock:
                    # Update experiment time
                    experiment_data['current_time'] += self.experiment_parameters['time_step']
                    
                    # Run experiment step
                    self._run_experiment_step(experiment_id, experiment, civilization_ids)
                    
                    # Check for experiment completion
                    if experiment_data['current_time'] >= duration:
                        self._complete_experiment(experiment_id)
                        break
                    
                    # Sleep to control experiment speed
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error running experiment {experiment_id}: {e}")
            experiment_data['status'] = 'failed'
    
    def _run_experiment_step(self, experiment_id: str, experiment: Any, civilization_ids: List[str]):
        """Run a single step of an experiment."""
        # Apply experiment effects to civilizations
        for civ_id in civilization_ids:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                
                # Apply experiment-specific effects
                if isinstance(experiment, GovernanceExperiment):
                    self._apply_governance_effects(civilization, experiment)
                elif isinstance(experiment, TechnologyEvolutionExperiment):
                    self._apply_technology_effects(civilization, experiment)
                elif isinstance(experiment, CulturalMemeExperiment):
                    self._apply_cultural_effects(civilization, experiment)
                elif isinstance(experiment, KnowledgeNetworkExperiment):
                    self._apply_knowledge_network_effects(civilization, experiment)
    
    def _apply_governance_effects(self, civilization: Any, experiment: GovernanceExperiment):
        """Apply governance experiment effects."""
        # Update governance effectiveness
        if 'governance_effectiveness' in civilization.performance_metrics:
            current = civilization.performance_metrics['governance_effectiveness']
            change = experiment.parameters.get('participation_level', 0.5) * 0.1
            civilization.performance_metrics['governance_effectiveness'] = max(0.0, min(1.0, current + change))
        
        # Update participation level
        if 'participation_level' in civilization.value_system.__dict__:
            current = getattr(civilization.value_system, 'participation_level', 0.5)
            change = experiment.parameters.get('participation_level', 0.5) * 0.05
            setattr(civilization.value_system, 'participation_level', max(0.0, min(1.0, current + change)))
    
    def _apply_technology_effects(self, civilization: Any, experiment: TechnologyEvolutionExperiment):
        """Apply technology evolution experiment effects."""
        # Update innovation rate
        if hasattr(civilization.knowledge_base, 'innovation_rate'):
            current = civilization.knowledge_base.innovation_rate
            change = experiment.parameters.get('innovation_rate', 0.3) * 0.1
            civilization.knowledge_base.innovation_rate = max(0.0, min(1.0, current + change))
        
        # Update technological advancement
        if 'technological_advancement' in civilization.performance_metrics:
            current = civilization.performance_metrics['technological_advancement']
            change = experiment.parameters.get('technological_advancement', 0.5) * 0.1
            civilization.performance_metrics['technological_advancement'] = max(0.0, min(1.0, current + change))
    
    def _apply_cultural_effects(self, civilization: Any, experiment: CulturalMemeExperiment):
        """Apply cultural meme experiment effects."""
        # Update cultural vitality
        if 'cultural_vitality' in civilization.performance_metrics:
            current = civilization.performance_metrics['cultural_vitality']
            change = experiment.parameters.get('cultural_impact', 0.5) * 0.1
            civilization.performance_metrics['cultural_vitality'] = max(0.0, min(1.0, current + change))
        
        # Update social cohesion
        if 'social_cohesion' in civilization.performance_metrics:
            current = civilization.performance_metrics['social_cohesion']
            change = experiment.parameters.get('meme_strength', 0.5) * 0.05
            civilization.performance_metrics['social_cohesion'] = max(0.0, min(1.0, current + change))
    
    def _apply_knowledge_network_effects(self, civilization: Any, experiment: KnowledgeNetworkExperiment):
        """Apply knowledge network experiment effects."""
        # Update knowledge diffusion rate
        if hasattr(civilization.knowledge_base, 'knowledge_diffusion_rate'):
            current = civilization.knowledge_base.knowledge_diffusion_rate
            change = experiment.parameters.get('knowledge_diffusion', 0.5) * 0.1
            civilization.knowledge_base.knowledge_diffusion_rate = max(0.0, min(1.0, current + change))
        
        # Update communication efficiency
        if hasattr(civilization.communication_network, 'communication_efficiency'):
            current = civilization.communication_network.communication_efficiency
            change = experiment.parameters.get('information_flow', 0.5) * 0.1
            civilization.communication_network.communication_efficiency = max(0.0, min(1.0, current + change))
    
    def _complete_experiment(self, experiment_id: str):
        """Complete an experiment."""
        experiment_data = self.active_experiments[experiment_id]
        experiment = experiment_data['experiment']
        civilization_ids = experiment_data['civilization_ids']
        
        # Calculate experiment results
        results = self._calculate_experiment_results(experiment_id, experiment, civilization_ids)
        
        # Create experiment result
        experiment_result = ExperimentResult(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            experiment_type=ExperimentType.GOVERNANCE if isinstance(experiment, GovernanceExperiment) else
                          ExperimentType.TECHNOLOGY_EVOLUTION if isinstance(experiment, TechnologyEvolutionExperiment) else
                          ExperimentType.CULTURAL_MEME if isinstance(experiment, CulturalMemeExperiment) else
                          ExperimentType.KNOWLEDGE_NETWORK,
            success=results['success'],
            results=results,
            metrics=results.get('metrics', {}),
            insights=results.get('insights', []),
            recommendations=results.get('recommendations', [])
        )
        
        # Store result
        self.experiment_results[experiment_result.id] = experiment_result
        
        # Update experiment status
        experiment_data['status'] = 'completed'
        experiment.status = 'completed'
        
        # Update experiment metrics
        self._update_experiment_metrics(experiment_result)
        
        # Record experiment completion
        self._record_experiment_completion(experiment_id, experiment_result)
    
    def _calculate_experiment_results(self, experiment_id: str, experiment: Any, civilization_ids: List[str]) -> Dict[str, Any]:
        """Calculate results of an experiment."""
        results = {
            'experiment_id': experiment_id,
            'experiment_type': type(experiment).__name__,
            'duration': experiment.duration,
            'participating_civilizations': civilization_ids,
            'success': False,
            'metrics': {},
            'insights': [],
            'recommendations': []
        }
        
        # Calculate metrics for each participating civilization
        for civ_id in civilization_ids:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                
                # Calculate civilization-specific metrics
                civ_metrics = {
                    'stability': civilization.stability,
                    'innovation_capacity': civilization.innovation_capacity,
                    'adaptability': civilization.adaptability,
                    'resilience': civilization.resilience,
                    'performance_metrics': civilization.performance_metrics.copy(),
                    'strategic_capabilities': civilization.strategic_capabilities.copy()
                }
                
                results['metrics'][civ_id] = civ_metrics
        
        # Determine experiment success
        success_criteria = experiment.success_criteria
        success_count = 0
        total_criteria = len(success_criteria)
        
        for criterion, threshold in success_criteria.items():
            # Check if criterion is met across participating civilizations
            criterion_met = True
            for civ_id in civilization_ids:
                if civ_id in self.civilization_architect.civilizations:
                    civilization = self.civilization_architect.civilizations[civ_id]
                    
                    # Get current value for criterion
                    if criterion in civilization.performance_metrics:
                        current_value = civilization.performance_metrics[criterion]
                    elif hasattr(civilization, criterion):
                        current_value = getattr(civilization, criterion)
                    else:
                        current_value = 0.0
                    
                    if current_value < threshold:
                        criterion_met = False
                        break
            
            if criterion_met:
                success_count += 1
        
        results['success'] = success_count >= total_criteria * 0.7  # 70% of criteria must be met
        
        # Generate insights and recommendations
        results['insights'] = self._generate_insights(experiment, results)
        results['recommendations'] = self._generate_recommendations(experiment, results)
        
        return results
    
    def _generate_insights(self, experiment: Any, results: Dict[str, Any]) -> List[str]:
        """Generate insights from experiment results."""
        insights = []
        
        if results['success']:
            insights.append(f"Experiment {experiment.name} was successful")
            insights.append(f"Key factors: {', '.join(experiment.parameters.keys())}")
        else:
            insights.append(f"Experiment {experiment.name} did not meet success criteria")
            insights.append("Consider adjusting experiment parameters")
        
        # Add experiment-specific insights
        if isinstance(experiment, GovernanceExperiment):
            insights.append("Governance effectiveness improved")
        elif isinstance(experiment, TechnologyEvolutionExperiment):
            insights.append("Technology evolution accelerated")
        elif isinstance(experiment, CulturalMemeExperiment):
            insights.append("Cultural memes propagated successfully")
        elif isinstance(experiment, KnowledgeNetworkExperiment):
            insights.append("Knowledge network efficiency increased")
        
        return insights
    
    def _generate_recommendations(self, experiment: Any, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations from experiment results."""
        recommendations = []
        
        if results['success']:
            recommendations.append("Consider scaling up successful experiment")
            recommendations.append("Apply lessons learned to other civilizations")
        else:
            recommendations.append("Review and adjust experiment parameters")
            recommendations.append("Consider alternative approaches")
        
        # Add experiment-specific recommendations
        if isinstance(experiment, GovernanceExperiment):
            recommendations.append("Implement governance improvements")
        elif isinstance(experiment, TechnologyEvolutionExperiment):
            recommendations.append("Invest in technology development")
        elif isinstance(experiment, CulturalMemeExperiment):
            recommendations.append("Promote cultural exchange")
        elif isinstance(experiment, KnowledgeNetworkExperiment):
            recommendations.append("Enhance knowledge sharing mechanisms")
        
        return recommendations
    
    def _update_experiment_metrics(self, experiment_result: ExperimentResult):
        """Update experiment metrics."""
        self.experiment_metrics['total_experiments'] += 1
        
        if experiment_result.success:
            self.experiment_metrics['successful_experiments'] += 1
        else:
            self.experiment_metrics['failed_experiments'] += 1
        
        # Update experiment type counts
        experiment_type = experiment_result.experiment_type.name
        if experiment_type not in self.experiment_metrics['experiment_types']:
            self.experiment_metrics['experiment_types'][experiment_type] = 0
        self.experiment_metrics['experiment_types'][experiment_type] += 1
        
        # Update average success rate
        self.experiment_metrics['average_success_rate'] = (
            self.experiment_metrics['successful_experiments'] / 
            self.experiment_metrics['total_experiments']
        )
    
    def _record_experiment_completion(self, experiment_id: str, experiment_result: ExperimentResult):
        """Record experiment completion."""
        completion_data = {
            'time': self.temporal_evolution_engine.current_tick,
            'experiment_id': experiment_id,
            'experiment_type': experiment_result.experiment_type.name,
            'success': experiment_result.success,
            'duration': experiment_result.results.get('duration', 0),
            'participating_civilizations': experiment_result.results.get('participating_civilizations', []),
            'metrics': experiment_result.metrics,
            'insights': experiment_result.insights,
            'recommendations': experiment_result.recommendations
        }
        
        self.experiment_history.append(completion_data)
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        try:
            if experiment_id in self.active_experiments:
                experiment_data = self.active_experiments[experiment_id]
                experiment_data['status'] = 'stopped'
                
                if experiment_id in self.experiment_threads:
                    del self.experiment_threads[experiment_id]
                
                return True
            return False
            
        except Exception as e:
            print(f"Error stopping experiment {experiment_id}: {e}")
            return False
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an experiment."""
        if experiment_id in self.active_experiments:
            experiment_data = self.active_experiments[experiment_id]
            experiment = experiment_data['experiment']
            
            return {
                'experiment_id': experiment_id,
                'name': experiment.name,
                'description': experiment.description,
                'status': experiment_data['status'],
                'current_time': experiment_data['current_time'],
                'duration': experiment.duration,
                'participating_civilizations': experiment_data['civilization_ids'],
                'parameters': experiment.parameters,
                'success_criteria': experiment.success_criteria
            }
        return None
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get results of an experiment."""
        for result in self.experiment_results.values():
            if result.experiment_id == experiment_id:
                return result
        return None
    
    def get_experiment_metrics(self) -> Dict[str, Any]:
        """Get experiment metrics."""
        return self.experiment_metrics.copy()
    
    def get_experiment_history(self) -> List[Dict[str, Any]]:
        """Get experiment history."""
        return self.experiment_history.copy()
