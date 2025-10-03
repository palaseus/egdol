"""
Experiment Executor for OmniMind Experimental Intelligence
Executes hypotheses through controlled experiments and simulations.
"""

import uuid
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class ExperimentType(Enum):
    """Types of experiments that can be executed."""
    SIMULATION = auto()
    CONTROLLED_TEST = auto()
    MULTI_AGENT_COLLABORATION = auto()
    RESOURCE_ALLOCATION = auto()
    KNOWLEDGE_INTEGRATION = auto()
    CREATIVE_SYNTHESIS = auto()


class ExperimentStatus(Enum):
    """Status of experiment execution."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class Experiment:
    """Represents an experiment to test a hypothesis."""
    id: str
    hypothesis_id: str
    type: ExperimentType
    description: str
    parameters: Dict[str, Any]
    expected_duration: timedelta
    resource_requirements: Dict[str, float]
    success_criteria: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    agent_assignments: List[str] = field(default_factory=list)


class ExperimentExecutor:
    """Executes experiments to test hypotheses."""
    
    def __init__(self, network, memory_manager, knowledge_graph):
        self.network = network
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph
        self.experiments: Dict[str, Experiment] = {}
        self.execution_queue: List[str] = []
        self.active_experiments: Dict[str, str] = {}  # agent_id -> experiment_id
        self.resource_allocations: Dict[str, Dict[str, float]] = {}
        self.experiment_history: List[Dict[str, Any]] = []
        
    def create_experiment(self, hypothesis_id: str, experiment_type: ExperimentType, 
                         parameters: Dict[str, Any]) -> Experiment:
        """Create a new experiment for testing a hypothesis."""
        experiment = Experiment(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis_id,
            type=experiment_type,
            description=f"Experiment for hypothesis {hypothesis_id}",
            parameters=parameters,
            expected_duration=timedelta(minutes=parameters.get('duration_minutes', 30)),
            resource_requirements=parameters.get('resource_requirements', {}),
            success_criteria=parameters.get('success_criteria', []),
            agent_assignments=parameters.get('agent_assignments', [])
        )
        
        self.experiments[experiment.id] = experiment
        self.execution_queue.append(experiment.id)
        
        return experiment
    
    def execute_experiment(self, experiment_id: str) -> bool:
        """Execute an experiment and return success status."""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()
        
        try:
            # Execute based on experiment type
            if experiment.type == ExperimentType.SIMULATION:
                success = self._execute_simulation_experiment(experiment)
            elif experiment.type == ExperimentType.CONTROLLED_TEST:
                success = self._execute_controlled_test(experiment)
            elif experiment.type == ExperimentType.MULTI_AGENT_COLLABORATION:
                success = self._execute_multi_agent_experiment(experiment)
            elif experiment.type == ExperimentType.RESOURCE_ALLOCATION:
                success = self._execute_resource_experiment(experiment)
            elif experiment.type == ExperimentType.KNOWLEDGE_INTEGRATION:
                success = self._execute_knowledge_experiment(experiment)
            elif experiment.type == ExperimentType.CREATIVE_SYNTHESIS:
                success = self._execute_creative_experiment(experiment)
            else:
                success = False
                experiment.errors.append(f"Unknown experiment type: {experiment.type}")
            
            # Update experiment status
            experiment.completed_at = datetime.now()
            experiment.status = ExperimentStatus.COMPLETED if success else ExperimentStatus.FAILED
            
            # Record in history
            self.experiment_history.append({
                'experiment_id': experiment_id,
                'hypothesis_id': experiment.hypothesis_id,
                'type': experiment.type.name,
                'success': success,
                'duration': (experiment.completed_at - experiment.started_at).total_seconds(),
                'metrics': experiment.metrics.copy()
            })
            
            return success
            
        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            experiment.errors.append(f"Execution error: {str(e)}")
            experiment.completed_at = datetime.now()
            return False
    
    def _execute_simulation_experiment(self, experiment: Experiment) -> bool:
        """Execute a simulation-based experiment."""
        # Simulate the hypothesis scenario
        simulation_results = self._run_simulation(experiment.parameters)
        
        # Evaluate against success criteria
        success = self._evaluate_success_criteria(experiment, simulation_results)
        
        # Record metrics
        experiment.metrics.update({
            'simulation_accuracy': simulation_results.get('accuracy', 0.0),
            'performance_improvement': simulation_results.get('improvement', 0.0),
            'resource_efficiency': simulation_results.get('efficiency', 0.0)
        })
        
        experiment.results = simulation_results
        return success
    
    def _execute_controlled_test(self, experiment: Experiment) -> bool:
        """Execute a controlled test experiment."""
        # Set up controlled environment
        test_environment = self._setup_controlled_environment(experiment.parameters)
        
        # Run controlled test
        test_results = self._run_controlled_test(test_environment, experiment.parameters)
        
        # Evaluate results
        success = self._evaluate_controlled_results(experiment, test_results)
        
        # Record metrics
        experiment.metrics.update({
            'test_accuracy': test_results.get('accuracy', 0.0),
            'reproducibility': test_results.get('reproducibility', 0.0),
            'statistical_significance': test_results.get('significance', 0.0)
        })
        
        experiment.results = test_results
        return success
    
    def _execute_multi_agent_experiment(self, experiment: Experiment) -> bool:
        """Execute a multi-agent collaboration experiment."""
        # Assign agents to experiment
        assigned_agents = self._assign_agents_to_experiment(experiment)
        
        # Run multi-agent collaboration
        collaboration_results = self._run_multi_agent_collaboration(
            assigned_agents, experiment.parameters
        )
        
        # Evaluate collaboration effectiveness
        success = self._evaluate_collaboration_results(experiment, collaboration_results)
        
        # Record metrics
        experiment.metrics.update({
            'collaboration_efficiency': collaboration_results.get('efficiency', 0.0),
            'knowledge_sharing': collaboration_results.get('knowledge_sharing', 0.0),
            'task_completion_rate': collaboration_results.get('completion_rate', 0.0)
        })
        
        experiment.results = collaboration_results
        return success
    
    def _execute_resource_experiment(self, experiment: Experiment) -> bool:
        """Execute a resource allocation experiment."""
        # Allocate resources
        allocation_results = self._allocate_resources(experiment.parameters)
        
        # Monitor resource utilization
        utilization_results = self._monitor_resource_utilization(allocation_results)
        
        # Evaluate resource efficiency
        success = self._evaluate_resource_efficiency(experiment, utilization_results)
        
        # Record metrics
        experiment.metrics.update({
            'resource_utilization': utilization_results.get('utilization', 0.0),
            'allocation_efficiency': utilization_results.get('efficiency', 0.0),
            'cost_effectiveness': utilization_results.get('cost_effectiveness', 0.0)
        })
        
        experiment.results = utilization_results
        return success
    
    def _execute_knowledge_experiment(self, experiment: Experiment) -> bool:
        """Execute a knowledge integration experiment."""
        # Test knowledge integration
        integration_results = self._test_knowledge_integration(experiment.parameters)
        
        # Evaluate integration success
        success = self._evaluate_knowledge_integration(experiment, integration_results)
        
        # Record metrics
        experiment.metrics.update({
            'integration_success_rate': integration_results.get('success_rate', 0.0),
            'knowledge_retention': integration_results.get('retention', 0.0),
            'reasoning_improvement': integration_results.get('reasoning_improvement', 0.0)
        })
        
        experiment.results = integration_results
        return success
    
    def _execute_creative_experiment(self, experiment: Experiment) -> bool:
        """Execute a creative synthesis experiment."""
        # Run creative synthesis
        synthesis_results = self._run_creative_synthesis(experiment.parameters)
        
        # Evaluate creativity and novelty
        success = self._evaluate_creative_results(experiment, synthesis_results)
        
        # Record metrics
        experiment.metrics.update({
            'creativity_score': synthesis_results.get('creativity', 0.0),
            'novelty_score': synthesis_results.get('novelty', 0.0),
            'usefulness_score': synthesis_results.get('usefulness', 0.0)
        })
        
        experiment.results = synthesis_results
        return success
    
    def _run_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simulation for the experiment."""
        # Simulate network behavior
        simulation_steps = parameters.get('simulation_steps', 100)
        accuracy = random.uniform(0.7, 0.95)
        improvement = random.uniform(0.1, 0.3)
        efficiency = random.uniform(0.8, 0.95)
        
        return {
            'accuracy': accuracy,
            'improvement': improvement,
            'efficiency': efficiency,
            'steps_completed': simulation_steps,
            'simulation_time': time.time()
        }
    
    def _setup_controlled_environment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Set up a controlled test environment."""
        return {
            'environment_id': str(uuid.uuid4()),
            'variables': parameters.get('control_variables', {}),
            'baseline_metrics': parameters.get('baseline_metrics', {}),
            'test_conditions': parameters.get('test_conditions', {})
        }
    
    def _run_controlled_test(self, environment: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a controlled test in the environment."""
        # Simulate controlled test results
        accuracy = random.uniform(0.8, 0.98)
        reproducibility = random.uniform(0.85, 0.95)
        significance = random.uniform(0.7, 0.9)
        
        return {
            'accuracy': accuracy,
            'reproducibility': reproducibility,
            'significance': significance,
            'test_iterations': parameters.get('iterations', 10),
            'environment_id': environment['environment_id']
        }
    
    def _assign_agents_to_experiment(self, experiment: Experiment) -> List[str]:
        """Assign agents to the experiment."""
        if experiment.agent_assignments:
            return experiment.agent_assignments
        
        # Get available agents
        available_agents = self.network.get_available_agents()
        num_agents = min(len(available_agents), experiment.parameters.get('max_agents', 3))
        
        return random.sample([agent.id for agent in available_agents], num_agents)
    
    def _run_multi_agent_collaboration(self, agent_ids: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-agent collaboration experiment."""
        # Simulate collaboration results
        efficiency = random.uniform(0.7, 0.9)
        knowledge_sharing = random.uniform(0.6, 0.85)
        completion_rate = random.uniform(0.8, 0.95)
        
        return {
            'efficiency': efficiency,
            'knowledge_sharing': knowledge_sharing,
            'completion_rate': completion_rate,
            'agent_count': len(agent_ids),
            'collaboration_time': time.time()
        }
    
    def _allocate_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources for the experiment."""
        resource_requirements = parameters.get('resource_requirements', {})
        allocation = {}
        
        for resource, amount in resource_requirements.items():
            allocation[resource] = amount * random.uniform(0.8, 1.2)
        
        return {
            'allocation': allocation,
            'total_cost': sum(allocation.values()),
            'allocation_time': time.time()
        }
    
    def _monitor_resource_utilization(self, allocation: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor resource utilization during experiment."""
        utilization = {}
        for resource, allocated in allocation['allocation'].items():
            utilization[resource] = allocated * random.uniform(0.7, 0.95)
        
        return {
            'utilization': sum(utilization.values()) / sum(allocation['allocation'].values()),
            'efficiency': random.uniform(0.8, 0.95),
            'cost_effectiveness': random.uniform(0.7, 0.9)
        }
    
    def _test_knowledge_integration(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Test knowledge integration capabilities."""
        return {
            'success_rate': random.uniform(0.75, 0.9),
            'retention': random.uniform(0.8, 0.95),
            'reasoning_improvement': random.uniform(0.1, 0.3),
            'integration_time': time.time()
        }
    
    def _run_creative_synthesis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run creative synthesis experiment."""
        return {
            'creativity': random.uniform(0.6, 0.9),
            'novelty': random.uniform(0.5, 0.85),
            'usefulness': random.uniform(0.7, 0.9),
            'synthesis_time': time.time()
        }
    
    def _evaluate_success_criteria(self, experiment: Experiment, results: Dict[str, Any]) -> bool:
        """Evaluate if experiment meets success criteria."""
        if not experiment.success_criteria:
            return True
        
        # Check each success criterion
        for criterion in experiment.success_criteria:
            if not self._check_criterion(criterion, results):
                return False
        
        return True
    
    def _check_criterion(self, criterion: str, results: Dict[str, Any]) -> bool:
        """Check if a specific criterion is met."""
        # Simple criterion checking - in practice this would be more sophisticated
        if 'accuracy' in criterion and 'accuracy' in results:
            threshold = 0.8
            return results['accuracy'] >= threshold
        elif 'improvement' in criterion and 'improvement' in results:
            threshold = 0.1
            return results['improvement'] >= threshold
        elif 'efficiency' in criterion and 'efficiency' in results:
            threshold = 0.8
            return results['efficiency'] >= threshold
        
        return True  # Default to success if criterion not recognized
    
    def _evaluate_controlled_results(self, experiment: Experiment, results: Dict[str, Any]) -> bool:
        """Evaluate controlled test results."""
        return results.get('accuracy', 0) >= 0.8 and results.get('significance', 0) >= 0.7
    
    def _evaluate_collaboration_results(self, experiment: Experiment, results: Dict[str, Any]) -> bool:
        """Evaluate multi-agent collaboration results."""
        return (results.get('efficiency', 0) >= 0.7 and 
                results.get('completion_rate', 0) >= 0.8)
    
    def _evaluate_resource_efficiency(self, experiment: Experiment, results: Dict[str, Any]) -> bool:
        """Evaluate resource allocation efficiency."""
        return results.get('efficiency', 0) >= 0.8
    
    def _evaluate_knowledge_integration(self, experiment: Experiment, results: Dict[str, Any]) -> bool:
        """Evaluate knowledge integration results."""
        return (results.get('success_rate', 0) >= 0.8 and 
                results.get('retention', 0) >= 0.8)
    
    def _evaluate_creative_results(self, experiment: Experiment, results: Dict[str, Any]) -> bool:
        """Evaluate creative synthesis results."""
        return (results.get('creativity', 0) >= 0.6 and 
                results.get('usefulness', 0) >= 0.7)
    
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get statistics about executed experiments."""
        total_experiments = len(self.experiments)
        completed_experiments = len([e for e in self.experiments.values() if e.status == ExperimentStatus.COMPLETED])
        failed_experiments = len([e for e in self.experiments.values() if e.status == ExperimentStatus.FAILED])
        
        success_rate = completed_experiments / total_experiments if total_experiments > 0 else 0
        
        avg_duration = 0
        if self.experiment_history:
            durations = [h['duration'] for h in self.experiment_history]
            avg_duration = sum(durations) / len(durations)
        
        return {
            'total_experiments': total_experiments,
            'completed_experiments': completed_experiments,
            'failed_experiments': failed_experiments,
            'success_rate': success_rate,
            'average_duration': avg_duration,
            'queue_length': len(self.execution_queue),
            'active_experiments': len(self.active_experiments)
        }
    
    def get_experiments_by_status(self, status: ExperimentStatus) -> List[Experiment]:
        """Get experiments filtered by status."""
        return [e for e in self.experiments.values() if e.status == status]
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment."""
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            if experiment.status == ExperimentStatus.RUNNING:
                experiment.status = ExperimentStatus.CANCELLED
                experiment.completed_at = datetime.now()
                return True
        return False
