"""
Autonomous Experimenter for Next-Generation OmniMind
Plans, executes, and monitors experiments offline with full resource and constraint management.
"""

import uuid
import random
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import os


class ExperimentType(Enum):
    """Types of experiments that can be executed."""
    SIMULATION = auto()
    CONTROLLED_TEST = auto()
    MULTI_AGENT_COLLABORATION = auto()
    RESOURCE_ALLOCATION = auto()
    KNOWLEDGE_INTEGRATION = auto()
    CREATIVE_SYNTHESIS = auto()
    PERFORMANCE_BENCHMARK = auto()
    ADAPTIVE_LEARNING = auto()
    EMERGENT_BEHAVIOR = auto()
    CROSS_DOMAIN_FUSION = auto()


class ExperimentStatus(Enum):
    """Status of experiment execution."""
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class ResourceType(Enum):
    """Types of resources that can be managed."""
    COMPUTATIONAL = auto()
    MEMORY = auto()
    NETWORK = auto()
    STORAGE = auto()
    ENERGY = auto()
    TIME = auto()


@dataclass
class ResourceConstraint:
    """Resource constraint for experiment execution."""
    resource_type: ResourceType
    max_usage: float  # 0.0-1.0
    priority: int  # 1-10, higher is more important
    current_usage: float = 0.0
    allocated: float = 0.0


@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment execution."""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    success_indicators: Dict[str, bool] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0
    data_points_collected: int = 0
    iterations_completed: int = 0


@dataclass
class Experiment:
    """Autonomous experiment with full resource management."""
    id: str
    name: str
    description: str
    experiment_type: ExperimentType
    status: ExperimentStatus
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Experiment configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    objectives: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    expected_duration: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    max_duration: timedelta = field(default_factory=lambda: timedelta(hours=2))
    
    # Resource management
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    resource_constraints: List[ResourceConstraint] = field(default_factory=list)
    priority: int = 5  # 1-10, higher is more important
    
    # Execution
    execution_function: Optional[Callable] = None
    monitoring_functions: List[Callable] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: ExperimentMetrics = field(default_factory=lambda: ExperimentMetrics(datetime.now()))
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Dependencies and relationships
    dependencies: List[str] = field(default_factory=list)
    blocking_experiments: List[str] = field(default_factory=list)
    parallel_experiments: List[str] = field(default_factory=list)
    
    # Safety and rollback
    safety_checks: List[str] = field(default_factory=list)
    rollback_plan: Optional[Dict[str, Any]] = None
    deterministic: bool = True


class ResourceManager:
    """Manages system resources for experiment execution."""
    
    def __init__(self):
        self.total_resources = {
            ResourceType.COMPUTATIONAL: 1.0,
            ResourceType.MEMORY: 1.0,
            ResourceType.NETWORK: 1.0,
            ResourceType.STORAGE: 1.0,
            ResourceType.ENERGY: 1.0,
            ResourceType.TIME: 1.0
        }
        
        self.allocated_resources = {resource: 0.0 for resource in list(ResourceType)}
        self.resource_queues = {resource: queue.Queue() for resource in list(ResourceType)}
        self.resource_locks = {resource: threading.Lock() for resource in list(ResourceType)}
        
        # Resource monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.resource_history: List[Dict[str, Any]] = []
        
        # Start resource monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start monitoring system resources."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring system resources."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitor_resources(self):
        """Monitor system resources continuously."""
        while self.monitoring_active:
            try:
                # Get current system resource usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Update resource usage
                current_usage = {
                    'timestamp': datetime.now(),
                    'computational': cpu_percent / 100.0,
                    'memory': memory.percent / 100.0,
                    'storage': disk.percent / 100.0,
                    'network': 0.0,  # Would need network monitoring
                    'energy': 0.0,  # Would need energy monitoring
                    'time': 1.0  # Time is always available
                }
                
                # Store in history
                self.resource_history.append(current_usage)
                
                # Keep only recent history
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-500:]
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                time.sleep(10)
    
    def allocate_resources(self, experiment: Experiment) -> bool:
        """Allocate resources for an experiment."""
        # If no resource requirements, allocation is always successful
        if not experiment.resource_requirements:
            return True
            
        with threading.Lock():
            # Check if resources are available
            for resource_type, required in experiment.resource_requirements.items():
                available = self.total_resources[resource_type] - self.allocated_resources[resource_type]
                if available < required:
                    return False
            
            # Allocate resources
            for resource_type, required in experiment.resource_requirements.items():
                self.allocated_resources[resource_type] += required
            
            return True
    
    def deallocate_resources(self, experiment: Experiment):
        """Deallocate resources from an experiment."""
        with threading.Lock():
            for resource_type, allocated in experiment.resource_requirements.items():
                self.allocated_resources[resource_type] -= allocated
    
    def get_available_resources(self) -> Dict[ResourceType, float]:
        """Get currently available resources."""
        available = {}
        for resource_type in list(ResourceType):
            available[resource_type] = self.total_resources[resource_type] - self.allocated_resources[resource_type]
        return available
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        if not self.resource_history:
            return {'message': 'No resource history available'}
        
        latest = self.resource_history[-1]
        return {
            'current_usage': latest,
            'allocated_resources': dict(self.allocated_resources),
            'available_resources': self.get_available_resources(),
            'history_length': len(self.resource_history)
        }


class ConstraintManager:
    """Manages constraints for experiment execution."""
    
    def __init__(self):
        self.constraints: List[ResourceConstraint] = []
        self.constraint_violations: List[Dict[str, Any]] = []
        self.constraint_monitoring = True
    
    def add_constraint(self, constraint: ResourceConstraint):
        """Add a resource constraint."""
        self.constraints.append(constraint)
    
    def check_constraints(self, experiment: Experiment) -> Tuple[bool, List[str]]:
        """Check if experiment meets all constraints."""
        violations = []
        
        for constraint in self.constraints:
            if constraint.resource_type in experiment.resource_requirements:
                required = experiment.resource_requirements[constraint.resource_type]
                if required > constraint.max_usage:
                    violations.append(f"Resource {constraint.resource_type.name} exceeds constraint: {required:.2f} > {constraint.max_usage:.2f}")
        
        # Check system-level constraints
        if experiment.expected_duration > timedelta(hours=24):
            violations.append("Experiment duration exceeds 24 hours")
        
        if len(experiment.objectives) == 0:
            violations.append("Experiment has no objectives")
        
        if not experiment.deterministic and experiment.experiment_type in [ExperimentType.SIMULATION, ExperimentType.CONTROLLED_TEST]:
            violations.append("Non-deterministic experiments should not be simulations or controlled tests")
        
        return len(violations) == 0, violations
    
    def monitor_constraints(self, experiment: Experiment) -> bool:
        """Monitor constraints during experiment execution."""
        if not self.constraint_monitoring:
            return True
        
        for constraint in self.constraints:
            if constraint.resource_type in experiment.resource_requirements:
                current_usage = constraint.current_usage
                if current_usage > constraint.max_usage:
                    violation = {
                        'timestamp': datetime.now(),
                        'experiment_id': experiment.id,
                        'constraint_type': constraint.resource_type.name,
                        'current_usage': current_usage,
                        'max_usage': constraint.max_usage,
                        'violation_amount': current_usage - constraint.max_usage
                    }
                    self.constraint_violations.append(violation)
                    return False
        
        return True


class AutonomousExperimenter:
    """Autonomous experimenter with full resource and constraint management."""
    
    def __init__(self, network, memory_manager, knowledge_graph, experimental_system):
        self.network = network
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph
        self.experimental_system = experimental_system
        
        # Experiment management
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: List[str] = []
        self.completed_experiments: List[str] = []
        self.failed_experiments: List[str] = []
        
        # Resource and constraint management
        self.resource_manager = ResourceManager()
        self.constraint_manager = ConstraintManager()
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.experiment_futures: Dict[str, Future] = {}
        
        # Monitoring and logging
        self.monitoring_active = True
        self.monitoring_thread = None
        self.experiment_log: List[Dict[str, Any]] = []
        
        # Initialize constraint system
        self._initialize_constraints()
        
        # Start monitoring
        self.start_monitoring()
    
    def _initialize_constraints(self):
        """Initialize default constraints for experiment execution."""
        # Default resource constraints
        constraints = [
            ResourceConstraint(ResourceType.COMPUTATIONAL, 0.8, 10),
            ResourceConstraint(ResourceType.MEMORY, 0.7, 9),
            ResourceConstraint(ResourceType.NETWORK, 0.6, 8),
            ResourceConstraint(ResourceType.STORAGE, 0.5, 7),
            ResourceConstraint(ResourceType.ENERGY, 0.9, 10),
            ResourceConstraint(ResourceType.TIME, 1.0, 10)
        ]
        
        for constraint in constraints:
            self.constraint_manager.add_constraint(constraint)
    
    def start_monitoring(self):
        """Start monitoring experiment execution."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_experiments, daemon=True)
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring experiment execution."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitor_experiments(self):
        """Monitor active experiments continuously."""
        while self.monitoring_active:
            try:
                for experiment_id in self.active_experiments[:]:  # Copy to avoid modification during iteration
                    if experiment_id in self.experiments:
                        experiment = self.experiments[experiment_id]
                        
                        # Check if experiment should be terminated
                        if self._should_terminate_experiment(experiment):
                            self._terminate_experiment(experiment_id, "Termination conditions met")
                        
                        # Check constraints
                        if not self.constraint_manager.monitor_constraints(experiment):
                            self._terminate_experiment(experiment_id, "Constraint violation")
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logging.error(f"Experiment monitoring error: {e}")
                time.sleep(5)
    
    def _should_terminate_experiment(self, experiment: Experiment) -> bool:
        """Check if experiment should be terminated."""
        # Check timeout
        if experiment.started_at:
            elapsed = datetime.now() - experiment.started_at
            if elapsed > experiment.max_duration:
                return True
        
        # Check for critical errors
        if len(experiment.errors) > 10:
            return True
        
        # Check success criteria
        if experiment.status == ExperimentStatus.RUNNING:
            success_count = sum(1 for criteria in experiment.success_criteria 
                              if criteria in experiment.results and experiment.results[criteria])
            if success_count == len(experiment.success_criteria):
                return True
        
        return False
    
    def create_experiment(self, 
                        name: str,
                        description: str,
                        experiment_type: ExperimentType,
                        parameters: Dict[str, Any],
                        objectives: List[str],
                        success_criteria: List[str],
                        resource_requirements: Dict[ResourceType, float],
                        execution_function: Optional[Callable] = None,
                        expected_duration: timedelta = timedelta(minutes=30),
                        priority: int = 5) -> Experiment:
        """Create a new autonomous experiment."""
        
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            experiment_type=experiment_type,
            status=ExperimentStatus.PENDING,
            parameters=parameters,
            objectives=objectives,
            success_criteria=success_criteria,
            resource_requirements=resource_requirements,
            execution_function=execution_function,
            expected_duration=expected_duration,
            priority=priority
        )
        
        # Validate experiment
        is_valid, violations = self.constraint_manager.check_constraints(experiment)
        if not is_valid:
            experiment.errors.extend(violations)
            experiment.status = ExperimentStatus.FAILED
        
        # Store experiment
        self.experiments[experiment.id] = experiment
        
        # Log experiment creation
        self.experiment_log.append({
            'timestamp': datetime.now(),
            'action': 'experiment_created',
            'experiment_id': experiment.id,
            'name': name,
            'type': experiment_type.name,
            'violations': violations
        })
        
        return experiment
    
    def execute_experiment(self, experiment_id: str) -> bool:
        """Execute an autonomous experiment."""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        
        # Check if experiment can be executed
        if experiment.status != ExperimentStatus.PENDING:
            return False
        
        # Allocate resources
        if not self.resource_manager.allocate_resources(experiment):
            experiment.errors.append("Insufficient resources available")
            experiment.status = ExperimentStatus.FAILED
            return False
        
        # Start experiment
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()
        self.active_experiments.append(experiment_id)
        
        # Execute experiment in thread pool
        future = self.executor.submit(self._execute_experiment_worker, experiment)
        self.experiment_futures[experiment_id] = future
        
        # Log experiment start
        self.experiment_log.append({
            'timestamp': datetime.now(),
            'action': 'experiment_started',
            'experiment_id': experiment_id,
            'name': experiment.name
        })
        
        return True
    
    def _execute_experiment_worker(self, experiment: Experiment):
        """Worker function for experiment execution."""
        try:
            # Execute the experiment
            if experiment.execution_function:
                results = experiment.execution_function(experiment.parameters)
                experiment.results.update(results)
            else:
                # Default experiment execution
                results = self._default_experiment_execution(experiment)
                experiment.results.update(results)
            
            # Update metrics
            experiment.metrics.end_time = datetime.now()
            experiment.metrics.duration = experiment.metrics.end_time - experiment.metrics.start_time
            
            # Check success criteria
            success_count = 0
            for criteria in experiment.success_criteria:
                if criteria in experiment.results and experiment.results[criteria]:
                    success_count += 1
                    experiment.metrics.success_indicators[criteria] = True
                else:
                    experiment.metrics.success_indicators[criteria] = False
            
            # Determine final status
            if success_count == len(experiment.success_criteria):
                experiment.status = ExperimentStatus.COMPLETED
                self.completed_experiments.append(experiment.id)
            else:
                experiment.status = ExperimentStatus.FAILED
                self.failed_experiments.append(experiment.id)
                experiment.errors.append(f"Only {success_count}/{len(experiment.success_criteria)} success criteria met")
            
            # Deallocate resources
            self.resource_manager.deallocate_resources(experiment)
            
            # Remove from active experiments
            if experiment.id in self.active_experiments:
                self.active_experiments.remove(experiment.id)
            
            # Log experiment completion
            self.experiment_log.append({
                'timestamp': datetime.now(),
                'action': 'experiment_completed',
                'experiment_id': experiment.id,
                'status': experiment.status.name,
                'success_count': success_count,
                'total_criteria': len(experiment.success_criteria)
            })
            
        except Exception as e:
            # Handle execution errors
            experiment.status = ExperimentStatus.FAILED
            experiment.errors.append(f"Execution error: {str(e)}")
            self.failed_experiments.append(experiment.id)
            
            # Deallocate resources
            self.resource_manager.deallocate_resources(experiment)
            
            # Remove from active experiments
            if experiment.id in self.active_experiments:
                self.active_experiments.remove(experiment.id)
            
            # Log experiment failure
            self.experiment_log.append({
                'timestamp': datetime.now(),
                'action': 'experiment_failed',
                'experiment_id': experiment.id,
                'error': str(e)
            })
    
    def _default_experiment_execution(self, experiment: Experiment) -> Dict[str, Any]:
        """Default experiment execution when no custom function is provided."""
        results = {}
        
        # Simulate experiment execution based on type
        if experiment.experiment_type == ExperimentType.SIMULATION:
            results = self._execute_simulation_experiment(experiment)
        elif experiment.experiment_type == ExperimentType.CONTROLLED_TEST:
            results = self._execute_controlled_test_experiment(experiment)
        elif experiment.experiment_type == ExperimentType.MULTI_AGENT_COLLABORATION:
            results = self._execute_multi_agent_experiment(experiment)
        elif experiment.experiment_type == ExperimentType.PERFORMANCE_BENCHMARK:
            results = self._execute_benchmark_experiment(experiment)
        else:
            results = self._execute_generic_experiment(experiment)
        
        return results
    
    def _execute_simulation_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Execute a simulation experiment."""
        # Simulate simulation execution
        iterations = experiment.parameters.get('iterations', 100)
        success_rate = random.uniform(0.6, 0.95)
        
        results = {
            'simulation_completed': True,
            'iterations_run': iterations,
            'success_rate': success_rate,
            'performance_improvement': random.uniform(0.1, 0.5),
            'resource_efficiency': random.uniform(0.7, 0.9)
        }
        
        # Update metrics
        experiment.metrics.iterations_completed = iterations
        experiment.metrics.data_points_collected = iterations * 10
        
        return results
    
    def _execute_controlled_test_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Execute a controlled test experiment."""
        # Simulate controlled test execution
        test_cases = experiment.parameters.get('test_cases', 50)
        pass_rate = random.uniform(0.8, 0.98)
        
        results = {
            'test_completed': True,
            'test_cases_run': test_cases,
            'pass_rate': pass_rate,
            'accuracy_improvement': random.uniform(0.05, 0.3),
            'reliability_score': random.uniform(0.85, 0.98)
        }
        
        # Update metrics
        experiment.metrics.data_points_collected = test_cases
        
        return results
    
    def _execute_multi_agent_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Execute a multi-agent collaboration experiment."""
        # Simulate multi-agent experiment
        agents = experiment.parameters.get('agents', 3)
        collaboration_efficiency = random.uniform(0.7, 0.95)
        
        results = {
            'collaboration_completed': True,
            'agents_involved': agents,
            'collaboration_efficiency': collaboration_efficiency,
            'task_completion_rate': random.uniform(0.8, 0.95),
            'communication_effectiveness': random.uniform(0.75, 0.9)
        }
        
        return results
    
    def _execute_benchmark_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Execute a performance benchmark experiment."""
        # Simulate benchmark execution
        benchmark_duration = experiment.parameters.get('duration_seconds', 60)
        performance_score = random.uniform(0.6, 0.95)
        
        results = {
            'benchmark_completed': True,
            'duration_seconds': benchmark_duration,
            'performance_score': performance_score,
            'throughput_improvement': random.uniform(0.1, 0.4),
            'latency_reduction': random.uniform(0.05, 0.3)
        }
        
        return results
    
    def _execute_generic_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Execute a generic experiment."""
        # Simulate generic experiment execution
        duration = experiment.expected_duration.total_seconds()
        success_probability = random.uniform(0.7, 0.95)
        
        results = {
            'experiment_completed': True,
            'duration_seconds': duration,
            'success_probability': success_probability,
            'data_quality': random.uniform(0.8, 0.95),
            'insights_generated': random.randint(3, 10)
        }
        
        return results
    
    def _terminate_experiment(self, experiment_id: str, reason: str):
        """Terminate an experiment."""
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            experiment.status = ExperimentStatus.CANCELLED
            experiment.errors.append(f"Terminated: {reason}")
            
            # Deallocate resources
            self.resource_manager.deallocate_resources(experiment)
            
            # Remove from active experiments
            if experiment_id in self.active_experiments:
                self.active_experiments.remove(experiment_id)
            
            # Log termination
            self.experiment_log.append({
                'timestamp': datetime.now(),
                'action': 'experiment_terminated',
                'experiment_id': experiment_id,
                'reason': reason
            })
    
    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an experiment."""
        if experiment_id in self.experiments and experiment_id in self.active_experiments:
            experiment = self.experiments[experiment_id]
            experiment.status = ExperimentStatus.PAUSED
            
            # Log pause
            self.experiment_log.append({
                'timestamp': datetime.now(),
                'action': 'experiment_paused',
                'experiment_id': experiment_id
            })
            
            return True
        return False
    
    def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused experiment."""
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            if experiment.status == ExperimentStatus.PAUSED:
                experiment.status = ExperimentStatus.RUNNING
                
                # Log resume
                self.experiment_log.append({
                    'timestamp': datetime.now(),
                    'action': 'experiment_resumed',
                    'experiment_id': experiment_id
                })
                
                return True
        return False
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel an experiment."""
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            experiment.status = ExperimentStatus.CANCELLED
            
            # Deallocate resources
            self.resource_manager.deallocate_resources(experiment)
            
            # Remove from active experiments
            if experiment_id in self.active_experiments:
                self.active_experiments.remove(experiment_id)
            
            # Log cancellation
            self.experiment_log.append({
                'timestamp': datetime.now(),
                'action': 'experiment_cancelled',
                'experiment_id': experiment_id
            })
            
            return True
        return False
    
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get statistics about experiment execution."""
        total_experiments = len(self.experiments)
        active_experiments = len(self.active_experiments)
        completed_experiments = len(self.completed_experiments)
        failed_experiments = len(self.failed_experiments)
        
        # Status distribution
        status_counts = {}
        for experiment in self.experiments.values():
            status = experiment.status.name
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Type distribution
        type_counts = {}
        for experiment in self.experiments.values():
            exp_type = experiment.experiment_type.name
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
        
        # Resource usage
        resource_usage = self.resource_manager.get_resource_usage()
        
        return {
            'total_experiments': total_experiments,
            'active_experiments': active_experiments,
            'completed_experiments': completed_experiments,
            'failed_experiments': failed_experiments,
            'status_distribution': status_counts,
            'type_distribution': type_counts,
            'resource_usage': resource_usage,
            'constraint_violations': len(self.constraint_manager.constraint_violations),
            'experiment_log_entries': len(self.experiment_log)
        }
    
    def get_active_experiments(self) -> List[Experiment]:
        """Get all currently active experiments."""
        return [self.experiments[exp_id] for exp_id in self.active_experiments 
                if exp_id in self.experiments]
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed results for a specific experiment."""
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiments[experiment_id]
        
        return {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status.name,
            'results': experiment.results,
            'metrics': {
                'start_time': experiment.metrics.start_time.isoformat() if experiment.metrics.start_time else None,
                'end_time': experiment.metrics.end_time.isoformat() if experiment.metrics.end_time else None,
                'duration': str(experiment.metrics.duration) if experiment.metrics.duration else None,
                'resource_usage': experiment.metrics.resource_usage,
                'performance_metrics': experiment.metrics.performance_metrics,
                'success_indicators': experiment.metrics.success_indicators,
                'error_count': experiment.metrics.error_count,
                'warning_count': experiment.metrics.warning_count,
                'data_points_collected': experiment.metrics.data_points_collected,
                'iterations_completed': experiment.metrics.iterations_completed
            },
            'errors': experiment.errors,
            'warnings': experiment.warnings
        }
    
    def cleanup_completed_experiments(self):
        """Clean up completed and failed experiments."""
        # This would implement cleanup logic for completed experiments
        # For now, just log the action
        self.experiment_log.append({
            'timestamp': datetime.now(),
            'action': 'cleanup_completed',
            'completed_count': len(self.completed_experiments),
            'failed_count': len(self.failed_experiments)
        })
    
    def shutdown(self):
        """Shutdown the experimenter and clean up resources."""
        # Stop monitoring
        self.stop_monitoring()
        
        # Cancel all active experiments
        for experiment_id in self.active_experiments[:]:
            self.cancel_experiment(experiment_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Stop resource monitoring
        self.resource_manager.stop_monitoring()
