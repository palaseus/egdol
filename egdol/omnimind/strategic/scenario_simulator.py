"""
Scenario Simulator for OmniMind
Simulates potential outcomes for strategic goals before commitment.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque
import statistics
import random

# Import ResourceType from the network module
try:
    from ..network.coordination import ResourceType
except ImportError:
    # Fallback if import fails
    class ResourceType(Enum):
        COMPUTATIONAL = auto()
        MEMORY = auto()
        SKILL = auto()
        KNOWLEDGE = auto()
        NETWORK = auto()


class ScenarioType(Enum):
    """Types of scenarios to simulate."""
    GOAL_EXECUTION = auto()
    RESOURCE_ALLOCATION = auto()
    NETWORK_OPTIMIZATION = auto()
    COLLABORATION_PATTERN = auto()
    RISK_MITIGATION = auto()
    PERFORMANCE_IMPROVEMENT = auto()


class SimulationStatus(Enum):
    """Status of simulation runs."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class SimulationResult:
    """Result of a scenario simulation."""
    id: str
    scenario_type: ScenarioType
    goal_id: Optional[str]
    status: SimulationStatus
    created_at: float
    completed_at: Optional[float] = None
    duration: float = 0.0
    success_probability: float = 0.0
    expected_outcome: Dict[str, Any] = None
    resource_impact: Dict[str, float] = None
    risk_factors: List[str] = None
    confidence_score: float = 0.0
    recommendations: List[str] = None
    simulation_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.expected_outcome is None:
            self.expected_outcome = {}
        if self.resource_impact is None:
            self.resource_impact = {}
        if self.risk_factors is None:
            self.risk_factors = []
        if self.recommendations is None:
            self.recommendations = []
        if self.simulation_data is None:
            self.simulation_data = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'id': self.id,
            'scenario_type': self.scenario_type.name,
            'goal_id': self.goal_id,
            'status': self.status.name,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'duration': self.duration,
            'success_probability': self.success_probability,
            'expected_outcome': self.expected_outcome,
            'resource_impact': self.resource_impact,
            'risk_factors': self.risk_factors,
            'confidence_score': self.confidence_score,
            'recommendations': self.recommendations,
            'simulation_data': self.simulation_data
        }


class ScenarioSimulator:
    """Simulates potential outcomes for strategic decisions."""
    
    def __init__(self, network, coordinator, resource_manager):
        self.network = network
        self.coordinator = coordinator
        self.resource_manager = resource_manager
        self.simulations: Dict[str, SimulationResult] = {}
        self.simulation_history: List[Dict[str, Any]] = []
        self.simulation_models: Dict[str, Any] = {}
        
    def simulate_goal_execution(self, goal_id: str, goal_data: Dict[str, Any]) -> SimulationResult:
        """Simulate execution of a strategic goal."""
        simulation_id = str(uuid.uuid4())
        
        simulation = SimulationResult(
            id=simulation_id,
            scenario_type=ScenarioType.GOAL_EXECUTION,
            goal_id=goal_id,
            status=SimulationStatus.RUNNING,
            created_at=time.time()
        )
        
        # Run simulation
        result = self._simulate_goal_scenario(goal_data)
        
        # Update simulation with results
        simulation.status = SimulationStatus.COMPLETED
        simulation.completed_at = time.time()
        simulation.duration = simulation.completed_at - simulation.created_at
        simulation.success_probability = result['success_probability']
        simulation.expected_outcome = result['expected_outcome']
        simulation.resource_impact = result['resource_impact']
        simulation.risk_factors = result['risk_factors']
        simulation.confidence_score = result['confidence_score']
        simulation.recommendations = result['recommendations']
        simulation.simulation_data = result['simulation_data']
        
        # Store simulation
        self.simulations[simulation_id] = simulation
        
        # Log simulation
        self._log_simulation_event('simulation_completed', {
            'simulation_id': simulation_id,
            'scenario_type': 'goal_execution',
            'goal_id': goal_id,
            'success_probability': simulation.success_probability
        })
        
        return simulation
        
    def simulate_resource_allocation(self, allocation_plan: Dict[str, Any]) -> SimulationResult:
        """Simulate resource allocation scenario."""
        simulation_id = str(uuid.uuid4())
        
        simulation = SimulationResult(
            id=simulation_id,
            scenario_type=ScenarioType.RESOURCE_ALLOCATION,
            goal_id=None,
            status=SimulationStatus.RUNNING,
            created_at=time.time()
        )
        
        # Run simulation
        result = self._simulate_resource_scenario(allocation_plan)
        
        # Update simulation with results
        simulation.status = SimulationStatus.COMPLETED
        simulation.completed_at = time.time()
        simulation.duration = simulation.completed_at - simulation.created_at
        simulation.success_probability = result['success_probability']
        simulation.expected_outcome = result['expected_outcome']
        simulation.resource_impact = result['resource_impact']
        simulation.risk_factors = result['risk_factors']
        simulation.confidence_score = result['confidence_score']
        simulation.recommendations = result['recommendations']
        simulation.simulation_data = result['simulation_data']
        
        # Store simulation
        self.simulations[simulation_id] = simulation
        
        # Log simulation
        self._log_simulation_event('simulation_completed', {
            'simulation_id': simulation_id,
            'scenario_type': 'resource_allocation',
            'success_probability': simulation.success_probability
        })
        
        return simulation
        
    def simulate_network_optimization(self, optimization_plan: Dict[str, Any]) -> SimulationResult:
        """Simulate network optimization scenario."""
        simulation_id = str(uuid.uuid4())
        
        simulation = SimulationResult(
            id=simulation_id,
            scenario_type=ScenarioType.NETWORK_OPTIMIZATION,
            goal_id=None,
            status=SimulationStatus.RUNNING,
            created_at=time.time()
        )
        
        # Run simulation
        result = self._simulate_optimization_scenario(optimization_plan)
        
        # Update simulation with results
        simulation.status = SimulationStatus.COMPLETED
        simulation.completed_at = time.time()
        simulation.duration = simulation.completed_at - simulation.created_at
        simulation.success_probability = result['success_probability']
        simulation.expected_outcome = result['expected_outcome']
        simulation.resource_impact = result['resource_impact']
        simulation.risk_factors = result['risk_factors']
        simulation.confidence_score = result['confidence_score']
        simulation.recommendations = result['recommendations']
        simulation.simulation_data = result['simulation_data']
        
        # Store simulation
        self.simulations[simulation_id] = simulation
        
        # Log simulation
        self._log_simulation_event('simulation_completed', {
            'simulation_id': simulation_id,
            'scenario_type': 'network_optimization',
            'success_probability': simulation.success_probability
        })
        
        return simulation
        
    def _simulate_goal_scenario(self, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a goal execution scenario."""
        # Analyze goal requirements
        required_skills = goal_data.get('required_skills', [])
        required_resources = goal_data.get('required_resources', {})
        participating_agents = goal_data.get('participating_agents', [])
        
        # Calculate success probability
        success_probability = self._calculate_goal_success_probability(
            required_skills, required_resources, participating_agents
        )
        
        # Simulate resource impact
        resource_impact = self._simulate_resource_impact(required_resources)
        
        # Identify risk factors
        risk_factors = self._identify_goal_risks(goal_data)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            success_probability, resource_impact, risk_factors
        )
        
        # Generate recommendations
        recommendations = self._generate_goal_recommendations(
            success_probability, resource_impact, risk_factors
        )
        
        # Simulate expected outcome
        expected_outcome = self._simulate_expected_outcome(
            goal_data, success_probability, resource_impact
        )
        
        return {
            'success_probability': success_probability,
            'expected_outcome': expected_outcome,
            'resource_impact': resource_impact,
            'risk_factors': risk_factors,
            'confidence_score': confidence_score,
            'recommendations': recommendations,
            'simulation_data': {
                'required_skills': required_skills,
                'required_resources': required_resources,
                'participating_agents': participating_agents
            }
        }
        
    def _simulate_resource_scenario(self, allocation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a resource allocation scenario."""
        # Analyze resource requirements
        total_resources = allocation_plan.get('total_resources', {})
        allocation_strategy = allocation_plan.get('strategy', 'equal')
        
        # Calculate allocation efficiency
        efficiency = self._calculate_allocation_efficiency(total_resources, allocation_strategy)
        
        # Simulate resource conflicts
        conflicts = self._simulate_resource_conflicts(total_resources)
        
        # Calculate success probability
        success_probability = efficiency * (1 - len(conflicts) * 0.1)
        
        # Generate recommendations
        recommendations = self._generate_resource_recommendations(efficiency, conflicts)
        
        return {
            'success_probability': success_probability,
            'expected_outcome': {'efficiency': efficiency, 'conflicts': len(conflicts)},
            'resource_impact': total_resources,
            'risk_factors': conflicts,
            'confidence_score': success_probability,
            'recommendations': recommendations,
            'simulation_data': {
                'total_resources': total_resources,
                'allocation_strategy': allocation_strategy,
                'efficiency': efficiency
            }
        }
        
    def _simulate_optimization_scenario(self, optimization_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a network optimization scenario."""
        # Analyze optimization targets
        targets = optimization_plan.get('targets', [])
        optimization_strategy = optimization_plan.get('strategy', 'gradual')
        
        # Calculate optimization impact
        impact = self._calculate_optimization_impact(targets, optimization_strategy)
        
        # Simulate network disruption
        disruption = self._simulate_network_disruption(optimization_strategy)
        
        # Calculate success probability
        success_probability = impact * (1 - disruption)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(impact, disruption)
        
        return {
            'success_probability': success_probability,
            'expected_outcome': {'impact': impact, 'disruption': disruption},
            'resource_impact': {},
            'risk_factors': ['network_disruption', 'coordination_complexity'],
            'confidence_score': success_probability,
            'recommendations': recommendations,
            'simulation_data': {
                'targets': targets,
                'optimization_strategy': optimization_strategy,
                'impact': impact
            }
        }
        
    def _calculate_goal_success_probability(self, required_skills: List[str], 
                                          required_resources: Dict[str, float],
                                          participating_agents: List[str]) -> float:
        """Calculate success probability for a goal."""
        # Check skill availability
        skill_coverage = 0.0
        if required_skills:
            available_skills = set()
            for agent in self.network.agents.values():
                available_skills.update(agent.skills)
            skill_coverage = len(set(required_skills) & available_skills) / len(required_skills)
            
        # Check resource availability
        resource_coverage = 0.0
        if required_resources:
            available_resources = self.resource_manager.get_resource_statistics()['resource_usage']
            total_available = sum(resource.get('available', 0) for resource in available_resources.values())
            total_required = sum(required_resources.values())
            resource_coverage = min(total_available / total_required, 1.0) if total_required > 0 else 1.0
            
        # Check agent availability
        agent_coverage = 0.0
        if participating_agents:
            available_agents = [aid for aid in participating_agents if aid in self.network.agents]
            agent_coverage = len(available_agents) / len(participating_agents)
            
        # Calculate overall success probability
        success_probability = (skill_coverage * 0.4 + resource_coverage * 0.3 + agent_coverage * 0.3)
        return min(success_probability, 1.0)
        
    def _simulate_resource_impact(self, required_resources: Dict[str, float]) -> Dict[str, float]:
        """Simulate resource impact of a goal."""
        impact = {}
        
        for resource_type, amount in required_resources.items():
            # Calculate impact based on current resource availability
            current_availability = self.resource_manager.get_resource_availability(
                getattr(ResourceType, resource_type.upper(), ResourceType.COMPUTATIONAL)
            )
            
            if current_availability > 0:
                impact[resource_type] = amount / current_availability
            else:
                impact[resource_type] = 1.0  # Full impact if no availability
                
        return impact
        
    def _identify_goal_risks(self, goal_data: Dict[str, Any]) -> List[str]:
        """Identify risks for a goal."""
        risks = []
        
        # Check for high resource requirements
        required_resources = goal_data.get('required_resources', {})
        if any(amount > 50 for amount in required_resources.values()):
            risks.append('high_resource_requirements')
            
        # Check for complex dependencies
        dependencies = goal_data.get('dependencies', [])
        if len(dependencies) > 3:
            risks.append('complex_dependencies')
            
        # Check for low success probability
        success_probability = goal_data.get('success_probability', 0.5)
        if success_probability < 0.3:
            risks.append('low_success_probability')
            
        return risks
        
    def _calculate_confidence_score(self, success_probability: float, 
                                  resource_impact: Dict[str, float],
                                  risk_factors: List[str]) -> float:
        """Calculate confidence score for a simulation."""
        # Base confidence on success probability
        confidence = success_probability
        
        # Adjust for resource impact
        if resource_impact:
            avg_impact = statistics.mean(resource_impact.values())
            confidence *= (1 - avg_impact * 0.2)
            
        # Adjust for risk factors
        risk_penalty = len(risk_factors) * 0.1
        confidence *= (1 - risk_penalty)
        
        return max(0.0, min(confidence, 1.0))
        
    def _generate_goal_recommendations(self, success_probability: float,
                                     resource_impact: Dict[str, float],
                                     risk_factors: List[str]) -> List[str]:
        """Generate recommendations for a goal."""
        recommendations = []
        
        if success_probability < 0.5:
            recommendations.append("Consider breaking down the goal into smaller, more achievable tasks")
            
        if resource_impact:
            max_impact_resource = max(resource_impact.items(), key=lambda x: x[1])
            if max_impact_resource[1] > 0.8:
                recommendations.append(f"High resource impact on {max_impact_resource[0]}, consider alternative approaches")
                
        if 'complex_dependencies' in risk_factors:
            recommendations.append("Simplify dependencies to reduce complexity")
            
        if 'low_success_probability' in risk_factors:
            recommendations.append("Increase success probability by improving prerequisites")
            
        return recommendations
        
    def _simulate_expected_outcome(self, goal_data: Dict[str, Any], 
                                 success_probability: float,
                                 resource_impact: Dict[str, float]) -> Dict[str, Any]:
        """Simulate expected outcome of a goal."""
        outcome = {
            'success_probability': success_probability,
            'expected_duration': goal_data.get('estimated_duration', 3600),  # 1 hour default
            'resource_utilization': resource_impact,
            'quality_score': success_probability * 0.8 + 0.2,  # Quality based on success probability
            'efficiency_gain': success_probability * 0.3,  # Efficiency gain
            'risk_level': 'high' if success_probability < 0.3 else 'medium' if success_probability < 0.7 else 'low'
        }
        
        return outcome
        
    def _calculate_allocation_efficiency(self, total_resources: Dict[str, float], 
                                       strategy: str) -> float:
        """Calculate efficiency of resource allocation."""
        # Base efficiency on strategy
        if strategy == 'equal':
            return 0.8
        elif strategy == 'priority':
            return 0.9
        elif strategy == 'optimal':
            return 0.95
        else:
            return 0.7
            
    def _simulate_resource_conflicts(self, total_resources: Dict[str, float]) -> List[str]:
        """Simulate resource conflicts."""
        conflicts = []
        
        # Check for resource over-allocation
        for resource_type, amount in total_resources.items():
            available = self.resource_manager.get_resource_availability(
                getattr(ResourceType, resource_type.upper(), ResourceType.COMPUTATIONAL)
            )
            if amount > available:
                conflicts.append(f'{resource_type}_over_allocation')
                
        return conflicts
        
    def _generate_resource_recommendations(self, efficiency: float, conflicts: List[str]) -> List[str]:
        """Generate recommendations for resource allocation."""
        recommendations = []
        
        if efficiency < 0.8:
            recommendations.append("Consider optimizing resource allocation strategy")
            
        if conflicts:
            recommendations.append("Resolve resource conflicts before proceeding")
            
        return recommendations
        
    def _calculate_optimization_impact(self, targets: List[str], strategy: str) -> float:
        """Calculate impact of network optimization."""
        # Base impact on number of targets and strategy
        base_impact = len(targets) * 0.1
        
        if strategy == 'gradual':
            return base_impact * 0.8
        elif strategy == 'aggressive':
            return base_impact * 1.2
        else:
            return base_impact
            
    def _simulate_network_disruption(self, strategy: str) -> float:
        """Simulate network disruption from optimization."""
        if strategy == 'gradual':
            return 0.1
        elif strategy == 'aggressive':
            return 0.3
        else:
            return 0.2
            
    def _generate_optimization_recommendations(self, impact: float, disruption: float) -> List[str]:
        """Generate recommendations for network optimization."""
        recommendations = []
        
        if impact > 0.5:
            recommendations.append("High impact optimization, ensure proper planning")
            
        if disruption > 0.2:
            recommendations.append("Consider gradual implementation to minimize disruption")
            
        return recommendations
        
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        total_simulations = len(self.simulations)
        completed_simulations = sum(1 for sim in self.simulations.values() 
                                 if sim.status == SimulationStatus.COMPLETED)
        
        # Calculate average success probability
        success_probabilities = [sim.success_probability for sim in self.simulations.values()]
        average_success_probability = statistics.mean(success_probabilities) if success_probabilities else 0
        
        # Calculate average confidence score
        confidence_scores = [sim.confidence_score for sim in self.simulations.values()]
        average_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        # Calculate scenario type distribution
        type_distribution = defaultdict(int)
        for sim in self.simulations.values():
            type_distribution[sim.scenario_type.name] += 1
            
        return {
            'total_simulations': total_simulations,
            'completed_simulations': completed_simulations,
            'average_success_probability': average_success_probability,
            'average_confidence': average_confidence,
            'type_distribution': dict(type_distribution)
        }
        
    def _log_simulation_event(self, event_type: str, data: Dict[str, Any]):
        """Log a simulation event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.simulation_history.append(event)
        
    def get_simulation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get simulation history."""
        return list(self.simulation_history[-limit:])
