"""
Strategic Civilizational Orchestrator for Transcendence Layer
Directs multi-civilization simulations to explore strategic domains.
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


class StrategicDomain(Enum):
    """Strategic domains for multi-civilization simulations."""
    RESOURCE_ACQUISITION = auto()
    KNOWLEDGE_DIFFUSION = auto()
    COOPERATION_VS_CONFLICT = auto()
    TECHNOLOGICAL_COMPETITION = auto()
    CULTURAL_INFLUENCE = auto()
    GOVERNANCE_EVOLUTION = auto()
    ENVIRONMENTAL_ADAPTATION = auto()
    STRATEGIC_ALLIANCES = auto()
    ECONOMIC_DOMINANCE = auto()
    INNOVATION_RACE = auto()


class PolicyArchetype(Enum):
    """Policy archetypes for testing evolutionary stability."""
    ISOLATIONIST = auto()
    EXPANSIONIST = auto()
    COOPERATIVE = auto()
    COMPETITIVE = auto()
    ADAPTIVE = auto()
    CONSERVATIVE = auto()
    INNOVATIVE = auto()
    TRADITIONAL = auto()
    AGGRESSIVE = auto()
    DEFENSIVE = auto()


class EvolutionaryStability(Enum):
    """Evolutionary stability levels."""
    HIGHLY_STABLE = auto()
    STABLE = auto()
    MODERATELY_STABLE = auto()
    UNSTABLE = auto()
    HIGHLY_UNSTABLE = auto()


@dataclass
class StrategicIntelligence:
    """Represents strategic intelligence capabilities."""
    id: str
    name: str
    description: str
    intelligence_type: str  # e.g., 'tactical', 'strategic', 'operational'
    effectiveness: float  # 0.0 to 1.0
    adaptability: float  # 0.0 to 1.0
    innovation_capacity: float  # 0.0 to 1.0
    decision_speed: float  # 0.0 to 1.0
    risk_tolerance: float  # 0.0 to 1.0
    collaboration_capacity: float  # 0.0 to 1.0
    competitive_advantage: float  # 0.0 to 1.0
    strategic_depth: float  # 0.0 to 1.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MultiCivilizationSimulation:
    """Represents a multi-civilization simulation."""
    id: str
    name: str
    description: str
    strategic_domain: StrategicDomain
    policy_archetypes: Dict[str, PolicyArchetype] = field(default_factory=dict)  # civilization_id: archetype
    participating_civilizations: List[str] = field(default_factory=list)
    simulation_parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: int = 0
    end_time: int = 0
    current_time: int = 0
    status: str = 'pending'  # pending, running, completed, failed
    results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    evolutionary_stability: EvolutionaryStability = EvolutionaryStability.MODERATELY_STABLE
    strategic_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    policy_effectiveness: Dict[str, float] = field(default_factory=dict)
    competitive_dynamics: Dict[str, Any] = field(default_factory=dict)
    cooperation_patterns: List[Dict[str, Any]] = field(default_factory=list)
    conflict_resolution: List[Dict[str, Any]] = field(default_factory=list)


class StrategicCivilizationalOrchestrator:
    """Directs multi-civilization simulations to explore strategic domains."""
    
    def __init__(self, civilization_architect, temporal_evolution_engine, macro_pattern_detector, network, memory_manager, knowledge_graph):
        self.civilization_architect = civilization_architect
        self.temporal_evolution_engine = temporal_evolution_engine
        self.macro_pattern_detector = macro_pattern_detector
        self.network = network
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph
        
        # Simulation management
        self.active_simulations: Dict[str, MultiCivilizationSimulation] = {}
        self.simulation_history: List[Dict[str, Any]] = []
        self.strategic_intelligence: Dict[str, StrategicIntelligence] = {}
        
        # Strategic parameters
        self.strategic_parameters: Dict[str, Any] = {
            'simulation_duration': 100,
            'time_step': 1,
            'policy_evaluation_interval': 10,
            'stability_threshold': 0.6,
            'cooperation_threshold': 0.5,
            'conflict_threshold': 0.7,
            'innovation_threshold': 0.6,
            'adaptability_threshold': 0.5
        }
        
        # Policy effectiveness tracking
        self.policy_effectiveness: Dict[PolicyArchetype, Dict[str, float]] = {}
        self._initialize_policy_effectiveness()
        
        # Strategic intelligence tracking
        self.strategic_metrics: Dict[str, Any] = {
            'total_simulations': 0,
            'successful_simulations': 0,
            'failed_simulations': 0,
            'average_stability': 0.0,
            'average_cooperation': 0.0,
            'average_innovation': 0.0,
            'policy_archetype_performance': {},
            'strategic_domain_performance': {}
        }
        
        # Threading for parallel simulations
        self.simulation_threads: Dict[str, threading.Thread] = {}
        self.simulation_lock = threading.Lock()
        
    def _initialize_policy_effectiveness(self):
        """Initialize policy effectiveness tracking."""
        for archetype in list(PolicyArchetype):
            self.policy_effectiveness[archetype] = {
                'success_rate': 0.0,
                'stability_score': 0.0,
                'cooperation_score': 0.0,
                'innovation_score': 0.0,
                'adaptability_score': 0.0,
                'competitive_advantage': 0.0,
                'simulation_count': 0
            }
    
    def create_multi_civilization_simulation(self, 
                                           name: str,
                                           description: str,
                                           strategic_domain: StrategicDomain,
                                           civilization_ids: List[str],
                                           policy_archetypes: Optional[Dict[str, PolicyArchetype]] = None,
                                           simulation_parameters: Optional[Dict[str, Any]] = None) -> MultiCivilizationSimulation:
        """Create a new multi-civilization simulation."""
        
        # Generate simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Assign policy archetypes if not provided
        if policy_archetypes is None:
            policy_archetypes = {}
            for civ_id in civilization_ids:
                policy_archetypes[civ_id] = random.choice(list(PolicyArchetype))
        
        # Set default simulation parameters
        if simulation_parameters is None:
            simulation_parameters = {
                'duration': self.strategic_parameters['simulation_duration'],
                'time_step': self.strategic_parameters['time_step'],
                'evaluation_interval': self.strategic_parameters['policy_evaluation_interval'],
                'stability_threshold': self.strategic_parameters['stability_threshold'],
                'cooperation_threshold': self.strategic_parameters['cooperation_threshold'],
                'conflict_threshold': self.strategic_parameters['conflict_threshold']
            }
        
        # Create simulation
        simulation = MultiCivilizationSimulation(
            id=simulation_id,
            name=name,
            description=description,
            strategic_domain=strategic_domain,
            policy_archetypes=policy_archetypes,
            participating_civilizations=civilization_ids,
            simulation_parameters=simulation_parameters,
            start_time=self.temporal_evolution_engine.current_time,
            status='pending'
        )
        
        # Store simulation
        self.active_simulations[simulation_id] = simulation
        
        return simulation
    
    def start_simulation(self, simulation_id: str) -> bool:
        """Start a multi-civilization simulation."""
        try:
            if simulation_id not in self.active_simulations:
                return False
            
            simulation = self.active_simulations[simulation_id]
            simulation.status = 'running'
            simulation.start_time = self.temporal_evolution_engine.current_time
            simulation.current_time = simulation.start_time
            
            # Start simulation thread
            thread = threading.Thread(
                target=self._run_simulation,
                args=(simulation_id,),
                daemon=True
            )
            self.simulation_threads[simulation_id] = thread
            thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting simulation {simulation_id}: {e}")
            return False
    
    def _run_simulation(self, simulation_id: str):
        """Run a multi-civilization simulation."""
        try:
            simulation = self.active_simulations[simulation_id]
            end_time = simulation.start_time + simulation.simulation_parameters['duration']
            
            while (simulation.current_time < end_time and 
                   simulation.status == 'running' and 
                   simulation_id in self.active_simulations):
                
                with self.simulation_lock:
                    # Update simulation time
                    simulation.current_time += simulation.simulation_parameters['time_step']
                    
                    # Evaluate policies
                    if simulation.current_time % simulation.simulation_parameters['evaluation_interval'] == 0:
                        self._evaluate_policies(simulation_id)
                    
                    # Update strategic dynamics
                    self._update_strategic_dynamics(simulation_id)
                    
                    # Update competitive dynamics
                    self._update_competitive_dynamics(simulation_id)
                    
                    # Update cooperation patterns
                    self._update_cooperation_patterns(simulation_id)
                    
                    # Update conflict resolution
                    self._update_conflict_resolution(simulation_id)
                    
                    # Record simulation step
                    self._record_simulation_step(simulation_id)
                    
                    # Check for simulation completion
                    if simulation.current_time >= end_time:
                        self._complete_simulation(simulation_id)
                        break
                    
                    # Sleep to control simulation speed
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error running simulation {simulation_id}: {e}")
            simulation.status = 'failed'
            simulation.results['error'] = str(e)
    
    def _evaluate_policies(self, simulation_id: str):
        """Evaluate policy effectiveness in a simulation."""
        simulation = self.active_simulations[simulation_id]
        
        for civ_id, archetype in simulation.policy_archetypes.items():
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                
                # Evaluate policy effectiveness
                effectiveness = self._calculate_policy_effectiveness(civilization, archetype, simulation)
                
                # Update policy effectiveness tracking
                if civ_id not in simulation.policy_effectiveness:
                    simulation.policy_effectiveness[civ_id] = {}
                simulation.policy_effectiveness[civ_id][archetype.name] = effectiveness
                
                # Update global policy effectiveness
                self._update_global_policy_effectiveness(archetype, effectiveness)
    
    def _calculate_policy_effectiveness(self, civilization: Any, archetype: PolicyArchetype, simulation: MultiCivilizationSimulation) -> float:
        """Calculate policy effectiveness for a civilization."""
        # Base effectiveness from civilization characteristics
        base_effectiveness = (
            civilization.stability * 0.3 +
            civilization.adaptability * 0.3 +
            civilization.innovation_capacity * 0.2 +
            civilization.resilience * 0.2
        )
        
        # Archetype-specific adjustments
        if archetype == PolicyArchetype.ISOLATIONIST:
            # Isolationist policies work better with high stability and low adaptability
            archetype_bonus = civilization.stability * 0.2 - civilization.adaptability * 0.1
        elif archetype == PolicyArchetype.EXPANSIONIST:
            # Expansionist policies work better with high adaptability and innovation
            archetype_bonus = civilization.adaptability * 0.2 + civilization.innovation_capacity * 0.1
        elif archetype == PolicyArchetype.COOPERATIVE:
            # Cooperative policies work better with high cooperation and stability
            archetype_bonus = civilization.value_system.cooperation_level * 0.2 + civilization.stability * 0.1
        elif archetype == PolicyArchetype.COMPETITIVE:
            # Competitive policies work better with high innovation and adaptability
            archetype_bonus = civilization.innovation_capacity * 0.2 + civilization.adaptability * 0.1
        elif archetype == PolicyArchetype.ADAPTIVE:
            # Adaptive policies work better with high adaptability and resilience
            archetype_bonus = civilization.adaptability * 0.3 + civilization.resilience * 0.1
        elif archetype == PolicyArchetype.CONSERVATIVE:
            # Conservative policies work better with high stability and low innovation
            archetype_bonus = civilization.stability * 0.2 - civilization.innovation_capacity * 0.1
        elif archetype == PolicyArchetype.INNOVATIVE:
            # Innovative policies work better with high innovation capacity
            archetype_bonus = civilization.innovation_capacity * 0.3
        elif archetype == PolicyArchetype.TRADITIONAL:
            # Traditional policies work better with high stability and low adaptability
            archetype_bonus = civilization.stability * 0.2 - civilization.adaptability * 0.1
        elif archetype == PolicyArchetype.AGGRESSIVE:
            # Aggressive policies work better with high innovation and low cooperation
            archetype_bonus = civilization.innovation_capacity * 0.2 - civilization.value_system.cooperation_level * 0.1
        elif archetype == PolicyArchetype.DEFENSIVE:
            # Defensive policies work better with high stability and resilience
            archetype_bonus = civilization.stability * 0.2 + civilization.resilience * 0.1
        else:
            archetype_bonus = 0.0
        
        # Calculate final effectiveness
        effectiveness = base_effectiveness + archetype_bonus
        return max(0.0, min(1.0, effectiveness))
    
    def _update_global_policy_effectiveness(self, archetype: PolicyArchetype, effectiveness: float):
        """Update global policy effectiveness tracking."""
        if archetype in self.policy_effectiveness:
            current = self.policy_effectiveness[archetype]
            current['simulation_count'] += 1
            current['success_rate'] = (current['success_rate'] * (current['simulation_count'] - 1) + effectiveness) / current['simulation_count']
            
            # Update other metrics based on effectiveness
            if effectiveness > 0.7:
                current['stability_score'] = (current['stability_score'] * (current['simulation_count'] - 1) + 0.8) / current['simulation_count']
                current['cooperation_score'] = (current['cooperation_score'] * (current['simulation_count'] - 1) + 0.7) / current['simulation_count']
                current['innovation_score'] = (current['innovation_score'] * (current['simulation_count'] - 1) + 0.6) / current['simulation_count']
                current['adaptability_score'] = (current['adaptability_score'] * (current['simulation_count'] - 1) + 0.7) / current['simulation_count']
                current['competitive_advantage'] = (current['competitive_advantage'] * (current['simulation_count'] - 1) + 0.8) / current['simulation_count']
    
    def _update_strategic_dynamics(self, simulation_id: str):
        """Update strategic dynamics in a simulation."""
        simulation = self.active_simulations[simulation_id]
        
        # Update strategic outcomes based on current state
        for civ_id in simulation.participating_civilizations:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                
                # Calculate strategic outcome
                strategic_outcome = {
                    'civilization_id': civ_id,
                    'time': simulation.current_time,
                    'strategic_position': self._calculate_strategic_position(civilization, simulation),
                    'competitive_advantage': self._calculate_competitive_advantage(civilization, simulation, civ_id),
                    'cooperation_level': civilization.value_system.cooperation_level,
                    'innovation_capacity': civilization.innovation_capacity,
                    'stability': civilization.stability,
                    'adaptability': civilization.adaptability
                }
                
                simulation.strategic_outcomes.append(strategic_outcome)
    
    def _calculate_strategic_position(self, civilization: Any, simulation: MultiCivilizationSimulation) -> float:
        """Calculate strategic position of a civilization."""
        # Base strategic position from civilization characteristics
        base_position = (
            civilization.complexity * 0.3 +
            civilization.stability * 0.2 +
            civilization.innovation_capacity * 0.2 +
            civilization.adaptability * 0.2 +
            civilization.resilience * 0.1
        )
        
        # Adjust based on strategic domain
        if simulation.strategic_domain == StrategicDomain.RESOURCE_ACQUISITION:
            domain_bonus = civilization.environment.resource_abundance * 0.2
        elif simulation.strategic_domain == StrategicDomain.KNOWLEDGE_DIFFUSION:
            domain_bonus = civilization.knowledge_base.knowledge_diffusion_rate * 0.2
        elif simulation.strategic_domain == StrategicDomain.COOPERATION_VS_CONFLICT:
            domain_bonus = civilization.value_system.cooperation_level * 0.2
        elif simulation.strategic_domain == StrategicDomain.TECHNOLOGICAL_COMPETITION:
            domain_bonus = civilization.innovation_capacity * 0.2
        elif simulation.strategic_domain == StrategicDomain.CULTURAL_INFLUENCE:
            domain_bonus = civilization.value_system.cultural_norms.__len__() * 0.01
        elif simulation.strategic_domain == StrategicDomain.GOVERNANCE_EVOLUTION:
            domain_bonus = civilization.performance_metrics.get('governance_effectiveness', 0.5) * 0.2
        elif simulation.strategic_domain == StrategicDomain.ENVIRONMENTAL_ADAPTATION:
            domain_bonus = civilization.adaptability * 0.2
        elif simulation.strategic_domain == StrategicDomain.STRATEGIC_ALLIANCES:
            domain_bonus = civilization.value_system.cooperation_level * 0.2
        elif simulation.strategic_domain == StrategicDomain.ECONOMIC_DOMINANCE:
            domain_bonus = civilization.performance_metrics.get('economic_efficiency', 0.5) * 0.2
        elif simulation.strategic_domain == StrategicDomain.INNOVATION_RACE:
            domain_bonus = civilization.innovation_capacity * 0.2
        else:
            domain_bonus = 0.0
        
        return max(0.0, min(1.0, base_position + domain_bonus))
    
    def _calculate_competitive_advantage(self, civilization: Any, simulation: MultiCivilizationSimulation, civ_id: str = None) -> float:
        """Calculate competitive advantage of a civilization."""
        # Base competitive advantage
        base_advantage = (
            civilization.strategic_capabilities.get('military_strength', 0.5) * 0.2 +
            civilization.strategic_capabilities.get('economic_power', 0.5) * 0.2 +
            civilization.strategic_capabilities.get('technological_superiority', 0.5) * 0.2 +
            civilization.strategic_capabilities.get('cultural_influence', 0.5) * 0.2 +
            civilization.strategic_capabilities.get('strategic_intelligence', 0.5) * 0.2
        )
        
        # Adjust based on policy archetype
        if civ_id and civ_id in simulation.policy_archetypes:
            archetype = simulation.policy_archetypes[civ_id]
            if archetype == PolicyArchetype.COMPETITIVE:
                archetype_bonus = 0.1
            elif archetype == PolicyArchetype.AGGRESSIVE:
                archetype_bonus = 0.15
            elif archetype == PolicyArchetype.DEFENSIVE:
                archetype_bonus = -0.05
            else:
                archetype_bonus = 0.0
        else:
            archetype_bonus = 0.0
        
        return max(0.0, min(1.0, base_advantage + archetype_bonus))
    
    def _update_competitive_dynamics(self, simulation_id: str):
        """Update competitive dynamics in a simulation."""
        simulation = self.active_simulations[simulation_id]
        
        # Analyze competitive dynamics between civilizations
        competitive_dynamics = {
            'time': simulation.current_time,
            'civilization_rankings': [],
            'competitive_advantages': {},
            'conflict_levels': {},
            'cooperation_levels': {}
        }
        
        # Calculate rankings based on strategic position
        rankings = []
        for civ_id in simulation.participating_civilizations:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                strategic_position = self._calculate_strategic_position(civilization, simulation)
                competitive_advantage = self._calculate_competitive_advantage(civilization, simulation)
                
                rankings.append({
                    'civilization_id': civ_id,
                    'strategic_position': strategic_position,
                    'competitive_advantage': competitive_advantage,
                    'overall_score': strategic_position * 0.6 + competitive_advantage * 0.4
                })
        
        # Sort by overall score
        rankings.sort(key=lambda x: x['overall_score'], reverse=True)
        competitive_dynamics['civilization_rankings'] = rankings
        
        # Calculate competitive advantages
        for ranking in rankings:
            civ_id = ranking['civilization_id']
            competitive_dynamics['competitive_advantages'][civ_id] = ranking['competitive_advantage']
        
        # Calculate conflict and cooperation levels
        for civ_id in simulation.participating_civilizations:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                competitive_dynamics['conflict_levels'][civ_id] = 1.0 - civilization.value_system.cooperation_level
                competitive_dynamics['cooperation_levels'][civ_id] = civilization.value_system.cooperation_level
        
        simulation.competitive_dynamics = competitive_dynamics
    
    def _update_cooperation_patterns(self, simulation_id: str):
        """Update cooperation patterns in a simulation."""
        simulation = self.active_simulations[simulation_id]
        
        # Analyze cooperation patterns between civilizations
        cooperation_patterns = {
            'time': simulation.current_time,
            'cooperation_networks': [],
            'alliance_formations': [],
            'cooperation_effectiveness': {},
            'cooperation_barriers': []
        }
        
        # Identify cooperation networks
        for i, civ_id1 in enumerate(simulation.participating_civilizations):
            for civ_id2 in simulation.participating_civilizations[i+1:]:
                if (civ_id1 in self.civilization_architect.civilizations and 
                    civ_id2 in self.civilization_architect.civilizations):
                    
                    civ1 = self.civilization_architect.civilizations[civ_id1]
                    civ2 = self.civilization_architect.civilizations[civ_id2]
                    
                    # Calculate cooperation potential
                    cooperation_potential = (
                        civ1.value_system.cooperation_level * 0.5 +
                        civ2.value_system.cooperation_level * 0.5
                    )
                    
                    if cooperation_potential > simulation.simulation_parameters['cooperation_threshold']:
                        cooperation_network = {
                            'civilization_1': civ_id1,
                            'civilization_2': civ_id2,
                            'cooperation_potential': cooperation_potential,
                            'cooperation_effectiveness': cooperation_potential * 0.8
                        }
                        cooperation_patterns['cooperation_networks'].append(cooperation_network)
        
        # Identify alliance formations
        for civ_id in simulation.participating_civilizations:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                
                if civilization.value_system.cooperation_level > 0.7:
                    alliance_formation = {
                        'civilization_id': civ_id,
                        'alliance_potential': civilization.value_system.cooperation_level,
                        'alliance_effectiveness': civilization.value_system.cooperation_level * 0.9
                    }
                    cooperation_patterns['alliance_formations'].append(alliance_formation)
        
        # Calculate cooperation effectiveness
        for civ_id in simulation.participating_civilizations:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                cooperation_effectiveness = civilization.value_system.cooperation_level * civilization.stability
                cooperation_patterns['cooperation_effectiveness'][civ_id] = cooperation_effectiveness
        
        simulation.cooperation_patterns = cooperation_patterns
    
    def _update_conflict_resolution(self, simulation_id: str):
        """Update conflict resolution in a simulation."""
        simulation = self.active_simulations[simulation_id]
        
        # Analyze conflict resolution mechanisms
        conflict_resolution = {
            'time': simulation.current_time,
            'conflict_levels': {},
            'resolution_mechanisms': {},
            'resolution_effectiveness': {},
            'conflict_outcomes': []
        }
        
        # Calculate conflict levels
        for civ_id in simulation.participating_civilizations:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                
                # Conflict level based on cooperation level and stability
                conflict_level = 1.0 - civilization.value_system.cooperation_level
                if civilization.stability < 0.5:
                    conflict_level += 0.2
                
                conflict_resolution['conflict_levels'][civ_id] = max(0.0, min(1.0, conflict_level))
        
        # Identify resolution mechanisms
        for civ_id in simulation.participating_civilizations:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                
                # Resolution mechanisms based on civilization characteristics
                mechanisms = []
                if civilization.value_system.cooperation_level > 0.6:
                    mechanisms.append('diplomatic')
                if civilization.stability > 0.7:
                    mechanisms.append('institutional')
                if civilization.adaptability > 0.6:
                    mechanisms.append('adaptive')
                if civilization.resilience > 0.7:
                    mechanisms.append('resilient')
                
                conflict_resolution['resolution_mechanisms'][civ_id] = mechanisms
        
        # Calculate resolution effectiveness
        for civ_id in simulation.participating_civilizations:
            if civ_id in self.civilization_architect.civilizations:
                civilization = self.civilization_architect.civilizations[civ_id]
                
                # Resolution effectiveness based on cooperation and stability
                resolution_effectiveness = (
                    civilization.value_system.cooperation_level * 0.6 +
                    civilization.stability * 0.4
                )
                
                conflict_resolution['resolution_effectiveness'][civ_id] = resolution_effectiveness
        
        simulation.conflict_resolution = conflict_resolution
    
    def _record_simulation_step(self, simulation_id: str):
        """Record a simulation step."""
        simulation = self.active_simulations[simulation_id]
        
        step_data = {
            'time': simulation.current_time,
            'simulation_id': simulation_id,
            'strategic_domain': simulation.strategic_domain.name,
            'participating_civilizations': simulation.participating_civilizations,
            'policy_archetypes': {civ_id: archetype.name for civ_id, archetype in simulation.policy_archetypes.items()},
            'strategic_outcomes': simulation.strategic_outcomes[-len(simulation.participating_civilizations):] if simulation.strategic_outcomes else [],
            'competitive_dynamics': simulation.competitive_dynamics,
            'cooperation_patterns': simulation.cooperation_patterns,
            'conflict_resolution': simulation.conflict_resolution
        }
        
        self.simulation_history.append(step_data)
    
    def _complete_simulation(self, simulation_id: str):
        """Complete a simulation."""
        simulation = self.active_simulations[simulation_id]
        simulation.status = 'completed'
        simulation.end_time = simulation.current_time
        
        # Calculate final results
        simulation.results = self._calculate_simulation_results(simulation_id)
        
        # Calculate performance metrics
        simulation.performance_metrics = self._calculate_performance_metrics(simulation_id)
        
        # Determine evolutionary stability
        simulation.evolutionary_stability = self._determine_evolutionary_stability(simulation_id)
        
        # Update strategic metrics
        self._update_strategic_metrics(simulation_id)
        
        # Record simulation completion
        self._record_simulation_completion(simulation_id)
    
    def _calculate_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """Calculate final results of a simulation."""
        simulation = self.active_simulations[simulation_id]
        
        results = {
            'simulation_id': simulation_id,
            'duration': simulation.end_time - simulation.start_time,
            'strategic_domain': simulation.strategic_domain.name,
            'participating_civilizations': simulation.participating_civilizations,
            'policy_effectiveness': simulation.policy_effectiveness,
            'strategic_outcomes': simulation.strategic_outcomes,
            'competitive_dynamics': simulation.competitive_dynamics,
            'cooperation_patterns': simulation.cooperation_patterns,
            'conflict_resolution': simulation.conflict_resolution,
            'evolutionary_stability': simulation.evolutionary_stability.name
        }
        
        return results
    
    def _calculate_performance_metrics(self, simulation_id: str) -> Dict[str, float]:
        """Calculate performance metrics for a simulation."""
        simulation = self.active_simulations[simulation_id]
        
        # Calculate average metrics across participating civilizations
        metrics = {
            'average_stability': 0.0,
            'average_cooperation': 0.0,
            'average_innovation': 0.0,
            'average_adaptability': 0.0,
            'average_resilience': 0.0,
            'average_competitive_advantage': 0.0,
            'average_strategic_position': 0.0
        }
        
        if simulation.participating_civilizations:
            for civ_id in simulation.participating_civilizations:
                if civ_id in self.civilization_architect.civilizations:
                    civilization = self.civilization_architect.civilizations[civ_id]
                    
                    metrics['average_stability'] += civilization.stability
                    metrics['average_cooperation'] += civilization.value_system.cooperation_level
                    metrics['average_innovation'] += civilization.innovation_capacity
                    metrics['average_adaptability'] += civilization.adaptability
                    metrics['average_resilience'] += civilization.resilience
                    
                    # Calculate competitive advantage and strategic position
                    competitive_advantage = self._calculate_competitive_advantage(civilization, simulation)
                    strategic_position = self._calculate_strategic_position(civilization, simulation)
                    
                    metrics['average_competitive_advantage'] += competitive_advantage
                    metrics['average_strategic_position'] += strategic_position
            
            # Average the metrics
            num_civilizations = len(simulation.participating_civilizations)
            for metric in metrics:
                metrics[metric] /= num_civilizations
        
        return metrics
    
    def _determine_evolutionary_stability(self, simulation_id: str) -> EvolutionaryStability:
        """Determine evolutionary stability of a simulation."""
        simulation = self.active_simulations[simulation_id]
        
        # Calculate stability score based on performance metrics
        stability_score = (
            simulation.performance_metrics.get('average_stability', 0.5) * 0.3 +
            simulation.performance_metrics.get('average_cooperation', 0.5) * 0.2 +
            simulation.performance_metrics.get('average_adaptability', 0.5) * 0.2 +
            simulation.performance_metrics.get('average_resilience', 0.5) * 0.3
        )
        
        # Determine stability level
        if stability_score >= 0.9:
            return EvolutionaryStability.HIGHLY_STABLE
        elif stability_score >= 0.8:
            return EvolutionaryStability.STABLE
        elif stability_score >= 0.6:
            return EvolutionaryStability.MODERATELY_STABLE
        elif stability_score >= 0.4:
            return EvolutionaryStability.UNSTABLE
        else:
            return EvolutionaryStability.HIGHLY_UNSTABLE
    
    def _update_strategic_metrics(self, simulation_id: str):
        """Update strategic metrics."""
        simulation = self.active_simulations[simulation_id]
        
        self.strategic_metrics['total_simulations'] += 1
        
        if simulation.status == 'completed':
            self.strategic_metrics['successful_simulations'] += 1
        else:
            self.strategic_metrics['failed_simulations'] += 1
        
        # Update average metrics
        if simulation.performance_metrics:
            self.strategic_metrics['average_stability'] = (
                self.strategic_metrics['average_stability'] * (self.strategic_metrics['total_simulations'] - 1) +
                simulation.performance_metrics.get('average_stability', 0.5)
            ) / self.strategic_metrics['total_simulations']
            
            self.strategic_metrics['average_cooperation'] = (
                self.strategic_metrics['average_cooperation'] * (self.strategic_metrics['total_simulations'] - 1) +
                simulation.performance_metrics.get('average_cooperation', 0.5)
            ) / self.strategic_metrics['total_simulations']
            
            self.strategic_metrics['average_innovation'] = (
                self.strategic_metrics['average_innovation'] * (self.strategic_metrics['total_simulations'] - 1) +
                simulation.performance_metrics.get('average_innovation', 0.5)
            ) / self.strategic_metrics['total_simulations']
        
        # Update policy archetype performance
        for civ_id, archetype in simulation.policy_archetypes.items():
            if archetype.name not in self.strategic_metrics['policy_archetype_performance']:
                self.strategic_metrics['policy_archetype_performance'][archetype.name] = {
                    'success_count': 0,
                    'total_count': 0,
                    'average_effectiveness': 0.0
                }
            
            perf = self.strategic_metrics['policy_archetype_performance'][archetype.name]
            perf['total_count'] += 1
            if simulation.status == 'completed':
                perf['success_count'] += 1
            
            # Update average effectiveness
            if civ_id in simulation.policy_effectiveness:
                effectiveness = simulation.policy_effectiveness[civ_id].get(archetype.name, 0.5)
                perf['average_effectiveness'] = (
                    perf['average_effectiveness'] * (perf['total_count'] - 1) + effectiveness
                ) / perf['total_count']
        
        # Update strategic domain performance
        domain_name = simulation.strategic_domain.name
        if domain_name not in self.strategic_metrics['strategic_domain_performance']:
            self.strategic_metrics['strategic_domain_performance'][domain_name] = {
                'success_count': 0,
                'total_count': 0,
                'average_performance': 0.0
            }
        
        domain_perf = self.strategic_metrics['strategic_domain_performance'][domain_name]
        domain_perf['total_count'] += 1
        if simulation.status == 'completed':
            domain_perf['success_count'] += 1
        
        # Update average performance
        if simulation.performance_metrics:
            avg_performance = statistics.mean(simulation.performance_metrics.values())
            domain_perf['average_performance'] = (
                domain_perf['average_performance'] * (domain_perf['total_count'] - 1) + avg_performance
            ) / domain_perf['total_count']
    
    def _record_simulation_completion(self, simulation_id: str):
        """Record simulation completion."""
        simulation = self.active_simulations[simulation_id]
        
        completion_data = {
            'time': simulation.end_time,
            'simulation_id': simulation_id,
            'status': simulation.status,
            'duration': simulation.end_time - simulation.start_time,
            'strategic_domain': simulation.strategic_domain.name,
            'participating_civilizations': simulation.participating_civilizations,
            'evolutionary_stability': simulation.evolutionary_stability.name,
            'performance_metrics': simulation.performance_metrics,
            'results': simulation.results
        }
        
        self.simulation_history.append(completion_data)
    
    def stop_simulation(self, simulation_id: str) -> bool:
        """Stop a simulation."""
        try:
            if simulation_id in self.active_simulations:
                simulation = self.active_simulations[simulation_id]
                simulation.status = 'failed'
                simulation.end_time = simulation.current_time
                
                if simulation_id in self.simulation_threads:
                    del self.simulation_threads[simulation_id]
                
                return True
            return False
            
        except Exception as e:
            print(f"Error stopping simulation {simulation_id}: {e}")
            return False
    
    def get_simulation_status(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a simulation."""
        if simulation_id in self.active_simulations:
            simulation = self.active_simulations[simulation_id]
            return {
                'simulation_id': simulation_id,
                'status': simulation.status,
                'current_time': simulation.current_time,
                'start_time': simulation.start_time,
                'end_time': simulation.end_time,
                'strategic_domain': simulation.strategic_domain.name,
                'participating_civilizations': simulation.participating_civilizations,
                'policy_archetypes': {civ_id: archetype.name for civ_id, archetype in simulation.policy_archetypes.items()},
                'evolutionary_stability': simulation.evolutionary_stability.name,
                'performance_metrics': simulation.performance_metrics
            }
        return None
    
    def get_simulation_results(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get results of a simulation."""
        if simulation_id in self.active_simulations:
            simulation = self.active_simulations[simulation_id]
            return simulation.results
        return None
    
    def get_strategic_metrics(self) -> Dict[str, Any]:
        """Get strategic metrics."""
        return self.strategic_metrics.copy()
    
    def get_simulation_history(self) -> List[Dict[str, Any]]:
        """Get simulation history."""
        return self.simulation_history.copy()
    
    def get_policy_effectiveness(self) -> Dict[str, Any]:
        """Get policy effectiveness data."""
        return {archetype.name: data for archetype, data in self.policy_effectiveness.items()}
