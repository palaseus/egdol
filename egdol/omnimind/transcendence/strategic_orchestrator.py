"""
Strategic Civilizational Orchestrator for OmniMind Civilization Intelligence Layer
Manages multi-civilization interactions, strategic domains, and policy archetypes.
"""

import uuid
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import time

from .core_structures import Civilization, AgentCluster, CivilizationIntelligenceCore


class StrategicDomain(Enum):
    """Strategic domains for multi-civilization interactions."""
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
    """Policy archetypes for civilization strategies."""
    ISOLATIONIST = auto()
    EXPANSIONIST = auto()
    COMPETITIVE = auto()
    COOPERATIVE = auto()
    ADAPTIVE = auto()
    INNOVATIVE = auto()
    CONSERVATIVE = auto()
    AGGRESSIVE = auto()
    DEFENSIVE = auto()
    DIPLOMATIC = auto()


class InteractionType(Enum):
    """Types of civilization interactions."""
    TRADE = auto()
    ALLIANCE = auto()
    CONFLICT = auto()
    KNOWLEDGE_SHARING = auto()
    RESOURCE_SHARING = auto()
    CULTURAL_EXCHANGE = auto()
    TECHNOLOGICAL_COOPERATION = auto()
    DIPLOMATIC_NEGOTIATION = auto()
    COMPETITIVE_RIVALRY = auto()
    MUTUAL_AID = auto()


@dataclass
class StrategicInteraction:
    """Represents a strategic interaction between civilizations."""
    id: str
    interaction_type: InteractionType
    participants: List[str]  # Civilization IDs
    domain: StrategicDomain
    start_time: datetime = field(default_factory=datetime.now)
    duration: int = 0  # in ticks
    intensity: float = 0.0  # 0.0 to 1.0
    success_probability: float = 0.0  # 0.0 to 1.0
    
    # Interaction parameters
    resource_flow: Dict[str, float] = field(default_factory=dict)
    knowledge_transfer: List[str] = field(default_factory=list)
    cultural_exchange: List[str] = field(default_factory=list)
    technological_cooperation: List[str] = field(default_factory=list)
    
    # Outcomes
    outcomes: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    completion_time: Optional[datetime] = None


@dataclass
class StrategicSimulation:
    """Represents a strategic simulation scenario."""
    id: str
    name: str
    description: str
    domain: StrategicDomain
    participating_civilizations: List[str]
    policy_assignments: Dict[str, PolicyArchetype]  # civ_id -> policy
    
    # Simulation parameters
    duration: int = 1000  # ticks
    time_step: int = 1
    evaluation_interval: int = 10
    
    # Simulation state
    current_time: int = 0
    status: str = "pending"  # pending, running, completed, failed
    
    # Results
    interactions: List[StrategicInteraction] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    civilization_outcomes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Meta-intelligence integration
    meta_intelligence_proposals: List[Dict[str, Any]] = field(default_factory=list)
    architectural_insights: List[Dict[str, Any]] = field(default_factory=list)


class StrategicCivilizationalOrchestrator:
    """Orchestrates multi-civilization strategic interactions."""
    
    def __init__(self, core: CivilizationIntelligenceCore):
        """Initialize the strategic orchestrator."""
        self.core = core
        self.active_simulations: Dict[str, StrategicSimulation] = {}
        self.interaction_history: List[StrategicInteraction] = []
        self.strategic_metrics: Dict[str, Any] = {}
        
        # Policy effectiveness tracking
        self.policy_effectiveness: Dict[PolicyArchetype, Dict[str, float]] = {}
        self.domain_performance: Dict[StrategicDomain, Dict[str, float]] = {}
        
        # Threading for parallel simulations
        self.simulation_threads: Dict[str, threading.Thread] = {}
        self.simulation_lock = threading.Lock()
        
        # Meta-intelligence integration
        self.meta_intelligence_hooks: List[callable] = []
        
        # Strategic parameters
        self.strategic_parameters = {
            'interaction_probability': 0.1,
            'cooperation_threshold': 0.6,
            'conflict_threshold': 0.4,
            'alliance_threshold': 0.7,
            'trade_efficiency': 0.8,
            'knowledge_diffusion_rate': 0.3,
            'cultural_influence_rate': 0.2
        }
    
    def create_strategic_simulation(self, name: str, description: str, 
                                   domain: StrategicDomain,
                                   civilization_ids: List[str],
                                   policy_assignments: Optional[Dict[str, PolicyArchetype]] = None,
                                   duration: int = 1000,
                                   time_step: int = 1,
                                   evaluation_interval: int = 10) -> StrategicSimulation:
        """Create a new strategic simulation."""
        simulation_id = str(uuid.uuid4())
        
        # Assign random policies if not specified
        if policy_assignments is None:
            policy_assignments = {}
            for civ_id in civilization_ids:
                policy_assignments[civ_id] = random.choice(list(PolicyArchetype))
        
        simulation = StrategicSimulation(
            id=simulation_id,
            name=name,
            description=description,
            domain=domain,
            participating_civilizations=civilization_ids,
            policy_assignments=policy_assignments,
            duration=duration,
            time_step=time_step,
            evaluation_interval=evaluation_interval
        )
        
        self.active_simulations[simulation_id] = simulation
        return simulation
    
    def start_simulation(self, simulation_id: str) -> bool:
        """Start a strategic simulation."""
        if simulation_id not in self.active_simulations:
            return False
        
        simulation = self.active_simulations[simulation_id]
        simulation.status = "running"
        
        # Start simulation thread
        thread = threading.Thread(
            target=self._run_simulation,
            args=(simulation_id,),
            daemon=True
        )
        self.simulation_threads[simulation_id] = thread
        thread.start()
        
        return True
    
    def _run_simulation(self, simulation_id: str):
        """Run a strategic simulation."""
        simulation = self.active_simulations[simulation_id]
        
        try:
            while (simulation.current_time < simulation.duration and 
                   simulation.status == "running"):
                
                with self.simulation_lock:
                    # Update simulation time
                    simulation.current_time += simulation.time_step
                    
                    # Generate interactions
                    if simulation.current_time % simulation.evaluation_interval == 0:
                        self._generate_interactions(simulation)
                    
                    # Process interactions
                    self._process_interactions(simulation)
                    
                    # Update civilization states
                    self._update_civilization_states(simulation)
                    
                    # Evaluate performance
                    if simulation.current_time % simulation.evaluation_interval == 0:
                        self._evaluate_performance(simulation)
                    
                    # Generate meta-intelligence insights
                    if simulation.current_time % (simulation.evaluation_interval * 2) == 0:
                        self._generate_meta_intelligence_insights(simulation)
                    
                    # Sleep to control simulation speed
                    time.sleep(0.01)
            
            # Complete simulation
            simulation.status = "completed"
            self._finalize_simulation(simulation)
            
        except Exception as e:
            print(f"Simulation error: {e}")
            simulation.status = "failed"
    
    def _generate_interactions(self, simulation: StrategicSimulation):
        """Generate strategic interactions between civilizations."""
        if len(simulation.participating_civilizations) < 2:
            return
        
        # Generate interactions based on domain and policies
        for i, civ_id_1 in enumerate(simulation.participating_civilizations):
            for civ_id_2 in simulation.participating_civilizations[i+1:]:
                if random.random() < self.strategic_parameters['interaction_probability']:
                    interaction = self._create_interaction(
                        civ_id_1, civ_id_2, simulation
                    )
                    if interaction:
                        simulation.interactions.append(interaction)
    
    def _create_interaction(self, civ_id_1: str, civ_id_2: str, 
                          simulation: StrategicSimulation) -> Optional[StrategicInteraction]:
        """Create a strategic interaction between two civilizations."""
        civ_1 = self.core.get_civilization(civ_id_1)
        civ_2 = self.core.get_civilization(civ_id_2)
        
        if not civ_1 or not civ_2:
            return None
        
        # Determine interaction type based on policies and domain
        interaction_type = self._determine_interaction_type(
            civ_1, civ_2, simulation
        )
        
        if not interaction_type:
            return None
        
        # Calculate interaction parameters
        intensity = self._calculate_interaction_intensity(civ_1, civ_2, simulation)
        success_probability = self._calculate_success_probability(
            civ_1, civ_2, interaction_type, simulation
        )
        
        # Create interaction
        interaction = StrategicInteraction(
            id=str(uuid.uuid4()),
            interaction_type=interaction_type,
            participants=[civ_id_1, civ_id_2],
            domain=simulation.domain,
            duration=random.randint(10, 50),
            intensity=intensity,
            success_probability=success_probability
        )
        
        # Set interaction parameters based on type
        self._configure_interaction_parameters(interaction, civ_1, civ_2, simulation)
        
        return interaction
    
    def _determine_interaction_type(self, civ_1: Civilization, civ_2: Civilization,
                                  simulation: StrategicSimulation) -> Optional[InteractionType]:
        """Determine the type of interaction between two civilizations."""
        policy_1 = simulation.policy_assignments.get(civ_1.id)
        policy_2 = simulation.policy_assignments.get(civ_2.id)
        
        if not policy_1 or not policy_2:
            return None
        
        # Determine interaction type based on policies and domain
        if (policy_1 == PolicyArchetype.COOPERATIVE and 
            policy_2 == PolicyArchetype.COOPERATIVE):
            return random.choice([InteractionType.ALLIANCE, InteractionType.KNOWLEDGE_SHARING,
                                InteractionType.CULTURAL_EXCHANGE, InteractionType.MUTUAL_AID])
        
        elif (policy_1 == PolicyArchetype.COMPETITIVE and 
              policy_2 == PolicyArchetype.COMPETITIVE):
            return random.choice([InteractionType.CONFLICT, InteractionType.COMPETITIVE_RIVALRY,
                                InteractionType.TRADE])
        
        elif (policy_1 == PolicyArchetype.AGGRESSIVE or 
              policy_2 == PolicyArchetype.AGGRESSIVE):
            return random.choice([InteractionType.CONFLICT, InteractionType.COMPETITIVE_RIVALRY])
        
        elif (policy_1 == PolicyArchetype.DIPLOMATIC and 
              policy_2 == PolicyArchetype.DIPLOMATIC):
            return random.choice([InteractionType.DIPLOMATIC_NEGOTIATION, InteractionType.TRADE,
                                InteractionType.ALLIANCE])
        
        else:
            # Mixed policies - determine based on domain
            if simulation.domain == StrategicDomain.RESOURCE_ACQUISITION:
                return random.choice([InteractionType.TRADE, InteractionType.CONFLICT])
            elif simulation.domain == StrategicDomain.KNOWLEDGE_DIFFUSION:
                return random.choice([InteractionType.KNOWLEDGE_SHARING, InteractionType.TECHNOLOGICAL_COOPERATION])
            elif simulation.domain == StrategicDomain.CULTURAL_INFLUENCE:
                return random.choice([InteractionType.CULTURAL_EXCHANGE, InteractionType.DIPLOMATIC_NEGOTIATION])
            else:
                return random.choice(list(InteractionType))
    
    def _calculate_interaction_intensity(self, civ_1: Civilization, civ_2: Civilization,
                                       simulation: StrategicSimulation) -> float:
        """Calculate the intensity of an interaction."""
        # Base intensity on civilization characteristics
        base_intensity = (civ_1.complexity + civ_2.complexity) / 2.0
        
        # Adjust based on policies
        policy_1 = simulation.policy_assignments.get(civ_1.id)
        policy_2 = simulation.policy_assignments.get(civ_2.id)
        
        if policy_1 == PolicyArchetype.AGGRESSIVE or policy_2 == PolicyArchetype.AGGRESSIVE:
            base_intensity *= 1.5
        elif policy_1 == PolicyArchetype.COOPERATIVE and policy_2 == PolicyArchetype.COOPERATIVE:
            base_intensity *= 0.8
        
        return min(1.0, base_intensity)
    
    def _calculate_success_probability(self, civ_1: Civilization, civ_2: Civilization,
                                     interaction_type: InteractionType,
                                     simulation: StrategicSimulation) -> float:
        """Calculate the success probability of an interaction."""
        # Base success probability
        base_success = 0.5
        
        # Adjust based on civilization compatibility
        compatibility = self._calculate_civilization_compatibility(civ_1, civ_2)
        base_success *= compatibility
        
        # Adjust based on interaction type
        if interaction_type in [InteractionType.ALLIANCE, InteractionType.MUTUAL_AID]:
            base_success *= 1.2
        elif interaction_type == InteractionType.CONFLICT:
            base_success *= 0.8
        
        # Adjust based on domain
        if simulation.domain == StrategicDomain.COOPERATION_VS_CONFLICT:
            if interaction_type in [InteractionType.ALLIANCE, InteractionType.MUTUAL_AID]:
                base_success *= 1.3
            elif interaction_type == InteractionType.CONFLICT:
                base_success *= 0.7
        
        return min(1.0, max(0.0, base_success))
    
    def _calculate_civilization_compatibility(self, civ_1: Civilization, civ_2: Civilization) -> float:
        """Calculate compatibility between two civilizations."""
        # Base compatibility on shared characteristics
        stability_diff = abs(civ_1.stability - civ_2.stability)
        complexity_diff = abs(civ_1.complexity - civ_2.complexity)
        cooperation_diff = abs(civ_1.cooperation_level - civ_2.cooperation_level)
        
        # Calculate compatibility score
        compatibility = 1.0 - (stability_diff + complexity_diff + cooperation_diff) / 3.0
        
        # Adjust based on governance models
        if civ_1.governance_model == civ_2.governance_model:
            compatibility *= 1.2
        
        return min(1.0, max(0.0, compatibility))
    
    def _configure_interaction_parameters(self, interaction: StrategicInteraction,
                                        civ_1: Civilization, civ_2: Civilization,
                                        simulation: StrategicSimulation):
        """Configure interaction parameters based on type and civilizations."""
        if interaction.interaction_type == InteractionType.TRADE:
            # Configure trade parameters
            interaction.resource_flow = {
                'energy': random.uniform(10, 50),
                'materials': random.uniform(5, 25),
                'information': random.uniform(1, 10)
            }
        
        elif interaction.interaction_type == InteractionType.KNOWLEDGE_SHARING:
            # Configure knowledge transfer
            knowledge_items = list(civ_1.knowledge_base.keys())[:3]
            interaction.knowledge_transfer = knowledge_items
        
        elif interaction.interaction_type == InteractionType.CULTURAL_EXCHANGE:
            # Configure cultural exchange
            cultural_traits = list(civ_1.cultural_traits)[:2]
            interaction.cultural_exchange = cultural_traits
        
        elif interaction.interaction_type == InteractionType.TECHNOLOGICAL_COOPERATION:
            # Configure technological cooperation
            tech_items = list(civ_1.knowledge_base.keys())[:2]
            interaction.technological_cooperation = tech_items
    
    def _process_interactions(self, simulation: StrategicSimulation):
        """Process active interactions."""
        for interaction in simulation.interactions:
            if not interaction.completion_time:
                # Check if interaction should complete
                if simulation.current_time >= interaction.duration:
                    self._complete_interaction(interaction, simulation)
    
    def _complete_interaction(self, interaction: StrategicInteraction, simulation: StrategicSimulation):
        """Complete an interaction and apply outcomes."""
        # Determine success
        interaction.success = random.random() < interaction.success_probability
        interaction.completion_time = datetime.now()
        
        # Apply outcomes
        if interaction.success:
            self._apply_successful_interaction_outcomes(interaction, simulation)
        else:
            self._apply_failed_interaction_outcomes(interaction, simulation)
        
        # Record interaction
        self.interaction_history.append(interaction)
    
    def _apply_successful_interaction_outcomes(self, interaction: StrategicInteraction,
                                            simulation: StrategicSimulation):
        """Apply outcomes of a successful interaction."""
        civ_1 = self.core.get_civilization(interaction.participants[0])
        civ_2 = self.core.get_civilization(interaction.participants[1])
        
        if not civ_1 or not civ_2:
            return
        
        # Apply resource flow
        for resource, amount in interaction.resource_flow.items():
            if resource in civ_1.resource_pools:
                civ_1.resource_pools[resource] += amount
            if resource in civ_2.resource_pools:
                civ_2.resource_pools[resource] += amount
        
        # Apply knowledge transfer
        for knowledge in interaction.knowledge_transfer:
            if knowledge not in civ_2.knowledge_base:
                civ_2.knowledge_base[knowledge] = 1.0
        
        # Apply cultural exchange
        for trait in interaction.cultural_exchange:
            if trait not in civ_2.cultural_traits:
                civ_2.cultural_traits.add(trait)
        
        # Apply technological cooperation
        for tech in interaction.technological_cooperation:
            if tech not in civ_2.knowledge_base:
                civ_2.knowledge_base[tech] = 1.0
        
        # Update cooperation levels
        civ_1.cooperation_level = min(1.0, civ_1.cooperation_level + 0.05)
        civ_2.cooperation_level = min(1.0, civ_2.cooperation_level + 0.05)
    
    def _apply_failed_interaction_outcomes(self, interaction: StrategicInteraction,
                                         simulation: StrategicSimulation):
        """Apply outcomes of a failed interaction."""
        civ_1 = self.core.get_civilization(interaction.participants[0])
        civ_2 = self.core.get_civilization(interaction.participants[1])
        
        if not civ_1 or not civ_2:
            return
        
        # Reduce cooperation levels
        civ_1.cooperation_level = max(0.0, civ_1.cooperation_level - 0.02)
        civ_2.cooperation_level = max(0.0, civ_2.cooperation_level - 0.02)
        
        # For conflicts, reduce stability
        if interaction.interaction_type == InteractionType.CONFLICT:
            civ_1.stability = max(0.0, civ_1.stability - 0.05)
            civ_2.stability = max(0.0, civ_2.stability - 0.05)
    
    def _update_civilization_states(self, simulation: StrategicSimulation):
        """Update civilization states based on interactions and policies."""
        for civ_id in simulation.participating_civilizations:
            civilization = self.core.get_civilization(civ_id)
            if not civilization:
                continue
            
            policy = simulation.policy_assignments.get(civ_id)
            if not policy:
                continue
            
            # Apply policy effects
            self._apply_policy_effects(civilization, policy, simulation)
    
    def _apply_policy_effects(self, civilization: Civilization, policy: PolicyArchetype,
                            simulation: StrategicSimulation):
        """Apply policy effects to a civilization."""
        if policy == PolicyArchetype.INNOVATIVE:
            civilization.innovation_capacity = min(1.0, civilization.innovation_capacity + 0.01)
        elif policy == PolicyArchetype.COOPERATIVE:
            civilization.cooperation_level = min(1.0, civilization.cooperation_level + 0.01)
        elif policy == PolicyArchetype.ADAPTIVE:
            civilization.adaptability = min(1.0, civilization.adaptability + 0.01)
        elif policy == PolicyArchetype.CONSERVATIVE:
            civilization.stability = min(1.0, civilization.stability + 0.01)
        elif policy == PolicyArchetype.AGGRESSIVE:
            civilization.strategic_capabilities['military_strength'] = min(1.0, 
                civilization.strategic_capabilities.get('military_strength', 0.5) + 0.01)
    
    def _evaluate_performance(self, simulation: StrategicSimulation):
        """Evaluate performance of civilizations in the simulation."""
        for civ_id in simulation.participating_civilizations:
            civilization = self.core.get_civilization(civ_id)
            if not civilization:
                continue
            
            # Calculate performance metrics
            performance = {
                'stability': civilization.stability,
                'complexity': civilization.complexity,
                'innovation_capacity': civilization.innovation_capacity,
                'cooperation_level': civilization.cooperation_level,
                'adaptability': civilization.adaptability,
                'resilience': civilization.resilience
            }
            
            simulation.civilization_outcomes[civ_id] = performance
            
            # Update policy effectiveness
            policy = simulation.policy_assignments.get(civ_id)
            if policy:
                self._update_policy_effectiveness(policy, performance)
    
    def _update_policy_effectiveness(self, policy: PolicyArchetype, performance: Dict[str, float]):
        """Update policy effectiveness tracking."""
        if policy not in self.policy_effectiveness:
            self.policy_effectiveness[policy] = {
                'total_performance': 0.0,
                'count': 0,
                'average_performance': 0.0
            }
        
        # Calculate overall performance
        overall_performance = sum(performance.values()) / len(performance)
        
        # Update tracking
        self.policy_effectiveness[policy]['total_performance'] += overall_performance
        self.policy_effectiveness[policy]['count'] += 1
        self.policy_effectiveness[policy]['average_performance'] = (
            self.policy_effectiveness[policy]['total_performance'] / 
            self.policy_effectiveness[policy]['count']
        )
    
    def _generate_meta_intelligence_insights(self, simulation: StrategicSimulation):
        """Generate meta-intelligence insights from simulation."""
        # Analyze policy effectiveness
        policy_insights = self._analyze_policy_effectiveness(simulation)
        
        # Analyze interaction patterns
        interaction_insights = self._analyze_interaction_patterns(simulation)
        
        # Generate architectural insights
        architectural_insights = self._generate_architectural_insights(simulation)
        
        # Store insights
        simulation.meta_intelligence_proposals.extend(policy_insights)
        simulation.architectural_insights.extend(architectural_insights)
        
        # Trigger meta-intelligence hooks
        for hook in self.meta_intelligence_hooks:
            try:
                hook(simulation, policy_insights, interaction_insights, architectural_insights)
            except Exception as e:
                print(f"Meta-intelligence hook error: {e}")
    
    def _analyze_policy_effectiveness(self, simulation: StrategicSimulation) -> List[Dict[str, Any]]:
        """Analyze policy effectiveness in the simulation."""
        insights = []
        
        # Compare policy performance
        policy_performance = {}
        for civ_id, policy in simulation.policy_assignments.items():
            if civ_id in simulation.civilization_outcomes:
                performance = simulation.civilization_outcomes[civ_id]
                overall_performance = sum(performance.values()) / len(performance)
                policy_performance[policy] = overall_performance
        
        # Identify best performing policies
        if policy_performance:
            best_policy = max(policy_performance, key=policy_performance.get)
            worst_policy = min(policy_performance, key=policy_performance.get)
            
            insights.append({
                'type': 'policy_effectiveness',
                'best_policy': best_policy.name,
                'worst_policy': worst_policy.name,
                'performance_gap': policy_performance[best_policy] - policy_performance[worst_policy]
            })
        
        return insights
    
    def _analyze_interaction_patterns(self, simulation: StrategicSimulation) -> List[Dict[str, Any]]:
        """Analyze interaction patterns in the simulation."""
        insights = []
        
        # Analyze interaction types
        interaction_types = {}
        for interaction in simulation.interactions:
            interaction_type = interaction.interaction_type
            if interaction_type not in interaction_types:
                interaction_types[interaction_type] = {'total': 0, 'successful': 0}
            interaction_types[interaction_type]['total'] += 1
            if interaction.success:
                interaction_types[interaction_type]['successful'] += 1
        
        # Calculate success rates
        for interaction_type, stats in interaction_types.items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            insights.append({
                'type': 'interaction_pattern',
                'interaction_type': interaction_type.name,
                'success_rate': success_rate,
                'total_interactions': stats['total']
            })
        
        return insights
    
    def _generate_architectural_insights(self, simulation: StrategicSimulation) -> List[Dict[str, Any]]:
        """Generate architectural insights from simulation."""
        insights = []
        
        # Analyze governance model effectiveness
        governance_performance = {}
        for civ_id in simulation.participating_civilizations:
            civilization = self.core.get_civilization(civ_id)
            if civilization and civ_id in simulation.civilization_outcomes:
                governance_model = civilization.governance_model
                performance = simulation.civilization_outcomes[civ_id]
                overall_performance = sum(performance.values()) / len(performance)
                
                if governance_model not in governance_performance:
                    governance_performance[governance_model] = []
                governance_performance[governance_model].append(overall_performance)
        
        # Identify best governance models
        if governance_performance:
            avg_performance = {model: np.mean(perfs) for model, perfs in governance_performance.items()}
            best_governance = max(avg_performance, key=avg_performance.get)
            
            insights.append({
                'type': 'architectural_insight',
                'best_governance_model': best_governance.name,
                'performance_advantage': avg_performance[best_governance] - np.mean(list(avg_performance.values()))
            })
        
        return insights
    
    def _finalize_simulation(self, simulation: StrategicSimulation):
        """Finalize a completed simulation."""
        # Calculate final performance metrics
        simulation.performance_metrics = {
            'total_interactions': len(simulation.interactions),
            'successful_interactions': sum(1 for i in simulation.interactions if i.success),
            'average_intensity': np.mean([i.intensity for i in simulation.interactions]) if simulation.interactions else 0.0,
            'policy_effectiveness': self.policy_effectiveness.copy()
        }
        
        # Update strategic metrics
        self.strategic_metrics[simulation.id] = simulation.performance_metrics
    
    def stop_simulation(self, simulation_id: str) -> bool:
        """Stop a running simulation."""
        if simulation_id in self.active_simulations:
            simulation = self.active_simulations[simulation_id]
            simulation.status = "completed"
            return True
        return False
    
    def get_simulation(self, simulation_id: str) -> Optional[StrategicSimulation]:
        """Get a simulation by ID."""
        return self.active_simulations.get(simulation_id)
    
    def get_simulation_results(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get simulation results."""
        simulation = self.get_simulation(simulation_id)
        if not simulation:
            return None
        
        return {
            'simulation': simulation,
            'performance_metrics': simulation.performance_metrics,
            'civilization_outcomes': simulation.civilization_outcomes,
            'meta_intelligence_proposals': simulation.meta_intelligence_proposals,
            'architectural_insights': simulation.architectural_insights
        }
    
    def get_policy_effectiveness(self) -> Dict[PolicyArchetype, Dict[str, float]]:
        """Get policy effectiveness metrics."""
        return self.policy_effectiveness.copy()
    
    def get_strategic_metrics(self) -> Dict[str, Any]:
        """Get overall strategic metrics."""
        return self.strategic_metrics.copy()
    
    def add_meta_intelligence_hook(self, hook: callable):
        """Add a meta-intelligence hook."""
        self.meta_intelligence_hooks.append(hook)
