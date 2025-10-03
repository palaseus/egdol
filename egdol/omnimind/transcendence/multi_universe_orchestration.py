"""
Multi-Universe Simulation Orchestration for OmniMind Civilization Intelligence Layer
Enables running entire "universes" with different fundamental rule sets and comparing emergent macro-laws.
"""

import json
import uuid
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core_structures import (
    Civilization, AgentCluster, EnvironmentState, TemporalState,
    CivilizationIntelligenceCore, GovernanceModel, AgentType, CivilizationArchetype
)
from ...utils.pretty_printing import print_universe_status, pp
from .pattern_codification_engine import PatternBlueprint, BlueprintType, StrategicDoctrine
from .civilizational_genetic_archive import CivilizationDNA, LineageType, ArchiveStatus
from .strategic_feedback_loop import FeedbackApplication, FeedbackType
from .reflexive_introspection_layer import IntrospectionAnalysis, MetaRule, MetaRuleType


class UniverseType(Enum):
    """Types of universes with different fundamental rule sets."""
    STANDARD = auto()
    HIGH_RESOURCE_SCARCITY = auto()
    LOW_COMMUNICATION_COST = auto()
    HIGH_MUTATION_RATE = auto()
    COOPERATIVE_BIAS = auto()
    COMPETITIVE_BIAS = auto()
    HIERARCHICAL_CONSTRAINT = auto()
    FLAT_STRUCTURE = auto()
    RAPID_EVOLUTION = auto()
    STABLE_ENVIRONMENT = auto()


class UniverseStatus(Enum):
    """Status of universe simulation."""
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    ARCHIVED = auto()


@dataclass
class UniverseParameters:
    """Parameters defining a universe's fundamental rule set."""
    universe_id: str
    universe_type: UniverseType
    name: str
    
    # Physical laws
    resource_scarcity_multiplier: float = 1.0
    communication_cost_multiplier: float = 1.0
    mutation_rate_multiplier: float = 1.0
    evolution_speed_multiplier: float = 1.0
    
    # Social laws
    cooperation_bias: float = 0.0  # -1.0 to 1.0
    competition_bias: float = 0.0  # -1.0 to 1.0
    hierarchy_constraint: float = 0.0  # 0.0 to 1.0
    innovation_rate_multiplier: float = 1.0
    
    # Environmental laws
    environmental_stability: float = 1.0
    natural_disaster_frequency: float = 0.1
    resource_renewal_rate: float = 1.0
    
    # Temporal laws
    time_dilation_factor: float = 1.0
    temporal_resolution: float = 1.0
    simulation_duration: int = 1000  # ticks
    
    # Meta-parameters
    max_civilizations: int = 10
    min_civilizations: int = 3
    universe_seed: int = 42


@dataclass
class UniverseState:
    """Current state of a universe simulation."""
    universe_id: str
    status: UniverseStatus
    current_tick: int = 0
    civilizations: List[str] = field(default_factory=list)
    active_civilizations: List[str] = field(default_factory=list)
    
    # Universe metrics
    total_energy: float = 1000.0
    total_resources: Dict[str, float] = field(default_factory=dict)
    universe_stability: float = 1.0
    universe_complexity: float = 0.0
    
    # Emergent properties
    emergent_laws: List[Dict[str, Any]] = field(default_factory=list)
    cross_civilization_patterns: List[Dict[str, Any]] = field(default_factory=list)
    universe_meta_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    simulation_start_time: datetime = field(default_factory=datetime.now)
    last_update_time: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CrossUniverseAnalysis:
    """Analysis comparing multiple universes."""
    analysis_id: str
    universe_ids: List[str]
    analysis_timestamp: datetime
    
    # Comparative metrics
    universe_performance: Dict[str, float] = field(default_factory=dict)
    emergent_law_comparison: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    meta_rule_consistency: Dict[str, float] = field(default_factory=dict)
    
    # Universal laws
    universal_laws: List[Dict[str, Any]] = field(default_factory=list)
    universe_specific_laws: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # Insights
    cross_universe_insights: List[str] = field(default_factory=list)
    optimization_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Meta-analysis
    analysis_confidence: float = 0.0
    universal_truths: List[Dict[str, Any]] = field(default_factory=list)


class MultiUniverseOrchestrator:
    """Orchestrator for multi-universe simulation and analysis."""
    
    def __init__(self, core: CivilizationIntelligenceCore, 
                 pattern_codification_engine, genetic_archive, 
                 strategic_feedback_loop, introspection_layer):
        """Initialize the multi-universe orchestrator."""
        self.core = core
        self.pattern_codification_engine = pattern_codification_engine
        self.genetic_archive = genetic_archive
        self.strategic_feedback_loop = strategic_feedback_loop
        self.introspection_layer = introspection_layer
        
        # Universe management
        self.universes: Dict[str, UniverseState] = {}
        self.universe_parameters: Dict[str, UniverseParameters] = {}
        self.active_simulations: Dict[str, threading.Thread] = {}
        
        # Cross-universe analysis
        self.cross_universe_analyses: Dict[str, CrossUniverseAnalysis] = {}
        
        # Orchestration parameters
        self.orchestration_parameters = {
            'max_concurrent_universes': 5,
            'universe_timeout': 300.0,  # seconds
            'analysis_frequency': 100,  # ticks
            'cross_universe_analysis_threshold': 3,  # minimum universes
            'universal_law_confidence_threshold': 0.8
        }
        
        # Performance tracking
        self.orchestration_stats = {
            'total_universes_created': 0,
            'total_universes_completed': 0,
            'total_cross_universe_analyses': 0,
            'universal_laws_discovered': 0,
            'universe_specific_laws_discovered': 0,
            'average_universe_performance': 0.0,
            'best_performing_universe': None,
            'most_innovative_universe': None
        }
        
        # Initialize default universe types
        self._initialize_default_universe_types()
    
    def _initialize_default_universe_types(self):
        """Initialize default universe parameter sets."""
        # Standard universe
        self._create_universe_parameters(
            UniverseType.STANDARD, "Standard Universe",
            resource_scarcity_multiplier=1.0,
            communication_cost_multiplier=1.0,
            mutation_rate_multiplier=1.0,
            evolution_speed_multiplier=1.0
        )
        
        # High resource scarcity universe
        self._create_universe_parameters(
            UniverseType.HIGH_RESOURCE_SCARCITY, "Resource Scarcity Universe",
            resource_scarcity_multiplier=3.0,
            communication_cost_multiplier=1.5,
            mutation_rate_multiplier=1.2,
            evolution_speed_multiplier=0.8
        )
        
        # Low communication cost universe
        self._create_universe_parameters(
            UniverseType.LOW_COMMUNICATION_COST, "High Connectivity Universe",
            resource_scarcity_multiplier=1.0,
            communication_cost_multiplier=0.3,
            mutation_rate_multiplier=1.5,
            evolution_speed_multiplier=1.2
        )
        
        # Cooperative bias universe
        self._create_universe_parameters(
            UniverseType.COOPERATIVE_BIAS, "Cooperative Universe",
            cooperation_bias=0.7,
            competition_bias=-0.3,
            hierarchy_constraint=0.2,
            innovation_rate_multiplier=1.3
        )
        
        # Competitive bias universe
        self._create_universe_parameters(
            UniverseType.COMPETITIVE_BIAS, "Competitive Universe",
            cooperation_bias=-0.3,
            competition_bias=0.7,
            hierarchy_constraint=0.8,
            innovation_rate_multiplier=0.8
        )
    
    def _create_universe_parameters(self, universe_type: UniverseType, name: str, **kwargs):
        """Create universe parameters for a specific type."""
        params = UniverseParameters(
            universe_id=str(uuid.uuid4()),
            universe_type=universe_type,
            name=name,
            **kwargs
        )
        self.universe_parameters[params.universe_id] = params
    
    def create_universe(self, universe_type: UniverseType, 
                       custom_parameters: Optional[Dict[str, Any]] = None) -> str:
        """Create a new universe with specified parameters."""
        # Find or create universe parameters
        universe_params = None
        for params in self.universe_parameters.values():
            if params.universe_type == universe_type:
                universe_params = params
                break
        
        if not universe_params:
            # Create new parameters for this type
            universe_params = UniverseParameters(
                universe_id=str(uuid.uuid4()),
                universe_type=universe_type,
                name=f"{universe_type.name} Universe"
            )
            self.universe_parameters[universe_params.universe_id] = universe_params
        
        # Apply custom parameters if provided
        if custom_parameters:
            for key, value in custom_parameters.items():
                if hasattr(universe_params, key):
                    setattr(universe_params, key, value)
        
        # Create universe state
        universe_state = UniverseState(
            universe_id=universe_params.universe_id,
            status=UniverseStatus.INITIALIZING
        )
        
        self.universes[universe_state.universe_id] = universe_state
        self.orchestration_stats['total_universes_created'] += 1
        
        return universe_state.universe_id
    
    def initialize_universe_civilizations(self, universe_id: str, 
                                          civilization_count: int = None) -> bool:
        """Initialize civilizations in a universe."""
        universe_state = self.universes.get(universe_id)
        universe_params = self.universe_parameters.get(universe_id)
        
        if not universe_state or not universe_params:
            return False
        
        if civilization_count is None:
            civilization_count = random.randint(
                universe_params.min_civilizations, 
                universe_params.max_civilizations
            )
        
        # Create civilizations with universe-specific characteristics
        civilizations = []
        for i in range(civilization_count):
            civ = self._create_universe_civilization(universe_id, i)
            if civ:
                civilizations.append(civ.id)
        
        universe_state.civilizations = civilizations
        universe_state.active_civilizations = civilizations.copy()
        universe_state.status = UniverseStatus.RUNNING
        
        return len(civilizations) > 0
    
    def _create_universe_civilization(self, universe_id: str, index: int) -> Optional[Civilization]:
        """Create a civilization adapted to universe parameters."""
        universe_params = self.universe_parameters.get(universe_id)
        if not universe_params:
            return None
        
        # Create civilization with universe-specific characteristics
        from .civilization_architect import CivilizationArchitect
        
        architect = CivilizationArchitect(self.core)
        
        # Choose archetype based on universe parameters
        if universe_params.cooperation_bias > 0.3:
            archetype = CivilizationArchetype.COOPERATIVE
        elif universe_params.competition_bias > 0.3:
            archetype = CivilizationArchetype.HIERARCHICAL
        else:
            archetype = random.choice(list(CivilizationArchetype))
        
        # Create civilization
        civilization = architect.generate_civilization(
            name=f"Universe {universe_id[:8]} Civ {index+1}",
            archetype=archetype,
            population_size=random.randint(50, 200),
            deterministic_seed=universe_params.universe_seed + index
        )
        
        # Apply universe-specific modifications
        self._apply_universe_parameters_to_civilization(civilization, universe_params)
        
        return civilization
    
    def _apply_universe_parameters_to_civilization(self, civilization: Civilization, 
                                                 universe_params: UniverseParameters):
        """Apply universe parameters to a civilization."""
        # Apply resource scarcity
        if universe_params.resource_scarcity_multiplier > 1.0:
            for resource in civilization.resource_pools:
                civilization.resource_pools[resource] *= (1.0 / universe_params.resource_scarcity_multiplier)
        
        # Apply cooperation bias
        if universe_params.cooperation_bias != 0.0:
            civilization.cooperation_level = max(0.0, min(1.0, 
                civilization.cooperation_level + universe_params.cooperation_bias * 0.3))
        
        # Apply innovation rate
        if universe_params.innovation_rate_multiplier != 1.0:
            civilization.innovation_capacity *= universe_params.innovation_rate_multiplier
            civilization.innovation_capacity = max(0.0, min(1.0, civilization.innovation_capacity))
        
        # Apply hierarchy constraints
        if universe_params.hierarchy_constraint > 0.0:
            # Modify governance model based on hierarchy constraint
            if universe_params.hierarchy_constraint > 0.5:
                civilization.governance_model = GovernanceModel.AUTOCRATIC
            else:
                civilization.governance_model = GovernanceModel.DEMOCRATIC
    
    def run_universe_simulation(self, universe_id: str, 
                               duration: int = None) -> bool:
        """Run a universe simulation."""
        universe_state = self.universes.get(universe_id)
        universe_params = self.universe_parameters.get(universe_id)
        
        if not universe_state or not universe_params:
            return False
        
        if universe_state.status != UniverseStatus.RUNNING:
            return False
        
        if duration is None:
            duration = universe_params.simulation_duration
        
        try:
            # Run simulation in a separate thread
            simulation_thread = threading.Thread(
                target=self._run_universe_simulation_thread,
                args=(universe_id, duration),
                daemon=True
            )
            
            self.active_simulations[universe_id] = simulation_thread
            simulation_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting universe simulation {universe_id}: {e}")
            universe_state.status = UniverseStatus.FAILED
            return False
    
    def _run_universe_simulation_thread(self, universe_id: str, duration: int):
        """Run universe simulation in a separate thread."""
        universe_state = self.universes.get(universe_id)
        universe_params = self.universe_parameters.get(universe_id)
        
        if not universe_state or not universe_params:
            return
        
        try:
            # Initialize evolution engine for this universe
            from .temporal_evolution_engine import TemporalEvolutionEngine
            evolution_engine = TemporalEvolutionEngine(self.core)
            
            # Run simulation
            for tick in range(duration):
                universe_state.current_tick = tick
                
                # Apply universe-specific rules each tick
                self._apply_universe_rules(universe_id, tick)
                
                # Run civilization evolution
                if universe_state.active_civilizations:
                    evolution_engine.start_evolution(
                        universe_state.active_civilizations, 
                        deterministic_seed=universe_params.universe_seed + tick
                    )
                    
                    # Let evolution run for a short time
                    time.sleep(0.1 * universe_params.evolution_speed_multiplier)
                    
                    evolution_engine.stop_evolution()
                
                # Update universe metrics
                self._update_universe_metrics(universe_id)
                
                # Perform periodic analysis
                if tick % self.orchestration_parameters['analysis_frequency'] == 0:
                    self._analyze_universe_emergent_patterns(universe_id)
            
            # Final analysis
            self._perform_final_universe_analysis(universe_id)
            
            universe_state.status = UniverseStatus.COMPLETED
            self.orchestration_stats['total_universes_completed'] += 1
            
        except Exception as e:
            print(f"Error in universe simulation {universe_id}: {e}")
            universe_state.status = UniverseStatus.FAILED
    
    def _apply_universe_rules(self, universe_id: str, tick: int):
        """Apply universe-specific rules to the simulation."""
        universe_state = self.universes.get(universe_id)
        universe_params = self.universe_parameters.get(universe_id)
        
        if not universe_state or not universe_params:
            return
        
        # Apply environmental changes
        if universe_params.natural_disaster_frequency > 0.0:
            if random.random() < universe_params.natural_disaster_frequency:
                self._apply_natural_disaster(universe_id)
        
        # Apply resource renewal
        if universe_params.resource_renewal_rate != 1.0:
            self._apply_resource_renewal(universe_id)
        
        # Apply communication cost changes
        if universe_params.communication_cost_multiplier != 1.0:
            self._apply_communication_cost_changes(universe_id)
    
    def _apply_natural_disaster(self, universe_id: str):
        """Apply a natural disaster to the universe."""
        universe_state = self.universes.get(universe_id)
        if not universe_state:
            return
        
        # Randomly affect some civilizations
        affected_civs = random.sample(
            universe_state.active_civilizations, 
            min(2, len(universe_state.active_civilizations))
        )
        
        for civ_id in affected_civs:
            civilization = self.core.get_civilization(civ_id)
            if civilization:
                # Reduce stability and resources
                civilization.stability *= 0.8
                for resource in civilization.resource_pools:
                    civilization.resource_pools[resource] *= 0.7
    
    def _apply_resource_renewal(self, universe_id: str):
        """Apply resource renewal to the universe."""
        universe_state = self.universes.get(universe_id)
        universe_params = self.universe_parameters.get(universe_id)
        
        if not universe_state or not universe_params:
            return
        
        for civ_id in universe_state.active_civilizations:
            civilization = self.core.get_civilization(civ_id)
            if civilization:
                # Renew resources
                for resource in civilization.resource_pools:
                    civilization.resource_pools[resource] *= universe_params.resource_renewal_rate
    
    def _apply_communication_cost_changes(self, universe_id: str):
        """Apply communication cost changes to the universe."""
        universe_state = self.universes.get(universe_id)
        universe_params = self.universe_parameters.get(universe_id)
        
        if not universe_state or not universe_params:
            return
        
        for civ_id in universe_state.active_civilizations:
            civilization = self.core.get_civilization(civ_id)
            if civilization:
                # Modify communication network topology (proxy for communication efficiency)
                if hasattr(civilization, 'communication_network_topology'):
                    # Apply communication cost changes through network modifications
                    pass  # For now, skip direct modification as the attribute doesn't exist
    
    def _update_universe_metrics(self, universe_id: str):
        """Update universe-level metrics."""
        universe_state = self.universes.get(universe_id)
        if not universe_state:
            return
        
        # Calculate universe stability
        stability_values = []
        for civ_id in universe_state.active_civilizations:
            civilization = self.core.get_civilization(civ_id)
            if civilization:
                stability_values.append(civilization.stability)
        
        if stability_values:
            universe_state.universe_stability = sum(stability_values) / len(stability_values)
        
        # Calculate universe complexity
        complexity_values = []
        for civ_id in universe_state.active_civilizations:
            civilization = self.core.get_civilization(civ_id)
            if civilization:
                complexity_values.append(civilization.complexity)
        
        if complexity_values:
            universe_state.universe_complexity = sum(complexity_values) / len(complexity_values)
        
        # Update performance metrics
        universe_state.performance_metrics = {
            'stability': universe_state.universe_stability,
            'complexity': universe_state.universe_complexity,
            'active_civilizations': len(universe_state.active_civilizations),
            'total_civilizations': len(universe_state.civilizations)
        }
    
    def _analyze_universe_emergent_patterns(self, universe_id: str):
        """Analyze emergent patterns in a universe."""
        universe_state = self.universes.get(universe_id)
        if not universe_state:
            return
        
        # Use introspection layer to analyze universe
        analysis_id = self.introspection_layer.perform_comprehensive_introspection(
            universe_state.active_civilizations
        )
        
        if analysis_id:
            analysis = self.introspection_layer.get_analysis(analysis_id)
            if analysis:
                # Store emergent patterns
                universe_state.emergent_laws.extend(analysis.discovered_meta_rules)
                universe_state.cross_civilization_patterns.extend(analysis.discovered_patterns)
    
    def _perform_final_universe_analysis(self, universe_id: str):
        """Perform final analysis of a universe."""
        universe_state = self.universes.get(universe_id)
        if not universe_state:
            return
        
        # Perform comprehensive introspection
        analysis_id = self.introspection_layer.perform_comprehensive_introspection(
            universe_state.active_civilizations
        )
        
        if analysis_id:
            analysis = self.introspection_layer.get_analysis(analysis_id)
            if analysis:
                # Store final analysis results
                universe_state.universe_meta_rules = analysis.discovered_meta_rules.copy()
                
                # Update orchestration statistics
                if analysis.confidence_score > self.orchestration_stats['average_universe_performance']:
                    self.orchestration_stats['best_performing_universe'] = universe_id
                    self.orchestration_stats['average_universe_performance'] = analysis.confidence_score
    
    def run_cross_universe_analysis(self, universe_ids: List[str] = None) -> str:
        """Run cross-universe analysis to discover universal laws."""
        if universe_ids is None:
            universe_ids = [uid for uid, state in self.universes.items() 
                          if state.status == UniverseStatus.COMPLETED]
        
        if len(universe_ids) < self.orchestration_parameters['cross_universe_analysis_threshold']:
            print(f"Not enough completed universes for cross-universe analysis ({len(universe_ids)} < {self.orchestration_parameters['cross_universe_analysis_threshold']})")
            return None
        
        # Create cross-universe analysis
        analysis = CrossUniverseAnalysis(
            analysis_id=str(uuid.uuid4()),
            universe_ids=universe_ids,
            analysis_timestamp=datetime.now()
        )
        
        try:
            # Analyze each universe
            for universe_id in universe_ids:
                universe_state = self.universes.get(universe_id)
                if universe_state:
                    # Calculate universe performance
                    performance = self._calculate_universe_performance(universe_id)
                    analysis.universe_performance[universe_id] = performance
                    
                    # Collect emergent laws
                    analysis.emergent_law_comparison[universe_id] = universe_state.universe_meta_rules.copy()
            
            # Discover universal laws
            self._discover_universal_laws(analysis)
            
            # Discover universe-specific laws
            self._discover_universe_specific_laws(analysis)
            
            # Generate cross-universe insights
            self._generate_cross_universe_insights(analysis)
            
            # Calculate analysis confidence
            analysis.analysis_confidence = self._calculate_cross_universe_confidence(analysis)
            
            # Store analysis
            self.cross_universe_analyses[analysis.analysis_id] = analysis
            self.orchestration_stats['total_cross_universe_analyses'] += 1
            
            return analysis.analysis_id
            
        except Exception as e:
            print(f"Error in cross-universe analysis: {e}")
            return None
    
    def _calculate_universe_performance(self, universe_id: str) -> float:
        """Calculate performance score for a universe."""
        universe_state = self.universes.get(universe_id)
        if not universe_state:
            return 0.0
        
        # Calculate performance based on multiple factors
        performance_factors = []
        
        # Stability factor
        performance_factors.append(universe_state.universe_stability)
        
        # Complexity factor (moderate complexity is good)
        complexity_score = 1.0 - abs(universe_state.universe_complexity - 0.5) * 2
        performance_factors.append(complexity_score)
        
        # Civilization survival rate
        survival_rate = len(universe_state.active_civilizations) / max(len(universe_state.civilizations), 1)
        performance_factors.append(survival_rate)
        
        # Emergent law discovery rate
        law_discovery_rate = len(universe_state.emergent_laws) / max(len(universe_state.civilizations), 1)
        performance_factors.append(min(1.0, law_discovery_rate))
        
        return sum(performance_factors) / len(performance_factors)
    
    def _discover_universal_laws(self, analysis: CrossUniverseAnalysis):
        """Discover laws that hold across all universes."""
        # Collect all meta-rules from all universes
        all_meta_rules = []
        for universe_id in analysis.universe_ids:
            universe_rules = analysis.emergent_law_comparison.get(universe_id, [])
            all_meta_rules.extend(universe_rules)
        
        # Find rules that appear in multiple universes
        rule_frequency = {}
        for rule in all_meta_rules:
            rule_name = rule.get('name', 'Unknown Rule')
            if rule_name not in rule_frequency:
                rule_frequency[rule_name] = 0
            rule_frequency[rule_name] += 1
        
        # Identify universal laws (appear in multiple universes)
        total_universes = len(analysis.universe_ids)
        for rule_name, frequency in rule_frequency.items():
            if frequency >= total_universes * 0.5:  # Appears in at least 50% of universes
                universal_law = {
                    'name': rule_name,
                    'frequency': frequency,
                    'universality_score': frequency / total_universes,
                    'description': f"Universal law appearing in {frequency}/{total_universes} universes"
                }
                analysis.universal_laws.append(universal_law)
                self.orchestration_stats['universal_laws_discovered'] += 1
    
    def _discover_universe_specific_laws(self, analysis: CrossUniverseAnalysis):
        """Discover laws specific to certain universe types."""
        # Group rules by universe type
        universe_type_rules = {}
        for universe_id in analysis.universe_ids:
            universe_params = self.universe_parameters.get(universe_id)
            if universe_params:
                universe_type = universe_params.universe_type
                if universe_type not in universe_type_rules:
                    universe_type_rules[universe_type] = []
                universe_type_rules[universe_type].extend(
                    analysis.emergent_law_comparison.get(universe_id, [])
                )
        
        # Find rules specific to universe types
        for universe_type, rules in universe_type_rules.items():
            if len(rules) > 0:
                universe_specific_laws = {
                    'universe_type': universe_type.name,
                    'laws': rules,
                    'law_count': len(rules),
                    'description': f"Laws specific to {universe_type.name} universes"
                }
                analysis.universe_specific_laws[universe_type.name] = universe_specific_laws
                self.orchestration_stats['universe_specific_laws_discovered'] += len(rules)
    
    def _generate_cross_universe_insights(self, analysis: CrossUniverseAnalysis):
        """Generate insights from cross-universe analysis."""
        insights = []
        
        # Performance insights
        if analysis.universe_performance:
            best_universe = max(analysis.universe_performance, key=analysis.universe_performance.get)
            worst_universe = min(analysis.universe_performance, key=analysis.universe_performance.get)
            
            insights.append(f"Best performing universe: {best_universe} (score: {analysis.universe_performance[best_universe]:.3f})")
            insights.append(f"Worst performing universe: {worst_universe} (score: {analysis.universe_performance[worst_universe]:.3f})")
        
        # Universal law insights
        if analysis.universal_laws:
            insights.append(f"Discovered {len(analysis.universal_laws)} universal laws")
            for law in analysis.universal_laws:
                insights.append(f"Universal law: {law['name']} (universality: {law['universality_score']:.3f})")
        
        # Universe-specific insights
        if analysis.universe_specific_laws:
            insights.append(f"Discovered laws specific to {len(analysis.universe_specific_laws)} universe types")
            for universe_type, laws in analysis.universe_specific_laws.items():
                insights.append(f"{universe_type} universes have {laws['law_count']} specific laws")
        
        analysis.cross_universe_insights = insights
    
    def _calculate_cross_universe_confidence(self, analysis: CrossUniverseAnalysis) -> float:
        """Calculate confidence score for cross-universe analysis."""
        confidence_factors = []
        
        # Factor 1: Number of universes analyzed
        universe_count = len(analysis.universe_ids)
        universe_confidence = min(1.0, universe_count / 10.0)  # Normalize to 0-1
        confidence_factors.append(universe_confidence)
        
        # Factor 2: Universal law discovery rate
        if analysis.universal_laws:
            law_confidence = min(1.0, len(analysis.universal_laws) / 5.0)  # Normalize to 0-1
            confidence_factors.append(law_confidence)
        
        # Factor 3: Universe performance variance
        if analysis.universe_performance:
            performance_values = list(analysis.universe_performance.values())
            if len(performance_values) > 1:
                import statistics
                performance_variance = statistics.stdev(performance_values)
                variance_confidence = min(1.0, performance_variance)  # Higher variance = more interesting
                confidence_factors.append(variance_confidence)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
    
    def get_universe_status(self, universe_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a universe."""
        universe_state = self.universes.get(universe_id)
        if not universe_state:
            return None
        
        return {
            'universe_id': universe_id,
            'status': universe_state.status.name,
            'current_tick': universe_state.current_tick,
            'civilizations': len(universe_state.civilizations),
            'active_civilizations': len(universe_state.active_civilizations),
            'universe_stability': universe_state.universe_stability,
            'universe_complexity': universe_state.universe_complexity,
            'emergent_laws': len(universe_state.emergent_laws),
            'performance_metrics': universe_state.performance_metrics
        }
    
    def get_cross_universe_analysis(self, analysis_id: str) -> Optional[CrossUniverseAnalysis]:
        """Get cross-universe analysis by ID."""
        return self.cross_universe_analyses.get(analysis_id)
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        return {
            'orchestration_stats': self.orchestration_stats.copy(),
            'total_universes': len(self.universes),
            'active_simulations': len(self.active_simulations),
            'completed_universes': len([u for u in self.universes.values() if u.status == UniverseStatus.COMPLETED]),
            'failed_universes': len([u for u in self.universes.values() if u.status == UniverseStatus.FAILED]),
            'total_cross_universe_analyses': len(self.cross_universe_analyses),
            'universe_types': {
                universe_type.name: len([u for u in self.universes.values() 
                                      if self.universe_parameters.get(u.universe_id) and 
                                      self.universe_parameters[u.universe_id].universe_type == universe_type])
                for universe_type in list(UniverseType)
            }
        }
