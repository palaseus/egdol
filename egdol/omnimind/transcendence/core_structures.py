"""
Core Data Structures for OmniMind Civilization Intelligence Layer
Foundation for civilization-scale simulation and emergence detection.
"""

import uuid
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import networkx as nx
import numpy as np


class CivilizationArchetype(Enum):
    """Core civilization archetypes for procedural generation."""
    HIERARCHICAL = auto()
    DECENTRALIZED = auto()
    HYBRID = auto()
    EXPLORATORY = auto()
    COMPETITIVE = auto()
    COOPERATIVE = auto()


class AgentType(Enum):
    """Types of agents within civilizations."""
    LEADER = auto()
    WORKER = auto()
    INNOVATOR = auto()
    COORDINATOR = auto()
    SPECIALIST = auto()
    EXPLORER = auto()


class GovernanceModel(Enum):
    """Governance structure types."""
    AUTOCRATIC = auto()
    DEMOCRATIC = auto()
    CONSENSUS = auto()
    MERITOCRATIC = auto()
    TECHNOCRATIC = auto()
    ANARCHIST = auto()


@dataclass
class AgentCluster:
    """Grouped progeny agents with shared characteristics."""
    id: str
    cluster_type: AgentType
    size: int
    capabilities: Dict[str, float] = field(default_factory=dict)
    knowledge_domains: Set[str] = field(default_factory=set)
    communication_links: Set[str] = field(default_factory=set)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    productivity: float = 0.0
    innovation_rate: float = 0.0
    cooperation_level: float = 0.0


@dataclass
class EnvironmentState:
    """Environmental conditions affecting civilization development."""
    id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Resource pools
    resource_abundance: Dict[str, float] = field(default_factory=dict)
    resource_scarcity: Dict[str, float] = field(default_factory=dict)
    
    # Environmental factors
    climate_stability: float = 1.0
    natural_hazards: List[str] = field(default_factory=list)
    technological_opportunities: List[str] = field(default_factory=list)
    
    # Spatial constraints
    territorial_bounds: Dict[str, Any] = field(default_factory=dict)
    connectivity_network: nx.Graph = field(default_factory=nx.Graph)
    
    # External pressures
    competitive_pressure: float = 0.0
    cooperation_opportunities: float = 0.0


@dataclass
class TemporalState:
    """Temporal evolution state with rollback capabilities."""
    current_tick: int = 0
    epoch: int = 0
    simulation_time: float = 0.0
    
    # Rollback system
    checkpoint_interval: int = 100
    last_checkpoint: int = 0
    rollback_points: List[Dict[str, Any]] = field(default_factory=list)
    
    # Evolution tracking
    major_events: List[Dict[str, Any]] = field(default_factory=list)
    phase_transitions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    evolution_speed: float = 1.0
    stability_threshold: float = 0.8
    collapse_threshold: float = 0.2


@dataclass
class Civilization:
    """Core civilization structure for large-scale simulation."""
    id: str
    name: str
    archetype: CivilizationArchetype
    created_at: datetime = field(default_factory=datetime.now)
    
    # Population structure
    agent_clusters: Dict[str, AgentCluster] = field(default_factory=dict)
    total_population: int = 0
    population_growth_rate: float = 0.0
    
    # Governance and organization
    governance_model: GovernanceModel = GovernanceModel.DEMOCRATIC
    decision_making_efficiency: float = 0.5
    resource_allocation_system: Dict[str, Any] = field(default_factory=dict)
    
    # Knowledge and technology
    knowledge_base: Dict[str, float] = field(default_factory=dict)
    technological_level: float = 1.0
    innovation_capacity: float = 0.5
    knowledge_diffusion_rate: float = 0.1
    
    # Communication network
    communication_network: nx.Graph = field(default_factory=nx.Graph)
    information_flow_efficiency: float = 0.5
    coordination_capability: float = 0.5
    
    # Resource management
    resource_pools: Dict[str, float] = field(default_factory=dict)
    resource_efficiency: Dict[str, float] = field(default_factory=dict)
    sustainability_index: float = 0.5
    
    # Environmental interaction
    environment: EnvironmentState = field(default_factory=lambda: EnvironmentState(id=str(uuid.uuid4())))
    environmental_adaptation: float = 0.5
    ecological_footprint: float = 0.5
    
    # Temporal state
    temporal_state: TemporalState = field(default_factory=TemporalState)
    
    # Performance metrics
    stability: float = 0.5
    complexity: float = 0.5
    adaptability: float = 0.5
    resilience: float = 0.5
    growth_potential: float = 0.5
    cooperation_level: float = 0.5
    
    # Relationships with other civilizations
    diplomatic_relations: Dict[str, str] = field(default_factory=dict)  # civ_id -> relation_type
    trade_networks: Set[str] = field(default_factory=set)
    conflict_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Emergent properties
    emergent_patterns: List[str] = field(default_factory=list)
    cultural_traits: List[str] = field(default_factory=list)
    strategic_capabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class CivilizationSnapshot:
    """Complete civilization state for rollback system."""
    civilization_id: str
    timestamp: datetime
    tick: int
    
    # Core state
    civilization: Civilization
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Relationships
    relationships: Dict[str, str] = field(default_factory=dict)
    
    # Environment
    environment_state: EnvironmentState = field(default_factory=lambda: EnvironmentState(id=str(uuid.uuid4())))


class CivilizationIntelligenceCore:
    """Core intelligence layer for civilization-scale simulation."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the civilization intelligence core."""
        self.seed = seed or random.randint(0, 2**32-1)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Core registries
        self.civilizations: Dict[str, Civilization] = {}
        self.environments: Dict[str, EnvironmentState] = {}
        self.snapshots: List[CivilizationSnapshot] = []
        
        # Simulation state
        self.global_tick: int = 0
        self.simulation_active: bool = False
        self.rollback_enabled: bool = True
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {
            'total_civilizations': 0,
            'active_simulations': 0,
            'patterns_detected': 0,
            'experiments_completed': 0
        }
    
    def create_civilization(self, name: str, archetype: CivilizationArchetype, 
                          population_size: int, seed: Optional[int] = None) -> Civilization:
        """Create a new civilization with deterministic seeding."""
        civ_id = str(uuid.uuid4())
        
        # Set deterministic seed for this civilization
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate agent clusters based on archetype
        agent_clusters = self._generate_agent_clusters(archetype, population_size)
        
        # Create governance model
        governance_model = self._select_governance_model(archetype)
        
        # Initialize civilization
        civilization = Civilization(
            id=civ_id,
            name=name,
            archetype=archetype,
            agent_clusters=agent_clusters,
            total_population=population_size,
            governance_model=governance_model
        )
        
        # Initialize communication network
        self._initialize_communication_network(civilization)
        
        # Set initial resource pools
        self._initialize_resource_pools(civilization)
        
        # Register civilization
        self.civilizations[civ_id] = civilization
        self.performance_metrics['total_civilizations'] += 1
        
        return civilization
    
    def _generate_agent_clusters(self, archetype: CivilizationArchetype, 
                                population_size: int) -> Dict[str, AgentCluster]:
        """Generate agent clusters based on civilization archetype."""
        clusters = {}
        
        # Define cluster distributions by archetype
        if archetype == CivilizationArchetype.HIERARCHICAL:
            cluster_distribution = {
                AgentType.LEADER: 0.05,
                AgentType.COORDINATOR: 0.15,
                AgentType.WORKER: 0.60,
                AgentType.SPECIALIST: 0.15,
                AgentType.INNOVATOR: 0.05
            }
        elif archetype == CivilizationArchetype.DECENTRALIZED:
            cluster_distribution = {
                AgentType.COORDINATOR: 0.30,
                AgentType.WORKER: 0.40,
                AgentType.SPECIALIST: 0.20,
                AgentType.INNOVATOR: 0.10
            }
        elif archetype == CivilizationArchetype.EXPLORATORY:
            cluster_distribution = {
                AgentType.EXPLORER: 0.25,
                AgentType.INNOVATOR: 0.20,
                AgentType.SPECIALIST: 0.25,
                AgentType.WORKER: 0.30
            }
        else:  # Default hybrid
            cluster_distribution = {
                AgentType.LEADER: 0.03,
                AgentType.COORDINATOR: 0.12,
                AgentType.WORKER: 0.50,
                AgentType.SPECIALIST: 0.25,
                AgentType.INNOVATOR: 0.10
            }
        
        # Generate clusters
        for agent_type, proportion in cluster_distribution.items():
            cluster_size = max(1, int(population_size * proportion))
            cluster = AgentCluster(
                id=str(uuid.uuid4()),
                cluster_type=agent_type,
                size=cluster_size,
                capabilities=self._generate_capabilities(agent_type),
                knowledge_domains=self._assign_knowledge_domains(agent_type),
                resource_requirements=self._calculate_resource_requirements(agent_type),
                productivity=random.uniform(0.3, 0.9),
                innovation_rate=random.uniform(0.1, 0.8),
                cooperation_level=random.uniform(0.4, 0.9)
            )
            clusters[cluster.id] = cluster
        
        return clusters
    
    def _generate_capabilities(self, agent_type: AgentType) -> Dict[str, float]:
        """Generate capabilities for agent type."""
        base_capabilities = {
            'intelligence': random.uniform(0.4, 0.9),
            'creativity': random.uniform(0.3, 0.8),
            'cooperation': random.uniform(0.5, 0.9),
            'leadership': random.uniform(0.2, 0.8),
            'technical_skill': random.uniform(0.3, 0.9),
            'social_skill': random.uniform(0.4, 0.8)
        }
        
        # Adjust based on agent type
        if agent_type == AgentType.LEADER:
            base_capabilities['leadership'] = random.uniform(0.7, 0.95)
            base_capabilities['intelligence'] = random.uniform(0.6, 0.9)
        elif agent_type == AgentType.INNOVATOR:
            base_capabilities['creativity'] = random.uniform(0.7, 0.95)
            base_capabilities['technical_skill'] = random.uniform(0.6, 0.9)
        elif agent_type == AgentType.COORDINATOR:
            base_capabilities['cooperation'] = random.uniform(0.7, 0.95)
            base_capabilities['social_skill'] = random.uniform(0.6, 0.9)
        
        return base_capabilities
    
    def _assign_knowledge_domains(self, agent_type: AgentType) -> Set[str]:
        """Assign knowledge domains to agent type."""
        all_domains = {
            'governance', 'technology', 'economics', 'culture', 'environment',
            'communication', 'resource_management', 'innovation', 'coordination'
        }
        
        # Assign 2-4 domains based on agent type
        num_domains = random.randint(2, 4)
        return set(random.sample(list(all_domains), num_domains))
    
    def _calculate_resource_requirements(self, agent_type: AgentType) -> Dict[str, float]:
        """Calculate resource requirements for agent type."""
        base_requirements = {
            'energy': random.uniform(0.1, 0.5),
            'materials': random.uniform(0.1, 0.4),
            'information': random.uniform(0.1, 0.6),
            'coordination': random.uniform(0.1, 0.5)
        }
        
        # Adjust based on agent type
        if agent_type == AgentType.LEADER:
            base_requirements['coordination'] = random.uniform(0.6, 0.9)
        elif agent_type == AgentType.INNOVATOR:
            base_requirements['information'] = random.uniform(0.6, 0.9)
        
        return base_requirements
    
    def _select_governance_model(self, archetype: CivilizationArchetype) -> GovernanceModel:
        """Select governance model based on archetype."""
        if archetype == CivilizationArchetype.HIERARCHICAL:
            return random.choice([GovernanceModel.AUTOCRATIC, GovernanceModel.MERITOCRATIC])
        elif archetype == CivilizationArchetype.DECENTRALIZED:
            return random.choice([GovernanceModel.DEMOCRATIC, GovernanceModel.CONSENSUS])
        elif archetype == CivilizationArchetype.COMPETITIVE:
            return random.choice([GovernanceModel.MERITOCRATIC, GovernanceModel.TECHNOCRATIC])
        else:
            return random.choice(list(GovernanceModel))
    
    def _initialize_communication_network(self, civilization: Civilization):
        """Initialize communication network for civilization."""
        # Create network based on governance model
        if civilization.governance_model in [GovernanceModel.AUTOCRATIC, GovernanceModel.MERITOCRATIC]:
            # Hierarchical network
            self._create_hierarchical_network(civilization)
        else:
            # Decentralized network
            self._create_decentralized_network(civilization)
    
    def _create_hierarchical_network(self, civilization: Civilization):
        """Create hierarchical communication network."""
        # Implementation for hierarchical network structure
        pass
    
    def _create_decentralized_network(self, civilization: Civilization):
        """Create decentralized communication network."""
        # Implementation for decentralized network structure
        pass
    
    def _initialize_resource_pools(self, civilization: Civilization):
        """Initialize resource pools for civilization."""
        # Set initial resource levels based on archetype and environment
        base_resources = {
            'energy': random.uniform(50, 200),
            'materials': random.uniform(30, 150),
            'information': random.uniform(20, 100),
            'coordination': random.uniform(10, 80)
        }
        
        civilization.resource_pools = base_resources
        civilization.resource_efficiency = {
            resource: random.uniform(0.3, 0.8) for resource in base_resources.keys()
        }
    
    def create_snapshot(self, civilization_id: str) -> CivilizationSnapshot:
        """Create a snapshot for rollback capability."""
        if civilization_id not in self.civilizations:
            raise ValueError(f"Civilization {civilization_id} not found")
        
        civilization = self.civilizations[civilization_id]
        snapshot = CivilizationSnapshot(
            civilization_id=civilization_id,
            timestamp=datetime.now(),
            tick=self.global_tick,
            civilization=civilization,
            metrics={
                'stability': civilization.stability,
                'complexity': civilization.complexity,
                'adaptability': civilization.adaptability,
                'resilience': civilization.resilience
            },
            relationships=civilization.diplomatic_relations.copy(),
            environment_state=civilization.environment
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def rollback_to_snapshot(self, snapshot: CivilizationSnapshot) -> bool:
        """Rollback civilization to a previous snapshot."""
        try:
            # Restore civilization state
            self.civilizations[snapshot.civilization_id] = snapshot.civilization
            
            # Restore global state
            self.global_tick = snapshot.tick
            
            return True
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False
    
    def get_civilization(self, civilization_id: str) -> Optional[Civilization]:
        """Get civilization by ID."""
        return self.civilizations.get(civilization_id)
    
    def list_civilizations(self) -> List[Civilization]:
        """List all civilizations."""
        return list(self.civilizations.values())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
