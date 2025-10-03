"""
Civilization Architect for OmniMind Civilization Intelligence Layer
Procedural generator with configurable archetypes and deterministic seeding.
"""

import uuid
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import networkx as nx

from ...utils.pretty_printing import print_civilization, pp

from .core_structures import (
    Civilization, AgentCluster, EnvironmentState, CivilizationArchetype, 
    AgentType, GovernanceModel, CivilizationIntelligenceCore
)


class CivilizationSize(Enum):
    """Civilization size categories."""
    SMALL = auto()      # 10-100 agents
    MEDIUM = auto()     # 100-1000 agents
    LARGE = auto()      # 1000-10000 agents
    MASSIVE = auto()    # 10000+ agents


class ResourceType(Enum):
    """Types of resources in civilizations."""
    ENERGY = auto()
    MATERIALS = auto()
    INFORMATION = auto()
    COORDINATION = auto()
    KNOWLEDGE = auto()
    INNOVATION = auto()


@dataclass
class CivilizationBlueprint:
    """Template for generating civilizations."""
    name: str
    archetype: CivilizationArchetype
    size_category: CivilizationSize
    population_range: Tuple[int, int]
    
    # Governance characteristics
    preferred_governance: List[GovernanceModel]
    decision_efficiency_range: Tuple[float, float]
    
    # Agent distributions
    agent_type_distribution: Dict[AgentType, float]
    capability_ranges: Dict[str, Tuple[float, float]]
    
    # Resource characteristics
    resource_abundance: Dict[ResourceType, float]
    resource_efficiency: Dict[ResourceType, float]
    
    # Environmental preferences
    climate_preference: float  # 0.0 (harsh) to 1.0 (benign)
    resource_richness: float  # 0.0 (scarce) to 1.0 (abundant)
    
    # Cultural traits
    cooperation_tendency: float  # 0.0 (competitive) to 1.0 (cooperative)
    innovation_focus: float  # 0.0 (conservative) to 1.0 (innovative)
    hierarchy_preference: float  # 0.0 (egalitarian) to 1.0 (hierarchical)


class CivilizationArchitect:
    """Procedural civilization generator with archetype support."""
    
    def __init__(self, core: CivilizationIntelligenceCore):
        """Initialize the civilization architect."""
        self.core = core
        self.blueprints: Dict[str, CivilizationBlueprint] = {}
        self.generation_history: List[Dict[str, Any]] = []
        
        # Initialize default blueprints
        self._initialize_default_blueprints()
        
        # Generation statistics
        self.generation_stats = {
            'total_generated': 0,
            'by_archetype': {archetype.name: 0 for archetype in list(CivilizationArchetype)},
            'by_size': {size.name: 0 for size in list(CivilizationSize)},
            'success_rate': 0.0
        }
    
    def _initialize_default_blueprints(self):
        """Initialize default civilization blueprints."""
        # Hierarchical Empire
        self.blueprints['hierarchical_empire'] = CivilizationBlueprint(
            name="Hierarchical Empire",
            archetype=CivilizationArchetype.HIERARCHICAL,
            size_category=CivilizationSize.LARGE,
            population_range=(1000, 5000),
            preferred_governance=[GovernanceModel.AUTOCRATIC, GovernanceModel.MERITOCRATIC],
            decision_efficiency_range=(0.6, 0.9),
            agent_type_distribution={
                AgentType.LEADER: 0.05,
                AgentType.COORDINATOR: 0.15,
                AgentType.WORKER: 0.60,
                AgentType.SPECIALIST: 0.15,
                AgentType.INNOVATOR: 0.05
            },
            capability_ranges={
                'leadership': (0.7, 0.95),
                'cooperation': (0.5, 0.8),
                'technical_skill': (0.4, 0.8)
            },
            resource_abundance={
                ResourceType.ENERGY: 0.8,
                ResourceType.MATERIALS: 0.7,
                ResourceType.INFORMATION: 0.6,
                ResourceType.COORDINATION: 0.9
            },
            resource_efficiency={
                ResourceType.ENERGY: 0.7,
                ResourceType.MATERIALS: 0.8,
                ResourceType.INFORMATION: 0.6,
                ResourceType.COORDINATION: 0.9
            },
            climate_preference=0.7,
            resource_richness=0.8,
            cooperation_tendency=0.6,
            innovation_focus=0.4,
            hierarchy_preference=0.9
        )
        
        # Decentralized Collective
        self.blueprints['decentralized_collective'] = CivilizationBlueprint(
            name="Decentralized Collective",
            archetype=CivilizationArchetype.DECENTRALIZED,
            size_category=CivilizationSize.MEDIUM,
            population_range=(100, 1000),
            preferred_governance=[GovernanceModel.DEMOCRATIC, GovernanceModel.CONSENSUS],
            decision_efficiency_range=(0.4, 0.7),
            agent_type_distribution={
                AgentType.COORDINATOR: 0.30,
                AgentType.WORKER: 0.40,
                AgentType.SPECIALIST: 0.20,
                AgentType.INNOVATOR: 0.10
            },
            capability_ranges={
                'cooperation': (0.8, 0.95),
                'creativity': (0.6, 0.9),
                'social_skill': (0.7, 0.9)
            },
            resource_abundance={
                ResourceType.ENERGY: 0.6,
                ResourceType.MATERIALS: 0.5,
                ResourceType.INFORMATION: 0.8,
                ResourceType.COORDINATION: 0.7
            },
            resource_efficiency={
                ResourceType.ENERGY: 0.8,
                ResourceType.MATERIALS: 0.7,
                ResourceType.INFORMATION: 0.9,
                ResourceType.COORDINATION: 0.8
            },
            climate_preference=0.5,
            resource_richness=0.6,
            cooperation_tendency=0.9,
            innovation_focus=0.7,
            hierarchy_preference=0.2
        )
        
        # Exploratory Federation
        self.blueprints['exploratory_federation'] = CivilizationBlueprint(
            name="Exploratory Federation",
            archetype=CivilizationArchetype.EXPLORATORY,
            size_category=CivilizationSize.MEDIUM,
            population_range=(200, 1500),
            preferred_governance=[GovernanceModel.DEMOCRATIC, GovernanceModel.TECHNOCRATIC],
            decision_efficiency_range=(0.5, 0.8),
            agent_type_distribution={
                AgentType.EXPLORER: 0.25,
                AgentType.INNOVATOR: 0.20,
                AgentType.SPECIALIST: 0.25,
                AgentType.WORKER: 0.30
            },
            capability_ranges={
                'creativity': (0.7, 0.95),
                'technical_skill': (0.6, 0.9),
                'intelligence': (0.6, 0.9)
            },
            resource_abundance={
                ResourceType.ENERGY: 0.7,
                ResourceType.MATERIALS: 0.6,
                ResourceType.INFORMATION: 0.9,
                ResourceType.KNOWLEDGE: 0.8
            },
            resource_efficiency={
                ResourceType.ENERGY: 0.6,
                ResourceType.MATERIALS: 0.7,
                ResourceType.INFORMATION: 0.9,
                ResourceType.KNOWLEDGE: 0.8
            },
            climate_preference=0.4,
            resource_richness=0.7,
            cooperation_tendency=0.7,
            innovation_focus=0.9,
            hierarchy_preference=0.3
        )
        
        # Competitive Alliance
        self.blueprints['competitive_alliance'] = CivilizationBlueprint(
            name="Competitive Alliance",
            archetype=CivilizationArchetype.COMPETITIVE,
            size_category=CivilizationSize.LARGE,
            population_range=(500, 3000),
            preferred_governance=[GovernanceModel.MERITOCRATIC, GovernanceModel.TECHNOCRATIC],
            decision_efficiency_range=(0.7, 0.9),
            agent_type_distribution={
                AgentType.LEADER: 0.08,
                AgentType.COORDINATOR: 0.12,
                AgentType.WORKER: 0.50,
                AgentType.SPECIALIST: 0.20,
                AgentType.INNOVATOR: 0.10
            },
            capability_ranges={
                'leadership': (0.6, 0.9),
                'technical_skill': (0.7, 0.95),
                'intelligence': (0.6, 0.9)
            },
            resource_abundance={
                ResourceType.ENERGY: 0.8,
                ResourceType.MATERIALS: 0.8,
                ResourceType.INFORMATION: 0.7,
                ResourceType.INNOVATION: 0.8
            },
            resource_efficiency={
                ResourceType.ENERGY: 0.8,
                ResourceType.MATERIALS: 0.8,
                ResourceType.INFORMATION: 0.7,
                ResourceType.INNOVATION: 0.8
            },
            climate_preference=0.6,
            resource_richness=0.8,
            cooperation_tendency=0.4,
            innovation_focus=0.8,
            hierarchy_preference=0.6
        )
        
        # Cooperative Union
        self.blueprints['cooperative_union'] = CivilizationBlueprint(
            name="Cooperative Union",
            archetype=CivilizationArchetype.COOPERATIVE,
            size_category=CivilizationSize.MEDIUM,
            population_range=(300, 2000),
            preferred_governance=[GovernanceModel.DEMOCRATIC, GovernanceModel.CONSENSUS],
            decision_efficiency_range=(0.5, 0.8),
            agent_type_distribution={
                AgentType.COORDINATOR: 0.25,
                AgentType.WORKER: 0.45,
                AgentType.SPECIALIST: 0.20,
                AgentType.INNOVATOR: 0.10
            },
            capability_ranges={
                'cooperation': (0.8, 0.95),
                'social_skill': (0.7, 0.9),
                'creativity': (0.5, 0.8)
            },
            resource_abundance={
                ResourceType.ENERGY: 0.6,
                ResourceType.MATERIALS: 0.6,
                ResourceType.INFORMATION: 0.8,
                ResourceType.COORDINATION: 0.9
            },
            resource_efficiency={
                ResourceType.ENERGY: 0.8,
                ResourceType.MATERIALS: 0.8,
                ResourceType.INFORMATION: 0.9,
                ResourceType.COORDINATION: 0.9
            },
            climate_preference=0.7,
            resource_richness=0.6,
            cooperation_tendency=0.95,
            innovation_focus=0.6,
            hierarchy_preference=0.2
        )
    
    def generate_civilization(self, name: str, 
                            archetype: Optional[CivilizationArchetype] = None,
                            blueprint_name: Optional[str] = None,
                            population_size: Optional[int] = None,
                            deterministic_seed: Optional[int] = None) -> Optional[Civilization]:
        """Generate a new civilization with specified parameters."""
        try:
            # Set deterministic seed
            if deterministic_seed is not None:
                random.seed(deterministic_seed)
                np.random.seed(deterministic_seed)
            
            # Select blueprint
            if blueprint_name and blueprint_name in self.blueprints:
                blueprint = self.blueprints[blueprint_name]
            elif archetype:
                blueprint = self._select_blueprint_by_archetype(archetype)
            else:
                blueprint = self._select_random_blueprint()
            
            # Determine population size
            if population_size is None:
                population_size = self._determine_population_size(blueprint)
            
            # Generate civilization using blueprint
            civilization = self._generate_from_blueprint(name, blueprint, population_size)
            
            # Register with core
            self.core.civilizations[civilization.id] = civilization
            
            # Update statistics
            self._update_generation_stats(civilization, blueprint)
            
            # Record generation
            self._record_generation(civilization, blueprint, deterministic_seed)
            
            # Pretty print the generated civilization
            print_civilization(civilization, f"Generated Civilization: {name}")
            
            return civilization
            
        except Exception as e:
            print(f"Error generating civilization: {e}")
            return None
    
    def _select_blueprint_by_archetype(self, archetype: CivilizationArchetype) -> CivilizationBlueprint:
        """Select blueprint by archetype."""
        matching_blueprints = [bp for bp in self.blueprints.values() if bp.archetype == archetype]
        if matching_blueprints:
            return random.choice(matching_blueprints)
        else:
            return random.choice(list(self.blueprints.values()))
    
    def _select_random_blueprint(self) -> CivilizationBlueprint:
        """Select a random blueprint."""
        return random.choice(list(self.blueprints.values()))
    
    def _determine_population_size(self, blueprint: CivilizationBlueprint) -> int:
        """Determine population size based on blueprint."""
        min_pop, max_pop = blueprint.population_range
        return random.randint(min_pop, max_pop)
    
    def _generate_from_blueprint(self, name: str, blueprint: CivilizationBlueprint, 
                                population_size: int) -> Civilization:
        """Generate civilization from blueprint."""
        civ_id = str(uuid.uuid4())
        
        # Generate agent clusters based on blueprint
        agent_clusters = self._generate_agent_clusters_from_blueprint(blueprint, population_size)
        
        # Select governance model
        governance_model = random.choice(blueprint.preferred_governance)
        
        # Generate decision efficiency
        decision_efficiency = random.uniform(*blueprint.decision_efficiency_range)
        
        # Generate resource pools
        resource_pools = self._generate_resource_pools_from_blueprint(blueprint)
        resource_efficiency = self._generate_resource_efficiency_from_blueprint(blueprint)
        
        # Generate environment
        environment = self._generate_environment_from_blueprint(blueprint)
        
        # Create civilization
        civilization = Civilization(
            id=civ_id,
            name=name,
            archetype=blueprint.archetype,
            agent_clusters=agent_clusters,
            total_population=population_size,
            governance_model=governance_model,
            decision_making_efficiency=decision_efficiency,
            resource_pools=resource_pools,
            resource_efficiency=resource_efficiency,
            environment=environment,
            innovation_capacity=blueprint.innovation_focus,
            adaptability=1.0 - blueprint.hierarchy_preference,
            cooperation_level=blueprint.cooperation_tendency
        )
        
        # Initialize communication network
        self._initialize_communication_network(civilization, blueprint)
        
        # Set initial stability and complexity
        civilization.stability = random.uniform(0.4, 0.8)
        civilization.complexity = random.uniform(0.3, 0.7)
        civilization.resilience = random.uniform(0.4, 0.8)
        
        return civilization
    
    def _generate_agent_clusters_from_blueprint(self, blueprint: CivilizationBlueprint, 
                                               population_size: int) -> Dict[str, AgentCluster]:
        """Generate agent clusters from blueprint."""
        clusters = {}
        
        for agent_type, proportion in blueprint.agent_type_distribution.items():
            cluster_size = max(1, int(population_size * proportion))
            
            # Generate capabilities based on blueprint ranges
            capabilities = {}
            for capability, (min_val, max_val) in blueprint.capability_ranges.items():
                capabilities[capability] = random.uniform(min_val, max_val)
            
            # Generate knowledge domains
            knowledge_domains = self._generate_knowledge_domains_for_agent_type(agent_type)
            
            # Generate resource requirements
            resource_requirements = self._generate_resource_requirements_for_agent_type(agent_type)
            
            cluster = AgentCluster(
                id=str(uuid.uuid4()),
                cluster_type=agent_type,
                size=cluster_size,
                capabilities=capabilities,
                knowledge_domains=knowledge_domains,
                resource_requirements=resource_requirements,
                productivity=random.uniform(0.4, 0.9),
                innovation_rate=random.uniform(0.1, 0.8),
                cooperation_level=blueprint.cooperation_tendency + random.uniform(-0.2, 0.2)
            )
            
            clusters[cluster.id] = cluster
        
        return clusters
    
    def _generate_knowledge_domains_for_agent_type(self, agent_type: AgentType) -> Set[str]:
        """Generate knowledge domains for agent type."""
        all_domains = {
            'governance', 'technology', 'economics', 'culture', 'environment',
            'communication', 'resource_management', 'innovation', 'coordination'
        }
        
        # Assign domains based on agent type
        if agent_type == AgentType.LEADER:
            domains = {'governance', 'coordination', 'communication'}
        elif agent_type == AgentType.INNOVATOR:
            domains = {'technology', 'innovation', 'research'}
        elif agent_type == AgentType.COORDINATOR:
            domains = {'coordination', 'communication', 'governance'}
        elif agent_type == AgentType.SPECIALIST:
            domains = {'technology', 'economics', 'resource_management'}
        else:  # WORKER, EXPLORER
            domains = {'economics', 'resource_management', 'environment'}
        
        # Add 1-2 additional random domains
        additional_domains = random.sample(list(all_domains - domains), random.randint(1, 2))
        domains.update(additional_domains)
        
        return domains
    
    def _generate_resource_requirements_for_agent_type(self, agent_type: AgentType) -> Dict[str, float]:
        """Generate resource requirements for agent type."""
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
        elif agent_type == AgentType.WORKER:
            base_requirements['energy'] = random.uniform(0.3, 0.7)
        
        return base_requirements
    
    def _generate_resource_pools_from_blueprint(self, blueprint: CivilizationBlueprint) -> Dict[str, float]:
        """Generate resource pools from blueprint."""
        pools = {}
        
        for resource_type, abundance in blueprint.resource_abundance.items():
            # Scale abundance to actual resource amounts
            base_amount = abundance * 100  # Scale factor
            variance = base_amount * 0.2  # 20% variance
            pools[resource_type.name.lower()] = max(10, base_amount + random.uniform(-variance, variance))
        
        return pools
    
    def _generate_resource_efficiency_from_blueprint(self, blueprint: CivilizationBlueprint) -> Dict[str, float]:
        """Generate resource efficiency from blueprint."""
        efficiency = {}
        
        for resource_type, eff in blueprint.resource_efficiency.items():
            variance = eff * 0.1  # 10% variance
            efficiency[resource_type.name.lower()] = max(0.1, min(1.0, eff + random.uniform(-variance, variance)))
        
        return efficiency
    
    def _generate_environment_from_blueprint(self, blueprint: CivilizationBlueprint) -> EnvironmentState:
        """Generate environment from blueprint."""
        environment = EnvironmentState(id=str(uuid.uuid4()))
        
        # Set climate stability based on preference
        climate_variance = (1.0 - blueprint.climate_preference) * 0.3
        environment.climate_stability = max(0.1, min(1.0, 
            blueprint.climate_preference + random.uniform(-climate_variance, climate_variance)))
        
        # Set resource abundance
        for resource_type, abundance in blueprint.resource_abundance.items():
            variance = abundance * 0.2
            environment.resource_abundance[resource_type.name.lower()] = max(0.1, 
                abundance + random.uniform(-variance, variance))
        
        # Set environmental factors
        environment.natural_hazards = self._generate_natural_hazards(blueprint)
        environment.technological_opportunities = self._generate_tech_opportunities(blueprint)
        
        return environment
    
    def _generate_natural_hazards(self, blueprint: CivilizationBlueprint) -> List[str]:
        """Generate natural hazards based on blueprint."""
        all_hazards = ['storms', 'droughts', 'floods', 'earthquakes', 'pandemics', 'resource_depletion']
        
        # More hazards for less climate-preferred civilizations
        num_hazards = int((1.0 - blueprint.climate_preference) * 3) + 1
        return random.sample(all_hazards, min(num_hazards, len(all_hazards)))
    
    def _generate_tech_opportunities(self, blueprint: CivilizationBlueprint) -> List[str]:
        """Generate technological opportunities based on blueprint."""
        all_opportunities = ['energy_breakthrough', 'materials_innovation', 'communication_advance', 
                           'automation_tech', 'biotech_advance', 'ai_development']
        
        # More opportunities for innovation-focused civilizations
        num_opportunities = int(blueprint.innovation_focus * 4) + 1
        return random.sample(all_opportunities, min(num_opportunities, len(all_opportunities)))
    
    def _initialize_communication_network(self, civilization: Civilization, blueprint: CivilizationBlueprint):
        """Initialize communication network based on blueprint."""
        # Create network based on hierarchy preference
        if blueprint.hierarchy_preference > 0.7:
            self._create_hierarchical_network(civilization)
        else:
            self._create_decentralized_network(civilization)
    
    def _create_hierarchical_network(self, civilization: Civilization):
        """Create hierarchical communication network."""
        # Implementation for hierarchical network
        # This would create a tree-like structure with clear leadership hierarchy
        pass
    
    def _create_decentralized_network(self, civilization: Civilization):
        """Create decentralized communication network."""
        # Implementation for decentralized network
        # This would create a more mesh-like structure with distributed connections
        pass
    
    def _update_generation_stats(self, civilization: Civilization, blueprint: CivilizationBlueprint):
        """Update generation statistics."""
        self.generation_stats['total_generated'] += 1
        self.generation_stats['by_archetype'][civilization.archetype.name] += 1
        self.generation_stats['by_size'][blueprint.size_category.name] += 1
    
    def _record_generation(self, civilization: Civilization, blueprint: CivilizationBlueprint, 
                          seed: Optional[int]):
        """Record generation details."""
        record = {
            'timestamp': datetime.now(),
            'civilization_id': civilization.id,
            'name': civilization.name,
            'archetype': civilization.archetype.name,
            'blueprint': blueprint.name,
            'population_size': civilization.total_population,
            'governance_model': civilization.governance_model.name,
            'seed': seed,
            'success': True
        }
        
        self.generation_history.append(record)
    
    def create_custom_blueprint(self, name: str, **kwargs) -> CivilizationBlueprint:
        """Create a custom civilization blueprint."""
        # This would allow for custom blueprint creation
        # Implementation would validate parameters and create blueprint
        pass
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get civilization generation statistics."""
        return self.generation_stats.copy()
    
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Get civilization generation history."""
        return self.generation_history.copy()
    
    def list_available_blueprints(self) -> List[str]:
        """List available civilization blueprints."""
        return list(self.blueprints.keys())
    
    def get_blueprint(self, blueprint_name: str) -> Optional[CivilizationBlueprint]:
        """Get blueprint by name."""
        return self.blueprints.get(blueprint_name)