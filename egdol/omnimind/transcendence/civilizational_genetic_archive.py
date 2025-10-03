"""
Civilizational Genetic Archive for OmniMind Civilization Intelligence Layer
Stores and manages high-performing civilization blueprints and governance models.
"""

import json
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from .core_structures import (
    Civilization, AgentCluster, EnvironmentState, TemporalState,
    CivilizationIntelligenceCore, GovernanceModel, AgentType, CivilizationArchetype
)
from .pattern_codification_engine import PatternBlueprint, BlueprintType, StrategicDoctrine


class ArchiveStatus(Enum):
    """Status of archived civilization blueprints."""
    ACTIVE = auto()
    DEPRECATED = auto()
    EXPERIMENTAL = auto()
    LEGACY = auto()
    QUARANTINED = auto()


class LineageType(Enum):
    """Types of civilization lineages."""
    GOVERNANCE_EVOLUTION = auto()
    TECHNOLOGICAL_PATHWAY = auto()
    CULTURAL_TRADITION = auto()
    STRATEGIC_DOCTRINE = auto()
    HYBRID_ARCHITECTURE = auto()


@dataclass
class CivilizationDNA:
    """Genetic representation of a civilization's core characteristics."""
    id: str
    name: str
    source_civilization_id: str
    lineage_type: LineageType
    
    # Core genetic markers
    governance_genome: Dict[str, Any]
    resource_genome: Dict[str, Any]
    communication_genome: Dict[str, Any]
    innovation_genome: Dict[str, Any]
    cultural_genome: Dict[str, Any]
    
    # Performance characteristics
    fitness_score: float
    adaptability_score: float
    stability_score: float
    innovation_potential: float
    cooperation_capacity: float
    
    # Archive metadata
    archive_timestamp: datetime
    last_accessed: datetime
    
    # Lineage information
    parent_lineages: List[str] = field(default_factory=list)
    child_lineages: List[str] = field(default_factory=list)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Archive status
    access_count: int = 0
    success_rate: float = 0.0
    status: ArchiveStatus = ArchiveStatus.ACTIVE
    
    # Compatibility markers
    compatible_archetypes: List[str] = field(default_factory=list)
    incompatible_archetypes: List[str] = field(default_factory=list)
    environmental_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageTree:
    """Tree structure representing civilization lineage evolution."""
    root_dna_id: str
    lineage_name: str
    lineage_type: LineageType
    
    # Tree structure
    nodes: Dict[str, CivilizationDNA] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    
    # Lineage statistics
    total_generations: int = 0
    successful_mutations: int = 0
    failed_mutations: int = 0
    average_fitness: float = 0.0
    
    # Meta-information
    creation_timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class CivilizationalGeneticArchive:
    """Archive for storing and managing civilization genetic information."""
    
    def __init__(self, core: CivilizationIntelligenceCore):
        """Initialize the genetic archive."""
        self.core = core
        
        # Archive storage
        self.dna_archive: Dict[str, CivilizationDNA] = {}
        self.lineage_trees: Dict[str, LineageTree] = {}
        self.blueprint_lineage_map: Dict[str, str] = {}  # blueprint_id -> lineage_id
        
        # Archive parameters
        self.archive_parameters = {
            'min_fitness_threshold': 0.6,
            'min_stability_threshold': 0.5,
            'max_archive_size': 10000,
            'lineage_depth_limit': 20,
            'mutation_rate': 0.1,
            'crossover_rate': 0.3,
            'archive_cleanup_interval': 30  # days
        }
        
        # Archive statistics
        self.archive_stats = {
            'total_dna_sequences': 0,
            'active_lineages': 0,
            'successful_mutations': 0,
            'failed_mutations': 0,
            'crossovers_performed': 0,
            'lineages_created': 0,
            'lineages_merged': 0,
            'archive_cleanups': 0
        }
        
        # Performance tracking
        self.performance_tracking = {
            'dna_retrieval_times': [],
            'lineage_search_times': [],
            'mutation_success_rates': [],
            'crossover_success_rates': []
        }
    
    def archive_civilization_dna(self, civilization: Civilization, 
                                performance_metrics: Dict[str, float]) -> Optional[CivilizationDNA]:
        """Archive a civilization's genetic information."""
        try:
            # Extract genetic markers from civilization
            governance_genome = self._extract_governance_genome(civilization)
            resource_genome = self._extract_resource_genome(civilization)
            communication_genome = self._extract_communication_genome(civilization)
            innovation_genome = self._extract_innovation_genome(civilization)
            cultural_genome = self._extract_cultural_genome(civilization)
            
            # Calculate fitness scores
            fitness_score = self._calculate_fitness_score(civilization, performance_metrics)
            adaptability_score = self._calculate_adaptability_score(civilization)
            stability_score = self._calculate_stability_score(civilization)
            innovation_potential = self._calculate_innovation_potential(civilization)
            cooperation_capacity = self._calculate_cooperation_capacity(civilization)
            
            # Create DNA sequence
            dna = CivilizationDNA(
                id=str(uuid.uuid4()),
                name=f"{civilization.name} DNA",
                source_civilization_id=civilization.id,
                lineage_type=self._determine_lineage_type(civilization),
                governance_genome=governance_genome,
                resource_genome=resource_genome,
                communication_genome=communication_genome,
                innovation_genome=innovation_genome,
                cultural_genome=cultural_genome,
                fitness_score=fitness_score,
                adaptability_score=adaptability_score,
                stability_score=stability_score,
                innovation_potential=innovation_potential,
                cooperation_capacity=cooperation_capacity,
                archive_timestamp=datetime.now(),
                last_accessed=datetime.now(),
                compatible_archetypes=[civilization.archetype.name],
                environmental_requirements={
                    'climate_stability': civilization.environment.climate_stability,
                    'natural_hazards': civilization.environment.natural_hazards,
                    'technological_opportunities': civilization.environment.technological_opportunities,
                    'competitive_pressure': civilization.environment.competitive_pressure
                }
            )
            
            # Archive DNA
            self.dna_archive[dna.id] = dna
            self.archive_stats['total_dna_sequences'] += 1
            
            # Create or update lineage tree
            self._update_lineage_tree(dna)
            
            return dna
            
        except Exception as e:
            print(f"Error archiving civilization DNA: {e}")
            return None
    
    def _extract_governance_genome(self, civilization: Civilization) -> Dict[str, Any]:
        """Extract governance genetic markers."""
        return {
            'governance_model': civilization.governance_model.name,
            'decision_efficiency': civilization.decision_making_efficiency,
            'hierarchy_levels': getattr(civilization.governance_model, 'hierarchy_levels', 1),
            'decision_making_process': getattr(civilization.governance_model, 'decision_making_process', 'unknown'),
            'authority_distribution': getattr(civilization.governance_model, 'authority_distribution', 'unknown'),
            'accountability_mechanisms': getattr(civilization.governance_model, 'accountability_mechanisms', 'unknown'),
            'conflict_resolution': getattr(civilization.governance_model, 'conflict_resolution', 'unknown'),
            'participation_level': getattr(civilization.governance_model, 'participation_level', 0.5)
        }
    
    def _extract_resource_genome(self, civilization: Civilization) -> Dict[str, Any]:
        """Extract resource genetic markers."""
        return {
            'resource_pools': civilization.resource_pools.copy(),
            'resource_efficiency': civilization.resource_efficiency.copy(),
            'resource_allocation_strategy': getattr(civilization, 'resource_allocation_strategy', 'balanced'),
            'resource_management_approach': getattr(civilization, 'resource_management_approach', 'centralized'),
            'resource_scarcity_handling': getattr(civilization, 'resource_scarcity_handling', 'conservation'),
            'resource_sharing_policies': getattr(civilization, 'resource_sharing_policies', 'limited'),
            'resource_innovation_rate': getattr(civilization, 'resource_innovation_rate', 0.5),
            'resource_sustainability': getattr(civilization, 'resource_sustainability', 0.5)
        }
    
    def _extract_communication_genome(self, civilization: Civilization) -> Dict[str, Any]:
        """Extract communication genetic markers."""
        return {
            'network_topology': getattr(civilization, 'communication_network_topology', 'mesh'),
            'communication_efficiency': getattr(civilization, 'communication_efficiency', 0.5),
            'information_flow_patterns': getattr(civilization, 'information_flow_patterns', 'hierarchical'),
            'knowledge_sharing_rate': getattr(civilization, 'knowledge_sharing_rate', 0.5),
            'communication_protocols': getattr(civilization, 'communication_protocols', 'standard'),
            'network_resilience': getattr(civilization, 'network_resilience', 0.5),
            'information_processing_capacity': getattr(civilization, 'information_processing_capacity', 0.5),
            'communication_barriers': getattr(civilization, 'communication_barriers', 'low')
        }
    
    def _extract_innovation_genome(self, civilization: Civilization) -> Dict[str, Any]:
        """Extract innovation genetic markers."""
        return {
            'innovation_capacity': civilization.innovation_capacity,
            'innovation_rate': getattr(civilization, 'innovation_rate', 0.5),
            'innovation_focus_areas': getattr(civilization, 'innovation_focus_areas', []),
            'innovation_collaboration': getattr(civilization, 'innovation_collaboration', 0.5),
            'innovation_funding': getattr(civilization, 'innovation_funding', 0.5),
            'innovation_risk_tolerance': getattr(civilization, 'innovation_risk_tolerance', 0.5),
            'innovation_impact_assessment': getattr(civilization, 'innovation_impact_assessment', 'basic'),
            'innovation_scaling': getattr(civilization, 'innovation_scaling', 0.5)
        }
    
    def _extract_cultural_genome(self, civilization: Civilization) -> Dict[str, Any]:
        """Extract cultural genetic markers."""
        return {
            'cooperation_level': civilization.cooperation_level,
            'cultural_values': getattr(civilization, 'cultural_values', []),
            'social_norms': getattr(civilization, 'social_norms', []),
            'cultural_diversity': getattr(civilization, 'cultural_diversity', 0.5),
            'cultural_adaptability': getattr(civilization, 'cultural_adaptability', 0.5),
            'cultural_transmission': getattr(civilization, 'cultural_transmission', 0.5),
            'cultural_innovation': getattr(civilization, 'cultural_innovation', 0.5),
            'cultural_stability': getattr(civilization, 'cultural_stability', 0.5)
        }
    
    def _calculate_fitness_score(self, civilization: Civilization, 
                               performance_metrics: Dict[str, float]) -> float:
        """Calculate fitness score for a civilization."""
        # Base fitness from performance metrics
        base_fitness = sum(performance_metrics.values()) / len(performance_metrics) if performance_metrics else 0.5
        
        # Adjust for civilization characteristics
        stability_factor = civilization.stability
        complexity_factor = civilization.complexity
        adaptability_factor = civilization.adaptability
        resilience_factor = civilization.resilience
        
        # Weighted fitness calculation
        fitness = (
            base_fitness * 0.4 +
            stability_factor * 0.2 +
            complexity_factor * 0.2 +
            adaptability_factor * 0.1 +
            resilience_factor * 0.1
        )
        
        return min(1.0, max(0.0, fitness))
    
    def _calculate_adaptability_score(self, civilization: Civilization) -> float:
        """Calculate adaptability score for a civilization."""
        return civilization.adaptability
    
    def _calculate_stability_score(self, civilization: Civilization) -> float:
        """Calculate stability score for a civilization."""
        return civilization.stability
    
    def _calculate_innovation_potential(self, civilization: Civilization) -> float:
        """Calculate innovation potential for a civilization."""
        return civilization.innovation_capacity
    
    def _calculate_cooperation_capacity(self, civilization: Civilization) -> float:
        """Calculate cooperation capacity for a civilization."""
        return civilization.cooperation_level
    
    def _determine_lineage_type(self, civilization: Civilization) -> LineageType:
        """Determine the lineage type for a civilization."""
        # Analyze civilization characteristics to determine lineage type
        if civilization.innovation_capacity > 0.7:
            return LineageType.TECHNOLOGICAL_PATHWAY
        elif civilization.cooperation_level > 0.7:
            return LineageType.CULTURAL_TRADITION
        elif getattr(civilization.governance_model, 'hierarchy_levels', 1) > 3:
            return LineageType.GOVERNANCE_EVOLUTION
        elif civilization.archetype in [CivilizationArchetype.HYBRID, CivilizationArchetype.DECENTRALIZED]:
            return LineageType.HYBRID_ARCHITECTURE
        else:
            return LineageType.STRATEGIC_DOCTRINE
    
    def _update_lineage_tree(self, dna: CivilizationDNA):
        """Update lineage tree with new DNA sequence."""
        lineage_id = f"{dna.lineage_type.name}_{dna.source_civilization_id}"
        
        if lineage_id not in self.lineage_trees:
            # Create new lineage tree
            tree = LineageTree(
                root_dna_id=dna.id,
                lineage_name=f"{dna.lineage_type.name} Lineage",
                lineage_type=dna.lineage_type
            )
            self.lineage_trees[lineage_id] = tree
            self.archive_stats['lineages_created'] += 1
        
        # Add DNA to lineage tree
        tree = self.lineage_trees[lineage_id]
        tree.nodes[dna.id] = dna
        
        # Update lineage statistics
        tree.total_generations += 1
        tree.average_fitness = (
            (tree.average_fitness * (tree.total_generations - 1) + dna.fitness_score) / 
            tree.total_generations
        )
        tree.last_updated = datetime.now()
    
    def create_dna_mutation(self, parent_dna_id: str, mutation_type: str = "random") -> Optional[CivilizationDNA]:
        """Create a mutated version of existing DNA."""
        parent_dna = self.dna_archive.get(parent_dna_id)
        if not parent_dna:
            return None
        
        try:
            # Create mutated DNA
            mutated_dna = CivilizationDNA(
                id=str(uuid.uuid4()),
                name=f"{parent_dna.name} Mutation",
                source_civilization_id=parent_dna.source_civilization_id,
                lineage_type=parent_dna.lineage_type,
                governance_genome=self._mutate_governance_genome(parent_dna.governance_genome, mutation_type),
                resource_genome=self._mutate_resource_genome(parent_dna.resource_genome, mutation_type),
                communication_genome=self._mutate_communication_genome(parent_dna.communication_genome, mutation_type),
                innovation_genome=self._mutate_innovation_genome(parent_dna.innovation_genome, mutation_type),
                cultural_genome=self._mutate_cultural_genome(parent_dna.cultural_genome, mutation_type),
                fitness_score=parent_dna.fitness_score,
                adaptability_score=parent_dna.adaptability_score,
                stability_score=parent_dna.stability_score,
                innovation_potential=parent_dna.innovation_potential,
                cooperation_capacity=parent_dna.cooperation_capacity,
                parent_lineages=[parent_dna_id],
                archive_timestamp=datetime.now(),
                last_accessed=datetime.now(),
                compatible_archetypes=parent_dna.compatible_archetypes.copy(),
                incompatible_archetypes=parent_dna.incompatible_archetypes.copy(),
                environmental_requirements=parent_dna.environmental_requirements.copy()
            )
            
            # Record mutation
            mutation_record = {
                'timestamp': datetime.now(),
                'mutation_type': mutation_type,
                'parent_dna_id': parent_dna_id,
                'changes': self._calculate_mutation_changes(parent_dna, mutated_dna)
            }
            mutated_dna.mutation_history.append(mutation_record)
            
            # Archive mutated DNA
            self.dna_archive[mutated_dna.id] = mutated_dna
            self.archive_stats['total_dna_sequences'] += 1
            self.archive_stats['successful_mutations'] += 1
            
            # Update lineage tree
            self._update_lineage_tree(mutated_dna)
            
            return mutated_dna
            
        except Exception as e:
            print(f"Error creating DNA mutation: {e}")
            self.archive_stats['failed_mutations'] += 1
            return None
    
    def _mutate_governance_genome(self, genome: Dict[str, Any], mutation_type: str) -> Dict[str, Any]:
        """Mutate governance genome."""
        mutated = genome.copy()
        
        if mutation_type == "random":
            # Random mutation of governance parameters
            if 'decision_efficiency' in mutated:
                mutated['decision_efficiency'] = max(0.0, min(1.0, 
                    mutated['decision_efficiency'] + random.uniform(-0.1, 0.1)))
            if 'hierarchy_levels' in mutated:
                mutated['hierarchy_levels'] = max(1, 
                    mutated['hierarchy_levels'] + random.randint(-1, 1))
        
        return mutated
    
    def _mutate_resource_genome(self, genome: Dict[str, Any], mutation_type: str) -> Dict[str, Any]:
        """Mutate resource genome."""
        mutated = genome.copy()
        
        if mutation_type == "random":
            # Random mutation of resource parameters
            for resource, efficiency in mutated.get('resource_efficiency', {}).items():
                mutated['resource_efficiency'][resource] = max(0.0, min(1.0, 
                    efficiency + random.uniform(-0.1, 0.1)))
        
        return mutated
    
    def _mutate_communication_genome(self, genome: Dict[str, Any], mutation_type: str) -> Dict[str, Any]:
        """Mutate communication genome."""
        mutated = genome.copy()
        
        if mutation_type == "random":
            # Random mutation of communication parameters
            if 'communication_efficiency' in mutated:
                mutated['communication_efficiency'] = max(0.0, min(1.0, 
                    mutated['communication_efficiency'] + random.uniform(-0.1, 0.1)))
        
        return mutated
    
    def _mutate_innovation_genome(self, genome: Dict[str, Any], mutation_type: str) -> Dict[str, Any]:
        """Mutate innovation genome."""
        mutated = genome.copy()
        
        if mutation_type == "random":
            # Random mutation of innovation parameters
            if 'innovation_capacity' in mutated:
                mutated['innovation_capacity'] = max(0.0, min(1.0, 
                    mutated['innovation_capacity'] + random.uniform(-0.1, 0.1)))
        
        return mutated
    
    def _mutate_cultural_genome(self, genome: Dict[str, Any], mutation_type: str) -> Dict[str, Any]:
        """Mutate cultural genome."""
        mutated = genome.copy()
        
        if mutation_type == "random":
            # Random mutation of cultural parameters
            if 'cooperation_level' in mutated:
                mutated['cooperation_level'] = max(0.0, min(1.0, 
                    mutated['cooperation_level'] + random.uniform(-0.1, 0.1)))
        
        return mutated
    
    def _calculate_mutation_changes(self, parent_dna: CivilizationDNA, 
                                  mutated_dna: CivilizationDNA) -> Dict[str, Any]:
        """Calculate changes between parent and mutated DNA."""
        changes = {}
        
        # Compare genomes
        for genome_name in ['governance_genome', 'resource_genome', 'communication_genome', 
                           'innovation_genome', 'cultural_genome']:
            parent_genome = getattr(parent_dna, genome_name)
            mutated_genome = getattr(mutated_dna, genome_name)
            
            genome_changes = {}
            for key in parent_genome:
                if key in mutated_genome:
                    if isinstance(parent_genome[key], (int, float)):
                        change = mutated_genome[key] - parent_genome[key]
                        if abs(change) > 0.001:  # Significant change
                            genome_changes[key] = change
            
            if genome_changes:
                changes[genome_name] = genome_changes
        
        return changes
    
    def create_dna_crossover(self, parent1_dna_id: str, parent2_dna_id: str) -> Optional[CivilizationDNA]:
        """Create a crossover between two DNA sequences."""
        parent1 = self.dna_archive.get(parent1_dna_id)
        parent2 = self.dna_archive.get(parent2_dna_id)
        
        if not parent1 or not parent2:
            return None
        
        try:
            # Create crossover DNA
            crossover_dna = CivilizationDNA(
                id=str(uuid.uuid4()),
                name=f"{parent1.name} x {parent2.name} Crossover",
                source_civilization_id=parent1.source_civilization_id,
                lineage_type=parent1.lineage_type,
                governance_genome=self._crossover_genomes(parent1.governance_genome, parent2.governance_genome),
                resource_genome=self._crossover_genomes(parent1.resource_genome, parent2.resource_genome),
                communication_genome=self._crossover_genomes(parent1.communication_genome, parent2.communication_genome),
                innovation_genome=self._crossover_genomes(parent1.innovation_genome, parent2.innovation_genome),
                cultural_genome=self._crossover_genomes(parent1.cultural_genome, parent2.cultural_genome),
                fitness_score=(parent1.fitness_score + parent2.fitness_score) / 2,
                adaptability_score=(parent1.adaptability_score + parent2.adaptability_score) / 2,
                stability_score=(parent1.stability_score + parent2.stability_score) / 2,
                innovation_potential=(parent1.innovation_potential + parent2.innovation_potential) / 2,
                cooperation_capacity=(parent1.cooperation_capacity + parent2.cooperation_capacity) / 2,
                parent_lineages=[parent1_dna_id, parent2_dna_id],
                archive_timestamp=datetime.now(),
                last_accessed=datetime.now(),
                compatible_archetypes=list(set(parent1.compatible_archetypes + parent2.compatible_archetypes)),
                incompatible_archetypes=list(set(parent1.incompatible_archetypes + parent2.incompatible_archetypes)),
                environmental_requirements=self._merge_environmental_requirements(
                    parent1.environmental_requirements, parent2.environmental_requirements
                )
            )
            
            # Archive crossover DNA
            self.dna_archive[crossover_dna.id] = crossover_dna
            self.archive_stats['total_dna_sequences'] += 1
            self.archive_stats['crossovers_performed'] += 1
            
            # Update lineage trees
            self._update_lineage_tree(crossover_dna)
            
            return crossover_dna
            
        except Exception as e:
            print(f"Error creating DNA crossover: {e}")
            return None
    
    def _crossover_genomes(self, genome1: Dict[str, Any], genome2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two genomes."""
        crossover = {}
        
        # Get all keys from both genomes
        all_keys = set(genome1.keys()) | set(genome2.keys())
        
        for key in all_keys:
            if key in genome1 and key in genome2:
                # Both parents have this trait - choose randomly or average
                if isinstance(genome1[key], (int, float)) and isinstance(genome2[key], (int, float)):
                    # Numeric traits - average with some randomness
                    crossover[key] = (genome1[key] + genome2[key]) / 2 + random.uniform(-0.05, 0.05)
                else:
                    # Categorical traits - choose randomly
                    crossover[key] = random.choice([genome1[key], genome2[key]])
            elif key in genome1:
                crossover[key] = genome1[key]
            else:
                crossover[key] = genome2[key]
        
        return crossover
    
    def _merge_environmental_requirements(self, env1: Dict[str, Any], env2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environmental requirements from two DNA sequences."""
        merged = {}
        
        # Get all environmental keys
        all_keys = set(env1.keys()) | set(env2.keys())
        
        for key in all_keys:
            if key in env1 and key in env2:
                if isinstance(env1[key], (int, float)) and isinstance(env2[key], (int, float)):
                    # Numeric requirements - take the more restrictive
                    merged[key] = min(env1[key], env2[key])
                else:
                    # Categorical requirements - choose randomly
                    merged[key] = random.choice([env1[key], env2[key]])
            elif key in env1:
                merged[key] = env1[key]
            else:
                merged[key] = env2[key]
        
        return merged
    
    def search_dna_by_fitness(self, min_fitness: float = 0.7) -> List[CivilizationDNA]:
        """Search for DNA sequences by fitness score."""
        return [
            dna for dna in self.dna_archive.values() 
            if dna.fitness_score >= min_fitness and dna.status == ArchiveStatus.ACTIVE
        ]
    
    def search_dna_by_lineage(self, lineage_type: LineageType) -> List[CivilizationDNA]:
        """Search for DNA sequences by lineage type."""
        return [
            dna for dna in self.dna_archive.values() 
            if dna.lineage_type == lineage_type and dna.status == ArchiveStatus.ACTIVE
        ]
    
    def search_dna_by_compatibility(self, archetype: str, 
                                  environmental_conditions: Dict[str, Any]) -> List[CivilizationDNA]:
        """Search for DNA sequences by compatibility."""
        compatible_dna = []
        
        for dna in self.dna_archive.values():
            if dna.status != ArchiveStatus.ACTIVE:
                continue
            
            # Check archetype compatibility
            if archetype in dna.incompatible_archetypes:
                continue
            
            # Check environmental compatibility
            env_compatibility = self._calculate_environmental_compatibility(
                dna.environmental_requirements, environmental_conditions
            )
            
            if env_compatibility > 0.5:  # Minimum compatibility threshold
                compatible_dna.append(dna)
        
        return compatible_dna
    
    def _calculate_environmental_compatibility(self, requirements: Dict[str, Any], 
                                             conditions: Dict[str, Any]) -> float:
        """Calculate environmental compatibility between requirements and conditions."""
        if not requirements or not conditions:
            return 0.5  # Neutral compatibility
        
        common_keys = set(requirements.keys()) & set(conditions.keys())
        if not common_keys:
            return 0.5
        
        compatibility_scores = []
        for key in common_keys:
            req_val = requirements[key]
            cond_val = conditions[key]
            
            if isinstance(req_val, (int, float)) and isinstance(cond_val, (int, float)):
                # Numeric compatibility
                diff = abs(req_val - cond_val)
                max_val = max(abs(req_val), abs(cond_val), 1)
                compatibility = max(0, 1 - diff / max_val)
                compatibility_scores.append(compatibility)
            else:
                # Categorical compatibility
                compatibility = 1.0 if req_val == cond_val else 0.0
                compatibility_scores.append(compatibility)
        
        return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.5
    
    def get_lineage_tree(self, lineage_id: str) -> Optional[LineageTree]:
        """Get a lineage tree by ID."""
        return self.lineage_trees.get(lineage_id)
    
    def get_dna_lineage(self, dna_id: str) -> List[CivilizationDNA]:
        """Get the lineage of a DNA sequence."""
        dna = self.dna_archive.get(dna_id)
        if not dna:
            return []
        
        lineage = [dna]
        
        # Add parent lineages
        for parent_id in dna.parent_lineages:
            parent_lineage = self.get_dna_lineage(parent_id)
            lineage.extend(parent_lineage)
        
        return lineage
    
    def cleanup_archive(self, days_old: int = 30) -> int:
        """Clean up old and low-performing DNA sequences."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0
        
        # Find DNA sequences to clean up
        to_cleanup = []
        for dna_id, dna in self.dna_archive.items():
            if (dna.last_accessed < cutoff_date and 
                dna.fitness_score < self.archive_parameters['min_fitness_threshold'] and
                dna.status in [ArchiveStatus.DEPRECATED, ArchiveStatus.LEGACY]):
                to_cleanup.append(dna_id)
        
        # Remove DNA sequences
        for dna_id in to_cleanup:
            del self.dna_archive[dna_id]
            cleaned_count += 1
        
        # Update statistics
        self.archive_stats['archive_cleanups'] += 1
        
        return cleaned_count
    
    def get_archive_statistics(self) -> Dict[str, Any]:
        """Get archive statistics."""
        return {
            'total_dna_sequences': len(self.dna_archive),
            'active_lineages': len(self.lineage_trees),
            'archive_stats': self.archive_stats.copy(),
            'lineage_distribution': {
                lineage_type.name: len(self.search_dna_by_lineage(lineage_type))
                for lineage_type in list(LineageType)
            },
            'fitness_distribution': {
                'high_fitness': len(self.search_dna_by_fitness(0.8)),
                'medium_fitness': len(self.search_dna_by_fitness(0.6)) - len(self.search_dna_by_fitness(0.8)),
                'low_fitness': len(self.search_dna_by_fitness(0.4)) - len(self.search_dna_by_fitness(0.6))
            },
            'average_fitness': sum(dna.fitness_score for dna in self.dna_archive.values()) / max(len(self.dna_archive), 1),
            'archive_size_mb': self._calculate_archive_size()
        }
    
    def _calculate_archive_size(self) -> float:
        """Calculate approximate archive size in MB."""
        # Rough estimation based on DNA sequences
        dna_size = len(self.dna_archive) * 0.001  # 1KB per DNA sequence
        lineage_size = len(self.lineage_trees) * 0.0005  # 0.5KB per lineage tree
        return (dna_size + lineage_size) / 1024  # Convert to MB
