"""
Pattern Codification Engine for OmniMind Civilization Intelligence Layer
Translates detected macro-patterns into formalized civilization blueprints and strategic doctrines.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from .core_structures import (
    Civilization, AgentCluster, EnvironmentState, TemporalState,
    CivilizationIntelligenceCore, GovernanceModel, AgentType, CivilizationArchetype
)
from .macro_pattern_detector import MacroPatternDetector, PatternType, PatternSignificance


class BlueprintType(Enum):
    """Types of civilization blueprints."""
    GOVERNANCE_MODEL = auto()
    RESOURCE_STRATEGY = auto()
    COMMUNICATION_PROTOCOL = auto()
    INNOVATION_PATHWAY = auto()
    CULTURAL_MEME = auto()
    STRATEGIC_DOCTRINE = auto()


class CodificationStatus(Enum):
    """Status of pattern codification."""
    DETECTED = auto()
    ANALYZING = auto()
    CODIFIED = auto()
    VALIDATED = auto()
    ARCHIVED = auto()
    DEPRECATED = auto()


@dataclass
class PatternBlueprint:
    """Formalized blueprint derived from detected patterns."""
    id: str
    name: str
    blueprint_type: BlueprintType
    source_pattern_id: str
    source_civilization_id: str
    
    # Pattern characteristics
    pattern_description: str
    emergence_conditions: Dict[str, Any]
    stability_requirements: Dict[str, float]
    scalability_factors: Dict[str, float]
    
    # Blueprint specifications
    implementation_requirements: Dict[str, Any]
    performance_metrics: Dict[str, float]
    compatibility_constraints: List[str]
    
    # Meta-information
    codification_timestamp: datetime
    validation_score: float
    reproducibility_score: float
    generalizability_score: float
    
    # Usage tracking
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    
    # Relationships
    derived_from: List[str] = field(default_factory=list)  # Other blueprint IDs
    influences: List[str] = field(default_factory=list)  # Blueprint IDs this influences
    conflicts_with: List[str] = field(default_factory=list)  # Conflicting blueprint IDs


@dataclass
class StrategicDoctrine:
    """Strategic doctrine derived from civilization patterns."""
    id: str
    name: str
    doctrine_type: str
    source_blueprints: List[str]  # Blueprint IDs
    
    # Doctrine specifications
    core_principles: List[str]
    implementation_strategy: Dict[str, Any]
    success_conditions: Dict[str, Any]
    failure_conditions: Dict[str, Any]
    
    # Performance characteristics
    effectiveness_score: float
    adaptability_score: float
    sustainability_score: float
    
    # Meta-information
    creation_timestamp: datetime
    last_updated: datetime
    usage_count: int = 0
    success_rate: float = 0.0


class PatternCodificationEngine:
    """Engine for codifying detected patterns into reusable blueprints."""
    
    def __init__(self, core: CivilizationIntelligenceCore, pattern_detector: MacroPatternDetector):
        """Initialize the pattern codification engine."""
        self.core = core
        self.pattern_detector = pattern_detector
        
        # Blueprint storage
        self.blueprints: Dict[str, PatternBlueprint] = {}
        self.doctrines: Dict[str, StrategicDoctrine] = {}
        
        # Codification parameters
        self.codification_parameters = {
            'min_pattern_significance': 0.7,
            'min_stability_threshold': 0.6,
            'min_reproducibility_threshold': 0.5,
            'max_blueprint_complexity': 0.8,
            'codification_timeout': 30.0  # seconds
        }
        
        # Pattern analysis cache
        self.pattern_analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Codification statistics
        self.codification_stats = {
            'total_patterns_analyzed': 0,
            'successful_codifications': 0,
            'failed_codifications': 0,
            'blueprint_usage_count': 0,
            'doctrine_creation_count': 0
        }
    
    def codify_detected_patterns(self, civilization_id: str) -> List[str]:
        """Codify detected patterns for a civilization into blueprints."""
        civilization = self.core.get_civilization(civilization_id)
        if not civilization:
            return []
        
        # Get detected patterns for this civilization
        detected_patterns = self.pattern_detector.detected_patterns.get(civilization_id, {})
        if not detected_patterns:
            return []
        
        codified_blueprints = []
        
        for pattern_id, pattern_data in detected_patterns.items():
            try:
                # Analyze pattern for codification potential
                analysis = self._analyze_pattern_for_codification(pattern_data, civilization)
                
                if analysis['codification_potential'] > self.codification_parameters['min_pattern_significance']:
                    # Create blueprint from pattern
                    blueprint = self._create_blueprint_from_pattern(
                        pattern_id, pattern_data, civilization, analysis
                    )
                    
                    if blueprint:
                        self.blueprints[blueprint.id] = blueprint
                        codified_blueprints.append(blueprint.id)
                        self.codification_stats['successful_codifications'] += 1
                    else:
                        self.codification_stats['failed_codifications'] += 1
                else:
                    self.codification_stats['failed_codifications'] += 1
                    
            except Exception as e:
                print(f"Error codifying pattern {pattern_id}: {e}")
                self.codification_stats['failed_codifications'] += 1
        
        self.codification_stats['total_patterns_analyzed'] += len(detected_patterns)
        return codified_blueprints
    
    def _analyze_pattern_for_codification(self, pattern_data: Dict[str, Any], 
                                        civilization: Civilization) -> Dict[str, Any]:
        """Analyze a pattern for codification potential."""
        pattern_id = pattern_data.get('id', 'unknown')
        
        # Check cache first
        if pattern_id in self.pattern_analysis_cache:
            return self.pattern_analysis_cache[pattern_id]
        
        analysis = {
            'codification_potential': 0.0,
            'stability_score': 0.0,
            'reproducibility_score': 0.0,
            'generalizability_score': 0.0,
            'complexity_score': 0.0,
            'implementation_difficulty': 0.0
        }
        
        # Analyze pattern characteristics
        pattern_type = pattern_data.get('type', 'unknown')
        significance = pattern_data.get('significance', 0.0)
        stability = pattern_data.get('stability', 0.0)
        
        # Calculate codification potential
        analysis['codification_potential'] = (
            significance * 0.4 + 
            stability * 0.3 + 
            pattern_data.get('novelty', 0.0) * 0.3
        )
        
        # Calculate stability score
        analysis['stability_score'] = stability
        
        # Calculate reproducibility score
        reproducibility_factors = [
            pattern_data.get('frequency', 0.0),
            pattern_data.get('consistency', 0.0),
            pattern_data.get('predictability', 0.0)
        ]
        analysis['reproducibility_score'] = sum(reproducibility_factors) / len(reproducibility_factors)
        
        # Calculate generalizability score
        generalizability_factors = [
            pattern_data.get('scalability', 0.0),
            pattern_data.get('adaptability', 0.0),
            pattern_data.get('transferability', 0.0)
        ]
        analysis['generalizability_score'] = sum(generalizability_factors) / len(generalizability_factors)
        
        # Calculate complexity score
        complexity_factors = [
            len(pattern_data.get('components', [])),
            pattern_data.get('interaction_complexity', 0.0),
            pattern_data.get('dependency_count', 0)
        ]
        analysis['complexity_score'] = min(1.0, sum(complexity_factors) / 10.0)
        
        # Calculate implementation difficulty
        analysis['implementation_difficulty'] = (
            analysis['complexity_score'] * 0.5 + 
            (1.0 - analysis['reproducibility_score']) * 0.3 +
            (1.0 - analysis['generalizability_score']) * 0.2
        )
        
        # Cache analysis
        self.pattern_analysis_cache[pattern_id] = analysis
        
        return analysis
    
    def _create_blueprint_from_pattern(self, pattern_id: str, pattern_data: Dict[str, Any],
                                     civilization: Civilization, analysis: Dict[str, Any]) -> Optional[PatternBlueprint]:
        """Create a blueprint from a detected pattern."""
        try:
            # Determine blueprint type based on pattern characteristics
            blueprint_type = self._determine_blueprint_type(pattern_data, civilization)
            
            # Generate blueprint specifications
            implementation_requirements = self._generate_implementation_requirements(
                pattern_data, civilization, blueprint_type
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                pattern_data, civilization, analysis
            )
            
            # Generate compatibility constraints
            compatibility_constraints = self._generate_compatibility_constraints(
                pattern_data, civilization, blueprint_type
            )
            
            # Create blueprint
            blueprint = PatternBlueprint(
                id=str(uuid.uuid4()),
                name=f"{pattern_data.get('name', 'Unnamed Pattern')} Blueprint",
                blueprint_type=blueprint_type,
                source_pattern_id=pattern_id,
                source_civilization_id=civilization.id,
                pattern_description=pattern_data.get('description', 'No description available'),
                emergence_conditions=pattern_data.get('conditions', {}),
                stability_requirements=pattern_data.get('stability_requirements', {}),
                scalability_factors=pattern_data.get('scalability_factors', {}),
                implementation_requirements=implementation_requirements,
                performance_metrics=performance_metrics,
                compatibility_constraints=compatibility_constraints,
                codification_timestamp=datetime.now(),
                validation_score=analysis['stability_score'],
                reproducibility_score=analysis['reproducibility_score'],
                generalizability_score=analysis['generalizability_score']
            )
            
            return blueprint
            
        except Exception as e:
            print(f"Error creating blueprint from pattern {pattern_id}: {e}")
            return None
    
    def _determine_blueprint_type(self, pattern_data: Dict[str, Any], 
                                civilization: Civilization) -> BlueprintType:
        """Determine the appropriate blueprint type for a pattern."""
        pattern_type = pattern_data.get('type', 'unknown')
        pattern_components = pattern_data.get('components', [])
        
        # Analyze pattern components to determine type
        if 'governance' in pattern_components or 'decision_making' in pattern_components:
            return BlueprintType.GOVERNANCE_MODEL
        elif 'resource' in pattern_components or 'allocation' in pattern_components:
            return BlueprintType.RESOURCE_STRATEGY
        elif 'communication' in pattern_components or 'network' in pattern_components:
            return BlueprintType.COMMUNICATION_PROTOCOL
        elif 'innovation' in pattern_components or 'technology' in pattern_components:
            return BlueprintType.INNOVATION_PATHWAY
        elif 'culture' in pattern_components or 'meme' in pattern_components:
            return BlueprintType.CULTURAL_MEME
        else:
            return BlueprintType.STRATEGIC_DOCTRINE
    
    def _generate_implementation_requirements(self, pattern_data: Dict[str, Any],
                                            civilization: Civilization, 
                                            blueprint_type: BlueprintType) -> Dict[str, Any]:
        """Generate implementation requirements for a blueprint."""
        requirements = {
            'minimum_population': civilization.total_population * 0.1,
            'required_resources': list(civilization.resource_pools.keys()),
            'governance_compatibility': [civilization.governance_model.name],
            'environmental_conditions': civilization.environment.conditions.copy(),
            'temporal_requirements': {
                'minimum_duration': pattern_data.get('duration', 10),
                'stability_period': pattern_data.get('stability_period', 5)
            }
        }
        
        # Add type-specific requirements
        if blueprint_type == BlueprintType.GOVERNANCE_MODEL:
            requirements['decision_making_efficiency'] = civilization.decision_making_efficiency
            requirements['hierarchy_levels'] = pattern_data.get('hierarchy_levels', 1)
        elif blueprint_type == BlueprintType.RESOURCE_STRATEGY:
            requirements['resource_efficiency'] = civilization.resource_efficiency.copy()
            requirements['allocation_strategy'] = pattern_data.get('allocation_strategy', 'balanced')
        elif blueprint_type == BlueprintType.COMMUNICATION_PROTOCOL:
            requirements['network_topology'] = pattern_data.get('network_topology', 'mesh')
            requirements['communication_efficiency'] = pattern_data.get('communication_efficiency', 0.5)
        
        return requirements
    
    def _calculate_performance_metrics(self, pattern_data: Dict[str, Any],
                                    civilization: Civilization, 
                                    analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for a blueprint."""
        metrics = {
            'efficiency': pattern_data.get('efficiency', 0.5),
            'stability': analysis['stability_score'],
            'scalability': analysis['generalizability_score'],
            'adaptability': pattern_data.get('adaptability', 0.5),
            'innovation_potential': pattern_data.get('innovation_potential', 0.5),
            'resource_efficiency': pattern_data.get('resource_efficiency', 0.5),
            'cooperation_level': civilization.cooperation_level,
            'complexity': analysis['complexity_score']
        }
        
        return metrics
    
    def _generate_compatibility_constraints(self, pattern_data: Dict[str, Any],
                                          civilization: Civilization,
                                          blueprint_type: BlueprintType) -> List[str]:
        """Generate compatibility constraints for a blueprint."""
        constraints = []
        
        # Add civilization-specific constraints
        constraints.append(f"Compatible with {civilization.archetype.name} archetype")
        constraints.append(f"Requires minimum population of {civilization.total_population * 0.1}")
        
        # Add type-specific constraints
        if blueprint_type == BlueprintType.GOVERNANCE_MODEL:
            constraints.append(f"Compatible with {civilization.governance_model.name} governance")
        elif blueprint_type == BlueprintType.RESOURCE_STRATEGY:
            constraints.append("Requires resource management capabilities")
        elif blueprint_type == BlueprintType.COMMUNICATION_PROTOCOL:
            constraints.append("Requires communication network infrastructure")
        
        # Add pattern-specific constraints
        if pattern_data.get('requires_hierarchy', False):
            constraints.append("Requires hierarchical structure")
        if pattern_data.get('requires_cooperation', False):
            constraints.append("Requires high cooperation level")
        
        return constraints
    
    def create_strategic_doctrine(self, blueprint_ids: List[str], 
                                doctrine_name: str) -> Optional[StrategicDoctrine]:
        """Create a strategic doctrine from multiple blueprints."""
        if not blueprint_ids:
            return None
        
        # Validate blueprints exist
        valid_blueprints = [bp_id for bp_id in blueprint_ids if bp_id in self.blueprints]
        if not valid_blueprints:
            return None
        
        # Analyze blueprint compatibility
        compatibility_score = self._analyze_blueprint_compatibility(valid_blueprints)
        if compatibility_score < 0.5:
            return None
        
        # Extract core principles from blueprints
        core_principles = self._extract_core_principles(valid_blueprints)
        
        # Generate implementation strategy
        implementation_strategy = self._generate_implementation_strategy(valid_blueprints)
        
        # Calculate performance characteristics
        performance_characteristics = self._calculate_doctrine_performance(valid_blueprints)
        
        # Create doctrine
        doctrine = StrategicDoctrine(
            id=str(uuid.uuid4()),
            name=doctrine_name,
            doctrine_type="Composite",
            source_blueprints=valid_blueprints,
            core_principles=core_principles,
            implementation_strategy=implementation_strategy,
            success_conditions=self._generate_success_conditions(valid_blueprints),
            failure_conditions=self._generate_failure_conditions(valid_blueprints),
            effectiveness_score=performance_characteristics['effectiveness'],
            adaptability_score=performance_characteristics['adaptability'],
            sustainability_score=performance_characteristics['sustainability'],
            creation_timestamp=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.doctrines[doctrine.id] = doctrine
        self.codification_stats['doctrine_creation_count'] += 1
        
        return doctrine
    
    def _analyze_blueprint_compatibility(self, blueprint_ids: List[str]) -> float:
        """Analyze compatibility between blueprints."""
        if len(blueprint_ids) < 2:
            return 1.0
        
        compatibility_scores = []
        
        for i, bp_id_1 in enumerate(blueprint_ids):
            for bp_id_2 in blueprint_ids[i+1:]:
                bp_1 = self.blueprints[bp_id_1]
                bp_2 = self.blueprints[bp_id_2]
                
                # Check for direct conflicts
                if bp_id_2 in bp_1.conflicts_with or bp_id_1 in bp_2.conflicts_with:
                    compatibility_scores.append(0.0)
                    continue
                
                # Calculate compatibility based on requirements
                compatibility = self._calculate_blueprint_compatibility(bp_1, bp_2)
                compatibility_scores.append(compatibility)
        
        return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.0
    
    def _calculate_blueprint_compatibility(self, bp_1: PatternBlueprint, 
                                         bp_2: PatternBlueprint) -> float:
        """Calculate compatibility between two blueprints."""
        # Check governance compatibility
        gov_1 = bp_1.implementation_requirements.get('governance_compatibility', [])
        gov_2 = bp_2.implementation_requirements.get('governance_compatibility', [])
        governance_compatibility = len(set(gov_1) & set(gov_2)) / max(len(gov_1), len(gov_2), 1)
        
        # Check resource compatibility
        res_1 = set(bp_1.implementation_requirements.get('required_resources', []))
        res_2 = set(bp_2.implementation_requirements.get('required_resources', []))
        resource_compatibility = len(res_1 & res_2) / max(len(res_1), len(res_2), 1)
        
        # Check environmental compatibility
        env_1 = bp_1.implementation_requirements.get('environmental_conditions', {})
        env_2 = bp_2.implementation_requirements.get('environmental_conditions', {})
        environmental_compatibility = self._calculate_environmental_compatibility(env_1, env_2)
        
        # Weighted average
        compatibility = (
            governance_compatibility * 0.4 +
            resource_compatibility * 0.3 +
            environmental_compatibility * 0.3
        )
        
        return compatibility
    
    def _calculate_environmental_compatibility(self, env_1: Dict[str, Any], 
                                             env_2: Dict[str, Any]) -> float:
        """Calculate environmental compatibility between two environments."""
        if not env_1 or not env_2:
            return 0.5  # Neutral if no environmental data
        
        common_conditions = set(env_1.keys()) & set(env_2.keys())
        if not common_conditions:
            return 0.5
        
        compatibility_scores = []
        for condition in common_conditions:
            val_1 = env_1[condition]
            val_2 = env_2[condition]
            
            if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                # Numeric compatibility
                diff = abs(val_1 - val_2)
                max_val = max(abs(val_1), abs(val_2), 1)
                compatibility = max(0, 1 - diff / max_val)
                compatibility_scores.append(compatibility)
            else:
                # Categorical compatibility
                compatibility = 1.0 if val_1 == val_2 else 0.0
                compatibility_scores.append(compatibility)
        
        return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.5
    
    def _extract_core_principles(self, blueprint_ids: List[str]) -> List[str]:
        """Extract core principles from blueprints."""
        principles = set()
        
        for bp_id in blueprint_ids:
            blueprint = self.blueprints[bp_id]
            
            # Extract principles from pattern description
            description = blueprint.pattern_description.lower()
            
            # Common principle keywords
            principle_keywords = [
                'cooperation', 'collaboration', 'efficiency', 'innovation',
                'stability', 'adaptability', 'scalability', 'sustainability',
                'transparency', 'accountability', 'flexibility', 'resilience'
            ]
            
            for keyword in principle_keywords:
                if keyword in description:
                    principles.add(keyword.title())
        
        return list(principles)
    
    def _generate_implementation_strategy(self, blueprint_ids: List[str]) -> Dict[str, Any]:
        """Generate implementation strategy for a doctrine."""
        strategy = {
            'implementation_order': blueprint_ids.copy(),
            'parallel_implementation': [],
            'sequential_implementation': [],
            'dependencies': {},
            'timeline': {}
        }
        
        # Analyze blueprint dependencies
        for bp_id in blueprint_ids:
            blueprint = self.blueprints[bp_id]
            dependencies = blueprint.derived_from + blueprint.influences
            strategy['dependencies'][bp_id] = dependencies
        
        # Determine implementation order
        strategy['sequential_implementation'] = self._determine_implementation_order(blueprint_ids)
        
        return strategy
    
    def _determine_implementation_order(self, blueprint_ids: List[str]) -> List[str]:
        """Determine the optimal implementation order for blueprints."""
        # Simple topological sort based on dependencies
        ordered = []
        remaining = blueprint_ids.copy()
        
        while remaining:
            # Find blueprints with no unresolved dependencies
            ready = []
            for bp_id in remaining:
                blueprint = self.blueprints[bp_id]
                dependencies = blueprint.derived_from + blueprint.influences
                unresolved_deps = [dep for dep in dependencies if dep in remaining]
                
                if not unresolved_deps:
                    ready.append(bp_id)
            
            if not ready:
                # Circular dependency - add remaining in original order
                ordered.extend(remaining)
                break
            
            # Add ready blueprints to ordered list
            ordered.extend(ready)
            for bp_id in ready:
                remaining.remove(bp_id)
        
        return ordered
    
    def _calculate_doctrine_performance(self, blueprint_ids: List[str]) -> Dict[str, float]:
        """Calculate performance characteristics for a doctrine."""
        if not blueprint_ids:
            return {'effectiveness': 0.0, 'adaptability': 0.0, 'sustainability': 0.0}
        
        # Aggregate performance metrics from blueprints
        effectiveness_scores = []
        adaptability_scores = []
        sustainability_scores = []
        
        for bp_id in blueprint_ids:
            blueprint = self.blueprints[bp_id]
            metrics = blueprint.performance_metrics
            
            effectiveness_scores.append(metrics.get('efficiency', 0.5))
            adaptability_scores.append(metrics.get('adaptability', 0.5))
            sustainability_scores.append(metrics.get('stability', 0.5))
        
        return {
            'effectiveness': sum(effectiveness_scores) / len(effectiveness_scores),
            'adaptability': sum(adaptability_scores) / len(adaptability_scores),
            'sustainability': sum(sustainability_scores) / len(sustainability_scores)
        }
    
    def _generate_success_conditions(self, blueprint_ids: List[str]) -> Dict[str, Any]:
        """Generate success conditions for a doctrine."""
        conditions = {
            'performance_thresholds': {},
            'stability_requirements': {},
            'resource_requirements': {},
            'temporal_requirements': {}
        }
        
        for bp_id in blueprint_ids:
            blueprint = self.blueprints[bp_id]
            requirements = blueprint.implementation_requirements
            
            # Aggregate requirements
            if 'performance_thresholds' in requirements:
                conditions['performance_thresholds'].update(requirements['performance_thresholds'])
            if 'stability_requirements' in requirements:
                conditions['stability_requirements'].update(requirements['stability_requirements'])
            if 'resource_requirements' in requirements:
                conditions['resource_requirements'].update(requirements['resource_requirements'])
            if 'temporal_requirements' in requirements:
                conditions['temporal_requirements'].update(requirements['temporal_requirements'])
        
        return conditions
    
    def _generate_failure_conditions(self, blueprint_ids: List[str]) -> Dict[str, Any]:
        """Generate failure conditions for a doctrine."""
        conditions = {
            'performance_failures': [],
            'stability_failures': [],
            'resource_failures': [],
            'temporal_failures': []
        }
        
        for bp_id in blueprint_ids:
            blueprint = self.blueprints[bp_id]
            constraints = blueprint.compatibility_constraints
            
            # Extract failure conditions from constraints
            for constraint in constraints:
                if 'minimum' in constraint.lower() or 'requires' in constraint.lower():
                    conditions['performance_failures'].append(constraint)
                elif 'stability' in constraint.lower():
                    conditions['stability_failures'].append(constraint)
                elif 'resource' in constraint.lower():
                    conditions['resource_failures'].append(constraint)
                elif 'time' in constraint.lower() or 'duration' in constraint.lower():
                    conditions['temporal_failures'].append(constraint)
        
        return conditions
    
    def get_blueprint(self, blueprint_id: str) -> Optional[PatternBlueprint]:
        """Get a blueprint by ID."""
        return self.blueprints.get(blueprint_id)
    
    def get_doctrine(self, doctrine_id: str) -> Optional[StrategicDoctrine]:
        """Get a doctrine by ID."""
        return self.doctrines.get(doctrine_id)
    
    def get_blueprints_by_type(self, blueprint_type: BlueprintType) -> List[PatternBlueprint]:
        """Get all blueprints of a specific type."""
        return [bp for bp in self.blueprints.values() if bp.blueprint_type == blueprint_type]
    
    def get_high_performing_blueprints(self, min_score: float = 0.7) -> List[PatternBlueprint]:
        """Get high-performing blueprints above a threshold."""
        return [
            bp for bp in self.blueprints.values() 
            if bp.validation_score >= min_score and bp.success_rate >= min_score
        ]
    
    def get_codification_statistics(self) -> Dict[str, Any]:
        """Get codification statistics."""
        return {
            'total_blueprints': len(self.blueprints),
            'total_doctrines': len(self.doctrines),
            'codification_stats': self.codification_stats.copy(),
            'blueprint_types': {
                bp_type.name: len(self.get_blueprints_by_type(bp_type))
                for bp_type in list(BlueprintType)
            },
            'high_performing_blueprints': len(self.get_high_performing_blueprints()),
            'average_validation_score': sum(bp.validation_score for bp in self.blueprints.values()) / max(len(self.blueprints), 1),
            'average_success_rate': sum(bp.success_rate for bp in self.blueprints.values()) / max(len(self.blueprints), 1)
        }
