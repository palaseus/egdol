"""
Macro-Pattern Detector for OmniMind Civilization Intelligence Layer
Detects emergent governance structures, network motifs, and technological clusters.
"""

import uuid
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import networkx as nx

from ...utils.pretty_printing import print_pattern_analysis, pp
from collections import defaultdict

from .core_structures import Civilization, AgentCluster, CivilizationIntelligenceCore


class PatternType(Enum):
    """Types of macro-patterns that can be detected."""
    GOVERNANCE_STRUCTURE = auto()
    NETWORK_TOPOLOGY = auto()
    TECHNOLOGICAL_CLUSTER = auto()
    CULTURAL_MEME = auto()
    RESOURCE_FLOW = auto()
    EMERGENT_BEHAVIOR = auto()
    STRATEGIC_ALLIANCE = auto()
    INNOVATION_CASCADE = auto()


class PatternSignificance(Enum):
    """Significance levels for detected patterns."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class MacroPattern:
    """Represents a detected macro-pattern."""
    id: str
    name: str
    pattern_type: PatternType
    civilization_id: str
    discovered_at: datetime = field(default_factory=datetime.now)
    
    # Pattern characteristics
    novelty_score: float = 0.0  # 0.0 to 1.0
    significance_score: float = 0.0  # 0.0 to 1.0
    stability_score: float = 0.0  # 0.0 to 1.0
    scalability_score: float = 0.0  # 0.0 to 1.0
    
    # Pattern data
    pattern_data: Dict[str, Any] = field(default_factory=dict)
    emergence_conditions: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    
    # Evolution tracking
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    cross_civilization_applicability: float = 0.0
    
    # Performance metrics
    performance_impact: Dict[str, float] = field(default_factory=dict)
    maintenance_requirements: List[str] = field(default_factory=list)


@dataclass
class PatternDetectionMetrics:
    """Metrics for pattern detection performance."""
    total_patterns_detected: int = 0
    patterns_by_type: Dict[PatternType, int] = field(default_factory=dict)
    average_novelty: float = 0.0
    average_significance: float = 0.0
    detection_accuracy: float = 0.0
    false_positive_rate: float = 0.0


class MacroPatternDetector:
    """Detects emergent macro-patterns in civilization evolution."""
    
    def __init__(self, core: CivilizationIntelligenceCore):
        """Initialize the macro-pattern detector."""
        self.core = core
        self.detected_patterns: Dict[str, MacroPattern] = {}
        self.pattern_candidates: List[Dict[str, Any]] = []
        self.detection_history: List[Dict[str, Any]] = []
        
        # Detection parameters
        self.detection_parameters = {
            'novelty_threshold': 0.6,
            'significance_threshold': 0.5,
            'stability_threshold': 0.7,
            'detection_interval': 10,  # Check every N ticks
            'pattern_persistence_required': 5  # Patterns must persist for N ticks
        }
        
        # Pattern classifiers
        self.pattern_classifiers = self._initialize_pattern_classifiers()
        
        # Detection metrics
        self.metrics = PatternDetectionMetrics()
        
        # Detection hooks
        self.detection_hooks: List[callable] = []
    
    def _initialize_pattern_classifiers(self) -> Dict[PatternType, Dict[str, Any]]:
        """Initialize pattern classification algorithms."""
        return {
            PatternType.GOVERNANCE_STRUCTURE: {
                'hierarchical': {'centralization': 0.8, 'authority_level': 0.7, 'decision_speed': 0.6},
                'democratic': {'participation': 0.9, 'consensus': 0.8, 'transparency': 0.7},
                'consensus': {'participation': 0.9, 'consensus': 0.9, 'efficiency': 0.5},
                'autocratic': {'centralization': 0.9, 'authority_level': 0.9, 'efficiency': 0.8}
            },
            PatternType.NETWORK_TOPOLOGY: {
                'star': {'centralization': 0.9, 'clustering': 0.1, 'path_length': 0.2},
                'mesh': {'centralization': 0.1, 'clustering': 0.8, 'path_length': 0.6},
                'hierarchical': {'centralization': 0.7, 'clustering': 0.6, 'path_length': 0.4},
                'small_world': {'centralization': 0.3, 'clustering': 0.7, 'path_length': 0.3}
            },
            PatternType.TECHNOLOGICAL_CLUSTER: {
                'innovation_hub': {'innovation_rate': 0.9, 'knowledge_density': 0.8, 'connectivity': 0.7},
                'specialization': {'expertise_depth': 0.9, 'diversity': 0.3, 'efficiency': 0.8},
                'diffusion_center': {'knowledge_flow': 0.9, 'adoption_rate': 0.8, 'reach': 0.7}
            },
            PatternType.CULTURAL_MEME: {
                'viral_spread': {'adoption_rate': 0.9, 'persistence': 0.6, 'mutation_rate': 0.7},
                'stable_tradition': {'persistence': 0.9, 'resistance': 0.8, 'transmission': 0.6},
                'adaptive_meme': {'adaptability': 0.8, 'evolution_rate': 0.7, 'resilience': 0.6}
            }
        }
    
    def detect_patterns(self, civilization_id: str, 
                       detection_context: Optional[Dict[str, Any]] = None) -> List[MacroPattern]:
        """Detect macro-patterns in a civilization."""
        civilization = self.core.get_civilization(civilization_id)
        if not civilization:
            return []
        
        detected_patterns = []
        
        # Detect different types of patterns
        governance_patterns = self._detect_governance_patterns(civilization)
        network_patterns = self._detect_network_patterns(civilization)
        technological_patterns = self._detect_technological_patterns(civilization)
        cultural_patterns = self._detect_cultural_patterns(civilization)
        resource_patterns = self._detect_resource_patterns(civilization)
        
        # Combine all patterns
        all_patterns = (governance_patterns + network_patterns + 
                       technological_patterns + cultural_patterns + resource_patterns)
        
        # Filter and validate patterns
        for pattern in all_patterns:
            if self._validate_pattern(pattern, civilization):
                self._register_pattern(pattern)
                detected_patterns.append(pattern)
        
        # Update metrics
        self._update_detection_metrics(detected_patterns)
        
        # Trigger detection hooks
        for hook in self.detection_hooks:
            try:
                hook(civilization_id, detected_patterns)
            except Exception as e:
                print(f"Pattern detection hook error: {e}")
        
        # Pretty print detected patterns
        if detected_patterns:
            print_pattern_analysis(detected_patterns, f"Pattern Analysis - Civilization {civilization_id}")
        
        return detected_patterns
    
    def _detect_governance_patterns(self, civilization: Civilization) -> List[MacroPattern]:
        """Detect governance structure patterns."""
        patterns = []
        
        # Analyze governance characteristics
        governance_metrics = {
            'centralization': self._calculate_centralization(civilization),
            'participation': self._calculate_participation(civilization),
            'authority_level': self._calculate_authority_level(civilization),
            'decision_speed': civilization.decision_making_efficiency,
            'consensus_level': self._calculate_consensus_level(civilization),
            'transparency': self._calculate_transparency(civilization)
        }
        
        # Classify governance pattern
        pattern_type = self._classify_governance_pattern(governance_metrics)
        
        if pattern_type:
            pattern = MacroPattern(
                id=str(uuid.uuid4()),
                name=f"Governance Pattern: {pattern_type.title()}",
                pattern_type=PatternType.GOVERNANCE_STRUCTURE,
                civilization_id=civilization.id,
                novelty_score=self._calculate_novelty(governance_metrics),
                significance_score=self._calculate_significance(governance_metrics),
                stability_score=self._calculate_stability(governance_metrics),
                scalability_score=self._calculate_scalability(governance_metrics),
                pattern_data=governance_metrics,
                emergence_conditions=self._identify_emergence_conditions(civilization),
                affected_components=['governance', 'decision_making', 'coordination']
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_network_patterns(self, civilization: Civilization) -> List[MacroPattern]:
        """Detect network topology patterns."""
        patterns = []
        
        # Analyze network characteristics
        network_metrics = {
            'centralization': nx.degree_centrality(civilization.communication_network),
            'clustering': nx.average_clustering(civilization.communication_network) if civilization.communication_network.number_of_nodes() > 0 else 0.0,
            'path_length': nx.average_shortest_path_length(civilization.communication_network) if civilization.communication_network.number_of_nodes() > 1 else 0,
            'density': nx.density(civilization.communication_network),
            'connectivity': nx.is_connected(civilization.communication_network) if civilization.communication_network.number_of_nodes() > 0 else False
        }
        
        # Classify network pattern
        pattern_type = self._classify_network_pattern(network_metrics)
        
        if pattern_type:
            pattern = MacroPattern(
                id=str(uuid.uuid4()),
                name=f"Network Pattern: {pattern_type.title()}",
                pattern_type=PatternType.NETWORK_TOPOLOGY,
                civilization_id=civilization.id,
                novelty_score=self._calculate_novelty(network_metrics),
                significance_score=self._calculate_significance(network_metrics),
                stability_score=self._calculate_stability(network_metrics),
                scalability_score=self._calculate_scalability(network_metrics),
                pattern_data=network_metrics,
                emergence_conditions=self._identify_emergence_conditions(civilization),
                affected_components=['communication', 'coordination', 'information_flow']
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_technological_patterns(self, civilization: Civilization) -> List[MacroPattern]:
        """Detect technological cluster patterns."""
        patterns = []
        
        # Analyze technological characteristics
        tech_metrics = {
            'innovation_rate': civilization.innovation_capacity,
            'knowledge_density': len(civilization.knowledge_base),
            'tech_diversity': self._calculate_tech_diversity(civilization),
            'adoption_rate': self._calculate_adoption_rate(civilization),
            'diffusion_speed': civilization.knowledge_diffusion_rate,
            'specialization': self._calculate_specialization(civilization)
        }
        
        # Classify technological pattern
        pattern_type = self._classify_technological_pattern(tech_metrics)
        
        if pattern_type:
            pattern = MacroPattern(
                id=str(uuid.uuid4()),
                name=f"Technological Pattern: {pattern_type.title()}",
                pattern_type=PatternType.TECHNOLOGICAL_CLUSTER,
                civilization_id=civilization.id,
                novelty_score=self._calculate_novelty(tech_metrics),
                significance_score=self._calculate_significance(tech_metrics),
                stability_score=self._calculate_stability(tech_metrics),
                scalability_score=self._calculate_scalability(tech_metrics),
                pattern_data=tech_metrics,
                emergence_conditions=self._identify_emergence_conditions(civilization),
                affected_components=['technology', 'innovation', 'knowledge']
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_cultural_patterns(self, civilization: Civilization) -> List[MacroPattern]:
        """Detect cultural meme patterns."""
        patterns = []
        
        # Analyze cultural characteristics
        cultural_metrics = {
            'meme_diversity': len(civilization.cultural_traits),
            'transmission_rate': self._calculate_transmission_rate(civilization),
            'persistence': self._calculate_cultural_persistence(civilization),
            'mutation_rate': self._calculate_mutation_rate(civilization),
            'adoption_rate': self._calculate_cultural_adoption_rate(civilization),
            'resistance': self._calculate_cultural_resistance(civilization)
        }
        
        # Classify cultural pattern
        pattern_type = self._classify_cultural_pattern(cultural_metrics)
        
        if pattern_type:
            pattern = MacroPattern(
                id=str(uuid.uuid4()),
                name=f"Cultural Pattern: {pattern_type.title()}",
                pattern_type=PatternType.CULTURAL_MEME,
                civilization_id=civilization.id,
                novelty_score=self._calculate_novelty(cultural_metrics),
                significance_score=self._calculate_significance(cultural_metrics),
                stability_score=self._calculate_stability(cultural_metrics),
                scalability_score=self._calculate_scalability(cultural_metrics),
                pattern_data=cultural_metrics,
                emergence_conditions=self._identify_emergence_conditions(civilization),
                affected_components=['culture', 'values', 'behavior']
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_resource_patterns(self, civilization: Civilization) -> List[MacroPattern]:
        """Detect resource flow patterns."""
        patterns = []
        
        # Analyze resource characteristics
        resource_metrics = {
            'efficiency': sum(civilization.resource_efficiency.values()) / len(civilization.resource_efficiency),
            'abundance': sum(civilization.resource_pools.values()) / len(civilization.resource_pools),
            'distribution': self._calculate_resource_distribution(civilization),
            'flow_rate': self._calculate_resource_flow_rate(civilization),
            'sustainability': civilization.sustainability_index,
            'scarcity': self._calculate_resource_scarcity(civilization)
        }
        
        # Check for significant resource patterns
        if resource_metrics['efficiency'] > 0.8 or resource_metrics['scarcity'] > 0.7:
            pattern = MacroPattern(
                id=str(uuid.uuid4()),
                name="Resource Flow Pattern",
                pattern_type=PatternType.RESOURCE_FLOW,
                civilization_id=civilization.id,
                novelty_score=self._calculate_novelty(resource_metrics),
                significance_score=self._calculate_significance(resource_metrics),
                stability_score=self._calculate_stability(resource_metrics),
                scalability_score=self._calculate_scalability(resource_metrics),
                pattern_data=resource_metrics,
                emergence_conditions=self._identify_emergence_conditions(civilization),
                affected_components=['resources', 'economy', 'sustainability']
            )
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_centralization(self, civilization: Civilization) -> float:
        """Calculate governance centralization."""
        if civilization.governance_model in [civilization.governance_model.AUTOCRATIC, civilization.governance_model.MERITOCRATIC]:
            return 0.8
        elif civilization.governance_model == civilization.governance_model.DEMOCRATIC:
            return 0.4
        else:
            return 0.2
    
    def _calculate_participation(self, civilization: Civilization) -> float:
        """Calculate participation level in governance."""
        if civilization.governance_model == civilization.governance_model.DEMOCRATIC:
            return 0.9
        elif civilization.governance_model == civilization.governance_model.CONSENSUS:
            return 0.95
        else:
            return 0.3
    
    def _calculate_authority_level(self, civilization: Civilization) -> float:
        """Calculate authority level in governance."""
        if civilization.governance_model == civilization.governance_model.AUTOCRATIC:
            return 0.9
        elif civilization.governance_model == civilization.governance_model.MERITOCRATIC:
            return 0.7
        else:
            return 0.3
    
    def _calculate_consensus_level(self, civilization: Civilization) -> float:
        """Calculate consensus level in governance."""
        if civilization.governance_model == civilization.governance_model.CONSENSUS:
            return 0.9
        elif civilization.governance_model == civilization.governance_model.DEMOCRATIC:
            return 0.6
        else:
            return 0.3
    
    def _calculate_transparency(self, civilization: Civilization) -> float:
        """Calculate transparency level in governance."""
        if civilization.governance_model == civilization.governance_model.DEMOCRATIC:
            return 0.8
        elif civilization.governance_model == civilization.governance_model.CONSENSUS:
            return 0.9
        else:
            return 0.4
    
    def _calculate_tech_diversity(self, civilization: Civilization) -> float:
        """Calculate technological diversity."""
        return len(civilization.knowledge_base) / 10.0  # Normalize
    
    def _calculate_adoption_rate(self, civilization: Civilization) -> float:
        """Calculate technology adoption rate."""
        return civilization.knowledge_diffusion_rate
    
    def _calculate_specialization(self, civilization: Civilization) -> float:
        """Calculate technological specialization."""
        # Based on agent cluster specialization
        total_specialists = sum(1 for cluster in civilization.agent_clusters.values() 
                               if cluster.cluster_type.name == 'SPECIALIST')
        return total_specialists / len(civilization.agent_clusters) if civilization.agent_clusters else 0.0
    
    def _calculate_transmission_rate(self, civilization: Civilization) -> float:
        """Calculate cultural transmission rate."""
        return civilization.cooperation_level * 0.8
    
    def _calculate_cultural_persistence(self, civilization: Civilization) -> float:
        """Calculate cultural persistence."""
        return civilization.stability * 0.7
    
    def _calculate_mutation_rate(self, civilization: Civilization) -> float:
        """Calculate cultural mutation rate."""
        return civilization.innovation_capacity * 0.5
    
    def _calculate_cultural_adoption_rate(self, civilization: Civilization) -> float:
        """Calculate cultural adoption rate."""
        return civilization.cooperation_level * civilization.innovation_capacity
    
    def _calculate_cultural_resistance(self, civilization: Civilization) -> float:
        """Calculate cultural resistance to change."""
        return (1.0 - civilization.innovation_capacity) * 0.8
    
    def _calculate_resource_distribution(self, civilization: Civilization) -> float:
        """Calculate resource distribution equality."""
        if not civilization.resource_pools:
            return 0.0
        values = list(civilization.resource_pools.values())
        return 1.0 - (np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0
    
    def _calculate_resource_flow_rate(self, civilization: Civilization) -> float:
        """Calculate resource flow rate."""
        return civilization.resource_efficiency.get('energy', 0.5) * 0.8
    
    def _calculate_resource_scarcity(self, civilization: Civilization) -> float:
        """Calculate resource scarcity level."""
        if not civilization.resource_pools:
            return 0.0
        avg_resources = np.mean(list(civilization.resource_pools.values()))
        return max(0.0, 1.0 - (avg_resources / 100.0))  # Normalize
    
    def _classify_governance_pattern(self, metrics: Dict[str, float]) -> Optional[str]:
        """Classify governance pattern based on metrics."""
        classifiers = self.pattern_classifiers[PatternType.GOVERNANCE_STRUCTURE]
        
        best_match = None
        best_score = 0.0
        
        for pattern_name, weights in classifiers.items():
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    score += weight * metrics[metric]
                    total_weight += weight
            
            if total_weight > 0:
                score /= total_weight
                
                if score > best_score and score > self.detection_parameters['significance_threshold']:
                    best_score = score
                    best_match = pattern_name
        
        return best_match
    
    def _classify_network_pattern(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Classify network pattern based on metrics."""
        classifiers = self.pattern_classifiers[PatternType.NETWORK_TOPOLOGY]
        
        best_match = None
        best_score = 0.0
        
        for pattern_name, weights in classifiers.items():
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    if isinstance(metrics[metric], dict):
                        # Handle centrality metrics
                        avg_value = np.mean(list(metrics[metric].values())) if metrics[metric] else 0.0
                    else:
                        avg_value = metrics[metric]
                    
                    score += weight * avg_value
                    total_weight += weight
            
            if total_weight > 0:
                score /= total_weight
                
                if score > best_score and score > self.detection_parameters['significance_threshold']:
                    best_score = score
                    best_match = pattern_name
        
        return best_match
    
    def _classify_technological_pattern(self, metrics: Dict[str, float]) -> Optional[str]:
        """Classify technological pattern based on metrics."""
        classifiers = self.pattern_classifiers[PatternType.TECHNOLOGICAL_CLUSTER]
        
        best_match = None
        best_score = 0.0
        
        for pattern_name, weights in classifiers.items():
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    score += weight * metrics[metric]
                    total_weight += weight
            
            if total_weight > 0:
                score /= total_weight
                
                if score > best_score and score > self.detection_parameters['significance_threshold']:
                    best_score = score
                    best_match = pattern_name
        
        return best_match
    
    def _classify_cultural_pattern(self, metrics: Dict[str, float]) -> Optional[str]:
        """Classify cultural pattern based on metrics."""
        classifiers = self.pattern_classifiers[PatternType.CULTURAL_MEME]
        
        best_match = None
        best_score = 0.0
        
        for pattern_name, weights in classifiers.items():
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    score += weight * metrics[metric]
                    total_weight += weight
            
            if total_weight > 0:
                score /= total_weight
                
                if score > best_score and score > self.detection_parameters['significance_threshold']:
                    best_score = score
                    best_match = pattern_name
        
        return best_match
    
    def _calculate_novelty(self, metrics: Dict[str, float]) -> float:
        """Calculate pattern novelty score."""
        # Simple novelty calculation based on metric variance
        if not metrics:
            return 0.0
        values = list(metrics.values())
        return min(1.0, np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0
    
    def _calculate_significance(self, metrics: Dict[str, float]) -> float:
        """Calculate pattern significance score."""
        # Significance based on metric magnitudes
        if not metrics:
            return 0.0
        return min(1.0, np.mean(list(metrics.values())))
    
    def _calculate_stability(self, metrics: Dict[str, float]) -> float:
        """Calculate pattern stability score."""
        # Stability based on metric consistency
        if not metrics:
            return 0.0
        values = list(metrics.values())
        return 1.0 - (np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0
    
    def _calculate_scalability(self, metrics: Dict[str, float]) -> float:
        """Calculate pattern scalability score."""
        # Scalability based on metric magnitudes and consistency
        if not metrics:
            return 0.0
        values = list(metrics.values())
        magnitude_score = min(1.0, np.mean(values))
        consistency_score = 1.0 - (np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0
        return (magnitude_score + consistency_score) / 2.0
    
    def _identify_emergence_conditions(self, civilization: Civilization) -> List[str]:
        """Identify conditions that led to pattern emergence."""
        conditions = []
        
        if civilization.stability > 0.7:
            conditions.append('high_stability')
        if civilization.complexity > 0.6:
            conditions.append('high_complexity')
        if civilization.innovation_capacity > 0.8:
            conditions.append('high_innovation')
        if civilization.cooperation_level > 0.8:
            conditions.append('high_cooperation')
        if civilization.adaptability > 0.7:
            conditions.append('high_adaptability')
        
        return conditions
    
    def _validate_pattern(self, pattern: MacroPattern, civilization: Civilization) -> bool:
        """Validate that a pattern meets detection criteria."""
        return (pattern.novelty_score >= self.detection_parameters['novelty_threshold'] and
                pattern.significance_score >= self.detection_parameters['significance_threshold'] and
                pattern.stability_score >= self.detection_parameters['stability_threshold'])
    
    def _register_pattern(self, pattern: MacroPattern):
        """Register a detected pattern."""
        self.detected_patterns[pattern.id] = pattern
        
        # Record detection
        self.detection_history.append({
            'timestamp': datetime.now(),
            'pattern_id': pattern.id,
            'civilization_id': pattern.civilization_id,
            'pattern_type': pattern.pattern_type.name,
            'novelty': pattern.novelty_score,
            'significance': pattern.significance_score
        })
    
    def _update_detection_metrics(self, patterns: List[MacroPattern]):
        """Update pattern detection metrics."""
        self.metrics.total_patterns_detected += len(patterns)
        
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in self.metrics.patterns_by_type:
                self.metrics.patterns_by_type[pattern_type] = 0
            self.metrics.patterns_by_type[pattern_type] += 1
        
        # Update averages
        if patterns:
            self.metrics.average_novelty = np.mean([p.novelty_score for p in patterns])
            self.metrics.average_significance = np.mean([p.significance_score for p in patterns])
    
    def add_detection_hook(self, hook: callable):
        """Add a pattern detection hook."""
        self.detection_hooks.append(hook)
    
    def get_pattern(self, pattern_id: str) -> Optional[MacroPattern]:
        """Get a pattern by ID."""
        return self.detected_patterns.get(pattern_id)
    
    def get_patterns_by_civilization(self, civilization_id: str) -> List[MacroPattern]:
        """Get all patterns for a civilization."""
        return [pattern for pattern in self.detected_patterns.values() 
                if pattern.civilization_id == civilization_id]
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[MacroPattern]:
        """Get all patterns of a specific type."""
        return [pattern for pattern in self.detected_patterns.values() 
                if pattern.pattern_type == pattern_type]
    
    def get_detection_metrics(self) -> PatternDetectionMetrics:
        """Get pattern detection metrics."""
        return self.metrics
    
    def get_detection_history(self) -> List[Dict[str, Any]]:
        """Get pattern detection history."""
        return self.detection_history.copy()