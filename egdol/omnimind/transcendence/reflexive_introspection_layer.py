"""
Reflexive Introspection Layer for OmniMind Civilization Intelligence Layer
Enables the system to audit its own simulation histories and adapt meta-rules accordingly.
This is where civilization discovers civilizational laws.
"""

import json
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import statistics
import numpy as np

from .core_structures import (
    Civilization, AgentCluster, EnvironmentState, TemporalState,
    CivilizationIntelligenceCore, GovernanceModel, AgentType, CivilizationArchetype
)
from .pattern_codification_engine import PatternBlueprint, BlueprintType, StrategicDoctrine
from .civilizational_genetic_archive import CivilizationDNA, LineageType, ArchiveStatus
from .strategic_feedback_loop import FeedbackApplication, FeedbackType


class IntrospectionType(Enum):
    """Types of introspection analysis."""
    HISTORICAL_ANALYSIS = auto()
    PATTERN_DOMINANCE = auto()
    META_RULE_DISCOVERY = auto()
    EVOLUTIONARY_TRAJECTORY = auto()
    SYSTEM_PERFORMANCE = auto()
    EMERGENT_LAW_DETECTION = auto()


class IntrospectionStatus(Enum):
    """Status of introspection analysis."""
    PENDING = auto()
    ANALYZING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ARCHIVED = auto()


class MetaRuleType(Enum):
    """Types of discovered meta-rules."""
    GOVERNANCE_LAW = auto()
    RESOURCE_LAW = auto()
    COMMUNICATION_LAW = auto()
    INNOVATION_LAW = auto()
    CULTURAL_LAW = auto()
    EVOLUTIONARY_LAW = auto()
    STRATEGIC_LAW = auto()


@dataclass
class IntrospectionAnalysis:
    """Results of introspection analysis."""
    id: str
    analysis_type: IntrospectionType
    target_civilization_ids: List[str]
    analysis_timestamp: datetime
    
    # Analysis results
    discovered_patterns: List[Dict[str, Any]] = field(default_factory=list)
    dominance_analysis: Dict[str, float] = field(default_factory=dict)
    evolutionary_trajectories: List[Dict[str, Any]] = field(default_factory=list)
    performance_correlations: Dict[str, float] = field(default_factory=dict)
    
    # Meta-rule discoveries
    discovered_meta_rules: List[Dict[str, Any]] = field(default_factory=list)
    law_confidence_scores: Dict[str, float] = field(default_factory=dict)
    law_applicability_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # System insights
    system_insights: List[str] = field(default_factory=list)
    optimization_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    meta_rule_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis metadata
    analysis_duration: float = 0.0
    confidence_score: float = 0.0
    status: IntrospectionStatus = IntrospectionStatus.PENDING


@dataclass
class MetaRule:
    """A discovered meta-rule governing civilization behavior."""
    id: str
    name: str
    rule_type: MetaRuleType
    source_analysis_id: str
    
    # Rule definition
    rule_description: str
    rule_conditions: Dict[str, Any]
    rule_effects: Dict[str, Any]
    rule_exceptions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Rule validation
    confidence_score: float = 0.0
    validation_count: int = 0
    success_rate: float = 0.0
    applicability_score: float = 0.0
    
    # Rule evolution
    discovery_timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    effectiveness_score: float = 0.0
    
    # Rule relationships
    related_rules: List[str] = field(default_factory=list)
    conflicting_rules: List[str] = field(default_factory=list)
    prerequisite_rules: List[str] = field(default_factory=list)


@dataclass
class SystemInsight:
    """A high-level insight about the system's behavior."""
    id: str
    insight_type: str
    description: str
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence_level: float = 0.0
    discovery_timestamp: datetime = field(default_factory=datetime.now)
    impact_score: float = 0.0


class ReflexiveIntrospectionLayer:
    """Layer for reflexive introspection and meta-rule discovery."""
    
    def __init__(self, core: CivilizationIntelligenceCore, 
                 pattern_codification_engine, genetic_archive, strategic_feedback_loop):
        """Initialize the reflexive introspection layer."""
        self.core = core
        self.pattern_codification_engine = pattern_codification_engine
        self.genetic_archive = genetic_archive
        self.strategic_feedback_loop = strategic_feedback_loop
        
        # Introspection storage
        self.introspection_analyses: Dict[str, IntrospectionAnalysis] = {}
        self.discovered_meta_rules: Dict[str, MetaRule] = {}
        self.system_insights: Dict[str, SystemInsight] = {}
        
        # Introspection parameters
        self.introspection_parameters = {
            'min_civilization_count': 3,
            'min_analysis_period': 7,  # days
            'confidence_threshold': 0.7,
            'meta_rule_min_confidence': 0.8,
            'insight_min_confidence': 0.6,
            'analysis_timeout': 60.0  # seconds
        }
        
        # Introspection statistics
        self.introspection_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'meta_rules_discovered': 0,
            'system_insights_generated': 0,
            'optimization_recommendations': 0
        }
        
        # Analysis cache
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Meta-rule evolution tracking
        self.meta_rule_evolution: Dict[str, List[Dict[str, Any]]] = {}
    
    def perform_comprehensive_introspection(self, civilization_ids: List[str] = None) -> str:
        """Perform comprehensive introspection analysis."""
        if civilization_ids is None:
            civilization_ids = list(self.core.civilizations.keys())
        
        if len(civilization_ids) < self.introspection_parameters['min_civilization_count']:
            print(f"Warning: Not enough civilizations for introspection ({len(civilization_ids)} < {self.introspection_parameters['min_civilization_count']})")
            return None
        
        # Create introspection analysis
        analysis = IntrospectionAnalysis(
            id=str(uuid.uuid4()),
            analysis_type=IntrospectionType.HISTORICAL_ANALYSIS,
            target_civilization_ids=civilization_ids,
            analysis_timestamp=datetime.now(),
            status=IntrospectionStatus.ANALYZING
        )
        
        try:
            start_time = datetime.now()
            
            # Perform various types of introspection
            self._analyze_historical_patterns(analysis)
            self._analyze_pattern_dominance(analysis)
            self._analyze_evolutionary_trajectories(analysis)
            self._analyze_performance_correlations(analysis)
            self._discover_meta_rules(analysis)
            self._generate_system_insights(analysis)
            self._generate_optimization_recommendations(analysis)
            
            # Complete analysis
            analysis.analysis_duration = (datetime.now() - start_time).total_seconds()
            analysis.status = IntrospectionStatus.COMPLETED
            analysis.confidence_score = self._calculate_analysis_confidence(analysis)
            
            # Store analysis
            self.introspection_analyses[analysis.id] = analysis
            self.introspection_stats['total_analyses'] += 1
            self.introspection_stats['successful_analyses'] += 1
            
            return analysis.id
            
        except Exception as e:
            import traceback
            print(f"Error in comprehensive introspection: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            analysis.status = IntrospectionStatus.FAILED
            self.introspection_stats['failed_analyses'] += 1
            return None
    
    def _analyze_historical_patterns(self, analysis: IntrospectionAnalysis):
        """Analyze historical patterns across civilizations."""
        patterns = []
        
        for civ_id in analysis.target_civilization_ids:
            civilization = self.core.get_civilization(civ_id)
            if not civilization:
                continue
            
            # Analyze civilization characteristics
            civ_patterns = {
                'civilization_id': civ_id,
                'archetype': civilization.archetype.name,
                'governance_model': civilization.governance_model.name,
                'stability_trend': civilization.stability,
                'complexity_trend': civilization.complexity,
                'adaptability_trend': civilization.adaptability,
                'innovation_capacity': civilization.innovation_capacity,
                'cooperation_level': civilization.cooperation_level,
                'population_size': civilization.total_population,
                'resource_efficiency': sum(civilization.resource_efficiency.values()) / len(civilization.resource_efficiency) if civilization.resource_efficiency else 0.0
            }
            
            patterns.append(civ_patterns)
        
        analysis.discovered_patterns = patterns
    
    def _analyze_pattern_dominance(self, analysis: IntrospectionAnalysis):
        """Analyze which patterns dominate across civilizations."""
        if not analysis.discovered_patterns:
            return
        
        # Analyze archetype dominance
        archetypes = [p['archetype'] for p in analysis.discovered_patterns]
        archetype_counts = {arch: archetypes.count(arch) for arch in set(archetypes)}
        total_civs = len(archetypes)
        archetype_dominance = {arch: count / total_civs for arch, count in archetype_counts.items()}
        
        # Analyze governance dominance
        governance_models = [p['governance_model'] for p in analysis.discovered_patterns]
        governance_counts = {gov: governance_models.count(gov) for gov in set(governance_models)}
        governance_dominance = {gov: count / total_civs for gov, count in governance_counts.items()}
        
        # Analyze performance dominance
        stability_values = [p['stability_trend'] for p in analysis.discovered_patterns]
        complexity_values = [p['complexity_trend'] for p in analysis.discovered_patterns]
        adaptability_values = [p['adaptability_trend'] for p in analysis.discovered_patterns]
        
        performance_dominance = {
            'stability_dominance': statistics.mean(stability_values),
            'complexity_dominance': statistics.mean(complexity_values),
            'adaptability_dominance': statistics.mean(adaptability_values),
            'innovation_dominance': statistics.mean([p['innovation_capacity'] for p in analysis.discovered_patterns]),
            'cooperation_dominance': statistics.mean([p['cooperation_level'] for p in analysis.discovered_patterns])
        }
        
        analysis.dominance_analysis = {
            'archetype_dominance': archetype_dominance,
            'governance_dominance': governance_dominance,
            'performance_dominance': performance_dominance
        }
    
    def _analyze_evolutionary_trajectories(self, analysis: IntrospectionAnalysis):
        """Analyze evolutionary trajectories of civilizations."""
        trajectories = []
        
        for civ_id in analysis.target_civilization_ids:
            civilization = self.core.get_civilization(civ_id)
            if not civilization:
                continue
            
            # Analyze temporal state
            temporal_state = civilization.temporal_state
            trajectory = {
                'civilization_id': civ_id,
                'current_phase': temporal_state.current_phase.name if hasattr(temporal_state, 'current_phase') else 'unknown',
                'phase_transitions': len(temporal_state.phase_transitions),
                'simulation_time': temporal_state.simulation_time,
                'evolution_events': len(temporal_state.major_events),
                'stability_evolution': civilization.stability,
                'complexity_evolution': civilization.complexity,
                'adaptability_evolution': civilization.adaptability
            }
            
            trajectories.append(trajectory)
        
        analysis.evolutionary_trajectories = trajectories
    
    def _analyze_performance_correlations(self, analysis: IntrospectionAnalysis):
        """Analyze correlations between different performance metrics."""
        if len(analysis.discovered_patterns) < 2:
            return
        
        # Extract metrics
        stability_values = [p['stability_trend'] for p in analysis.discovered_patterns]
        complexity_values = [p['complexity_trend'] for p in analysis.discovered_patterns]
        adaptability_values = [p['adaptability_trend'] for p in analysis.discovered_patterns]
        innovation_values = [p['innovation_capacity'] for p in analysis.discovered_patterns]
        cooperation_values = [p['cooperation_level'] for p in analysis.discovered_patterns]
        
        # Calculate correlations
        correlations = {}
        
        # Stability vs other metrics
        correlations['stability_complexity'] = self._calculate_correlation(stability_values, complexity_values)
        correlations['stability_adaptability'] = self._calculate_correlation(stability_values, adaptability_values)
        correlations['stability_innovation'] = self._calculate_correlation(stability_values, innovation_values)
        correlations['stability_cooperation'] = self._calculate_correlation(stability_values, cooperation_values)
        
        # Complexity vs other metrics
        correlations['complexity_adaptability'] = self._calculate_correlation(complexity_values, adaptability_values)
        correlations['complexity_innovation'] = self._calculate_correlation(complexity_values, innovation_values)
        correlations['complexity_cooperation'] = self._calculate_correlation(complexity_values, cooperation_values)
        
        # Innovation vs cooperation
        correlations['innovation_cooperation'] = self._calculate_correlation(innovation_values, cooperation_values)
        
        analysis.performance_correlations = correlations
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate correlation coefficient between two sets of values."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        try:
            # Ensure values are numeric
            x_vals = [float(x) for x in x_values if isinstance(x, (int, float))]
            y_vals = [float(y) for y in y_values if isinstance(y, (int, float))]
            
            if len(x_vals) != len(y_vals) or len(x_vals) < 2:
                return 0.0
            
            # Check for constant values (zero variance) to avoid division by zero
            x_std = np.std(x_vals)
            y_std = np.std(y_vals)
            
            if x_std == 0.0 or y_std == 0.0:
                return 0.0
            
            # Use numpy for correlation calculation with proper error handling
            with np.errstate(divide='ignore', invalid='ignore'):
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                return correlation if not np.isnan(correlation) and not np.isinf(correlation) else 0.0
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 0.0
    
    def _discover_meta_rules(self, analysis: IntrospectionAnalysis):
        """Discover meta-rules governing civilization behavior."""
        meta_rules = []
        
        # Analyze dominance patterns for rule discovery
        dominance = analysis.dominance_analysis
        correlations = analysis.performance_correlations
        
        # Rule 1: Governance-Adaptability Law
        if 'governance_dominance' in dominance:
            governance_dom = dominance['governance_dominance']
            if max(governance_dom.values()) > 0.6:  # Strong dominance
                dominant_gov = max(governance_dom, key=governance_dom.get)
                
                rule = {
                    'rule_type': MetaRuleType.GOVERNANCE_LAW,
                    'name': f"{dominant_gov} Dominance Law",
                    'description': f"Civilizations tend to evolve toward {dominant_gov} governance when certain conditions are met",
                    'conditions': {
                        'population_threshold': 100,
                        'stability_threshold': 0.6,
                        'complexity_threshold': 0.5
                    },
                    'effects': {
                        'governance_convergence': 0.8,
                        'stability_improvement': 0.1,
                        'decision_efficiency_improvement': 0.15
                    },
                    'confidence': governance_dom[dominant_gov]
                }
                meta_rules.append(rule)
        
        # Rule 2: Innovation-Cooperation Synergy Law
        if 'innovation_cooperation' in correlations:
            corr = correlations['innovation_cooperation']
            if abs(corr) > 0.5:  # Strong correlation
                rule = {
                    'rule_type': MetaRuleType.INNOVATION_LAW,
                    'name': "Innovation-Cooperation Synergy Law",
                    'description': f"High cooperation levels (correlation: {corr:.3f}) are associated with innovation capacity",
                    'conditions': {
                        'cooperation_threshold': 0.7,
                        'innovation_threshold': 0.6
                    },
                    'effects': {
                        'innovation_boost': 0.2,
                        'cooperation_boost': 0.1,
                        'synergy_multiplier': 1.5
                    },
                    'confidence': abs(corr)
                }
                meta_rules.append(rule)
        
        # Rule 3: Stability-Complexity Trade-off Law
        if 'stability_complexity' in correlations:
            corr = correlations['stability_complexity']
            if corr < -0.3:  # Negative correlation
                rule = {
                    'rule_type': MetaRuleType.EVOLUTIONARY_LAW,
                    'name': "Stability-Complexity Trade-off Law",
                    'description': f"High complexity tends to reduce stability (correlation: {corr:.3f})",
                    'conditions': {
                        'complexity_threshold': 0.7,
                        'stability_threshold': 0.5
                    },
                    'effects': {
                        'complexity_penalty': 0.1,
                        'stability_boost': 0.05,
                        'optimization_opportunity': True
                    },
                    'confidence': abs(corr)
                }
                meta_rules.append(rule)
        
        # Rule 4: Resource Efficiency Law
        if analysis.discovered_patterns:
            resource_efficiencies = [p['resource_efficiency'] for p in analysis.discovered_patterns]
            avg_efficiency = statistics.mean(resource_efficiencies)
            
            if avg_efficiency > 0.6:  # High average efficiency
                rule = {
                    'rule_type': MetaRuleType.RESOURCE_LAW,
                    'name': "Resource Efficiency Optimization Law",
                    'description': f"Civilizations with resource efficiency > 0.6 show improved overall performance",
                    'conditions': {
                        'resource_efficiency_threshold': 0.6,
                        'population_threshold': 50
                    },
                    'effects': {
                        'performance_boost': 0.15,
                        'resource_optimization': 0.2,
                        'sustainability_improvement': 0.1
                    },
                    'confidence': min(1.0, avg_efficiency)
                }
                meta_rules.append(rule)
        
        # Store discovered meta-rules
        for rule_data in meta_rules:
            if rule_data['confidence'] >= self.introspection_parameters['meta_rule_min_confidence']:
                meta_rule = MetaRule(
                    id=str(uuid.uuid4()),
                    name=rule_data['name'],
                    rule_type=rule_data['rule_type'],
                    source_analysis_id=analysis.id,
                    rule_description=rule_data['description'],
                    rule_conditions=rule_data['conditions'],
                    rule_effects=rule_data['effects'],
                    confidence_score=rule_data['confidence']
                )
                
                self.discovered_meta_rules[meta_rule.id] = meta_rule
                analysis.discovered_meta_rules.append(rule_data)
                self.introspection_stats['meta_rules_discovered'] += 1
        
        analysis.law_confidence_scores = {rule['name']: rule['confidence'] for rule in meta_rules}
    
    def _generate_system_insights(self, analysis: IntrospectionAnalysis):
        """Generate high-level system insights."""
        insights = []
        
        # Insight 1: Archetype Performance Analysis
        if analysis.dominance_analysis and 'archetype_dominance' in analysis.dominance_analysis:
            arch_dom = analysis.dominance_analysis['archetype_dominance']
            dominant_archetype = max(arch_dom, key=arch_dom.get)
            dominance_score = arch_dom[dominant_archetype]
            
            if dominance_score > 0.5:
                insight = {
                    'type': 'archetype_performance',
                    'description': f"{dominant_archetype} archetype shows dominance ({dominance_score:.1%}) across civilizations",
                    'confidence': dominance_score,
                    'evidence': [f"Found in {dominance_score:.1%} of analyzed civilizations"]
                }
                insights.append(insight)
        
        # Insight 2: Performance Correlation Insights
        if analysis.performance_correlations:
            strong_correlations = [(k, v) for k, v in analysis.performance_correlations.items() if abs(v) > 0.5]
            
            for corr_name, corr_value in strong_correlations:
                insight = {
                    'type': 'performance_correlation',
                    'description': f"Strong correlation ({corr_value:.3f}) between {corr_name.replace('_', ' ')}",
                    'confidence': abs(corr_value),
                    'evidence': [f"Correlation coefficient: {corr_value:.3f}"]
                }
                insights.append(insight)
        
        # Insight 3: Evolutionary Trajectory Insights
        if analysis.evolutionary_trajectories:
            avg_transitions = statistics.mean([t['phase_transitions'] for t in analysis.evolutionary_trajectories])
            
            if avg_transitions > 2:
                insight = {
                    'type': 'evolutionary_activity',
                    'description': f"High evolutionary activity with average {avg_transitions:.1f} phase transitions per civilization",
                    'confidence': min(1.0, avg_transitions / 5.0),
                    'evidence': [f"Average transitions: {avg_transitions:.1f}"]
                }
                insights.append(insight)
        
        # Store insights
        for insight_data in insights:
            if insight_data['confidence'] >= self.introspection_parameters['insight_min_confidence']:
                insight = SystemInsight(
                    id=str(uuid.uuid4()),
                    insight_type=insight_data['type'],
                    description=insight_data['description'],
                    supporting_evidence=insight_data['evidence'],
                    confidence_level=insight_data['confidence'],
                    impact_score=insight_data['confidence'] * 0.8
                )
                
                self.system_insights[insight.id] = insight
                analysis.system_insights.append(insight_data['description'])
                self.introspection_stats['system_insights_generated'] += 1
    
    def _generate_optimization_recommendations(self, analysis: IntrospectionAnalysis):
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Recommendation 1: Governance Optimization
        if analysis.dominance_analysis and 'governance_dominance' in analysis.dominance_analysis:
            gov_dom = analysis.dominance_analysis['governance_dominance']
            if max(gov_dom.values()) < 0.6:  # No clear dominance
                recommendation = {
                    'type': 'governance_optimization',
                    'priority': 'high',
                    'description': "No single governance model dominates - consider hybrid approaches",
                    'action': "Implement adaptive governance systems that can switch between models",
                    'expected_benefit': "Improved adaptability and performance across different conditions"
                }
                recommendations.append(recommendation)
        
        # Recommendation 2: Performance Correlation Optimization
        if analysis.performance_correlations:
            negative_correlations = [(k, v) for k, v in analysis.performance_correlations.items() if v < -0.3]
            
            for corr_name, corr_value in negative_correlations:
                recommendation = {
                    'type': 'correlation_optimization',
                    'priority': 'medium',
                    'description': f"Negative correlation ({corr_value:.3f}) between {corr_name.replace('_', ' ')}",
                    'action': f"Investigate and optimize the relationship between {corr_name.replace('_', ' ')}",
                    'expected_benefit': "Improved overall system performance through better balance"
                }
                recommendations.append(recommendation)
        
        # Recommendation 3: Resource Efficiency Optimization
        if analysis.discovered_patterns:
            resource_efficiencies = [p['resource_efficiency'] for p in analysis.discovered_patterns]
            avg_efficiency = statistics.mean(resource_efficiencies)
            
            if avg_efficiency < 0.5:  # Low efficiency
                recommendation = {
                    'type': 'resource_optimization',
                    'priority': 'high',
                    'description': f"Low average resource efficiency ({avg_efficiency:.3f})",
                    'action': "Implement resource management optimization strategies",
                    'expected_benefit': "Improved resource utilization and overall performance"
                }
                recommendations.append(recommendation)
        
        analysis.optimization_recommendations = recommendations
        self.introspection_stats['optimization_recommendations'] += len(recommendations)
    
    def _calculate_analysis_confidence(self, analysis: IntrospectionAnalysis) -> float:
        """Calculate confidence score for the analysis."""
        confidence_factors = []
        
        # Factor 1: Number of civilizations analyzed
        civ_count = len(analysis.target_civilization_ids)
        civ_confidence = min(1.0, civ_count / 10.0)  # Normalize to 0-1
        confidence_factors.append(civ_confidence)
        
        # Factor 2: Number of patterns discovered
        pattern_count = len(analysis.discovered_patterns)
        pattern_confidence = min(1.0, pattern_count / 5.0)  # Normalize to 0-1
        confidence_factors.append(pattern_confidence)
        
        # Factor 3: Meta-rule confidence
        if analysis.discovered_meta_rules:
            meta_rule_confidences = [rule['confidence'] for rule in analysis.discovered_meta_rules]
            avg_meta_confidence = statistics.mean(meta_rule_confidences)
            confidence_factors.append(avg_meta_confidence)
        
        # Factor 4: System insights confidence
        if analysis.system_insights:
            # analysis.system_insights contains strings (descriptions), not dictionaries
            # Use a default confidence for system insights
            avg_insight_confidence = 0.7  # Default confidence for system insights
            confidence_factors.append(avg_insight_confidence)
        
        # Calculate overall confidence
        if confidence_factors:
            return statistics.mean(confidence_factors)
        else:
            return 0.0
    
    def apply_meta_rule(self, meta_rule_id: str, target_civilization_id: str) -> bool:
        """Apply a discovered meta-rule to a civilization."""
        meta_rule = self.discovered_meta_rules.get(meta_rule_id)
        if not meta_rule:
            return False
        
        civilization = self.core.get_civilization(target_civilization_id)
        if not civilization:
            return False
        
        try:
            # Apply rule based on type
            if meta_rule.rule_type == MetaRuleType.GOVERNANCE_LAW:
                return self._apply_governance_rule(civilization, meta_rule)
            elif meta_rule.rule_type == MetaRuleType.INNOVATION_LAW:
                return self._apply_innovation_rule(civilization, meta_rule)
            elif meta_rule.rule_type == MetaRuleType.RESOURCE_LAW:
                return self._apply_resource_rule(civilization, meta_rule)
            elif meta_rule.rule_type == MetaRuleType.EVOLUTIONARY_LAW:
                return self._apply_evolutionary_rule(civilization, meta_rule)
            else:
                return False
                
        except Exception as e:
            print(f"Error applying meta-rule {meta_rule_id}: {e}")
            return False
    
    def _apply_governance_rule(self, civilization: Civilization, meta_rule: MetaRule) -> bool:
        """Apply a governance meta-rule to a civilization."""
        # Check if conditions are met
        conditions = meta_rule.rule_conditions
        if civilization.total_population < conditions.get('population_threshold', 0):
            return False
        if civilization.stability < conditions.get('stability_threshold', 0):
            return False
        if civilization.complexity < conditions.get('complexity_threshold', 0):
            return False
        
        # Apply effects
        effects = meta_rule.rule_effects
        if 'stability_improvement' in effects:
            civilization.stability = min(1.0, civilization.stability + effects['stability_improvement'])
        if 'decision_efficiency_improvement' in effects:
            civilization.decision_making_efficiency = min(1.0, 
                civilization.decision_making_efficiency + effects['decision_efficiency_improvement'])
        
        # Update meta-rule usage
        meta_rule.usage_count += 1
        meta_rule.last_updated = datetime.now()
        
        return True
    
    def _apply_innovation_rule(self, civilization: Civilization, meta_rule: MetaRule) -> bool:
        """Apply an innovation meta-rule to a civilization."""
        # Check conditions
        conditions = meta_rule.rule_conditions
        if civilization.cooperation_level < conditions.get('cooperation_threshold', 0):
            return False
        if civilization.innovation_capacity < conditions.get('innovation_threshold', 0):
            return False
        
        # Apply effects
        effects = meta_rule.rule_effects
        if 'innovation_boost' in effects:
            civilization.innovation_capacity = min(1.0, civilization.innovation_capacity + effects['innovation_boost'])
        if 'cooperation_boost' in effects:
            civilization.cooperation_level = min(1.0, civilization.cooperation_level + effects['cooperation_boost'])
        
        # Update meta-rule usage
        meta_rule.usage_count += 1
        meta_rule.last_updated = datetime.now()
        
        return True
    
    def _apply_resource_rule(self, civilization: Civilization, meta_rule: MetaRule) -> bool:
        """Apply a resource meta-rule to a civilization."""
        # Check conditions
        conditions = meta_rule.rule_conditions
        if civilization.total_population < conditions.get('population_threshold', 0):
            return False
        
        # Calculate current resource efficiency
        current_efficiency = sum(civilization.resource_efficiency.values()) / len(civilization.resource_efficiency) if civilization.resource_efficiency else 0.0
        if current_efficiency < conditions.get('resource_efficiency_threshold', 0):
            return False
        
        # Apply effects
        effects = meta_rule.rule_effects
        if 'performance_boost' in effects:
            civilization.stability = min(1.0, civilization.stability + effects['performance_boost'])
            civilization.complexity = min(1.0, civilization.complexity + effects['performance_boost'])
        
        # Update meta-rule usage
        meta_rule.usage_count += 1
        meta_rule.last_updated = datetime.now()
        
        return True
    
    def _apply_evolutionary_rule(self, civilization: Civilization, meta_rule: MetaRule) -> bool:
        """Apply an evolutionary meta-rule to a civilization."""
        # Check conditions
        conditions = meta_rule.rule_conditions
        if civilization.complexity < conditions.get('complexity_threshold', 0):
            return False
        if civilization.stability < conditions.get('stability_threshold', 0):
            return False
        
        # Apply effects
        effects = meta_rule.rule_effects
        if 'complexity_penalty' in effects:
            civilization.complexity = max(0.0, civilization.complexity - effects['complexity_penalty'])
        if 'stability_boost' in effects:
            civilization.stability = min(1.0, civilization.stability + effects['stability_boost'])
        
        # Update meta-rule usage
        meta_rule.usage_count += 1
        meta_rule.last_updated = datetime.now()
        
        return True
    
    def get_analysis(self, analysis_id: str) -> Optional[IntrospectionAnalysis]:
        """Get introspection analysis by ID."""
        return self.introspection_analyses.get(analysis_id)
    
    def get_meta_rule(self, meta_rule_id: str) -> Optional[MetaRule]:
        """Get meta-rule by ID."""
        return self.discovered_meta_rules.get(meta_rule_id)
    
    def get_meta_rules_by_type(self, rule_type: MetaRuleType) -> List[MetaRule]:
        """Get meta-rules by type."""
        return [rule for rule in self.discovered_meta_rules.values() if rule.rule_type == rule_type]
    
    def get_high_confidence_meta_rules(self, min_confidence: float = 0.8) -> List[MetaRule]:
        """Get high-confidence meta-rules."""
        return [
            rule for rule in self.discovered_meta_rules.values() 
            if rule.confidence_score >= min_confidence
        ]
    
    def get_system_insights(self) -> List[SystemInsight]:
        """Get all system insights."""
        return list(self.system_insights.values())
    
    def get_introspection_statistics(self) -> Dict[str, Any]:
        """Get introspection layer statistics."""
        return {
            'introspection_stats': self.introspection_stats.copy(),
            'total_analyses': len(self.introspection_analyses),
            'total_meta_rules': len(self.discovered_meta_rules),
            'total_insights': len(self.system_insights),
            'meta_rule_types': {
                rule_type.name: len(self.get_meta_rules_by_type(rule_type))
                for rule_type in list(MetaRuleType)
            },
            'high_confidence_meta_rules': len(self.get_high_confidence_meta_rules()),
            'average_meta_rule_confidence': sum(rule.confidence_score for rule in self.discovered_meta_rules.values()) / max(len(self.discovered_meta_rules), 1),
            'average_insight_confidence': sum(insight.confidence_level for insight in self.system_insights.values()) / max(len(self.system_insights), 1)
        }
