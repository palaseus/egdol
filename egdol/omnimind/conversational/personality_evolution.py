"""
Personality Evolution Module
Handles dynamic refinement of personality behaviors, language styles,
epistemic patterns, and meta-rule incorporation based on meta-learning insights.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .personality_framework import Personality, PersonalityType
from .meta_learning_engine import MetaLearningEngine, LearningInsight, HeuristicUpdate
from .reflexive_audit import ReflexiveAuditModule, PersonalityPerformance


class EvolutionStage(Enum):
    """Stages of personality evolution."""
    INITIAL = auto()
    LEARNING = auto()
    REFINING = auto()
    OPTIMIZING = auto()
    MASTERED = auto()


@dataclass
class PersonalityEvolutionState:
    """State of personality evolution."""
    personality: str
    stage: EvolutionStage
    evolution_score: float
    learned_heuristics: Dict[str, Any]
    language_patterns: Dict[str, float]
    epistemic_patterns: Dict[str, float]
    meta_rules_applied: List[str]
    fallback_threshold: float
    confidence_trend: float
    improvement_rate: float
    last_evolution: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'personality': self.personality,
            'stage': self.stage.name,
            'evolution_score': self.evolution_score,
            'learned_heuristics': self.learned_heuristics,
            'language_patterns': self.language_patterns,
            'epistemic_patterns': self.epistemic_patterns,
            'meta_rules_applied': self.meta_rules_applied,
            'fallback_threshold': self.fallback_threshold,
            'confidence_trend': self.confidence_trend,
            'improvement_rate': self.improvement_rate,
            'last_evolution': self.last_evolution.isoformat()
        }


@dataclass
class EvolutionUpdate:
    """Update to personality evolution."""
    personality: str
    update_type: str
    old_value: Any
    new_value: Any
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


class PersonalityEvolutionEngine:
    """Engine for personality evolution and dynamic refinement."""
    
    def __init__(self, meta_learning_engine: MetaLearningEngine, 
                 audit_module: ReflexiveAuditModule):
        self.meta_learning_engine = meta_learning_engine
        self.audit_module = audit_module
        self.evolution_states: Dict[str, PersonalityEvolutionState] = {}
        self.evolution_history: List[EvolutionUpdate] = []
        
        # Evolution parameters
        self.evolution_thresholds = {
            'stage_advancement': 0.8,
            'heuristic_confidence': 0.7,
            'pattern_stability': 0.6,
            'meta_rule_application': 0.5
        }
        
        self._initialize_personality_evolution_states()
    
    def _initialize_personality_evolution_states(self):
        """Initialize evolution states for all personalities."""
        personalities = ['Strategos', 'Archivist', 'Lawmaker', 'Oracle']
        
        for personality in personalities:
            self.evolution_states[personality] = PersonalityEvolutionState(
                personality=personality,
                stage=EvolutionStage.INITIAL,
                evolution_score=0.0,
                learned_heuristics={},
                language_patterns=self._get_initial_language_patterns(personality),
                epistemic_patterns=self._get_initial_epistemic_patterns(personality),
                meta_rules_applied=[],
                fallback_threshold=0.5,
                confidence_trend=0.0,
                improvement_rate=0.0
            )
    
    def _get_initial_language_patterns(self, personality: str) -> Dict[str, float]:
        """Get initial language patterns for personality."""
        patterns = {
            'Strategos': {
                'military_terminology': 0.8,
                'tactical_language': 0.7,
                'command_structure': 0.9,
                'strategic_thinking': 0.6
            },
            'Archivist': {
                'scholarly_language': 0.8,
                'historical_references': 0.7,
                'archival_terminology': 0.9,
                'philosophical_depth': 0.6
            },
            'Lawmaker': {
                'legal_terminology': 0.8,
                'governance_language': 0.7,
                'structural_thinking': 0.9,
                'principle_based': 0.6
            },
            'Oracle': {
                'mystical_language': 0.8,
                'cosmic_references': 0.7,
                'universal_terminology': 0.9,
                'transcendent_thinking': 0.6
            }
        }
        return patterns.get(personality, {})
    
    def _get_initial_epistemic_patterns(self, personality: str) -> Dict[str, float]:
        """Get initial epistemic patterns for personality."""
        patterns = {
            'Strategos': {
                'tactical_analysis': 0.8,
                'strategic_planning': 0.7,
                'military_reasoning': 0.9,
                'operational_thinking': 0.6
            },
            'Archivist': {
                'historical_analysis': 0.8,
                'scholarly_research': 0.7,
                'philosophical_inquiry': 0.9,
                'cultural_understanding': 0.6
            },
            'Lawmaker': {
                'legal_analysis': 0.8,
                'governance_reasoning': 0.7,
                'structural_thinking': 0.9,
                'principle_application': 0.6
            },
            'Oracle': {
                'cosmic_analysis': 0.8,
                'universal_reasoning': 0.7,
                'transcendent_thinking': 0.9,
                'mystical_insight': 0.6
            }
        }
        return patterns.get(personality, {})
    
    def evolve_personality(self, personality: str, 
                          learning_insights: List[LearningInsight],
                          performance_data: Optional[PersonalityPerformance] = None) -> EvolutionUpdate:
        """
        Evolve a personality based on learning insights and performance data.
        
        Args:
            personality: Name of personality to evolve
            learning_insights: Insights from meta-learning
            performance_data: Optional performance data from audit
            
        Returns:
            EvolutionUpdate describing the evolution
        """
        if personality not in self.evolution_states:
            raise ValueError(f"Unknown personality: {personality}")
        
        evolution_state = self.evolution_states[personality]
        evolution_updates = []
        
        # Update language patterns based on insights
        language_updates = self._evolve_language_patterns(personality, learning_insights)
        evolution_updates.extend(language_updates)
        
        # Update epistemic patterns based on insights
        epistemic_updates = self._evolve_epistemic_patterns(personality, learning_insights)
        evolution_updates.extend(epistemic_updates)
        
        # Update learned heuristics
        heuristic_updates = self._evolve_learned_heuristics(personality, learning_insights)
        evolution_updates.extend(heuristic_updates)
        
        # Update meta-rules applied
        meta_rule_updates = self._evolve_meta_rules(personality, learning_insights)
        evolution_updates.extend(meta_rule_updates)
        
        # Update fallback threshold based on performance
        if performance_data:
            fallback_updates = self._evolve_fallback_threshold(personality, performance_data)
            evolution_updates.extend(fallback_updates)
        
        # Update evolution score and stage
        self._update_evolution_score(personality)
        self._advance_evolution_stage(personality)
        
        # Store evolution updates
        self.evolution_history.extend(evolution_updates)
        
        # Return primary update
        if evolution_updates:
            return evolution_updates[0]
        else:
            return EvolutionUpdate(
                personality=personality,
                update_type='no_change',
                old_value=None,
                new_value=None,
                confidence=0.0,
                reasoning='No significant evolution detected'
            )
    
    def _evolve_language_patterns(self, personality: str, 
                                learning_insights: List[LearningInsight]) -> List[EvolutionUpdate]:
        """Evolve language patterns based on learning insights."""
        updates = []
        evolution_state = self.evolution_states[personality]
        
        # Filter insights for language patterns
        language_insights = [insight for insight in learning_insights 
                           if insight.insight_type == 'response_pattern']
        
        for insight in language_insights:
            pattern = insight.pattern
            confidence = insight.confidence
            
            # Update or add language pattern
            if pattern in evolution_state.language_patterns:
                old_value = evolution_state.language_patterns[pattern]
                new_value = (old_value + confidence) / 2
                evolution_state.language_patterns[pattern] = new_value
                
                if abs(new_value - old_value) > 0.1:
                    update = EvolutionUpdate(
                        personality=personality,
                        update_type='language_pattern_refinement',
                        old_value=old_value,
                        new_value=new_value,
                        confidence=confidence,
                        reasoning=f'Refined {pattern} language pattern based on learning'
                    )
                    updates.append(update)
            else:
                evolution_state.language_patterns[pattern] = confidence
                
                update = EvolutionUpdate(
                    personality=personality,
                    update_type='language_pattern_discovery',
                    old_value=None,
                    new_value=confidence,
                    confidence=confidence,
                    reasoning=f'Discovered new {pattern} language pattern'
                )
                updates.append(update)
        
        return updates
    
    def _evolve_epistemic_patterns(self, personality: str, 
                                 learning_insights: List[LearningInsight]) -> List[EvolutionUpdate]:
        """Evolve epistemic patterns based on learning insights."""
        updates = []
        evolution_state = self.evolution_states[personality]
        
        # Filter insights for reasoning patterns
        reasoning_insights = [insight for insight in learning_insights 
                            if insight.insight_type == 'reasoning_pattern']
        
        for insight in reasoning_insights:
            pattern = insight.pattern
            confidence = insight.confidence
            
            # Update or add epistemic pattern
            if pattern in evolution_state.epistemic_patterns:
                old_value = evolution_state.epistemic_patterns[pattern]
                new_value = (old_value + confidence) / 2
                evolution_state.epistemic_patterns[pattern] = new_value
                
                if abs(new_value - old_value) > 0.1:
                    update = EvolutionUpdate(
                        personality=personality,
                        update_type='epistemic_pattern_refinement',
                        old_value=old_value,
                        new_value=new_value,
                        confidence=confidence,
                        reasoning=f'Refined {pattern} epistemic pattern based on learning'
                    )
                    updates.append(update)
            else:
                evolution_state.epistemic_patterns[pattern] = confidence
                
                update = EvolutionUpdate(
                    personality=personality,
                    update_type='epistemic_pattern_discovery',
                    old_value=None,
                    new_value=confidence,
                    confidence=confidence,
                    reasoning=f'Discovered new {pattern} epistemic pattern'
                )
                updates.append(update)
        
        return updates
    
    def _evolve_learned_heuristics(self, personality: str, 
                                 learning_insights: List[LearningInsight]) -> List[EvolutionUpdate]:
        """Evolve learned heuristics based on learning insights."""
        updates = []
        evolution_state = self.evolution_states[personality]
        
        # Get heuristic updates from meta-learning engine
        heuristic_updates = self.meta_learning_engine.get_heuristic_updates(personality)
        
        for heuristic_update in heuristic_updates:
            heuristic_name = heuristic_update.heuristic_name
            new_value = heuristic_update.new_value
            confidence = heuristic_update.confidence
            
            # Update learned heuristics
            if heuristic_name in evolution_state.learned_heuristics:
                old_value = evolution_state.learned_heuristics[heuristic_name]
                evolution_state.learned_heuristics[heuristic_name] = new_value
                
                update = EvolutionUpdate(
                    personality=personality,
                    update_type='heuristic_update',
                    old_value=old_value,
                    new_value=new_value,
                    confidence=confidence,
                    reasoning=f'Updated {heuristic_name} heuristic based on meta-learning'
                )
                updates.append(update)
            else:
                evolution_state.learned_heuristics[heuristic_name] = new_value
                
                update = EvolutionUpdate(
                    personality=personality,
                    update_type='heuristic_discovery',
                    old_value=None,
                    new_value=new_value,
                    confidence=confidence,
                    reasoning=f'Discovered new {heuristic_name} heuristic'
                )
                updates.append(update)
        
        return updates
    
    def _evolve_meta_rules(self, personality: str, 
                          learning_insights: List[LearningInsight]) -> List[EvolutionUpdate]:
        """Evolve meta-rules based on learning insights."""
        updates = []
        evolution_state = self.evolution_states[personality]
        
        # Get meta-rule discoveries from meta-learning engine
        meta_rule_discoveries = self.meta_learning_engine.get_meta_rule_discoveries()
        
        for discovery in meta_rule_discoveries:
            if personality in discovery.source_personalities or 'all_personalities' in discovery.applicability:
                rule_name = discovery.rule_name
                
                if rule_name not in evolution_state.meta_rules_applied:
                    evolution_state.meta_rules_applied.append(rule_name)
                    
                    update = EvolutionUpdate(
                        personality=personality,
                        update_type='meta_rule_incorporation',
                        old_value=None,
                        new_value=rule_name,
                        confidence=discovery.confidence,
                        reasoning=f'Incorporated {rule_name} meta-rule based on discovery'
                    )
                    updates.append(update)
        
        return updates
    
    def _evolve_fallback_threshold(self, personality: str, 
                                 performance_data: PersonalityPerformance) -> List[EvolutionUpdate]:
        """Evolve fallback threshold based on performance data."""
        updates = []
        evolution_state = self.evolution_states[personality]
        
        # Calculate new fallback threshold based on performance
        fallback_ratio = performance_data.fallback_usage / max(1, performance_data.total_turns)
        confidence_score = performance_data.average_confidence
        
        # Adjust threshold based on performance
        if confidence_score > 0.8 and fallback_ratio < 0.3:
            # High confidence, low fallback - can reduce threshold
            old_threshold = evolution_state.fallback_threshold
            new_threshold = max(0.1, old_threshold - 0.05)
            evolution_state.fallback_threshold = new_threshold
            
            update = EvolutionUpdate(
                personality=personality,
                update_type='fallback_threshold_reduction',
                old_value=old_threshold,
                new_value=new_threshold,
                confidence=confidence_score,
                reasoning='Reduced fallback threshold due to high performance'
            )
            updates.append(update)
        
        elif confidence_score < 0.5 and fallback_ratio > 0.7:
            # Low confidence, high fallback - should increase threshold
            old_threshold = evolution_state.fallback_threshold
            new_threshold = min(0.9, old_threshold + 0.05)
            evolution_state.fallback_threshold = new_threshold
            
            update = EvolutionUpdate(
                personality=personality,
                update_type='fallback_threshold_increase',
                old_value=old_threshold,
                new_value=new_threshold,
                confidence=1.0 - confidence_score,
                reasoning='Increased fallback threshold due to low performance'
            )
            updates.append(update)
        
        return updates
    
    def _update_evolution_score(self, personality: str):
        """Update evolution score for personality."""
        evolution_state = self.evolution_states[personality]
        
        # Calculate evolution score based on various factors
        language_diversity = len(evolution_state.language_patterns)
        epistemic_diversity = len(evolution_state.epistemic_patterns)
        heuristic_count = len(evolution_state.learned_heuristics)
        meta_rule_count = len(evolution_state.meta_rules_applied)
        
        # Normalize scores
        language_score = min(1.0, language_diversity / 10.0)
        epistemic_score = min(1.0, epistemic_diversity / 10.0)
        heuristic_score = min(1.0, heuristic_count / 20.0)
        meta_rule_score = min(1.0, meta_rule_count / 10.0)
        
        # Calculate weighted evolution score
        evolution_state.evolution_score = (
            language_score * 0.25 +
            epistemic_score * 0.25 +
            heuristic_score * 0.25 +
            meta_rule_score * 0.25
        )
        
        # Update confidence trend
        evolution_state.confidence_trend = self._calculate_confidence_trend(personality)
        
        # Update improvement rate
        evolution_state.improvement_rate = self._calculate_improvement_rate(personality)
    
    def _calculate_confidence_trend(self, personality: str) -> float:
        """Calculate confidence trend for personality."""
        # Get recent performance data
        performance = self.audit_module.get_personality_performance(personality)
        if not performance:
            return 0.0
        
        # Calculate trend based on recent vs overall performance
        if performance.total_turns < 5:
            return 0.0
        
        # Simple trend calculation
        recent_turns = min(5, performance.total_turns)
        recent_success_rate = performance.successful_turns / performance.total_turns
        
        return recent_success_rate - 0.5  # Positive = improving, negative = declining
    
    def _calculate_improvement_rate(self, personality: str) -> float:
        """Calculate improvement rate for personality."""
        # Get recent evolution updates
        recent_updates = [update for update in self.evolution_history 
                         if update.personality == personality][-10:]
        
        if len(recent_updates) < 3:
            return 0.0
        
        # Calculate improvement rate based on update frequency and confidence
        total_confidence = sum(update.confidence for update in recent_updates)
        avg_confidence = total_confidence / len(recent_updates)
        
        return min(1.0, avg_confidence)
    
    def _advance_evolution_stage(self, personality: str):
        """Advance evolution stage if criteria are met."""
        evolution_state = self.evolution_states[personality]
        current_stage = evolution_state.stage
        
        # Check if ready to advance
        if current_stage == EvolutionStage.INITIAL and evolution_state.evolution_score > 0.3:
            evolution_state.stage = EvolutionStage.LEARNING
        elif current_stage == EvolutionStage.LEARNING and evolution_state.evolution_score > 0.5:
            evolution_state.stage = EvolutionStage.REFINING
        elif current_stage == EvolutionStage.REFINING and evolution_state.evolution_score > 0.7:
            evolution_state.stage = EvolutionStage.OPTIMIZING
        elif current_stage == EvolutionStage.OPTIMIZING and evolution_state.evolution_score > 0.9:
            evolution_state.stage = EvolutionStage.MASTERED
    
    def get_evolution_state(self, personality: str) -> Optional[PersonalityEvolutionState]:
        """Get evolution state for personality."""
        return self.evolution_states.get(personality)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of all personality evolution."""
        summary = {
            'total_personalities': len(self.evolution_states),
            'evolution_stages': {},
            'average_evolution_score': 0.0,
            'total_evolution_updates': len(self.evolution_history),
            'personalities_by_stage': {}
        }
        
        # Calculate stage distribution
        stage_counts = {}
        total_score = 0.0
        
        for personality, state in self.evolution_states.items():
            stage = state.stage.name
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            total_score += state.evolution_score
            
            if stage not in summary['personalities_by_stage']:
                summary['personalities_by_stage'][stage] = []
            summary['personalities_by_stage'][stage].append(personality)
        
        summary['evolution_stages'] = stage_counts
        summary['average_evolution_score'] = total_score / len(self.evolution_states)
        
        return summary
    
    def get_personality_evolution_report(self, personality: str) -> Dict[str, Any]:
        """Get detailed evolution report for personality."""
        evolution_state = self.evolution_states.get(personality)
        if not evolution_state:
            return {}
        
        # Get recent evolution updates
        recent_updates = [update for update in self.evolution_history 
                         if update.personality == personality][-20:]
        
        # Get performance data
        performance = self.audit_module.get_personality_performance(personality)
        
        return {
            'personality': personality,
            'evolution_state': evolution_state.to_dict(),
            'recent_updates': [update.__dict__ for update in recent_updates],
            'performance_data': performance.__dict__ if performance else {},
            'evolution_metrics': {
                'total_updates': len([u for u in self.evolution_history if u.personality == personality]),
                'language_patterns': len(evolution_state.language_patterns),
                'epistemic_patterns': len(evolution_state.epistemic_patterns),
                'learned_heuristics': len(evolution_state.learned_heuristics),
                'meta_rules_applied': len(evolution_state.meta_rules_applied)
            }
        }
    
    def apply_evolution_to_personality(self, personality: str) -> Dict[str, Any]:
        """Apply evolved patterns to personality behavior."""
        evolution_state = self.evolution_states.get(personality)
        if not evolution_state:
            return {}
        
        # Create updated personality configuration
        updated_config = {
            'personality': personality,
            'stage': evolution_state.stage.name,
            'evolution_score': evolution_state.evolution_score,
            'language_patterns': evolution_state.language_patterns,
            'epistemic_patterns': evolution_state.epistemic_patterns,
            'learned_heuristics': evolution_state.learned_heuristics,
            'meta_rules_applied': evolution_state.meta_rules_applied,
            'fallback_threshold': evolution_state.fallback_threshold,
            'confidence_trend': evolution_state.confidence_trend,
            'improvement_rate': evolution_state.improvement_rate
        }
        
        return updated_config
