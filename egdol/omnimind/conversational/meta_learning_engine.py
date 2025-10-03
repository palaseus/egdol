"""
Meta-Learning Engine
Consumes conversation logs, reasoning traces, and introspection outputs to
continuously improve heuristics, meta-rules, and personality behaviors.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .reflexive_audit import ReflexiveAuditModule, AuditResult, PersonalityPerformance
from .context_intent_resolver import ContextIntentResolver, ReasoningContext
from .personality_framework import Personality, PersonalityType
from .personality_fallbacks import PersonalityFallbackReasoner


class LearningStrategy(Enum):
    """Types of learning strategies."""
    REINFORCEMENT = auto()
    SUPERVISED = auto()
    HEURISTIC_REFINEMENT = auto()
    META_RULE_DISCOVERY = auto()
    CROSS_PERSONALITY = auto()


@dataclass
class LearningInsight:
    """Insight gained from meta-learning."""
    insight_type: str
    personality: str
    pattern: str
    confidence: float
    applicable_to: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'insight_type': self.insight_type,
            'personality': self.personality,
            'pattern': self.pattern,
            'confidence': self.confidence,
            'applicable_to': self.applicable_to,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class HeuristicUpdate:
    """Update to a heuristic based on learning."""
    heuristic_name: str
    personality: str
    old_value: Any
    new_value: Any
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetaRuleDiscovery:
    """New meta-rule discovered through conversation analysis."""
    rule_name: str
    pattern: str
    confidence: float
    source_personalities: List[str]
    applicable_to: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MetaLearningEngine:
    """Engine for meta-learning and continuous improvement."""
    
    def __init__(self):
        self.audit_module = ReflexiveAuditModule()
        self.learning_insights: List[LearningInsight] = []
        self.heuristic_updates: List[HeuristicUpdate] = []
        self.meta_rule_discoveries: List[MetaRuleDiscovery] = []
        self.learning_history: List[Dict[str, Any]] = []
        
        # Learning state
        self.personality_learning_state: Dict[str, Dict[str, Any]] = {}
        self.cross_personality_insights: Dict[str, List[str]] = {}
        self.fallback_reduction_targets: Dict[str, float] = {}
        
        self._initialize_learning_parameters()
    
    def _initialize_learning_parameters(self):
        """Initialize learning parameters and thresholds."""
        self.learning_thresholds = {
            'confidence_improvement': 0.1,
            'fallback_reduction': 0.05,
            'reasoning_accuracy_improvement': 0.08,
            'pattern_confidence': 0.7
        }
        
        self.learning_rates = {
            'heuristic_refinement': 0.1,
            'meta_rule_discovery': 0.05,
            'cross_personality': 0.15,
            'fallback_optimization': 0.2
        }
    
    def process_conversation_batch(self, conversation_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of conversation logs for meta-learning.
        
        Args:
            conversation_logs: List of conversation turn data
            
        Returns:
            Dictionary with learning results and updates
        """
        learning_results = {
            'insights_generated': 0,
            'heuristics_updated': 0,
            'meta_rules_discovered': 0,
            'personalities_improved': [],
            'cross_personality_insights': 0
        }
        
        # Process each conversation turn
        for log in conversation_logs:
            turn_data = self._extract_turn_data(log)
            if not turn_data:
                continue
            
            # Generate insights from this turn
            insights = self._generate_insights_from_turn(turn_data)
            learning_results['insights_generated'] += len(insights)
            
            # Apply learning strategies
            heuristic_updates = self._apply_heuristic_learning(turn_data, insights)
            learning_results['heuristics_updated'] += len(heuristic_updates)
            
            # Discover meta-rules
            meta_rules = self._discover_meta_rules(turn_data, insights)
            learning_results['meta_rules_discovered'] += len(meta_rules)
            
            # Cross-personality learning
            cross_insights = self._apply_cross_personality_learning(turn_data, insights)
            learning_results['cross_personality_insights'] += len(cross_insights)
        
        # Update personality learning states
        self._update_personality_learning_states()
        
        # Generate improvement recommendations
        recommendations = self._generate_improvement_recommendations()
        
        learning_results['recommendations'] = recommendations
        learning_results['learning_summary'] = self._generate_learning_summary()
        
        return learning_results
    
    def _extract_turn_data(self, log: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract relevant data from conversation log."""
        try:
            return {
                'turn_id': log.get('turn_id'),
                'personality': log.get('personality'),
                'user_input': log.get('user_input'),
                'system_response': log.get('system_response'),
                'reasoning_trace': log.get('reasoning_trace'),
                'context': log.get('context'),
                'confidence_score': log.get('confidence_score', 0.5),
                'fallback_used': log.get('fallback_used', False),
                'timestamp': log.get('timestamp')
            }
        except Exception:
            return None
    
    def _generate_insights_from_turn(self, turn_data: Dict[str, Any]) -> List[LearningInsight]:
        """Generate learning insights from a conversation turn."""
        insights = []
        personality = turn_data['personality']
        
        # Analyze response patterns
        response_patterns = self._analyze_response_patterns(turn_data)
        for pattern, confidence in response_patterns.items():
            insight = LearningInsight(
                insight_type='response_pattern',
                personality=personality,
                pattern=pattern,
                confidence=confidence,
                applicable_to=['response_generation'],
                metadata={'turn_id': turn_data['turn_id']}
            )
            insights.append(insight)
        
        # Analyze reasoning patterns
        reasoning_patterns = self._analyze_reasoning_patterns(turn_data)
        for pattern, confidence in reasoning_patterns.items():
            insight = LearningInsight(
                insight_type='reasoning_pattern',
                personality=personality,
                pattern=pattern,
                confidence=confidence,
                applicable_to=['reasoning_engine'],
                metadata={'turn_id': turn_data['turn_id']}
            )
            insights.append(insight)
        
        # Analyze fallback patterns
        if turn_data.get('fallback_used', False):
            fallback_patterns = self._analyze_fallback_patterns(turn_data)
            for pattern, confidence in fallback_patterns.items():
                insight = LearningInsight(
                    insight_type='fallback_pattern',
                    personality=personality,
                    pattern=pattern,
                    confidence=confidence,
                    applicable_to=['fallback_optimization'],
                    metadata={'turn_id': turn_data['turn_id']}
                )
                insights.append(insight)
        
        # Store insights
        self.learning_insights.extend(insights)
        
        return insights
    
    def _analyze_response_patterns(self, turn_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze patterns in system responses."""
        patterns = {}
        response = turn_data.get('system_response', '').lower()
        personality = turn_data.get('personality', '')
        
        # Personality-specific patterns
        if personality == 'Strategos':
            if 'commander' in response:
                patterns['military_terminology'] = 0.8
            if 'tactical' in response or 'strategic' in response:
                patterns['strategic_language'] = 0.9
        
        elif personality == 'Archivist':
            if 'archives' in response or 'historical' in response:
                patterns['scholarly_language'] = 0.8
            if 'ancient' in response or 'chronicles' in response:
                patterns['historical_references'] = 0.9
        
        elif personality == 'Lawmaker':
            if 'governance' in response or 'legal' in response:
                patterns['legal_terminology'] = 0.8
            if 'principles' in response or 'framework' in response:
                patterns['structural_language'] = 0.9
        
        elif personality == 'Oracle':
            if 'cosmic' in response or 'universal' in response:
                patterns['mystical_language'] = 0.8
            if 'tapestry' in response or 'infinite' in response:
                patterns['cosmic_references'] = 0.9
        
        # General patterns
        if len(response.split()) > 20:
            patterns['detailed_response'] = 0.7
        if '?' in turn_data.get('user_input', ''):
            patterns['question_handling'] = 0.6
        
        return patterns
    
    def _analyze_reasoning_patterns(self, turn_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze patterns in reasoning traces."""
        patterns = {}
        reasoning_trace = turn_data.get('reasoning_trace', {})
        
        if not reasoning_trace:
            patterns['no_reasoning_trace'] = 1.0
            return patterns
        
        # Analyze reasoning steps
        if 'processing_steps' in reasoning_trace:
            steps = reasoning_trace['processing_steps']
            patterns['multi_step_reasoning'] = min(1.0, len(steps) * 0.2)
        
        # Analyze meta-insights
        if 'meta_insights' in reasoning_trace:
            insights = reasoning_trace['meta_insights']
            patterns['meta_rule_application'] = min(1.0, len(insights) * 0.3)
        
        # Analyze confidence
        if 'confidence' in reasoning_trace:
            confidence = reasoning_trace['confidence']
            if confidence > 0.8:
                patterns['high_confidence_reasoning'] = 0.9
            elif confidence < 0.4:
                patterns['low_confidence_reasoning'] = 0.8
        
        return patterns
    
    def _analyze_fallback_patterns(self, turn_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze patterns in fallback usage."""
        patterns = {}
        response = turn_data.get('system_response', '').lower()
        personality = turn_data.get('personality', '')
        
        # Identify fallback triggers
        if 'i understand' in response:
            patterns['generic_fallback'] = 0.8
        if 'let me try' in response:
            patterns['retry_fallback'] = 0.7
        if 'alternative' in response:
            patterns['alternative_fallback'] = 0.6
        
        # Personality-specific fallback patterns
        if personality == 'Strategos' and 'commander' in response:
            patterns['tactical_fallback'] = 0.9
        elif personality == 'Archivist' and 'archives' in response:
            patterns['scholarly_fallback'] = 0.9
        elif personality == 'Lawmaker' and 'governance' in response:
            patterns['legal_fallback'] = 0.9
        elif personality == 'Oracle' and 'cosmic' in response:
            patterns['mystical_fallback'] = 0.9
        
        return patterns
    
    def _apply_heuristic_learning(self, turn_data: Dict[str, Any], 
                                insights: List[LearningInsight]) -> List[HeuristicUpdate]:
        """Apply heuristic learning based on insights."""
        updates = []
        personality = turn_data['personality']
        
        # Initialize personality learning state if needed
        if personality not in self.personality_learning_state:
            self.personality_learning_state[personality] = {
                'response_patterns': {},
                'reasoning_patterns': {},
                'fallback_patterns': {},
                'confidence_trend': 0.0,
                'improvement_rate': 0.0
            }
        
        # Update response patterns
        for insight in insights:
            if insight.insight_type == 'response_pattern':
                self._update_response_heuristics(personality, insight, updates)
            elif insight.insight_type == 'reasoning_pattern':
                self._update_reasoning_heuristics(personality, insight, updates)
            elif insight.insight_type == 'fallback_pattern':
                self._update_fallback_heuristics(personality, insight, updates)
        
        return updates
    
    def _update_response_heuristics(self, personality: str, insight: LearningInsight, 
                                   updates: List[HeuristicUpdate]):
        """Update response generation heuristics."""
        pattern = insight.pattern
        confidence = insight.confidence
        
        # Update personality-specific response patterns
        if personality not in self.personality_learning_state:
            return
        
        current_patterns = self.personality_learning_state[personality]['response_patterns']
        
        if pattern in current_patterns:
            # Refine existing pattern
            old_confidence = current_patterns[pattern]
            new_confidence = (old_confidence + confidence) / 2
            current_patterns[pattern] = new_confidence
            
            if abs(new_confidence - old_confidence) > 0.1:
                update = HeuristicUpdate(
                    heuristic_name=f'response_pattern_{pattern}',
                    personality=personality,
                    old_value=old_confidence,
                    new_value=new_confidence,
                    confidence=confidence,
                    reasoning=f'Refined {pattern} pattern based on conversation analysis'
                )
                updates.append(update)
        else:
            # Add new pattern
            current_patterns[pattern] = confidence
            
            update = HeuristicUpdate(
                heuristic_name=f'response_pattern_{pattern}',
                personality=personality,
                old_value=None,
                new_value=confidence,
                confidence=confidence,
                reasoning=f'Discovered new {pattern} pattern'
            )
            updates.append(update)
    
    def _update_reasoning_heuristics(self, personality: str, insight: LearningInsight, 
                                   updates: List[HeuristicUpdate]):
        """Update reasoning heuristics."""
        pattern = insight.pattern
        confidence = insight.confidence
        
        if personality not in self.personality_learning_state:
            return
        
        current_patterns = self.personality_learning_state[personality]['reasoning_patterns']
        
        if pattern in current_patterns:
            old_confidence = current_patterns[pattern]
            new_confidence = (old_confidence + confidence) / 2
            current_patterns[pattern] = new_confidence
            
            if abs(new_confidence - old_confidence) > 0.1:
                update = HeuristicUpdate(
                    heuristic_name=f'reasoning_pattern_{pattern}',
                    personality=personality,
                    old_value=old_confidence,
                    new_value=new_confidence,
                    confidence=confidence,
                    reasoning=f'Refined {pattern} reasoning pattern'
                )
                updates.append(update)
        else:
            current_patterns[pattern] = confidence
            
            update = HeuristicUpdate(
                heuristic_name=f'reasoning_pattern_{pattern}',
                personality=personality,
                old_value=None,
                new_value=confidence,
                confidence=confidence,
                reasoning=f'Discovered new {pattern} reasoning pattern'
            )
            updates.append(update)
    
    def _update_fallback_heuristics(self, personality: str, insight: LearningInsight, 
                                   updates: List[HeuristicUpdate]):
        """Update fallback heuristics."""
        pattern = insight.pattern
        confidence = insight.confidence
        
        if personality not in self.personality_learning_state:
            return
        
        current_patterns = self.personality_learning_state[personality]['fallback_patterns']
        
        # Track fallback patterns for optimization
        if pattern in current_patterns:
            old_confidence = current_patterns[pattern]
            new_confidence = (old_confidence + confidence) / 2
            current_patterns[pattern] = new_confidence
        else:
            current_patterns[pattern] = confidence
        
        # Set fallback reduction target
        if personality not in self.fallback_reduction_targets:
            self.fallback_reduction_targets[personality] = 0.5
        
        # Reduce fallback target if pattern is well-established
        if confidence > 0.8:
            current_target = self.fallback_reduction_targets[personality]
            new_target = max(0.1, current_target - 0.05)
            self.fallback_reduction_targets[personality] = new_target
            
            update = HeuristicUpdate(
                heuristic_name='fallback_reduction_target',
                personality=personality,
                old_value=current_target,
                new_value=new_target,
                confidence=confidence,
                reasoning=f'Reduced fallback target due to strong {pattern} pattern'
            )
            updates.append(update)
    
    def _discover_meta_rules(self, turn_data: Dict[str, Any], 
                           insights: List[LearningInsight]) -> List[MetaRuleDiscovery]:
        """Discover new meta-rules from conversation analysis."""
        discoveries = []
        personality = turn_data['personality']
        
        # Analyze for cross-personality patterns
        cross_patterns = self._analyze_cross_personality_patterns(turn_data, insights)
        
        for pattern, confidence in cross_patterns.items():
            if confidence > self.learning_thresholds['pattern_confidence']:
                discovery = MetaRuleDiscovery(
                    rule_name=f'cross_personality_{pattern}',
                    pattern=pattern,
                    confidence=confidence,
                    source_personalities=[personality],
                    applicable_to=['all_personalities'],
                    metadata={'turn_id': turn_data['turn_id']}
                )
                discoveries.append(discovery)
        
        # Analyze for universal patterns
        universal_patterns = self._analyze_universal_patterns(turn_data, insights)
        
        for pattern, confidence in universal_patterns.items():
            if confidence > self.learning_thresholds['pattern_confidence']:
                discovery = MetaRuleDiscovery(
                    rule_name=f'universal_{pattern}',
                    pattern=pattern,
                    confidence=confidence,
                    source_personalities=[personality],
                    applicable_to=['all_personalities'],
                    metadata={'turn_id': turn_data['turn_id']}
                )
                discoveries.append(discovery)
        
        # Store discoveries
        self.meta_rule_discoveries.extend(discoveries)
        
        return discoveries
    
    def _analyze_cross_personality_patterns(self, turn_data: Dict[str, Any], 
                                         insights: List[LearningInsight]) -> Dict[str, float]:
        """Analyze patterns that could apply across personalities."""
        patterns = {}
        
        # Look for successful response patterns that could be adapted
        for insight in insights:
            if insight.insight_type == 'response_pattern' and insight.confidence > 0.7:
                pattern_name = f'adaptable_{insight.pattern}'
                patterns[pattern_name] = insight.confidence * 0.8
        
        return patterns
    
    def _analyze_universal_patterns(self, turn_data: Dict[str, Any], 
                                  insights: List[LearningInsight]) -> Dict[str, float]:
        """Analyze patterns that apply universally."""
        patterns = {}
        
        # Look for universal reasoning patterns
        reasoning_trace = turn_data.get('reasoning_trace', {})
        if reasoning_trace and 'meta_insights' in reasoning_trace:
            meta_insights = reasoning_trace['meta_insights']
            if len(meta_insights) > 0:
                patterns['meta_insight_application'] = 0.8
        
        # Look for confidence patterns
        confidence = turn_data.get('confidence_score', 0.5)
        if confidence > 0.8:
            patterns['high_confidence_handling'] = 0.9
        elif confidence < 0.3:
            patterns['low_confidence_handling'] = 0.7
        
        return patterns
    
    def _apply_cross_personality_learning(self, turn_data: Dict[str, Any], 
                                        insights: List[LearningInsight]) -> List[str]:
        """Apply cross-personality learning."""
        cross_insights = []
        personality = turn_data['personality']
        
        # Share insights between personalities
        for insight in insights:
            if insight.confidence > 0.8:
                # This insight could be useful for other personalities
                applicable_personalities = [p for p in ['Strategos', 'Archivist', 'Lawmaker', 'Oracle'] 
                                          if p != personality]
                
                for other_personality in applicable_personalities:
                    if other_personality not in self.cross_personality_insights:
                        self.cross_personality_insights[other_personality] = []
                    
                    cross_insight = f"{insight.pattern} (from {personality})"
                    self.cross_personality_insights[other_personality].append(cross_insight)
                    cross_insights.append(cross_insight)
        
        return cross_insights
    
    def _update_personality_learning_states(self):
        """Update personality learning states based on accumulated data."""
        for personality, state in self.personality_learning_state.items():
            # Calculate confidence trend
            recent_insights = [insight for insight in self.learning_insights 
                             if insight.personality == personality][-10:]
            
            if len(recent_insights) >= 5:
                recent_confidence = sum(insight.confidence for insight in recent_insights) / len(recent_insights)
                state['confidence_trend'] = recent_confidence - 0.5
            
            # Calculate improvement rate
            if personality in self.fallback_reduction_targets:
                current_target = self.fallback_reduction_targets[personality]
                state['improvement_rate'] = 0.5 - current_target  # Higher rate = better improvement
    
    def _generate_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Generate improvement recommendations based on learning."""
        recommendations = []
        
        for personality, state in self.personality_learning_state.items():
            # Confidence improvement recommendations
            if state['confidence_trend'] < 0:
                recommendations.append({
                    'personality': personality,
                    'area': 'confidence',
                    'priority': 'high',
                    'recommendation': 'Improve response confidence through better reasoning',
                    'specific_actions': [
                        'Strengthen reasoning trace generation',
                        'Apply more meta-rules in responses',
                        'Improve context-intent alignment'
                    ]
                })
            
            # Fallback reduction recommendations
            if personality in self.fallback_reduction_targets:
                target = self.fallback_reduction_targets[personality]
                if target > 0.3:
                    recommendations.append({
                        'personality': personality,
                        'area': 'fallback_reduction',
                        'priority': 'medium',
                        'recommendation': f'Reduce fallback usage to {target:.1%}',
                        'specific_actions': [
                            'Improve primary reasoning accuracy',
                            'Enhance context resolution',
                            'Apply learned patterns more effectively'
                        ]
                    })
            
            # Cross-personality learning recommendations
            if personality in self.cross_personality_insights:
                cross_insights = self.cross_personality_insights[personality]
                if len(cross_insights) > 0:
                    recommendations.append({
                        'personality': personality,
                        'area': 'cross_personality_learning',
                        'priority': 'low',
                        'recommendation': f'Apply {len(cross_insights)} cross-personality insights',
                        'specific_actions': [
                            'Incorporate successful patterns from other personalities',
                            'Adapt reasoning strategies from high-performing personalities',
                            'Share meta-rules across personality boundaries'
                        ]
                    })
        
        return recommendations
    
    def _generate_learning_summary(self) -> Dict[str, Any]:
        """Generate summary of learning activities."""
        return {
            'total_insights': len(self.learning_insights),
            'total_heuristic_updates': len(self.heuristic_updates),
            'total_meta_rule_discoveries': len(self.meta_rule_discoveries),
            'personalities_with_learning': list(self.personality_learning_state.keys()),
            'cross_personality_insights': sum(len(insights) for insights in self.cross_personality_insights.values()),
            'fallback_reduction_targets': self.fallback_reduction_targets,
            'learning_effectiveness': self._calculate_learning_effectiveness()
        }
    
    def _calculate_learning_effectiveness(self) -> float:
        """Calculate overall learning effectiveness."""
        if not self.learning_insights:
            return 0.0
        
        # Calculate based on insight quality and diversity
        avg_confidence = sum(insight.confidence for insight in self.learning_insights) / len(self.learning_insights)
        
        # Calculate diversity (number of unique patterns)
        unique_patterns = len(set(insight.pattern for insight in self.learning_insights))
        pattern_diversity = min(1.0, unique_patterns / 10.0)  # Normalize to 0-1
        
        # Calculate cross-personality learning
        cross_learning = len(self.cross_personality_insights) / 4.0  # 4 personalities
        
        # Combine metrics
        effectiveness = (avg_confidence * 0.4 + pattern_diversity * 0.3 + cross_learning * 0.3)
        
        return min(1.0, effectiveness)
    
    def get_learning_insights(self, personality: Optional[str] = None) -> List[LearningInsight]:
        """Get learning insights, optionally filtered by personality."""
        if personality:
            return [insight for insight in self.learning_insights if insight.personality == personality]
        return self.learning_insights
    
    def get_heuristic_updates(self, personality: Optional[str] = None) -> List[HeuristicUpdate]:
        """Get heuristic updates, optionally filtered by personality."""
        if personality:
            return [update for update in self.heuristic_updates if update.personality == personality]
        return self.heuristic_updates
    
    def get_meta_rule_discoveries(self) -> List[MetaRuleDiscovery]:
        """Get discovered meta-rules."""
        return self.meta_rule_discoveries
    
    def clear_learning_data(self):
        """Clear all learning data (for testing or reset)."""
        self.learning_insights.clear()
        self.heuristic_updates.clear()
        self.meta_rule_discoveries.clear()
        self.learning_history.clear()
        self.personality_learning_state.clear()
        self.cross_personality_insights.clear()
        self.fallback_reduction_targets.clear()
