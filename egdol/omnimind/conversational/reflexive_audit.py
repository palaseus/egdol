"""
Reflexive Conversation Audit Module
Analyzes conversation turns, reasoning traces, and performance to identify
improvement opportunities and track personality-specific metrics.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .conversation_state import ConversationTurn, ConversationState
from .context_intent_resolver import ReasoningContext
from .personality_framework import Personality, PersonalityType


class AuditMetric(Enum):
    """Types of audit metrics."""
    RESPONSE_QUALITY = auto()
    REASONING_ACCURACY = auto()
    FALLBACK_USAGE = auto()
    CONTEXT_ALIGNMENT = auto()
    PERSONALITY_CONSISTENCY = auto()
    META_RULE_APPLICATION = auto()


@dataclass
class AuditResult:
    """Result of conversation turn audit."""
    turn_id: str
    personality: str
    metrics: Dict[AuditMetric, float]
    gaps_identified: List[str]
    improvement_suggestions: List[str]
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'turn_id': self.turn_id,
            'personality': self.personality,
            'metrics': {metric.name: score for metric, score in self.metrics.items()},
            'gaps_identified': self.gaps_identified,
            'improvement_suggestions': self.improvement_suggestions,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PersonalityPerformance:
    """Performance tracking for a specific personality."""
    personality: str
    total_turns: int = 0
    successful_turns: int = 0
    fallback_usage: int = 0
    average_confidence: float = 0.0
    reasoning_accuracy: float = 0.0
    context_alignment: float = 0.0
    meta_rule_applications: int = 0
    improvement_trend: float = 0.0
    
    def update_from_audit(self, audit_result: AuditResult):
        """Update performance metrics from audit result."""
        self.total_turns += 1
        if audit_result.confidence_score > 0.7:
            self.successful_turns += 1
        
        # Update running averages
        self.average_confidence = (
            (self.average_confidence * (self.total_turns - 1) + audit_result.confidence_score) 
            / self.total_turns
        )
        
        # Update other metrics
        if audit_result.metrics.get(AuditMetric.FALLBACK_USAGE, 0) > 0.5:
            self.fallback_usage += 1
        
        self.reasoning_accuracy = (
            (self.reasoning_accuracy * (self.total_turns - 1) + 
             audit_result.metrics.get(AuditMetric.REASONING_ACCURACY, 0.5)) 
            / self.total_turns
        )
        
        self.context_alignment = (
            (self.context_alignment * (self.total_turns - 1) + 
             audit_result.metrics.get(AuditMetric.CONTEXT_ALIGNMENT, 0.5)) 
            / self.total_turns
        )
        
        if audit_result.metrics.get(AuditMetric.META_RULE_APPLICATION, 0) > 0.5:
            self.meta_rule_applications += 1


class ReflexiveAuditModule:
    """Audits conversation turns and tracks performance metrics."""
    
    def __init__(self):
        self.audit_history: List[AuditResult] = []
        self.personality_performance: Dict[str, PersonalityPerformance] = {}
        self.audit_patterns: Dict[str, List[str]] = {}
        self._initialize_audit_heuristics()
    
    def _initialize_audit_heuristics(self):
        """Initialize audit heuristics and patterns."""
        self.quality_indicators = {
            'high_quality': ['precise', 'accurate', 'comprehensive', 'insightful'],
            'medium_quality': ['adequate', 'reasonable', 'acceptable'],
            'low_quality': ['vague', 'incomplete', 'unclear', 'generic']
        }
        
        self.reasoning_indicators = {
            'strong_reasoning': ['because', 'therefore', 'since', 'given that', 'it follows'],
            'weak_reasoning': ['maybe', 'perhaps', 'possibly', 'might be'],
            'fallback_indicators': ['i understand', 'let me try', 'alternative approach', 'different way']
        }
        
        self.personality_consistency_patterns = {
            'Strategos': ['commander', 'tactical', 'strategic', 'military', 'battle'],
            'Archivist': ['archives', 'historical', 'records', 'chronicles', 'ancient'],
            'Lawmaker': ['governance', 'legal', 'principles', 'framework', 'law'],
            'Oracle': ['cosmic', 'universal', 'infinite', 'tapestry', 'reality']
        }
    
    def audit_conversation_turn(self, turn: ConversationTurn, 
                              reasoning_trace: Optional[Dict[str, Any]] = None,
                              context: Optional[ReasoningContext] = None) -> AuditResult:
        """
        Audit a conversation turn for performance and improvement opportunities.
        
        Args:
            turn: The conversation turn to audit
            reasoning_trace: Optional reasoning trace from the turn
            context: Optional context used for the turn
            
        Returns:
            AuditResult with metrics and suggestions
        """
        # Calculate response quality
        response_quality = self._assess_response_quality(turn.system_response)
        
        # Calculate reasoning accuracy
        reasoning_accuracy = self._assess_reasoning_accuracy(turn, reasoning_trace)
        
        # Calculate fallback usage
        fallback_usage = self._assess_fallback_usage(turn.system_response)
        
        # Calculate context alignment
        context_alignment = self._assess_context_alignment(turn, context)
        
        # Calculate personality consistency
        personality_consistency = self._assess_personality_consistency(
            turn.system_response, turn.personality_used
        )
        
        # Calculate meta-rule application
        meta_rule_application = self._assess_meta_rule_application(turn, reasoning_trace)
        
        # Compile metrics
        metrics = {
            AuditMetric.RESPONSE_QUALITY: response_quality,
            AuditMetric.REASONING_ACCURACY: reasoning_accuracy,
            AuditMetric.FALLBACK_USAGE: fallback_usage,
            AuditMetric.CONTEXT_ALIGNMENT: context_alignment,
            AuditMetric.PERSONALITY_CONSISTENCY: personality_consistency,
            AuditMetric.META_RULE_APPLICATION: meta_rule_application
        }
        
        # Identify gaps and improvement suggestions
        gaps_identified = self._identify_gaps(turn, metrics, reasoning_trace)
        improvement_suggestions = self._generate_improvement_suggestions(
            turn, metrics, gaps_identified
        )
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(metrics)
        
        # Create audit result
        audit_result = AuditResult(
            turn_id=turn.turn_id,
            personality=turn.personality_used,
            metrics=metrics,
            gaps_identified=gaps_identified,
            improvement_suggestions=improvement_suggestions,
            confidence_score=confidence_score
        )
        
        # Store audit result
        self.audit_history.append(audit_result)
        
        # Update personality performance
        self._update_personality_performance(audit_result)
        
        return audit_result
    
    def _assess_response_quality(self, response: str) -> float:
        """Assess the quality of a response."""
        response_lower = response.lower()
        
        # Check for high-quality indicators
        high_quality_score = sum(1 for indicator in self.quality_indicators['high_quality'] 
                                if indicator in response_lower)
        
        # Check for low-quality indicators
        low_quality_score = sum(1 for indicator in self.quality_indicators['low_quality'] 
                              if indicator in response_lower)
        
        # Calculate quality score (0.0 to 1.0)
        if high_quality_score > 0 and low_quality_score == 0:
            return min(1.0, 0.7 + (high_quality_score * 0.1))
        elif low_quality_score > 0:
            return max(0.0, 0.5 - (low_quality_score * 0.1))
        else:
            return 0.6  # Default medium quality
    
    def _assess_reasoning_accuracy(self, turn: ConversationTurn, 
                                 reasoning_trace: Optional[Dict[str, Any]]) -> float:
        """Assess the accuracy of reasoning used."""
        if not reasoning_trace:
            return 0.3  # Low score for no reasoning trace
        
        # Check reasoning trace quality
        trace_quality = 0.0
        
        if 'processing_steps' in reasoning_trace:
            trace_quality += 0.3
        
        if 'meta_insights' in reasoning_trace and reasoning_trace['meta_insights']:
            trace_quality += 0.4
        
        if 'confidence' in reasoning_trace:
            trace_quality += reasoning_trace['confidence'] * 0.3
        
        return min(1.0, trace_quality)
    
    def _assess_fallback_usage(self, response: str) -> float:
        """Assess whether fallback reasoning was used."""
        response_lower = response.lower()
        
        fallback_indicators = self.reasoning_indicators['fallback_indicators']
        fallback_score = sum(1 for indicator in fallback_indicators 
                           if indicator in response_lower)
        
        return min(1.0, fallback_score * 0.5)  # 0.0 = no fallback, 1.0 = heavy fallback
    
    def _assess_context_alignment(self, turn: ConversationTurn, 
                                context: Optional[ReasoningContext]) -> float:
        """Assess how well the response aligns with the context."""
        if not context:
            return 0.5  # Default score
        
        alignment_score = 0.0
        
        # Check if response matches intent type
        if turn.intent and context.intent_type:
            if turn.intent.lower() in context.intent_type.lower():
                alignment_score += 0.3
        
        # Check if response matches reasoning type
        if context.reasoning_type:
            if 'strategic' in context.reasoning_type and 'commander' in turn.system_response.lower():
                alignment_score += 0.2
            elif 'civilizational' in context.reasoning_type and 'archives' in turn.system_response.lower():
                alignment_score += 0.2
            elif 'meta_rule' in context.reasoning_type and 'governance' in turn.system_response.lower():
                alignment_score += 0.2
            elif 'universe' in context.reasoning_type and 'cosmic' in turn.system_response.lower():
                alignment_score += 0.2
        
        # Check confidence alignment
        if turn.confidence_score and context.confidence:
            confidence_diff = abs(turn.confidence_score - context.confidence)
            alignment_score += max(0.0, 0.3 - confidence_diff)
        
        return min(1.0, alignment_score)
    
    def _assess_personality_consistency(self, response: str, personality: str) -> float:
        """Assess how consistent the response is with the personality."""
        if personality not in self.personality_consistency_patterns:
            return 0.5
        
        response_lower = response.lower()
        expected_patterns = self.personality_consistency_patterns[personality]
        
        pattern_matches = sum(1 for pattern in expected_patterns 
                            if pattern in response_lower)
        
        return min(1.0, pattern_matches / len(expected_patterns))
    
    def _assess_meta_rule_application(self, turn: ConversationTurn, 
                                    reasoning_trace: Optional[Dict[str, Any]]) -> float:
        """Assess whether meta-rules were properly applied."""
        if not reasoning_trace or 'meta_insights' not in reasoning_trace:
            return 0.0
        
        meta_insights = reasoning_trace.get('meta_insights', [])
        if not meta_insights:
            return 0.0
        
        # Score based on number and quality of meta-insights
        insight_count = len(meta_insights)
        return min(1.0, insight_count * 0.3)
    
    def _identify_gaps(self, turn: ConversationTurn, metrics: Dict[AuditMetric, float],
                      reasoning_trace: Optional[Dict[str, Any]]) -> List[str]:
        """Identify gaps in reasoning or performance."""
        gaps = []
        
        # Check for low response quality
        if metrics[AuditMetric.RESPONSE_QUALITY] < 0.5:
            gaps.append("Low response quality - consider more specific or detailed responses")
        
        # Check for low reasoning accuracy
        if metrics[AuditMetric.REASONING_ACCURACY] < 0.4:
            gaps.append("Weak reasoning trace - improve reasoning pipeline")
        
        # Check for high fallback usage
        if metrics[AuditMetric.FALLBACK_USAGE] > 0.7:
            gaps.append("Excessive fallback usage - improve primary reasoning")
        
        # Check for low context alignment
        if metrics[AuditMetric.CONTEXT_ALIGNMENT] < 0.4:
            gaps.append("Poor context alignment - improve intent resolution")
        
        # Check for low personality consistency
        if metrics[AuditMetric.PERSONALITY_CONSISTENCY] < 0.5:
            gaps.append("Inconsistent personality expression - strengthen personality patterns")
        
        # Check for missing meta-rule application
        if metrics[AuditMetric.META_RULE_APPLICATION] < 0.3:
            gaps.append("Limited meta-rule application - enhance meta-learning")
        
        return gaps
    
    def _generate_improvement_suggestions(self, turn: ConversationTurn, 
                                        metrics: Dict[AuditMetric, float],
                                        gaps: List[str]) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        # Response quality improvements
        if metrics[AuditMetric.RESPONSE_QUALITY] < 0.6:
            suggestions.append("Use more specific terminology and provide concrete examples")
            suggestions.append("Incorporate domain-specific knowledge in responses")
        
        # Reasoning improvements
        if metrics[AuditMetric.REASONING_ACCURACY] < 0.5:
            suggestions.append("Strengthen reasoning trace generation")
            suggestions.append("Apply more meta-rules in reasoning process")
        
        # Fallback reduction
        if metrics[AuditMetric.FALLBACK_USAGE] > 0.6:
            suggestions.append("Improve primary reasoning to reduce fallback dependency")
            suggestions.append("Enhance context-intent resolution accuracy")
        
        # Context alignment improvements
        if metrics[AuditMetric.CONTEXT_ALIGNMENT] < 0.5:
            suggestions.append("Better align responses with detected intent")
            suggestions.append("Improve context understanding and application")
        
        # Personality consistency improvements
        if metrics[AuditMetric.PERSONALITY_CONSISTENCY] < 0.6:
            personality = turn.personality_used
            if personality == 'Strategos':
                suggestions.append("Use more military and tactical terminology")
            elif personality == 'Archivist':
                suggestions.append("Incorporate more historical and scholarly language")
            elif personality == 'Lawmaker':
                suggestions.append("Apply more legal and governance terminology")
            elif personality == 'Oracle':
                suggestions.append("Use more cosmic and mystical language")
        
        return suggestions
    
    def _calculate_confidence_score(self, metrics: Dict[AuditMetric, float]) -> float:
        """Calculate overall confidence score from metrics."""
        weights = {
            AuditMetric.RESPONSE_QUALITY: 0.25,
            AuditMetric.REASONING_ACCURACY: 0.20,
            AuditMetric.FALLBACK_USAGE: 0.15,  # Lower is better
            AuditMetric.CONTEXT_ALIGNMENT: 0.20,
            AuditMetric.PERSONALITY_CONSISTENCY: 0.10,
            AuditMetric.META_RULE_APPLICATION: 0.10
        }
        
        weighted_score = 0.0
        for metric, weight in weights.items():
            score = metrics[metric]
            if metric == AuditMetric.FALLBACK_USAGE:
                # Invert fallback usage (lower is better)
                score = 1.0 - score
            weighted_score += score * weight
        
        return min(1.0, max(0.0, weighted_score))
    
    def _update_personality_performance(self, audit_result: AuditResult):
        """Update personality performance tracking."""
        personality = audit_result.personality
        
        if personality not in self.personality_performance:
            self.personality_performance[personality] = PersonalityPerformance(personality)
        
        self.personality_performance[personality].update_from_audit(audit_result)
    
    def get_personality_performance(self, personality: str) -> Optional[PersonalityPerformance]:
        """Get performance metrics for a specific personality."""
        return self.personality_performance.get(personality)
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of all audit results."""
        if not self.audit_history:
            return {}
        
        total_audits = len(self.audit_history)
        average_confidence = sum(audit.confidence_score for audit in self.audit_history) / total_audits
        
        # Calculate improvement trends
        recent_audits = self.audit_history[-10:] if len(self.audit_history) >= 10 else self.audit_history
        recent_confidence = sum(audit.confidence_score for audit in recent_audits) / len(recent_audits)
        
        improvement_trend = recent_confidence - average_confidence
        
        return {
            'total_audits': total_audits,
            'average_confidence': average_confidence,
            'recent_confidence': recent_confidence,
            'improvement_trend': improvement_trend,
            'personality_performance': {
                personality: perf.__dict__ 
                for personality, perf in self.personality_performance.items()
            }
        }
    
    def get_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Get prioritized list of improvement opportunities."""
        opportunities = []
        
        for personality, performance in self.personality_performance.items():
            if performance.total_turns < 3:  # Need minimum data
                continue
            
            # Identify improvement areas
            if performance.average_confidence < 0.6:
                opportunities.append({
                    'personality': personality,
                    'area': 'confidence',
                    'priority': 'high',
                    'current_score': performance.average_confidence,
                    'suggestion': 'Improve overall response confidence'
                })
            
            if performance.fallback_usage / performance.total_turns > 0.5:
                opportunities.append({
                    'personality': personality,
                    'area': 'fallback_reduction',
                    'priority': 'high',
                    'current_score': performance.fallback_usage / performance.total_turns,
                    'suggestion': 'Reduce fallback usage through better primary reasoning'
                })
            
            if performance.reasoning_accuracy < 0.5:
                opportunities.append({
                    'personality': personality,
                    'area': 'reasoning_accuracy',
                    'priority': 'medium',
                    'current_score': performance.reasoning_accuracy,
                    'suggestion': 'Strengthen reasoning pipeline and meta-rule application'
                })
        
        # Sort by priority and current score
        opportunities.sort(key=lambda x: (x['priority'] == 'high', -x['current_score']))
        
        return opportunities
