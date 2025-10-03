"""
Rule Scorer and Confidence Tracker for Egdol.
Provides scoring and confidence tracking for rules and facts.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from ..memory import MemoryStore, MemoryItem


class RuleScorer:
    """Scores rules based on their usefulness and reliability."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.usage_history: Dict[int, List[float]] = defaultdict(list)
        self.success_history: Dict[int, List[bool]] = defaultdict(list)
        
    def score_rule(self, rule_id: int) -> float:
        """Calculate a score for a rule based on its history."""
        if rule_id not in self.usage_history:
            return 0.5  # Default score for new rules
            
        usage_count = len(self.usage_history[rule_id])
        if usage_count == 0:
            return 0.5
            
        # Calculate success rate
        success_count = sum(self.success_history[rule_id])
        success_rate = success_count / usage_count if usage_count > 0 else 0.5
        
        # Calculate recency bonus (more recent usage gets higher weight)
        recent_usage = self._calculate_recent_usage(rule_id)
        
        # Calculate frequency bonus
        frequency_score = min(usage_count / 10.0, 1.0)  # Cap at 1.0
        
        # Combine scores
        score = (success_rate * 0.4 + recent_usage * 0.3 + frequency_score * 0.3)
        return max(0.0, min(1.0, score))
        
    def _calculate_recent_usage(self, rule_id: int) -> float:
        """Calculate recency bonus for rule usage."""
        if rule_id not in self.usage_history:
            return 0.0
            
        timestamps = self.usage_history[rule_id]
        if not timestamps:
            return 0.0
            
        # Calculate time since last usage
        time_since_last = time.time() - max(timestamps)
        
        # Recent usage gets higher score
        if time_since_last < 3600:  # Last hour
            return 1.0
        elif time_since_last < 86400:  # Last day
            return 0.8
        elif time_since_last < 604800:  # Last week
            return 0.6
        else:
            return 0.2
            
    def record_rule_usage(self, rule_id: int, success: bool = True):
        """Record that a rule was used."""
        current_time = time.time()
        self.usage_history[rule_id].append(current_time)
        self.success_history[rule_id].append(success)
        
        # Keep only recent history (last 100 uses)
        if len(self.usage_history[rule_id]) > 100:
            self.usage_history[rule_id] = self.usage_history[rule_id][-100:]
            self.success_history[rule_id] = self.success_history[rule_id][-100:]
            
    def get_top_rules(self, limit: int = 10) -> List[Tuple[int, float]]:
        """Get top-scoring rules."""
        all_rules = self.memory_store.search(item_type='rule')
        scored_rules = []
        
        for rule in all_rules:
            score = self.score_rule(rule.id)
            scored_rules.append((rule.id, score))
            
        # Sort by score (descending)
        scored_rules.sort(key=lambda x: x[1], reverse=True)
        return scored_rules[:limit]
        
    def get_underperforming_rules(self, threshold: float = 0.3) -> List[Tuple[int, float]]:
        """Get rules that are underperforming."""
        all_rules = self.memory_store.search(item_type='rule')
        underperforming = []
        
        for rule in all_rules:
            score = self.score_rule(rule.id)
            if score < threshold:
                underperforming.append((rule.id, score))
                
        return underperforming
        
    def suggest_rule_improvements(self) -> List[Dict[str, Any]]:
        """Suggest improvements for rules."""
        suggestions = []
        
        # Find underperforming rules
        underperforming = self.get_underperforming_rules()
        for rule_id, score in underperforming:
            rule = self.memory_store.retrieve(rule_id)
            if rule:
                suggestions.append({
                    'type': 'improve_rule',
                    'rule_id': rule_id,
                    'current_score': score,
                    'suggestion': f"Rule {rule_id} has low score ({score:.2f}). Consider reviewing or removing it."
                })
                
        # Find unused rules
        all_rules = self.memory_store.search(item_type='rule')
        for rule in all_rules:
            if rule.id not in self.usage_history:
                suggestions.append({
                    'type': 'unused_rule',
                    'rule_id': rule.id,
                    'suggestion': f"Rule {rule.id} has never been used. Consider removing it."
                })
                
        return suggestions


class ConfidenceTracker:
    """Tracks and manages confidence scores for memories."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.confidence_history: Dict[int, List[float]] = defaultdict(list)
        
    def update_confidence(self, memory_id: int, new_confidence: float, reason: str = ""):
        """Update confidence for a memory item."""
        # Record the change
        self.confidence_history[memory_id].append({
            'confidence': new_confidence,
            'timestamp': time.time(),
            'reason': reason
        })
        
        # Update the memory
        self.memory_store.update(memory_id, confidence=new_confidence)
        
    def get_confidence_trend(self, memory_id: int) -> Dict[str, Any]:
        """Get confidence trend for a memory item."""
        if memory_id not in self.confidence_history:
            return {'trend': 'stable', 'changes': 0}
            
        history = self.confidence_history[memory_id]
        if len(history) < 2:
            return {'trend': 'stable', 'changes': len(history)}
            
        # Calculate trend
        confidences = [entry['confidence'] for entry in history]
        first_conf = confidences[0]
        last_conf = confidences[-1]
        
        if last_conf > first_conf + 0.1:
            trend = 'increasing'
        elif last_conf < first_conf - 0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'
            
        return {
            'trend': trend,
            'changes': len(history),
            'first_confidence': first_conf,
            'last_confidence': last_conf,
            'history': history
        }
        
    def find_confidence_anomalies(self) -> List[Dict[str, Any]]:
        """Find memories with unusual confidence patterns."""
        anomalies = []
        
        for memory_id, history in self.confidence_history.items():
            if len(history) < 3:
                continue
                
            confidences = [entry['confidence'] for entry in history]
            
            # Check for sudden drops
            for i in range(1, len(confidences)):
                if confidences[i] < confidences[i-1] - 0.3:
                    anomalies.append({
                        'memory_id': memory_id,
                        'type': 'sudden_drop',
                        'from': confidences[i-1],
                        'to': confidences[i],
                        'timestamp': history[i]['timestamp']
                    })
                    
            # Check for sudden increases
            for i in range(1, len(confidences)):
                if confidences[i] > confidences[i-1] + 0.3:
                    anomalies.append({
                        'memory_id': memory_id,
                        'type': 'sudden_increase',
                        'from': confidences[i-1],
                        'to': confidences[i],
                        'timestamp': history[i]['timestamp']
                    })
                    
        return anomalies
        
    def get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence scores."""
        memories = self.memory_store.search(limit=10000)
        distribution = {
            'very_low': 0,    # 0.0 - 0.2
            'low': 0,         # 0.2 - 0.4
            'medium': 0,      # 0.4 - 0.6
            'high': 0,        # 0.6 - 0.8
            'very_high': 0    # 0.8 - 1.0
        }
        
        for memory in memories:
            conf = memory.confidence
            if conf < 0.2:
                distribution['very_low'] += 1
            elif conf < 0.4:
                distribution['low'] += 1
            elif conf < 0.6:
                distribution['medium'] += 1
            elif conf < 0.8:
                distribution['high'] += 1
            else:
                distribution['very_high'] += 1
                
        return distribution
        
    def suggest_confidence_adjustments(self) -> List[Dict[str, Any]]:
        """Suggest confidence adjustments based on patterns."""
        suggestions = []
        
        # Find memories with consistently low confidence
        low_confidence_memories = []
        for memory in self.memory_store.search(limit=1000):
            if memory.confidence < 0.3:
                low_confidence_memories.append(memory)
                
        if low_confidence_memories:
            suggestions.append({
                'type': 'review_low_confidence',
                'count': len(low_confidence_memories),
                'suggestion': f"Review {len(low_confidence_memories)} memories with low confidence"
            })
            
        # Find memories with unstable confidence
        unstable_memories = []
        for memory_id, history in self.confidence_history.items():
            if len(history) > 5:
                confidences = [entry['confidence'] for entry in history]
                variance = max(confidences) - min(confidences)
                if variance > 0.5:
                    unstable_memories.append(memory_id)
                    
        if unstable_memories:
            suggestions.append({
                'type': 'review_unstable_confidence',
                'count': len(unstable_memories),
                'suggestion': f"Review {len(unstable_memories)} memories with unstable confidence"
            })
            
        return suggestions
