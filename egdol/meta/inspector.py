"""
Memory and Rule Inspector for Egdol.
Provides introspection capabilities for the reasoning system.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from ..rules_engine import RulesEngine
from ..parser import Term, Variable, Constant, Rule, Fact
from ..interpreter import Interpreter
from ..memory import MemoryStore, MemoryItem


class MemoryInspector:
    """Inspects and analyzes memory contents."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in memory."""
        memories = self.memory_store.search(limit=10000)
        
        # Analyze by type
        by_type = {}
        for memory in memories:
            item_type = memory.item_type
            if item_type not in by_type:
                by_type[item_type] = []
            by_type[item_type].append(memory)
            
        # Analyze by source
        by_source = {}
        for memory in memories:
            source = memory.source
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(memory)
            
        # Analyze confidence distribution
        confidence_scores = [memory.confidence for memory in memories]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Analyze temporal patterns
        timestamps = [memory.timestamp for memory in memories]
        if timestamps:
            time_span = max(timestamps) - min(timestamps)
            recent_memories = len([t for t in timestamps if t > time.time() - 86400])  # Last 24 hours
        else:
            time_span = 0
            recent_memories = 0
            
        return {
            'total_memories': len(memories),
            'by_type': {t: len(mems) for t, mems in by_type.items()},
            'by_source': {s: len(mems) for s, mems in by_source.items()},
            'avg_confidence': avg_confidence,
            'time_span_days': time_span / 86400 if time_span > 0 else 0,
            'recent_memories': recent_memories,
            'confidence_distribution': self._analyze_confidence_distribution(confidence_scores)
        }
        
    def _analyze_confidence_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Analyze confidence score distribution."""
        if not scores:
            return {}
            
        ranges = {
            'very_low': 0,    # 0.0 - 0.2
            'low': 0,         # 0.2 - 0.4
            'medium': 0,      # 0.4 - 0.6
            'high': 0,        # 0.6 - 0.8
            'very_high': 0    # 0.8 - 1.0
        }
        
        for score in scores:
            if score < 0.2:
                ranges['very_low'] += 1
            elif score < 0.4:
                ranges['low'] += 1
            elif score < 0.6:
                ranges['medium'] += 1
            elif score < 0.8:
                ranges['high'] += 1
            else:
                ranges['very_high'] += 1
                
        return ranges
        
    def find_memory_gaps(self) -> List[str]:
        """Find potential gaps in memory."""
        gaps = []
        
        # Find facts without supporting rules
        facts = self.memory_store.search(item_type='fact')
        rules = self.memory_store.search(item_type='rule')
        
        fact_predicates = set()
        for fact in facts:
            if isinstance(fact.content, dict) and fact.content.get('type') == 'Term':
                fact_predicates.add(fact.content['name'])
                
        rule_predicates = set()
        for rule in rules:
            if isinstance(rule.content, dict) and rule.content.get('type') == 'Rule':
                head = rule.content.get('head', {})
                if head.get('type') == 'Term':
                    rule_predicates.add(head['name'])
                    
        # Find predicates that appear in facts but not in rules
        for pred in fact_predicates:
            if pred not in rule_predicates:
                gaps.append(f"Predicate '{pred}' appears in facts but has no rules")
                
        return gaps
        
    def suggest_memory_consolidation(self) -> List[str]:
        """Suggest ways to consolidate memory."""
        suggestions = []
        
        # Find duplicate or similar memories
        memories = self.memory_store.search(limit=1000)
        similar_groups = self._find_similar_memories(memories)
        
        for group in similar_groups:
            if len(group) > 1:
                suggestions.append(f"Consider consolidating {len(group)} similar memories")
                
        # Find low-confidence memories
        low_confidence = [m for m in memories if m.confidence < 0.5]
        if low_confidence:
            suggestions.append(f"Review {len(low_confidence)} low-confidence memories")
            
        return suggestions
        
    def _find_similar_memories(self, memories: List[MemoryItem]) -> List[List[MemoryItem]]:
        """Find groups of similar memories."""
        groups = []
        processed = set()
        
        for i, memory1 in enumerate(memories):
            if i in processed:
                continue
                
            group = [memory1]
            processed.add(i)
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if j in processed:
                    continue
                    
                if self._are_similar(memory1, memory2):
                    group.append(memory2)
                    processed.add(j)
                    
            if len(group) > 1:
                groups.append(group)
                
        return groups
        
    def _are_similar(self, mem1: MemoryItem, mem2: MemoryItem) -> bool:
        """Check if two memories are similar."""
        # Simple similarity check based on content type and structure
        if mem1.item_type != mem2.item_type:
            return False
            
        # For now, just check if they have similar content structure
        if isinstance(mem1.content, dict) and isinstance(mem2.content, dict):
            return mem1.content.get('type') == mem2.content.get('type')
            
        return str(mem1.content) == str(mem2.content)


class RuleInspector:
    """Inspects and analyzes rule behavior."""
    
    def __init__(self, engine: RulesEngine, memory_store: MemoryStore):
        self.engine = engine
        self.memory_store = memory_store
        self.interpreter = Interpreter(engine)
        
    def analyze_rule_usage(self) -> Dict[str, Any]:
        """Analyze how rules are being used."""
        rules = self.memory_store.search(item_type='rule')
        
        usage_stats = {}
        for rule in rules:
            rule_id = rule.id
            rule_content = rule.content
            
            if isinstance(rule_content, dict) and rule_content.get('type') == 'Rule':
                head = rule_content.get('head', {})
                if head.get('type') == 'Term':
                    predicate = head['name']
                    
                    # Count how many times this rule fires
                    goal = Term(predicate, [Variable('X')])
                    results = list(self.interpreter.query(goal))
                    
                    usage_stats[predicate] = {
                        'rule_id': rule_id,
                        'usage_count': len(results),
                        'confidence': rule.confidence,
                        'source': rule.source
                    }
                    
        return usage_stats
        
    def find_unused_rules(self) -> List[Dict[str, Any]]:
        """Find rules that are never used."""
        usage_stats = self.analyze_rule_usage()
        unused = []
        
        for predicate, stats in usage_stats.items():
            if stats['usage_count'] == 0:
                unused.append({
                    'predicate': predicate,
                    'rule_id': stats['rule_id'],
                    'confidence': stats['confidence'],
                    'source': stats['source']
                })
                
        return unused
        
    def find_rule_conflicts(self) -> List[Dict[str, Any]]:
        """Find potential conflicts between rules."""
        conflicts = []
        
        # Get all rules
        rules = self.memory_store.search(item_type='rule')
        
        # Check for contradictory rules
        for i, rule1 in enumerate(rules):
            for j, rule2 in enumerate(rules[i+1:], i+1):
                if self._are_conflicting(rule1, rule2):
                    conflicts.append({
                        'rule1_id': rule1.id,
                        'rule2_id': rule2.id,
                        'conflict_type': 'contradiction',
                        'description': f"Rules {rule1.id} and {rule2.id} may conflict"
                    })
                    
        return conflicts
        
    def _are_conflicting(self, rule1: MemoryItem, rule2: MemoryItem) -> bool:
        """Check if two rules conflict."""
        # Simple conflict detection
        # For now, just check if they have the same head predicate but different bodies
        if not isinstance(rule1.content, dict) or not isinstance(rule2.content, dict):
            return False
            
        if rule1.content.get('type') != 'Rule' or rule2.content.get('type') != 'Rule':
            return False
            
        head1 = rule1.content.get('head', {})
        head2 = rule2.content.get('head', {})
        
        if head1.get('type') == 'Term' and head2.get('type') == 'Term':
            if head1.get('name') == head2.get('name'):
                # Same predicate, check if bodies are different
                body1 = rule1.content.get('body', [])
                body2 = rule2.content.get('body', [])
                return body1 != body2
                
        return False
        
    def suggest_rule_optimizations(self) -> List[str]:
        """Suggest optimizations for rules."""
        suggestions = []
        
        # Find unused rules
        unused = self.find_unused_rules()
        if unused:
            suggestions.append(f"Consider removing {len(unused)} unused rules")
            
        # Find conflicting rules
        conflicts = self.find_rule_conflicts()
        if conflicts:
            suggestions.append(f"Review {len(conflicts)} potentially conflicting rules")
            
        # Find rules with low confidence
        rules = self.memory_store.search(item_type='rule')
        low_confidence_rules = [r for r in rules if r.confidence < 0.5]
        if low_confidence_rules:
            suggestions.append(f"Review {len(low_confidence_rules)} low-confidence rules")
            
        return suggestions
        
    def explain_reasoning(self, query: str) -> Dict[str, Any]:
        """Explain how a query would be resolved."""
        try:
            # Parse the query
            goal = self._parse_query(query)
            if not goal:
                return {'error': 'Could not parse query'}
                
            # Find applicable rules
            applicable_rules = self._find_applicable_rules(goal)
            
            # Build explanation
            explanation = {
                'query': query,
                'goal': str(goal),
                'applicable_rules': applicable_rules,
                'reasoning_steps': self._build_reasoning_steps(goal, applicable_rules)
            }
            
            return explanation
            
        except Exception as e:
            return {'error': f'Could not explain reasoning: {e}'}
            
    def _parse_query(self, query: str) -> Optional[Term]:
        """Parse a query string into a Term."""
        # Simple query parsing
        if ' ' in query:
            parts = query.split()
            if len(parts) >= 2:
                predicate = parts[0]
                args = [Variable('X')]  # Simple variable for now
                return Term(predicate, args)
        return None
        
    def _find_applicable_rules(self, goal: Term) -> List[Dict[str, Any]]:
        """Find rules that could be applied to the goal."""
        applicable = []
        
        rules = self.memory_store.search(item_type='rule')
        for rule in rules:
            if isinstance(rule.content, dict) and rule.content.get('type') == 'Rule':
                head = rule.content.get('head', {})
                if head.get('type') == 'Term' and head.get('name') == goal.name:
                    applicable.append({
                        'rule_id': rule.id,
                        'head': head,
                        'body': rule.content.get('body', []),
                        'confidence': rule.confidence,
                        'source': rule.source
                    })
                    
        return applicable
        
    def _build_reasoning_steps(self, goal: Term, applicable_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build reasoning steps for the explanation."""
        steps = []
        
        for rule in applicable_rules:
            steps.append({
                'step': len(steps) + 1,
                'description': f"Apply rule {rule['rule_id']}",
                'rule': rule,
                'confidence': rule['confidence']
            })
            
        return steps
