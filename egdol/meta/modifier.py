"""
Self-Modification and Rule Modifier for Egdol.
Provides capabilities for the system to modify itself.
"""

import time
import random
from typing import List, Dict, Any, Optional, Tuple
from ..rules_engine import RulesEngine
from ..parser import Term, Variable, Constant, Rule, Fact
from ..memory import MemoryStore, MemoryItem
from .scorer import RuleScorer


class SelfModifier:
    """Enables self-modification of the reasoning system."""
    
    def __init__(self, engine: RulesEngine, memory_store: MemoryStore):
        self.engine = engine
        self.memory_store = memory_store
        self.scorer = RuleScorer(memory_store)
        self.modification_history: List[Dict[str, Any]] = []
        
    def propose_rule_modifications(self) -> List[Dict[str, Any]]:
        """Propose modifications to existing rules."""
        proposals = []
        
        # Get underperforming rules
        underperforming = self.scorer.get_underperforming_rules()
        
        for rule_id, score in underperforming:
            rule = self.memory_store.retrieve(rule_id)
            if rule and isinstance(rule.content, dict):
                proposal = self._analyze_rule_for_modification(rule)
                if proposal:
                    proposals.append(proposal)
                    
        return proposals
        
    def _analyze_rule_for_modification(self, rule: MemoryItem) -> Optional[Dict[str, Any]]:
        """Analyze a rule for potential modifications."""
        if rule.content.get('type') != 'Rule':
            return None
            
        head = rule.content.get('head', {})
        body = rule.content.get('body', [])
        
        # Check if rule has too many conditions
        if len(body) > 3:
            return {
                'type': 'simplify_rule',
                'rule_id': rule.id,
                'current_complexity': len(body),
                'suggestion': f"Rule {rule.id} has {len(body)} conditions. Consider simplifying.",
                'action': 'remove_conditions'
            }
            
        # Check if rule has no conditions (fact)
        if len(body) == 0:
            return {
                'type': 'add_conditions',
                'rule_id': rule.id,
                'suggestion': f"Rule {rule.id} has no conditions. Consider adding some.",
                'action': 'add_conditions'
            }
            
        return None
        
    def apply_rule_modification(self, rule_id: int, modification: Dict[str, Any]) -> bool:
        """Apply a rule modification."""
        try:
            rule = self.memory_store.retrieve(rule_id)
            if not rule:
                return False
                
            # Record the modification
            self.modification_history.append({
                'timestamp': time.time(),
                'rule_id': rule_id,
                'modification': modification,
                'success': False  # Will be updated if successful
            })
            
            # Apply the modification
            if modification['action'] == 'remove_conditions':
                success = self._remove_rule_conditions(rule_id)
            elif modification['action'] == 'add_conditions':
                success = self._add_rule_conditions(rule_id)
            else:
                success = False
                
            # Update modification history
            if success:
                self.modification_history[-1]['success'] = True
                
            return success
            
        except Exception as e:
            return False
            
    def _remove_rule_conditions(self, rule_id: int) -> bool:
        """Remove conditions from a rule."""
        rule = self.memory_store.retrieve(rule_id)
        if not rule or rule.content.get('type') != 'Rule':
            return False
            
        body = rule.content.get('body', [])
        if len(body) <= 1:
            return False  # Can't remove all conditions
            
        # Remove one condition (the last one)
        new_body = body[:-1]
        
        # Update the rule content
        new_content = rule.content.copy()
        new_content['body'] = new_body
        
        # Update in memory store
        self.memory_store.update(rule_id, content=new_content)
        
        return True
        
    def _add_rule_conditions(self, rule_id: int) -> bool:
        """Add conditions to a rule."""
        rule = self.memory_store.retrieve(rule_id)
        if not rule or rule.content.get('type') != 'Rule':
            return False
            
        body = rule.content.get('body', [])
        
        # Add a simple condition (this is a placeholder - in practice, you'd need more sophisticated logic)
        new_condition = {
            'type': 'Term',
            'name': 'condition',
            'args': [{'type': 'Variable', 'name': 'X'}]
        }
        
        new_body = body + [new_condition]
        new_content = rule.content.copy()
        new_content['body'] = new_body
        
        # Update in memory store
        self.memory_store.update(rule_id, content=new_content)
        
        return True
        
    def propose_new_rules(self) -> List[Dict[str, Any]]:
        """Propose new rules based on existing patterns."""
        proposals = []
        
        # Analyze existing facts to find patterns
        facts = self.memory_store.search(item_type='fact')
        
        # Find common patterns
        patterns = self._find_fact_patterns(facts)
        
        for pattern in patterns:
            if pattern['frequency'] > 2:  # Only propose rules for common patterns
                proposals.append({
                    'type': 'new_rule',
                    'pattern': pattern,
                    'suggestion': f"Consider creating a rule for pattern: {pattern['description']}",
                    'confidence': min(pattern['frequency'] / 10.0, 1.0)
                })
                
        return proposals
        
    def _find_fact_patterns(self, facts: List[MemoryItem]) -> List[Dict[str, Any]]:
        """Find patterns in facts."""
        patterns = []
        
        # Group facts by predicate
        predicate_groups = {}
        for fact in facts:
            if isinstance(fact.content, dict) and fact.content.get('type') == 'Term':
                predicate = fact.content['name']
                if predicate not in predicate_groups:
                    predicate_groups[predicate] = []
                predicate_groups[predicate].append(fact)
                
        # Find patterns
        for predicate, fact_list in predicate_groups.items():
            if len(fact_list) > 1:
                patterns.append({
                    'predicate': predicate,
                    'frequency': len(fact_list),
                    'description': f"Multiple facts with predicate '{predicate}'",
                    'facts': fact_list
                })
                
        return patterns
        
    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Get history of modifications."""
        return self.modification_history
        
    def get_modification_stats(self) -> Dict[str, Any]:
        """Get statistics about modifications."""
        if not self.modification_history:
            return {'total_modifications': 0, 'success_rate': 0.0}
            
        total = len(self.modification_history)
        successful = sum(1 for mod in self.modification_history if mod['success'])
        success_rate = successful / total if total > 0 else 0.0
        
        return {
            'total_modifications': total,
            'successful_modifications': successful,
            'success_rate': success_rate,
            'recent_modifications': len([m for m in self.modification_history 
                                      if m['timestamp'] > time.time() - 86400])
        }


class RuleModifier:
    """Specialized rule modification capabilities."""
    
    def __init__(self, engine: RulesEngine, memory_store: MemoryStore):
        self.engine = engine
        self.memory_store = memory_store
        
    def create_rule_from_facts(self, fact_ids: List[int]) -> Optional[Dict[str, Any]]:
        """Create a rule from a set of facts."""
        facts = []
        for fact_id in fact_ids:
            fact = self.memory_store.retrieve(fact_id)
            if fact and fact.item_type == 'fact':
                facts.append(fact)
                
        if len(facts) < 2:
            return None
            
        # Analyze facts to create rule
        rule = self._analyze_facts_for_rule(facts)
        if rule:
            # Store the new rule
            rule_id = self.memory_store.store(
                content=rule,
                item_type='rule',
                source='auto_generated',
                confidence=0.7,
                metadata={'generated_from_facts': fact_ids}
            )
            
            return {
                'rule_id': rule_id,
                'rule': rule,
                'generated_from': fact_ids
            }
            
        return None
        
    def _analyze_facts_for_rule(self, facts: List[MemoryItem]) -> Optional[Dict[str, Any]]:
        """Analyze facts to create a rule."""
        # Simple rule generation - look for common patterns
        if len(facts) < 2:
            return None
            
        # For now, create a simple rule that generalizes the facts
        # This is a placeholder - in practice, you'd need more sophisticated analysis
        
        # Extract predicates from facts
        predicates = []
        for fact in facts:
            if isinstance(fact.content, dict) and fact.content.get('type') == 'Term':
                predicates.append(fact.content['name'])
                
        if not predicates:
            return None
            
        # Create a simple rule
        most_common_predicate = max(set(predicates), key=predicates.count)
        
        rule = {
            'type': 'Rule',
            'head': {
                'type': 'Term',
                'name': most_common_predicate,
                'args': [{'type': 'Variable', 'name': 'X'}]
            },
            'body': [
                {
                    'type': 'Term',
                    'name': 'condition',
                    'args': [{'type': 'Variable', 'name': 'X'}]
                }
            ]
        }
        
        return rule
        
    def merge_similar_rules(self, rule_ids: List[int]) -> Optional[Dict[str, Any]]:
        """Merge similar rules into one."""
        rules = []
        for rule_id in rule_ids:
            rule = self.memory_store.retrieve(rule_id)
            if rule and rule.item_type == 'rule':
                rules.append(rule)
                
        if len(rules) < 2:
            return None
            
        # Analyze rules for merging
        merged_rule = self._analyze_rules_for_merging(rules)
        if merged_rule:
            # Store the merged rule
            rule_id = self.memory_store.store(
                content=merged_rule,
                item_type='rule',
                source='auto_merged',
                confidence=0.8,
                metadata={'merged_from': rule_ids}
            )
            
            return {
                'rule_id': rule_id,
                'rule': merged_rule,
                'merged_from': rule_ids
            }
            
        return None
        
    def _analyze_rules_for_merging(self, rules: List[MemoryItem]) -> Optional[Dict[str, Any]]:
        """Analyze rules to see if they can be merged."""
        # Simple merging logic - look for rules with same head predicate
        head_predicates = []
        for rule in rules:
            if isinstance(rule.content, dict) and rule.content.get('type') == 'Rule':
                head = rule.content.get('head', {})
                if head.get('type') == 'Term':
                    head_predicates.append(head['name'])
                    
        if not head_predicates:
            return None
            
        # Check if all rules have the same head predicate
        if len(set(head_predicates)) == 1:
            # Create merged rule
            merged_rule = {
                'type': 'Rule',
                'head': {
                    'type': 'Term',
                    'name': head_predicates[0],
                    'args': [{'type': 'Variable', 'name': 'X'}]
                },
                'body': [
                    {
                        'type': 'Term',
                        'name': 'condition',
                        'args': [{'type': 'Variable', 'name': 'X'}]
                    }
                ]
            }
            return merged_rule
            
        return None
