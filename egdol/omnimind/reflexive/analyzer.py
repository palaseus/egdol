"""
Analyzers for OmniMind Reflexive System
Specialized analyzers for reasoning, skills, and memory.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Result of analysis operation."""
    analyzer_type: str
    findings: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    confidence: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'analyzer_type': self.analyzer_type,
            'findings': self.findings,
            'recommendations': self.recommendations,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


class ReasoningAnalyzer:
    """Analyzes reasoning patterns and efficiency."""
    
    def __init__(self, memory_manager=None, planner=None):
        self.memory_manager = memory_manager
        self.planner = planner
        
    def analyze_reasoning_patterns(self) -> AnalysisResult:
        """Analyze reasoning patterns for inefficiencies."""
        findings = []
        recommendations = []
        
        # Analyze rule usage patterns
        if self.memory_manager:
            rules = self.memory_manager.get_all_rules()
            facts = self.memory_manager.get_all_facts()
            
            # Find unused rules
            unused_rules = self._find_unused_rules(rules, facts)
            if unused_rules:
                findings.append({
                    'type': 'unused_rules',
                    'count': len(unused_rules),
                    'severity': 'medium',
                    'description': f'Found {len(unused_rules)} unused rules'
                })
                
                recommendations.append({
                    'type': 'remove_unused_rules',
                    'priority': 'medium',
                    'description': 'Remove unused rules to improve efficiency',
                    'estimated_impact': 'memory_reduction'
                })
                
            # Find conflicting rules
            conflicts = self._find_rule_conflicts(rules)
            if conflicts:
                findings.append({
                    'type': 'rule_conflicts',
                    'count': len(conflicts),
                    'severity': 'high',
                    'description': f'Found {len(conflicts)} rule conflicts'
                })
                
                recommendations.append({
                    'type': 'resolve_conflicts',
                    'priority': 'high',
                    'description': 'Resolve rule conflicts to ensure consistency',
                    'estimated_impact': 'reasoning_consistency'
                })
                
            # Find complex rules
            complex_rules = self._find_complex_rules(rules)
            if complex_rules:
                findings.append({
                    'type': 'complex_rules',
                    'count': len(complex_rules),
                    'severity': 'low',
                    'description': f'Found {len(complex_rules)} overly complex rules'
                })
                
                recommendations.append({
                    'type': 'simplify_rules',
                    'priority': 'low',
                    'description': 'Simplify complex rules for better maintainability',
                    'estimated_impact': 'reasoning_efficiency'
                })
                
        # Analyze goal complexity
        if self.planner:
            goals = self.planner.get_all_goals()
            complex_goals = self._find_complex_goals(goals)
            if complex_goals:
                findings.append({
                    'type': 'complex_goals',
                    'count': len(complex_goals),
                    'severity': 'medium',
                    'description': f'Found {len(complex_goals)} overly complex goals'
                })
                
                recommendations.append({
                    'type': 'simplify_goals',
                    'priority': 'medium',
                    'description': 'Break down complex goals into smaller ones',
                    'estimated_impact': 'reasoning_efficiency'
                })
                
        # Calculate confidence
        confidence = self._calculate_confidence(findings)
        
        return AnalysisResult(
            analyzer_type='reasoning',
            findings=findings,
            recommendations=recommendations,
            confidence=confidence,
            timestamp=time.time()
        )
        
    def _find_unused_rules(self, rules, facts) -> List[Dict[str, Any]]:
        """Find rules that are never used."""
        unused = []
        
        for rule in rules:
            if not self._is_rule_used(rule, facts):
                unused.append({
                    'id': getattr(rule, 'id', 'unknown'),
                    'name': getattr(rule, 'name', 'unknown'),
                    'description': getattr(rule, 'description', 'No description')
                })
                
        return unused
        
    def _find_rule_conflicts(self, rules) -> List[Dict[str, Any]]:
        """Find conflicting rules."""
        conflicts = []
        
        for i, rule1 in enumerate(rules):
            for j, rule2 in enumerate(rules[i+1:], i+1):
                if self._rules_conflict(rule1, rule2):
                    conflicts.append({
                        'rule1': {
                            'id': getattr(rule1, 'id', 'unknown'),
                            'name': getattr(rule1, 'name', 'unknown')
                        },
                        'rule2': {
                            'id': getattr(rule2, 'id', 'unknown'),
                            'name': getattr(rule2, 'name', 'unknown')
                        }
                    })
                    
        return conflicts
        
    def _find_complex_rules(self, rules) -> List[Dict[str, Any]]:
        """Find overly complex rules."""
        complex_rules = []
        
        for rule in rules:
            if self._is_rule_complex(rule):
                complex_rules.append({
                    'id': getattr(rule, 'id', 'unknown'),
                    'name': getattr(rule, 'name', 'unknown'),
                    'complexity_score': self._calculate_rule_complexity(rule)
                })
                
        return complex_rules
        
    def _find_complex_goals(self, goals) -> List[Dict[str, Any]]:
        """Find overly complex goals."""
        complex_goals = []
        
        for goal in goals.get('active_goals', []):
            task_count = len(goal.get('tasks', []))
            if task_count > 10:  # Threshold for complexity
                complex_goals.append({
                    'id': goal.get('id', 'unknown'),
                    'description': goal.get('description', 'No description'),
                    'task_count': task_count
                })
                
        return complex_goals
        
    def _is_rule_used(self, rule, facts) -> bool:
        """Check if a rule is being used."""
        # Simple heuristic: check if rule conditions match any facts
        if hasattr(rule, 'conditions'):
            for fact in facts:
                if fact in rule.conditions:
                    return True
        return False
        
    def _rules_conflict(self, rule1, rule2) -> bool:
        """Check if two rules conflict."""
        # Simple heuristic: check if rules have opposite conclusions
        if hasattr(rule1, 'conclusion') and hasattr(rule2, 'conclusion'):
            return rule1.conclusion != rule2.conclusion
        return False
        
    def _is_rule_complex(self, rule) -> bool:
        """Check if a rule is overly complex."""
        # Simple heuristic: check rule length or number of conditions
        if hasattr(rule, 'conditions'):
            return len(rule.conditions) > 5
        return len(str(rule)) > 100
        
    def _calculate_rule_complexity(self, rule) -> float:
        """Calculate complexity score for a rule."""
        score = 0.0
        
        if hasattr(rule, 'conditions'):
            score += len(rule.conditions) * 0.2
            
        if hasattr(rule, 'conclusion'):
            score += len(str(rule.conclusion)) * 0.01
            
        score += len(str(rule)) * 0.001
        
        return score
        
    def _calculate_confidence(self, findings) -> float:
        """Calculate confidence in analysis results."""
        if not findings:
            return 0.0
            
        # Base confidence on number and severity of findings
        high_severity = sum(1 for f in findings if f.get('severity') == 'high')
        medium_severity = sum(1 for f in findings if f.get('severity') == 'medium')
        
        if high_severity >= 2:
            return 0.9
        elif high_severity >= 1 or medium_severity >= 3:
            return 0.8
        elif medium_severity >= 1:
            return 0.7
        else:
            return 0.6


class SkillAnalyzer:
    """Analyzes skill usage and performance."""
    
    def __init__(self, skill_router=None):
        self.skill_router = skill_router
        
    def analyze_skill_performance(self) -> AnalysisResult:
        """Analyze skill performance and usage patterns."""
        findings = []
        recommendations = []
        
        if not self.skill_router:
            return AnalysisResult(
                analyzer_type='skills',
                findings=findings,
                recommendations=recommendations,
                confidence=0.0,
                timestamp=time.time()
            )
            
        # Get skill statistics
        skill_stats = self.skill_router.get_skill_statistics()
        
        # Analyze underperforming skills
        underperforming = self._find_underperforming_skills(skill_stats)
        if underperforming:
            findings.append({
                'type': 'underperforming_skills',
                'count': len(underperforming),
                'severity': 'high',
                'description': f'Found {len(underperforming)} underperforming skills'
            })
            
            recommendations.append({
                'type': 'improve_skills',
                'priority': 'high',
                'description': 'Improve underperforming skills',
                'estimated_impact': 'task_success_rate'
            })
            
        # Analyze rarely used skills
        rarely_used = self._find_rarely_used_skills(skill_stats)
        if rarely_used:
            findings.append({
                'type': 'rarely_used_skills',
                'count': len(rarely_used),
                'severity': 'low',
                'description': f'Found {len(rarely_used)} rarely used skills'
            })
            
            recommendations.append({
                'type': 'review_skills',
                'priority': 'low',
                'description': 'Review rarely used skills for removal',
                'estimated_impact': 'skill_maintenance'
            })
            
        # Analyze skill load balancing
        load_imbalance = self._find_load_imbalance(skill_stats)
        if load_imbalance:
            findings.append({
                'type': 'load_imbalance',
                'severity': 'medium',
                'description': 'Uneven skill usage detected'
            })
            
            recommendations.append({
                'type': 'balance_skills',
                'priority': 'medium',
                'description': 'Balance skill usage across available skills',
                'estimated_impact': 'performance'
            })
            
        # Calculate confidence
        confidence = self._calculate_confidence(findings)
        
        return AnalysisResult(
            analyzer_type='skills',
            findings=findings,
            recommendations=recommendations,
            confidence=confidence,
            timestamp=time.time()
        )
        
    def _find_underperforming_skills(self, skill_stats) -> List[Dict[str, Any]]:
        """Find skills with low success rates."""
        underperforming = []
        
        for skill_name, stats in skill_stats.items():
            success_rate = stats.get('success_rate', 1.0)
            execution_count = stats.get('execution_count', 0)
            
            if execution_count > 5 and success_rate < 0.7:
                underperforming.append({
                    'name': skill_name,
                    'success_rate': success_rate,
                    'execution_count': execution_count
                })
                
        return underperforming
        
    def _find_rarely_used_skills(self, skill_stats) -> List[Dict[str, Any]]:
        """Find skills that are rarely used."""
        rarely_used = []
        
        for skill_name, stats in skill_stats.items():
            execution_count = stats.get('execution_count', 0)
            
            if execution_count < 3:
                rarely_used.append({
                    'name': skill_name,
                    'execution_count': execution_count
                })
                
        return rarely_used
        
    def _find_load_imbalance(self, skill_stats) -> bool:
        """Check for load imbalance across skills."""
        if len(skill_stats) < 2:
            return False
            
        execution_counts = [stats.get('execution_count', 0) for stats in skill_stats.values()]
        
        if not execution_counts:
            return False
            
        max_count = max(execution_counts)
        min_count = min(execution_counts)
        
        # Consider imbalanced if max is more than 3x min
        return max_count > 3 * min_count and max_count > 0
        
    def _calculate_confidence(self, findings) -> float:
        """Calculate confidence in analysis results."""
        if not findings:
            return 0.0
            
        high_severity = sum(1 for f in findings if f.get('severity') == 'high')
        medium_severity = sum(1 for f in findings if f.get('severity') == 'medium')
        
        if high_severity >= 1:
            return 0.9
        elif medium_severity >= 2:
            return 0.8
        elif medium_severity >= 1:
            return 0.7
        else:
            return 0.6


class MemoryAnalyzer:
    """Analyzes memory usage and organization."""
    
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        
    def analyze_memory_usage(self) -> AnalysisResult:
        """Analyze memory usage and organization."""
        findings = []
        recommendations = []
        
        if not self.memory_manager:
            return AnalysisResult(
                analyzer_type='memory',
                findings=findings,
                recommendations=recommendations,
                confidence=0.0,
                timestamp=time.time()
            )
            
        # Get memory statistics
        memory_stats = self.memory_manager.get_memory_statistics()
        
        # Analyze memory size
        total_entries = memory_stats.get('total_entries', 0)
        if total_entries > 1000:
            findings.append({
                'type': 'large_memory',
                'count': total_entries,
                'severity': 'medium',
                'description': f'Large memory footprint: {total_entries} entries'
            })
            
            recommendations.append({
                'type': 'memory_cleanup',
                'priority': 'medium',
                'description': 'Clean up memory to reduce footprint',
                'estimated_impact': 'memory_efficiency'
            })
            
        # Analyze duplicate facts
        facts = self.memory_manager.get_all_facts()
        duplicates = self._find_duplicate_facts(facts)
        if duplicates:
            findings.append({
                'type': 'duplicate_facts',
                'count': len(duplicates),
                'severity': 'low',
                'description': f'Found {len(duplicates)} duplicate facts'
            })
            
            recommendations.append({
                'type': 'remove_duplicates',
                'priority': 'low',
                'description': 'Remove duplicate facts',
                'estimated_impact': 'memory_cleanup'
            })
            
        # Analyze memory fragmentation
        fragmentation = self._analyze_fragmentation(memory_stats)
        if fragmentation > 0.5:  # 50% fragmentation threshold
            findings.append({
                'type': 'memory_fragmentation',
                'fragmentation_ratio': fragmentation,
                'severity': 'medium',
                'description': f'High memory fragmentation: {fragmentation:.2f}'
            })
            
            recommendations.append({
                'type': 'defragment_memory',
                'priority': 'medium',
                'description': 'Defragment memory to improve efficiency',
                'estimated_impact': 'memory_efficiency'
            })
            
        # Calculate confidence
        confidence = self._calculate_confidence(findings)
        
        return AnalysisResult(
            analyzer_type='memory',
            findings=findings,
            recommendations=recommendations,
            confidence=confidence,
            timestamp=time.time()
        )
        
    def _find_duplicate_facts(self, facts) -> List[Dict[str, Any]]:
        """Find duplicate facts in memory."""
        seen = set()
        duplicates = []
        
        for fact in facts:
            fact_key = str(fact)
            if fact_key in seen:
                duplicates.append({
                    'id': getattr(fact, 'id', 'unknown'),
                    'content': str(fact)
                })
            else:
                seen.add(fact_key)
                
        return duplicates
        
    def _analyze_fragmentation(self, memory_stats) -> float:
        """Analyze memory fragmentation."""
        # Simple heuristic: check if memory is fragmented
        # This would need actual memory layout analysis in a real implementation
        total_entries = memory_stats.get('total_entries', 0)
        
        if total_entries < 100:
            return 0.0
            
        # Simulate fragmentation based on entry count
        # More entries = higher potential fragmentation
        return min(total_entries / 1000, 1.0)
        
    def _calculate_confidence(self, findings) -> float:
        """Calculate confidence in analysis results."""
        if not findings:
            return 0.0
            
        high_severity = sum(1 for f in findings if f.get('severity') == 'high')
        medium_severity = sum(1 for f in findings if f.get('severity') == 'medium')
        
        if high_severity >= 1:
            return 0.9
        elif medium_severity >= 2:
            return 0.8
        elif medium_severity >= 1:
            return 0.7
        else:
            return 0.6
