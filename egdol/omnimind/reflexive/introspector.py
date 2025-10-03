"""
Self-Introspector for OmniMind
Analyzes reasoning patterns, skills, and memory to identify optimization opportunities.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class IntrospectionType(Enum):
    """Types of introspection analysis."""
    REASONING = auto()
    SKILLS = auto()
    MEMORY = auto()
    PERFORMANCE = auto()
    RULES = auto()
    PATTERNS = auto()


class ConfidenceLevel(Enum):
    """Confidence levels for introspection results."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    VERY_HIGH = auto()


@dataclass
class IntrospectionResult:
    """Result of introspection analysis."""
    id: str
    introspection_type: IntrospectionType
    findings: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    confidence: ConfidenceLevel
    timestamp: float
    reasoning_trace: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'id': self.id,
            'type': self.introspection_type.name,
            'findings': self.findings,
            'recommendations': self.recommendations,
            'confidence': self.confidence.name,
            'timestamp': self.timestamp,
            'reasoning_trace': self.reasoning_trace
        }


class SelfIntrospector:
    """Main introspector for self-analysis and optimization."""
    
    def __init__(self, memory_manager=None, skill_router=None, planner=None):
        self.memory_manager = memory_manager
        self.skill_router = skill_router
        self.planner = planner
        self.introspection_history: List[IntrospectionResult] = []
        self.analysis_cache: Dict[str, Any] = {}
        
    def introspect(self, introspection_type: IntrospectionType, 
                   context: Dict[str, Any] = None) -> IntrospectionResult:
        """Perform introspection analysis."""
        context = context or {}
        result_id = str(uuid.uuid4())
        reasoning_trace = []
        
        reasoning_trace.append(f"Starting {introspection_type.name} introspection")
        
        # Perform analysis based on type
        if introspection_type == IntrospectionType.REASONING:
            findings, recommendations = self._analyze_reasoning(context, reasoning_trace)
        elif introspection_type == IntrospectionType.SKILLS:
            findings, recommendations = self._analyze_skills(context, reasoning_trace)
        elif introspection_type == IntrospectionType.MEMORY:
            findings, recommendations = self._analyze_memory(context, reasoning_trace)
        elif introspection_type == IntrospectionType.PERFORMANCE:
            findings, recommendations = self._analyze_performance(context, reasoning_trace)
        elif introspection_type == IntrospectionType.RULES:
            findings, recommendations = self._analyze_rules(context, reasoning_trace)
        elif introspection_type == IntrospectionType.PATTERNS:
            findings, recommendations = self._analyze_patterns(context, reasoning_trace)
        else:
            findings, recommendations = [], []
            
        # Determine confidence level
        confidence = self._calculate_confidence(findings, recommendations)
        
        # Create result
        result = IntrospectionResult(
            id=result_id,
            introspection_type=introspection_type,
            findings=findings,
            recommendations=recommendations,
            confidence=confidence,
            timestamp=time.time(),
            reasoning_trace=reasoning_trace
        )
        
        # Store result
        self.introspection_history.append(result)
        
        return result
        
    def _analyze_reasoning(self, context: Dict[str, Any], 
                          reasoning_trace: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze reasoning patterns and efficiency."""
        findings = []
        recommendations = []
        
        reasoning_trace.append("Analyzing reasoning patterns")
        
        # Analyze rule efficiency
        if self.memory_manager:
            rules = self.memory_manager.get_all_rules()
            facts = self.memory_manager.get_all_facts()
            
            # Find unused rules
            unused_rules = []
            for rule in rules:
                if not self._is_rule_used(rule, facts):
                    unused_rules.append(rule)
                    
            if unused_rules:
                findings.append({
                    'type': 'unused_rules',
                    'count': len(unused_rules),
                    'rules': [rule.to_dict() for rule in unused_rules[:5]],  # Limit for readability
                    'severity': 'medium'
                })
                
                recommendations.append({
                    'type': 'remove_unused_rules',
                    'description': f'Remove {len(unused_rules)} unused rules to improve efficiency',
                    'priority': 'medium',
                    'estimated_impact': 'memory_reduction'
                })
                
        # Analyze reasoning complexity
        if self.planner:
            goals = self.planner.get_all_goals()
            total_tasks = sum(len(goal['tasks']) for goal in goals['active_goals'])
            
            if total_tasks > 10:
                findings.append({
                    'type': 'high_complexity',
                    'total_tasks': total_tasks,
                    'severity': 'low'
                })
                
                recommendations.append({
                    'type': 'simplify_goals',
                    'description': 'Consider breaking down complex goals into smaller ones',
                    'priority': 'low',
                    'estimated_impact': 'reasoning_efficiency'
                })
                
        reasoning_trace.append(f"Found {len(findings)} issues and {len(recommendations)} recommendations")
        
        return findings, recommendations
        
    def _analyze_skills(self, context: Dict[str, Any], 
                       reasoning_trace: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze skill usage and performance."""
        findings = []
        recommendations = []
        
        reasoning_trace.append("Analyzing skill usage patterns")
        
        if self.skill_router:
            # Get skill statistics
            skill_stats = self.skill_router.get_skill_statistics()
            
            # Find underperforming skills
            for skill_name, stats in skill_stats.items():
                if stats.get('success_rate', 1.0) < 0.7:
                    findings.append({
                        'type': 'underperforming_skill',
                        'skill': skill_name,
                        'success_rate': stats.get('success_rate', 0),
                        'execution_count': stats.get('execution_count', 0),
                        'severity': 'high'
                    })
                    
                    recommendations.append({
                        'type': 'improve_skill',
                        'description': f'Improve skill {skill_name} (success rate: {stats.get("success_rate", 0):.2f})',
                        'priority': 'high',
                        'estimated_impact': 'task_success_rate'
                    })
                    
            # Find rarely used skills
            for skill_name, stats in skill_stats.items():
                if stats.get('execution_count', 0) < 3:
                    findings.append({
                        'type': 'rarely_used_skill',
                        'skill': skill_name,
                        'execution_count': stats.get('execution_count', 0),
                        'severity': 'low'
                    })
                    
                    recommendations.append({
                        'type': 'review_skill_necessity',
                        'description': f'Review if skill {skill_name} is still needed',
                        'priority': 'low',
                        'estimated_impact': 'skill_maintenance'
                    })
                    
        reasoning_trace.append(f"Analyzed {len(skill_stats) if self.skill_router else 0} skills")
        
        return findings, recommendations
        
    def _analyze_memory(self, context: Dict[str, Any], 
                       reasoning_trace: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze memory usage and organization."""
        findings = []
        recommendations = []
        
        reasoning_trace.append("Analyzing memory usage")
        
        if self.memory_manager:
            # Get memory statistics
            memory_stats = self.memory_manager.get_memory_statistics()
            
            # Check memory size
            total_entries = memory_stats.get('total_entries', 0)
            if total_entries > 1000:
                findings.append({
                    'type': 'large_memory',
                    'total_entries': total_entries,
                    'severity': 'medium'
                })
                
                recommendations.append({
                    'type': 'memory_cleanup',
                    'description': f'Consider cleaning up memory ({total_entries} entries)',
                    'priority': 'medium',
                    'estimated_impact': 'memory_efficiency'
                })
                
            # Check for duplicate facts
            facts = self.memory_manager.get_all_facts()
            duplicates = self._find_duplicate_facts(facts)
            
            if duplicates:
                findings.append({
                    'type': 'duplicate_facts',
                    'count': len(duplicates),
                    'severity': 'low'
                })
                
                recommendations.append({
                    'type': 'remove_duplicates',
                    'description': f'Remove {len(duplicates)} duplicate facts',
                    'priority': 'low',
                    'estimated_impact': 'memory_cleanup'
                })
                
        reasoning_trace.append(f"Memory analysis complete")
        
        return findings, recommendations
        
    def _analyze_performance(self, context: Dict[str, Any], 
                           reasoning_trace: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze performance metrics and bottlenecks."""
        findings = []
        recommendations = []
        
        reasoning_trace.append("Analyzing performance metrics")
        
        # Analyze execution times
        if self.planner:
            execution_stats = self.planner.get_execution_statistics()
            
            avg_execution_time = execution_stats.get('average_execution_time', 0)
            if avg_execution_time > 5.0:  # 5 seconds
                findings.append({
                    'type': 'slow_execution',
                    'average_time': avg_execution_time,
                    'severity': 'high'
                })
                
                recommendations.append({
                    'type': 'optimize_execution',
                    'description': f'Optimize execution (avg: {avg_execution_time:.2f}s)',
                    'priority': 'high',
                    'estimated_impact': 'performance'
                })
                
        # Analyze memory usage
        import psutil
        memory_usage = psutil.virtual_memory().percent
        
        if memory_usage > 80:
            findings.append({
                'type': 'high_memory_usage',
                'usage_percent': memory_usage,
                'severity': 'high'
            })
            
            recommendations.append({
                'type': 'memory_optimization',
                'description': f'Optimize memory usage ({memory_usage:.1f}%)',
                'priority': 'high',
                'estimated_impact': 'system_stability'
            })
            
        reasoning_trace.append(f"Performance analysis complete")
        
        return findings, recommendations
        
    def _analyze_rules(self, context: Dict[str, Any], 
                      reasoning_trace: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze rule efficiency and conflicts."""
        findings = []
        recommendations = []
        
        reasoning_trace.append("Analyzing rule efficiency")
        
        if self.memory_manager:
            rules = self.memory_manager.get_all_rules()
            
            # Find conflicting rules
            conflicts = self._find_rule_conflicts(rules)
            if conflicts:
                findings.append({
                    'type': 'rule_conflicts',
                    'count': len(conflicts),
                    'conflicts': conflicts[:3],  # Limit for readability
                    'severity': 'high'
                })
                
                recommendations.append({
                    'type': 'resolve_conflicts',
                    'description': f'Resolve {len(conflicts)} rule conflicts',
                    'priority': 'high',
                    'estimated_impact': 'reasoning_consistency'
                })
                
            # Find overly complex rules
            complex_rules = [rule for rule in rules if self._is_rule_complex(rule)]
            if complex_rules:
                findings.append({
                    'type': 'complex_rules',
                    'count': len(complex_rules),
                    'severity': 'medium'
                })
                
                recommendations.append({
                    'type': 'simplify_rules',
                    'description': f'Simplify {len(complex_rules)} complex rules',
                    'priority': 'medium',
                    'estimated_impact': 'reasoning_efficiency'
                })
                
        reasoning_trace.append(f"Rule analysis complete")
        
        return findings, recommendations
        
    def _analyze_patterns(self, context: Dict[str, Any], 
                         reasoning_trace: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze usage patterns and trends."""
        findings = []
        recommendations = []
        
        reasoning_trace.append("Analyzing usage patterns")
        
        # Analyze introspection history
        if len(self.introspection_history) > 10:
            recent_results = self.introspection_history[-10:]
            
            # Find recurring issues
            issue_types = {}
            for result in recent_results:
                for finding in result.findings:
                    issue_type = finding.get('type', 'unknown')
                    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                    
            recurring_issues = {k: v for k, v in issue_types.items() if v >= 3}
            
            if recurring_issues:
                findings.append({
                    'type': 'recurring_issues',
                    'issues': recurring_issues,
                    'severity': 'medium'
                })
                
                recommendations.append({
                    'type': 'address_recurring_issues',
                    'description': f'Address recurring issues: {list(recurring_issues.keys())}',
                    'priority': 'medium',
                    'estimated_impact': 'system_stability'
                })
                
        reasoning_trace.append(f"Pattern analysis complete")
        
        return findings, recommendations
        
    def _calculate_confidence(self, findings: List[Dict[str, Any]], 
                             recommendations: List[Dict[str, Any]]) -> ConfidenceLevel:
        """Calculate confidence level for introspection result."""
        if not findings and not recommendations:
            return ConfidenceLevel.LOW
            
        # Base confidence on number of findings and their severity
        high_severity_count = sum(1 for f in findings if f.get('severity') == 'high')
        medium_severity_count = sum(1 for f in findings if f.get('severity') == 'medium')
        
        if high_severity_count >= 3:
            return ConfidenceLevel.VERY_HIGH
        elif high_severity_count >= 1 or medium_severity_count >= 3:
            return ConfidenceLevel.HIGH
        elif medium_severity_count >= 1 or len(findings) >= 3:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
            
    def _is_rule_used(self, rule, facts) -> bool:
        """Check if a rule is being used."""
        # Simple heuristic: check if rule conditions match any facts
        for fact in facts:
            if hasattr(rule, 'conditions') and fact in rule.conditions:
                return True
        return False
        
    def _find_duplicate_facts(self, facts) -> List[Dict[str, Any]]:
        """Find duplicate facts in memory."""
        seen = set()
        duplicates = []
        
        for fact in facts:
            fact_key = str(fact)
            if fact_key in seen:
                duplicates.append(fact.to_dict() if hasattr(fact, 'to_dict') else str(fact))
            else:
                seen.add(fact_key)
                
        return duplicates
        
    def _find_rule_conflicts(self, rules) -> List[Dict[str, Any]]:
        """Find conflicting rules."""
        conflicts = []
        
        # Simple heuristic: rules with opposite conclusions
        for i, rule1 in enumerate(rules):
            for j, rule2 in enumerate(rules[i+1:], i+1):
                if self._rules_conflict(rule1, rule2):
                    conflicts.append({
                        'rule1': rule1.to_dict() if hasattr(rule1, 'to_dict') else str(rule1),
                        'rule2': rule2.to_dict() if hasattr(rule2, 'to_dict') else str(rule2)
                    })
                    
        return conflicts
        
    def _rules_conflict(self, rule1, rule2) -> bool:
        """Check if two rules conflict."""
        # Simple heuristic: check if rules have opposite conclusions
        if hasattr(rule1, 'conclusion') and hasattr(rule2, 'conclusion'):
            return rule1.conclusion != rule2.conclusion
        elif isinstance(rule1, dict) and isinstance(rule2, dict):
            # Handle dictionary rules
            conclusion1 = rule1.get('conclusion', '')
            conclusion2 = rule2.get('conclusion', '')
            return conclusion1 != conclusion2 and conclusion1 and conclusion2
        return False
        
    def _is_rule_complex(self, rule) -> bool:
        """Check if a rule is overly complex."""
        # Simple heuristic: check rule length or number of conditions
        if hasattr(rule, 'conditions'):
            return len(rule.conditions) > 5
        return len(str(rule)) > 100
        
    def get_introspection_history(self) -> List[Dict[str, Any]]:
        """Get introspection history."""
        return [result.to_dict() for result in self.introspection_history]
        
    def get_introspection_stats(self) -> Dict[str, Any]:
        """Get introspection statistics."""
        if not self.introspection_history:
            return {
                'total_introspections': 0,
                'average_findings': 0,
                'average_recommendations': 0,
                'confidence_distribution': {}
            }
            
        total_introspections = len(self.introspection_history)
        total_findings = sum(len(result.findings) for result in self.introspection_history)
        total_recommendations = sum(len(result.recommendations) for result in self.introspection_history)
        
        # Calculate confidence distribution
        confidence_dist = {}
        for result in self.introspection_history:
            conf = result.confidence.name
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
            
        return {
            'total_introspections': total_introspections,
            'average_findings': total_findings / total_introspections,
            'average_recommendations': total_recommendations / total_introspections,
            'confidence_distribution': confidence_dist
        }
