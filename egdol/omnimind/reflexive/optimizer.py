"""
Self-Optimizer for OmniMind
Implements optimization strategies based on introspection results.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class OptimizationType(Enum):
    """Types of optimizations."""
    RULE_OPTIMIZATION = auto()
    SKILL_OPTIMIZATION = auto()
    MEMORY_OPTIMIZATION = auto()
    PERFORMANCE_OPTIMIZATION = auto()
    REASONING_OPTIMIZATION = auto()


class OptimizationStatus(Enum):
    """Status of optimization."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


@dataclass
class OptimizationResult:
    """Result of optimization operation."""
    id: str
    optimization_type: OptimizationType
    status: OptimizationStatus
    changes_made: List[Dict[str, Any]]
    performance_impact: Dict[str, Any]
    rollback_data: Optional[Dict[str, Any]]
    timestamp: float
    reasoning_trace: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'id': self.id,
            'type': self.optimization_type.name,
            'status': self.status.name,
            'changes_made': self.changes_made,
            'performance_impact': self.performance_impact,
            'rollback_data': self.rollback_data,
            'timestamp': self.timestamp,
            'reasoning_trace': self.reasoning_trace
        }


class SelfOptimizer:
    """Optimizes OmniMind based on introspection results."""
    
    def __init__(self, memory_manager=None, skill_router=None, planner=None):
        self.memory_manager = memory_manager
        self.skill_router = skill_router
        self.planner = planner
        self.optimization_history: List[OptimizationResult] = []
        self.rollback_stack: List[Dict[str, Any]] = []
        
    def optimize(self, introspection_result, auto_apply: bool = False) -> OptimizationResult:
        """Apply optimizations based on introspection results."""
        result_id = str(uuid.uuid4())
        reasoning_trace = []
        
        reasoning_trace.append(f"Starting optimization based on {introspection_result.introspection_type.name}")
        
        # Determine optimization type
        optimization_type = self._determine_optimization_type(introspection_result)
        
        # Create optimization result
        result = OptimizationResult(
            id=result_id,
            optimization_type=optimization_type,
            status=OptimizationStatus.PENDING,
            changes_made=[],
            performance_impact={},
            rollback_data=None,
            timestamp=time.time(),
            reasoning_trace=reasoning_trace
        )
        
        # Apply optimizations
        if auto_apply:
            result = self._apply_optimizations(result, introspection_result)
        else:
            result.status = OptimizationStatus.PENDING
            
        # Store result
        self.optimization_history.append(result)
        
        return result
        
    def _determine_optimization_type(self, introspection_result) -> OptimizationType:
        """Determine the type of optimization needed."""
        findings = introspection_result.findings
        
        # Check for rule-related issues
        if any(f.get('type') in ['unused_rules', 'rule_conflicts', 'complex_rules'] for f in findings):
            return OptimizationType.RULE_OPTIMIZATION
            
        # Check for skill-related issues
        if any(f.get('type') in ['underperforming_skill', 'rarely_used_skill'] for f in findings):
            return OptimizationType.SKILL_OPTIMIZATION
            
        # Check for memory-related issues
        if any(f.get('type') in ['large_memory', 'duplicate_facts'] for f in findings):
            return OptimizationType.MEMORY_OPTIMIZATION
            
        # Check for performance issues
        if any(f.get('type') in ['slow_execution', 'high_memory_usage'] for f in findings):
            return OptimizationType.PERFORMANCE_OPTIMIZATION
            
        # Default to reasoning optimization
        return OptimizationType.REASONING_OPTIMIZATION
        
    def _apply_optimizations(self, result: OptimizationResult, 
                           introspection_result) -> OptimizationResult:
        """Apply the actual optimizations."""
        result.status = OptimizationStatus.IN_PROGRESS
        result.reasoning_trace.append("Applying optimizations")
        
        try:
            if result.optimization_type == OptimizationType.RULE_OPTIMIZATION:
                result = self._optimize_rules(result, introspection_result)
            elif result.optimization_type == OptimizationType.SKILL_OPTIMIZATION:
                result = self._optimize_skills(result, introspection_result)
            elif result.optimization_type == OptimizationType.MEMORY_OPTIMIZATION:
                result = self._optimize_memory(result, introspection_result)
            elif result.optimization_type == OptimizationType.PERFORMANCE_OPTIMIZATION:
                result = self._optimize_performance(result, introspection_result)
            else:
                result = self._optimize_reasoning(result, introspection_result)
                
            result.status = OptimizationStatus.COMPLETED
            result.reasoning_trace.append("Optimization completed successfully")
            
        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.reasoning_trace.append(f"Optimization failed: {str(e)}")
            
        return result
        
    def _optimize_rules(self, result: OptimizationResult, 
                       introspection_result) -> OptimizationResult:
        """Optimize rules based on findings."""
        result.reasoning_trace.append("Optimizing rules")
        
        changes = []
        
        # Remove unused rules
        for finding in introspection_result.findings:
            if finding.get('type') == 'unused_rules':
                unused_rules = finding.get('rules', [])
                for rule in unused_rules:
                    if self.memory_manager:
                        # Store rollback data
                        rollback_data = {
                            'action': 'remove_rule',
                            'rule': rule,
                            'timestamp': time.time()
                        }
                        self.rollback_stack.append(rollback_data)
                        
                        # Remove rule
                        self.memory_manager.remove_rule(rule.get('id'))
                        changes.append({
                            'action': 'removed_unused_rule',
                            'rule_id': rule.get('id'),
                            'reason': 'unused'
                        })
                        
        # Resolve rule conflicts
        for finding in introspection_result.findings:
            if finding.get('type') == 'rule_conflicts':
                conflicts = finding.get('conflicts', [])
                for conflict in conflicts:
                    # Simple resolution: keep the more recent rule
                    rule1 = conflict.get('rule1', {})
                    rule2 = conflict.get('rule2', {})
                    
                    if self.memory_manager:
                        # Store rollback data
                        rollback_data = {
                            'action': 'resolve_conflict',
                            'removed_rule': rule2,
                            'kept_rule': rule1,
                            'timestamp': time.time()
                        }
                        self.rollback_stack.append(rollback_data)
                        
                        # Remove conflicting rule
                        self.memory_manager.remove_rule(rule2.get('id'))
                        changes.append({
                            'action': 'resolved_rule_conflict',
                            'removed_rule_id': rule2.get('id'),
                            'kept_rule_id': rule1.get('id')
                        })
                        
        result.changes_made = changes
        result.rollback_data = self.rollback_stack[-len(changes):] if changes else None
        
        return result
        
    def _optimize_skills(self, result: OptimizationResult, 
                        introspection_result) -> OptimizationResult:
        """Optimize skills based on findings."""
        result.reasoning_trace.append("Optimizing skills")
        
        changes = []
        
        # Improve underperforming skills
        for finding in introspection_result.findings:
            if finding.get('type') == 'underperforming_skill':
                skill_name = finding.get('skill')
                success_rate = finding.get('success_rate', 0)
                
                if self.skill_router:
                    # Try to improve skill performance
                    improvement = self.skill_router.improve_skill(skill_name)
                    
                    if improvement:
                        changes.append({
                            'action': 'improved_skill',
                            'skill_name': skill_name,
                            'improvement': improvement,
                            'previous_success_rate': success_rate
                        })
                        
        # Remove rarely used skills
        for finding in introspection_result.findings:
            if finding.get('type') == 'rarely_used_skill':
                skill_name = finding.get('skill')
                execution_count = finding.get('execution_count', 0)
                
                if execution_count == 0 and self.skill_router:
                    # Store rollback data
                    rollback_data = {
                        'action': 'remove_skill',
                        'skill_name': skill_name,
                        'timestamp': time.time()
                    }
                    self.rollback_stack.append(rollback_data)
                    
                    # Remove skill
                    self.skill_router.remove_skill(skill_name)
                    changes.append({
                        'action': 'removed_unused_skill',
                        'skill_name': skill_name,
                        'reason': 'never_used'
                    })
                    
        result.changes_made = changes
        result.rollback_data = self.rollback_stack[-len(changes):] if changes else None
        
        return result
        
    def _optimize_memory(self, result: OptimizationResult, 
                        introspection_result) -> OptimizationResult:
        """Optimize memory based on findings."""
        result.reasoning_trace.append("Optimizing memory")
        
        changes = []
        
        # Remove duplicate facts
        for finding in introspection_result.findings:
            if finding.get('type') == 'duplicate_facts':
                duplicates = finding.get('duplicates', [])
                
                if self.memory_manager:
                    for duplicate in duplicates:
                        # Store rollback data
                        rollback_data = {
                            'action': 'remove_duplicate_fact',
                            'fact': duplicate,
                            'timestamp': time.time()
                        }
                        self.rollback_stack.append(rollback_data)
                        
                        # Remove duplicate
                        self.memory_manager.remove_fact(duplicate.get('id'))
                        changes.append({
                            'action': 'removed_duplicate_fact',
                            'fact_id': duplicate.get('id')
                        })
                        
        # Clean up old memory entries
        if self.memory_manager:
            cleanup_count = self.memory_manager.cleanup_old_entries()
            if cleanup_count > 0:
                changes.append({
                    'action': 'cleaned_up_old_entries',
                    'count': cleanup_count
                })
                
        result.changes_made = changes
        result.rollback_data = self.rollback_stack[-len(changes):] if changes else None
        
        return result
        
    def _optimize_performance(self, result: OptimizationResult, 
                           introspection_result) -> OptimizationResult:
        """Optimize performance based on findings."""
        result.reasoning_trace.append("Optimizing performance")
        
        changes = []
        
        # Optimize execution
        for finding in introspection_result.findings:
            if finding.get('type') == 'slow_execution':
                if self.planner:
                    # Optimize task scheduling
                    optimization = self.planner.optimize_scheduling()
                    if optimization:
                        changes.append({
                            'action': 'optimized_scheduling',
                            'optimization': optimization
                        })
                        
        # Optimize memory usage
        for finding in introspection_result.findings:
            if finding.get('type') == 'high_memory_usage':
                if self.memory_manager:
                    # Compact memory
                    compacted = self.memory_manager.compact_memory()
                    if compacted:
                        changes.append({
                            'action': 'compacted_memory',
                            'compaction_ratio': compacted
                        })
                        
        result.changes_made = changes
        result.rollback_data = self.rollback_stack[-len(changes):] if changes else None
        
        return result
        
    def _optimize_reasoning(self, result: OptimizationResult, 
                          introspection_result) -> OptimizationResult:
        """Optimize reasoning based on findings."""
        result.reasoning_trace.append("Optimizing reasoning")
        
        changes = []
        
        # Simplify complex goals
        for finding in introspection_result.findings:
            if finding.get('type') == 'high_complexity':
                if self.planner:
                    # Break down complex goals
                    simplified = self.planner.simplify_complex_goals()
                    if simplified:
                        changes.append({
                            'action': 'simplified_goals',
                            'simplified_count': simplified
                        })
                        
        result.changes_made = changes
        result.rollback_data = self.rollback_stack[-len(changes):] if changes else None
        
        return result
        
    def rollback_optimization(self, optimization_id: str) -> bool:
        """Rollback a specific optimization."""
        # Find the optimization
        optimization = None
        for opt in self.optimization_history:
            if opt.id == optimization_id:
                optimization = opt
                break
                
        if not optimization or not optimization.rollback_data:
            return False
            
        try:
            # Apply rollback
            for rollback_action in optimization.rollback_data:
                self._apply_rollback(rollback_action)
                
            # Update status
            optimization.status = OptimizationStatus.ROLLED_BACK
            
            return True
            
        except Exception as e:
            return False
            
    def _apply_rollback(self, rollback_action: Dict[str, Any]):
        """Apply a rollback action."""
        action = rollback_action.get('action')
        
        if action == 'remove_rule':
            # Restore removed rule
            if self.memory_manager:
                rule = rollback_action.get('rule')
                self.memory_manager.add_rule(rule)
                
        elif action == 'remove_skill':
            # Restore removed skill
            if self.skill_router:
                skill_name = rollback_action.get('skill_name')
                # Note: This would require storing the skill definition
                pass
                
        elif action == 'remove_duplicate_fact':
            # Restore removed fact
            if self.memory_manager:
                fact = rollback_action.get('fact')
                self.memory_manager.add_fact(fact)
                
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return [result.to_dict() for result in self.optimization_history]
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_history:
            return {
                'total_optimizations': 0,
                'successful_optimizations': 0,
                'failed_optimizations': 0,
                'rolled_back_optimizations': 0
            }
            
        total = len(self.optimization_history)
        successful = sum(1 for opt in self.optimization_history 
                         if opt.status == OptimizationStatus.COMPLETED)
        failed = sum(1 for opt in self.optimization_history 
                    if opt.status == OptimizationStatus.FAILED)
        rolled_back = sum(1 for opt in self.optimization_history 
                         if opt.status == OptimizationStatus.ROLLED_BACK)
        
        return {
            'total_optimizations': total,
            'successful_optimizations': successful,
            'failed_optimizations': failed,
            'rolled_back_optimizations': rolled_back,
            'success_rate': successful / total if total > 0 else 0
        }
