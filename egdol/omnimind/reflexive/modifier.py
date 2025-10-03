"""
Self-Modifier for OmniMind
Handles self-modification operations and maintains modification logs.
"""

import time
import uuid
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum, auto


class ModificationType(Enum):
    """Types of modifications."""
    RULE_ADDITION = auto()
    RULE_REMOVAL = auto()
    RULE_MODIFICATION = auto()
    FACT_ADDITION = auto()
    FACT_REMOVAL = auto()
    SKILL_ADDITION = auto()
    SKILL_REMOVAL = auto()
    SKILL_MODIFICATION = auto()
    MEMORY_CLEANUP = auto()
    CONFIGURATION_CHANGE = auto()


class ModificationStatus(Enum):
    """Status of modifications."""
    PENDING = auto()
    APPLIED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


@dataclass
class ModificationLog:
    """Log entry for a modification."""
    id: str
    modification_type: ModificationType
    status: ModificationStatus
    description: str
    changes: Dict[str, Any]
    reasoning: str
    confidence_score: float
    timestamp: float
    rollback_data: Optional[Dict[str, Any]]
    performance_impact: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log to dictionary."""
        return {
            'id': self.id,
            'type': self.modification_type.name,
            'status': self.status.name,
            'description': self.description,
            'changes': self.changes,
            'reasoning': self.reasoning,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp,
            'rollback_data': self.rollback_data,
            'performance_impact': self.performance_impact
        }


class SelfModifier:
    """Handles self-modification operations with logging and rollback."""
    
    def __init__(self, memory_manager=None, skill_router=None, planner=None):
        self.memory_manager = memory_manager
        self.skill_router = skill_router
        self.planner = planner
        self.modification_logs: List[ModificationLog] = []
        self.rollback_stack: List[Dict[str, Any]] = []
        
    def apply_modification(self, modification_type: ModificationType,
                          changes: Dict[str, Any], reasoning: str,
                          confidence_score: float = 0.5) -> ModificationLog:
        """Apply a self-modification with logging."""
        log_id = str(uuid.uuid4())
        
        # Create modification log
        log = ModificationLog(
            id=log_id,
            modification_type=modification_type,
            status=ModificationStatus.PENDING,
            description=self._generate_description(modification_type, changes),
            changes=changes,
            reasoning=reasoning,
            confidence_score=confidence_score,
            timestamp=time.time(),
            rollback_data=None,
            performance_impact=None
        )
        
        try:
            # Apply the modification
            rollback_data = self._apply_changes(modification_type, changes)
            log.rollback_data = rollback_data
            log.status = ModificationStatus.APPLIED
            
            # Measure performance impact
            log.performance_impact = self._measure_performance_impact()
            
        except Exception as e:
            log.status = ModificationStatus.FAILED
            log.reasoning += f" (Failed: {str(e)})"
            
        # Store log
        self.modification_logs.append(log)
        
        return log
        
    def _generate_description(self, modification_type: ModificationType, 
                            changes: Dict[str, Any]) -> str:
        """Generate a description for the modification."""
        if modification_type == ModificationType.RULE_ADDITION:
            return f"Added rule: {changes.get('rule_name', 'Unknown')}"
        elif modification_type == ModificationType.RULE_REMOVAL:
            return f"Removed rule: {changes.get('rule_id', 'Unknown')}"
        elif modification_type == ModificationType.RULE_MODIFICATION:
            return f"Modified rule: {changes.get('rule_id', 'Unknown')}"
        elif modification_type == ModificationType.FACT_ADDITION:
            return f"Added fact: {changes.get('fact_name', 'Unknown')}"
        elif modification_type == ModificationType.FACT_REMOVAL:
            return f"Removed fact: {changes.get('fact_id', 'Unknown')}"
        elif modification_type == ModificationType.SKILL_ADDITION:
            return f"Added skill: {changes.get('skill_name', 'Unknown')}"
        elif modification_type == ModificationType.SKILL_REMOVAL:
            return f"Removed skill: {changes.get('skill_name', 'Unknown')}"
        elif modification_type == ModificationType.SKILL_MODIFICATION:
            return f"Modified skill: {changes.get('skill_name', 'Unknown')}"
        elif modification_type == ModificationType.MEMORY_CLEANUP:
            return f"Cleaned up {changes.get('items_removed', 0)} memory items"
        elif modification_type == ModificationType.CONFIGURATION_CHANGE:
            return f"Changed configuration: {changes.get('setting', 'Unknown')}"
        else:
            return f"Applied {modification_type.name}"
            
    def _apply_changes(self, modification_type: ModificationType, 
                      changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the actual changes and return rollback data."""
        rollback_data = {
            'modification_type': modification_type,
            'timestamp': time.time(),
            'original_state': {}
        }
        
        if modification_type == ModificationType.RULE_ADDITION:
            if self.memory_manager:
                rule = changes.get('rule')
                if rule:
                    # Store original state
                    rollback_data['original_state']['rule_exists'] = False
                    
                    # Add rule
                    self.memory_manager.add_rule(rule)
                    rollback_data['rule_id'] = rule.get('id')
                    
        elif modification_type == ModificationType.RULE_REMOVAL:
            if self.memory_manager:
                rule_id = changes.get('rule_id')
                if rule_id:
                    # Store original rule for rollback
                    original_rule = self.memory_manager.get_rule(rule_id)
                    rollback_data['original_state']['rule'] = original_rule
                    
                    # Remove rule
                    self.memory_manager.remove_rule(rule_id)
                    
        elif modification_type == ModificationType.RULE_MODIFICATION:
            if self.memory_manager:
                rule_id = changes.get('rule_id')
                new_rule = changes.get('new_rule')
                if rule_id and new_rule:
                    # Store original rule
                    original_rule = self.memory_manager.get_rule(rule_id)
                    rollback_data['original_state']['rule'] = original_rule
                    
                    # Update rule
                    self.memory_manager.update_rule(rule_id, new_rule)
                    
        elif modification_type == ModificationType.FACT_ADDITION:
            if self.memory_manager:
                fact = changes.get('fact')
                if fact:
                    # Store original state
                    rollback_data['original_state']['fact_exists'] = False
                    
                    # Add fact
                    self.memory_manager.add_fact(fact)
                    rollback_data['fact_id'] = fact.get('id')
                    
        elif modification_type == ModificationType.FACT_REMOVAL:
            if self.memory_manager:
                fact_id = changes.get('fact_id')
                if fact_id:
                    # Store original fact
                    original_fact = self.memory_manager.get_fact(fact_id)
                    rollback_data['original_state']['fact'] = original_fact
                    
                    # Remove fact
                    self.memory_manager.remove_fact(fact_id)
                    
        elif modification_type == ModificationType.SKILL_ADDITION:
            if self.skill_router:
                skill_name = changes.get('skill_name')
                skill_definition = changes.get('skill_definition')
                if skill_name and skill_definition:
                    # Store original state
                    rollback_data['original_state']['skill_exists'] = False
                    
                    # Add skill
                    self.skill_router.add_skill(skill_name, skill_definition)
                    
        elif modification_type == ModificationType.SKILL_REMOVAL:
            if self.skill_router:
                skill_name = changes.get('skill_name')
                if skill_name:
                    # Store original skill for rollback
                    original_skill = self.skill_router.get_skill(skill_name)
                    rollback_data['original_state']['skill'] = original_skill
                    
                    # Remove skill
                    self.skill_router.remove_skill(skill_name)
                    
        elif modification_type == ModificationType.SKILL_MODIFICATION:
            if self.skill_router:
                skill_name = changes.get('skill_name')
                new_skill = changes.get('new_skill')
                if skill_name and new_skill:
                    # Store original skill
                    original_skill = self.skill_router.get_skill(skill_name)
                    rollback_data['original_state']['skill'] = original_skill
                    
                    # Update skill
                    self.skill_router.update_skill(skill_name, new_skill)
                    
        elif modification_type == ModificationType.MEMORY_CLEANUP:
            if self.memory_manager:
                # Store original memory state
                original_memory = getattr(self.memory_manager, 'get_memory_snapshot', lambda: {})()
                rollback_data['original_state']['memory'] = original_memory
                
                # Clean up memory
                items_removed = getattr(self.memory_manager, 'cleanup_old_entries', lambda: 0)()
                rollback_data['items_removed'] = items_removed
                
        elif modification_type == ModificationType.CONFIGURATION_CHANGE:
            setting = changes.get('setting')
            new_value = changes.get('new_value')
            if setting and new_value:
                # Store original configuration
                rollback_data['original_state']['config'] = {
                    'setting': setting,
                    'value': getattr(self, setting, None)
                }
                
                # Apply configuration change
                setattr(self, setting, new_value)
                
        return rollback_data
        
    def _measure_performance_impact(self) -> Dict[str, Any]:
        """Measure the performance impact of modifications."""
        impact = {}
        
        # Measure memory usage
        import psutil
        memory_usage = psutil.virtual_memory().percent
        impact['memory_usage_percent'] = memory_usage
        
        # Measure execution time (if available)
        if self.planner:
            execution_stats = self.planner.get_execution_statistics()
            impact['average_execution_time'] = execution_stats.get('average_execution_time', 0)
            
        # Measure memory size (if available)
        if self.memory_manager:
            memory_stats = self.memory_manager.get_memory_statistics()
            impact['memory_entries'] = memory_stats.get('total_entries', 0)
            
        return impact
        
    def rollback_modification(self, log_id: str) -> bool:
        """Rollback a specific modification."""
        # Find the modification log
        log = None
        for l in self.modification_logs:
            if l.id == log_id:
                log = l
                break
                
        if not log or not log.rollback_data:
            return False
            
        try:
            # Apply rollback
            self._apply_rollback(log.rollback_data)
            
            # Update status
            log.status = ModificationStatus.ROLLED_BACK
            
            return True
            
        except Exception as e:
            return False
            
    def _apply_rollback(self, rollback_data: Dict[str, Any]):
        """Apply rollback data to restore original state."""
        modification_type = rollback_data.get('modification_type')
        original_state = rollback_data.get('original_state', {})
        
        if modification_type == ModificationType.RULE_ADDITION:
            if self.memory_manager:
                rule_id = rollback_data.get('rule_id')
                if rule_id:
                    self.memory_manager.remove_rule(rule_id)
                    
        elif modification_type == ModificationType.RULE_REMOVAL:
            if self.memory_manager:
                original_rule = original_state.get('rule')
                if original_rule:
                    self.memory_manager.add_rule(original_rule)
                    
        elif modification_type == ModificationType.RULE_MODIFICATION:
            if self.memory_manager:
                rule_id = rollback_data.get('rule_id')
                original_rule = original_state.get('rule')
                if rule_id and original_rule:
                    self.memory_manager.update_rule(rule_id, original_rule)
                    
        elif modification_type == ModificationType.FACT_ADDITION:
            if self.memory_manager:
                fact_id = rollback_data.get('fact_id')
                if fact_id:
                    self.memory_manager.remove_fact(fact_id)
                    
        elif modification_type == ModificationType.FACT_REMOVAL:
            if self.memory_manager:
                original_fact = original_state.get('fact')
                if original_fact:
                    self.memory_manager.add_fact(original_fact)
                    
        elif modification_type == ModificationType.SKILL_ADDITION:
            if self.skill_router:
                skill_name = rollback_data.get('skill_name')
                if skill_name:
                    self.skill_router.remove_skill(skill_name)
                    
        elif modification_type == ModificationType.SKILL_REMOVAL:
            if self.skill_router:
                original_skill = original_state.get('skill')
                if original_skill:
                    skill_name = original_skill.get('name')
                    if skill_name:
                        self.skill_router.add_skill(skill_name, original_skill)
                        
        elif modification_type == ModificationType.SKILL_MODIFICATION:
            if self.skill_router:
                skill_name = rollback_data.get('skill_name')
                original_skill = original_state.get('skill')
                if skill_name and original_skill:
                    self.skill_router.update_skill(skill_name, original_skill)
                    
        elif modification_type == ModificationType.MEMORY_CLEANUP:
            if self.memory_manager:
                original_memory = original_state.get('memory')
                if original_memory:
                    self.memory_manager.restore_memory_snapshot(original_memory)
                    
        elif modification_type == ModificationType.CONFIGURATION_CHANGE:
            original_config = original_state.get('config')
            if original_config:
                setting = original_config.get('setting')
                value = original_config.get('value')
                if setting:
                    setattr(self, setting, value)
                    
    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Get modification history."""
        return [log.to_dict() for log in self.modification_logs]
        
    def get_modification_stats(self) -> Dict[str, Any]:
        """Get modification statistics."""
        if not self.modification_logs:
            return {
                'total_modifications': 0,
                'successful_modifications': 0,
                'failed_modifications': 0,
                'rolled_back_modifications': 0,
                'average_confidence': 0
            }
            
        total = len(self.modification_logs)
        successful = sum(1 for log in self.modification_logs 
                     if log.status == ModificationStatus.APPLIED)
        failed = sum(1 for log in self.modification_logs 
                    if log.status == ModificationStatus.FAILED)
        rolled_back = sum(1 for log in self.modification_logs 
                         if log.status == ModificationStatus.ROLLED_BACK)
        
        average_confidence = sum(log.confidence_score for log in self.modification_logs) / total
        
        return {
            'total_modifications': total,
            'successful_modifications': successful,
            'failed_modifications': failed,
            'rolled_back_modifications': rolled_back,
            'success_rate': successful / total if total > 0 else 0,
            'average_confidence': average_confidence
        }
        
    def export_modification_log(self, filepath: str) -> bool:
        """Export modification log to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_modification_history(), f, indent=2)
            return True
        except Exception:
            return False
            
    def import_modification_log(self, filepath: str) -> bool:
        """Import modification log from file."""
        try:
            with open(filepath, 'r') as f:
                logs = json.load(f)
                
            # Clear existing logs
            self.modification_logs.clear()
            
            # Import logs
            for log_data in logs:
                log = ModificationLog(
                    id=log_data['id'],
                    modification_type=ModificationType[log_data['type']],
                    status=ModificationStatus[log_data['status']],
                    description=log_data['description'],
                    changes=log_data['changes'],
                    reasoning=log_data['reasoning'],
                    confidence_score=log_data['confidence_score'],
                    timestamp=log_data['timestamp'],
                    rollback_data=log_data.get('rollback_data'),
                    performance_impact=log_data.get('performance_impact')
                )
                self.modification_logs.append(log)
                
            return True
        except Exception:
            return False
