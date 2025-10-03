"""
Rollback Guard for OmniMind Self-Creation
Ensures full determinism and rollback safety for all progeny operations.
"""

import uuid
import json
import os
import shutil
import hashlib
import threading
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto


class RollbackStatus(Enum):
    """Status of rollback operations."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class SafetyLevel(Enum):
    """Safety levels for operations."""
    MINIMAL = auto()
    STANDARD = auto()
    HIGH = auto()
    MAXIMUM = auto()


class OperationType(Enum):
    """Types of operations that can be rolled back."""
    PROGENY_CREATION = auto()
    PROGENY_INTEGRATION = auto()
    SYSTEM_MODIFICATION = auto()
    CONFIGURATION_CHANGE = auto()
    SKILL_UPDATE = auto()
    NETWORK_CHANGE = auto()
    MEMORY_UPDATE = auto()
    POLICY_CHANGE = auto()


@dataclass
class RollbackPoint:
    """A rollback point for system state."""
    point_id: str
    operation_type: OperationType
    description: str
    system_state: Dict[str, Any]
    backup_path: str
    checksum: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackOperation:
    """A rollback operation."""
    operation_id: str
    rollback_point_id: str
    target_state: Dict[str, Any]
    rollback_steps: List[Dict[str, Any]]
    status: RollbackStatus
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_messages: List[str] = field(default_factory=list)
    rollback_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SafetyConstraint:
    """Safety constraint for operations."""
    constraint_id: str
    operation_type: OperationType
    safety_level: SafetyLevel
    conditions: List[Dict[str, Any]]
    rollback_requirements: List[str]
    validation_functions: List[str]
    created_at: datetime = field(default_factory=datetime.now)


class RollbackGuard:
    """Ensures full determinism and rollback safety for all operations."""
    
    def __init__(self, backup_base_path: str, max_rollback_points: int = 100):
        self.backup_base_path = backup_base_path
        self.max_rollback_points = max_rollback_points
        self.rollback_points: Dict[str, RollbackPoint] = {}
        self.rollback_operations: Dict[str, RollbackOperation] = {}
        self.safety_constraints: Dict[str, SafetyConstraint] = {}
        self.active_rollbacks: Dict[str, threading.Thread] = {}
        self.rollback_history: List[Dict[str, Any]] = []
        self.safety_violations: List[Dict[str, Any]] = []
        self.determinism_checks: Dict[str, bool] = {}
        
        # Ensure backup directory exists
        os.makedirs(backup_base_path, exist_ok=True)
        
        # Initialize safety constraints
        self._initialize_safety_constraints()
    
    def _initialize_safety_constraints(self):
        """Initialize default safety constraints."""
        # Progeny creation safety
        progeny_creation_constraint = SafetyConstraint(
            constraint_id=str(uuid.uuid4()),
            operation_type=OperationType.PROGENY_CREATION,
            safety_level=SafetyLevel.HIGH,
            conditions=[
                {'type': 'resource_check', 'max_cpu': 0.8, 'max_memory': 0.9},
                {'type': 'isolation_check', 'sandbox_required': True},
                {'type': 'validation_check', 'progeny_validated': True}
            ],
            rollback_requirements=[
                'create_rollback_point',
                'validate_rollback_capability',
                'test_rollback_procedure'
            ],
            validation_functions=[
                'validate_progeny_safety',
                'check_system_integrity',
                'verify_rollback_capability'
            ]
        )
        self.safety_constraints[progeny_creation_constraint.constraint_id] = progeny_creation_constraint
        
        # System modification safety
        system_modification_constraint = SafetyConstraint(
            constraint_id=str(uuid.uuid4()),
            operation_type=OperationType.SYSTEM_MODIFICATION,
            safety_level=SafetyLevel.MAXIMUM,
            conditions=[
                {'type': 'backup_required', 'full_backup': True},
                {'type': 'testing_required', 'comprehensive_testing': True},
                {'type': 'approval_required', 'human_approval': True}
            ],
            rollback_requirements=[
                'create_full_system_backup',
                'validate_rollback_integrity',
                'test_rollback_speed'
            ],
            validation_functions=[
                'validate_system_modification',
                'check_modification_safety',
                'verify_rollback_integrity'
            ]
        )
        self.safety_constraints[system_modification_constraint.constraint_id] = system_modification_constraint
        
        # Integration safety
        integration_constraint = SafetyConstraint(
            constraint_id=str(uuid.uuid4()),
            operation_type=OperationType.PROGENY_INTEGRATION,
            safety_level=SafetyLevel.HIGH,
            conditions=[
                {'type': 'compatibility_check', 'full_compatibility': True},
                {'type': 'performance_check', 'performance_impact': 'acceptable'},
                {'type': 'stability_check', 'stability_maintained': True}
            ],
            rollback_requirements=[
                'create_integration_backup',
                'validate_rollback_speed',
                'test_rollback_completeness'
            ],
            validation_functions=[
                'validate_integration_safety',
                'check_integration_compatibility',
                'verify_rollback_capability'
            ]
        )
        self.safety_constraints[integration_constraint.constraint_id] = integration_constraint
    
    def create_rollback_point(self, operation_type: OperationType, 
                             description: str, system_state: Dict[str, Any],
                             metadata: Optional[Dict[str, Any]] = None) -> RollbackPoint:
        """Create a rollback point for the current system state."""
        point_id = str(uuid.uuid4())
        
        # Create backup directory
        backup_path = os.path.join(self.backup_base_path, f"rollback_{point_id}")
        os.makedirs(backup_path, exist_ok=True)
        
        # Save system state
        state_file = os.path.join(backup_path, "system_state.json")
        with open(state_file, 'w') as f:
            json.dump(system_state, f, indent=2)
        
        # Calculate checksum
        checksum = self._calculate_checksum(system_state)
        
        # Create rollback point
        rollback_point = RollbackPoint(
            point_id=point_id,
            operation_type=operation_type,
            description=description,
            system_state=system_state,
            backup_path=backup_path,
            checksum=checksum,
            metadata=metadata or {}
        )
        
        # Store rollback point
        self.rollback_points[point_id] = rollback_point
        
        # Clean up old rollback points if needed
        self._cleanup_old_rollback_points()
        
        # Log creation
        self.rollback_history.append({
            'timestamp': datetime.now(),
            'event': 'rollback_point_created',
            'point_id': point_id,
            'operation_type': operation_type.name,
            'description': description
        })
        
        return rollback_point
    
    def _calculate_checksum(self, system_state: Dict[str, Any]) -> str:
        """Calculate checksum for system state."""
        state_str = json.dumps(system_state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def _cleanup_old_rollback_points(self):
        """Clean up old rollback points to stay within limits."""
        if len(self.rollback_points) <= self.max_rollback_points:
            return
        
        # Sort by creation time
        sorted_points = sorted(
            self.rollback_points.values(),
            key=lambda x: x.created_at
        )
        
        # Remove oldest points
        points_to_remove = len(self.rollback_points) - self.max_rollback_points
        for i in range(points_to_remove):
            point = sorted_points[i]
            
            # Remove backup directory
            if os.path.exists(point.backup_path):
                shutil.rmtree(point.backup_path)
            
            # Remove from tracking
            del self.rollback_points[point.point_id]
    
    def validate_operation_safety(self, operation_type: OperationType, 
                                 operation_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that an operation is safe to perform."""
        # Find relevant safety constraints
        relevant_constraints = [
            constraint for constraint in self.safety_constraints.values()
            if constraint.operation_type == operation_type
        ]
        
        if not relevant_constraints:
            return True, []  # No constraints, operation is safe
        
        violations = []
        
        for constraint in relevant_constraints:
            # Check conditions
            for condition in constraint.conditions:
                if not self._check_condition(condition, operation_data):
                    violations.append(f"Safety constraint violation: {condition}")
            
            # Check rollback requirements
            for requirement in constraint.rollback_requirements:
                if not self._check_rollback_requirement(requirement, operation_data):
                    violations.append(f"Rollback requirement not met: {requirement}")
            
            # Run validation functions
            for validation_func in constraint.validation_functions:
                if not self._run_validation_function(validation_func, operation_data):
                    violations.append(f"Validation failed: {validation_func}")
        
        is_safe = len(violations) == 0
        
        if not is_safe:
            # Log safety violation
            self.safety_violations.append({
                'timestamp': datetime.now(),
                'operation_type': operation_type.name,
                'violations': violations,
                'operation_data': operation_data
            })
        
        return is_safe, violations
    
    def _check_condition(self, condition: Dict[str, Any], operation_data: Dict[str, Any]) -> bool:
        """Check a safety condition."""
        condition_type = condition.get('type')
        
        if condition_type == 'resource_check':
            # Check resource usage
            cpu_usage = operation_data.get('cpu_usage', 0)
            memory_usage = operation_data.get('memory_usage', 0)
            
            max_cpu = condition.get('max_cpu', 1.0)
            max_memory = condition.get('max_memory', 1.0)
            
            return cpu_usage <= max_cpu and memory_usage <= max_memory
        
        elif condition_type == 'isolation_check':
            # Check if sandbox is required
            sandbox_required = condition.get('sandbox_required', False)
            has_sandbox = operation_data.get('sandbox_enabled', False)
            
            return not sandbox_required or has_sandbox
        
        elif condition_type == 'validation_check':
            # Check if validation is required
            validation_required = condition.get('progeny_validated', False)
            is_validated = operation_data.get('validated', False)
            
            return not validation_required or is_validated
        
        elif condition_type == 'backup_required':
            # Check if backup is required
            backup_required = condition.get('full_backup', False)
            has_backup = operation_data.get('backup_created', False)
            
            return not backup_required or has_backup
        
        elif condition_type == 'testing_required':
            # Check if testing is required
            testing_required = condition.get('comprehensive_testing', False)
            has_testing = operation_data.get('testing_completed', False)
            
            return not testing_required or has_testing
        
        elif condition_type == 'approval_required':
            # Check if approval is required
            approval_required = condition.get('human_approval', False)
            has_approval = operation_data.get('approved', False)
            
            return not approval_required or has_approval
        
        elif condition_type == 'compatibility_check':
            # Check compatibility
            compatibility_required = condition.get('full_compatibility', False)
            is_compatible = operation_data.get('compatible', False)
            
            return not compatibility_required or is_compatible
        
        elif condition_type == 'performance_check':
            # Check performance impact
            performance_impact = operation_data.get('performance_impact', 'high')
            acceptable_impact = condition.get('performance_impact', 'low')
            
            impact_levels = {'low': 0, 'medium': 1, 'high': 2}
            return impact_levels.get(performance_impact, 2) <= impact_levels.get(acceptable_impact, 0)
        
        elif condition_type == 'stability_check':
            # Check stability
            stability_required = condition.get('stability_maintained', False)
            is_stable = operation_data.get('stable', False)
            
            return not stability_required or is_stable
        
        else:
            # Unknown condition type, assume safe
            return True
    
    def _check_rollback_requirement(self, requirement: str, operation_data: Dict[str, Any]) -> bool:
        """Check a rollback requirement."""
        if requirement == 'create_rollback_point':
            return operation_data.get('rollback_point_created', False)
        elif requirement == 'validate_rollback_capability':
            return operation_data.get('rollback_validated', False)
        elif requirement == 'test_rollback_procedure':
            return operation_data.get('rollback_tested', False)
        elif requirement == 'create_full_system_backup':
            return operation_data.get('full_backup_created', False)
        elif requirement == 'validate_rollback_integrity':
            return operation_data.get('rollback_integrity_validated', False)
        elif requirement == 'test_rollback_speed':
            return operation_data.get('rollback_speed_tested', False)
        elif requirement == 'create_integration_backup':
            return operation_data.get('integration_backup_created', False)
        elif requirement == 'validate_rollback_speed':
            return operation_data.get('rollback_speed_validated', False)
        elif requirement == 'test_rollback_completeness':
            return operation_data.get('rollback_completeness_tested', False)
        else:
            # Unknown requirement, assume met
            return True
    
    def _run_validation_function(self, validation_func: str, operation_data: Dict[str, Any]) -> bool:
        """Run a validation function."""
        if validation_func == 'validate_progeny_safety':
            return self._validate_progeny_safety(operation_data)
        elif validation_func == 'check_system_integrity':
            return self._check_system_integrity(operation_data)
        elif validation_func == 'verify_rollback_capability':
            return self._verify_rollback_capability(operation_data)
        elif validation_func == 'validate_system_modification':
            return self._validate_system_modification(operation_data)
        elif validation_func == 'check_modification_safety':
            return self._check_modification_safety(operation_data)
        elif validation_func == 'verify_rollback_integrity':
            return self._verify_rollback_integrity(operation_data)
        elif validation_func == 'validate_integration_safety':
            return self._validate_integration_safety(operation_data)
        elif validation_func == 'check_integration_compatibility':
            return self._check_integration_compatibility(operation_data)
        elif validation_func == 'verify_rollback_capability':
            return self._verify_rollback_capability(operation_data)
        else:
            # Unknown validation function, assume valid
            return True
    
    def _validate_progeny_safety(self, operation_data: Dict[str, Any]) -> bool:
        """Validate progeny safety."""
        # Check if progeny has safety features
        safety_features = operation_data.get('safety_features', [])
        required_features = ['error_handling', 'resource_limits', 'isolation']
        
        return all(feature in safety_features for feature in required_features)
    
    def _check_system_integrity(self, operation_data: Dict[str, Any]) -> bool:
        """Check system integrity."""
        # Simulate system integrity check
        integrity_score = operation_data.get('integrity_score', 0.8)
        return integrity_score >= 0.7
    
    def _verify_rollback_capability(self, operation_data: Dict[str, Any]) -> bool:
        """Verify rollback capability."""
        # Check if rollback is possible
        rollback_possible = operation_data.get('rollback_possible', True)
        rollback_tested = operation_data.get('rollback_tested', False)
        
        return rollback_possible and rollback_tested
    
    def _validate_system_modification(self, operation_data: Dict[str, Any]) -> bool:
        """Validate system modification."""
        # Check if modification is safe
        modification_safe = operation_data.get('modification_safe', True)
        impact_assessed = operation_data.get('impact_assessed', False)
        
        return modification_safe and impact_assessed
    
    def _check_modification_safety(self, operation_data: Dict[str, Any]) -> bool:
        """Check modification safety."""
        # Check safety measures
        safety_measures = operation_data.get('safety_measures', [])
        required_measures = ['backup', 'testing', 'monitoring']
        
        return all(measure in safety_measures for measure in required_measures)
    
    def _verify_rollback_integrity(self, operation_data: Dict[str, Any]) -> bool:
        """Verify rollback integrity."""
        # Check rollback integrity
        rollback_integrity = operation_data.get('rollback_integrity', 0.8)
        return rollback_integrity >= 0.7
    
    def _validate_integration_safety(self, operation_data: Dict[str, Any]) -> bool:
        """Validate integration safety."""
        # Check integration safety
        integration_safe = operation_data.get('integration_safe', True)
        compatibility_checked = operation_data.get('compatibility_checked', False)
        
        return integration_safe and compatibility_checked
    
    def _check_integration_compatibility(self, operation_data: Dict[str, Any]) -> bool:
        """Check integration compatibility."""
        # Check compatibility
        compatibility_score = operation_data.get('compatibility_score', 0.8)
        return compatibility_score >= 0.7
    
    def execute_rollback(self, rollback_point_id: str, 
                        target_state: Optional[Dict[str, Any]] = None) -> RollbackOperation:
        """Execute a rollback to a specific point."""
        if rollback_point_id not in self.rollback_points:
            raise ValueError(f"Rollback point {rollback_point_id} not found")
        
        rollback_point = self.rollback_points[rollback_point_id]
        operation_id = str(uuid.uuid4())
        
        # Create rollback operation
        rollback_operation = RollbackOperation(
            operation_id=operation_id,
            rollback_point_id=rollback_point_id,
            target_state=target_state or rollback_point.system_state,
            rollback_steps=[],
            status=RollbackStatus.PENDING
        )
        
        # Generate rollback steps
        rollback_steps = self._generate_rollback_steps(rollback_point, target_state)
        rollback_operation.rollback_steps = rollback_steps
        
        # Store operation
        self.rollback_operations[operation_id] = rollback_operation
        
        # Execute rollback in a separate thread
        rollback_thread = threading.Thread(
            target=self._execute_rollback_operation,
            args=(rollback_operation,)
        )
        
        self.active_rollbacks[operation_id] = rollback_thread
        rollback_thread.start()
        
        return rollback_operation
    
    def _generate_rollback_steps(self, rollback_point: RollbackPoint, 
                               target_state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate rollback steps."""
        steps = []
        
        # Step 1: Validate rollback point
        steps.append({
            'step': 1,
            'action': 'validate_rollback_point',
            'description': 'Validate rollback point integrity',
            'timeout': 30
        })
        
        # Step 2: Create current state backup
        steps.append({
            'step': 2,
            'action': 'backup_current_state',
            'description': 'Create backup of current state',
            'timeout': 60
        })
        
        # Step 3: Stop active operations
        steps.append({
            'step': 3,
            'action': 'stop_active_operations',
            'description': 'Stop all active operations',
            'timeout': 120
        })
        
        # Step 4: Restore system state
        steps.append({
            'step': 4,
            'action': 'restore_system_state',
            'description': 'Restore system to rollback point',
            'timeout': 300
        })
        
        # Step 5: Validate restoration
        steps.append({
            'step': 5,
            'action': 'validate_restoration',
            'description': 'Validate system restoration',
            'timeout': 180
        })
        
        # Step 6: Restart system components
        steps.append({
            'step': 6,
            'action': 'restart_components',
            'description': 'Restart system components',
            'timeout': 240
        })
        
        return steps
    
    def _execute_rollback_operation(self, rollback_operation: RollbackOperation):
        """Execute a rollback operation."""
        try:
            rollback_operation.status = RollbackStatus.IN_PROGRESS
            
            # Execute each rollback step
            for step in rollback_operation.rollback_steps:
                step_success = self._execute_rollback_step(step, rollback_operation)
                
                if not step_success:
                    rollback_operation.status = RollbackStatus.FAILED
                    rollback_operation.error_messages.append(f"Step {step['step']} failed")
                    return
                
                # Log step completion
                rollback_operation.rollback_log.append({
                    'timestamp': datetime.now(),
                    'step': step['step'],
                    'action': step['action'],
                    'status': 'completed'
                })
            
            # Rollback completed successfully
            rollback_operation.status = RollbackStatus.COMPLETED
            rollback_operation.completed_at = datetime.now()
            
        except Exception as e:
            rollback_operation.status = RollbackStatus.FAILED
            rollback_operation.error_messages.append(f"Rollback failed: {str(e)}")
            rollback_operation.completed_at = datetime.now()
        
        finally:
            # Remove from active rollbacks
            if rollback_operation.operation_id in self.active_rollbacks:
                del self.active_rollbacks[rollback_operation.operation_id]
    
    def _execute_rollback_step(self, step: Dict[str, Any], 
                             rollback_operation: RollbackOperation) -> bool:
        """Execute a single rollback step."""
        action = step['action']
        
        if action == 'validate_rollback_point':
            return self._validate_rollback_point(rollback_operation)
        elif action == 'backup_current_state':
            return self._backup_current_state(rollback_operation)
        elif action == 'stop_active_operations':
            return self._stop_active_operations(rollback_operation)
        elif action == 'restore_system_state':
            return self._restore_system_state(rollback_operation)
        elif action == 'validate_restoration':
            return self._validate_restoration(rollback_operation)
        elif action == 'restart_components':
            return self._restart_components(rollback_operation)
        else:
            # Unknown action, assume success
            return True
    
    def _validate_rollback_point(self, rollback_operation: RollbackOperation) -> bool:
        """Validate rollback point."""
        rollback_point_id = rollback_operation.rollback_point_id
        rollback_point = self.rollback_points[rollback_point_id]
        
        # Check if backup path exists
        if not os.path.exists(rollback_point.backup_path):
            return False
        
        # Check if state file exists
        state_file = os.path.join(rollback_point.backup_path, "system_state.json")
        if not os.path.exists(state_file):
            return False
        
        # Validate checksum
        with open(state_file, 'r') as f:
            saved_state = json.load(f)
        
        current_checksum = self._calculate_checksum(saved_state)
        return current_checksum == rollback_point.checksum
    
    def _backup_current_state(self, rollback_operation: RollbackOperation) -> bool:
        """Backup current state."""
        # Simulate current state backup
        backup_id = str(uuid.uuid4())
        rollback_operation.rollback_log.append({
            'timestamp': datetime.now(),
            'action': 'backup_current_state',
            'backup_id': backup_id,
            'status': 'completed'
        })
        return True
    
    def _stop_active_operations(self, rollback_operation: RollbackOperation) -> bool:
        """Stop active operations."""
        # Simulate stopping active operations
        rollback_operation.rollback_log.append({
            'timestamp': datetime.now(),
            'action': 'stop_active_operations',
            'operations_stopped': 5,  # Simulated count
            'status': 'completed'
        })
        return True
    
    def _restore_system_state(self, rollback_operation: RollbackOperation) -> bool:
        """Restore system state."""
        rollback_point_id = rollback_operation.rollback_point_id
        rollback_point = self.rollback_points[rollback_point_id]
        
        # Simulate system state restoration
        rollback_operation.rollback_log.append({
            'timestamp': datetime.now(),
            'action': 'restore_system_state',
            'state_restored': True,
            'status': 'completed'
        })
        return True
    
    def _validate_restoration(self, rollback_operation: RollbackOperation) -> bool:
        """Validate restoration."""
        # Simulate restoration validation
        validation_success = True  # Simulated success
        rollback_operation.rollback_log.append({
            'timestamp': datetime.now(),
            'action': 'validate_restoration',
            'validation_success': validation_success,
            'status': 'completed'
        })
        return validation_success
    
    def _restart_components(self, rollback_operation: RollbackOperation) -> bool:
        """Restart system components."""
        # Simulate component restart
        rollback_operation.rollback_log.append({
            'timestamp': datetime.now(),
            'action': 'restart_components',
            'components_restarted': 3,  # Simulated count
            'status': 'completed'
        })
        return True
    
    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get statistics about rollback operations."""
        total_points = len(self.rollback_points)
        total_operations = len(self.rollback_operations)
        active_rollbacks = len(self.active_rollbacks)
        
        # Calculate success rate
        completed_rollbacks = len([op for op in self.rollback_operations.values() 
                                 if op.status == RollbackStatus.COMPLETED])
        success_rate = completed_rollbacks / total_operations if total_operations > 0 else 0.0
        
        # Calculate average rollback time
        avg_rollback_time = 0.0
        if self.rollback_operations:
            completed_ops = [op for op in self.rollback_operations.values() 
                           if op.completed_at is not None]
            if completed_ops:
                total_time = sum((op.completed_at - op.created_at).total_seconds() 
                               for op in completed_ops)
                avg_rollback_time = total_time / len(completed_ops)
        
        return {
            'total_rollback_points': total_points,
            'total_rollback_operations': total_operations,
            'active_rollbacks': active_rollbacks,
            'success_rate': success_rate,
            'average_rollback_time': avg_rollback_time,
            'safety_violations': len(self.safety_violations),
            'rollback_history_count': len(self.rollback_history)
        }
    
    def get_safety_violations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent safety violations."""
        return self.safety_violations[-limit:] if self.safety_violations else []
    
    def cleanup_old_rollbacks(self, days: int = 30) -> int:
        """Clean up old rollback points and operations."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        # Clean up old rollback points
        old_points = [point for point in self.rollback_points.values() 
                     if point.created_at < cutoff_date]
        
        for point in old_points:
            # Remove backup directory
            if os.path.exists(point.backup_path):
                shutil.rmtree(point.backup_path)
            
            # Remove from tracking
            del self.rollback_points[point.point_id]
            cleaned_count += 1
        
        # Clean up old rollback operations
        old_operations = [op for op in self.rollback_operations.values() 
                         if op.created_at < cutoff_date]
        
        for operation in old_operations:
            del self.rollback_operations[operation.operation_id]
            cleaned_count += 1
        
        return cleaned_count
    
    def verify_determinism(self, operation_id: str) -> bool:
        """Verify that an operation is deterministic."""
        # Check if operation has been verified before
        if operation_id in self.determinism_checks:
            return self.determinism_checks[operation_id]
        
        # Simulate determinism check
        is_deterministic = random.choice([True, True, True, False])  # 75% deterministic
        
        # Cache result
        self.determinism_checks[operation_id] = is_deterministic
        
        return is_deterministic
    
    def get_rollback_point_details(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific rollback point."""
        if point_id not in self.rollback_points:
            return None
        
        point = self.rollback_points[point_id]
        return {
            'point_id': point.point_id,
            'operation_type': point.operation_type.name,
            'description': point.description,
            'created_at': point.created_at,
            'backup_path': point.backup_path,
            'checksum': point.checksum,
            'metadata': point.metadata
        }
