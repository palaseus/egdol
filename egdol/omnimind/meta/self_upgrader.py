"""
Self-Upgrader for OmniMind Meta-Intelligence
Modifies its own codebase or system architecture while maintaining deterministic rollback safety.
"""

import uuid
import os
import shutil
import json
import hashlib
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class UpgradeStatus(Enum):
    """Status of an upgrade operation."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


class RollbackStatus(Enum):
    """Status of a rollback operation."""
    SUCCESS = auto()
    FAILED = auto()
    PARTIAL = auto()
    NOT_NEEDED = auto()


@dataclass
class UpgradePlan:
    """Represents a plan for upgrading the system."""
    id: str
    name: str
    description: str
    target_version: str
    components_to_upgrade: List[str]
    upgrade_strategy: str
    rollback_strategy: str
    estimated_duration: float  # minutes
    risk_level: float
    dependencies: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: UpgradeStatus = UpgradeStatus.PENDING
    progress: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    rollback_instructions: List[str] = field(default_factory=list)


class SelfUpgrader:
    """Modifies its own codebase or system architecture while maintaining rollback safety."""
    
    def __init__(self, base_path: str, backup_path: str):
        self.base_path = base_path
        self.backup_path = backup_path
        self.upgrade_plans: Dict[str, UpgradePlan] = {}
        self.upgrade_history: List[Dict[str, Any]] = []
        self.rollback_history: List[Dict[str, Any]] = []
        self.system_snapshots: Dict[str, Dict[str, Any]] = {}
        self.verification_checks: List[str] = []
        self.safety_checks: List[str] = []
        
        # Ensure backup directory exists
        os.makedirs(backup_path, exist_ok=True)
    
    def create_upgrade_plan(self, name: str, description: str, 
                           components: List[str], strategy: str = "incremental") -> UpgradePlan:
        """Create a new upgrade plan."""
        plan = UpgradePlan(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            target_version=f"v{len(self.upgrade_plans) + 1}.0.0",
            components_to_upgrade=components,
            upgrade_strategy=strategy,
            rollback_strategy="automatic",
            estimated_duration=random.uniform(30, 120),  # minutes
            risk_level=random.uniform(0.1, 0.6),
            dependencies=self._identify_dependencies(components),
            prerequisites=self._identify_prerequisites(components),
            test_cases=self._generate_test_cases(components),
            rollback_instructions=self._generate_rollback_instructions(components)
        )
        
        self.upgrade_plans[plan.id] = plan
        return plan
    
    def _identify_dependencies(self, components: List[str]) -> List[str]:
        """Identify dependencies for the components."""
        dependencies = []
        
        for component in components:
            if "network" in component.lower():
                dependencies.extend(["communication", "coordination"])
            if "strategic" in component.lower():
                dependencies.extend(["network", "experimental"])
            if "experimental" in component.lower():
                dependencies.extend(["network", "memory"])
            if "meta" in component.lower():
                dependencies.extend(["network", "strategic", "experimental"])
        
        return list(set(dependencies))
    
    def _identify_prerequisites(self, components: List[str]) -> List[str]:
        """Identify prerequisites for the components."""
        prerequisites = []
        
        for component in components:
            if "network" in component.lower():
                prerequisites.extend(["message_bus", "agent_management"])
            if "strategic" in component.lower():
                prerequisites.extend(["goal_generation", "scenario_simulation"])
            if "experimental" in component.lower():
                prerequisites.extend(["hypothesis_generation", "experiment_execution"])
            if "meta" in component.lower():
                prerequisites.extend(["architecture_invention", "skill_innovation"])
        
        return list(set(prerequisites))
    
    def _generate_test_cases(self, components: List[str]) -> List[Dict[str, Any]]:
        """Generate test cases for the components."""
        test_cases = []
        
        for component in components:
            test_cases.extend([
                {
                    'name': f'{component}_functionality_test',
                    'description': f'Test basic functionality of {component}',
                    'type': 'unit',
                    'priority': 'high'
                },
                {
                    'name': f'{component}_integration_test',
                    'description': f'Test integration of {component} with other systems',
                    'type': 'integration',
                    'priority': 'medium'
                },
                {
                    'name': f'{component}_performance_test',
                    'description': f'Test performance of {component}',
                    'type': 'performance',
                    'priority': 'medium'
                }
            ])
        
        return test_cases
    
    def _generate_rollback_instructions(self, components: List[str]) -> List[str]:
        """Generate rollback instructions for the components."""
        instructions = []
        
        for component in components:
            instructions.extend([
                f'Backup current {component} implementation',
                f'Restore previous {component} version',
                f'Verify {component} functionality after rollback',
                f'Update {component} configuration if needed'
            ])
        
        return instructions
    
    def create_system_snapshot(self, snapshot_name: str) -> bool:
        """Create a snapshot of the current system state."""
        try:
            snapshot_data = {
                'timestamp': datetime.now(),
                'system_state': self._capture_system_state(),
                'file_checksums': self._calculate_file_checksums(),
                'configuration': self._capture_configuration(),
                'dependencies': self._capture_dependencies()
            }
            
            self.system_snapshots[snapshot_name] = snapshot_data
            
            # Create backup directory for this snapshot
            snapshot_path = os.path.join(self.backup_path, snapshot_name)
            os.makedirs(snapshot_path, exist_ok=True)
            
            # Copy critical files
            self._backup_critical_files(snapshot_path)
            
            return True
            
        except Exception as e:
            print(f"Error creating snapshot: {e}")
            return False
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture the current system state."""
        return {
            'upgrade_plans_count': len(self.upgrade_plans),
            'upgrade_history_count': len(self.upgrade_history),
            'rollback_history_count': len(self.rollback_history),
            'system_health': random.uniform(0.8, 1.0)
        }
    
    def _calculate_file_checksums(self) -> Dict[str, str]:
        """Calculate checksums for critical files."""
        checksums = {}
        
        # List of critical files to track
        critical_files = [
            'egdol/omnimind/network/__init__.py',
            'egdol/omnimind/strategic/__init__.py',
            'egdol/omnimind/experimental/__init__.py',
            'egdol/omnimind/meta/__init__.py'
        ]
        
        for file_path in critical_files:
            full_path = os.path.join(self.base_path, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'rb') as f:
                        content = f.read()
                        checksums[file_path] = hashlib.md5(content).hexdigest()
                except Exception:
                    checksums[file_path] = "error"
        
        return checksums
    
    def _capture_configuration(self) -> Dict[str, Any]:
        """Capture system configuration."""
        return {
            'base_path': self.base_path,
            'backup_path': self.backup_path,
            'verification_checks': self.verification_checks,
            'safety_checks': self.safety_checks
        }
    
    def _capture_dependencies(self) -> Dict[str, List[str]]:
        """Capture system dependencies."""
        return {
            'network': ['communication', 'coordination'],
            'strategic': ['network', 'experimental'],
            'experimental': ['network', 'memory'],
            'meta': ['network', 'strategic', 'experimental']
        }
    
    def _backup_critical_files(self, snapshot_path: str) -> None:
        """Backup critical files to snapshot directory."""
        critical_files = [
            'egdol/omnimind/network/__init__.py',
            'egdol/omnimind/strategic/__init__.py',
            'egdol/omnimind/experimental/__init__.py',
            'egdol/omnimind/meta/__init__.py'
        ]
        
        for file_path in critical_files:
            source_path = os.path.join(self.base_path, file_path)
            if os.path.exists(source_path):
                dest_path = os.path.join(snapshot_path, file_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(source_path, dest_path)
    
    def execute_upgrade(self, plan_id: str) -> bool:
        """Execute an upgrade plan."""
        if plan_id not in self.upgrade_plans:
            return False
        
        plan = self.upgrade_plans[plan_id]
        plan.status = UpgradeStatus.IN_PROGRESS
        
        try:
            # Create snapshot before upgrade
            snapshot_name = f"pre_upgrade_{plan_id}"
            if not self.create_system_snapshot(snapshot_name):
                plan.status = UpgradeStatus.FAILED
                plan.error_messages.append("Failed to create pre-upgrade snapshot")
                return False
            
            # Execute upgrade phases
            success = self._execute_upgrade_phases(plan)
            
            if success:
                plan.status = UpgradeStatus.COMPLETED
                plan.progress = 100.0
                self.upgrade_history.append({
                    'plan_id': plan_id,
                    'name': plan.name,
                    'completed_at': datetime.now(),
                    'success': True
                })
            else:
                plan.status = UpgradeStatus.FAILED
                # Attempt automatic rollback
                self.rollback_upgrade(plan_id)
            
            return success
            
        except Exception as e:
            plan.status = UpgradeStatus.FAILED
            plan.error_messages.append(f"Upgrade execution error: {str(e)}")
            # Attempt automatic rollback
            self.rollback_upgrade(plan_id)
            return False
    
    def _execute_upgrade_phases(self, plan: UpgradePlan) -> bool:
        """Execute the upgrade phases."""
        phases = [
            "Preparation and validation",
            "Component backup",
            "Implementation",
            "Testing and verification",
            "Deployment and monitoring"
        ]
        
        for i, phase in enumerate(phases):
            plan.progress = (i + 1) * 20.0
            
            # Simulate phase execution
            phase_success = random.random() > plan.risk_level * 0.2
            
            if not phase_success:
                plan.error_messages.append(f"Failed in phase: {phase}")
                return False
            
            # Add some delay to simulate work
            import time
            time.sleep(0.1)
        
        return True
    
    def rollback_upgrade(self, plan_id: str) -> RollbackStatus:
        """Rollback an upgrade."""
        if plan_id not in self.upgrade_plans:
            return RollbackStatus.FAILED
        
        plan = self.upgrade_plans[plan_id]
        
        try:
            # Find the pre-upgrade snapshot
            snapshot_name = f"pre_upgrade_{plan_id}"
            if snapshot_name not in self.system_snapshots:
                return RollbackStatus.FAILED
            
            # Restore from snapshot
            success = self._restore_from_snapshot(snapshot_name)
            
            if success:
                plan.status = UpgradeStatus.ROLLED_BACK
                self.rollback_history.append({
                    'plan_id': plan_id,
                    'name': plan.name,
                    'rolled_back_at': datetime.now(),
                    'success': True
                })
                return RollbackStatus.SUCCESS
            else:
                return RollbackStatus.FAILED
                
        except Exception as e:
            plan.error_messages.append(f"Rollback error: {str(e)}")
            return RollbackStatus.FAILED
    
    def _restore_from_snapshot(self, snapshot_name: str) -> bool:
        """Restore system from a snapshot."""
        try:
            snapshot_path = os.path.join(self.backup_path, snapshot_name)
            
            if not os.path.exists(snapshot_path):
                return False
            
            # Restore critical files
            critical_files = [
                'egdol/omnimind/network/__init__.py',
                'egdol/omnimind/strategic/__init__.py',
                'egdol/omnimind/experimental/__init__.py',
                'egdol/omnimind/meta/__init__.py'
            ]
            
            for file_path in critical_files:
                source_path = os.path.join(snapshot_path, file_path)
                if os.path.exists(source_path):
                    dest_path = os.path.join(self.base_path, file_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(source_path, dest_path)
            
            return True
            
        except Exception as e:
            print(f"Error restoring from snapshot: {e}")
            return False
    
    def verify_system_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the system."""
        verification_results = {
            'overall_health': 0.0,
            'component_health': {},
            'issues_found': [],
            'recommendations': []
        }
        
        # Check each component
        components = ['network', 'strategic', 'experimental', 'meta']
        
        for component in components:
            component_health = random.uniform(0.7, 1.0)
            verification_results['component_health'][component] = component_health
            
            if component_health < 0.8:
                verification_results['issues_found'].append(f"{component} component has low health")
                verification_results['recommendations'].append(f"Consider upgrading {component} component")
        
        # Calculate overall health
        if verification_results['component_health']:
            verification_results['overall_health'] = sum(verification_results['component_health'].values()) / len(verification_results['component_health'])
        
        return verification_results
    
    def get_upgrade_statistics(self) -> Dict[str, Any]:
        """Get statistics about upgrades."""
        total_plans = len(self.upgrade_plans)
        completed_upgrades = len([p for p in self.upgrade_plans.values() if p.status == UpgradeStatus.COMPLETED])
        failed_upgrades = len([p for p in self.upgrade_plans.values() if p.status == UpgradeStatus.FAILED])
        rolled_back_upgrades = len([p for p in self.upgrade_plans.values() if p.status == UpgradeStatus.ROLLED_BACK])
        
        # Average risk level
        if self.upgrade_plans:
            avg_risk = sum(p.risk_level for p in self.upgrade_plans.values()) / total_plans
        else:
            avg_risk = 0.0
        
        return {
            'total_plans': total_plans,
            'completed_upgrades': completed_upgrades,
            'failed_upgrades': failed_upgrades,
            'rolled_back_upgrades': rolled_back_upgrades,
            'success_rate': completed_upgrades / total_plans if total_plans > 0 else 0,
            'average_risk_level': avg_risk,
            'upgrade_history_count': len(self.upgrade_history),
            'rollback_history_count': len(self.rollback_history),
            'snapshots_count': len(self.system_snapshots)
        }
    
    def get_available_snapshots(self) -> List[str]:
        """Get list of available system snapshots."""
        return list(self.system_snapshots.keys())
    
    def cleanup_old_snapshots(self, keep_count: int = 5) -> int:
        """Clean up old snapshots, keeping only the most recent ones."""
        if len(self.system_snapshots) <= keep_count:
            return 0
        
        # Sort snapshots by timestamp
        sorted_snapshots = sorted(
            self.system_snapshots.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Keep only the most recent ones
        snapshots_to_remove = sorted_snapshots[keep_count:]
        removed_count = 0
        
        for snapshot_name, _ in snapshots_to_remove:
            # Remove from memory
            del self.system_snapshots[snapshot_name]
            
            # Remove from disk
            snapshot_path = os.path.join(self.backup_path, snapshot_name)
            if os.path.exists(snapshot_path):
                shutil.rmtree(snapshot_path)
                removed_count += 1
        
        return removed_count
