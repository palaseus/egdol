"""
System Evolution Orchestrator
Gives OmniMind the ability to refactor its own architecture intelligently.
Analyzes performance, complexity, and redundancy to propose architectural reorganizations.
"""

import uuid
import time
import ast
import importlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import sqlite3
from collections import defaultdict, deque
import statistics
import numpy as np
import os
import shutil
from pathlib import Path

from ..conversational.personality_framework import Personality, PersonalityType
from ..civilization.multi_agent_system import CivilizationAgent


class RefactorType(Enum):
    """Types of architectural refactoring."""
    MODULE_MERGE = "module_merge"
    MODULE_SPLIT = "module_split"
    INTERFACE_CONSOLIDATION = "interface_consolidation"
    DEPENDENCY_OPTIMIZATION = "dependency_optimization"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CODE_DEDUPLICATION = "code_deduplication"
    ARCHITECTURE_RESTRUCTURE = "architecture_restructure"


class RefactorStatus(Enum):
    """Status of refactoring operations."""
    PROPOSED = "proposed"
    ANALYZING = "analyzing"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ComplexityLevel(Enum):
    """Complexity levels for refactoring."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RefactorProposal:
    """A proposal for system refactoring."""
    proposal_id: str
    refactor_type: RefactorType
    target_modules: List[str]
    description: str
    rationale: str
    expected_benefits: List[str]
    potential_risks: List[str]
    complexity: ComplexityLevel
    estimated_effort: int  # Hours
    dependencies: List[str] = field(default_factory=list)
    status: RefactorStatus = RefactorStatus.PROPOSED
    performance_impact: Dict[str, float] = field(default_factory=dict)
    migration_plan: List[Dict[str, Any]] = field(default_factory=list)
    rollback_plan: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "refactor_type": self.refactor_type.value,
            "target_modules": self.target_modules,
            "description": self.description,
            "rationale": self.rationale,
            "expected_benefits": self.expected_benefits,
            "potential_risks": self.potential_risks,
            "complexity": self.complexity.value,
            "estimated_effort": self.estimated_effort,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "performance_impact": self.performance_impact,
            "migration_plan": self.migration_plan,
            "rollback_plan": self.rollback_plan,
            "created_at": self.created_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class SystemMetrics:
    """System performance and complexity metrics."""
    timestamp: datetime
    module_count: int
    total_lines_of_code: int
    cyclomatic_complexity: float
    coupling_score: float
    cohesion_score: float
    test_coverage: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    code_duplication_rate: float = 0.0
    interface_complexity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "module_count": self.module_count,
            "total_lines_of_code": self.total_lines_of_code,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "coupling_score": self.coupling_score,
            "cohesion_score": self.cohesion_score,
            "test_coverage": self.test_coverage,
            "performance_metrics": self.performance_metrics,
            "dependency_graph": self.dependency_graph,
            "code_duplication_rate": self.code_duplication_rate,
            "interface_complexity": self.interface_complexity
        }


@dataclass
class RefactorResult:
    """Result of a refactoring operation."""
    refactor_id: str
    proposal_id: str
    success: bool
    performance_improvement: Dict[str, float] = field(default_factory=dict)
    complexity_reduction: float = 0.0
    issues_introduced: List[str] = field(default_factory=list)
    rollback_required: bool = False
    completion_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "refactor_id": self.refactor_id,
            "proposal_id": self.proposal_id,
            "success": self.success,
            "performance_improvement": self.performance_improvement,
            "complexity_reduction": self.complexity_reduction,
            "issues_introduced": self.issues_introduced,
            "rollback_required": self.rollback_required,
            "completion_time": self.completion_time.isoformat() if self.completion_time else None
        }


class SystemEvolutionOrchestrator:
    """Orchestrator for intelligent system refactoring."""
    
    def __init__(self, base_path: str = "/home/dubius/Documents/egdol", db_path: str = "system_evolution.db"):
        self.base_path = Path(base_path)
        self.db_path = db_path
        self.refactor_proposals: Dict[str, RefactorProposal] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.refactor_results: Dict[str, RefactorResult] = {}
        self.module_registry: Dict[str, Dict[str, Any]] = {}
        self.performance_baseline: Dict[str, float] = {}
        self._init_database()
        self._scan_system_modules()
    
    def _init_database(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create refactor proposals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS refactor_proposals (
                proposal_id TEXT PRIMARY KEY,
                refactor_type TEXT NOT NULL,
                target_modules TEXT NOT NULL,
                description TEXT NOT NULL,
                rationale TEXT NOT NULL,
                expected_benefits TEXT,
                potential_risks TEXT,
                complexity TEXT NOT NULL,
                estimated_effort INTEGER,
                dependencies TEXT,
                status TEXT NOT NULL,
                performance_impact TEXT,
                migration_plan TEXT,
                rollback_plan TEXT,
                created_at TEXT NOT NULL,
                approved_at TEXT,
                completed_at TEXT
            )
        ''')
        
        # Create system metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp TEXT PRIMARY KEY,
                module_count INTEGER,
                total_lines_of_code INTEGER,
                cyclomatic_complexity REAL,
                coupling_score REAL,
                cohesion_score REAL,
                test_coverage REAL,
                performance_metrics TEXT,
                dependency_graph TEXT,
                code_duplication_rate REAL,
                interface_complexity REAL
            )
        ''')
        
        # Create refactor results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS refactor_results (
                refactor_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL,
                success BOOLEAN,
                performance_improvement TEXT,
                complexity_reduction REAL,
                issues_introduced TEXT,
                rollback_required BOOLEAN,
                completion_time TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _scan_system_modules(self):
        """Scan and analyze system modules."""
        for module_path in self.base_path.rglob("*.py"):
            if "__pycache__" in str(module_path):
                continue
            
            relative_path = module_path.relative_to(self.base_path)
            module_name = str(relative_path).replace("/", ".").replace(".py", "")
            
            try:
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST for analysis
                tree = ast.parse(content)
                
                # Extract module information
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
                
                self.module_registry[module_name] = {
                    "path": str(module_path),
                    "lines_of_code": len(content.splitlines()),
                    "classes": classes,
                    "functions": functions,
                    "imports": imports,
                    "complexity": self._calculate_module_complexity(tree),
                    "last_modified": datetime.fromtimestamp(module_path.stat().st_mtime)
                }
            except Exception as e:
                print(f"Warning: Could not analyze module {module_path}: {e}")
    
    def _calculate_module_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity of a module."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def analyze_system_performance(self) -> SystemMetrics:
        """Analyze current system performance and complexity."""
        total_modules = len(self.module_registry)
        total_loc = sum(module["lines_of_code"] for module in self.module_registry.values())
        avg_complexity = statistics.mean([module["complexity"] for module in self.module_registry.values()])
        
        # Calculate coupling and cohesion
        coupling_score = self._calculate_coupling_score()
        cohesion_score = self._calculate_cohesion_score()
        
        # Calculate test coverage (simplified)
        test_coverage = self._calculate_test_coverage()
        
        # Calculate code duplication
        duplication_rate = self._calculate_code_duplication()
        
        # Calculate interface complexity
        interface_complexity = self._calculate_interface_complexity()
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph()
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            module_count=total_modules,
            total_lines_of_code=total_loc,
            cyclomatic_complexity=avg_complexity,
            coupling_score=coupling_score,
            cohesion_score=cohesion_score,
            test_coverage=test_coverage,
            dependency_graph=dependency_graph,
            code_duplication_rate=duplication_rate,
            interface_complexity=interface_complexity
        )
        
        self.system_metrics.append(metrics)
        self._save_system_metrics(metrics)
        
        return metrics
    
    def _calculate_coupling_score(self) -> float:
        """Calculate system coupling score."""
        total_couplings = 0
        total_modules = len(self.module_registry)
        
        for module_name, module_info in self.module_registry.items():
            # Count imports as coupling indicators
            total_couplings += len(module_info["imports"])
        
        # Normalize by module count
        return total_couplings / max(1, total_modules * (total_modules - 1))
    
    def _calculate_cohesion_score(self) -> float:
        """Calculate system cohesion score."""
        # Simplified cohesion calculation based on related functionality
        cohesion_scores = []
        
        for module_name, module_info in self.module_registry.items():
            # Cohesion based on related classes and functions
            related_items = len(module_info["classes"]) + len(module_info["functions"])
            if related_items > 0:
                # Higher cohesion if related items are grouped together
                cohesion = min(1.0, related_items / 10.0)  # Normalize
                cohesion_scores.append(cohesion)
        
        return statistics.mean(cohesion_scores) if cohesion_scores else 0.0
    
    def _calculate_test_coverage(self) -> float:
        """Calculate test coverage (simplified)."""
        test_files = list(self.base_path.rglob("test_*.py"))
        total_files = len(self.module_registry)
        test_file_count = len(test_files)
        
        return min(1.0, test_file_count / max(1, total_files))
    
    def _calculate_code_duplication(self) -> float:
        """Calculate code duplication rate."""
        # Simplified duplication detection
        function_signatures = defaultdict(int)
        
        for module_info in self.module_registry.values():
            for func in module_info["functions"]:
                function_signatures[func] += 1
        
        duplicated_functions = sum(1 for count in function_signatures.values() if count > 1)
        total_functions = len(function_signatures)
        
        return duplicated_functions / max(1, total_functions)
    
    def _calculate_interface_complexity(self) -> float:
        """Calculate interface complexity."""
        total_interfaces = 0
        total_complexity = 0
        
        for module_info in self.module_registry.values():
            # Count classes as interfaces
            class_count = len(module_info["classes"])
            total_interfaces += class_count
            total_complexity += class_count * module_info["complexity"]
        
        return total_complexity / max(1, total_interfaces)
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph of modules."""
        dependency_graph = defaultdict(list)
        
        for module_name, module_info in self.module_registry.items():
            for import_module in module_info["imports"]:
                if import_module and import_module in self.module_registry:
                    dependency_graph[module_name].append(import_module)
        
        return dict(dependency_graph)
    
    def propose_refactor(self,
                        refactor_type: RefactorType,
                        target_modules: List[str],
                        description: str,
                        rationale: str) -> str:
        """Propose a system refactor."""
        proposal_id = str(uuid.uuid4())
        
        # Analyze refactor complexity
        complexity = self._analyze_refactor_complexity(refactor_type, target_modules)
        
        # Generate benefits and risks
        benefits = self._generate_refactor_benefits(refactor_type, target_modules)
        risks = self._generate_refactor_risks(refactor_type, target_modules)
        
        # Estimate effort
        effort = self._estimate_refactor_effort(refactor_type, target_modules)
        
        # Generate migration and rollback plans
        migration_plan = self._generate_migration_plan(refactor_type, target_modules)
        rollback_plan = self._generate_rollback_plan(refactor_type, target_modules)
        
        proposal = RefactorProposal(
            proposal_id=proposal_id,
            refactor_type=refactor_type,
            target_modules=target_modules,
            description=description,
            rationale=rationale,
            expected_benefits=benefits,
            potential_risks=risks,
            complexity=complexity,
            estimated_effort=effort,
            migration_plan=migration_plan,
            rollback_plan=rollback_plan
        )
        
        self.refactor_proposals[proposal_id] = proposal
        self._save_refactor_proposal(proposal)
        
        return proposal_id
    
    def _analyze_refactor_complexity(self, refactor_type: RefactorType, target_modules: List[str]) -> ComplexityLevel:
        """Analyze complexity of refactor operation."""
        if not target_modules:
            return ComplexityLevel.LOW
        
        # Calculate complexity based on module characteristics
        total_complexity = 0
        for module in target_modules:
            if module in self.module_registry:
                total_complexity += self.module_registry[module]["complexity"]
        
        avg_complexity = total_complexity / len(target_modules)
        
        if avg_complexity > 20:
            return ComplexityLevel.CRITICAL
        elif avg_complexity > 15:
            return ComplexityLevel.HIGH
        elif avg_complexity > 10:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
    
    def _generate_refactor_benefits(self, refactor_type: RefactorType, target_modules: List[str]) -> List[str]:
        """Generate expected benefits for refactor."""
        benefits_map = {
            RefactorType.MODULE_MERGE: [
                "Reduced coupling",
                "Simplified interfaces",
                "Better code organization"
            ],
            RefactorType.MODULE_SPLIT: [
                "Improved modularity",
                "Reduced complexity",
                "Better testability"
            ],
            RefactorType.INTERFACE_CONSOLIDATION: [
                "Unified interfaces",
                "Reduced duplication",
                "Better maintainability"
            ],
            RefactorType.DEPENDENCY_OPTIMIZATION: [
                "Reduced coupling",
                "Faster imports",
                "Better modularity"
            ],
            RefactorType.PERFORMANCE_OPTIMIZATION: [
                "Faster execution",
                "Reduced memory usage",
                "Better scalability"
            ],
            RefactorType.CODE_DEDUPLICATION: [
                "Reduced duplication",
                "Better maintainability",
                "Consistent behavior"
            ],
            RefactorType.ARCHITECTURE_RESTRUCTURE: [
                "Improved architecture",
                "Better separation of concerns",
                "Enhanced scalability"
            ]
        }
        
        return benefits_map.get(refactor_type, ["General improvements"])
    
    def _generate_refactor_risks(self, refactor_type: RefactorType, target_modules: List[str]) -> List[str]:
        """Generate potential risks for refactor."""
        risks_map = {
            RefactorType.MODULE_MERGE: [
                "Increased coupling",
                "Interface conflicts",
                "Breaking changes"
            ],
            RefactorType.MODULE_SPLIT: [
                "Interface fragmentation",
                "Increased complexity",
                "Import issues"
            ],
            RefactorType.INTERFACE_CONSOLIDATION: [
                "Interface conflicts",
                "Breaking changes",
                "Migration complexity"
            ],
            RefactorType.DEPENDENCY_OPTIMIZATION: [
                "Circular dependencies",
                "Import errors",
                "Module loading issues"
            ],
            RefactorType.PERFORMANCE_OPTIMIZATION: [
                "Behavioral changes",
                "Compatibility issues",
                "Testing requirements"
            ],
            RefactorType.CODE_DEDUPLICATION: [
                "Behavioral changes",
                "Interface conflicts",
                "Testing complexity"
            ],
            RefactorType.ARCHITECTURE_RESTRUCTURE: [
                "Major breaking changes",
                "Migration complexity",
                "System instability"
            ]
        }
        
        return risks_map.get(refactor_type, ["General risks"])
    
    def _estimate_refactor_effort(self, refactor_type: RefactorType, target_modules: List[str]) -> int:
        """Estimate effort required for refactor."""
        base_effort = {
            RefactorType.MODULE_MERGE: 4,
            RefactorType.MODULE_SPLIT: 6,
            RefactorType.INTERFACE_CONSOLIDATION: 8,
            RefactorType.DEPENDENCY_OPTIMIZATION: 6,
            RefactorType.PERFORMANCE_OPTIMIZATION: 10,
            RefactorType.CODE_DEDUPLICATION: 8,
            RefactorType.ARCHITECTURE_RESTRUCTURE: 20
        }
        
        base = base_effort.get(refactor_type, 5)
        module_multiplier = len(target_modules) * 0.5
        
        return int(base + module_multiplier)
    
    def _generate_migration_plan(self, refactor_type: RefactorType, target_modules: List[str]) -> List[Dict[str, Any]]:
        """Generate migration plan for refactor."""
        plan = []
        
        if refactor_type == RefactorType.MODULE_MERGE:
            plan.extend([
                {"step": 1, "action": "Backup existing modules", "duration": "5 minutes"},
                {"step": 2, "action": "Create merged module structure", "duration": "30 minutes"},
                {"step": 3, "action": "Migrate code and resolve conflicts", "duration": "2 hours"},
                {"step": 4, "action": "Update imports and dependencies", "duration": "1 hour"},
                {"step": 5, "action": "Run tests and fix issues", "duration": "1 hour"}
            ])
        elif refactor_type == RefactorType.MODULE_SPLIT:
            plan.extend([
                {"step": 1, "action": "Backup existing module", "duration": "5 minutes"},
                {"step": 2, "action": "Create new module structure", "duration": "30 minutes"},
                {"step": 3, "action": "Split code into new modules", "duration": "2 hours"},
                {"step": 4, "action": "Update imports and dependencies", "duration": "1 hour"},
                {"step": 5, "action": "Run tests and fix issues", "duration": "1 hour"}
            ])
        else:
            plan.extend([
                {"step": 1, "action": "Backup system", "duration": "10 minutes"},
                {"step": 2, "action": "Implement refactor", "duration": "2 hours"},
                {"step": 3, "action": "Update dependencies", "duration": "30 minutes"},
                {"step": 4, "action": "Run tests", "duration": "30 minutes"},
                {"step": 5, "action": "Validate changes", "duration": "30 minutes"}
            ])
        
        return plan
    
    def _generate_rollback_plan(self, refactor_type: RefactorType, target_modules: List[str]) -> List[Dict[str, Any]]:
        """Generate rollback plan for refactor."""
        return [
            {"step": 1, "action": "Stop all system processes", "duration": "1 minute"},
            {"step": 2, "action": "Restore from backup", "duration": "5 minutes"},
            {"step": 3, "action": "Verify system integrity", "duration": "5 minutes"},
            {"step": 4, "action": "Restart system processes", "duration": "2 minutes"},
            {"step": 5, "action": "Run validation tests", "duration": "10 minutes"}
        ]
    
    def approve_refactor(self, proposal_id: str) -> bool:
        """Approve a refactor proposal."""
        if proposal_id not in self.refactor_proposals:
            return False
        
        proposal = self.refactor_proposals[proposal_id]
        proposal.status = RefactorStatus.APPROVED
        proposal.approved_at = datetime.now()
        
        self._save_refactor_proposal(proposal)
        return True
    
    def execute_refactor(self, proposal_id: str) -> str:
        """Execute an approved refactor."""
        if proposal_id not in self.refactor_proposals:
            return None
        
        proposal = self.refactor_proposals[proposal_id]
        if proposal.status != RefactorStatus.APPROVED:
            return None
        
        proposal.status = RefactorStatus.IN_PROGRESS
        refactor_id = str(uuid.uuid4())
        
        try:
            # Execute the refactor based on type
            success = self._execute_refactor_operation(proposal)
            
            if success:
                proposal.status = RefactorStatus.COMPLETED
                proposal.completed_at = datetime.now()
            else:
                proposal.status = RefactorStatus.FAILED
            
            # Create result
            result = RefactorResult(
                refactor_id=refactor_id,
                proposal_id=proposal_id,
                success=success,
                completion_time=datetime.now()
            )
            
            self.refactor_results[refactor_id] = result
            self._save_refactor_result(result)
            
        except Exception as e:
            proposal.status = RefactorStatus.FAILED
            print(f"Refactor execution failed: {e}")
        
        self._save_refactor_proposal(proposal)
        return refactor_id
    
    def _execute_refactor_operation(self, proposal: RefactorProposal) -> bool:
        """Execute the actual refactor operation."""
        # This is a simplified implementation
        # In a real system, this would perform actual code transformations
        
        print(f"🔧 Executing refactor: {proposal.description}")
        print(f"   Type: {proposal.refactor_type.value}")
        print(f"   Target modules: {proposal.target_modules}")
        
        # Simulate refactor execution
        time.sleep(0.1)  # Simulate processing time
        
        # In a real implementation, this would:
        # 1. Create backup of target modules
        # 2. Perform the actual refactoring
        # 3. Update imports and dependencies
        # 4. Run tests to validate changes
        # 5. Rollback if tests fail
        
        return True  # Simplified: always succeed
    
    def rollback_refactor(self, refactor_id: str) -> bool:
        """Rollback a refactor operation."""
        if refactor_id not in self.refactor_results:
            return False
        
        result = self.refactor_results[refactor_id]
        proposal = self.refactor_proposals[result.proposal_id]
        
        print(f"🔄 Rolling back refactor: {proposal.description}")
        
        # In a real implementation, this would:
        # 1. Stop system processes
        # 2. Restore from backup
        # 3. Verify system integrity
        # 4. Restart processes
        
        result.rollback_required = True
        proposal.status = RefactorStatus.ROLLED_BACK
        
        self._save_refactor_result(result)
        self._save_refactor_proposal(proposal)
        
        return True
    
    def get_refactor_proposals(self, status: Optional[RefactorStatus] = None) -> List[RefactorProposal]:
        """Get refactor proposals, optionally filtered by status."""
        if status is None:
            return list(self.refactor_proposals.values())
        return [p for p in self.refactor_proposals.values() if p.status == status]
    
    def get_system_health_score(self) -> float:
        """Get overall system health score."""
        if not self.system_metrics:
            return 0.0
        
        latest_metrics = self.system_metrics[-1]
        
        # Calculate health score based on various factors
        complexity_score = max(0, 1.0 - latest_metrics.cyclomatic_complexity / 20.0)
        coupling_score = max(0, 1.0 - latest_metrics.coupling_score)
        cohesion_score = latest_metrics.cohesion_score
        test_score = latest_metrics.test_coverage
        duplication_score = max(0, 1.0 - latest_metrics.code_duplication_rate)
        
        # Weighted average
        health_score = (
            complexity_score * 0.25 +
            coupling_score * 0.25 +
            cohesion_score * 0.20 +
            test_score * 0.20 +
            duplication_score * 0.10
        )
        
        return min(1.0, max(0.0, health_score))
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for system optimization."""
        recommendations = []
        
        if not self.system_metrics:
            return recommendations
        
        latest_metrics = self.system_metrics[-1]
        
        # High complexity recommendation
        if latest_metrics.cyclomatic_complexity > 15:
            recommendations.append({
                "type": "complexity_reduction",
                "priority": "high",
                "description": "High cyclomatic complexity detected",
                "suggestion": "Consider splitting complex modules",
                "target_modules": self._find_complex_modules()
            })
        
        # High coupling recommendation
        if latest_metrics.coupling_score > 0.7:
            recommendations.append({
                "type": "coupling_reduction",
                "priority": "medium",
                "description": "High coupling detected between modules",
                "suggestion": "Consider interface consolidation",
                "target_modules": self._find_highly_coupled_modules()
            })
        
        # Low cohesion recommendation
        if latest_metrics.cohesion_score < 0.3:
            recommendations.append({
                "type": "cohesion_improvement",
                "priority": "medium",
                "description": "Low cohesion detected",
                "suggestion": "Consider module reorganization",
                "target_modules": self._find_low_cohesion_modules()
            })
        
        # Code duplication recommendation
        if latest_metrics.code_duplication_rate > 0.2:
            recommendations.append({
                "type": "deduplication",
                "priority": "low",
                "description": "High code duplication detected",
                "suggestion": "Consider code deduplication",
                "target_modules": self._find_duplicated_code()
            })
        
        return recommendations
    
    def _find_complex_modules(self) -> List[str]:
        """Find modules with high complexity."""
        complex_modules = []
        for module_name, module_info in self.module_registry.items():
            if module_info["complexity"] > 15:
                complex_modules.append(module_name)
        return complex_modules
    
    def _find_highly_coupled_modules(self) -> List[str]:
        """Find modules with high coupling."""
        # Simplified: return modules with many imports
        coupled_modules = []
        for module_name, module_info in self.module_registry.items():
            if len(module_info["imports"]) > 10:
                coupled_modules.append(module_name)
        return coupled_modules
    
    def _find_low_cohesion_modules(self) -> List[str]:
        """Find modules with low cohesion."""
        # Simplified: return modules with few related items
        low_cohesion_modules = []
        for module_name, module_info in self.module_registry.items():
            related_items = len(module_info["classes"]) + len(module_info["functions"])
            if related_items < 3:
                low_cohesion_modules.append(module_name)
        return low_cohesion_modules
    
    def _find_duplicated_code(self) -> List[str]:
        """Find modules with duplicated code."""
        # This would require more sophisticated analysis
        return []
    
    def _save_refactor_proposal(self, proposal: RefactorProposal):
        """Save refactor proposal to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO refactor_proposals 
            (proposal_id, refactor_type, target_modules, description, rationale,
             expected_benefits, potential_risks, complexity, estimated_effort,
             dependencies, status, performance_impact, migration_plan, rollback_plan,
             created_at, approved_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            proposal.proposal_id, proposal.refactor_type.value, json.dumps(proposal.target_modules),
            proposal.description, proposal.rationale, json.dumps(proposal.expected_benefits),
            json.dumps(proposal.potential_risks), proposal.complexity.value, proposal.estimated_effort,
            json.dumps(proposal.dependencies), proposal.status.value, json.dumps(proposal.performance_impact),
            json.dumps(proposal.migration_plan), json.dumps(proposal.rollback_plan),
            proposal.created_at.isoformat(),
            proposal.approved_at.isoformat() if proposal.approved_at else None,
            proposal.completed_at.isoformat() if proposal.completed_at else None
        ))
        
        conn.commit()
        conn.close()
    
    def _save_system_metrics(self, metrics: SystemMetrics):
        """Save system metrics to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO system_metrics 
            (timestamp, module_count, total_lines_of_code, cyclomatic_complexity,
             coupling_score, cohesion_score, test_coverage, performance_metrics,
             dependency_graph, code_duplication_rate, interface_complexity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(), metrics.module_count, metrics.total_lines_of_code,
            metrics.cyclomatic_complexity, metrics.coupling_score, metrics.cohesion_score,
            metrics.test_coverage, json.dumps(metrics.performance_metrics),
            json.dumps(metrics.dependency_graph), metrics.code_duplication_rate,
            metrics.interface_complexity
        ))
        
        conn.commit()
        conn.close()
    
    def _save_refactor_result(self, result: RefactorResult):
        """Save refactor result to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO refactor_results 
            (refactor_id, proposal_id, success, performance_improvement,
             complexity_reduction, issues_introduced, rollback_required, completion_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.refactor_id, result.proposal_id, result.success,
            json.dumps(result.performance_improvement), result.complexity_reduction,
            json.dumps(result.issues_introduced), result.rollback_required,
            result.completion_time.isoformat() if result.completion_time else None
        ))
        
        conn.commit()
        conn.close()
