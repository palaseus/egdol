"""
Tests for Reflexive Self-Introspection System
Comprehensive testing of self-analysis, optimization, and modification capabilities.
"""

import unittest
import time
from egdol.omnimind.reflexive import (
    SelfIntrospector, IntrospectionType, ConfidenceLevel,
    SelfOptimizer, OptimizationType, OptimizationStatus,
    SelfModifier, ModificationType, ModificationStatus,
    PerformanceMonitor, PerformanceMetrics,
    ReasoningAnalyzer, SkillAnalyzer, MemoryAnalyzer
)


class MockMemoryManager:
    """Mock memory manager for testing."""
    
    def __init__(self):
        self.rules = []
        self.facts = []
        self.memory_stats = {'total_entries': 0}
        
    def get_all_rules(self):
        return self.rules
        
    def get_all_facts(self):
        return self.facts
        
    def get_memory_statistics(self):
        return self.memory_stats
        
    def add_rule(self, rule):
        self.rules.append(rule)
        self.memory_stats['total_entries'] = len(self.rules) + len(self.facts)
        
    def remove_rule(self, rule_id):
        self.rules = [r for r in self.rules if getattr(r, 'id', None) != rule_id]
        self.memory_stats['total_entries'] = len(self.rules) + len(self.facts)
        
    def get_rule(self, rule_id):
        for rule in self.rules:
            if getattr(rule, 'id', None) == rule_id:
                return rule
        return None
        
    def add_fact(self, fact):
        self.facts.append(fact)
        self.memory_stats['total_entries'] = len(self.rules) + len(self.facts)
        
    def remove_fact(self, fact_id):
        self.facts = [f for f in self.facts if getattr(f, 'id', None) != fact_id]
        self.memory_stats['total_entries'] = len(self.rules) + len(self.facts)
        
    def get_fact(self, fact_id):
        for fact in self.facts:
            if getattr(fact, 'id', None) == fact_id:
                return fact
        return None


class MockSkillRouter:
    """Mock skill router for testing."""
    
    def __init__(self):
        self.skills = {}
        self.skill_stats = {}
        
    def get_skill_statistics(self):
        return self.skill_stats
        
    def add_skill(self, name, definition):
        self.skills[name] = definition
        
    def remove_skill(self, name):
        if name in self.skills:
            del self.skills[name]
            
    def get_skill(self, name):
        return self.skills.get(name)
        
    def update_skill(self, name, new_skill):
        self.skills[name] = new_skill
        
    def improve_skill(self, name):
        if name in self.skills:
            return f"Improved {name}"
        return None


class MockPlanner:
    """Mock planner for testing."""
    
    def __init__(self):
        self.goals = {'active_goals': [], 'completed_goals': []}
        self.execution_stats = {'average_execution_time': 0}
        
    def get_all_goals(self):
        return self.goals
        
    def get_execution_statistics(self):
        return self.execution_stats
        
    def get_planner_stats(self):
        return {'active_goals': len(self.goals['active_goals'])}


class SelfIntrospectorTests(unittest.TestCase):
    """Test the self-introspector."""
    
    def setUp(self):
        self.memory_manager = MockMemoryManager()
        self.skill_router = MockSkillRouter()
        self.planner = MockPlanner()
        self.introspector = SelfIntrospector(
            memory_manager=self.memory_manager,
            skill_router=self.skill_router,
            planner=self.planner
        )
        
    def test_introspect_reasoning(self):
        """Test reasoning introspection."""
        result = self.introspector.introspect(IntrospectionType.REASONING)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.introspection_type, IntrospectionType.REASONING)
        self.assertIsInstance(result.findings, list)
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.confidence, ConfidenceLevel)
        self.assertIsInstance(result.reasoning_trace, list)
        
    def test_introspect_skills(self):
        """Test skills introspection."""
        # Add some skill statistics
        self.skill_router.skill_stats = {
            'analysis': {'success_rate': 0.5, 'execution_count': 10},
            'security': {'success_rate': 0.9, 'execution_count': 5}
        }
        
        result = self.introspector.introspect(IntrospectionType.SKILLS)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.introspection_type, IntrospectionType.SKILLS)
        self.assertGreater(len(result.findings), 0)
        
    def test_introspect_memory(self):
        """Test memory introspection."""
        # Add some memory entries
        self.memory_manager.memory_stats = {'total_entries': 1500}
        
        result = self.introspector.introspect(IntrospectionType.MEMORY)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.introspection_type, IntrospectionType.MEMORY)
        
    def test_introspect_performance(self):
        """Test performance introspection."""
        result = self.introspector.introspect(IntrospectionType.PERFORMANCE)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.introspection_type, IntrospectionType.PERFORMANCE)
        
    def test_introspect_rules(self):
        """Test rules introspection."""
        result = self.introspector.introspect(IntrospectionType.RULES)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.introspection_type, IntrospectionType.RULES)
        
    def test_introspect_patterns(self):
        """Test patterns introspection."""
        result = self.introspector.introspect(IntrospectionType.PATTERNS)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.introspection_type, IntrospectionType.PATTERNS)
        
    def test_get_introspection_history(self):
        """Test introspection history."""
        # Perform some introspections
        self.introspector.introspect(IntrospectionType.REASONING)
        self.introspector.introspect(IntrospectionType.SKILLS)
        
        history = self.introspector.get_introspection_history()
        
        self.assertEqual(len(history), 2)
        self.assertIsInstance(history[0], dict)
        self.assertIn('id', history[0])
        self.assertIn('type', history[0])
        
    def test_get_introspection_stats(self):
        """Test introspection statistics."""
        # Perform some introspections
        self.introspector.introspect(IntrospectionType.REASONING)
        self.introspector.introspect(IntrospectionType.SKILLS)
        
        stats = self.introspector.get_introspection_stats()
        
        self.assertIn('total_introspections', stats)
        self.assertIn('average_findings', stats)
        self.assertIn('average_recommendations', stats)
        self.assertIn('confidence_distribution', stats)
        
        self.assertEqual(stats['total_introspections'], 2)


class SelfOptimizerTests(unittest.TestCase):
    """Test the self-optimizer."""
    
    def setUp(self):
        self.memory_manager = MockMemoryManager()
        self.skill_router = MockSkillRouter()
        self.planner = MockPlanner()
        self.optimizer = SelfOptimizer(
            memory_manager=self.memory_manager,
            skill_router=self.skill_router,
            planner=self.planner
        )
        
    def test_optimize_rule_optimization(self):
        """Test rule optimization."""
        # Create introspection result with rule findings
        from egdol.omnimind.reflexive.introspector import IntrospectionResult
        
        introspection_result = IntrospectionResult(
            id="test",
            introspection_type=IntrospectionType.REASONING,
            findings=[{
                'type': 'unused_rules',
                'count': 2,
                'rules': [{'id': 'rule1'}, {'id': 'rule2'}]
            }],
            recommendations=[],
            confidence=ConfidenceLevel.HIGH,
            timestamp=time.time(),
            reasoning_trace=[]
        )
        
        result = self.optimizer.optimize(introspection_result, auto_apply=True)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.optimization_type, OptimizationType.RULE_OPTIMIZATION)
        self.assertIn(result.status, [OptimizationStatus.COMPLETED, OptimizationStatus.FAILED])
        
    def test_optimize_skill_optimization(self):
        """Test skill optimization."""
        from egdol.omnimind.reflexive.introspector import IntrospectionResult
        
        introspection_result = IntrospectionResult(
            id="test",
            introspection_type=IntrospectionType.SKILLS,
            findings=[{
                'type': 'underperforming_skill',
                'skill': 'analysis',
                'success_rate': 0.5
            }],
            recommendations=[],
            confidence=ConfidenceLevel.HIGH,
            timestamp=time.time(),
            reasoning_trace=[]
        )
        
        result = self.optimizer.optimize(introspection_result, auto_apply=True)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.optimization_type, OptimizationType.SKILL_OPTIMIZATION)
        
    def test_optimize_memory_optimization(self):
        """Test memory optimization."""
        from egdol.omnimind.reflexive.introspector import IntrospectionResult
        
        introspection_result = IntrospectionResult(
            id="test",
            introspection_type=IntrospectionType.MEMORY,
            findings=[{
                'type': 'large_memory',
                'total_entries': 1500
            }],
            recommendations=[],
            confidence=ConfidenceLevel.HIGH,
            timestamp=time.time(),
            reasoning_trace=[]
        )
        
        result = self.optimizer.optimize(introspection_result, auto_apply=True)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.optimization_type, OptimizationType.MEMORY_OPTIMIZATION)
        
    def test_optimize_performance_optimization(self):
        """Test performance optimization."""
        from egdol.omnimind.reflexive.introspector import IntrospectionResult
        
        introspection_result = IntrospectionResult(
            id="test",
            introspection_type=IntrospectionType.PERFORMANCE,
            findings=[{
                'type': 'slow_execution',
                'average_time': 15.0
            }],
            recommendations=[],
            confidence=ConfidenceLevel.HIGH,
            timestamp=time.time(),
            reasoning_trace=[]
        )
        
        result = self.optimizer.optimize(introspection_result, auto_apply=True)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.optimization_type, OptimizationType.PERFORMANCE_OPTIMIZATION)
        
    def test_get_optimization_history(self):
        """Test optimization history."""
        # Perform some optimizations
        from egdol.omnimind.reflexive.introspector import IntrospectionResult
        
        introspection_result = IntrospectionResult(
            id="test",
            introspection_type=IntrospectionType.REASONING,
            findings=[],
            recommendations=[],
            confidence=ConfidenceLevel.MEDIUM,
            timestamp=time.time(),
            reasoning_trace=[]
        )
        
        self.optimizer.optimize(introspection_result, auto_apply=True)
        
        history = self.optimizer.get_optimization_history()
        
        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0], dict)
        self.assertIn('id', history[0])
        self.assertIn('type', history[0])
        
    def test_get_optimization_stats(self):
        """Test optimization statistics."""
        # Perform some optimizations
        from egdol.omnimind.reflexive.introspector import IntrospectionResult
        
        introspection_result = IntrospectionResult(
            id="test",
            introspection_type=IntrospectionType.REASONING,
            findings=[],
            recommendations=[],
            confidence=ConfidenceLevel.MEDIUM,
            timestamp=time.time(),
            reasoning_trace=[]
        )
        
        self.optimizer.optimize(introspection_result, auto_apply=True)
        
        stats = self.optimizer.get_optimization_stats()
        
        self.assertIn('total_optimizations', stats)
        self.assertIn('successful_optimizations', stats)
        self.assertIn('failed_optimizations', stats)
        self.assertIn('rolled_back_optimizations', stats)
        self.assertIn('success_rate', stats)


class SelfModifierTests(unittest.TestCase):
    """Test the self-modifier."""
    
    def setUp(self):
        self.memory_manager = MockMemoryManager()
        self.skill_router = MockSkillRouter()
        self.planner = MockPlanner()
        self.modifier = SelfModifier(
            memory_manager=self.memory_manager,
            skill_router=self.skill_router,
            planner=self.planner
        )
        
    def test_apply_rule_addition(self):
        """Test rule addition modification."""
        changes = {
            'rule': {'id': 'new_rule', 'name': 'Test Rule', 'conditions': [], 'conclusion': 'test'}
        }
        
        log = self.modifier.apply_modification(
            ModificationType.RULE_ADDITION,
            changes,
            "Adding test rule",
            0.8
        )
        
        self.assertIsNotNone(log)
        self.assertEqual(log.modification_type, ModificationType.RULE_ADDITION)
        self.assertEqual(log.status, ModificationStatus.APPLIED)
        self.assertEqual(log.confidence_score, 0.8)
        
    def test_apply_rule_removal(self):
        """Test rule removal modification."""
        # Add a rule first
        rule = {'id': 'test_rule', 'name': 'Test Rule'}
        self.memory_manager.add_rule(rule)
        
        changes = {'rule_id': 'test_rule'}
        
        log = self.modifier.apply_modification(
            ModificationType.RULE_REMOVAL,
            changes,
            "Removing test rule",
            0.9
        )
        
        self.assertIsNotNone(log)
        self.assertEqual(log.modification_type, ModificationType.RULE_REMOVAL)
        self.assertEqual(log.status, ModificationStatus.APPLIED)
        
    def test_apply_skill_addition(self):
        """Test skill addition modification."""
        changes = {
            'skill_name': 'new_skill',
            'skill_definition': {'name': 'new_skill', 'handler': lambda x: x}
        }
        
        log = self.modifier.apply_modification(
            ModificationType.SKILL_ADDITION,
            changes,
            "Adding new skill",
            0.7
        )
        
        self.assertIsNotNone(log)
        self.assertEqual(log.modification_type, ModificationType.SKILL_ADDITION)
        self.assertEqual(log.status, ModificationStatus.APPLIED)
        
    def test_apply_memory_cleanup(self):
        """Test memory cleanup modification."""
        changes = {'items_removed': 10}
        
        log = self.modifier.apply_modification(
            ModificationType.MEMORY_CLEANUP,
            changes,
            "Cleaning up memory",
            0.6
        )
        
        self.assertIsNotNone(log)
        self.assertEqual(log.modification_type, ModificationType.MEMORY_CLEANUP)
        self.assertEqual(log.status, ModificationStatus.APPLIED)
        
    def test_get_modification_history(self):
        """Test modification history."""
        # Apply some modifications
        self.modifier.apply_modification(
            ModificationType.RULE_ADDITION,
            {'rule': {'id': 'rule1'}},
            "Test modification",
            0.8
        )
        
        history = self.modifier.get_modification_history()
        
        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0], dict)
        self.assertIn('id', history[0])
        self.assertIn('type', history[0])
        
    def test_get_modification_stats(self):
        """Test modification statistics."""
        # Apply some modifications
        self.modifier.apply_modification(
            ModificationType.RULE_ADDITION,
            {'rule': {'id': 'rule1'}},
            "Test modification",
            0.8
        )
        
        stats = self.modifier.get_modification_stats()
        
        self.assertIn('total_modifications', stats)
        self.assertIn('successful_modifications', stats)
        self.assertIn('failed_modifications', stats)
        self.assertIn('rolled_back_modifications', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('average_confidence', stats)


class PerformanceMonitorTests(unittest.TestCase):
    """Test the performance monitor."""
    
    def setUp(self):
        self.monitor = PerformanceMonitor()
        self.memory_manager = MockMemoryManager()
        self.skill_router = MockSkillRouter()
        self.planner = MockPlanner()
        
    def test_capture_metrics(self):
        """Test metrics capture."""
        metrics = self.monitor.capture_metrics(
            planner=self.planner,
            memory_manager=self.memory_manager,
            skill_router=self.skill_router
        )
        
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreaterEqual(metrics.cpu_usage, 0)
        self.assertGreaterEqual(metrics.memory_usage, 0)
        self.assertGreaterEqual(metrics.memory_available, 0)
        
    def test_get_performance_trends(self):
        """Test performance trends."""
        # Capture some metrics
        for _ in range(5):
            self.monitor.capture_metrics()
            time.sleep(0.01)
            
        trends = self.monitor.get_performance_trends(hours=1)
        
        self.assertIsInstance(trends, dict)
        self.assertIn('data_points', trends)
        self.assertIn('cpu_trend', trends)
        self.assertIn('memory_trend', trends)
        
    def test_get_bottleneck_analysis(self):
        """Test bottleneck analysis."""
        # Capture some metrics
        for _ in range(5):
            self.monitor.capture_metrics()
            time.sleep(0.01)
            
        analysis = self.monitor.get_bottleneck_analysis()
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('bottlenecks', analysis)
        self.assertIn('total_bottlenecks', analysis)
        
    def test_get_optimization_recommendations(self):
        """Test optimization recommendations."""
        # Capture some metrics
        for _ in range(5):
            self.monitor.capture_metrics()
            time.sleep(0.01)
            
        recommendations = self.monitor.get_optimization_recommendations()
        
        self.assertIsInstance(recommendations, list)
        
    def test_get_performance_summary(self):
        """Test performance summary."""
        # Capture some metrics
        for _ in range(5):
            self.monitor.capture_metrics()
            time.sleep(0.01)
            
        summary = self.monitor.get_performance_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_measurements', summary)
        self.assertIn('average_cpu', summary)
        self.assertIn('average_memory', summary)
        
    def test_set_bottleneck_thresholds(self):
        """Test setting bottleneck thresholds."""
        new_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 95.0
        }
        
        self.monitor.set_bottleneck_thresholds(new_thresholds)
        
        self.assertEqual(self.monitor.bottleneck_thresholds['cpu_usage'], 90.0)
        self.assertEqual(self.monitor.bottleneck_thresholds['memory_usage'], 95.0)


class ReasoningAnalyzerTests(unittest.TestCase):
    """Test the reasoning analyzer."""
    
    def setUp(self):
        self.memory_manager = MockMemoryManager()
        self.planner = MockPlanner()
        self.analyzer = ReasoningAnalyzer(
            memory_manager=self.memory_manager,
            planner=self.planner
        )
        
    def test_analyze_reasoning_patterns(self):
        """Test reasoning pattern analysis."""
        result = self.analyzer.analyze_reasoning_patterns()
        
        self.assertIsNotNone(result)
        self.assertEqual(result.analyzer_type, 'reasoning')
        self.assertIsInstance(result.findings, list)
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class SkillAnalyzerTests(unittest.TestCase):
    """Test the skill analyzer."""
    
    def setUp(self):
        self.skill_router = MockSkillRouter()
        self.analyzer = SkillAnalyzer(skill_router=self.skill_router)
        
    def test_analyze_skill_performance(self):
        """Test skill performance analysis."""
        # Add some skill statistics
        self.skill_router.skill_stats = {
            'analysis': {'success_rate': 0.5, 'execution_count': 10},
            'security': {'success_rate': 0.9, 'execution_count': 5}
        }
        
        result = self.analyzer.analyze_skill_performance()
        
        self.assertIsNotNone(result)
        self.assertEqual(result.analyzer_type, 'skills')
        self.assertIsInstance(result.findings, list)
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.confidence, float)


class MemoryAnalyzerTests(unittest.TestCase):
    """Test the memory analyzer."""
    
    def setUp(self):
        self.memory_manager = MockMemoryManager()
        self.analyzer = MemoryAnalyzer(memory_manager=self.memory_manager)
        
    def test_analyze_memory_usage(self):
        """Test memory usage analysis."""
        # Set up memory stats
        self.memory_manager.memory_stats = {'total_entries': 1500}
        
        result = self.analyzer.analyze_memory_usage()
        
        self.assertIsNotNone(result)
        self.assertEqual(result.analyzer_type, 'memory')
        self.assertIsInstance(result.findings, list)
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.confidence, float)


class IntegrationTests(unittest.TestCase):
    """Integration tests for the reflexive system."""
    
    def setUp(self):
        self.memory_manager = MockMemoryManager()
        self.skill_router = MockSkillRouter()
        self.planner = MockPlanner()
        
        self.introspector = SelfIntrospector(
            memory_manager=self.memory_manager,
            skill_router=self.skill_router,
            planner=self.planner
        )
        
        self.optimizer = SelfOptimizer(
            memory_manager=self.memory_manager,
            skill_router=self.skill_router,
            planner=self.planner
        )
        
        self.modifier = SelfModifier(
            memory_manager=self.memory_manager,
            skill_router=self.skill_router,
            planner=self.planner
        )
        
        self.monitor = PerformanceMonitor()
        
    def test_full_reflexive_cycle(self):
        """Test a complete reflexive cycle."""
        # 1. Capture performance metrics
        metrics = self.monitor.capture_metrics(
            planner=self.planner,
            memory_manager=self.memory_manager,
            skill_router=self.skill_router
        )
        
        self.assertIsNotNone(metrics)
        
        # 2. Perform introspection
        introspection_result = self.introspector.introspect(IntrospectionType.REASONING)
        
        self.assertIsNotNone(introspection_result)
        
        # 3. Apply optimizations
        optimization_result = self.optimizer.optimize(introspection_result, auto_apply=True)
        
        self.assertIsNotNone(optimization_result)
        
        # 4. Apply modifications
        modification_log = self.modifier.apply_modification(
            ModificationType.RULE_ADDITION,
            {'rule': {'id': 'test_rule', 'name': 'Test Rule'}},
            "Test modification",
            0.8
        )
        
        self.assertIsNotNone(modification_log)
        
        # 5. Verify all components are working
        self.assertGreater(len(self.introspector.get_introspection_history()), 0)
        self.assertGreater(len(self.optimizer.get_optimization_history()), 0)
        self.assertGreater(len(self.modifier.get_modification_history()), 0)
        
    def test_self_correction_workflow(self):
        """Test self-correction workflow."""
        # Simulate a problem by adding conflicting rules
        rule1 = {'id': 'rule1', 'name': 'Rule 1', 'conclusion': 'A'}
        rule2 = {'id': 'rule2', 'name': 'Rule 2', 'conclusion': 'not A'}
        
        self.memory_manager.add_rule(rule1)
        self.memory_manager.add_rule(rule2)
        
        # Introspect to find the problem
        introspection_result = self.introspector.introspect(IntrospectionType.RULES)
        
        # Should find rule conflicts
        rule_conflicts = [f for f in introspection_result.findings if f.get('type') == 'rule_conflicts']
        self.assertGreater(len(rule_conflicts), 0)
        
        # Apply optimization to fix the problem
        optimization_result = self.optimizer.optimize(introspection_result, auto_apply=True)
        
        self.assertIsNotNone(optimization_result)
        
        # Verify the system is working
        self.assertEqual(optimization_result.optimization_type, OptimizationType.RULE_OPTIMIZATION)


if __name__ == '__main__':
    unittest.main()
