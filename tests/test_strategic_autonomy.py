"""
Tests for OmniMind Strategic Autonomy System
Comprehensive testing of autonomous goal generation, scenario simulation, policy evolution, and strategic coordination.
"""

import unittest
import time
from egdol.omnimind.strategic import (
    GoalGenerator, StrategicGoal, GoalType, GoalPriority, GoalStatus,
    ScenarioSimulator, SimulationResult, ScenarioType, SimulationStatus,
    PolicyEvolver, NetworkPolicy, PolicyType, PolicyStatus,
    RiskAssessor, RiskAssessment, RiskLevel, RiskType, RiskMitigation,
    AutonomousOptimizer, OptimizationTask, OptimizationStrategy, OptimizationResult,
    KnowledgeLifecycleManager, KnowledgeItem, KnowledgeState, LifecycleAction,
    PerformanceForecaster, ForecastResult, TrendType,
    StrategicCoordinator, StrategicDecision, DecisionType
)


class MockNetwork:
    """Mock network for testing."""
    
    def __init__(self):
        self.agents = {}
        self.connections = {}
        
    def get_network_statistics(self):
        return {
            'network_efficiency': 0.7,
            'total_agents': len(self.agents),
            'unique_skills': 5
        }
        
    def detect_network_bottlenecks(self):
        return [
            {'type': 'overloaded_agent', 'agent_id': 'agent1', 'severity': 'high'}
        ]


class MockLearningSystem:
    """Mock learning system for testing."""
    
    def get_learning_statistics(self):
        return {
            'total_learning_events': 10,
            'success_rate': 0.8,
            'average_confidence': 0.7,
            'unique_skills_shared': 3
        }


class MockMonitor:
    """Mock monitor for testing."""
    
    def get_monitoring_statistics(self):
        return {
            'total_alerts': 5,
            'active_alerts': 2,
            'resolution_rate': 0.8
        }


class MockCoordinator:
    """Mock coordinator for testing."""
    
    def get_coordination_statistics(self):
        return {
            'total_tasks': 20,
            'success_rate': 0.75,
            'average_duration': 300.0
        }
        
    def detect_coordination_issues(self):
        return [
            {'type': 'agent_overload', 'severity': 'medium'}
        ]


class MockResourceManager:
    """Mock resource manager for testing."""
    
    def get_resource_statistics(self):
        return {
            'total_allocations': 15,
            'resource_usage': {
                'COMPUTATIONAL': {'available': 100.0, 'allocated': 60.0},
                'MEMORY': {'available': 50.0, 'allocated': 30.0}
            }
        }
        
    def get_resource_availability(self, resource_type):
        return 100.0


class GoalGeneratorTests(unittest.TestCase):
    """Test the goal generator."""
    
    def setUp(self):
        self.network = MockNetwork()
        self.learning_system = MockLearningSystem()
        self.monitor = MockMonitor()
        self.goal_generator = GoalGenerator(self.network, self.learning_system, self.monitor)
        
    def test_analyze_network_state(self):
        """Test network state analysis."""
        analysis = self.goal_generator.analyze_network_state()
        
        self.assertIn('timestamp', analysis)
        self.assertIn('network_stats', analysis)
        self.assertIn('learning_stats', analysis)
        self.assertIn('monitoring_stats', analysis)
        self.assertIn('knowledge_gaps', analysis)
        self.assertIn('performance_bottlenecks', analysis)
        self.assertIn('collaboration_opportunities', analysis)
        self.assertIn('skill_gaps', analysis)
        self.assertIn('risk_factors', analysis)
        
    def test_generate_strategic_goals(self):
        """Test strategic goal generation."""
        goals = self.goal_generator.generate_strategic_goals(max_goals=3)
        
        self.assertIsInstance(goals, list)
        self.assertLessEqual(len(goals), 3)
        
        for goal in goals:
            self.assertIsInstance(goal, StrategicGoal)
            self.assertIn(goal.goal_type, GoalType)
            self.assertIn(goal.priority, GoalPriority)
            self.assertEqual(goal.status, GoalStatus.PROPOSED)
            
    def test_approve_goal(self):
        """Test goal approval."""
        goal = StrategicGoal(
            id="test_goal",
            goal_type=GoalType.SKILL_DEVELOPMENT,
            title="Test Goal",
            description="Test description",
            priority=GoalPriority.HIGH,
            status=GoalStatus.PROPOSED,
            created_at=time.time()
        )
        self.goal_generator.goals[goal.id] = goal
        
        result = self.goal_generator.approve_goal(goal.id)
        
        self.assertTrue(result)
        self.assertEqual(goal.status, GoalStatus.APPROVED)
        
    def test_start_goal(self):
        """Test goal start."""
        goal = StrategicGoal(
            id="test_goal",
            goal_type=GoalType.SKILL_DEVELOPMENT,
            title="Test Goal",
            description="Test description",
            priority=GoalPriority.HIGH,
            status=GoalStatus.APPROVED,
            created_at=time.time()
        )
        self.goal_generator.goals[goal.id] = goal
        
        result = self.goal_generator.start_goal(goal.id)
        
        self.assertTrue(result)
        self.assertEqual(goal.status, GoalStatus.IN_PROGRESS)
        
    def test_complete_goal(self):
        """Test goal completion."""
        goal = StrategicGoal(
            id="test_goal",
            goal_type=GoalType.SKILL_DEVELOPMENT,
            title="Test Goal",
            description="Test description",
            priority=GoalPriority.HIGH,
            status=GoalStatus.IN_PROGRESS,
            created_at=time.time()
        )
        self.goal_generator.goals[goal.id] = goal
        
        result = self.goal_generator.complete_goal(goal.id, True, {'metric': 0.8})
        
        self.assertTrue(result)
        self.assertEqual(goal.status, GoalStatus.COMPLETED)
        self.assertIsNotNone(goal.actual_completion)
        
    def test_get_goal_statistics(self):
        """Test goal statistics."""
        # Add some test goals
        goal1 = StrategicGoal(
            id="goal1", goal_type=GoalType.SKILL_DEVELOPMENT, title="Goal 1",
            description="Test", priority=GoalPriority.HIGH, status=GoalStatus.COMPLETED,
            created_at=time.time(), expected_impact=0.8
        )
        goal2 = StrategicGoal(
            id="goal2", goal_type=GoalType.NETWORK_OPTIMIZATION, title="Goal 2",
            description="Test", priority=GoalPriority.MEDIUM, status=GoalStatus.FAILED,
            created_at=time.time(), expected_impact=0.6
        )
        
        self.goal_generator.goals[goal1.id] = goal1
        self.goal_generator.goals[goal2.id] = goal2
        
        stats = self.goal_generator.get_goal_statistics()
        
        self.assertEqual(stats['total_goals'], 2)
        self.assertEqual(stats['completed_goals'], 1)
        self.assertEqual(stats['failed_goals'], 1)
        self.assertEqual(stats['success_rate'], 0.5)
        self.assertIn('type_distribution', stats)


class ScenarioSimulatorTests(unittest.TestCase):
    """Test the scenario simulator."""
    
    def setUp(self):
        self.network = MockNetwork()
        self.coordinator = MockCoordinator()
        self.resource_manager = MockResourceManager()
        self.simulator = ScenarioSimulator(self.network, self.coordinator, self.resource_manager)
        
    def test_simulate_goal_execution(self):
        """Test goal execution simulation."""
        goal_data = {
            'required_skills': ['analysis', 'reasoning'],
            'required_resources': {'COMPUTATIONAL': 50.0},
            'participating_agents': ['agent1', 'agent2'],
            'success_probability': 0.8
        }
        
        result = self.simulator.simulate_goal_execution("test_goal", goal_data)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.scenario_type, ScenarioType.GOAL_EXECUTION)
        self.assertEqual(result.goal_id, "test_goal")
        self.assertEqual(result.status, SimulationStatus.COMPLETED)
        self.assertGreater(result.success_probability, 0)
        self.assertGreater(result.confidence_score, 0)
        
    def test_simulate_resource_allocation(self):
        """Test resource allocation simulation."""
        allocation_plan = {
            'total_resources': {'COMPUTATIONAL': 100.0, 'MEMORY': 50.0},
            'strategy': 'equal'
        }
        
        result = self.simulator.simulate_resource_allocation(allocation_plan)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.scenario_type, ScenarioType.RESOURCE_ALLOCATION)
        self.assertEqual(result.status, SimulationStatus.COMPLETED)
        self.assertGreater(result.success_probability, 0)
        
    def test_simulate_network_optimization(self):
        """Test network optimization simulation."""
        optimization_plan = {
            'targets': ['efficiency', 'communication'],
            'strategy': 'gradual'
        }
        
        result = self.simulator.simulate_network_optimization(optimization_plan)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.scenario_type, ScenarioType.NETWORK_OPTIMIZATION)
        self.assertEqual(result.status, SimulationStatus.COMPLETED)
        self.assertGreater(result.success_probability, 0)
        
    def test_get_simulation_statistics(self):
        """Test simulation statistics."""
        # Add some test simulations
        sim1 = SimulationResult(
            id="sim1", scenario_type=ScenarioType.GOAL_EXECUTION, goal_id="goal1",
            status=SimulationStatus.COMPLETED, created_at=time.time(),
            success_probability=0.8, confidence_score=0.7
        )
        sim2 = SimulationResult(
            id="sim2", scenario_type=ScenarioType.RESOURCE_ALLOCATION, goal_id=None,
            status=SimulationStatus.COMPLETED, created_at=time.time(),
            success_probability=0.6, confidence_score=0.8
        )
        
        self.simulator.simulations[sim1.id] = sim1
        self.simulator.simulations[sim2.id] = sim2
        
        stats = self.simulator.get_simulation_statistics()
        
        self.assertEqual(stats['total_simulations'], 2)
        self.assertEqual(stats['completed_simulations'], 2)
        self.assertAlmostEqual(stats['average_success_probability'], 0.7)
        self.assertAlmostEqual(stats['average_confidence'], 0.75)


class PolicyEvolverTests(unittest.TestCase):
    """Test the policy evolver."""
    
    def setUp(self):
        self.network = MockNetwork()
        self.monitor = MockMonitor()
        self.learning_system = MockLearningSystem()
        self.evolver = PolicyEvolver(self.network, self.monitor, self.learning_system)
        
    def test_analyze_network_performance(self):
        """Test network performance analysis."""
        analysis = self.evolver.analyze_network_performance()
        
        self.assertIn('timestamp', analysis)
        self.assertIn('network_efficiency', analysis)
        self.assertIn('communication_patterns', analysis)
        self.assertIn('coordination_effectiveness', analysis)
        self.assertIn('resource_utilization', analysis)
        self.assertIn('learning_effectiveness', analysis)
        self.assertIn('collaboration_patterns', analysis)
        self.assertIn('bottlenecks', analysis)
        self.assertIn('opportunities', analysis)
        
    def test_evolve_policies(self):
        """Test policy evolution."""
        policies = self.evolver.evolve_policies(max_evolutions=2)
        
        self.assertIsInstance(policies, list)
        self.assertLessEqual(len(policies), 2)
        
        for policy in policies:
            self.assertIsInstance(policy, NetworkPolicy)
            self.assertIn(policy.policy_type, PolicyType)
            self.assertEqual(policy.status, PolicyStatus.DRAFT)
            
    def test_activate_policy(self):
        """Test policy activation."""
        policy = NetworkPolicy(
            id="test_policy",
            policy_type=PolicyType.COMMUNICATION,
            name="Test Policy",
            description="Test description",
            rules={'test': 'value'},
            status=PolicyStatus.DRAFT,
            created_at=time.time(),
            last_modified=time.time()
        )
        self.evolver.policies[policy.id] = policy
        
        result = self.evolver.activate_policy(policy.id)
        
        self.assertTrue(result)
        self.assertEqual(policy.status, PolicyStatus.ACTIVE)
        
    def test_update_policy_effectiveness(self):
        """Test policy effectiveness update."""
        policy = NetworkPolicy(
            id="test_policy",
            policy_type=PolicyType.COMMUNICATION,
            name="Test Policy",
            description="Test description",
            rules={'test': 'value'},
            status=PolicyStatus.ACTIVE,
            created_at=time.time(),
            last_modified=time.time()
        )
        self.evolver.policies[policy.id] = policy
        
        result = self.evolver.update_policy_effectiveness(policy.id, 0.8)
        
        self.assertTrue(result)
        self.assertEqual(policy.effectiveness_score, 0.8)
        
    def test_get_policy_statistics(self):
        """Test policy statistics."""
        # Add some test policies
        policy1 = NetworkPolicy(
            id="policy1", policy_type=PolicyType.COMMUNICATION, name="Policy 1",
            description="Test", rules={}, status=PolicyStatus.ACTIVE,
            created_at=time.time(), last_modified=time.time(), effectiveness_score=0.8
        )
        policy2 = NetworkPolicy(
            id="policy2", policy_type=PolicyType.COORDINATION, name="Policy 2",
            description="Test", rules={}, status=PolicyStatus.DEPRECATED,
            created_at=time.time(), last_modified=time.time(), effectiveness_score=0.6
        )
        
        self.evolver.policies[policy1.id] = policy1
        self.evolver.policies[policy2.id] = policy2
        
        stats = self.evolver.get_policy_statistics()
        
        self.assertEqual(stats['total_policies'], 2)
        self.assertEqual(stats['active_policies'], 1)
        self.assertEqual(stats['deprecated_policies'], 1)
        self.assertAlmostEqual(stats['average_effectiveness'], 0.7)


class RiskAssessorTests(unittest.TestCase):
    """Test the risk assessor."""
    
    def setUp(self):
        self.network = MockNetwork()
        self.monitor = MockMonitor()
        self.coordinator = MockCoordinator()
        self.assessor = RiskAssessor(self.network, self.monitor, self.coordinator)
        
    def test_assess_network_risks(self):
        """Test network risk assessment."""
        risks = self.assessor.assess_network_risks()
        
        self.assertIsInstance(risks, list)
        
        for risk in risks:
            self.assertIsInstance(risk, RiskAssessment)
            self.assertIn(risk.risk_type, RiskType)
            self.assertIn(risk.level, RiskLevel)
            self.assertGreater(risk.probability, 0)
            self.assertGreater(risk.impact, 0)
            
    def test_mitigate_risk(self):
        """Test risk mitigation."""
        risk = RiskAssessment(
            id="test_risk",
            risk_type=RiskType.NETWORK_BOTTLENECK,
            level=RiskLevel.HIGH,
            description="Test risk",
            affected_components=["network"],
            probability=0.8,
            impact=0.7,
            detected_at=time.time()
        )
        self.assessor.assessments[risk.id] = risk
        
        result = self.assessor.mitigate_risk(risk.id, RiskMitigation.MONITORING)
        
        self.assertTrue(result)
        self.assertEqual(risk.status, "mitigated")
        self.assertIsNotNone(risk.resolved_at)
        
    def test_get_risk_statistics(self):
        """Test risk statistics."""
        # Add some test risks
        risk1 = RiskAssessment(
            id="risk1", risk_type=RiskType.NETWORK_BOTTLENECK, level=RiskLevel.HIGH,
            description="Risk 1", affected_components=["network"], probability=0.8,
            impact=0.7, detected_at=time.time()
        )
        risk2 = RiskAssessment(
            id="risk2", risk_type=RiskType.RESOURCE_CONFLICT, level=RiskLevel.MEDIUM,
            description="Risk 2", affected_components=["resources"], probability=0.6,
            impact=0.5, detected_at=time.time(), status="mitigated"
        )
        
        self.assessor.assessments[risk1.id] = risk1
        self.assessor.assessments[risk2.id] = risk2
        
        stats = self.assessor.get_risk_statistics()
        
        self.assertEqual(stats['total_risks'], 2)
        self.assertEqual(stats['active_risks'], 1)
        self.assertEqual(stats['mitigated_risks'], 1)
        self.assertEqual(stats['mitigation_rate'], 0.5)


class AutonomousOptimizerTests(unittest.TestCase):
    """Test the autonomous optimizer."""
    
    def setUp(self):
        self.network = MockNetwork()
        self.monitor = MockMonitor()
        self.learning_system = MockLearningSystem()
        self.policy_evolver = PolicyEvolver(self.network, self.monitor, self.learning_system)
        self.optimizer = AutonomousOptimizer(self.network, self.monitor, self.learning_system, self.policy_evolver)
        
    def test_analyze_optimization_opportunities(self):
        """Test optimization opportunity analysis."""
        opportunities = self.optimizer.analyze_optimization_opportunities()
        
        self.assertIsInstance(opportunities, list)
        
        for task in opportunities:
            self.assertIsInstance(task, OptimizationTask)
            self.assertIn(task.strategy, OptimizationStrategy)
            self.assertGreater(len(task.optimization_actions), 0)
            
    def test_execute_optimization(self):
        """Test optimization execution."""
        task = OptimizationTask(
            id="test_task",
            strategy=OptimizationStrategy.PERFORMANCE,
            description="Test optimization",
            target_metrics={'efficiency': 0.9},
            current_metrics={'efficiency': 0.7},
            optimization_actions=[{'type': 'optimize_connections', 'priority': 'high'}]
        )
        self.optimizer.optimization_tasks[task.id] = task
        
        result = self.optimizer.execute_optimization(task.id)
        
        self.assertTrue(result)
        self.assertEqual(task.status, "completed")
        self.assertIsNotNone(task.result)
        
    def test_rollback_optimization(self):
        """Test optimization rollback."""
        task = OptimizationTask(
            id="test_task",
            strategy=OptimizationStrategy.PERFORMANCE,
            description="Test optimization",
            target_metrics={'efficiency': 0.9},
            current_metrics={'efficiency': 0.7},
            optimization_actions=[{'type': 'optimize_connections', 'priority': 'high'}]
        )
        self.optimizer.optimization_tasks[task.id] = task
        
        result = self.optimizer.rollback_optimization(task.id)
        
        self.assertTrue(result)
        self.assertEqual(task.status, "rolled_back")
        
    def test_get_optimization_statistics(self):
        """Test optimization statistics."""
        # Add some test tasks
        task1 = OptimizationTask(
            id="task1", strategy=OptimizationStrategy.PERFORMANCE, description="Task 1",
            target_metrics={}, current_metrics={}, optimization_actions=[],
            status="completed", result=OptimizationResult.SUCCESS, improvement_score=0.8
        )
        task2 = OptimizationTask(
            id="task2", strategy=OptimizationStrategy.EFFICIENCY, description="Task 2",
            target_metrics={}, current_metrics={}, optimization_actions=[],
            status="completed", result=OptimizationResult.FAILED, improvement_score=0.3
        )
        
        self.optimizer.optimization_tasks[task1.id] = task1
        self.optimizer.optimization_tasks[task2.id] = task2
        
        stats = self.optimizer.get_optimization_statistics()
        
        self.assertEqual(stats['total_tasks'], 2)
        self.assertEqual(stats['completed_tasks'], 2)
        self.assertEqual(stats['successful_tasks'], 1)
        self.assertEqual(stats['success_rate'], 0.5)
        self.assertAlmostEqual(stats['average_improvement'], 0.55)


class KnowledgeLifecycleManagerTests(unittest.TestCase):
    """Test the knowledge lifecycle manager."""
    
    def setUp(self):
        self.network = MockNetwork()
        self.learning_system = MockLearningSystem()
        self.lifecycle_manager = KnowledgeLifecycleManager(self.network, self.learning_system)
        
    def test_analyze_knowledge_lifecycle(self):
        """Test knowledge lifecycle analysis."""
        items = self.lifecycle_manager.analyze_knowledge_lifecycle()
        
        self.assertIsInstance(items, list)
        
        for item in items:
            self.assertIsInstance(item, KnowledgeItem)
            self.assertIn(item.state, KnowledgeState)
            self.assertGreater(len(item.lifecycle_actions), 0)
            
    def test_apply_lifecycle_action(self):
        """Test lifecycle action application."""
        item = KnowledgeItem(
            id="test_item",
            content="Test knowledge",
            knowledge_type="fact",
            state=KnowledgeState.UNUSED,
            created_at=time.time(),
            last_accessed=time.time() - 86400,  # 24 hours ago
            lifecycle_actions=[LifecycleAction.RETIRE]
        )
        self.lifecycle_manager.knowledge_items[item.id] = item
        
        result = self.lifecycle_manager.apply_lifecycle_action(item.id, LifecycleAction.RETIRE)
        
        self.assertTrue(result)
        self.assertEqual(item.state, KnowledgeState.OBSOLETE)
        
    def test_get_lifecycle_statistics(self):
        """Test lifecycle statistics."""
        # Add some test items
        item1 = KnowledgeItem(
            id="item1", content="Item 1", knowledge_type="fact",
            state=KnowledgeState.ACTIVE, created_at=time.time(),
            last_accessed=time.time(), usage_count=5, confidence_score=0.8
        )
        item2 = KnowledgeItem(
            id="item2", content="Item 2", knowledge_type="rule",
            state=KnowledgeState.OBSOLETE, created_at=time.time(),
            last_accessed=time.time(), usage_count=1, confidence_score=0.3
        )
        
        self.lifecycle_manager.knowledge_items[item1.id] = item1
        self.lifecycle_manager.knowledge_items[item2.id] = item2
        
        stats = self.lifecycle_manager.get_lifecycle_statistics()
        
        self.assertEqual(stats['total_items'], 2)
        self.assertEqual(stats['active_items'], 1)
        self.assertEqual(stats['obsolete_items'], 1)
        self.assertAlmostEqual(stats['average_usage'], 3.0)
        self.assertAlmostEqual(stats['average_confidence'], 0.55)


class PerformanceForecasterTests(unittest.TestCase):
    """Test the performance forecaster."""
    
    def setUp(self):
        self.network = MockNetwork()
        self.monitor = MockMonitor()
        self.resource_manager = MockResourceManager()
        self.forecaster = PerformanceForecaster(self.network, self.monitor, self.resource_manager)
        
    def test_forecast_network_performance(self):
        """Test network performance forecasting."""
        forecasts = self.forecaster.forecast_network_performance(horizon_hours=24.0)
        
        self.assertIsInstance(forecasts, list)
        self.assertGreater(len(forecasts), 0)
        
        for forecast in forecasts:
            self.assertIsInstance(forecast, ForecastResult)
            self.assertGreater(forecast.confidence, 0)
            self.assertIn(forecast.trend_type, TrendType)
            self.assertGreater(len(forecast.recommendations), 0)
            
    def test_update_performance_data(self):
        """Test performance data update."""
        self.forecaster.update_performance_data('network_efficiency', 0.8)
        self.forecaster.update_performance_data('network_efficiency', 0.9)
        
        self.assertEqual(len(self.forecaster.performance_data['network_efficiency']), 2)
        self.assertEqual(self.forecaster.performance_data['network_efficiency'][-1], 0.9)
        
    def test_get_forecast_statistics(self):
        """Test forecast statistics."""
        # Add some test forecasts
        forecast1 = ForecastResult(
            id="forecast1", metric_name="network_efficiency", current_value=0.7,
            predicted_value=0.8, confidence=0.8, trend_type=TrendType.IMPROVING,
            forecast_horizon=24.0, created_at=time.time()
        )
        forecast2 = ForecastResult(
            id="forecast2", metric_name="communication_volume", current_value=50,
            predicted_value=60, confidence=0.6, trend_type=TrendType.DECLINING,
            forecast_horizon=24.0, created_at=time.time()
        )
        
        self.forecaster.forecasts[forecast1.id] = forecast1
        self.forecaster.forecasts[forecast2.id] = forecast2
        
        stats = self.forecaster.get_forecast_statistics()
        
        self.assertEqual(stats['total_forecasts'], 2)
        self.assertAlmostEqual(stats['average_confidence'], 0.7)


class StrategicCoordinatorTests(unittest.TestCase):
    """Test the strategic coordinator."""
    
    def setUp(self):
        self.network = MockNetwork()
        self.goal_generator = GoalGenerator(self.network, MockLearningSystem(), MockMonitor())
        self.scenario_simulator = ScenarioSimulator(self.network, MockCoordinator(), MockResourceManager())
        self.policy_evolver = PolicyEvolver(self.network, MockMonitor(), MockLearningSystem())
        self.risk_assessor = RiskAssessor(self.network, MockMonitor(), MockCoordinator())
        self.autonomous_optimizer = AutonomousOptimizer(self.network, MockMonitor(), MockLearningSystem(), self.policy_evolver)
        self.knowledge_lifecycle = KnowledgeLifecycleManager(self.network, MockLearningSystem())
        self.performance_forecaster = PerformanceForecaster(self.network, MockMonitor(), MockResourceManager())
        
        self.coordinator = StrategicCoordinator(
            self.network, self.goal_generator, self.scenario_simulator,
            self.policy_evolver, self.risk_assessor, self.autonomous_optimizer,
            self.knowledge_lifecycle, self.performance_forecaster
        )
        
    def test_execute_strategic_cycle(self):
        """Test strategic cycle execution."""
        cycle_results = self.coordinator.execute_strategic_cycle()
        
        self.assertIn('cycle_id', cycle_results)
        self.assertIn('start_time', cycle_results)
        self.assertIn('end_time', cycle_results)
        self.assertIn('duration', cycle_results)
        self.assertIn('goals_generated', cycle_results)
        self.assertIn('simulations_run', cycle_results)
        self.assertIn('policies_evolved', cycle_results)
        self.assertIn('risks_assessed', cycle_results)
        self.assertIn('optimizations_executed', cycle_results)
        self.assertIn('knowledge_actions', cycle_results)
        self.assertIn('forecasts_made', cycle_results)
        self.assertIn('decisions_made', cycle_results)
        self.assertIn('success_rate', cycle_results)
        
    def test_get_strategic_statistics(self):
        """Test strategic statistics."""
        # Add some test decisions
        decision1 = StrategicDecision(
            id="decision1", decision_type=DecisionType.GOAL_APPROVAL,
            description="Approve goal", rationale="High confidence",
            confidence=0.8, impact_assessment={}, created_at=time.time(),
            status="executed"
        )
        decision2 = StrategicDecision(
            id="decision2", decision_type=DecisionType.POLICY_CHANGE,
            description="Change policy", rationale="Low confidence",
            confidence=0.4, impact_assessment={}, created_at=time.time(),
            status="failed"
        )
        
        self.coordinator.decisions[decision1.id] = decision1
        self.coordinator.decisions[decision2.id] = decision2
        
        stats = self.coordinator.get_strategic_statistics()
        
        self.assertEqual(stats['total_decisions'], 2)
        self.assertEqual(stats['executed_decisions'], 1)
        self.assertEqual(stats['failed_decisions'], 1)
        self.assertEqual(stats['execution_rate'], 0.5)
        self.assertAlmostEqual(stats['average_confidence'], 0.6)


class IntegrationTests(unittest.TestCase):
    """Integration tests for the strategic autonomy system."""
    
    def setUp(self):
        self.network = MockNetwork()
        self.learning_system = MockLearningSystem()
        self.monitor = MockMonitor()
        self.coordinator = MockCoordinator()
        self.resource_manager = MockResourceManager()
        
        # Initialize all components
        self.goal_generator = GoalGenerator(self.network, self.learning_system, self.monitor)
        self.scenario_simulator = ScenarioSimulator(self.network, self.coordinator, self.resource_manager)
        self.policy_evolver = PolicyEvolver(self.network, self.monitor, self.learning_system)
        self.risk_assessor = RiskAssessor(self.network, self.monitor, self.coordinator)
        self.autonomous_optimizer = AutonomousOptimizer(self.network, self.monitor, self.learning_system, self.policy_evolver)
        self.knowledge_lifecycle = KnowledgeLifecycleManager(self.network, self.learning_system)
        self.performance_forecaster = PerformanceForecaster(self.network, self.monitor, self.resource_manager)
        
        self.strategic_coordinator = StrategicCoordinator(
            self.network, self.goal_generator, self.scenario_simulator,
            self.policy_evolver, self.risk_assessor, self.autonomous_optimizer,
            self.knowledge_lifecycle, self.performance_forecaster
        )
        
    def test_full_strategic_autonomy_workflow(self):
        """Test complete strategic autonomy workflow."""
        # Execute strategic cycle
        cycle_results = self.strategic_coordinator.execute_strategic_cycle()
        
        # Verify cycle completed successfully
        self.assertIn('cycle_id', cycle_results)
        self.assertGreater(cycle_results['duration'], 0)
        
        # Verify all components were involved
        self.assertGreaterEqual(cycle_results['goals_generated'], 0)
        self.assertGreaterEqual(cycle_results['simulations_run'], 0)
        self.assertGreaterEqual(cycle_results['policies_evolved'], 0)
        self.assertGreaterEqual(cycle_results['risks_assessed'], 0)
        self.assertGreaterEqual(cycle_results['optimizations_executed'], 0)
        self.assertGreaterEqual(cycle_results['knowledge_actions'], 0)
        self.assertGreaterEqual(cycle_results['forecasts_made'], 0)
        self.assertGreaterEqual(cycle_results['decisions_made'], 0)
        
        # Verify success rate is calculated
        self.assertGreaterEqual(cycle_results['success_rate'], 0)
        self.assertLessEqual(cycle_results['success_rate'], 1)
        
    def test_strategic_autonomy_resilience(self):
        """Test strategic autonomy system resilience."""
        # Execute multiple cycles
        cycles = []
        for i in range(3):
            cycle_results = self.strategic_coordinator.execute_strategic_cycle()
            cycles.append(cycle_results)
            
        # Verify all cycles completed
        self.assertEqual(len(cycles), 3)
        for cycle in cycles:
            self.assertIn('cycle_id', cycle)
            self.assertGreater(cycle['duration'], 0)
            
        # Verify system maintains state
        self.assertGreater(len(self.strategic_coordinator.strategic_cycles), 0)
        
    def test_strategic_autonomy_performance(self):
        """Test strategic autonomy system performance."""
        start_time = time.time()
        
        # Execute strategic cycle
        cycle_results = self.strategic_coordinator.execute_strategic_cycle()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify reasonable execution time (should be fast for mock components)
        self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds
        
        # Verify cycle duration is recorded
        self.assertGreater(cycle_results['duration'], 0)
        self.assertLessEqual(cycle_results['duration'], execution_time)


if __name__ == '__main__':
    unittest.main()
