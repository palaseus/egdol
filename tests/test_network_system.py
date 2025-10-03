"""
Tests for OmniMind Network System
Comprehensive testing of multi-agent network, communication, coordination, learning, monitoring, and emergent behavior.
"""

import unittest
import time
from egdol.omnimind.network import (
    AgentNetwork, NetworkTopology, AgentStatus,
    MessageBus, Message, MessageType, MessagePriority,
    TaskCoordinator, GoalNegotiator, ResourceManager, TaskStatus, ResourceType,
    NetworkLearning, KnowledgePropagation, SkillSharing, LearningType, LearningStatus,
    NetworkMonitor, PerformanceAnalyzer, ConflictDetector, MonitorType, AlertLevel,
    EmergentBehavior, PatternDetector, CollaborationEngine, EmergentType, PatternStatus
)


class MockMemoryManager:
    """Mock memory manager for testing."""
    
    def __init__(self):
        self.facts = []
        self.rules = []
        
    def add_fact(self, fact):
        self.facts.append(fact)
        
    def add_rule(self, rule):
        self.rules.append(rule)
        
    def get_all_facts(self):
        return self.facts
        
    def get_all_rules(self):
        return self.rules


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


class AgentNetworkTests(unittest.TestCase):
    """Test the agent network."""
    
    def setUp(self):
        self.network = AgentNetwork(network_id="test_network", topology=NetworkTopology.MESH)
        self.memory_manager = MockMemoryManager()
        self.skill_router = MockSkillRouter()
        self.planner = MockPlanner()
        
    def test_add_agent(self):
        """Test adding an agent to the network."""
        agent = self.network.add_agent(
            name="TestAgent",
            persona_type="general",
            skills=["analysis", "reasoning"],
            memory_manager=self.memory_manager,
            skill_router=self.skill_router,
            planner=self.planner
        )
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.persona_type, "general")
        self.assertEqual(agent.skills, ["analysis", "reasoning"])
        self.assertEqual(agent.status, AgentStatus.IDLE)
        
    def test_remove_agent(self):
        """Test removing an agent from the network."""
        agent = self.network.add_agent("TestAgent", "general", ["analysis"])
        
        result = self.network.remove_agent(agent.id)
        self.assertTrue(result)
        self.assertIsNone(self.network.get_agent(agent.id))
        
    def test_connect_agents(self):
        """Test connecting agents."""
        agent1 = self.network.add_agent("Agent1", "general", ["analysis"])
        agent2 = self.network.add_agent("Agent2", "general", ["reasoning"])
        
        result = self.network.connect_agents(agent1.id, agent2.id)
        self.assertTrue(result)
        
        connected = self.network.get_connected_agents(agent1.id)
        self.assertEqual(len(connected), 1)
        self.assertEqual(connected[0].id, agent2.id)
        
    def test_broadcast_message(self):
        """Test broadcasting messages."""
        agent1 = self.network.add_agent("Agent1", "general", ["analysis"])
        agent2 = self.network.add_agent("Agent2", "general", ["reasoning"])
        
        message = {"type": "test", "content": "Hello"}
        count = self.network.broadcast_message(agent1.id, message, [agent2.id])
        
        self.assertEqual(count, 1)
        
        messages = self.network.get_messages_for_agent(agent2.id)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["message"], message)
        
    def test_get_network_topology(self):
        """Test getting network topology."""
        self.network.add_agent("Agent1", "general", ["analysis"])
        self.network.add_agent("Agent2", "general", ["reasoning"])
        
        topology = self.network.get_network_topology()
        
        self.assertEqual(topology["network_id"], "test_network")
        self.assertEqual(topology["topology"], "MESH")
        self.assertEqual(topology["agent_count"], 2)
        
    def test_get_network_statistics(self):
        """Test getting network statistics."""
        self.network.add_agent("Agent1", "general", ["analysis"])
        self.network.add_agent("Agent2", "general", ["reasoning"])
        
        stats = self.network.get_network_statistics()
        
        self.assertEqual(stats["total_agents"], 2)
        self.assertEqual(stats["unique_skills"], 2)
        self.assertIn("network_efficiency", stats)
        
    def test_detect_network_bottlenecks(self):
        """Test detecting network bottlenecks."""
        agent = self.network.add_agent("TestAgent", "general", ["analysis"])
        agent.status = AgentStatus.BUSY
        agent.last_activity = time.time() - 400  # 6+ minutes ago
        
        bottlenecks = self.network.detect_network_bottlenecks()
        
        self.assertGreater(len(bottlenecks), 0)
        self.assertEqual(bottlenecks[0]["type"], "overloaded_agent")


class MessageBusTests(unittest.TestCase):
    """Test the message bus."""
    
    def setUp(self):
        self.message_bus = MessageBus("test_network")
        
    def test_send_message(self):
        """Test sending a message."""
        message_id = self.message_bus.send_message(
            sender_id="agent1",
            recipient_id="agent2",
            message_type=MessageType.QUERY,
            content={"question": "What is the weather?"},
            priority=MessagePriority.NORMAL
        )
        
        self.assertIsNotNone(message_id)
        self.assertIn(message_id, self.message_bus.messages)
        
    def test_broadcast_message(self):
        """Test broadcasting messages."""
        message_ids = self.message_bus.broadcast_message(
            sender_id="agent1",
            recipient_ids=["agent2", "agent3"],
            message_type=MessageType.QUERY,
            content={"question": "What is the weather?"}
        )
        
        self.assertEqual(len(message_ids), 2)
        
    def test_get_messages_for_agent(self):
        """Test getting messages for an agent."""
        self.message_bus.send_message("agent1", "agent2", MessageType.QUERY, {"test": "data"})
        
        messages = self.message_bus.get_messages_for_agent("agent2")
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].sender_id, "agent1")
        self.assertEqual(messages[0].message_type, MessageType.QUERY)
        
    def test_send_response(self):
        """Test sending a response."""
        original_id = self.message_bus.send_message(
            "agent1", "agent2", MessageType.QUERY, {"question": "test"}
        )
        
        response_id = self.message_bus.send_response(
            original_id, "agent2", {"answer": "response"}
        )
        
        self.assertIsNotNone(response_id)
        
    def test_get_message_statistics(self):
        """Test getting message statistics."""
        self.message_bus.send_message("agent1", "agent2", MessageType.QUERY, {"test": "data"})
        
        stats = self.message_bus.get_message_statistics()
        
        self.assertEqual(stats["total_messages"], 1)
        self.assertEqual(stats["total_sent"], 1)
        self.assertEqual(stats["total_received"], 1)
        
    def test_detect_communication_patterns(self):
        """Test detecting communication patterns."""
        # Send multiple messages to create patterns
        for i in range(15):
            self.message_bus.send_message("agent1", "agent2", MessageType.QUERY, {"test": i})
            
        patterns = self.message_bus.detect_communication_patterns()
        
        self.assertGreater(len(patterns), 0)
        self.assertEqual(patterns[0]["type"], "high_communication_agent")


class TaskCoordinatorTests(unittest.TestCase):
    """Test the task coordinator."""
    
    def setUp(self):
        self.network = AgentNetwork()
        self.coordinator = TaskCoordinator(self.network)
        
    def test_create_task(self):
        """Test creating a task."""
        task_id = self.coordinator.create_task(
            goal="Analyze data",
            description="Perform data analysis",
            required_skills=["analysis"],
            priority=1
        )
        
        self.assertIsNotNone(task_id)
        self.assertIn(task_id, self.coordinator.tasks)
        
    def test_assign_task(self):
        """Test assigning a task."""
        # Add agents to network
        agent1 = self.network.add_agent("Agent1", "general", ["analysis"])
        agent2 = self.network.add_agent("Agent2", "general", ["analysis"])
        
        # Create task
        task_id = self.coordinator.create_task(
            goal="Analyze data",
            description="Perform data analysis",
            required_skills=["analysis"]
        )
        
        # Assign task
        result = self.coordinator.assign_task(task_id, [agent1.id, agent2.id])
        
        self.assertTrue(result)
        self.assertEqual(self.coordinator.tasks[task_id].assigned_agents, [agent1.id, agent2.id])
        
    def test_start_task(self):
        """Test starting a task."""
        agent = self.network.add_agent("Agent1", "general", ["analysis"])
        task_id = self.coordinator.create_task("Test goal", "Test description", ["analysis"])
        self.coordinator.assign_task(task_id, [agent.id])
        
        result = self.coordinator.start_task(task_id)
        
        self.assertTrue(result)
        self.assertEqual(self.coordinator.tasks[task_id].status, TaskStatus.IN_PROGRESS)
        
    def test_complete_task(self):
        """Test completing a task."""
        agent = self.network.add_agent("Agent1", "general", ["analysis"])
        task_id = self.coordinator.create_task("Test goal", "Test description", ["analysis"])
        self.coordinator.assign_task(task_id, [agent.id])
        self.coordinator.start_task(task_id)
        
        result = self.coordinator.complete_task(task_id, {"result": "success"})
        
        self.assertTrue(result)
        self.assertEqual(self.coordinator.tasks[task_id].status, TaskStatus.COMPLETED)
        
    def test_get_available_agents(self):
        """Test getting available agents."""
        agent1 = self.network.add_agent("Agent1", "general", ["analysis"])
        agent2 = self.network.add_agent("Agent2", "general", ["reasoning"])
        
        # Set agents to ACTIVE status
        agent1.status = AgentStatus.ACTIVE
        agent2.status = AgentStatus.ACTIVE
        
        available = self.coordinator.get_available_agents(["analysis"])
        
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0], agent1.id)
        
    def test_get_coordination_statistics(self):
        """Test getting coordination statistics."""
        agent = self.network.add_agent("Agent1", "general", ["analysis"])
        task_id = self.coordinator.create_task("Test goal", "Test description", ["analysis"])
        self.coordinator.assign_task(task_id, [agent.id])
        self.coordinator.start_task(task_id)
        self.coordinator.complete_task(task_id, {"result": "success"})
        
        stats = self.coordinator.get_coordination_statistics()
        
        self.assertEqual(stats["total_tasks"], 1)
        self.assertEqual(stats["completed_tasks"], 1)
        self.assertEqual(stats["success_rate"], 1.0)
        
    def test_detect_coordination_issues(self):
        """Test detecting coordination issues."""
        agent = self.network.add_agent("Agent1", "general", ["analysis"])
        
        # Overload agent
        for i in range(6):
            task_id = self.coordinator.create_task(f"Task {i}", "Description", ["analysis"])
            self.coordinator.assign_task(task_id, [agent.id])
            
        issues = self.coordinator.detect_coordination_issues()
        
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0]["type"], "agent_overload")


class GoalNegotiatorTests(unittest.TestCase):
    """Test the goal negotiator."""
    
    def setUp(self):
        self.network = AgentNetwork()
        self.negotiator = GoalNegotiator(self.network)
        
    def test_start_negotiation(self):
        """Test starting a negotiation."""
        negotiation_id = self.negotiator.start_negotiation(
            goal="Analyze data",
            initiator_id="agent1",
            target_agents=["agent2", "agent3"]
        )
        
        self.assertIsNotNone(negotiation_id)
        self.assertIn(negotiation_id, self.negotiator.negotiations)
        
    def test_submit_proposal(self):
        """Test submitting a proposal."""
        negotiation_id = self.negotiator.start_negotiation(
            "Analyze data", "agent1", ["agent2", "agent3"]
        )
        
        proposal = {"method": "statistical_analysis", "confidence": 0.8}
        result = self.negotiator.submit_proposal(negotiation_id, "agent2", proposal)
        
        self.assertTrue(result)
        self.assertIn("agent2", self.negotiator.negotiations[negotiation_id]["proposals"])
        
    def test_vote_on_proposal(self):
        """Test voting on a proposal."""
        negotiation_id = self.negotiator.start_negotiation(
            "Analyze data", "agent1", ["agent2", "agent3"]
        )
        
        # Submit proposal
        self.negotiator.submit_proposal(negotiation_id, "agent2", {"method": "analysis"})
        
        # Vote on proposal
        result = self.negotiator.vote_on_proposal(negotiation_id, "agent3", "agent2", True)
        
        self.assertTrue(result)
        self.assertIn("agent3", self.negotiator.negotiations[negotiation_id]["votes"])
        
    def test_get_negotiation_status(self):
        """Test getting negotiation status."""
        negotiation_id = self.negotiator.start_negotiation(
            "Analyze data", "agent1", ["agent2"]
        )
        
        status = self.negotiator.get_negotiation_status(negotiation_id)
        
        self.assertIsNotNone(status)
        self.assertEqual(status["goal"], "Analyze data")
        self.assertEqual(status["initiator_id"], "agent1")
        
    def test_get_active_negotiations(self):
        """Test getting active negotiations."""
        self.negotiator.start_negotiation("Goal1", "agent1", ["agent2"])
        self.negotiator.start_negotiation("Goal2", "agent2", ["agent3"])
        
        active = self.negotiator.get_active_negotiations()
        
        self.assertEqual(len(active), 2)


class ResourceManagerTests(unittest.TestCase):
    """Test the resource manager."""
    
    def setUp(self):
        self.network = AgentNetwork()
        self.resource_manager = ResourceManager(self.network)
        
    def test_allocate_resource(self):
        """Test allocating a resource."""
        # Set up resource pool
        self.resource_manager.resource_pools[ResourceType.COMPUTATIONAL] = 100.0
        
        result = self.resource_manager.allocate_resource(
            "agent1", ResourceType.COMPUTATIONAL, 50.0
        )
        
        self.assertTrue(result)
        self.assertEqual(self.resource_manager.resource_pools[ResourceType.COMPUTATIONAL], 50.0)
        
    def test_deallocate_resource(self):
        """Test deallocating a resource."""
        # Set up resource pool and allocation
        self.resource_manager.resource_pools[ResourceType.COMPUTATIONAL] = 100.0
        self.resource_manager.allocate_resource("agent1", ResourceType.COMPUTATIONAL, 50.0)
        
        result = self.resource_manager.deallocate_resource(
            "agent1", ResourceType.COMPUTATIONAL, 25.0
        )
        
        self.assertTrue(result)
        self.assertEqual(self.resource_manager.resource_pools[ResourceType.COMPUTATIONAL], 75.0)
        
    def test_get_resource_availability(self):
        """Test getting resource availability."""
        self.resource_manager.resource_pools[ResourceType.COMPUTATIONAL] = 75.0
        
        availability = self.resource_manager.get_resource_availability(ResourceType.COMPUTATIONAL)
        
        self.assertEqual(availability, 75.0)
        
    def test_get_agent_allocations(self):
        """Test getting agent allocations."""
        self.resource_manager.resource_pools[ResourceType.COMPUTATIONAL] = 100.0
        self.resource_manager.allocate_resource("agent1", ResourceType.COMPUTATIONAL, 50.0)
        
        allocations = self.resource_manager.get_agent_allocations("agent1")
        
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].resource_type, ResourceType.COMPUTATIONAL)
        self.assertEqual(allocations[0].allocated_amount, 50.0)
        
    def test_get_resource_statistics(self):
        """Test getting resource statistics."""
        self.resource_manager.resource_pools[ResourceType.COMPUTATIONAL] = 100.0
        self.resource_manager.allocate_resource("agent1", ResourceType.COMPUTATIONAL, 50.0)
        
        stats = self.resource_manager.get_resource_statistics()
        
        self.assertEqual(stats["total_allocations"], 1)
        self.assertIn("resource_usage", stats)
        self.assertIn("agent_allocations", stats)


class NetworkLearningTests(unittest.TestCase):
    """Test the network learning system."""
    
    def setUp(self):
        self.network = AgentNetwork()
        self.learning = NetworkLearning(self.network)
        
    def test_initiate_learning(self):
        """Test initiating a learning process."""
        learning_id = self.learning.initiate_learning(
            LearningType.SKILL_ACQUISITION,
            "agent1",
            ["agent2", "agent3"],
            {"skill": "analysis", "description": "Data analysis skill"}
        )
        
        self.assertIsNotNone(learning_id)
        self.assertIn(learning_id, self.learning.learning_events)
        
    def test_propagate_knowledge(self):
        """Test propagating knowledge."""
        learning_id = self.learning.initiate_learning(
            LearningType.KNOWLEDGE_PROPAGATION,
            "agent1",
            ["agent2"],
            {"knowledge": "test knowledge"}
        )
        
        result = self.learning.propagate_knowledge(
            learning_id, "agent2", {"validated": True, "confidence": 0.8}
        )
        
        self.assertTrue(result)
        
    def test_share_skill(self):
        """Test sharing a skill."""
        # Add agents to network
        agent1 = self.network.add_agent("Agent1", "general", ["analysis"])
        agent2 = self.network.add_agent("Agent2", "general", [])
        
        result = self.learning.share_skill(
            agent1.id, agent2.id, "analysis", {"handler": "analyze_data"}
        )
        
        self.assertTrue(result)
        self.assertIn("analysis", agent2.skills)
        
    def test_discover_pattern(self):
        """Test discovering a pattern."""
        agent = self.network.add_agent("Agent1", "general", ["analysis"])
        
        pattern_id = self.learning.discover_pattern(
            agent.id, "collaboration", "Agents work together effectively", 5, 0.8
        )
        
        self.assertIsNotNone(pattern_id)
        self.assertIn(pattern_id, self.learning.knowledge_patterns)
        
    def test_validate_learning(self):
        """Test validating learning."""
        learning_id = self.learning.initiate_learning(
            LearningType.SKILL_ACQUISITION,
            "agent1",
            ["agent2"],
            {"skill": "analysis", "description": "Data analysis skill"}
        )
        
        validation = self.learning.validate_learning(
            learning_id, "agent2", {"min_confidence": 0.5, "min_description_length": 10}
        )
        
        self.assertIn("valid", validation)
        self.assertIn("score", validation)
        
    def test_get_learning_statistics(self):
        """Test getting learning statistics."""
        # Create some learning events
        self.learning.initiate_learning(LearningType.SKILL_ACQUISITION, "agent1", ["agent2"], {"skill": "test"})
        self.learning.initiate_learning(LearningType.KNOWLEDGE_PROPAGATION, "agent2", ["agent3"], {"knowledge": "test"})
        
        stats = self.learning.get_learning_statistics()
        
        self.assertEqual(stats["total_learning_events"], 2)
        self.assertIn("success_rate", stats)
        self.assertIn("average_confidence", stats)


class NetworkMonitorTests(unittest.TestCase):
    """Test the network monitor."""
    
    def setUp(self):
        self.network = AgentNetwork()
        self.monitor = NetworkMonitor(self.network)
        
    def test_add_monitoring_rule(self):
        """Test adding a monitoring rule."""
        result = self.monitor.add_monitoring_rule(
            "rule1", "cpu_usage", {"threshold": 80}, AlertLevel.WARNING
        )
        
        self.assertTrue(result)
        self.assertIn("rule1", self.monitor.monitoring_rules)
        
    def test_check_network_health(self):
        """Test checking network health."""
        # Add an agent with high CPU usage
        agent = self.network.add_agent("TestAgent", "general", ["analysis"])
        agent.performance_metrics = {"cpu_usage": 95}
        
        alerts = self.monitor.check_network_health()
        
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].alert_type, "overloaded_agent")
        
    def test_capture_performance_metrics(self):
        """Test capturing performance metrics."""
        self.network.add_agent("Agent1", "general", ["analysis"])
        
        metrics = self.monitor.capture_performance_metrics()
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_agents, 1)
        self.assertGreaterEqual(metrics.network_efficiency, 0)
        
    def test_get_performance_trends(self):
        """Test getting performance trends."""
        # Capture some metrics
        for _ in range(5):
            self.monitor.capture_performance_metrics()
            time.sleep(0.01)
            
        trends = self.monitor.get_performance_trends(hours=1)
        
        self.assertIn("data_points", trends)
        self.assertIn("network_efficiency_trend", trends)
        
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Create some alerts
        self.monitor.check_network_health()
        
        active_alerts = self.monitor.get_active_alerts()
        
        self.assertIsInstance(active_alerts, list)
        
    def test_resolve_alert(self):
        """Test resolving an alert."""
        # Create an alert
        alerts = self.monitor.check_network_health()
        if alerts:
            alert_id = alerts[0].id
            
            result = self.monitor.resolve_alert(alert_id, {"resolved_by": "test"})
            
            self.assertTrue(result)
            self.assertTrue(self.monitor.alerts[alert_id].resolved)
            
    def test_get_monitoring_statistics(self):
        """Test getting monitoring statistics."""
        # Create some alerts
        self.monitor.check_network_health()
        
        stats = self.monitor.get_monitoring_statistics()
        
        self.assertIn("total_alerts", stats)
        self.assertIn("resolution_rate", stats)
        self.assertIn("alert_distribution", stats)


class EmergentBehaviorTests(unittest.TestCase):
    """Test the emergent behavior system."""
    
    def setUp(self):
        self.network = AgentNetwork()
        self.emergent = EmergentBehavior(self.network)
        
    def test_detect_emergent_patterns(self):
        """Test detecting emergent patterns."""
        # Add agents with communication history
        agent1 = self.network.add_agent("Agent1", "general", ["analysis"])
        agent2 = self.network.add_agent("Agent2", "general", ["reasoning"])
        
        # Add communication history
        agent1.communication_history = [
            {"target_agent": agent2.id, "timestamp": time.time()},
            {"target_agent": agent2.id, "timestamp": time.time()}
        ]
        agent2.communication_history = [
            {"target_agent": agent1.id, "timestamp": time.time()}
        ]
        
        patterns = self.emergent.detect_emergent_patterns()
        
        self.assertIsInstance(patterns, list)
        
    def test_record_collaboration_event(self):
        """Test recording a collaboration event."""
        event_id = self.emergent.record_collaboration_event(
            "task_collaboration",
            ["agent1", "agent2"],
            "Collaborative task execution",
            120.0,
            True,
            {"result": "success"}
        )
        
        self.assertIsNotNone(event_id)
        self.assertEqual(len(self.emergent.collaboration_events), 1)
        
    def test_update_pattern_status(self):
        """Test updating pattern status."""
        # Create a pattern
        pattern = self.emergent.patterns["test_pattern"] = type('Pattern', (), {
            'id': 'test_pattern',
            'status': PatternStatus.EMERGING,
            'last_observed': time.time(),
            'evolution_history': []
        })()
        
        result = self.emergent.update_pattern_status("test_pattern", PatternStatus.STABLE)
        
        self.assertTrue(result)
        self.assertEqual(pattern.status, PatternStatus.STABLE)
        
    def test_get_emergent_statistics(self):
        """Test getting emergent statistics."""
        # Record some collaboration events
        self.emergent.record_collaboration_event("test", ["agent1"], "test", 60.0, True)
        
        stats = self.emergent.get_emergent_statistics()
        
        self.assertIn("total_patterns", stats)
        self.assertIn("total_collaborations", stats)
        self.assertIn("collaboration_success_rate", stats)


class IntegrationTests(unittest.TestCase):
    """Integration tests for the network system."""
    
    def setUp(self):
        self.network = AgentNetwork()
        self.message_bus = MessageBus("test_network")
        self.coordinator = TaskCoordinator(self.network)
        self.negotiator = GoalNegotiator(self.network)
        self.resource_manager = ResourceManager(self.network)
        self.learning = NetworkLearning(self.network)
        self.monitor = NetworkMonitor(self.network)
        self.emergent = EmergentBehavior(self.network)
        
    def test_full_network_workflow(self):
        """Test a complete network workflow."""
        # Add agents
        agent1 = self.network.add_agent("Agent1", "analyst", ["analysis", "data_processing"])
        agent2 = self.network.add_agent("Agent2", "researcher", ["research", "reasoning"])
        
        # Connect agents
        self.network.connect_agents(agent1.id, agent2.id)
        
        # Send message
        message_id = self.message_bus.send_message(
            agent1.id, agent2.id, MessageType.QUERY,
            {"question": "Can you analyze this data?"}
        )
        
        # Create and assign task
        task_id = self.coordinator.create_task(
            "Analyze dataset", "Perform comprehensive data analysis",
            ["analysis", "data_processing"], priority=1
        )
        self.coordinator.assign_task(task_id, [agent1.id])
        
        # Start negotiation
        negotiation_id = self.negotiator.start_negotiation(
            "Collaborative analysis", agent1.id, [agent2.id]
        )
        
        # Allocate resources
        self.resource_manager.resource_pools[ResourceType.COMPUTATIONAL] = 100.0
        self.resource_manager.allocate_resource(agent1.id, ResourceType.COMPUTATIONAL, 50.0)
        
        # Initiate learning
        learning_id = self.learning.initiate_learning(
            LearningType.SKILL_ACQUISITION, agent1.id, [agent2.id],
            {"skill": "advanced_analysis", "description": "Advanced data analysis techniques"}
        )
        
        # Record collaboration
        self.emergent.record_collaboration_event(
            "data_analysis", [agent1.id, agent2.id],
            "Collaborative data analysis project", 300.0, True
        )
        
        # Check network health
        alerts = self.monitor.check_network_health()
        
        # Verify all components are working
        self.assertEqual(len(self.network.agents), 2)
        self.assertEqual(len(self.message_bus.messages), 1)
        self.assertEqual(len(self.coordinator.tasks), 1)
        self.assertEqual(len(self.negotiator.negotiations), 1)
        self.assertEqual(len(self.resource_manager.allocations[agent1.id]), 1)
        self.assertEqual(len(self.learning.learning_events), 1)
        self.assertEqual(len(self.emergent.collaboration_events), 1)
        
    def test_network_scalability(self):
        """Test network scalability."""
        # Add multiple agents
        agents = []
        for i in range(10):
            agent = self.network.add_agent(f"Agent{i}", "general", [f"skill{i}"])
            agents.append(agent)
            
        # Connect agents in a mesh topology
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                self.network.connect_agents(agents[i].id, agents[j].id)
                
        # Create multiple tasks
        task_ids = []
        for i in range(5):
            task_id = self.coordinator.create_task(
                f"Task {i}", f"Description {i}", [f"skill{i % 10}"]
            )
            task_ids.append(task_id)
            
        # Assign tasks to agents
        for i, task_id in enumerate(task_ids):
            agent_ids = [agents[i % len(agents)].id, agents[(i + 1) % len(agents)].id]
            self.coordinator.assign_task(task_id, agent_ids)
            
        # Verify network can handle the load
        stats = self.network.get_network_statistics()
        self.assertEqual(stats["total_agents"], 10)
        
        # Check topology for connection count
        topology = self.network.get_network_topology()
        self.assertGreater(topology["connection_count"], 0)
        
        coord_stats = self.coordinator.get_coordination_statistics()
        self.assertEqual(coord_stats["total_tasks"], 5)
        
    def test_network_resilience(self):
        """Test network resilience."""
        # Add agents
        agent1 = self.network.add_agent("Agent1", "general", ["analysis"])
        agent2 = self.network.add_agent("Agent2", "general", ["reasoning"])
        agent3 = self.network.add_agent("Agent3", "general", ["planning"])
        
        # Set agents to ACTIVE status
        agent1.status = AgentStatus.ACTIVE
        agent2.status = AgentStatus.ACTIVE
        agent3.status = AgentStatus.ACTIVE
        
        # Connect agents
        self.network.connect_agents(agent1.id, agent2.id)
        self.network.connect_agents(agent2.id, agent3.id)
        
        # Create task
        task_id = self.coordinator.create_task("Test task", "Description", ["analysis"])
        self.coordinator.assign_task(task_id, [agent1.id, agent2.id])
        
        # Remove an agent (simulate failure)
        self.network.remove_agent(agent2.id)
        
        # Check if network can adapt
        available_agents = self.coordinator.get_available_agents(["analysis"])
        self.assertEqual(len(available_agents), 1)
        self.assertEqual(available_agents[0], agent1.id)
        
        # Check for bottlenecks
        bottlenecks = self.network.detect_network_bottlenecks()
        self.assertIsInstance(bottlenecks, list)


if __name__ == '__main__':
    unittest.main()
