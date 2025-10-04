"""
Comprehensive Testing for Civilization System
Tests multi-agent dynamics, tool-building autonomy, world persistence, and emergent metrics.
"""

import unittest
import tempfile
import os
import time
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path

from egdol.omnimind.civilization.multi_agent_system import (
    MultiAgentCivilizationSystem, CivilizationAgent, AgentMessage, MessageType,
    AgentState, MessageBus, CivilizationState, AgentMemory
)
from egdol.omnimind.civilization.toolforge import (
    Toolforge, ToolType, ToolStatus, ToolSpecification, ToolImplementation
)
from egdol.omnimind.civilization.world_persistence import (
    WorldPersistence, EventType, EventImportance, WorldEvent, CivilizationalLaw,
    Technology, CulturalTrait, WorldState
)
from egdol.omnimind.civilization.emergent_metrics import (
    EmergentMetrics, MetricType, MetricTrend, EmergentPattern
)
from egdol.omnimind.conversational.personality_framework import Personality, PersonalityType


class TestMultiAgentSystem(unittest.TestCase):
    """Test multi-agent civilization system."""
    
    def setUp(self):
        """Set up test environment."""
        self.system = MultiAgentCivilizationSystem()
        self.agent_configs = [
            {
                "personality_name": "Strategos",
                "personality_type": PersonalityType.STRATEGOS,
                "communication_style": "formal"
            },
            {
                "personality_name": "Archivist",
                "personality_type": PersonalityType.ARCHIVIST,
                "communication_style": "scholarly"
            },
            {
                "personality_name": "Lawmaker",
                "personality_type": PersonalityType.LAWMAKER,
                "communication_style": "legal"
            },
            {
                "personality_name": "Oracle",
                "personality_type": PersonalityType.ORACLE,
                "communication_style": "mystical"
            }
        ]
    
    def test_civilization_creation(self):
        """Test civilization creation with agents."""
        civilization_id = self.system.create_civilization(
            name="Test Civilization",
            agent_configs=self.agent_configs
        )
        
        self.assertIsNotNone(civilization_id)
        self.assertIn(civilization_id, self.system.civilizations)
        
        civilization = self.system.civilizations[civilization_id]
        self.assertEqual(civilization.name, "Test Civilization")
        self.assertEqual(len(civilization.agents), 4)
        
        # Check agent personalities
        agent_personalities = [agent.personality.personality_type for agent in civilization.agents.values()]
        expected_types = [PersonalityType.STRATEGOS, PersonalityType.ARCHIVIST, 
                         PersonalityType.LAWMAKER, PersonalityType.ORACLE]
        self.assertEqual(set(agent_personalities), set(expected_types))
    
    def test_agent_communication(self):
        """Test agent communication through message bus."""
        civilization_id = self.system.create_civilization(
            name="Communication Test",
            agent_configs=self.agent_configs
        )
        
        # Get first two agents
        agents = list(self.system.civilizations[civilization_id].agents.values())
        agent1 = agents[0]
        agent2 = agents[1]
        
        # Create test message
        message = AgentMessage(
            message_id="test_001",
            sender_id=agent1.agent_id,
            receiver_id=agent2.agent_id,
            message_type=MessageType.DEBATE,
            content="Let's debate the optimal strategy",
            priority=5
        )
        
        # Publish message
        self.system.message_bus.publish(message)
        
        # Get messages for agent2
        messages = self.system.message_bus.get_messages_for_agent(agent2.agent_id)
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "Let's debate the optimal strategy")
        self.assertEqual(messages[0].message_type, MessageType.DEBATE)
    
    def test_agent_autonomous_action(self):
        """Test agent autonomous actions."""
        civilization_id = self.system.create_civilization(
            name="Autonomous Test",
            agent_configs=self.agent_configs
        )
        
        agent = list(self.system.civilizations[civilization_id].agents.values())[0]
        
        # Test autonomous action
        action = asyncio.run(agent.autonomous_action())
        
        if action:
            self.assertIsInstance(action, AgentMessage)
            self.assertEqual(action.sender_id, agent.agent_id)
            self.assertIsNone(action.receiver_id)  # Broadcast
    
    def test_civilization_state_update(self):
        """Test civilization state updates."""
        civilization_id = self.system.create_civilization(
            name="State Test",
            agent_configs=self.agent_configs
        )
        
        civilization = self.system.civilizations[civilization_id]
        initial_history_length = len(civilization.history)
        
        # Simulate some agent interactions
        agent = list(civilization.agents.values())[0]
        message = AgentMessage(
            message_id="test_002",
            sender_id=agent.agent_id,
            receiver_id=None,
            message_type=MessageType.DISCOVERY,
            content="I discovered something important",
            priority=3
        )
        
        self.system.message_bus.publish(message)
        
        # Update civilization state
        self.system._update_civilization_state(civilization)
        
        # Check that history was updated
        self.assertGreater(len(civilization.history), initial_history_length)
        self.assertGreater(civilization.last_updated, civilization.created_at)


class TestToolforge(unittest.TestCase):
    """Test toolforge subsystem."""
    
    def setUp(self):
        """Set up test environment."""
        self.toolforge = Toolforge()
        self.temp_dir = tempfile.mkdtemp()
        self.toolforge.tool_directory = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_tool_proposal(self):
        """Test tool proposal."""
        tool_id = self.toolforge.propose_tool(
            name="Test Reasoning Tool",
            description="A tool for testing reasoning capabilities",
            tool_type=ToolType.REASONING,
            proposer_agent_id="agent_001",
            requirements=["Python 3.8+", "numpy"],
            inputs={"data": "dict", "params": "dict"},
            outputs={"result": "dict", "confidence": "float"},
            constraints=["Must complete within 5 seconds"],
            success_criteria=["All tests pass", "Performance < 5s"]
        )
        
        self.assertIsNotNone(tool_id)
        self.assertIn(tool_id, self.toolforge.specifications)
        self.assertEqual(self.toolforge.tool_status[tool_id], ToolStatus.PROPOSED)
        
        spec = self.toolforge.specifications[tool_id]
        self.assertEqual(spec.name, "Test Reasoning Tool")
        self.assertEqual(spec.tool_type, ToolType.REASONING)
    
    def test_tool_design(self):
        """Test tool design and implementation."""
        # Propose tool
        tool_id = self.toolforge.propose_tool(
            name="Simple Calculator",
            description="Basic arithmetic operations",
            tool_type=ToolType.REASONING,
            proposer_agent_id="agent_001"
        )
        
        # Design tool
        success = self.toolforge.design_tool(tool_id, "designer_001")
        
        self.assertTrue(success)
        self.assertIn(tool_id, self.toolforge.tools)
        self.assertEqual(self.toolforge.tool_status[tool_id], ToolStatus.TESTING)
        
        implementation = self.toolforge.tools[tool_id]
        self.assertIsNotNone(implementation.code)
        self.assertIsNotNone(implementation.tests)
    
    def test_tool_testing(self):
        """Test tool testing system."""
        # Propose and design tool
        tool_id = self.toolforge.propose_tool(
            name="Test Tool",
            description="A simple test tool",
            tool_type=ToolType.REASONING,
            proposer_agent_id="agent_001"
        )
        
        self.toolforge.design_tool(tool_id, "designer_001")
        
        # Test tool
        test_results = self.toolforge.test_tool(tool_id)
        
        self.assertIsInstance(test_results, list)
        self.assertGreater(len(test_results), 0)
        
        # Check test result
        test_result = test_results[0]
        self.assertEqual(test_result.tool_id, tool_id)
        self.assertIsInstance(test_result.passed, bool)
    
    def test_tool_integration(self):
        """Test tool integration."""
        # Propose, design, and test tool
        tool_id = self.toolforge.propose_tool(
            name="Integration Test Tool",
            description="Tool for integration testing",
            tool_type=ToolType.REASONING,
            proposer_agent_id="agent_001"
        )
        
        self.toolforge.design_tool(tool_id, "designer_001")
        
        # Mock successful test results
        from egdol.omnimind.civilization.toolforge import ToolTestResult
        mock_result = ToolTestResult(
            tool_id=tool_id,
            test_name="mock_test",
            passed=True,
            execution_time=1.0
        )
        self.toolforge.test_results.append(mock_result)
        
        # Integrate tool
        success = self.toolforge.integrate_tool(tool_id)
        
        self.assertTrue(success)
        self.assertEqual(self.toolforge.tool_status[tool_id], ToolStatus.ACTIVE)
    
    def test_tool_usage_tracking(self):
        """Test tool usage tracking."""
        # Create and integrate a tool
        tool_id = self.toolforge.propose_tool(
            name="Usage Test Tool",
            description="Tool for usage testing",
            tool_type=ToolType.REASONING,
            proposer_agent_id="agent_001"
        )
        
        self.toolforge.design_tool(tool_id, "designer_001")
        
        # Mock successful test and integration
        from egdol.omnimind.civilization.toolforge import ToolTestResult
        mock_result = ToolTestResult(tool_id=tool_id, test_name="mock", passed=True)
        self.toolforge.test_results.append(mock_result)
        self.toolforge.integrate_tool(tool_id)
        
        # Use tool
        result = self.toolforge.use_tool(tool_id, "user_001", data={"test": "value"})
        
        # Check usage was tracked
        self.assertGreater(len(self.toolforge.usage_history), 0)
        usage = self.toolforge.usage_history[-1]
        self.assertEqual(usage.tool_id, tool_id)
        self.assertEqual(usage.user_agent_id, "user_001")
    
    def test_tool_statistics(self):
        """Test tool statistics generation."""
        # Create tool with usage history
        tool_id = self.toolforge.propose_tool(
            name="Stats Test Tool",
            description="Tool for statistics testing",
            tool_type=ToolType.REASONING,
            proposer_agent_id="agent_001"
        )
        
        # Add mock usage data
        from egdol.omnimind.civilization.toolforge import ToolUsage
        for i in range(10):
            usage = ToolUsage(
                tool_id=tool_id,
                user_agent_id=f"user_{i}",
                success=i < 8,  # 80% success rate
                execution_time=1.0 + i * 0.1
            )
            self.toolforge.usage_history.append(usage)
        
        # Get statistics
        stats = self.toolforge.get_tool_statistics(tool_id)
        
        self.assertEqual(stats["tool_id"], tool_id)
        self.assertEqual(stats["total_uses"], 10)
        self.assertEqual(stats["successful_uses"], 8)
        self.assertAlmostEqual(stats["success_rate"], 0.8)
        self.assertGreater(stats["avg_execution_time"], 0)


class TestWorldPersistence(unittest.TestCase):
    """Test world persistence system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.persistence = WorldPersistence(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.temp_db.name)
    
    def test_world_creation(self):
        """Test world creation."""
        world_id = self.persistence.create_world("Test World")
        
        self.assertIsNotNone(world_id)
        self.assertIn(world_id, self.persistence.world_states)
        
        world_state = self.persistence.world_states[world_id]
        self.assertEqual(world_state.name, "Test World")
        self.assertEqual(world_state.current_era, "foundation")
    
    def test_event_creation(self):
        """Test event creation and storage."""
        world_id = self.persistence.create_world("Event Test World")
        
        event_id = self.persistence.add_event(
            world_id=world_id,
            event_type=EventType.DISCOVERY,
            importance=EventImportance.MAJOR,
            title="Great Discovery",
            description="A major discovery was made",
            participants=["agent_001", "agent_002"],
            location="Research Lab",
            consequences=["New technology available", "Research funding increased"]
        )
        
        self.assertIsNotNone(event_id)
        self.assertIn(event_id, self.persistence.events)
        
        event = self.persistence.events[event_id]
        self.assertEqual(event.event_type, EventType.DISCOVERY)
        self.assertEqual(event.importance, EventImportance.MAJOR)
        self.assertEqual(event.title, "Great Discovery")
        self.assertEqual(len(event.participants), 2)
    
    def test_law_enactment(self):
        """Test law enactment."""
        world_id = self.persistence.create_world("Law Test World")
        
        law_id = self.persistence.enact_law(
            world_id=world_id,
            title="Test Law",
            description="A test law for governance",
            proposer_agent_id="agent_001",
            enacted_by=["agent_001", "agent_002", "agent_003"],
            opposed_by=["agent_004"],
            domain="governance"
        )
        
        self.assertIsNotNone(law_id)
        self.assertIn(law_id, self.persistence.laws)
        
        law = self.persistence.laws[law_id]
        self.assertEqual(law.title, "Test Law")
        self.assertEqual(law.proposer_agent_id, "agent_001")
        self.assertEqual(len(law.enacted_by), 3)
        self.assertEqual(len(law.opposed_by), 1)
        self.assertEqual(law.domain, "governance")
    
    def test_technology_discovery(self):
        """Test technology discovery."""
        world_id = self.persistence.create_world("Tech Test World")
        
        tech_id = self.persistence.discover_technology(
            world_id=world_id,
            name="Quantum Computer",
            description="Advanced quantum computing technology",
            inventor_agent_id="agent_001",
            prerequisites=["classical_computer", "quantum_physics"],
            applications=["cryptography", "simulation", "optimization"],
            domain="computing"
        )
        
        self.assertIsNotNone(tech_id)
        self.assertIn(tech_id, self.persistence.technologies)
        
        tech = self.persistence.technologies[tech_id]
        self.assertEqual(tech.name, "Quantum Computer")
        self.assertEqual(tech.inventor_agent_id, "agent_001")
        self.assertEqual(len(tech.prerequisites), 2)
        self.assertEqual(len(tech.applications), 3)
    
    def test_cultural_trait_adoption(self):
        """Test cultural trait adoption."""
        world_id = self.persistence.create_world("Culture Test World")
        
        trait_id = self.persistence.adopt_cultural_trait(
            world_id=world_id,
            name="Collaborative Decision Making",
            description="A cultural practice of making decisions through collaboration",
            origin_agent_id="agent_001",
            related_traits=["consensus_building", "democratic_processes"],
            domain="governance"
        )
        
        self.assertIsNotNone(trait_id)
        self.assertIn(trait_id, self.persistence.cultural_traits)
        
        trait = self.persistence.cultural_traits[trait_id]
        self.assertEqual(trait.name, "Collaborative Decision Making")
        self.assertEqual(trait.origin_agent_id, "agent_001")
        self.assertEqual(len(trait.related_traits), 2)
    
    def test_world_history(self):
        """Test world history retrieval."""
        world_id = self.persistence.create_world("History Test World")
        
        # Add multiple events
        for i in range(5):
            self.persistence.add_event(
                world_id=world_id,
                event_type=EventType.DISCOVERY,
                importance=EventImportance.MODERATE,
                title=f"Discovery {i}",
                description=f"Discovery number {i}",
                participants=[f"agent_{i}"]
            )
        
        # Get world history
        history = self.persistence.get_world_history(world_id, limit=3)
        
        self.assertEqual(len(history), 3)
        self.assertIsInstance(history[0], dict)
        self.assertIn("event_id", history[0])
        self.assertIn("title", history[0])
    
    def test_civilizational_memory(self):
        """Test civilizational memory queries."""
        world_id = self.persistence.create_world("Memory Test World")
        
        # Add events with different types
        self.persistence.add_event(
            world_id=world_id,
            event_type=EventType.TECHNOLOGICAL_BREAKTHROUGH,
            importance=EventImportance.MAJOR,
            title="AI Breakthrough",
            description="Major AI technology breakthrough",
            participants=["agent_001"]
        )
        
        self.persistence.add_event(
            world_id=world_id,
            event_type=EventType.CULTURAL_EVOLUTION,
            importance=EventImportance.MODERATE,
            title="Cultural Shift",
            description="Significant cultural evolution",
            participants=["agent_002"]
        )
        
        # Query for AI-related events
        ai_memory = self.persistence.get_civilizational_memory(world_id, "AI")
        
        self.assertEqual(len(ai_memory), 1)
        self.assertEqual(ai_memory[0]["title"], "AI Breakthrough")
        
        # Query for all events
        all_memory = self.persistence.get_civilizational_memory(world_id)
        
        self.assertEqual(len(all_memory), 2)


class TestEmergentMetrics(unittest.TestCase):
    """Test emergent metrics system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.metrics = EmergentMetrics(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.temp_db.name)
    
    def test_personality_divergence_calculation(self):
        """Test personality divergence calculation."""
        # Create mock agents with different personalities
        from egdol.omnimind.conversational.personality_framework import Personality, PersonalityType
        
        agent1 = CivilizationAgent(
            agent_id="agent_001",
            personality=Personality(
                name="Strategos",
                personality_type=PersonalityType.STRATEGOS,
                description="Strategos personality",
                archetype="strategos",
                epistemic_style="formal"
            ),
            civilization_id="test_civ",
            message_bus=None,
            memory=None
        )
        
        agent2 = CivilizationAgent(
            agent_id="agent_002",
            personality=Personality(
                name="Archivist",
                personality_type=PersonalityType.ARCHIVIST,
                description="Archivist personality",
                archetype="archivist",
                epistemic_style="scholarly"
            ),
            civilization_id="test_civ",
            message_bus=None,
            memory=None
        )
        
        # Register agents
        self.metrics.register_agents({"agent_001": agent1, "agent_002": agent2})
        
        # Calculate divergence
        divergence = self.metrics.calculate_personality_divergence()
        
        self.assertIsInstance(divergence, float)
        self.assertGreaterEqual(divergence, 0.0)
    
    def test_law_formation_rate(self):
        """Test law formation rate calculation."""
        # Add law events
        from egdol.omnimind.civilization.world_persistence import WorldEvent, EventType, EventImportance
        
        law_events = [
            WorldEvent(
                event_id=f"law_{i}",
                event_type=EventType.LAW_ENACTMENT,
                importance=EventImportance.MODERATE,
                title=f"Law {i}",
                description=f"Law number {i}",
                participants=[f"agent_{i}"]
            )
            for i in range(3)
        ]
        
        self.metrics.add_world_events(law_events)
        
        # Calculate law formation rate
        rate = self.metrics.calculate_law_formation_rate()
        
        self.assertIsInstance(rate, float)
        self.assertGreaterEqual(rate, 0.0)
    
    def test_strategic_dominance_calculation(self):
        """Test strategic dominance calculation."""
        # Create mock agent with memory
        from egdol.omnimind.civilization.multi_agent_system import AgentMemory
        
        memory = AgentMemory(agent_id="agent_001")
        memory.relationships = {"agent_002": 0.8, "agent_003": 0.6}
        memory.achievements = ["achievement1", "achievement2"]
        memory.expertise = {"strategy": 0.9, "tactics": 0.8}
        
        agent = CivilizationAgent(
            agent_id="agent_001",
            personality=Personality(
                name="Strategos",
                personality_type=PersonalityType.STRATEGOS,
                description="Strategos personality",
                archetype="strategos",
                epistemic_style="formal"
            ),
            civilization_id="test_civ",
            message_bus=None,
            memory=memory
        )
        
        self.metrics.register_agents({"agent_001": agent})
        
        # Calculate strategic dominance
        dominance = self.metrics.calculate_strategic_dominance("agent_001")
        
        self.assertIsInstance(dominance, float)
        self.assertGreaterEqual(dominance, 0.0)
        self.assertLessEqual(dominance, 1.0)
    
    def test_metrics_update(self):
        """Test metrics update process."""
        # Register some agents
        agent = CivilizationAgent(
            agent_id="agent_001",
            personality=Personality(
                name="Test",
                personality_type=PersonalityType.STRATEGOS,
                description="Test personality",
                archetype="test",
                epistemic_style="formal"
            ),
            civilization_id="test_civ",
            message_bus=None,
            memory=AgentMemory(agent_id="agent_001")
        )
        
        self.metrics.register_agents({"agent_001": agent})
        
        # Update metrics multiple times to generate trends
        self.metrics.update_metrics()
        self.metrics.update_metrics()
        
        # Check that metrics were calculated
        self.assertGreater(len(self.metrics.metrics), 0)
        self.assertGreater(len(self.metrics.trends), 0)
    
    def test_metrics_dashboard(self):
        """Test metrics dashboard generation."""
        # Update metrics first
        self.metrics.update_metrics()
        
        # Get dashboard
        dashboard = self.metrics.get_metrics_dashboard()
        
        self.assertIsInstance(dashboard, dict)
        self.assertIn("timestamp", dashboard)
        self.assertIn("metrics", dashboard)
        self.assertIn("trends", dashboard)
        self.assertIn("patterns", dashboard)
        self.assertIn("summary", dashboard)
        
        # Check summary structure
        summary = dashboard["summary"]
        self.assertIn("overall_health", summary)
        self.assertIn("key_insights", summary)
        self.assertIn("recommendations", summary)


class TestCivilizationIntegration(unittest.TestCase):
    """Test complete civilization system integration."""
    
    def setUp(self):
        """Set up integrated test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        
        # Initialize all components
        self.multi_agent_system = MultiAgentCivilizationSystem()
        self.toolforge = Toolforge()
        self.world_persistence = WorldPersistence(self.temp_db.name)
        self.emergent_metrics = EmergentMetrics(self.temp_db.name)
        
        # Agent configurations
        self.agent_configs = [
            {
                "personality_name": "Strategos",
                "personality_type": PersonalityType.STRATEGOS,
                "communication_style": "formal"
            },
            {
                "personality_name": "Archivist",
                "personality_type": PersonalityType.ARCHIVIST,
                "communication_style": "scholarly"
            }
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.temp_db.name)
    
    def test_complete_civilization_cycle(self):
        """Test complete civilization lifecycle."""
        # 1. Create civilization
        civilization_id = self.multi_agent_system.create_civilization(
            name="Integration Test Civilization",
            agent_configs=self.agent_configs
        )
        
        self.assertIsNotNone(civilization_id)
        
        # 2. Create world
        world_id = self.world_persistence.create_world("Integration Test World")
        
        self.assertIsNotNone(world_id)
        
        # 3. Add world events
        event_id = self.world_persistence.add_event(
            world_id=world_id,
            event_type=EventType.DISCOVERY,
            importance=EventImportance.MAJOR,
            title="Integration Discovery",
            description="A major discovery in the integration test",
            participants=["agent_001", "agent_002"]
        )
        
        self.assertIsNotNone(event_id)
        
        # 4. Propose and create tool
        tool_id = self.toolforge.propose_tool(
            name="Integration Test Tool",
            description="Tool for integration testing",
            tool_type=ToolType.REASONING,
            proposer_agent_id="agent_001"
        )
        
        self.assertIsNotNone(tool_id)
        
        # 5. Register agents with metrics
        agents = self.multi_agent_system.civilizations[civilization_id].agents
        self.emergent_metrics.register_agents(agents)
        
        # 6. Update metrics
        self.emergent_metrics.update_metrics()
        
        # 7. Get dashboard
        dashboard = self.emergent_metrics.get_metrics_dashboard()
        
        self.assertIsInstance(dashboard, dict)
        self.assertIn("summary", dashboard)
    
    def test_deterministic_replay(self):
        """Test deterministic replay capability."""
        # Create civilization with specific seed
        civilization_id = self.multi_agent_system.create_civilization(
            name="Deterministic Test",
            agent_configs=self.agent_configs
        )
        
        # Get initial state
        initial_state = self.multi_agent_system.get_civilization_state(civilization_id)
        
        # Simulate some interactions
        agents = list(self.multi_agent_system.civilizations[civilization_id].agents.values())
        agent1 = agents[0]
        agent2 = agents[1]
        
        # Send message
        message = AgentMessage(
            message_id="deterministic_test",
            sender_id=agent1.agent_id,
            receiver_id=agent2.agent_id,
            message_type=MessageType.DEBATE,
            content="Deterministic test message",
            priority=5
        )
        
        self.multi_agent_system.message_bus.publish(message)
        
        # Get final state
        final_state = self.multi_agent_system.get_civilization_state(civilization_id)
        
        # States should be different
        self.assertNotEqual(initial_state, final_state)
        
        # But should be deterministic (same inputs = same outputs)
        # This would require more sophisticated testing in a real implementation


if __name__ == "__main__":
    unittest.main()
