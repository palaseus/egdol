"""
Tests for Enhanced Egdol Features.
Tests persistent memory, autonomous behaviors, and multi-agent support.
"""

import unittest
import tempfile
import os
import time
from egdol.memory import MemoryStore, MemoryItem
from egdol.meta import MemoryInspector, RuleInspector, RuleScorer, ConfidenceTracker
from egdol.agents import AgentManager, AgentProfile, Agent
from egdol.autonomous import BehaviorScheduler, WatcherManager, ActionManager
from egdol.autonomous.actions import ActionType


class MemoryStoreTests(unittest.TestCase):
    """Test the persistent memory store."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory_store = MemoryStore(os.path.join(self.temp_dir, "test_memory.db"))
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_store_and_retrieve(self):
        """Test storing and retrieving memories."""
        memory_id = self.memory_store.store(
            content="Alice is a human",
            item_type="fact",
            source="user",
            confidence=0.9
        )
        
        self.assertIsNotNone(memory_id)
        
        memory = self.memory_store.retrieve(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, "Alice is a human")
        self.assertEqual(memory.item_type, "fact")
        self.assertEqual(memory.source, "user")
        self.assertEqual(memory.confidence, 0.9)
        
    def test_search_memories(self):
        """Test searching memories."""
        # Store some memories
        self.memory_store.store("Alice is human", "fact", "user", 0.9)
        self.memory_store.store("Bob is human", "fact", "user", 0.8)
        self.memory_store.store("If X is human then X is mortal", "rule", "user", 0.7)
        
        # Search by type
        facts = self.memory_store.search(item_type="fact")
        self.assertEqual(len(facts), 2)
        
        # Search by source
        user_memories = self.memory_store.search(source="user")
        self.assertEqual(len(user_memories), 3)
        
        # Search by confidence
        high_confidence = self.memory_store.search(min_confidence=0.8)
        self.assertEqual(len(high_confidence), 2)
        
    def test_update_memory(self):
        """Test updating memories."""
        memory_id = self.memory_store.store("Alice is human", "fact", "user", 0.9)
        
        # Update confidence
        success = self.memory_store.update(memory_id, confidence=0.95)
        self.assertTrue(success)
        
        memory = self.memory_store.retrieve(memory_id)
        self.assertEqual(memory.confidence, 0.95)
        
    def test_delete_memory(self):
        """Test deleting memories."""
        memory_id = self.memory_store.store("Alice is human", "fact", "user", 0.9)
        
        # Delete memory
        success = self.memory_store.delete(memory_id)
        self.assertTrue(success)
        
        # Verify deletion
        memory = self.memory_store.retrieve(memory_id)
        self.assertIsNone(memory)
        
    def test_forget_memories(self):
        """Test forgetting memories by pattern."""
        self.memory_store.store("Alice is human", "fact", "user", 0.9)
        self.memory_store.store("Bob is human", "fact", "user", 0.8)
        self.memory_store.store("Charlie is robot", "fact", "user", 0.7)
        
        # Forget memories containing "human"
        deleted_count = self.memory_store.forget(pattern="human")
        self.assertEqual(deleted_count, 2)
        
        # Verify remaining memories
        remaining = self.memory_store.search()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].content, "Charlie is robot")
        
    def test_get_stats(self):
        """Test getting memory statistics."""
        self.memory_store.store("Alice is human", "fact", "user", 0.9)
        self.memory_store.store("If X is human then X is mortal", "rule", "user", 0.8)
        
        stats = self.memory_store.get_stats()
        self.assertEqual(stats['total_items'], 2)
        self.assertEqual(stats['by_type']['fact'], 1)
        self.assertEqual(stats['by_type']['rule'], 1)


class MemoryInspectorTests(unittest.TestCase):
    """Test the memory inspector."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory_store = MemoryStore(os.path.join(self.temp_dir, "test_memory.db"))
        self.inspector = MemoryInspector(self.memory_store)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_analyze_memory_patterns(self):
        """Test memory pattern analysis."""
        # Add some memories
        self.memory_store.store("Alice is human", "fact", "user", 0.9)
        self.memory_store.store("Bob is human", "fact", "user", 0.8)
        self.memory_store.store("If X is human then X is mortal", "rule", "user", 0.7)
        
        analysis = self.inspector.analyze_memory_patterns()
        
        self.assertEqual(analysis['total_memories'], 3)
        self.assertEqual(analysis['by_type']['fact'], 2)
        self.assertEqual(analysis['by_type']['rule'], 1)
        self.assertIn('avg_confidence', analysis)
        
    def test_find_memory_gaps(self):
        """Test finding memory gaps."""
        # Add facts without supporting rules
        self.memory_store.store("Alice is human", "fact", "user", 0.9)
        self.memory_store.store("Bob is human", "fact", "user", 0.8)
        
        gaps = self.inspector.find_memory_gaps()
        self.assertIsInstance(gaps, list)
        
    def test_suggest_memory_consolidation(self):
        """Test memory consolidation suggestions."""
        # Add some memories
        self.memory_store.store("Alice is human", "fact", "user", 0.9)
        self.memory_store.store("Alice is human", "fact", "user", 0.8)  # Duplicate
        
        suggestions = self.inspector.suggest_memory_consolidation()
        self.assertIsInstance(suggestions, list)


class AgentManagerTests(unittest.TestCase):
    """Test the agent manager."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.agent_manager = AgentManager(self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_create_agent(self):
        """Test creating an agent."""
        agent = self.agent_manager.create_agent(
            name="test_agent",
            description="A test agent",
            expertise=["reasoning", "memory"]
        )
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.profile.name, "test_agent")
        self.assertEqual(agent.profile.description, "A test agent")
        self.assertEqual(agent.profile.expertise, ["reasoning", "memory"])
        
    def test_get_agent(self):
        """Test getting an agent."""
        self.agent_manager.create_agent("test_agent", "A test agent")
        
        agent = self.agent_manager.get_agent("test_agent")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.profile.name, "test_agent")
        
    def test_list_agents(self):
        """Test listing agents."""
        self.agent_manager.create_agent("agent1", "First agent")
        self.agent_manager.create_agent("agent2", "Second agent")
        
        agents = self.agent_manager.list_agents()
        self.assertEqual(len(agents), 2)
        self.assertEqual(agents[0]['name'], "agent1")
        self.assertEqual(agents[1]['name'], "agent2")
        
    def test_delete_agent(self):
        """Test deleting an agent."""
        self.agent_manager.create_agent("test_agent", "A test agent")
        
        success = self.agent_manager.delete_agent("test_agent")
        self.assertTrue(success)
        
        agent = self.agent_manager.get_agent("test_agent")
        self.assertIsNone(agent)
        
    def test_switch_agent(self):
        """Test switching to an agent."""
        agent = self.agent_manager.create_agent("test_agent", "A test agent")
        
        switched_agent = self.agent_manager.switch_agent("test_agent")
        self.assertEqual(switched_agent, agent)


class BehaviorSchedulerTests(unittest.TestCase):
    """Test the behavior scheduler."""
    
    def setUp(self):
        self.scheduler = BehaviorScheduler()
        self.execution_count = 0
        
    def test_add_task(self):
        """Test adding a task."""
        def test_function():
            self.execution_count += 1
            
        success = self.scheduler.add_task(
            name="test_task",
            function=test_function,
            interval=0.1
        )
        
        self.assertTrue(success)
        self.assertIn("test_task", self.scheduler.tasks)
        
    def test_remove_task(self):
        """Test removing a task."""
        def test_function():
            pass
            
        self.scheduler.add_task("test_task", test_function)
        
        success = self.scheduler.remove_task("test_task")
        self.assertTrue(success)
        self.assertNotIn("test_task", self.scheduler.tasks)
        
    def test_enable_disable_task(self):
        """Test enabling and disabling tasks."""
        def test_function():
            pass
            
        self.scheduler.add_task("test_task", test_function)
        
        # Disable task
        success = self.scheduler.disable_task("test_task")
        self.assertTrue(success)
        self.assertFalse(self.scheduler.tasks["test_task"].enabled)
        
        # Enable task
        success = self.scheduler.enable_task("test_task")
        self.assertTrue(success)
        self.assertTrue(self.scheduler.tasks["test_task"].enabled)
        
    def test_run_task_now(self):
        """Test running a task immediately."""
        def test_function():
            self.execution_count += 1
            
        self.scheduler.add_task("test_task", test_function)
        
        success = self.scheduler.run_task_now("test_task")
        self.assertTrue(success)
        self.assertEqual(self.execution_count, 1)
        
    def test_get_task_status(self):
        """Test getting task status."""
        def test_function():
            pass
            
        self.scheduler.add_task("test_task", test_function, interval=1.0)
        
        status = self.scheduler.get_task_status("test_task")
        self.assertIsNotNone(status)
        self.assertEqual(status['name'], "test_task")
        self.assertEqual(status['type'], "PERIODIC")
        self.assertTrue(status['enabled'])
        
    def test_get_scheduler_stats(self):
        """Test getting scheduler statistics."""
        def test_function():
            pass
            
        self.scheduler.add_task("task1", test_function, interval=1.0)
        self.scheduler.add_task("task2", test_function, interval=2.0)
        
        stats = self.scheduler.get_scheduler_stats()
        self.assertEqual(stats['total_tasks'], 2)
        self.assertEqual(stats['enabled_tasks'], 2)
        self.assertFalse(stats['running'])


class WatcherManagerTests(unittest.TestCase):
    """Test the watcher manager."""
    
    def setUp(self):
        self.watcher_manager = WatcherManager()
        self.trigger_count = 0
        
    def test_add_watcher(self):
        """Test adding a watcher."""
        def condition():
            return True
            
        def action():
            self.trigger_count += 1
            
        success = self.watcher_manager.add_watcher(
            name="test_watcher",
            condition=condition,
            action=action,
            priority=1
        )
        
        self.assertTrue(success)
        self.assertIn("test_watcher", self.watcher_manager.watchers)
        
    def test_remove_watcher(self):
        """Test removing a watcher."""
        def condition():
            return True
            
        def action():
            pass
            
        self.watcher_manager.add_watcher("test_watcher", condition, action)
        
        success = self.watcher_manager.remove_watcher("test_watcher")
        self.assertTrue(success)
        self.assertNotIn("test_watcher", self.watcher_manager.watchers)
        
    def test_check_all_watchers(self):
        """Test checking all watchers."""
        def condition1():
            return True
            
        def condition2():
            return False
            
        def action1():
            self.trigger_count += 1
            
        def action2():
            self.trigger_count += 1
            
        self.watcher_manager.add_watcher("watcher1", condition1, action1, priority=2)
        self.watcher_manager.add_watcher("watcher2", condition2, action2, priority=1)
        
        triggered = self.watcher_manager.check_all_watchers()
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0], "watcher1")
        self.assertEqual(self.trigger_count, 1)
        
    def test_get_watcher_status(self):
        """Test getting watcher status."""
        def condition():
            return True
            
        def action():
            pass
            
        self.watcher_manager.add_watcher("test_watcher", condition, action, priority=1)
        
        status = self.watcher_manager.get_watcher_status("test_watcher")
        self.assertIsNotNone(status)
        self.assertEqual(status['name'], "test_watcher")
        self.assertEqual(status['priority'], 1)
        self.assertTrue(status['enabled'])
        
    def test_get_watcher_stats(self):
        """Test getting watcher statistics."""
        def condition():
            return True
            
        def action():
            pass
            
        self.watcher_manager.add_watcher("watcher1", condition, action, priority=1)
        self.watcher_manager.add_watcher("watcher2", condition, action, priority=2)
        
        stats = self.watcher_manager.get_watcher_stats()
        self.assertEqual(stats['total_watchers'], 2)
        self.assertEqual(stats['enabled_watchers'], 2)
        self.assertEqual(stats['priority_distribution'][1], 1)
        self.assertEqual(stats['priority_distribution'][2], 1)


class ActionManagerTests(unittest.TestCase):
    """Test the action manager."""
    
    def setUp(self):
        self.action_manager = ActionManager()
        self.execution_count = 0
        
    def test_add_action(self):
        """Test adding an action."""
        def test_function():
            self.execution_count += 1
            
        success = self.action_manager.add_action(
            name="test_action",
            action_type=ActionType.MEMORY,
            function=test_function
        )
        
        self.assertTrue(success)
        self.assertIn("test_action", self.action_manager.actions)
        
    def test_remove_action(self):
        """Test removing an action."""
        def test_function():
            pass
            
        self.action_manager.add_action("test_action", ActionType.MEMORY, test_function)
        
        success = self.action_manager.remove_action("test_action")
        self.assertTrue(success)
        self.assertNotIn("test_action", self.action_manager.actions)
        
    def test_execute_action(self):
        """Test executing an action."""
        def test_function():
            self.execution_count += 1
            return True
            
        self.action_manager.add_action("test_action", ActionType.MEMORY, test_function)
        
        success = self.action_manager.execute_action("test_action")
        self.assertTrue(success)
        self.assertEqual(self.execution_count, 1)
        
    def test_get_action_status(self):
        """Test getting action status."""
        def test_function():
            pass
            
        self.action_manager.add_action("test_action", ActionType.MEMORY, test_function)
        
        status = self.action_manager.get_action_status("test_action")
        self.assertIsNotNone(status)
        self.assertEqual(status['name'], "test_action")
        self.assertEqual(status['type'], "MEMORY")
        self.assertTrue(status['enabled'])
        
    def test_get_action_stats(self):
        """Test getting action statistics."""
        def test_function():
            pass
            
        self.action_manager.add_action("action1", ActionType.MEMORY, test_function)
        self.action_manager.add_action("action2", ActionType.REASONING, test_function)
        
        stats = self.action_manager.get_action_stats()
        self.assertEqual(stats['total_actions'], 2)
        self.assertEqual(stats['enabled_actions'], 2)
        self.assertEqual(stats['type_distribution']['MEMORY'], 1)
        self.assertEqual(stats['type_distribution']['REASONING'], 1)


class IntegrationTests(unittest.TestCase):
    """Integration tests for enhanced features."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_memory_persistence(self):
        """Test that memories persist across sessions."""
        # Create memory store
        memory_store = MemoryStore(os.path.join(self.temp_dir, "test.db"))
        
        # Store some memories
        memory_store.store("Alice is human", "fact", "user", 0.9)
        memory_store.store("Bob is human", "fact", "user", 0.8)
        
        # Create new memory store (simulating restart)
        new_memory_store = MemoryStore(os.path.join(self.temp_dir, "test.db"))
        
        # Verify memories persist
        memories = new_memory_store.search()
        self.assertEqual(len(memories), 2)
        
    def test_agent_communication(self):
        """Test agent communication."""
        agent_manager = AgentManager(self.temp_dir)
        
        # Create two agents
        agent1 = agent_manager.create_agent("agent1", "First agent")
        agent2 = agent_manager.create_agent("agent2", "Second agent")
        
        # Send message from agent1 to agent2
        message = agent1.send_message("agent2", "Hello from agent1")
        self.assertIsNotNone(message)
        self.assertEqual(message.sender, "agent1")
        self.assertEqual(message.recipient, "agent2")
        
    def test_autonomous_behaviors(self):
        """Test autonomous behaviors."""
        scheduler = BehaviorScheduler()
        execution_count = 0
        
        def test_function():
            nonlocal execution_count
            execution_count += 1
            
        # Add a task
        scheduler.add_task("test_task", test_function, interval=0.1)
        
        # Start scheduler
        scheduler.start()
        
        # Let it run for a bit
        time.sleep(0.5)
        
        # Stop scheduler
        scheduler.stop()
        
        # Verify task executed
        self.assertGreater(execution_count, 0)
        
    def test_watcher_system(self):
        """Test watcher system."""
        watcher_manager = WatcherManager()
        trigger_count = 0
        
        def condition():
            return True
            
        def action():
            nonlocal trigger_count
            trigger_count += 1
            
        # Add watcher
        watcher_manager.add_watcher("test_watcher", condition, action)
        
        # Check watchers
        triggered = watcher_manager.check_all_watchers()
        self.assertEqual(len(triggered), 1)
        self.assertEqual(trigger_count, 1)


if __name__ == '__main__':
    unittest.main()
