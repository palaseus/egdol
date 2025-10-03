#!/usr/bin/env python3
"""
Egdol Enhanced Features Demonstration
Shows persistent memory, autonomous behaviors, and multi-agent capabilities.
"""

import time
import os
from egdol.memory import MemoryStore, MemoryItem
from egdol.meta import MemoryInspector, RuleInspector, RuleScorer, ConfidenceTracker
from egdol.agents import AgentManager, AgentProfile
from egdol.autonomous import BehaviorScheduler, WatcherManager, ActionManager
from egdol.autonomous.actions import ActionType
from egdol.dsl.simple_dsl import SimpleDSL
from egdol.rules_engine import RulesEngine


def demo_persistent_memory():
    """Demonstrate persistent memory capabilities."""
    print("🧠 Persistent Memory Demo")
    print("=" * 40)
    
    # Create memory store
    memory_store = MemoryStore("demo_memory.db")
    
    # Store some memories
    print("1. Storing memories...")
    memory_store.store("Alice is a human", "fact", "user", 0.9)
    memory_store.store("Bob is a human", "fact", "user", 0.8)
    memory_store.store("If X is human then X is mortal", "rule", "user", 0.7)
    memory_store.store("Alice is smart", "fact", "user", 0.6)
    
    # Search memories
    print("2. Searching memories...")
    facts = memory_store.search(item_type="fact")
    print(f"   Found {len(facts)} facts")
    
    high_confidence = memory_store.search(min_confidence=0.8)
    print(f"   Found {len(high_confidence)} high-confidence memories")
    
    # Get statistics
    stats = memory_store.get_stats()
    print(f"3. Memory statistics: {stats}")
    
    # Test memory inspector
    print("4. Memory analysis...")
    inspector = MemoryInspector(memory_store)
    analysis = inspector.analyze_memory_patterns()
    print(f"   Total memories: {analysis['total_memories']}")
    print(f"   By type: {analysis['by_type']}")
    print(f"   Average confidence: {analysis['avg_confidence']:.2f}")
    
    print()


def demo_autonomous_behaviors():
    """Demonstrate autonomous behaviors."""
    print("🔁 Autonomous Behaviors Demo")
    print("=" * 40)
    
    # Create scheduler
    scheduler = BehaviorScheduler()
    execution_count = 0
    
    def periodic_task():
        nonlocal execution_count
        execution_count += 1
        print(f"   Periodic task executed (count: {execution_count})")
    
    def cleanup_task():
        print("   Cleanup task executed")
    
    # Add tasks
    print("1. Adding scheduled tasks...")
    scheduler.add_task("periodic", periodic_task, interval=0.5)
    scheduler.add_task("cleanup", cleanup_task, interval=1.0)
    
    # Start scheduler
    print("2. Starting scheduler...")
    scheduler.start()
    
    # Let it run
    print("3. Running for 2 seconds...")
    time.sleep(2)
    
    # Stop scheduler
    scheduler.stop()
    print(f"4. Scheduler stopped. Total executions: {execution_count}")
    
    # Test watchers
    print("5. Testing watchers...")
    watcher_manager = WatcherManager()
    trigger_count = 0
    
    def condition():
        return True  # Always trigger for demo
    
    def action():
        nonlocal trigger_count
        trigger_count += 1
        print(f"   Watcher triggered (count: {trigger_count})")
    
    watcher_manager.add_watcher("demo_watcher", condition, action, priority=1)
    triggered = watcher_manager.check_all_watchers()
    print(f"   Watchers triggered: {triggered}")
    
    # Test actions
    print("6. Testing actions...")
    action_manager = ActionManager()
    
    def demo_action():
        print("   Action executed")
        return True
    
    action_manager.add_action("demo_action", ActionType.MEMORY, demo_action)
    action_manager.execute_action("demo_action")
    
    print()


def demo_multi_agent_system():
    """Demonstrate multi-agent capabilities."""
    print("🖥 Multi-Agent System Demo")
    print("=" * 40)
    
    # Create agent manager
    agent_manager = AgentManager("demo_agents")
    
    # Create agents
    print("1. Creating agents...")
    philosopher = agent_manager.create_agent(
        name="philosopher",
        description="A philosophical reasoning agent",
        expertise=["ethics", "logic", "metaphysics"],
        personality={"thinking_style": "deep", "curiosity": "high"}
    )
    
    engineer = agent_manager.create_agent(
        name="engineer", 
        description="A practical engineering agent",
        expertise=["systems", "optimization", "problem_solving"],
        personality={"thinking_style": "practical", "efficiency": "high"}
    )
    
    # List agents
    print("2. Available agents:")
    agents = agent_manager.list_agents()
    for agent in agents:
        print(f"   {agent['name']}: {agent['description']}")
    
    # Switch to philosopher
    print("3. Switching to philosopher...")
    current_agent = agent_manager.switch_agent("philosopher")
    print(f"   Current agent: {current_agent.profile.name}")
    
    # Test agent thinking
    print("4. Testing agent reasoning...")
    result = current_agent.think("What is the meaning of life?")
    print(f"   Agent response: {result}")
    
    # Test agent communication
    print("5. Testing agent communication...")
    message = philosopher.send_message("engineer", "Hello, can you help me solve a problem?")
    print(f"   Message sent: {message.content}")
    
    # Get communication stats
    stats = agent_manager.get_communication_stats()
    print(f"6. Communication stats: {stats}")
    
    print()


def demo_self_modification():
    """Demonstrate self-modification capabilities."""
    print("🧪 Self-Modification Demo")
    print("=" * 40)
    
    # Create components
    engine = RulesEngine()
    memory_store = MemoryStore("demo_self.db")
    
    # Add some facts and rules
    print("1. Adding initial knowledge...")
    memory_store.store("Alice is human", "fact", "user", 0.9)
    memory_store.store("Bob is human", "fact", "user", 0.8)
    memory_store.store("If X is human then X is mortal", "rule", "user", 0.7)
    
    # Test rule inspector
    print("2. Analyzing rules...")
    rule_inspector = RuleInspector(engine, memory_store)
    usage_stats = rule_inspector.analyze_rule_usage()
    print(f"   Rule usage stats: {usage_stats}")
    
    # Test rule scorer
    print("3. Scoring rules...")
    rule_scorer = RuleScorer(memory_store)
    top_rules = rule_scorer.get_top_rules(limit=5)
    print(f"   Top rules: {top_rules}")
    
    # Test confidence tracker
    print("4. Tracking confidence...")
    confidence_tracker = ConfidenceTracker(memory_store)
    
    # Update confidence
    memory_id = memory_store.search(limit=1)[0].id
    confidence_tracker.update_confidence(memory_id, 0.95, "User verification")
    
    # Get confidence trend
    trend = confidence_tracker.get_confidence_trend(memory_id)
    print(f"   Confidence trend: {trend}")
    
    # Test memory gaps
    print("5. Finding memory gaps...")
    memory_inspector = MemoryInspector(memory_store)
    gaps = memory_inspector.find_memory_gaps()
    print(f"   Memory gaps: {gaps}")
    
    print()


def demo_enhanced_repl():
    """Demonstrate enhanced REPL capabilities."""
    print("💻 Enhanced REPL Demo")
    print("=" * 40)
    
    # Create DSL instance
    engine = RulesEngine()
    dsl = SimpleDSL(engine)
    
    print("1. Testing DSL with memory...")
    
    # Add facts
    result = dsl.execute("Alice is a human")
    print(f"   Added fact: {result['description']}")
    
    result = dsl.execute("Bob is a human")
    print(f"   Added fact: {result['description']}")
    
    # Add rule
    result = dsl.execute("if X is a human then X is mortal")
    print(f"   Added rule: {result['description']}")
    
    # Query
    result = dsl.execute("who is mortal?")
    print(f"   Query results: {len(result['results'])} found")
    
    print("2. Enhanced features available:")
    print("   - :remember <content> - Remember something explicitly")
    print("   - :forget <pattern> - Forget memories matching pattern")
    print("   - :memories - List all memories")
    print("   - :introspect - Introspect system state")
    print("   - :agents - List all agents")
    print("   - :switch <agent> - Switch to agent")
    print("   - :watchers - List watchers")
    print("   - :tasks - List scheduled tasks")
    
    print()


def main():
    """Run all demonstrations."""
    print("🚀 Egdol Enhanced Features Demonstration")
    print("=" * 50)
    print()
    
    try:
        demo_persistent_memory()
        demo_autonomous_behaviors()
        demo_multi_agent_system()
        demo_self_modification()
        demo_enhanced_repl()
        
        print("🎉 All demonstrations completed successfully!")
        print()
        print("To start the enhanced REPL, run:")
        print("   python -m egdol.dsl.enhanced_repl")
        print()
        print("Or use the components programmatically:")
        print("   from egdol.memory import MemoryStore")
        print("   from egdol.agents import AgentManager")
        print("   from egdol.autonomous import BehaviorScheduler")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            os.remove("demo_memory.db")
            os.remove("demo_self.db")
            import shutil
            shutil.rmtree("demo_agents", ignore_errors=True)
        except:
            pass


if __name__ == '__main__':
    main()
