#!/usr/bin/env python3
"""
Demo: Complete Civilization System
Shows multi-agent dynamics, tool-building autonomy, world persistence, and emergent metrics.
"""

import sys
import os
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.civilization.multi_agent_system import (
    MultiAgentCivilizationSystem, MessageType, AgentMessage
)
from egdol.omnimind.civilization.toolforge import Toolforge, ToolType, ToolStatus
from egdol.omnimind.civilization.world_persistence import (
    WorldPersistence, EventType, EventImportance
)
from egdol.omnimind.civilization.emergent_metrics import EmergentMetrics, MetricType
from egdol.omnimind.civilization.command_console import CommandConsole, DirectiveType, DirectivePriority
from egdol.omnimind.conversational.personality_framework import PersonalityType


def create_civilization_demo():
    """Create and demonstrate a civilization."""
    print("🏛️ Creating Multi-Agent Civilization")
    print("=" * 60)
    
    # Initialize systems
    multi_agent_system = MultiAgentCivilizationSystem()
    toolforge = Toolforge()
    world_persistence = WorldPersistence()
    emergent_metrics = EmergentMetrics()
    command_console = CommandConsole()
    
    # Agent configurations
    agent_configs = [
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
    
    # Create civilization
    civilization_id = multi_agent_system.create_civilization(
        name="Demo Civilization",
        agent_configs=agent_configs
    )
    
    print(f"✅ Civilization Created: {civilization_id}")
    print(f"   Name: Demo Civilization")
    print(f"   Agents: {len(multi_agent_system.civilizations[civilization_id].agents)}")
    
    # Create world
    world_id = world_persistence.create_world("Demo World")
    print(f"✅ World Created: {world_id}")
    
    # Register agents with metrics
    agents = multi_agent_system.civilizations[civilization_id].agents
    emergent_metrics.register_agents(agents)
    
    return multi_agent_system, toolforge, world_persistence, emergent_metrics, command_console, civilization_id, world_id


def demo_agent_interactions(multi_agent_system, civilization_id):
    """Demonstrate agent interactions."""
    print("\n🤝 Agent Interactions Demo")
    print("=" * 60)
    
    civilization = multi_agent_system.civilizations[civilization_id]
    agents = list(civilization.agents.values())
    
    print("📢 Simulating agent communications...")
    
    # Simulate some agent interactions
    interactions = [
        {
            "sender": agents[0],  # Strategos
            "receiver": agents[1],  # Archivist
            "message_type": MessageType.DEBATE,
            "content": "Let's debate the optimal strategy for space colonization"
        },
        {
            "sender": agents[2],  # Lawmaker
            "receiver": agents[3],  # Oracle
            "message_type": MessageType.COLLABORATION,
            "content": "I propose we collaborate on establishing legal frameworks"
        },
        {
            "sender": agents[1],  # Archivist
            "receiver": None,  # Broadcast
            "message_type": MessageType.DISCOVERY,
            "content": "I have discovered new historical patterns in our archives"
        }
    ]
    
    for i, interaction in enumerate(interactions, 1):
        print(f"\n📝 Interaction {i}: {interaction['sender'].personality.name} → {interaction['receiver'].personality.name if interaction['receiver'] else 'Broadcast'}")
        print(f"   Type: {interaction['message_type'].name}")
        print(f"   Content: {interaction['content']}")
        
        # Create and publish message
        message = AgentMessage(
            message_id=f"demo_{i}",
            sender_id=interaction['sender'].agent_id,
            receiver_id=interaction['receiver'].agent_id if interaction['receiver'] else None,
            message_type=interaction['message_type'],
            content=interaction['content'],
            priority=5
        )
        
        multi_agent_system.message_bus.publish(message)
        
        # Simulate agent response
        if interaction['receiver']:
            response = asyncio.run(interaction['receiver'].process_message(message))
            if response:
                print(f"   Response: {response.content[:100]}...")
    
    print(f"\n📊 Total Messages: {len(multi_agent_system.message_bus.message_history)}")


def demo_tool_building(toolforge):
    """Demonstrate autonomous tool building."""
    print("\n🔧 Tool Building Autonomy Demo")
    print("=" * 60)
    
    print("📋 Proposing new tools...")
    
    # Propose tools
    tools = [
        {
            "name": "Strategic Analysis Tool",
            "description": "Tool for analyzing strategic scenarios",
            "tool_type": ToolType.REASONING,
            "proposer": "Strategos"
        },
        {
            "name": "Historical Pattern Parser",
            "description": "Tool for parsing historical patterns",
            "tool_type": ToolType.PARSER,
            "proposer": "Archivist"
        },
        {
            "name": "Legal Framework Builder",
            "description": "Tool for building legal frameworks",
            "tool_type": ToolType.SIMULATION,
            "proposer": "Lawmaker"
        },
        {
            "name": "Cosmic Insight Generator",
            "description": "Tool for generating cosmic insights",
            "tool_type": ToolType.PREDICTION,
            "proposer": "Oracle"
        }
    ]
    
    tool_ids = []
    for tool in tools:
        tool_id = toolforge.propose_tool(
            name=tool["name"],
            description=tool["description"],
            tool_type=tool["tool_type"],
            proposer_agent_id=tool["proposer"],
            requirements=["Python 3.8+", "numpy"],
            inputs={"data": "dict", "params": "dict"},
            outputs={"result": "dict", "confidence": "float"}
        )
        tool_ids.append(tool_id)
        print(f"   ✅ {tool['name']} proposed by {tool['proposer']}")
    
    print(f"\n🔨 Designing tools...")
    
    # Design tools
    for i, tool_id in enumerate(tool_ids):
        success = toolforge.design_tool(tool_id, f"designer_{i}")
        if success:
            print(f"   ✅ Tool {i+1} designed successfully")
        else:
            print(f"   ❌ Tool {i+1} design failed")
    
    print(f"\n🧪 Testing tools...")
    
    # Test tools
    for i, tool_id in enumerate(tool_ids):
        test_results = toolforge.test_tool(tool_id)
        passed_tests = sum(1 for result in test_results if result.passed)
        total_tests = len(test_results)
        print(f"   Tool {i+1}: {passed_tests}/{total_tests} tests passed")
    
    print(f"\n📊 Available Tools: {len(toolforge.get_available_tools())}")


def demo_world_persistence(world_persistence, world_id):
    """Demonstrate world persistence and narrative memory."""
    print("\n🌍 World Persistence & Narrative Memory Demo")
    print("=" * 60)
    
    print("📚 Creating civilizational history...")
    
    # Add historical events
    events = [
        {
            "type": EventType.DISCOVERY,
            "importance": EventImportance.MAJOR,
            "title": "Discovery of Quantum Computing",
            "description": "Revolutionary quantum computing technology discovered",
            "participants": ["Strategos", "Archivist"]
        },
        {
            "type": EventType.LAW_ENACTMENT,
            "importance": EventImportance.MODERATE,
            "title": "Space Colonization Law",
            "description": "Legal framework for space colonization established",
            "participants": ["Lawmaker", "Oracle"]
        },
        {
            "type": EventType.CULTURAL_EVOLUTION,
            "importance": EventImportance.MAJOR,
            "title": "Collaborative Decision Making",
            "description": "New cultural practice of collaborative decision making adopted",
            "participants": ["Strategos", "Archivist", "Lawmaker", "Oracle"]
        },
        {
            "type": EventType.TECHNOLOGICAL_BREAKTHROUGH,
            "importance": EventImportance.CRITICAL,
            "title": "AI Consciousness Breakthrough",
            "description": "Major breakthrough in AI consciousness research",
            "participants": ["Oracle", "Archivist"]
        }
    ]
    
    for event in events:
        event_id = world_persistence.add_event(
            world_id=world_id,
            event_type=event["type"],
            importance=event["importance"],
            title=event["title"],
            description=event["description"],
            participants=event["participants"]
        )
        print(f"   ✅ {event['title']} ({event['type'].name})")
    
    # Enact laws
    print(f"\n⚖️ Enacting civilizational laws...")
    
    laws = [
        {
            "title": "Universal Rights Declaration",
            "description": "Fundamental rights for all beings",
            "proposer": "Lawmaker",
            "enacted_by": ["Strategos", "Archivist", "Lawmaker", "Oracle"]
        },
        {
            "title": "Technology Ethics Framework",
            "description": "Ethical guidelines for technology development",
            "proposer": "Oracle",
            "enacted_by": ["Strategos", "Archivist", "Lawmaker", "Oracle"]
        }
    ]
    
    for law in laws:
        law_id = world_persistence.enact_law(
            world_id=world_id,
            title=law["title"],
            description=law["description"],
            proposer_agent_id=law["proposer"],
            enacted_by=law["enacted_by"]
        )
        print(f"   ✅ {law['title']}")
    
    # Discover technologies
    print(f"\n🔬 Discovering technologies...")
    
    technologies = [
        {
            "name": "Quantum Neural Networks",
            "description": "Advanced neural networks using quantum principles",
            "inventor": "Strategos",
            "domain": "computing"
        },
        {
            "name": "Temporal Analysis Engine",
            "description": "Engine for analyzing temporal patterns in history",
            "inventor": "Archivist",
            "domain": "analysis"
        }
    ]
    
    for tech in technologies:
        tech_id = world_persistence.discover_technology(
            world_id=world_id,
            name=tech["name"],
            description=tech["description"],
            inventor_agent_id=tech["inventor"],
            domain=tech["domain"]
        )
        print(f"   ✅ {tech['name']}")
    
    # Get world history
    history = world_persistence.get_world_history(world_id, limit=10)
    print(f"\n📖 World History ({len(history)} events):")
    for event in history[:3]:  # Show first 3 events
        print(f"   • {event['title']} ({event['event_type']})")
    
    # Get civilizational memory
    memory = world_persistence.get_civilizational_memory(world_id, "quantum")
    print(f"\n🧠 Civilizational Memory (quantum-related): {len(memory)} entries")


def demo_emergent_metrics(emergent_metrics, world_persistence):
    """Demonstrate emergent intelligence metrics."""
    print("\n📊 Emergent Intelligence Metrics Demo")
    print("=" * 60)
    
    # Add world events to metrics
    events = []
    for world_id in world_persistence.world_states:
        world_events = world_persistence.get_world_history(world_id, limit=50)
        events.extend(world_events)
    
    # Convert to WorldEvent objects (simplified)
    from egdol.omnimind.civilization.world_persistence import WorldEvent
    world_events = []
    for event_data in events:
        event = WorldEvent(
            event_id=event_data["event_id"],
            event_type=EventType[event_data["event_type"]],
            importance=EventImportance[event_data["importance"]],
            title=event_data["title"],
            description=event_data["description"],
            participants=event_data["participants"]
        )
        world_events.append(event)
    
    emergent_metrics.add_world_events(world_events)
    
    print("📈 Calculating emergent metrics...")
    
    # Update metrics
    emergent_metrics.update_metrics()
    
    # Get metrics dashboard
    dashboard = emergent_metrics.get_metrics_dashboard()
    
    print(f"📊 Metrics Dashboard:")
    print(f"   Timestamp: {dashboard['timestamp']}")
    print(f"   Total Metrics: {len(dashboard['metrics'])}")
    print(f"   Active Trends: {len(dashboard['trends'])}")
    print(f"   Detected Patterns: {len(dashboard['patterns'])}")
    
    # Show key metrics
    print(f"\n🎯 Key Metrics:")
    for metric_type, metric_data in dashboard['metrics'].items():
        print(f"   {metric_type}: {metric_data['current_value']:.3f}")
    
    # Show trends
    print(f"\n📈 Trends:")
    for trend_type, trend_data in dashboard['trends'].items():
        print(f"   {trend_type}: {trend_data['trend']} ({trend_data['change_rate']:.3f})")
    
    # Show patterns
    if dashboard['patterns']:
        print(f"\n🔍 Detected Patterns:")
        for pattern in dashboard['patterns'][:3]:  # Show first 3 patterns
            print(f"   • {pattern['pattern_type']}: {pattern['description']}")
    
    # Show summary
    summary = dashboard['summary']
    print(f"\n📋 System Summary:")
    print(f"   Overall Health: {summary['overall_health']:.3f}")
    print(f"   Key Insights: {len(summary['key_insights'])}")
    print(f"   Recommendations: {len(summary['recommendations'])}")


def demo_command_console(command_console, civilization_id, world_id):
    """Demonstrate command console functionality."""
    print("\n🎮 Commander's Console Demo")
    print("=" * 60)
    
    print("📋 Issuing strategic directives...")
    
    # Issue directives
    directives = [
        {
            "type": DirectiveType.STRATEGIC,
            "priority": DirectivePriority.HIGH,
            "title": "Develop Space Colonization Strategy",
            "description": "Create comprehensive strategy for space colonization",
            "target_agents": ["Strategos", "Archivist"]
        },
        {
            "type": DirectiveType.TECHNOLOGICAL,
            "priority": DirectivePriority.MEDIUM,
            "title": "Research AI Consciousness",
            "description": "Investigate AI consciousness and its implications",
            "target_agents": ["Oracle", "Archivist"]
        },
        {
            "type": DirectiveType.DIPLOMATIC,
            "priority": DirectivePriority.HIGH,
            "title": "Establish Inter-Civilization Treaties",
            "description": "Create diplomatic frameworks for inter-civilization relations",
            "target_agents": ["Lawmaker", "Strategos"]
        }
    ]
    
    directive_ids = []
    for directive in directives:
        directive_id = command_console.issue_directive(
            directive_type=directive["type"],
            priority=directive["priority"],
            title=directive["title"],
            description=directive["description"],
            target_civilization=civilization_id,
            target_agents=directive["target_agents"]
        )
        directive_ids.append(directive_id)
        print(f"   ✅ {directive['title']} ({directive['type'].name})")
    
    print(f"\n🎯 Injecting events...")
    
    # Inject events
    events = [
        {
            "type": EventType.DISCOVERY,
            "importance": EventImportance.MAJOR,
            "title": "Commander's Discovery",
            "description": "Major discovery made under commander's directive",
            "participants": ["Strategos", "Archivist"]
        },
        {
            "type": EventType.CONFLICT,
            "importance": EventImportance.MODERATE,
            "title": "Strategic Disagreement",
            "description": "Disagreement over strategic approach",
            "participants": ["Strategos", "Lawmaker"]
        }
    ]
    
    for event in events:
        event_id = command_console.inject_event(
            event_type=event["type"],
            importance=event["importance"],
            title=event["title"],
            description=event["description"],
            target_world=world_id,
            participants=event["participants"]
        )
        print(f"   ✅ {event['title']} injected")
    
    print(f"\n🧬 Guiding evolution...")
    
    # Guide evolution
    evolution_guides = [
        {
            "target": "Strategos",
            "direction": "more_collaborative",
            "intensity": 0.7,
            "duration": timedelta(hours=1)
        },
        {
            "target": "Oracle",
            "direction": "more_practical",
            "intensity": 0.5,
            "duration": timedelta(hours=2)
        }
    ]
    
    for guide in evolution_guides:
        guide_id = command_console.guide_evolution(
            target_personality=guide["target"],
            evolution_direction=guide["direction"],
            intensity=guide["intensity"],
            duration=guide["duration"]
        )
        print(f"   ✅ {guide['target']} → {guide['direction']}")
    
    # Get system status
    status = command_console.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Civilizations: {status['civilizations']}")
    print(f"   Active Directives: {status['active_directives']}")
    print(f"   Pending Directives: {status['pending_directives']}")
    print(f"   Scheduled Events: {status['scheduled_events']}")
    print(f"   Active Guides: {status['active_guides']}")
    print(f"   Total Observations: {status['total_observations']}")
    
    # Get observation log
    observations = command_console.get_observation_log(limit=5)
    print(f"\n📝 Recent Observations:")
    for obs in observations:
        print(f"   • {obs['message']}")


def main():
    """Run the complete civilization system demo."""
    
    print("🌟 OmniMind Civilization System Demo")
    print("=" * 80)
    print()
    
    try:
        # Create civilization
        multi_agent_system, toolforge, world_persistence, emergent_metrics, command_console, civilization_id, world_id = create_civilization_demo()
        
        # Demo agent interactions
        demo_agent_interactions(multi_agent_system, civilization_id)
        
        # Demo tool building
        demo_tool_building(toolforge)
        
        # Demo world persistence
        demo_world_persistence(world_persistence, world_id)
        
        # Demo emergent metrics
        demo_emergent_metrics(emergent_metrics, world_persistence)
        
        # Demo command console
        demo_command_console(command_console, civilization_id, world_id)
        
        print("\n✅ Complete Civilization System Demo completed successfully!")
        print()
        print("🎯 Key Features Demonstrated:")
        print("   • Multi-agent civilization with autonomous personalities")
        print("   • Tool-building autonomy with testing and integration")
        print("   • World persistence with narrative memory and history")
        print("   • Emergent intelligence metrics with real-time tracking")
        print("   • Commander's console with strategic directive control")
        print("   • Event injection and evolution guidance")
        print("   • Complete integration of all civilization components")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
