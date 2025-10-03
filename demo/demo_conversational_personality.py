#!/usr/bin/env python3
"""
OmniMind Conversational Personality Layer Demonstration
Shows the complete conversational personality system in action.
"""

import time
import os
from egdol.omnimind.conversational.api import OmniMindChat, quick_chat, chat_with_strategos, chat_with_archivist, chat_with_lawmaker, chat_with_oracle


def demo_basic_chat():
    """Demonstrate basic chat functionality."""
    print("🧠 OmniMind Conversational Personality Layer - Basic Chat Demo")
    print("=" * 70)
    print()
    
    # Initialize chat
    print("1. Initializing OmniMind Chat...")
    chat = OmniMindChat("demo_conversational_data")
    print("   ✅ OmniMind Chat initialized")
    print()
    
    # Basic conversation
    print("2. Basic Conversation")
    print("-" * 20)
    response = chat.chat("Hello, how are you?")
    print(f"   User: Hello, how are you?")
    print(f"   OmniMind: {response['response']}")
    print(f"   Personality: {response['personality']}")
    print()
    
    # Math conversation
    print("3. Mathematical Reasoning")
    print("-" * 25)
    response = chat.chat("What is 2 + 2?")
    print(f"   User: What is 2 + 2?")
    print(f"   OmniMind: {response['response']}")
    print()
    
    # Logic conversation
    print("4. Logical Reasoning")
    print("-" * 20)
    response = chat.chat("If all humans are mortal and Socrates is human, is Socrates mortal?")
    print(f"   User: If all humans are mortal and Socrates is human, is Socrates mortal?")
    print(f"   OmniMind: {response['response']}")
    print()
    
    # Get conversation summary
    print("5. Conversation Summary")
    print("-" * 20)
    summary = chat.get_conversation_summary()
    print(f"   Total turns: {summary.get('total_turns', 0)}")
    print(f"   Current phase: {summary.get('current_phase', 'Unknown')}")
    print(f"   Active personality: {summary.get('active_personality', 'Unknown')}")
    print()
    
    # End conversation
    print("6. Ending Conversation")
    print("-" * 20)
    end_result = chat.end_conversation()
    print(f"   Conversation ended successfully: {end_result['success']}")
    print()


def demo_personality_switching():
    """Demonstrate personality switching."""
    print("🎭 OmniMind Personality Switching Demo")
    print("=" * 50)
    print()
    
    chat = OmniMindChat("demo_personality_data")
    
    # Test all personalities
    personalities = chat.get_available_personalities()
    print(f"Available personalities: {', '.join(personalities)}")
    print()
    
    # Strategos personality
    print("1. Strategos (Military Strategist)")
    print("-" * 35)
    response = chat.chat("What is the best military strategy?", "Strategos")
    print(f"   User: What is the best military strategy?")
    print(f"   Strategos: {response['response']}")
    print()
    
    # Archivist personality
    print("2. Archivist (Historian-Philosopher)")
    print("-" * 40)
    response = chat.chat("Tell me about ancient civilizations", "Archivist")
    print(f"   User: Tell me about ancient civilizations")
    print(f"   Archivist: {response['response']}")
    print()
    
    # Lawmaker personality
    print("3. Lawmaker (Meta-Rule Discoverer)")
    print("-" * 40)
    response = chat.chat("What are the fundamental laws governing this system?", "Lawmaker")
    print(f"   User: What are the fundamental laws governing this system?")
    print(f"   Lawmaker: {response['response']}")
    print()
    
    # Oracle personality
    print("4. Oracle (Universe Comparer)")
    print("-" * 35)
    response = chat.chat("Compare different universes and their laws", "Oracle")
    print(f"   User: Compare different universes and their laws")
    print(f"   Oracle: {response['response']}")
    print()
    
    # Get personality insights
    print("5. Personality Usage Insights")
    print("-" * 30)
    insights = chat.get_personality_insights()
    print(f"   Total switches: {insights.get('total_switches', 0)}")
    print(f"   Most used: {insights.get('most_used', 'None')}")
    print()


def demo_civilizational_intelligence():
    """Demonstrate civilizational intelligence queries."""
    print("🏛️ OmniMind Civilizational Intelligence Demo")
    print("=" * 55)
    print()
    
    chat = OmniMindChat("demo_civilizational_data")
    
    # Civilizational analysis
    print("1. Civilizational Pattern Analysis")
    print("-" * 35)
    response = chat.chat("Analyze the evolution of civilizations and their patterns")
    print(f"   User: Analyze the evolution of civilizations and their patterns")
    print(f"   OmniMind: {response['response']}")
    print(f"   Reasoning available: {response['reasoning_available']}")
    print()
    
    # Strategic analysis
    print("2. Strategic Analysis")
    print("-" * 20)
    response = chat.chat("What is the optimal strategy for a civilization to thrive?", "Strategos")
    print(f"   User: What is the optimal strategy for a civilization to thrive?")
    print(f"   Strategos: {response['response']}")
    print()
    
    # Meta-rule discovery
    print("3. Meta-Rule Discovery")
    print("-" * 20)
    response = chat.chat("Discover the meta-rules governing civilizational development", "Lawmaker")
    print(f"   User: Discover the meta-rules governing civilizational development")
    print(f"   Lawmaker: {response['response']}")
    print()
    
    # Universe comparison
    print("4. Universe Comparison")
    print("-" * 20)
    response = chat.chat("Compare different universes and their civilizational laws", "Oracle")
    print(f"   User: Compare different universes and their civilizational laws")
    print(f"   Oracle: {response['response']}")
    print()
    
    # Get reasoning summary
    print("5. Reasoning Summary")
    print("-" * 20)
    reasoning_summary = chat.get_reasoning_summary()
    print(f"   Total reasoning traces: {reasoning_summary.get('total_traces', 0)}")
    print(f"   Total insights: {reasoning_summary.get('total_insights', 0)}")
    print(f"   Meta rules applied: {reasoning_summary.get('meta_rules_applied', 0)}")
    print()


def demo_quick_chat_functions():
    """Demonstrate quick chat convenience functions."""
    print("⚡ OmniMind Quick Chat Functions Demo")
    print("=" * 45)
    print()
    
    # Quick chat with different personalities
    print("1. Quick Chat with Strategos")
    print("-" * 30)
    response = chat_with_strategos("What is the best tactical approach?")
    print(f"   User: What is the best tactical approach?")
    print(f"   Strategos: {response}")
    print()
    
    print("2. Quick Chat with Archivist")
    print("-" * 30)
    response = chat_with_archivist("Tell me about the wisdom of the ages")
    print(f"   User: Tell me about the wisdom of the ages")
    print(f"   Archivist: {response}")
    print()
    
    print("3. Quick Chat with Lawmaker")
    print("-" * 30)
    response = chat_with_lawmaker("What are the universal principles of governance?")
    print(f"   User: What are the universal principles of governance?")
    print(f"   Lawmaker: {response}")
    print()
    
    print("4. Quick Chat with Oracle")
    print("-" * 30)
    response = chat_with_oracle("What are the cosmic patterns of reality?")
    print(f"   User: What are the cosmic patterns of reality?")
    print(f"   Oracle: {response}")
    print()


def demo_advanced_api():
    """Demonstrate advanced API features."""
    print("🔧 OmniMind Advanced API Demo")
    print("=" * 35)
    print()
    
    from egdol.omnimind.conversational.api import OmniMindChatAdvanced
    
    # Initialize advanced chat
    print("1. Initializing Advanced Chat...")
    chat = OmniMindChatAdvanced("demo_advanced_data")
    print("   ✅ Advanced chat initialized")
    print()
    
    # Process with reasoning
    print("2. Processing with Full Reasoning Trace")
    print("-" * 40)
    result = chat.process_with_reasoning("Analyze the strategic implications of civilizational evolution")
    print(f"   User: Analyze the strategic implications of civilizational evolution")
    print(f"   Response: {result['response']}")
    print(f"   Personality: {result['personality']}")
    print(f"   Reasoning trace available: {result['reasoning_trace'] is not None}")
    print()
    
    # Intent analysis
    print("3. Intent Analysis")
    print("-" * 15)
    intent = chat.analyze_intent("What are the fundamental laws of the universe?")
    print(f"   Message: What are the fundamental laws of the universe?")
    print(f"   Intent type: {intent['intent_type']}")
    print(f"   Confidence: {intent['confidence']}")
    print(f"   Domain: {intent['domain']}")
    print(f"   Complexity: {intent['complexity_level']}")
    print()
    
    # Get internal components
    print("4. Internal Components Access")
    print("-" * 30)
    interface = chat.get_interface()
    personality_framework = chat.get_personality_framework()
    reasoning_engine = chat.get_reasoning_engine()
    
    print(f"   Interface: {type(interface).__name__}")
    print(f"   Personality Framework: {type(personality_framework).__name__}")
    print(f"   Reasoning Engine: {type(reasoning_engine).__name__}")
    print()


def demo_multi_turn_conversation():
    """Demonstrate multi-turn conversation with personality switching."""
    print("🔄 OmniMind Multi-Turn Conversation Demo")
    print("=" * 45)
    print()
    
    chat = OmniMindChat("demo_multi_turn_data")
    
    # Multi-turn conversation
    print("1. Multi-Turn Conversation with Personality Switching")
    print("-" * 55)
    
    messages = [
        ("Hello, I need strategic advice", "Strategos"),
        ("What is the best approach for a complex situation?", None),  # Keep current personality
        ("Switch to Archivist", "Archivist"),
        ("Tell me about historical examples of similar situations", None),
        ("What wisdom can we learn from the past?", None),
        ("Switch to Lawmaker", "Lawmaker"),
        ("What are the fundamental principles governing this?", None),
        ("Switch to Oracle", "Oracle"),
        ("How does this relate to universal patterns?", None)
    ]
    
    for i, (message, personality) in enumerate(messages, 1):
        print(f"   Turn {i}:")
        print(f"   User: {message}")
        if personality:
            print(f"   Switching to: {personality}")
        
        response = chat.chat(message, personality)
        print(f"   {response['personality']}: {response['response']}")
        print()
    
    # Get final summary
    print("2. Final Conversation Summary")
    print("-" * 30)
    summary = chat.get_conversation_summary()
    print(f"   Total turns: {summary.get('total_turns', 0)}")
    print(f"   Current personality: {summary.get('active_personality', 'Unknown')}")
    print(f"   Context type: {summary.get('context_type', 'Unknown')}")
    print()
    
    # Get personality insights
    insights = chat.get_personality_insights()
    print(f"   Personality usage:")
    for personality, count in insights.get('usage_distribution', {}).items():
        print(f"     {personality}: {count:.1%}")
    print()


def main():
    """Run all demonstrations."""
    try:
        print("🚀 OmniMind Conversational Personality Layer - Complete Demo")
        print("=" * 70)
        print()
        
        # Run all demos
        demo_basic_chat()
        print()
        
        demo_personality_switching()
        print()
        
        demo_civilizational_intelligence()
        print()
        
        demo_quick_chat_functions()
        print()
        
        demo_advanced_api()
        print()
        
        demo_multi_turn_conversation()
        print()
        
        print("🎉 All demonstrations completed successfully!")
        print()
        print("To use OmniMind Chat in your own code:")
        print("   from egdol.omnimind.conversational.api import OmniMindChat")
        print("   chat = OmniMindChat()")
        print("   response = chat.chat('Your message here')")
        print()
        print("For quick one-liners:")
        print("   from egdol.omnimind.conversational.api import quick_chat")
        print("   response = quick_chat('Your message here')")
        print()
        print("For personality-specific chats:")
        print("   from egdol.omnimind.conversational.api import chat_with_strategos")
        print("   response = chat_with_strategos('Your message here')")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            import shutil
            for data_dir in ["demo_conversational_data", "demo_personality_data", 
                           "demo_civilizational_data", "demo_advanced_data", 
                           "demo_multi_turn_data"]:
                shutil.rmtree(data_dir, ignore_errors=True)
        except:
            pass


if __name__ == '__main__':
    main()
