#!/usr/bin/env python3
"""
Command Line Interface for OmniMind Conversational Personality Layer
Provides interactive chat interface with personality switching.
"""

import sys
import argparse
from typing import Optional
from .api import OmniMindChat, OmniMindChatAdvanced


def interactive_chat(data_dir: str = "omnimind_chat_data", advanced: bool = False):
    """
    Start interactive chat session.
    
    Args:
        data_dir: Directory for storing conversation data
        advanced: Whether to use advanced API features
    """
    # Initialize chat
    if advanced:
        chat = OmniMindChatAdvanced(data_dir)
        print("🔧 OmniMind Advanced Chat initialized")
    else:
        chat = OmniMindChat(data_dir)
        print("🧠 OmniMind Chat initialized")
    
    print(f"Available personalities: {', '.join(chat.get_available_personalities())}")
    print("Type 'help' for commands, 'quit' to exit")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'personalities':
                print_personalities(chat)
                continue
            elif user_input.lower() == 'current':
                print(f"Current personality: {chat.get_current_personality()}")
                continue
            elif user_input.lower() == 'summary':
                print_summary(chat)
                continue
            elif user_input.lower() == 'insights':
                print_insights(chat)
                continue
            elif user_input.lower() == 'reasoning':
                print_reasoning(chat)
                continue
            elif user_input.lower().startswith('switch '):
                personality = user_input[7:].strip()
                if chat.switch_personality(personality):
                    print(f"Switched to {personality}")
                else:
                    print(f"Personality '{personality}' not found")
                continue
            elif user_input.lower().startswith('analyze '):
                if advanced:
                    message = user_input[8:].strip()
                    intent = chat.analyze_intent(message)
                    print(f"Intent: {intent['intent_type']} (confidence: {intent['confidence']:.2f})")
                    print(f"Domain: {intent['domain']}, Complexity: {intent['complexity_level']:.2f}")
                else:
                    print("Intent analysis requires advanced mode")
                continue
            
            # Process message
            if advanced and user_input.lower().startswith('reasoning '):
                message = user_input[10:].strip()
                result = chat.process_with_reasoning(message)
                print(f"{result['personality']}: {result['response']}")
                if result['reasoning_trace']:
                    print(f"[Reasoning trace available]")
            else:
                result = chat.chat(user_input)
                print(f"{result['personality']}: {result['response']}")
                if result['reasoning_available']:
                    print(f"[Reasoning trace available]")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def print_help():
    """Print help information."""
    print("""
Available commands:
  help          - Show this help message
  personalities - List available personalities
  current       - Show current personality
  switch <name> - Switch to specified personality
  summary       - Show conversation summary
  insights      - Show personality usage insights
  reasoning     - Show reasoning summary
  analyze <msg> - Analyze intent of message (advanced mode)
  reasoning <msg> - Process message with full reasoning trace (advanced mode)
  quit/exit/bye - Exit the chat
""")


def print_personalities(chat):
    """Print available personalities."""
    personalities = chat.get_available_personalities()
    print(f"Available personalities: {', '.join(personalities)}")
    print(f"Current personality: {chat.get_current_personality()}")


def print_summary(chat):
    """Print conversation summary."""
    summary = chat.get_conversation_summary()
    print(f"Total turns: {summary.get('total_turns', 0)}")
    print(f"Current phase: {summary.get('current_phase', 'Unknown')}")
    print(f"Active personality: {summary.get('active_personality', 'Unknown')}")
    print(f"Context type: {summary.get('context_type', 'Unknown')}")


def print_insights(chat):
    """Print personality insights."""
    insights = chat.get_personality_insights()
    print(f"Total switches: {insights.get('total_switches', 0)}")
    print(f"Most used: {insights.get('most_used', 'None')}")
    if 'usage_distribution' in insights:
        print("Usage distribution:")
        for personality, percentage in insights['usage_distribution'].items():
            print(f"  {personality}: {percentage:.1%}")


def print_reasoning(chat):
    """Print reasoning summary."""
    reasoning = chat.get_reasoning_summary()
    print(f"Total reasoning traces: {reasoning.get('total_traces', 0)}")
    print(f"Total insights: {reasoning.get('total_insights', 0)}")
    print(f"Meta rules applied: {reasoning.get('meta_rules_applied', 0)}")
    print(f"Average confidence: {reasoning.get('average_confidence', 0):.2f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="OmniMind Conversational Personality Layer CLI")
    parser.add_argument("--data-dir", default="omnimind_chat_data", 
                       help="Directory for storing conversation data")
    parser.add_argument("--advanced", action="store_true", 
                       help="Use advanced API features")
    parser.add_argument("--personality", choices=["Strategos", "Archivist", "Lawmaker", "Oracle"],
                       help="Start with specific personality")
    parser.add_argument("--message", help="Send a single message and exit")
    
    args = parser.parse_args()
    
    if args.message:
        # Single message mode
        chat = OmniMindChat(args.data_dir)
        if args.personality:
            chat.switch_personality(args.personality)
        result = chat.chat(args.message)
        print(f"{result['personality']}: {result['response']}")
    else:
        # Interactive mode
        if args.personality:
            print(f"Starting with {args.personality} personality...")
        interactive_chat(args.data_dir, args.advanced)


if __name__ == '__main__':
    main()
