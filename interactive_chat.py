#!/usr/bin/env python3
"""
Interactive chat with OmniMind Civilizational Feedback System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.conversational.api import OmniMindChat

def main():
    print("🌟 OmniMind Civilizational Feedback System - Interactive Chat")
    print("=" * 70)
    print("Available personalities: Strategos, Archivist, Lawmaker, Oracle")
    print("Type 'quit' to exit, 'switch <personality>' to change personality")
    print("=" * 70)
    
    # Initialize the chat system
    chat = OmniMindChat("interactive_chat_data")
    
    print("\n🧠 OmniMind is ready! What would you like to discuss?")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🌟 OmniMind: Farewell! May your civilizations flourish!")
                break
            
            if user_input.lower().startswith('switch '):
                personality = user_input[7:].strip()
                if personality.lower() in ['strategos', 'archivist', 'lawmaker', 'oracle']:
                    print(f"\n🔄 Switching to {personality}...")
                    # The system will automatically detect personality hints
                    user_input = f"Switch to {personality}. {user_input}"
                else:
                    print(f"\n❌ Unknown personality: {personality}")
                    continue
            
            if not user_input:
                continue
            
            # Get response from OmniMind
            response = chat.chat(user_input)
            
            # Display response
            if response['success']:
                print(f"\n🧠 OmniMind ({response['personality']}): {response['response']}")
                if 'confidence' in response:
                    print(f"📊 Confidence: {response['confidence']:.2f}")
            else:
                print(f"\n❌ Error: {response.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n\n🌟 OmniMind: Farewell! May your civilizations flourish!")
            break
        except Exception as e:
            print(f"\n❌ System error: {e}")

if __name__ == "__main__":
    main()
