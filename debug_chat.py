#!/usr/bin/env python3
"""
Debug chat with OmniMind to see what's happening with reasoning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.conversational.api import OmniMindChat

def debug_chat():
    print("🔍 OmniMind Debug Chat - Investigating Reasoning Issues")
    print("=" * 60)
    
    # Initialize the chat system
    chat = OmniMindChat("debug_chat_data")
    
    # Test a simple query and see what happens internally
    query = "What is the optimal strategy for space colonization?"
    print(f"\n📋 Query: {query}")
    print("-" * 50)
    
    try:
        response = chat.chat(query)
        
        print(f"✅ Success: {response['success']}")
        print(f"🎯 Personality: {response['personality']}")
        print(f"📝 Response: {response['response']}")
        
        # Try to get more detailed information
        if hasattr(chat.interface, 'current_session') and chat.interface.current_session:
            session = chat.interface.current_session
            print(f"\n🔍 Session Debug Info:")
            print(f"  Active Personality: {session.active_personality}")
            print(f"  Current Phase: {session.current_phase}")
            print(f"  Turn Count: {len(session.conversation_history)}")
            
            if session.conversation_history:
                last_turn = session.conversation_history[-1]
                print(f"  Last Turn Intent: {last_turn.intent}")
                print(f"  Last Turn Context: {last_turn.context}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chat()
