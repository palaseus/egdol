#!/usr/bin/env python3
"""
Simple chat with OmniMind Civilizational Feedback System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.conversational.api import OmniMindChat

def test_chat():
    print("🌟 OmniMind Civilizational Feedback System - Chat Test")
    print("=" * 60)
    
    # Initialize the chat system
    chat = OmniMindChat("chat_test_data")
    
    # Test queries
    test_queries = [
        "Hello, I'm interested in strategic planning for space colonization.",
        "What historical precedents exist for large-scale migration?",
        "What legal frameworks should govern interplanetary trade?",
        "What does the future hold for human civilization?",
        "Switch to Strategos. What is the optimal military strategy for defending multiple planets?",
        "Switch to Archivist. What records should we preserve for future civilizations?",
        "Switch to Lawmaker. What constitutional principles should govern a multi-planetary federation?",
        "Switch to Oracle. What cosmic forces will shape our destiny?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 50)
        
        try:
            response = chat.chat(query)
            
            if response['success']:
                print(f"✅ Success: True")
                print(f"🎯 Personality: {response['personality']}")
                print(f"📊 Confidence: {response.get('confidence', 'N/A')}")
                print(f"📝 Response: {response['response']}")
            else:
                print(f"❌ Success: False")
                print(f"📝 Error: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print()

if __name__ == "__main__":
    test_chat()
