#!/usr/bin/env python3
"""
Multi-personality test with OmniMind
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.conversational.api import OmniMindChat

def multi_personality_test():
    print("🌟 OmniMind Multi-Personality Test")
    print("=" * 50)
    
    # Initialize the chat system
    chat = OmniMindChat("multi_personality_test_data")
    
    # Test queries for different personalities
    test_queries = [
        ("Strategos", "What is the optimal military strategy for defending multiple planets?"),
        ("Archivist", "What historical precedents exist for large-scale migration?"),
        ("Lawmaker", "What legal frameworks should govern interplanetary trade?"),
        ("Oracle", "What does the future hold for human civilization?")
    ]
    
    for personality, query in test_queries:
        print(f"\n🎯 Testing {personality}")
        print(f"📋 Query: {query}")
        print("-" * 50)
        
        try:
            # Switch personality and ask question
            switch_query = f"Switch to {personality}. {query}"
            response = chat.chat(switch_query)
            
            print(f"✅ Success: {response['success']}")
            print(f"🎯 Personality: {response['personality']}")
            print(f"📝 Response: {response['response']}")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print()

if __name__ == "__main__":
    multi_personality_test()
