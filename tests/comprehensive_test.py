#!/usr/bin/env python3
"""
Comprehensive test with OmniMind to see all reasoning types
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.conversational.api import OmniMindChat

def comprehensive_test():
    print("🔍 OmniMind Comprehensive Test - All Reasoning Types")
    print("=" * 70)
    
    # Initialize the chat system
    chat = OmniMindChat("comprehensive_test_data")
    
    # Test queries for different reasoning types
    test_queries = [
        ("What is the optimal strategy for space colonization?", "strategic"),
        ("What historical precedents exist for large-scale migration?", "historical"),
        ("What legal frameworks should govern interplanetary trade?", "legal"),
        ("What does the future hold for human civilization?", "mystical")
    ]
    
    for query, expected_type in test_queries:
        print(f"\n📋 Query: {query}")
        print(f"🎯 Expected Type: {expected_type}")
        print("-" * 50)
        
        try:
            # Get the interface to access internal components
            interface = chat.interface
            
            # Check context resolution
            context = interface.context_resolver.resolve_intent_and_context(query, "Strategos")
            print(f"🔍 Context Resolution:")
            print(f"  Reasoning Type: {context.reasoning_type}")
            print(f"  Intent Type: {context.intent_type}")
            print(f"  Confidence: {context.confidence}")
            print(f"  Domain: {context.domain}")
            
            # Check normalization
            normalized = interface.reasoning_normalizer.normalize_input(query, "Strategos", context)
            print(f"\n🔍 Normalization:")
            print(f"  Reasoning Type: {normalized.reasoning_type}")
            print(f"  Requires Fallback: {normalized.requires_fallback}")
            print(f"  Fallback Type: {normalized.fallback_type}")
            
            # Get the actual response
            response = chat.chat(query)
            print(f"\n📝 Final Response:")
            print(f"  Success: {response['success']}")
            print(f"  Personality: {response['personality']}")
            print(f"  Response: {response['response'][:100]}...")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print()

if __name__ == "__main__":
    comprehensive_test()
