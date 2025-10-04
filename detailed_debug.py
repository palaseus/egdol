#!/usr/bin/env python3
"""
Detailed debug chat with OmniMind to see reasoning type detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.conversational.api import OmniMindChat

def detailed_debug():
    print("🔍 OmniMind Detailed Debug - Reasoning Type Detection")
    print("=" * 70)
    
    # Initialize the chat system
    chat = OmniMindChat("detailed_debug_data")
    
    # Test a simple query and see what happens internally
    query = "What is the optimal strategy for space colonization?"
    print(f"\n📋 Query: {query}")
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
        print(f"  Entities: {context.entities}")
        
        # Check normalization
        normalized = interface.reasoning_normalizer.normalize_input(query, "Strategos", context)
        print(f"\n🔍 Normalization:")
        print(f"  Reasoning Type: {normalized.reasoning_type}")
        print(f"  Requires Fallback: {normalized.requires_fallback}")
        print(f"  Fallback Type: {normalized.fallback_type}")
        print(f"  Confidence: {normalized.confidence}")
        
        # Check if reasoning engine would be used
        would_use_reasoning = not normalized.requires_fallback and not interface.reasoning_normalizer.should_use_fallback(normalized)
        print(f"\n🔍 Reasoning Engine Decision:")
        print(f"  Would Use Reasoning Engine: {would_use_reasoning}")
        
        if would_use_reasoning:
            print(f"  Reasoning Type for Engine: {normalized.reasoning_type}")
            print(f"  Available Engine Methods:")
            print(f"    - civilizational_reasoning -> process_civilizational_query")
            print(f"    - strategic_reasoning -> process_strategic_analysis")
            print(f"    - meta_rule_reasoning -> process_meta_rule_discovery")
            print(f"    - universe_reasoning -> process_universe_comparison")
        
        # Now get the actual response
        response = chat.chat(query)
        print(f"\n📝 Final Response:")
        print(f"  Success: {response['success']}")
        print(f"  Personality: {response['personality']}")
        print(f"  Response: {response['response']}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    detailed_debug()
