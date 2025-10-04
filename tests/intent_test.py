#!/usr/bin/env python3
"""
Test intent detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.conversational.intent_parser import IntentParser

def test_intent_detection():
    print("🔍 Testing Intent Detection")
    print("=" * 50)
    
    # Initialize intent parser
    parser = IntentParser()
    
    # Test queries
    test_queries = [
        "What is the optimal strategy for space colonization?",
        "What historical precedents exist for large-scale migration?",
        "What legal frameworks should govern interplanetary trade?",
        "What does the future hold for human civilization?"
    ]
    
    for query in test_queries:
        print(f"\n📋 Query: {query}")
        print("-" * 30)
        
        try:
            intent = parser.parse(query)
            print(f"  Intent Type: {intent.intent_type.name}")
            print(f"  Confidence: {intent.confidence}")
            print(f"  Entities: {intent.entities}")
            print(f"  Context Clues: {intent.context_clues}")
            print(f"  Domain: {intent.domain}")
            
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_intent_detection()
