#!/usr/bin/env python3
"""
Test pattern matching for intent detection
"""

import re

def test_patterns():
    print("🔍 Testing Intent Pattern Matching")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "What is the optimal strategy for space colonization?",
        "What historical precedents exist for large-scale migration?",
        "What legal frameworks should govern interplanetary trade?",
        "What does the future hold for human civilization?"
    ]
    
    # Define patterns
    strategic_patterns = [
        r'\b(strategy|tactics|war|conflict|battle)\b',
        r'\b(planning|coordination|alliance|diplomacy)\b',
        r'\b(resources|territory|power|influence)\b'
    ]
    
    civilizational_patterns = [
        r'\b(civilization|culture|society|history)\b',
        r'\b(evolution|development|progress|advancement)\b',
        r'\b(historical|precedent|past|ancient)\b'
    ]
    
    meta_rule_patterns = [
        r'\b(meta|rule|principle|law|governance)\b',
        r'\b(universal|fundamental|underlying)\b',
        r'\b(discover|find|identify|detect)\b'
    ]
    
    universe_patterns = [
        r'\b(universe|reality|dimension|world)\b',
        r'\b(compare|contrast|different|alternative)\b',
        r'\b(parallel|multiverse|cosmic)\b'
    ]
    
    for query in test_queries:
        print(f"\n📋 Query: {query}")
        input_lower = query.lower()
        
        # Test strategic patterns
        strategic_match = any(re.search(pattern, input_lower) for pattern in strategic_patterns)
        print(f"  Strategic: {strategic_match}")
        if strategic_match:
            for pattern in strategic_patterns:
                if re.search(pattern, input_lower):
                    print(f"    Matched pattern: {pattern}")
        
        # Test civilizational patterns
        civilizational_match = any(re.search(pattern, input_lower) for pattern in civilizational_patterns)
        print(f"  Civilizational: {civilizational_match}")
        if civilizational_match:
            for pattern in civilizational_patterns:
                if re.search(pattern, input_lower):
                    print(f"    Matched pattern: {pattern}")
        
        # Test meta rule patterns
        meta_rule_match = any(re.search(pattern, input_lower) for pattern in meta_rule_patterns)
        print(f"  Meta Rule: {meta_rule_match}")
        if meta_rule_match:
            for pattern in meta_rule_patterns:
                if re.search(pattern, input_lower):
                    print(f"    Matched pattern: {pattern}")
        
        # Test universe patterns
        universe_match = any(re.search(pattern, input_lower) for pattern in universe_patterns)
        print(f"  Universe: {universe_match}")
        if universe_match:
            for pattern in universe_patterns:
                if re.search(pattern, input_lower):
                    print(f"    Matched pattern: {pattern}")

if __name__ == "__main__":
    test_patterns()
