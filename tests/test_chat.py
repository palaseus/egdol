#!/usr/bin/env python3
"""
Simple test script to interact with OmniMind
"""

from egdol.omnimind.core import OmniMind

def main():
    print("🧠 OmniMind Test Chat")
    print("=" * 50)
    
    # Initialize OmniMind
    omnimind = OmniMind()
    
    # Test conversations
    test_inputs = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Tell me about artificial intelligence",
        "Can you help me with a math problem?",
        "What do you know about machine learning?",
        "Calculate 5 * 3",
        "Explain quantum computing",
        "What are the benefits of renewable energy?"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{i}. User: {user_input}")
        
        try:
            response = omnimind.process_input(user_input)
            print(f"   OmniMind: {response}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main()
