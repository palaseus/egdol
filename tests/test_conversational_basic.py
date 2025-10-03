#!/usr/bin/env python3
"""
Basic test for Conversational Personality Layer
Tests the core functionality without running the full test suite.
"""

import tempfile
import os
import sys
from egdol.omnimind.conversational.api import OmniMindChat, quick_chat


def test_basic_functionality():
    """Test basic conversational functionality."""
    print("🧠 Testing OmniMind Conversational Personality Layer")
    print("=" * 55)
    print()
    
    # Test 1: Basic chat
    print("1. Testing basic chat...")
    try:
        chat = OmniMindChat("test_data")
        response = chat.chat("Hello")
        print(f"   ✅ Basic chat works: {response['success']}")
        print(f"   Response: {response['response'][:50]}...")
        print(f"   Personality: {response['personality']}")
    except Exception as e:
        print(f"   ❌ Basic chat failed: {e}")
        return False
    print()
    
    # Test 2: Personality switching
    print("2. Testing personality switching...")
    try:
        personalities = chat.get_available_personalities()
        print(f"   Available personalities: {personalities}")
        
        # Switch to Archivist
        success = chat.switch_personality("Archivist")
        print(f"   ✅ Personality switching works: {success}")
        print(f"   Current personality: {chat.get_current_personality()}")
    except Exception as e:
        print(f"   ❌ Personality switching failed: {e}")
        return False
    print()
    
    # Test 3: Quick chat functions
    print("3. Testing quick chat functions...")
    try:
        response = quick_chat("What is 2 + 2?")
        print(f"   ✅ Quick chat works: {len(response) > 0}")
        print(f"   Response: {response[:50]}...")
    except Exception as e:
        print(f"   ❌ Quick chat failed: {e}")
        return False
    print()
    
    # Test 4: Conversation summary
    print("4. Testing conversation summary...")
    try:
        summary = chat.get_conversation_summary()
        print(f"   ✅ Conversation summary works: {len(summary) > 0}")
        print(f"   Total turns: {summary.get('total_turns', 0)}")
    except Exception as e:
        print(f"   ❌ Conversation summary failed: {e}")
        return False
    print()
    
    # Test 5: End conversation
    print("5. Testing conversation ending...")
    try:
        end_result = chat.end_conversation()
        print(f"   ✅ Conversation ending works: {end_result['success']}")
    except Exception as e:
        print(f"   ❌ Conversation ending failed: {e}")
        return False
    print()
    
    return True


def test_personality_specific():
    """Test personality-specific functionality."""
    print("🎭 Testing Personality-Specific Functionality")
    print("=" * 45)
    print()
    
    from egdol.omnimind.conversational.api import chat_with_strategos, chat_with_archivist, chat_with_lawmaker, chat_with_oracle
    
    # Test Strategos
    print("1. Testing Strategos...")
    try:
        response = chat_with_strategos("What is the best military strategy?")
        print(f"   ✅ Strategos works: {len(response) > 0}")
        print(f"   Response: {response[:50]}...")
    except Exception as e:
        print(f"   ❌ Strategos failed: {e}")
        return False
    print()
    
    # Test Archivist
    print("2. Testing Archivist...")
    try:
        response = chat_with_archivist("Tell me about ancient wisdom")
        print(f"   ✅ Archivist works: {len(response) > 0}")
        print(f"   Response: {response[:50]}...")
    except Exception as e:
        print(f"   ❌ Archivist failed: {e}")
        return False
    print()
    
    # Test Lawmaker
    print("3. Testing Lawmaker...")
    try:
        response = chat_with_lawmaker("What are the fundamental laws?")
        print(f"   ✅ Lawmaker works: {len(response) > 0}")
        print(f"   Response: {response[:50]}...")
    except Exception as e:
        print(f"   ❌ Lawmaker failed: {e}")
        return False
    print()
    
    # Test Oracle
    print("4. Testing Oracle...")
    try:
        response = chat_with_oracle("What are the cosmic patterns?")
        print(f"   ✅ Oracle works: {len(response) > 0}")
        print(f"   Response: {response[:50]}...")
    except Exception as e:
        print(f"   ❌ Oracle failed: {e}")
        return False
    print()
    
    return True


def main():
    """Run basic tests."""
    print("🚀 OmniMind Conversational Personality Layer - Basic Test")
    print("=" * 65)
    print()
    
    try:
        # Test basic functionality
        basic_success = test_basic_functionality()
        
        if basic_success:
            print("✅ Basic functionality tests passed!")
            print()
            
            # Test personality-specific functionality
            personality_success = test_personality_specific()
            
            if personality_success:
                print("✅ Personality-specific tests passed!")
                print()
                print("🎉 All basic tests passed! The conversational layer is working.")
                print()
                print("To use in your code:")
                print("   from egdol.omnimind.conversational.api import OmniMindChat")
                print("   chat = OmniMindChat()")
                print("   response = chat.chat('Your message here')")
                print()
                print("To run the full demo:")
                print("   python demo/demo_conversational_personality.py")
                print()
                print("To start interactive chat:")
                print("   python -m egdol.omnimind.conversational.cli")
            else:
                print("❌ Personality-specific tests failed!")
                return False
        else:
            print("❌ Basic functionality tests failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree("test_data", ignore_errors=True)
        except:
            pass
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
