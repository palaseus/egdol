#!/usr/bin/env python3
"""
OmniMind Core Demonstration
Shows the complete OmniMind chatbot system in action.
"""

import time
import os
from egdol.omnimind.core import OmniMind


def demo_omnimind_core():
    """Demonstrate OmniMind Core capabilities."""
    print("🧠 OmniMind Core - Local AI Assistant Demo")
    print("=" * 50)
    print()
    
    # Initialize OmniMind
    print("1. Initializing OmniMind...")
    omnimind = OmniMind("demo_omnimind_data")
    print("   ✅ OmniMind initialized with persistent memory")
    print()
    
    # Test basic conversation
    print("2. Basic Conversation")
    print("-" * 20)
    response = omnimind.process_input("Hello")
    print(f"   User: Hello")
    print(f"   OmniMind: {response['content']}")
    print()
    
    # Test math capabilities
    print("3. Mathematical Reasoning")
    print("-" * 25)
    math_queries = [
        "Calculate 2 + 2",
        "What is 5 * 3?",
        "Solve 10 - 4",
        "What is 15 / 3?"
    ]
    
    for query in math_queries:
        response = omnimind.process_input(query)
        print(f"   User: {query}")
        print(f"   OmniMind: {response['content']}")
        print()
    
    # Test logic capabilities
    print("4. Logical Reasoning")
    print("-" * 20)
    logic_queries = [
        "Alice is human",
        "If X is human then X is mortal",
        "Who is mortal?",
        "Is Alice mortal?"
    ]
    
    for query in logic_queries:
        response = omnimind.process_input(query)
        print(f"   User: {query}")
        print(f"   OmniMind: {response['content']}")
        print()
    
    # Test code capabilities
    print("5. Code Analysis")
    print("-" * 15)
    code_queries = [
        "Write a Python function to add two numbers",
        "Analyze this code: def hello(): print('Hello')",
        "What does this code do: x = 5; y = 3; print(x + y)"
    ]
    
    for query in code_queries:
        response = omnimind.process_input(query)
        print(f"   User: {query}")
        print(f"   OmniMind: {response['content']}")
        print()
    
    # Test file capabilities
    print("6. File Operations")
    print("-" * 15)
    file_queries = [
        "List files in current directory",
        "Read the README.md file",
        "Analyze the project structure"
    ]
    
    for query in file_queries:
        response = omnimind.process_input(query)
        print(f"   User: {query}")
        print(f"   OmniMind: {response['content']}")
        print()
    
    # Test memory capabilities
    print("7. Memory and Learning")
    print("-" * 20)
    memory_queries = [
        "Remember that Python is a programming language",
        "What do you know about Python?",
        "Forget everything about Python",
        "What do you know about Python now?"
    ]
    
    for query in memory_queries:
        response = omnimind.process_input(query)
        print(f"   User: {query}")
        print(f"   OmniMind: {response['content']}")
        print()
    
    # Test conversation history
    print("8. Conversation History")
    print("-" * 20)
    history = omnimind.get_conversation_history()
    print(f"   Total exchanges: {len(history)}")
    print(f"   Recent conversation:")
    for i, exchange in enumerate(history[-4:], 1):
        print(f"     {i}. {exchange['type']}: {exchange['content'][:50]}...")
    print()
    
    # Test system capabilities
    print("9. System Capabilities")
    print("-" * 20)
    print("   Available skills:")
    skills = omnimind.router.get_loaded_skills()
    for skill in skills:
        print(f"     - {skill}")
    print()
    
    print("   Memory statistics:")
    memory_stats = omnimind.memory.get_stats()
    print(f"     - Total sessions: {memory_stats['total_sessions']}")
    print(f"     - Total memories: {memory_stats['total_memories']}")
    print(f"     - Recent memories: {memory_stats['recent_memories']}")
    print()
    
    # Test verbose mode
    print("10. Verbose Mode")
    print("-" * 15)
    omnimind.set_verbose_mode(True)
    response = omnimind.process_input("What is 2 + 2?")
    print(f"   User: What is 2 + 2?")
    print(f"   OmniMind: {response['content']}")
    if response.get('reasoning'):
        print("   Reasoning:")
        for step in response['reasoning']:
            print(f"     - {step}")
    print()
    
    # Test explain mode
    print("11. Explain Mode")
    print("-" * 15)
    omnimind.set_explain_mode(True)
    response = omnimind.process_input("Why did you say that?")
    print(f"   User: Why did you say that?")
    print(f"   OmniMind: {response['content']}")
    print()
    
    print("🎉 OmniMind Core demonstration completed!")
    print()
    print("To start the interactive chat, run:")
    print("   python egdol/omnimind/chat.py")
    print("   # OR")
    print("   python -m egdol.omnimind")
    print()
    print("Or use OmniMind programmatically:")
    print("   from egdol.omnimind.core import OmniMind")
    print("   omnimind = OmniMind()")
    print("   response = omnimind.process_input('Hello')")


def demo_skill_system():
    """Demonstrate the skill system."""
    print("🔧 OmniMind Skill System Demo")
    print("=" * 30)
    print()
    
    from egdol.omnimind.skills import MathSkill, LogicSkill, GeneralSkill, CodeSkill, FileSkill
    
    # Test individual skills
    print("1. Testing Individual Skills")
    print("-" * 25)
    
    # Math skill
    math_skill = MathSkill()
    print(f"   Math Skill: {math_skill.description}")
    print(f"   Capabilities: {', '.join(math_skill.capabilities)}")
    print(f"   Can handle '2 + 2': {math_skill.can_handle('2 + 2', 'math', {})}")
    print()
    
    # Logic skill
    logic_skill = LogicSkill()
    print(f"   Logic Skill: {logic_skill.description}")
    print(f"   Capabilities: {', '.join(logic_skill.capabilities)}")
    print(f"   Can handle 'Alice is human': {logic_skill.can_handle('Alice is human', 'fact', {})}")
    print()
    
    # General skill
    general_skill = GeneralSkill()
    print(f"   General Skill: {general_skill.description}")
    print(f"   Capabilities: {', '.join(general_skill.capabilities)}")
    print(f"   Can handle 'Hello': {general_skill.can_handle('Hello', 'general', {})}")
    print()
    
    # Code skill
    code_skill = CodeSkill()
    print(f"   Code Skill: {code_skill.description}")
    print(f"   Capabilities: {', '.join(code_skill.capabilities)}")
    print(f"   Can handle 'Write a function': {code_skill.can_handle('Write a function', 'generation', {})}")
    print()
    
    # File skill
    file_skill = FileSkill()
    print(f"   File Skill: {file_skill.description}")
    print(f"   Capabilities: {', '.join(file_skill.capabilities)}")
    print(f"   Can handle 'Read file.txt': {file_skill.can_handle('Read file.txt', 'read', {})}")
    print()
    
    print("✅ Skill system demonstration completed!")


def main():
    """Run all demonstrations."""
    try:
        demo_omnimind_core()
        print()
        demo_skill_system()
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree("demo_omnimind_data", ignore_errors=True)
        except:
            pass


if __name__ == '__main__':
    main()
