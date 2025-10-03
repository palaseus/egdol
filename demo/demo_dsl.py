#!/usr/bin/env python3
"""
Egdol DSL Interactive Assistant Demo
Demonstrates the natural language interface to the Egdol logic engine.
"""

from egdol.dsl.simple_dsl import SimpleDSL
from egdol.rules_engine import RulesEngine


def demo_dsl():
    """Demonstrate the Egdol DSL capabilities."""
    print("🤖 Egdol DSL Interactive Assistant Demo")
    print("=" * 50)
    print()
    
    # Create DSL instance
    engine = RulesEngine()
    dsl = SimpleDSL(engine)
    
    # Demo scenarios
    scenarios = [
        {
            "title": "🏠 Family Knowledge Base",
            "statements": [
                "Alice is a human",
                "Bob is a human", 
                "Alice is a parent",
                "Bob is a child",
                "if X is a human then X is mortal",
                "if X is a parent then X is responsible",
                "if X is a child then X is young"
            ],
            "queries": [
                "who is mortal?",
                "who is responsible?",
                "is Alice mortal?",
                "is Bob young?"
            ]
        },
        {
            "title": "🎓 Academic System",
            "statements": [
                "John is a student",
                "Mary is a professor",
                "if X is a student then X learns",
                "if X is a professor then X teaches",
                "if X learns and X is smart then X succeeds"
            ],
            "queries": [
                "who learns?",
                "who teaches?",
                "is John a student?"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"{scenario['title']}")
        print("-" * 30)
        
        # Add facts and rules
        for statement in scenario['statements']:
            result = dsl.execute(statement)
            if result['type'] == 'fact':
                print(f"✅ Added fact: {result['description']}")
            elif result['type'] == 'rule':
                print(f"✅ Added rule: {result['description']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        print()
        
        # Query the knowledge base
        for query in scenario['queries']:
            result = dsl.execute(query)
            if result['type'] == 'query':
                results = result['results']
                if results:
                    print(f"🔍 {result['query']}:")
                    for i, res in enumerate(results, 1):
                        print(f"   {i}. {res}")
                else:
                    print(f"🔍 {result['query']}: No results found")
            else:
                print(f"❌ Query error: {result.get('error', 'Unknown error')}")
        
        print()
        
        # Show stats
        result = dsl.execute(':stats')
        stats = result['stats']
        print(f"📊 Knowledge Base: {stats['num_facts']} facts, {stats['num_rules']} rules")
        print()
        print("=" * 50)
        print()
    
    # Demo advanced features
    print("🚀 Advanced Features Demo")
    print("-" * 30)
    
    # Context memory
    print("Context Memory:")
    dsl.execute("Alice is a programmer")
    dsl.execute("She is smart")  # 'She' refers to Alice
    result = dsl.execute("is Alice smart?")
    print(f"   Alice is smart: {len(result['results']) > 0}")
    
    # Commands
    print("\nCommands:")
    result = dsl.execute(':facts')
    print(f"   Facts count: {result['count']}")
    
    result = dsl.execute(':rules')
    print(f"   Rules count: {result['count']}")
    
    print()
    print("🎉 Demo Complete!")
    print()
    print("To start the interactive REPL, run:")
    print("   python -m egdol.dsl.simple_dsl")
    print()
    print("Or use the DSL programmatically:")
    print("   from egdol.dsl.simple_dsl import SimpleDSL")


if __name__ == '__main__':
    demo_dsl()
