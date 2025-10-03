#!/usr/bin/env python3
"""
Test script to demonstrate OmniMind's reflexive self-improvement capabilities
"""

from egdol.omnimind.core import OmniMind
from egdol.omnimind.conversational import (
    ReflexiveAuditModule, 
    MetaLearningEngine, 
    PersonalityEvolutionEngine,
    ReflectionModePlus
)
from egdol.omnimind.conversational.conversation_state import ConversationTurn
from datetime import datetime

def main():
    print("🧠 OmniMind Reflexive Self-Improvement Test")
    print("=" * 60)
    
    # Initialize OmniMind
    omnimind = OmniMind()
    
    # Initialize reflexive self-improvement components
    audit_module = ReflexiveAuditModule()
    meta_learning = MetaLearningEngine()
    personality_evolution = PersonalityEvolutionEngine(meta_learning, audit_module)
    reflection_plus = ReflectionModePlus(meta_learning, personality_evolution, audit_module)
    
    print("✅ Initialized reflexive self-improvement components")
    
    # Simulate a conversation with multiple turns
    conversation_turns = [
        {
            "user_input": "Hello, I need help with strategic planning",
            "personality": "Strategos",
            "response": "Commander, I recommend a tactical approach to strategic planning. Let's analyze the battlefield and develop a comprehensive strategy.",
            "reasoning_trace": ["analyzed strategic context", "applied military planning principles"],
            "fallback_used": False,
            "confidence": 0.8
        },
        {
            "user_input": "What about the legal implications?",
            "personality": "Lawmaker", 
            "response": "From a legal perspective, we must ensure compliance with all applicable regulations and consider the constitutional framework.",
            "reasoning_trace": ["identified legal context", "applied regulatory analysis"],
            "fallback_used": False,
            "confidence": 0.9
        },
        {
            "user_input": "Can you explain quantum mechanics?",
            "personality": "Oracle",
            "response": "Through the veil of existence, I see that quantum mechanics reveals the fundamental nature of reality at the smallest scales...",
            "reasoning_trace": ["accessed cosmic knowledge", "applied mystical understanding"],
            "fallback_used": True,
            "confidence": 0.6
        },
        {
            "user_input": "Store this information for later",
            "personality": "Archivist",
            "response": "I shall preserve this knowledge in the eternal archives, cataloging it for future reference and retrieval.",
            "reasoning_trace": ["identified storage need", "applied archival procedures"],
            "fallback_used": False,
            "confidence": 0.95
        }
    ]
    
    print("\n🔄 Processing conversation turns with reflexive audit...")
    
    # Process each conversation turn
    for i, turn in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {turn['user_input']}")
        print(f"Personality: {turn['personality']}")
        print(f"Response: {turn['response']}")
        
        # Create ConversationTurn object
        conversation_turn = ConversationTurn(
            turn_id=f'turn_{i}',
            timestamp=datetime.now(),
            user_input=turn['user_input'],
            system_response=turn['response'],
            intent='strategic_planning',  # Example intent
            personality_used=turn['personality'],
            reasoning_trace=turn['reasoning_trace'],
            confidence_score=turn['confidence']
        )
        
        # Audit the conversation turn
        audit_result = audit_module.audit_conversation_turn(conversation_turn)
        
        print(f"Audit Score: {audit_result.confidence_score:.2f}")
        print(f"Metrics: {audit_result.metrics}")
        print(f"Gaps identified: {len(audit_result.gaps_identified)}")
        print(f"Improvement suggestions: {len(audit_result.improvement_suggestions)}")
        
        # Generate learning insights
        turn_data = {
            'turn_id': f'turn_{i}',
            'user_input': turn['user_input'],
            'personality': turn['personality'],
            'response': turn['response'],
            'reasoning_trace': turn['reasoning_trace'],
            'fallback_used': turn['fallback_used'],
            'confidence': turn['confidence']
        }
        
        insights = meta_learning._generate_insights_from_turn(turn_data)
        print(f"Generated {len(insights)} learning insights")
        
        # Apply personality evolution
        if insights:
            evolution_result = personality_evolution.evolve_personality(
                personality=turn['personality'],
                learning_insights=insights
            )
            print(f"Personality evolution applied: {evolution_result.update_type}")
    
    print("\n📊 Meta-Learning Analysis...")
    
    # Get learning insights
    all_insights = meta_learning.get_learning_insights()
    print(f"Total learning insights: {len(all_insights)}")
    
    # Get meta-rule discoveries
    meta_rules = meta_learning.get_meta_rule_discoveries()
    print(f"Meta-rule discoveries: {len(meta_rules)}")
    
    # Get personality evolution summary
    evolution_summary = personality_evolution.get_evolution_summary()
    print(f"Personality evolution stages: {len(evolution_summary)}")
    
    print("\n🔄 Testing Reflection Mode Plus...")
    
    # Test reflection mode with a challenging input
    challenging_input = "Explain the philosophical implications of quantum entanglement in the context of free will"
    reflection_result = reflection_plus.reflect_and_retry_plus(
        user_input=challenging_input,
        personality="Oracle",
        original_error="Insufficient knowledge about quantum entanglement",
        context={"previous_topic": "quantum mechanics"}
    )
    
    print(f"Reflection result: {reflection_result.success}")
    if reflection_result.success:
        print(f"Enhanced response: {reflection_result.response}")
        print(f"Confidence improvement: {reflection_result.confidence_improvement:.2f}")
        print(f"Learning insights generated: {reflection_result.learning_insights_generated}")
    
    print("\n📈 Performance Metrics...")
    
    # Get audit summary
    audit_summary = audit_module.get_audit_summary()
    print(f"Total audits: {audit_summary['total_audits']}")
    print(f"Average score: {audit_summary['average_confidence']:.2f}")
    improvement_opportunities = audit_module.get_improvement_opportunities()
    print(f"Improvement opportunities: {len(improvement_opportunities)}")
    
    print("\n✅ Reflexive Self-Improvement Test Completed!")
    print("The system has successfully demonstrated:")
    print("- Conversation turn auditing")
    print("- Meta-learning from interactions") 
    print("- Personality evolution")
    print("- Enhanced reflection capabilities")
    print("- Performance tracking and optimization")

if __name__ == "__main__":
    main()
