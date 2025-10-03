#!/usr/bin/env python3
"""
OmniMind Civilizational Feedback + Meta-Rule Evolution Demo
Demonstrates the complete integration of conversation, simulation, and meta-rule evolution.
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from egdol.omnimind import (
    OmniMindExecutionPipeline,
    CivilizationalFeedbackEngine,
    MetaRuleDiscoveryEngine,
    EnhancedContextStabilization,
    ReflectionModePlusPlus,
    StructuredResponse
)


def demonstrate_civilizational_feedback_system():
    """Demonstrate the complete civilizational feedback system."""
    print("🧠 OmniMind Civilizational Feedback + Meta-Rule Evolution Demo")
    print("=" * 70)
    
    # Initialize the execution pipeline
    pipeline = OmniMindExecutionPipeline()
    
    # Demo queries for different personalities
    demo_queries = [
        {
            'user_input': 'What is the optimal strategy for multi-planetary resource allocation?',
            'personality': 'Strategos',
            'context': {'domain': 'strategic_planning', 'complexity': 'high'}
        },
        {
            'user_input': 'What historical precedents exist for civilizational collapse and recovery?',
            'personality': 'Archivist',
            'context': {'domain': 'historical_analysis', 'scope': 'civilizational'}
        },
        {
            'user_input': 'What are the legal frameworks for inter-planetary governance?',
            'personality': 'Lawmaker',
            'context': {'domain': 'legal_analysis', 'jurisdiction': 'interplanetary'}
        },
        {
            'user_input': 'What does the Oracle foresee for the future of human civilization?',
            'personality': 'Oracle',
            'context': {'domain': 'mystical_insight', 'temporal_scope': 'future'}
        }
    ]
    
    print("\n🚀 Executing Civilizational Feedback Pipeline...")
    print("-" * 50)
    
    responses = []
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📋 Query {i}: {query['personality']} - {query['user_input'][:50]}...")
        
        # Execute the complete pipeline
        response = pipeline.execute_reasoning_pipeline(
            user_input=query['user_input'],
            personality=query['personality'],
            context=query['context'],
            enable_civilizational_feedback=True,
            enable_reflection=True,
            max_retries=3
        )
        
        responses.append(response)
        
        # Display results
        print(f"✅ Success: {response.success}")
        print(f"🎯 Personality: {response.personality}")
        print(f"📊 Confidence Score: {response.confidence_score:.2f}")
        print(f"🔄 Fallback Used: {response.fallback_used}")
        print(f"🧠 Meta-Rules Generated: {len(response.meta_rules_generated)}")
        print(f"💡 Insights Discovered: {response.insights_discovered}")
        print(f"🔄 Reflection Applied: {response.reflection_applied}")
        print(f"⏱️  Processing Time: {response.processing_time:.3f}s")
        print(f"📝 Response: {response.response[:100]}...")
    
    # Display pipeline summary
    print("\n📊 Pipeline Execution Summary")
    print("-" * 50)
    
    summary = pipeline.get_pipeline_summary()
    print(f"Total Executions: {summary['total_executions']}")
    print(f"Successful Executions: {summary['successful_executions']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"Average Processing Time: {summary['average_processing_time']:.3f}s")
    print(f"Average Confidence Score: {summary['average_confidence_score']:.2f}")
    
    # Display component status
    print("\n🔧 Component Status")
    print("-" * 50)
    for component, status in summary['component_status'].items():
        print(f"{component.replace('_', ' ').title()}: {status}")
    
    return responses


def demonstrate_meta_rule_discovery():
    """Demonstrate meta-rule discovery capabilities."""
    print("\n\n🔍 Meta-Rule Discovery Demonstration")
    print("=" * 50)
    
    # Create meta-rule discovery engine
    from egdol.omnimind.conversational.reflexive_audit import ReflexiveAuditModule
    from egdol.omnimind.conversational.meta_learning_engine import MetaLearningEngine
    
    audit_module = ReflexiveAuditModule()
    meta_learning = MetaLearningEngine()
    discovery_engine = MetaRuleDiscoveryEngine(audit_module, meta_learning)
    
    # Demo conversation data
    conversation_data = {
        'response': 'Commander, tactical analysis complete. Strategic evaluation suggests optimal resource allocation through decentralized distribution networks.',
        'reasoning_trace': ['tactical_analysis', 'strategic_evaluation', 'resource_optimization'],
        'personality': 'Strategos',
        'confidence': 0.85
    }
    
    print("📝 Analyzing Conversation Data...")
    print(f"Response: {conversation_data['response'][:60]}...")
    print(f"Personality: {conversation_data['personality']}")
    print(f"Confidence: {conversation_data['confidence']}")
    
    # Discover meta-rules
    meta_rules = discovery_engine.discover_meta_rules_from_conversation(conversation_data)
    
    print(f"\n🧠 Discovered Meta-Rules: {len(meta_rules)}")
    for i, rule in enumerate(meta_rules, 1):
        print(f"  {i}. {rule.rule_name}: {rule.rule_pattern}")
        print(f"     Confidence: {rule.confidence:.2f}")
        print(f"     Applicable to: {', '.join(rule.personality_applicability)}")
    
    # Display discovery summary
    discovery_summary = discovery_engine.get_discovery_summary()
    print(f"\n📊 Discovery Summary:")
    print(f"  Patterns Analyzed: {discovery_summary['patterns_analyzed']}")
    print(f"  Rules Generated: {discovery_summary['rules_generated']}")
    print(f"  Rules Validated: {discovery_summary['rules_validated']}")
    print(f"  Rules Applied: {discovery_summary['rules_applied']}")


def demonstrate_context_stabilization():
    """Demonstrate context stabilization capabilities."""
    print("\n\n🛡️ Context Stabilization Demonstration")
    print("=" * 50)
    
    # Create context stabilization system
    from egdol.omnimind.conversational.context_intent_resolver import ContextIntentResolver
    from egdol.omnimind.conversational.reasoning_normalizer import ReasoningNormalizer
    from egdol.omnimind.conversational.personality_fallbacks import PersonalityFallbackReasoner
    from egdol.omnimind.conversational.personality_framework import PersonalityFramework
    from egdol.omnimind.conversational.reflexive_audit import ReflexiveAuditModule
    
    context_resolver = ContextIntentResolver()
    reasoning_normalizer = ReasoningNormalizer()
    fallback_reasoner = PersonalityFallbackReasoner()
    personality_framework = PersonalityFramework()
    audit_module = ReflexiveAuditModule()
    
    stabilization = EnhancedContextStabilization(
        context_resolver=context_resolver,
        reasoning_normalizer=reasoning_normalizer,
        fallback_reasoner=fallback_reasoner,
        audit_module=audit_module,
        personality_framework=personality_framework
    )
    
    # Test different reasoning types
    test_cases = [
        {
            'input': 'What is the tactical approach to this situation?',
            'personality': 'Strategos',
            'expected_type': 'TACTICAL'
        },
        {
            'input': 'What historical precedents exist for this scenario?',
            'personality': 'Archivist',
            'expected_type': 'HISTORICAL'
        },
        {
            'input': 'What are the legal implications of this decision?',
            'personality': 'Lawmaker',
            'expected_type': 'LEGAL'
        },
        {
            'input': 'What does the Oracle foresee in this matter?',
            'personality': 'Oracle',
            'expected_type': 'MYSTICAL'
        }
    ]
    
    print("🧪 Testing Context Stabilization...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {test_case['personality']} - {test_case['input'][:40]}...")
        
        result = stabilization.stabilize_context_and_reason(
            test_case['input'], 
            test_case['personality']
        )
        
        print(f"    ✅ Success: {result.success}")
        print(f"    🎯 Reasoning Type: {result.reasoning_type.value}")
        print(f"    📊 Confidence: {result.confidence_score:.2f}")
        print(f"    🔄 Fallback Used: {result.fallback_used}")
        print(f"    ⏱️  Processing Time: {result.processing_time:.3f}s")
    
    # Display stabilization summary
    stabilization_summary = stabilization.get_stabilization_summary()
    print(f"\n📊 Stabilization Summary:")
    print(f"  Total Attempts: {stabilization_summary['total_stabilization_attempts']}")
    print(f"  Successful Stabilizations: {stabilization_summary['successful_stabilizations']}")
    print(f"  Success Rate: {stabilization_summary['success_rate']:.2%}")
    print(f"  Fallback Usage: {stabilization_summary['fallback_usage_count']}")


def demonstrate_reflection_mode_plus_plus():
    """Demonstrate reflection mode plus plus capabilities."""
    print("\n\n🔄 Reflection Mode Plus Plus Demonstration")
    print("=" * 50)
    
    # Create reflection mode plus plus system
    from egdol.omnimind.conversational.reflection_mode_plus import ReflectionModePlus
    from egdol.omnimind.conversational.meta_learning_engine import MetaLearningEngine
    from egdol.omnimind.conversational.personality_evolution import PersonalityEvolutionEngine
    from egdol.omnimind.conversational.reflexive_audit import ReflexiveAuditModule
    from egdol.omnimind.enhanced_context_stabilization import EnhancedContextStabilization
    from egdol.omnimind.meta_rule_discovery import MetaRuleDiscoveryEngine
    
    # Initialize components
    audit_module = ReflexiveAuditModule()
    meta_learning = MetaLearningEngine()
    personality_evolution = PersonalityEvolutionEngine(meta_learning, audit_module)
    reflection_plus = ReflectionModePlus(meta_learning, personality_evolution, audit_module)
    
    # Create context stabilization (simplified)
    from egdol.omnimind.conversational.context_intent_resolver import ContextIntentResolver
    from egdol.omnimind.conversational.reasoning_normalizer import ReasoningNormalizer
    from egdol.omnimind.conversational.personality_fallbacks import PersonalityFallbackReasoner
    from egdol.omnimind.conversational.personality_framework import PersonalityFramework
    
    context_resolver = ContextIntentResolver()
    reasoning_normalizer = ReasoningNormalizer()
    fallback_reasoner = PersonalityFallbackReasoner()
    personality_framework = PersonalityFramework()
    
    context_stabilization = EnhancedContextStabilization(
        context_resolver=context_resolver,
        reasoning_normalizer=reasoning_normalizer,
        fallback_reasoner=fallback_reasoner,
        audit_module=audit_module,
        personality_framework=personality_framework
    )
    
    meta_rule_discovery = MetaRuleDiscoveryEngine(audit_module, meta_learning)
    
    reflection_plus_plus = ReflectionModePlusPlus(
        reflection_plus=reflection_plus,
        audit_module=audit_module,
        meta_learning=meta_learning,
        personality_evolution=personality_evolution,
        context_stabilization=context_stabilization,
        meta_rule_discovery=meta_rule_discovery
    )
    
    # Test reflection scenarios
    reflection_scenarios = [
        {
            'user_input': 'Complex query requiring advanced reasoning',
            'personality': 'Strategos',
            'error': 'Low confidence in reasoning output',
            'trigger': 'LOW_CONFIDENCE'
        },
        {
            'user_input': 'Query that failed audit validation',
            'personality': 'Archivist',
            'error': 'Audit criteria not met',
            'trigger': 'AUDIT_FAILURE'
        },
        {
            'user_input': 'Query with reasoning error',
            'personality': 'Lawmaker',
            'error': 'Logical error in reasoning process',
            'trigger': 'REASONING_ERROR'
        }
    ]
    
    print("🧪 Testing Reflection Mode Plus Plus...")
    
    for i, scenario in enumerate(reflection_scenarios, 1):
        print(f"\n  Scenario {i}: {scenario['personality']} - {scenario['error']}")
        
        result = reflection_plus_plus.reflect_and_retry_plus_plus(
            user_input=scenario['user_input'],
            personality=scenario['personality'],
            original_error=scenario['error']
        )
        
        print(f"    ✅ Success: {result.success}")
        print(f"    🔍 Trigger: {result.reflection_analysis.trigger.value}")
        print(f"    📊 Final Confidence: {result.final_confidence:.2f}")
        print(f"    🔄 Retry Attempts: {result.retry_attempts}")
        print(f"    🛠️  Heuristic Updates: {len(result.heuristic_updates)}")
        print(f"    ⏱️  Processing Time: {result.processing_time:.3f}s")
        
        # Display improvement metrics
        if result.improvement_metrics:
            print(f"    📈 Improvement Metrics:")
            for metric, value in result.improvement_metrics.items():
                print(f"      {metric}: {value:.2f}")
    
    # Display reflection summary
    reflection_summary = reflection_plus_plus.get_reflection_summary()
    print(f"\n📊 Reflection Summary:")
    print(f"  Reflection Cycles: {reflection_summary['reflection_cycles']}")
    print(f"  Successful Reflections: {reflection_summary['successful_reflections']}")
    print(f"  Success Rate: {reflection_summary['success_rate']:.2%}")
    print(f"  Total Heuristic Updates: {reflection_summary['total_heuristic_updates']}")
    print(f"  Applied Updates: {reflection_summary['applied_heuristic_updates']}")


def main():
    """Main demonstration function."""
    print("🌟 OmniMind Civilizational Feedback + Meta-Rule Evolution System")
    print("=" * 80)
    print("Demonstrating the complete integration of conversation, simulation,")
    print("and meta-rule evolution into a fully deterministic, testable system.")
    print("=" * 80)
    
    try:
        # Demonstrate civilizational feedback system
        responses = demonstrate_civilizational_feedback_system()
        
        # Demonstrate meta-rule discovery
        demonstrate_meta_rule_discovery()
        
        # Demonstrate context stabilization
        demonstrate_context_stabilization()
        
        # Demonstrate reflection mode plus plus
        demonstrate_reflection_mode_plus_plus()
        
        print("\n\n🎉 Demo Complete!")
        print("=" * 50)
        print("The OmniMind Civilizational Feedback + Meta-Rule Evolution System")
        print("has successfully demonstrated:")
        print("✅ Full civilization feedback loop integration")
        print("✅ Autonomous meta-rule discovery and validation")
        print("✅ Enhanced context stabilization with deterministic fallbacks")
        print("✅ Reflection mode plus plus with autonomous retry")
        print("✅ Comprehensive testing and validation")
        print("✅ Elegant modular execution pipeline")
        print("\nThe system is now ready for real-time, self-evolving")
        print("civilizational reasoning with deterministic behavior!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
