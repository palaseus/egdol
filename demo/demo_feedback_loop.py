#!/usr/bin/env python3
"""
Demo: Feedback Loop & Meta-Learning System
Shows complete feedback collection, analysis, learning, and adaptation cycle.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.conversational.feedback_loop import (
    FeedbackStorage, FeedbackCollector, UserFeedback, 
    FeedbackType, FeedbackSentiment
)
from egdol.omnimind.conversational.meta_learning_engine import (
    MetaLearningEngine, PersonalityProfile, LearningMetrics
)
from egdol.omnimind.conversational.self_healing import SelfHealingController
from egdol.omnimind.conversational.feedback_integration import FeedbackIntegration
from datetime import datetime
import tempfile


def create_sample_feedback_scenarios():
    """Create realistic feedback scenarios for demonstration."""
    scenarios = [
        # Scenario 1: Positive feedback for Strategos
        {
            "session_id": "session_001",
            "response_id": "response_001",
            "feedback_data": {
                "rating": 4.5,
                "text": "Excellent strategic analysis! Very helpful and clear.",
                "personality": "Strategos",
                "depth_preference": "standard",
                "confidence_feedback": 0.8
            }
        },
        # Scenario 2: Negative feedback for Archivist
        {
            "session_id": "session_002", 
            "response_id": "response_002",
            "feedback_data": {
                "rating": 2.0,
                "text": "This response was confusing and unclear. Too shallow.",
                "personality": "Archivist",
                "depth_preference": "deep",
                "confidence_feedback": 0.3
            }
        },
        # Scenario 3: Mixed feedback for Lawmaker
        {
            "session_id": "session_003",
            "response_id": "response_003", 
            "feedback_data": {
                "rating": 3.0,
                "text": "The legal framework analysis was okay but could be more detailed.",
                "personality": "Lawmaker",
                "depth_preference": "deep",
                "confidence_feedback": 0.6
            }
        },
        # Scenario 4: High confidence feedback for Oracle
        {
            "session_id": "session_004",
            "response_id": "response_004",
            "feedback_data": {
                "rating": 5.0,
                "text": "Brilliant mystical insights! The cosmic perspective was perfect.",
                "personality": "Oracle",
                "depth_preference": "deep",
                "confidence_feedback": 0.9
            }
        },
        # Scenario 5: Negative feedback for Strategos (learning trigger)
        {
            "session_id": "session_005",
            "response_id": "response_005",
            "feedback_data": {
                "rating": 1.5,
                "text": "Poor strategic analysis. Wrong approach and confusing.",
                "personality": "Strategos",
                "depth_preference": "deep",
                "confidence_feedback": 0.2
            }
        }
    ]
    
    return scenarios


def demo_feedback_collection():
    """Demo feedback collection system."""
    print("🔄 Feedback Collection System Demo")
    print("=" * 60)
    
    # Initialize components
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    temp_db.close()
    storage = FeedbackStorage(temp_db.name)
    collector = FeedbackCollector(storage)
    
    # Create sample scenarios
    scenarios = create_sample_feedback_scenarios()
    
    print(f"📊 Collecting {len(scenarios)} feedback scenarios...")
    print()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📝 Scenario {i}: {scenario['feedback_data']['personality']} Feedback")
        
        # Collect feedback
        feedback = collector.collect_feedback(
            session_id=scenario["session_id"],
            response_id=scenario["response_id"],
            feedback_data=scenario["feedback_data"]
        )
        
        # Display feedback details
        print(f"   Rating: {feedback.rating}/5.0")
        print(f"   Sentiment: {feedback.sentiment.name}")
        print(f"   Text: {feedback.text[:50]}..." if feedback.text else "   Text: None")
        print(f"   Personality: {feedback.personality}")
        print(f"   Depth Preference: {feedback.depth_preference}")
        print(f"   Confidence Feedback: {feedback.confidence_feedback}")
        print()
    
    # Get feedback summary
    summary = collector.get_feedback_summary("session_001")
    print(f"📈 Session Summary:")
    print(f"   Total Feedback: {summary['total']}")
    print(f"   Average Rating: {summary['average_rating']:.2f}")
    print(f"   Sentiment Distribution: {summary['sentiment_distribution']}")
    print()
    
    # Cleanup
    os.unlink(temp_db.name)
    
    return storage, collector


def demo_meta_learning_engine():
    """Demo meta-learning engine."""
    print("🧠 Meta-Learning Engine Demo")
    print("=" * 60)
    
    # Initialize components
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    temp_db.close()
    storage = FeedbackStorage(temp_db.name)
    engine = MetaLearningEngine(storage)
    
    # Create sample feedback
    scenarios = create_sample_feedback_scenarios()
    collector = FeedbackCollector(storage)
    
    print("📊 Collecting feedback for learning...")
    for scenario in scenarios:
        collector.collect_feedback(
            session_id=scenario["session_id"],
            response_id=scenario["response_id"],
            feedback_data=scenario["feedback_data"]
        )
    
    # Analyze feedback trends
    feedback_batch = storage.get_feedback_batch(10)
    trends = engine.analyze_feedback_trends(feedback_batch)
    
    print("📈 Feedback Trends Analysis:")
    for personality, trend in trends["trends"].items():
        print(f"   {personality}:")
        print(f"     Success Rate: {trend['success_rate']:.2f}")
        print(f"     Depth Misalignment: {trend['depth_misalignment']:.2f}")
        print(f"     Confidence Miscalibration: {trend['confidence_miscalibration']:.2f}")
        print(f"     Recent Trend: {trend['recent_trend']}")
        print()
    
    print(f"🎯 Overall Health: {trends['overall_health']:.2f}")
    print(f"💡 Recommendations: {trends['recommendations']}")
    print()
    
    # Generate learning update
    if engine.should_trigger_learning():
        print("🚀 Triggering Learning Cycle...")
        update = engine.generate_learning_update(feedback_batch)
        
        print(f"📋 Learning Update Generated:")
        print(f"   Update ID: {update.update_id}")
        print(f"   Feedback Batch Size: {len(update.feedback_batch)}")
        print(f"   Personality Adjustments: {len(update.personality_adjustments)}")
        print(f"   Depth Adjustments: {len(update.depth_preference_adjustments)}")
        print(f"   Confidence Adjustments: {len(update.confidence_calibration_adjustments)}")
        print(f"   Synthesis Improvements: {len(update.synthesis_improvements)}")
        print()
        
        # Apply learning update
        success = engine.apply_learning_update(update)
        print(f"✅ Learning Update Applied: {success}")
        
        # Show updated metrics
        metrics = engine.get_learning_metrics()
        print(f"📊 Updated Learning Metrics:")
        print(f"   Total Feedback: {metrics.total_feedback}")
        print(f"   Average Rating: {metrics.average_rating:.2f}")
        print(f"   Positive Feedback: {metrics.positive_feedback}")
        print(f"   Negative Feedback: {metrics.negative_feedback}")
        print()
    else:
        print("⏸️  Learning not triggered (insufficient feedback)")
        print()
    
    # Cleanup
    os.unlink(temp_db.name)
    
    return engine


def demo_self_healing_system():
    """Demo self-healing system."""
    print("🛡️ Self-Healing System Demo")
    print("=" * 60)
    
    # Initialize components
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    temp_db.close()
    storage = FeedbackStorage(temp_db.name)
    engine = MetaLearningEngine(storage)
    healing_controller = SelfHealingController(storage, engine)
    
    # Create system snapshot
    snapshot = healing_controller.create_system_snapshot({
        "demo": True,
        "timestamp": datetime.now().isoformat()
    })
    
    print(f"📸 System Snapshot Created:")
    print(f"   Snapshot ID: {snapshot.snapshot_id}")
    print(f"   System Health: {snapshot.system_health:.2f}")
    print(f"   Stability Score: {snapshot.stability_score:.2f}")
    print(f"   Is Stable: {snapshot.is_stable}")
    print()
    
    # Check system stability
    is_stable, stability_metrics = healing_controller.check_system_stability()
    print(f"🔍 System Stability Check:")
    print(f"   Is Stable: {is_stable}")
    print(f"   Response Quality: {stability_metrics.response_quality:.2f}")
    print(f"   User Satisfaction: {stability_metrics.user_satisfaction:.2f}")
    print(f"   Error Rate: {stability_metrics.error_rate:.2f}")
    print(f"   Performance Degradation: {stability_metrics.performance_degradation:.2f}")
    print(f"   Overall Stability: {stability_metrics.overall_stability:.2f}")
    print()
    
    # Check if rollback should be triggered
    should_rollback = healing_controller.should_trigger_rollback()
    print(f"🔄 Rollback Check:")
    print(f"   Should Rollback: {should_rollback}")
    
    if should_rollback:
        print("⚠️  System instability detected - rollback recommended")
    else:
        print("✅ System is stable - no rollback needed")
    
    print()
    
    # Get system status
    status = healing_controller.get_system_status()
    print(f"📊 System Status:")
    print(f"   Is Stable: {status['is_stable']}")
    print(f"   System Health: {status['system_health']:.2f}")
    print(f"   Should Rollback: {status['should_rollback']}")
    print(f"   Latest Stable Snapshot: {status['latest_stable_snapshot']}")
    print()
    
    # Cleanup
    os.unlink(temp_db.name)


def demo_feedback_integration():
    """Demo complete feedback integration."""
    print("🔗 Feedback Integration Demo")
    print("=" * 60)
    
    # Initialize components
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    temp_db.close()
    storage = FeedbackStorage(temp_db.name)
    collector = FeedbackCollector(storage)
    engine = MetaLearningEngine(storage)
    healing_controller = SelfHealingController(storage, engine)
    integration = FeedbackIntegration(storage, collector, engine, healing_controller)
    
    # Simulate response processing with feedback
    print("📝 Simulating Response Processing with Feedback...")
    print()
    
    # Process responses
    responses = [
        {
            "response": "Commander, strategic analysis complete. I recommend a systematic approach.",
            "personality": "Strategos",
            "user_input": "What is the optimal strategy for space colonization?"
        },
        {
            "response": "From the archives, historical analysis reveals key patterns.",
            "personality": "Archivist", 
            "user_input": "What historical precedents exist for large-scale migration?"
        },
        {
            "response": "According to legal principles, governance frameworks must be established.",
            "personality": "Lawmaker",
            "user_input": "What legal frameworks are needed for interplanetary trade?"
        }
    ]
    
    for i, response in enumerate(responses, 1):
        print(f"📤 Response {i}: {response['personality']}")
        print(f"   Input: {response['user_input']}")
        print(f"   Response: {response['response']}")
        
        # Process response with feedback
        processed_response = integration.process_response_with_feedback(
            response=response,
            session_id=f"session_{i}",
            user_input=response['user_input'],
            personality=response['personality']
        )
        
        print(f"   Feedback Required: {processed_response.get('feedback_required', False)}")
        if processed_response.get('feedback_required'):
            print(f"   Feedback Prompt: {processed_response.get('feedback_prompt', {}).get('text', 'N/A')}")
        print()
    
    # Simulate feedback collection
    print("📥 Simulating Feedback Collection...")
    feedback_scenarios = [
        {"rating": 4.0, "text": "Good strategic analysis", "personality": "Strategos"},
        {"rating": 2.0, "text": "Too shallow for historical analysis", "personality": "Archivist"},
        {"rating": 3.0, "text": "Legal framework was adequate", "personality": "Lawmaker"}
    ]
    
    for i, feedback_data in enumerate(feedback_scenarios, 1):
        feedback = integration.collect_user_feedback(
            session_id=f"session_{i}",
            response_id=f"response_{i}",
            feedback_data=feedback_data
        )
        print(f"   Feedback {i}: {feedback.rating}/5.0 - {feedback.sentiment.name}")
    
    print()
    
    # Get learning status
    learning_status = integration.get_system_learning_status()
    print(f"🧠 System Learning Status:")
    print(f"   Should Learn: {learning_status['should_learn']}")
    print(f"   Should Heal: {learning_status['should_heal']}")
    print(f"   Total Feedback: {learning_status['learning_metrics']['total_feedback']}")
    print(f"   Average Rating: {learning_status['learning_metrics']['average_rating']:.2f}")
    print()
    
    # Cleanup
    os.unlink(temp_db.name)


def main():
    """Run the complete feedback loop demo."""
    
    print("🌟 OmniMind Feedback Loop & Meta-Learning System Demo")
    print("=" * 80)
    print()
    
    try:
        # Demo 1: Feedback Collection
        demo_feedback_collection()
        
        # Demo 2: Meta-Learning Engine
        demo_meta_learning_engine()
        
        # Demo 3: Self-Healing System
        demo_self_healing_system()
        
        # Demo 4: Complete Integration
        demo_feedback_integration()
        
        print("✅ Feedback Loop & Meta-Learning Demo completed successfully!")
        print()
        print("🎯 Key Features Demonstrated:")
        print("   • Structured feedback collection with sentiment analysis")
        print("   • Meta-learning engine with trend analysis")
        print("   • Self-healing mechanisms with rollback capability")
        print("   • Complete feedback integration pipeline")
        print("   • Personality-specific learning and adaptation")
        print("   • System stability monitoring and auto-recovery")
        print("   • Deterministic replay with snapshot management")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
