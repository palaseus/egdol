"""
Simple Test Suite for Reflexive Self-Improvement and Meta-Learning Layer
Tests basic functionality without complex integration.
"""

import unittest
import tempfile
import os
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from egdol.omnimind.core import OmniMind
from egdol.omnimind.conversational import (
    ConversationalInterface, PersonalityFramework, Personality, PersonalityType,
    ConversationState, ConversationContext, ConversationPhase, ContextType,
    IntentParser, Intent, IntentType,
    ResponseGenerator, ResponseStyle
)
from egdol.omnimind.conversational.reflexive_audit import (
    ReflexiveAuditModule, AuditResult, PersonalityPerformance, AuditMetric
)
from egdol.omnimind.conversational.meta_learning_engine import (
    MetaLearningEngine, LearningInsight, HeuristicUpdate, LearningMetrics
)
from egdol.omnimind.conversational.feedback_loop import FeedbackStorage, UserFeedback, FeedbackType, FeedbackSentiment
from egdol.omnimind.conversational.personality_evolution import (
    PersonalityEvolutionEngine, PersonalityEvolutionState, EvolutionUpdate, EvolutionStage
)
from egdol.omnimind.conversational.reflection_mode_plus import (
    ReflectionModePlus, ReflectionResultPlus, ReflectionInsightPlus, ReflectionStrategy
)


class TestReflexiveAuditModule(unittest.TestCase):
    """Test the reflexive audit module."""
    
    def setUp(self):
        self.audit_module = ReflexiveAuditModule()
        self.sample_turn = Mock()
        self.sample_turn.turn_id = "test_turn_1"
        self.sample_turn.personality_used = "Strategos"
        self.sample_turn.system_response = "Commander, I recommend a tactical approach."
        self.sample_turn.confidence_score = 0.8
        self.sample_turn.intent = "STRATEGIC_ANALYSIS"
        self.sample_turn.reasoning_trace = []
        self.sample_turn.meta_insights = []
    
    def test_initialization(self):
        """Test audit module initialization."""
        self.assertIsNotNone(self.audit_module)
        self.assertIsInstance(self.audit_module.personality_performance, dict)
        self.assertIsInstance(self.audit_module.audit_history, list)
    
    def test_audit_conversation_turn(self):
        """Test auditing a conversation turn."""
        result = self.audit_module.audit_conversation_turn(self.sample_turn)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AuditResult)
        self.assertIsNotNone(result.turn_id)
        self.assertIsNotNone(result.personality)
        self.assertIsNotNone(result.metrics)
    
    def test_get_improvement_opportunities(self):
        """Test getting improvement opportunities."""
        # Add some performance data
        performance = PersonalityPerformance("TestPersonality")
        performance.total_turns = 10
        performance.successful_turns = 3
        performance.fallback_usage = 8
        performance.average_confidence = 0.3
        performance.reasoning_accuracy = 0.2
        
        self.audit_module.personality_performance["TestPersonality"] = performance
        
        opportunities = self.audit_module.get_improvement_opportunities()
        self.assertIsInstance(opportunities, list)


class TestMetaLearningEngine(unittest.TestCase):
    """Test the meta-learning engine."""
    
    def setUp(self):
        self.storage = FeedbackStorage()
        self.meta_learning_engine = MetaLearningEngine(self.storage)
        self.sample_feedback = [
            UserFeedback(
                feedback_id="test_1",
                session_id="session_1",
                response_id="response_1",
                feedback_type=FeedbackType.RATING,
                rating=4.0,
                sentiment=FeedbackSentiment.POSITIVE,
                personality="Strategos",
                depth_preference="standard",
                confidence_feedback=0.8
            )
        ]
    
    def test_initialization(self):
        """Test meta-learning engine initialization."""
        self.assertIsNotNone(self.meta_learning_engine.storage)
        self.assertIsNotNone(self.meta_learning_engine.personality_profiles)
        self.assertIsNotNone(self.meta_learning_engine.learning_metrics)
        self.assertIsNotNone(self.meta_learning_engine.depth_controller)
        self.assertIsNotNone(self.meta_learning_engine.confidence_calibrator)
        self.assertIsNotNone(self.meta_learning_engine.adaptation_threshold)
        self.assertIsNotNone(self.meta_learning_engine.learning_rate)
    
    def test_analyze_feedback_trends(self):
        """Test analyzing feedback trends."""
        results = self.meta_learning_engine.analyze_feedback_trends(self.sample_feedback)
        
        self.assertIn('trends', results)
        self.assertIn('recommendations', results)
        self.assertIn('overall_health', results)
    
    def test_get_learning_metrics(self):
        """Test getting learning metrics."""
        metrics = self.meta_learning_engine.get_learning_metrics()
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, LearningMetrics)
    
    def test_get_personality_profiles(self):
        """Test getting personality profiles."""
        profiles = self.meta_learning_engine.get_personality_profiles()
        self.assertIsNotNone(profiles)
        self.assertIsInstance(profiles, dict)
        self.assertGreater(len(profiles), 0)
    
    def test_should_trigger_learning(self):
        """Test learning trigger condition."""
        should_trigger = self.meta_learning_engine.should_trigger_learning(10)
        self.assertIsInstance(should_trigger, bool)


class TestPersonalityEvolutionEngine(unittest.TestCase):
    """Test the personality evolution engine."""
    
    def setUp(self):
        self.storage = FeedbackStorage()
        self.meta_learning_engine = MetaLearningEngine(self.storage)
        self.audit_module = ReflexiveAuditModule()
        self.evolution_engine = PersonalityEvolutionEngine(self.meta_learning_engine, self.audit_module)
    
    def test_initialization(self):
        """Test evolution engine initialization."""
        self.assertIsNotNone(self.evolution_engine)
        self.assertIsNotNone(self.evolution_engine.meta_learning_engine)
        self.assertIsNotNone(self.evolution_engine.audit_module)
    
    def test_get_evolution_summary(self):
        """Test getting evolution summary."""
        summary = self.evolution_engine.get_evolution_summary()
        self.assertIsNotNone(summary)
        self.assertIsInstance(summary, dict)


class TestReflectionModePlus(unittest.TestCase):
    """Test the enhanced reflection mode."""
    
    def setUp(self):
        self.storage = FeedbackStorage()
        self.meta_learning_engine = MetaLearningEngine(self.storage)
        self.audit_module = ReflexiveAuditModule()
        self.evolution_engine = PersonalityEvolutionEngine(self.meta_learning_engine, self.audit_module)
        self.reflection_engine = ReflectionModePlus(
            self.meta_learning_engine, 
            self.evolution_engine, 
            self.audit_module
        )
    
    def test_initialization(self):
        """Test reflection engine initialization."""
        self.assertIsNotNone(self.reflection_engine)
        self.assertIsNotNone(self.reflection_engine.meta_learning_engine)
        self.assertIsNotNone(self.reflection_engine.personality_evolution_engine)
        self.assertIsNotNone(self.reflection_engine.audit_module)
    
    def test_get_reflection_summary(self):
        """Test getting reflection summary."""
        summary = self.reflection_engine.get_reflection_summary()
        self.assertIsNotNone(summary)
        self.assertIsInstance(summary, dict)


class TestIntegrationReflexiveSelfImprovement(unittest.TestCase):
    """Test integration of reflexive self-improvement components."""
    
    def setUp(self):
        self.storage = FeedbackStorage()
        self.meta_learning_engine = MetaLearningEngine(self.storage)
        self.audit_module = ReflexiveAuditModule()
        self.evolution_engine = PersonalityEvolutionEngine(self.meta_learning_engine, self.audit_module)
        self.reflection_engine = ReflectionModePlus(
            self.meta_learning_engine, 
            self.evolution_engine, 
            self.audit_module
        )
    
    def test_audit_integration(self):
        """Test audit module integration."""
        self.assertIsNotNone(self.audit_module)
        
        # Test audit functionality
        mock_turn = Mock()
        mock_turn.turn_id = "test_turn"
        mock_turn.personality_used = "Strategos"
        mock_turn.system_response = "Test response"
        mock_turn.confidence_score = 0.8
        mock_turn.intent = "STRATEGIC_ANALYSIS"
        mock_turn.reasoning_trace = []
        mock_turn.meta_insights = []
        
        result = self.audit_module.audit_conversation_turn(mock_turn)
        self.assertIsNotNone(result)
    
    def test_full_meta_learning_pipeline(self):
        """Test complete meta-learning pipeline."""
        # Create sample feedback
        feedback = UserFeedback(
            feedback_id="test_1",
            session_id="session_1",
            response_id="response_1",
            feedback_type=FeedbackType.RATING,
            rating=4.0,
            sentiment=FeedbackSentiment.POSITIVE,
            personality="Strategos",
            depth_preference="standard",
            confidence_feedback=0.8
        )
        
        # Test feedback analysis
        results = self.meta_learning_engine.analyze_feedback_trends([feedback])
        self.assertIsNotNone(results)
        self.assertIn('trends', results)
    
    def test_continuous_improvement_simulation(self):
        """Test continuous improvement simulation."""
        # Simulate multiple feedback cycles
        for i in range(3):
            feedback = UserFeedback(
                feedback_id=f"test_{i}",
                session_id=f"session_{i}",
                response_id=f"response_{i}",
                feedback_type=FeedbackType.RATING,
                rating=4.0,
                sentiment=FeedbackSentiment.POSITIVE,
                personality="Strategos",
                depth_preference="standard",
                confidence_feedback=0.8
            )
            
            results = self.meta_learning_engine.analyze_feedback_trends([feedback])
            self.assertIsNotNone(results)
    
    def test_cross_component_learning(self):
        """Test cross-component learning."""
        # Test that components can learn from each other
        metrics = self.meta_learning_engine.get_learning_metrics()
        self.assertIsNotNone(metrics)
        
        profiles = self.meta_learning_engine.get_personality_profiles()
        self.assertIsNotNone(profiles)
        
        evolution_summary = self.evolution_engine.get_evolution_summary()
        self.assertIsNotNone(evolution_summary)
        
        reflection_summary = self.reflection_engine.get_reflection_summary()
        self.assertIsNotNone(reflection_summary)


if __name__ == '__main__':
    unittest.main()
