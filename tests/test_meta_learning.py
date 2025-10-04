"""
Test Meta-Learning Engine
Tests for feedback loop, learning updates, and system adaptation.
"""

import unittest
import tempfile
import os
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta

from egdol.omnimind.conversational.feedback_loop import (
    FeedbackStorage, FeedbackCollector, UserFeedback, 
    FeedbackType, FeedbackSentiment, LearningUpdate
)
from egdol.omnimind.conversational.meta_learning_engine import (
    MetaLearningEngine, PersonalityProfile, LearningMetrics
)


class TestFeedbackStorage(unittest.TestCase):
    """Test feedback storage system."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.storage = FeedbackStorage(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_store_and_retrieve_feedback(self):
        """Test storing and retrieving feedback."""
        feedback = UserFeedback(
            feedback_id="test_001",
            session_id="session_001",
            response_id="response_001",
            feedback_type=FeedbackType.RATING,
            rating=4.5,
            sentiment=FeedbackSentiment.POSITIVE,
            personality="Strategos"
        )
        
        # Store feedback
        self.storage.store_feedback(feedback)
        
        # Retrieve feedback
        retrieved = self.storage.get_feedback_batch(10)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].feedback_id, "test_001")
        self.assertEqual(retrieved[0].rating, 4.5)
    
    def test_store_learning_update(self):
        """Test storing learning updates."""
        feedback = UserFeedback(
            feedback_id="test_002",
            session_id="session_002",
            response_id="response_002",
            feedback_type=FeedbackType.TEXT,
            text="Great response!",
            sentiment=FeedbackSentiment.POSITIVE
        )
        
        update = LearningUpdate(
            update_id="update_001",
            feedback_batch=[feedback],
            personality_adjustments={"Strategos": {"success_rate_boost": 0.1}},
            depth_preference_adjustments={"Strategos_depth_weight": 0.05},
            confidence_calibration_adjustments={"Strategos_confidence_threshold": 0.02},
            synthesis_improvements={"Strategos_synthesis_quality": {"style_weight_adjustment": 0.1}}
        )
        
        # Store update
        self.storage.store_learning_update(update)
        
        # Retrieve updates
        updates = self.storage.get_learning_history(10)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].update_id, "update_001")


class TestFeedbackCollector(unittest.TestCase):
    """Test feedback collection system."""
    
    def setUp(self):
        """Set up test collector."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.storage = FeedbackStorage(self.temp_db.name)
        self.collector = FeedbackCollector(self.storage)
    
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_collect_rating_feedback(self):
        """Test collecting rating feedback."""
        feedback_data = {
            "rating": 4.0,
            "personality": "Strategos",
            "depth_preference": "standard"
        }
        
        feedback = self.collector.collect_feedback(
            session_id="session_001",
            response_id="response_001",
            feedback_data=feedback_data
        )
        
        self.assertEqual(feedback.feedback_type, FeedbackType.RATING)
        self.assertEqual(feedback.rating, 4.0)
        self.assertEqual(feedback.sentiment, FeedbackSentiment.POSITIVE)
        self.assertEqual(feedback.personality, "Strategos")
    
    def test_collect_text_feedback(self):
        """Test collecting text feedback."""
        feedback_data = {
            "text": "This response was confusing and unclear",
            "personality": "Archivist"
        }
        
        feedback = self.collector.collect_feedback(
            session_id="session_002",
            response_id="response_002",
            feedback_data=feedback_data
        )
        
        self.assertEqual(feedback.feedback_type, FeedbackType.TEXT)
        self.assertEqual(feedback.text, "This response was confusing and unclear")
        self.assertEqual(feedback.sentiment, FeedbackSentiment.NEGATIVE)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        # Test positive sentiment
        positive_data = {"text": "Great response! Very helpful and accurate."}
        feedback = self.collector.collect_feedback("session_003", "response_003", positive_data)
        self.assertEqual(feedback.sentiment, FeedbackSentiment.POSITIVE)
        
        # Test negative sentiment
        negative_data = {"text": "Bad response. Wrong and confusing."}
        feedback = self.collector.collect_feedback("session_004", "response_004", negative_data)
        self.assertEqual(feedback.sentiment, FeedbackSentiment.NEGATIVE)
        
        # Test neutral sentiment
        neutral_data = {"text": "The response was okay."}
        feedback = self.collector.collect_feedback("session_005", "response_005", neutral_data)
        self.assertEqual(feedback.sentiment, FeedbackSentiment.NEUTRAL)
    
    def test_feedback_summary(self):
        """Test feedback summary generation."""
        # Add multiple feedback items
        feedback_data_1 = {"rating": 4.0, "personality": "Strategos"}
        feedback_data_2 = {"rating": 2.0, "personality": "Strategos"}
        feedback_data_3 = {"rating": 5.0, "personality": "Archivist"}
        
        self.collector.collect_feedback("session_001", "response_001", feedback_data_1)
        self.collector.collect_feedback("session_001", "response_002", feedback_data_2)
        self.collector.collect_feedback("session_001", "response_003", feedback_data_3)
        
        summary = self.collector.get_feedback_summary("session_001")
        
        self.assertEqual(summary["total"], 3)
        self.assertAlmostEqual(summary["average_rating"], 3.67, places=1)
        self.assertIn("POSITIVE", summary["sentiment_distribution"])


class TestMetaLearningEngine(unittest.TestCase):
    """Test meta-learning engine."""
    
    def setUp(self):
        """Set up test engine."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.storage = FeedbackStorage(self.temp_db.name)
        self.engine = MetaLearningEngine(self.storage)
    
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_personality_profile_initialization(self):
        """Test personality profile initialization."""
        profiles = self.engine.get_personality_profiles()
        
        self.assertIn("Strategos", profiles)
        self.assertIn("Archivist", profiles)
        self.assertIn("Lawmaker", profiles)
        self.assertIn("Oracle", profiles)
        
        # Check Strategos has tactical focus
        strategos = profiles["Strategos"]
        self.assertGreater(strategos.synthesis_style_weights["tactical"], 0.7)
        
        # Check Archivist has historical focus
        archivist = profiles["Archivist"]
        self.assertGreater(archivist.synthesis_style_weights["historical"], 0.7)
    
    def test_feedback_trend_analysis(self):
        """Test feedback trend analysis."""
        # Create sample feedback
        feedback_batch = [
            UserFeedback(
                feedback_id="f1", session_id="s1", response_id="r1",
                feedback_type=FeedbackType.RATING, rating=4.0,
                sentiment=FeedbackSentiment.POSITIVE, personality="Strategos"
            ),
            UserFeedback(
                feedback_id="f2", session_id="s2", response_id="r2",
                feedback_type=FeedbackType.RATING, rating=2.0,
                sentiment=FeedbackSentiment.NEGATIVE, personality="Strategos"
            ),
            UserFeedback(
                feedback_id="f3", session_id="s3", response_id="r3",
                feedback_type=FeedbackType.RATING, rating=5.0,
                sentiment=FeedbackSentiment.POSITIVE, personality="Archivist"
            )
        ]
        
        trends = self.engine.analyze_feedback_trends(feedback_batch)
        
        self.assertIn("trends", trends)
        self.assertIn("recommendations", trends)
        self.assertIn("overall_health", trends)
        
        # Check Strategos trends
        strategos_trends = trends["trends"]["Strategos"]
        self.assertIn("success_rate", strategos_trends)
        self.assertIn("depth_misalignment", strategos_trends)
        self.assertIn("confidence_miscalibration", strategos_trends)
    
    def test_learning_update_generation(self):
        """Test learning update generation."""
        # Create feedback with negative sentiment
        feedback_batch = [
            UserFeedback(
                feedback_id="f1", session_id="s1", response_id="r1",
                feedback_type=FeedbackType.RATING, rating=2.0,
                sentiment=FeedbackSentiment.NEGATIVE, personality="Strategos",
                depth_preference="deep", confidence_feedback=0.3
            ),
            UserFeedback(
                feedback_id="f2", session_id="s2", response_id="r2",
                feedback_type=FeedbackType.TEXT, text="Too shallow",
                sentiment=FeedbackSentiment.NEGATIVE, personality="Strategos",
                depth_preference="deep"
            )
        ]
        
        update = self.engine.generate_learning_update(feedback_batch)
        
        self.assertIsInstance(update, LearningUpdate)
        self.assertEqual(update.feedback_batch, feedback_batch)
        self.assertIn("Strategos", update.personality_adjustments)
        self.assertIn("Strategos_depth_weight", update.depth_preference_adjustments)
        self.assertIn("Strategos_confidence_threshold", update.confidence_calibration_adjustments)
    
    def test_learning_trigger_conditions(self):
        """Test learning trigger conditions."""
        # Test with insufficient feedback
        self.assertFalse(self.engine.should_trigger_learning(5))
        
        # Add negative feedback
        negative_feedback = [
            UserFeedback(
                feedback_id=f"f{i}", session_id=f"s{i}", response_id=f"r{i}",
                feedback_type=FeedbackType.RATING, rating=2.0,
                sentiment=FeedbackSentiment.NEGATIVE, personality="Strategos"
            ) for i in range(3)
        ]
        
        for feedback in negative_feedback:
            self.storage.store_feedback(feedback)
        
        # Should trigger learning with enough negative feedback
        self.assertTrue(self.engine.should_trigger_learning(10))
    
    def test_learning_metrics_tracking(self):
        """Test learning metrics tracking."""
        metrics = self.engine.get_learning_metrics()
        
        self.assertIsInstance(metrics, LearningMetrics)
        self.assertEqual(metrics.total_feedback, 0)
        self.assertEqual(metrics.average_rating, 0.0)
        self.assertIsInstance(metrics.personality_accuracy, dict)
        self.assertIsInstance(metrics.depth_alignment, dict)


class TestMetaLearningIntegration(unittest.TestCase):
    """Test meta-learning integration with other components."""
    
    def setUp(self):
        """Set up integration test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.storage = FeedbackStorage(self.temp_db.name)
        self.collector = FeedbackCollector(self.storage)
        self.engine = MetaLearningEngine(self.storage)
    
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
    
    def test_end_to_end_learning_cycle(self):
        """Test complete learning cycle."""
        # 1. Collect feedback
        feedback_data = {
            "rating": 2.0,
            "text": "Response was too shallow",
            "personality": "Strategos",
            "depth_preference": "deep",
            "confidence_feedback": 0.3
        }
        
        feedback = self.collector.collect_feedback(
            session_id="session_001",
            response_id="response_001",
            feedback_data=feedback_data
        )
        
        # 2. Generate learning update
        feedback_batch = self.storage.get_feedback_batch(10)
        update = self.engine.generate_learning_update(feedback_batch)
        
        # 3. Apply learning update
        success = self.engine.apply_learning_update(update)
        self.assertTrue(success)
        
        # 4. Verify learning metrics updated
        metrics = self.engine.get_learning_metrics()
        self.assertGreater(metrics.total_feedback, 0)
    
    def test_learning_improves_system_behavior(self):
        """Test that learning improves system behavior."""
        # Initial state
        initial_profiles = self.engine.get_personality_profiles()
        initial_strategos = initial_profiles["Strategos"]
        
        # Add negative feedback for Strategos
        negative_feedback = [
            UserFeedback(
                feedback_id=f"f{i}", session_id=f"s{i}", response_id=f"r{i}",
                feedback_type=FeedbackType.RATING, rating=2.0,
                sentiment=FeedbackSentiment.NEGATIVE, personality="Strategos",
                depth_preference="deep"
            ) for i in range(5)
        ]
        
        for feedback in negative_feedback:
            self.storage.store_feedback(feedback)
        
        # Generate and apply learning update
        feedback_batch = self.storage.get_feedback_batch(10)
        update = self.engine.generate_learning_update(feedback_batch)
        self.engine.apply_learning_update(update)
        
        # Check that system adapted
        updated_profiles = self.engine.get_personality_profiles()
        updated_strategos = updated_profiles["Strategos"]
        
        # Should have improved adaptation rate
        self.assertGreaterEqual(updated_strategos.adaptation_rate, initial_strategos.adaptation_rate)
    
    def test_learning_preserves_determinism(self):
        """Test that learning preserves deterministic behavior."""
        # Create identical feedback
        feedback_data = {
            "rating": 4.0,
            "personality": "Strategos",
            "depth_preference": "standard"
        }
        
        # Collect feedback twice
        feedback1 = self.collector.collect_feedback("session_001", "response_001", feedback_data)
        feedback2 = self.collector.collect_feedback("session_002", "response_002", feedback_data)
        
        # Both should have same sentiment analysis
        self.assertEqual(feedback1.sentiment, feedback2.sentiment)
        self.assertEqual(feedback1.rating, feedback2.rating)
    
    def test_learning_handles_edge_cases(self):
        """Test learning handles edge cases gracefully."""
        # Test with empty feedback
        trends = self.engine.analyze_feedback_trends([])
        self.assertEqual(trends["trends"], {})
        self.assertEqual(trends["recommendations"], [])
        
        # Test with malformed feedback
        malformed_feedback = UserFeedback(
            feedback_id="malformed", session_id="s1", response_id="r1",
            feedback_type=FeedbackType.RATING, rating=None,
            sentiment=None, personality=None
        )
        
        # Should not crash
        trends = self.engine.analyze_feedback_trends([malformed_feedback])
        self.assertIsInstance(trends, dict)
    
    def test_learning_metrics_accuracy(self):
        """Test learning metrics accuracy."""
        # Add various feedback
        feedback_items = [
            {"rating": 5.0, "personality": "Strategos"},
            {"rating": 4.0, "personality": "Strategos"},
            {"rating": 3.0, "personality": "Archivist"},
            {"rating": 2.0, "personality": "Lawmaker"},
            {"rating": 1.0, "personality": "Oracle"}
        ]
        
        for i, feedback_data in enumerate(feedback_items):
            self.collector.collect_feedback(f"session_{i}", f"response_{i}", feedback_data)
        
        # Generate learning update
        feedback_batch = self.storage.get_feedback_batch(10)
        update = self.engine.generate_learning_update(feedback_batch)
        self.engine.apply_learning_update(update)
        
        # Check metrics
        metrics = self.engine.get_learning_metrics()
        self.assertEqual(metrics.total_feedback, 5)
        self.assertAlmostEqual(metrics.average_rating, 3.0, places=1)


if __name__ == "__main__":
    unittest.main()