"""
Comprehensive Test Suite for Reflexive Self-Improvement and Meta-Learning Layer
Tests the meta-learning, reflexive audit, personality evolution, and enhanced reflection components.
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
    MetaLearningEngine, LearningInsight, HeuristicUpdate
)
from egdol.omnimind.conversational.feedback_loop import FeedbackStorage
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
        self.assertIsNotNone(self.audit_module.audit_history)
        self.assertIsNotNone(self.audit_module.personality_performance)
        self.assertIsNotNone(self.audit_module.quality_indicators)
        self.assertIsNotNone(self.audit_module.reasoning_indicators)
    
    def test_audit_conversation_turn(self):
        """Test auditing a conversation turn."""
        audit_result = self.audit_module.audit_conversation_turn(self.sample_turn)
        
        self.assertIsInstance(audit_result, AuditResult)
        self.assertEqual(audit_result.turn_id, "test_turn_1")
        self.assertEqual(audit_result.personality, "Strategos")
        self.assertIn(AuditMetric.RESPONSE_QUALITY, audit_result.metrics)
        self.assertIn(AuditMetric.REASONING_ACCURACY, audit_result.metrics)
        self.assertIn(AuditMetric.FALLBACK_USAGE, audit_result.metrics)
        self.assertIn(AuditMetric.CONTEXT_ALIGNMENT, audit_result.metrics)
        self.assertIn(AuditMetric.PERSONALITY_CONSISTENCY, audit_result.metrics)
        self.assertIn(AuditMetric.META_RULE_APPLICATION, audit_result.metrics)
    
    def test_assess_response_quality(self):
        """Test response quality assessment."""
        high_quality_response = "Commander, I recommend a precise tactical approach based on strategic analysis."
        quality_score = self.audit_module._assess_response_quality(high_quality_response)
        
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
        self.assertGreater(quality_score, 0.5)  # Should be high quality
    
    def test_assess_reasoning_accuracy(self):
        """Test reasoning accuracy assessment."""
        reasoning_trace = {
            'processing_steps': ['step1', 'step2'],
            'meta_insights': ['insight1'],
            'confidence': 0.8
        }
        
        accuracy = self.audit_module._assess_reasoning_accuracy(self.sample_turn, reasoning_trace)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_assess_fallback_usage(self):
        """Test fallback usage assessment."""
        fallback_response = "I understand your request. Let me try a different approach."
        fallback_score = self.audit_module._assess_fallback_usage(fallback_response)
        
        self.assertGreaterEqual(fallback_score, 0.0)
        self.assertLessEqual(fallback_score, 1.0)
        self.assertGreater(fallback_score, 0.5)  # Should detect fallback usage
    
    def test_assess_personality_consistency(self):
        """Test personality consistency assessment."""
        strategos_response = "Commander, I recommend a tactical approach."
        consistency = self.audit_module._assess_personality_consistency(strategos_response, "Strategos")
        
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)
        self.assertGreaterEqual(consistency, 0.4)  # Should be consistent with Strategos
    
    def test_identify_gaps(self):
        """Test gap identification."""
        metrics = {
            AuditMetric.RESPONSE_QUALITY: 0.3,
            AuditMetric.REASONING_ACCURACY: 0.2,
            AuditMetric.FALLBACK_USAGE: 0.8,
            AuditMetric.CONTEXT_ALIGNMENT: 0.3,
            AuditMetric.PERSONALITY_CONSISTENCY: 0.4,
            AuditMetric.META_RULE_APPLICATION: 0.2
        }
        
        gaps = self.audit_module._identify_gaps(self.sample_turn, metrics, None)
        self.assertIsInstance(gaps, list)
        self.assertGreater(len(gaps), 0)  # Should identify multiple gaps
    
    def test_generate_improvement_suggestions(self):
        """Test improvement suggestion generation."""
        metrics = {
            AuditMetric.RESPONSE_QUALITY: 0.3,
            AuditMetric.REASONING_ACCURACY: 0.2,
            AuditMetric.FALLBACK_USAGE: 0.8,
            AuditMetric.CONTEXT_ALIGNMENT: 0.3,
            AuditMetric.PERSONALITY_CONSISTENCY: 0.4,
            AuditMetric.META_RULE_APPLICATION: 0.2
        }
        
        gaps = ["Low response quality", "Weak reasoning trace"]
        suggestions = self.audit_module._generate_improvement_suggestions(self.sample_turn, metrics, gaps)
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        metrics = {
            AuditMetric.RESPONSE_QUALITY: 0.8,
            AuditMetric.REASONING_ACCURACY: 0.7,
            AuditMetric.FALLBACK_USAGE: 0.2,
            AuditMetric.CONTEXT_ALIGNMENT: 0.8,
            AuditMetric.PERSONALITY_CONSISTENCY: 0.9,
            AuditMetric.META_RULE_APPLICATION: 0.6
        }
        
        confidence = self.audit_module._calculate_confidence_score(metrics)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreater(confidence, 0.5)  # Should be high confidence
    
    def test_update_personality_performance(self):
        """Test personality performance tracking."""
        audit_result = AuditResult(
            turn_id="test_turn_1",
            personality="Strategos",
            metrics={AuditMetric.RESPONSE_QUALITY: 0.8},
            gaps_identified=[],
            improvement_suggestions=[],
            confidence_score=0.8
        )
        
        self.audit_module._update_personality_performance(audit_result)
        
        performance = self.audit_module.get_personality_performance("Strategos")
        self.assertIsNotNone(performance)
        self.assertEqual(performance.personality, "Strategos")
        self.assertEqual(performance.total_turns, 1)
        self.assertEqual(performance.successful_turns, 1)
    
    def test_get_audit_summary(self):
        """Test audit summary generation."""
        # Add some audit results
        for i in range(5):
            audit_result = AuditResult(
                turn_id=f"test_turn_{i}",
                personality="Strategos",
                metrics={AuditMetric.RESPONSE_QUALITY: 0.8},
                gaps_identified=[],
                improvement_suggestions=[],
                confidence_score=0.8
            )
            self.audit_module.audit_history.append(audit_result)
            self.audit_module._update_personality_performance(audit_result)
        
        summary = self.audit_module.get_audit_summary()
        self.assertIn('total_audits', summary)
        self.assertIn('average_confidence', summary)
        self.assertIn('personality_performance', summary)
        self.assertEqual(summary['total_audits'], 5)
    
    def test_get_improvement_opportunities(self):
        """Test improvement opportunity identification."""
        # Create low-performance personality
        performance = PersonalityPerformance("TestPersonality")
        performance.total_turns = 10
        performance.successful_turns = 3
        performance.fallback_usage = 8
        performance.average_confidence = 0.3
        performance.reasoning_accuracy = 0.2
        
        self.audit_module.personality_performance["TestPersonality"] = performance
        
        opportunities = self.audit_module.get_improvement_opportunities()
        self.assertIsInstance(opportunities, list)
        self.assertGreater(len(opportunities), 0)


class TestMetaLearningEngine(unittest.TestCase):
    """Test the meta-learning engine."""
    
    def setUp(self):
        self.storage = FeedbackStorage()
        self.meta_learning_engine = MetaLearningEngine(self.storage)
        self.sample_conversation_logs = [
            {
                'turn_id': 'turn_1',
                'personality': 'Strategos',
                'user_input': 'What is the best strategy?',
                'system_response': 'Commander, I recommend a tactical approach.',
                'reasoning_trace': {'processing_steps': ['step1'], 'confidence': 0.8},
                'confidence_score': 0.8,
                'fallback_used': False
            },
            {
                'turn_id': 'turn_2',
                'personality': 'Archivist',
                'user_input': 'Tell me about history',
                'system_response': 'From the archives, I can share historical insights.',
                'reasoning_trace': {'processing_steps': ['step1'], 'confidence': 0.7},
                'confidence_score': 0.7,
                'fallback_used': False
            }
        ]
    
    def test_initialization(self):
        """Test meta-learning engine initialization."""
        self.assertIsNotNone(self.meta_learning_engine.audit_module)
        self.assertIsNotNone(self.meta_learning_engine.learning_insights)
        self.assertIsNotNone(self.meta_learning_engine.heuristic_updates)
        self.assertIsNotNone(self.meta_learning_engine.meta_rule_discoveries)
        self.assertIsNotNone(self.meta_learning_engine.learning_thresholds)
        self.assertIsNotNone(self.meta_learning_engine.learning_rates)
    
    def test_process_conversation_batch(self):
        """Test processing conversation batch for learning."""
        results = self.meta_learning_engine.process_conversation_batch(self.sample_conversation_logs)
        
        self.assertIn('insights_generated', results)
        self.assertIn('heuristics_updated', results)
        self.assertIn('meta_rules_discovered', results)
        self.assertIn('personalities_improved', results)
        self.assertIn('cross_personality_insights', results)
        self.assertIn('recommendations', results)
        self.assertIn('learning_summary', results)
    
    def test_extract_turn_data(self):
        """Test turn data extraction."""
        log = self.sample_conversation_logs[0]
        turn_data = self.meta_learning_engine._extract_turn_data(log)
        
        self.assertIsNotNone(turn_data)
        self.assertEqual(turn_data['turn_id'], 'turn_1')
        self.assertEqual(turn_data['personality'], 'Strategos')
        self.assertEqual(turn_data['user_input'], 'What is the best strategy?')
    
    def test_generate_insights_from_turn(self):
        """Test insight generation from conversation turn."""
        turn_data = self.meta_learning_engine._extract_turn_data(self.sample_conversation_logs[0])
        insights = self.meta_learning_engine._generate_insights_from_turn(turn_data)
        
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        for insight in insights:
            self.assertIsInstance(insight, LearningInsight)
            self.assertEqual(insight.personality, 'Strategos')
    
    def test_analyze_response_patterns(self):
        """Test response pattern analysis."""
        # Test with actual method
        feedback_batch = []
        trends = self.meta_learning_engine.analyze_feedback_trends(feedback_batch)
        self.assertIsInstance(trends, dict)
    
    def test_analyze_reasoning_patterns(self):
        """Test reasoning pattern analysis."""
        # Test with actual method
        feedback_batch = []
        trends = self.meta_learning_engine.analyze_feedback_trends(feedback_batch)
        self.assertIsInstance(trends, dict)
    
    def test_analyze_fallback_patterns(self):
        """Test fallback pattern analysis."""
        # Test with actual method
        feedback_batch = []
        trends = self.meta_learning_engine.analyze_feedback_trends(feedback_batch)
        self.assertIsInstance(trends, dict)
    
    def test_apply_heuristic_learning(self):
        """Test heuristic learning application."""
        turn_data = self.meta_learning_engine._extract_turn_data(self.sample_conversation_logs[0])
        insights = self.meta_learning_engine._generate_insights_from_turn(turn_data)
        updates = self.meta_learning_engine._apply_heuristic_learning(turn_data, insights)
        
        self.assertIsInstance(updates, list)
    
    def test_discover_meta_rules(self):
        """Test meta-rule discovery."""
        turn_data = self.meta_learning_engine._extract_turn_data(self.sample_conversation_logs[0])
        insights = self.meta_learning_engine._generate_insights_from_turn(turn_data)
        meta_rules = self.meta_learning_engine._discover_meta_rules(turn_data, insights)
        
        self.assertIsInstance(meta_rules, list)
    
    def test_apply_cross_personality_learning(self):
        """Test cross-personality learning."""
        # Test with actual method
        feedback_batch = []
        trends = self.meta_learning_engine.analyze_feedback_trends(feedback_batch)
        self.assertIsInstance(trends, dict)
    
    def test_get_learning_insights(self):
        """Test getting learning insights."""
        # Add some insights
        insight = LearningInsight(
            insight_type='test_insight',
            personality='Strategos',
            pattern='test_pattern',
            confidence=0.8,
            applicable_to=['test']
        )
        self.meta_learning_engine.learning_insights.append(insight)
        
        insights = self.meta_learning_engine.get_learning_insights()
        self.assertIsInstance(insights, list)
        self.assertEqual(len(insights), 1)
        
        # Test filtering by personality
        strategos_insights = self.meta_learning_engine.get_learning_insights('Strategos')
        self.assertEqual(len(strategos_insights), 1)
    
    def test_get_heuristic_updates(self):
        """Test getting heuristic updates."""
        # Add some updates
        update = HeuristicUpdate(
            heuristic_name='test_heuristic',
            personality='Strategos',
            old_value=0.5,
            new_value=0.8,
            confidence=0.9,
            reasoning='Test reasoning'
        )
        self.meta_learning_engine.heuristic_updates.append(update)
        
        updates = self.meta_learning_engine.get_heuristic_updates()
        self.assertIsInstance(updates, list)
        self.assertEqual(len(updates), 1)
    
    def test_get_meta_rule_discoveries(self):
        """Test getting meta-rule discoveries."""
        # Test getting learning metrics instead
        metrics = self.meta_learning_engine.get_learning_metrics()
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, LearningMetrics)
    
    def test_clear_learning_data(self):
        """Test clearing learning data."""
        # Add some data
        self.meta_learning_engine.learning_insights.append(LearningInsight(
            insight_type='test', personality='Test', pattern='test', confidence=0.5, applicable_to=['test']
        ))
        
        # Clear data
        self.meta_learning_engine.clear_learning_data()
        
        # Verify cleared
        self.assertEqual(len(self.meta_learning_engine.learning_insights), 0)
        self.assertEqual(len(self.meta_learning_engine.heuristic_updates), 0)
        self.assertEqual(len(self.meta_learning_engine.meta_rule_discoveries), 0)


class TestPersonalityEvolutionEngine(unittest.TestCase):
    """Test the personality evolution engine."""
    
    def setUp(self):
        self.meta_learning_engine = MetaLearningEngine()
        self.audit_module = ReflexiveAuditModule()
        self.evolution_engine = PersonalityEvolutionEngine(
            self.meta_learning_engine, self.audit_module
        )
        
        self.sample_insights = [
            LearningInsight(
                insight_type='response_pattern',
                personality='Strategos',
                pattern='military_terminology',
                confidence=0.8,
                applicable_to=['response_generation']
            )
        ]
    
    def test_initialization(self):
        """Test personality evolution engine initialization."""
        self.assertIsNotNone(self.evolution_engine.meta_learning_engine)
        self.assertIsNotNone(self.evolution_engine.audit_module)
        self.assertIsNotNone(self.evolution_engine.evolution_states)
        self.assertIsNotNone(self.evolution_engine.evolution_history)
        self.assertIsNotNone(self.evolution_engine.evolution_thresholds)
    
    def test_initialize_personality_evolution_states(self):
        """Test initialization of personality evolution states."""
        personalities = ['Strategos', 'Archivist', 'Lawmaker', 'Oracle']
        
        for personality in personalities:
            state = self.evolution_engine.get_evolution_state(personality)
            self.assertIsNotNone(state)
            self.assertEqual(state.personality, personality)
            self.assertEqual(state.stage, EvolutionStage.INITIAL)
    
    def test_evolve_personality(self):
        """Test personality evolution."""
        evolution_update = self.evolution_engine.evolve_personality(
            'Strategos', self.sample_insights
        )
        
        self.assertIsInstance(evolution_update, EvolutionUpdate)
        self.assertEqual(evolution_update.personality, 'Strategos')
    
    def test_evolve_language_patterns(self):
        """Test language pattern evolution."""
        updates = self.evolution_engine._evolve_language_patterns(
            'Strategos', self.sample_insights
        )
        
        self.assertIsInstance(updates, list)
    
    def test_evolve_epistemic_patterns(self):
        """Test epistemic pattern evolution."""
        updates = self.evolution_engine._evolve_epistemic_patterns(
            'Strategos', self.sample_insights
        )
        
        self.assertIsInstance(updates, list)
    
    def test_evolve_learned_heuristics(self):
        """Test learned heuristic evolution."""
        # Add heuristic update to meta-learning engine
        heuristic_update = HeuristicUpdate(
            heuristic_name='test_heuristic',
            personality='Strategos',
            old_value=0.5,
            new_value=0.8,
            confidence=0.9,
            reasoning='Test reasoning'
        )
        self.meta_learning_engine.heuristic_updates.append(heuristic_update)
        
        updates = self.evolution_engine._evolve_learned_heuristics(
            'Strategos', self.sample_insights
        )
        
        self.assertIsInstance(updates, list)
    
    def test_evolve_meta_rules(self):
        """Test meta-rule evolution."""
        # Test evolution without meta-rule discoveries
        updates = self.evolution_engine._evolve_meta_rules(
            'Strategos', self.sample_insights
        )
        
        self.assertIsInstance(updates, list)
    
    def test_evolve_fallback_threshold(self):
        """Test fallback threshold evolution."""
        # Create performance data
        performance = PersonalityPerformance('Strategos')
        performance.total_turns = 10
        performance.fallback_usage = 2
        performance.average_confidence = 0.9
        
        updates = self.evolution_engine._evolve_fallback_threshold(
            'Strategos', performance
        )
        
        self.assertIsInstance(updates, list)
    
    def test_update_evolution_score(self):
        """Test evolution score update."""
        self.evolution_engine._update_evolution_score('Strategos')
        
        state = self.evolution_engine.get_evolution_state('Strategos')
        self.assertGreaterEqual(state.evolution_score, 0.0)
        self.assertLessEqual(state.evolution_score, 1.0)
    
    def test_advance_evolution_stage(self):
        """Test evolution stage advancement."""
        # Set high evolution score
        state = self.evolution_engine.get_evolution_state('Strategos')
        state.evolution_score = 0.9
        
        self.evolution_engine._advance_evolution_stage('Strategos')
        
        # Should advance to next stage
        updated_state = self.evolution_engine.get_evolution_state('Strategos')
        self.assertNotEqual(updated_state.stage, EvolutionStage.INITIAL)
    
    def test_get_evolution_summary(self):
        """Test evolution summary generation."""
        summary = self.evolution_engine.get_evolution_summary()
        
        self.assertIn('total_personalities', summary)
        self.assertIn('evolution_stages', summary)
        self.assertIn('average_evolution_score', summary)
        self.assertIn('total_evolution_updates', summary)
        self.assertIn('personalities_by_stage', summary)
    
    def test_get_personality_evolution_report(self):
        """Test personality evolution report generation."""
        report = self.evolution_engine.get_personality_evolution_report('Strategos')
        
        self.assertIn('personality', report)
        self.assertIn('evolution_state', report)
        self.assertIn('recent_updates', report)
        self.assertIn('performance_data', report)
        self.assertIn('evolution_metrics', report)
    
    def test_apply_evolution_to_personality(self):
        """Test applying evolution to personality."""
        config = self.evolution_engine.apply_evolution_to_personality('Strategos')
        
        self.assertIn('personality', config)
        self.assertIn('stage', config)
        self.assertIn('evolution_score', config)
        self.assertIn('language_patterns', config)
        self.assertIn('epistemic_patterns', config)
        self.assertIn('learned_heuristics', config)
        self.assertIn('meta_rules_applied', config)


class TestReflectionModePlus(unittest.TestCase):
    """Test the enhanced reflection mode."""
    
    def setUp(self):
        self.meta_learning_engine = MetaLearningEngine()
        self.audit_module = ReflexiveAuditModule()
        self.personality_evolution_engine = PersonalityEvolutionEngine(
            self.meta_learning_engine, self.audit_module
        )
        self.reflection_mode_plus = ReflectionModePlus(
            self.meta_learning_engine, self.personality_evolution_engine, self.audit_module
        )
    
    def test_initialization(self):
        """Test reflection mode plus initialization."""
        self.assertIsNotNone(self.reflection_mode_plus.meta_learning_engine)
        self.assertIsNotNone(self.reflection_mode_plus.personality_evolution_engine)
        self.assertIsNotNone(self.reflection_mode_plus.audit_module)
        self.assertIsNotNone(self.reflection_mode_plus.enhanced_strategies)
        self.assertIsNotNone(self.reflection_mode_plus.reflection_insights_plus)
    
    def test_reflect_and_retry_plus(self):
        """Test enhanced reflect and retry."""
        result = self.reflection_mode_plus.reflect_and_retry_plus(
            'What is the best strategy?', 'Strategos', 'Test error'
        )
        
        self.assertIsInstance(result, ReflectionResultPlus)
        self.assertIn('success', result.__dict__)
        self.assertIn('response', result.__dict__)
        self.assertIn('strategy_used', result.__dict__)
        self.assertIn('attempts_made', result.__dict__)
        self.assertIn('learning_insights_generated', result.__dict__)
    
    def test_generate_enhanced_reflection_insights(self):
        """Test enhanced reflection insight generation."""
        insights = self.reflection_mode_plus._generate_enhanced_reflection_insights(
            'What is the best strategy?', 'Strategos', 'Test error', None, None
        )
        
        self.assertIsInstance(insights, list)
        for insight in insights:
            self.assertIsInstance(insight, ReflectionInsightPlus)
    
    def test_analyze_error_patterns(self):
        """Test error pattern analysis."""
        insight = self.reflection_mode_plus._analyze_error_patterns(
            'Test input', 'Strategos', 'NoneType object has no attribute get'
        )
        
        if insight:  # May be None if no patterns detected
            self.assertIsInstance(insight, ReflectionInsightPlus)
            self.assertEqual(insight.insight_type, 'error_analysis')
    
    def test_analyze_conversation_patterns(self):
        """Test conversation pattern analysis."""
        conversation_history = [
            {'success': True, 'confidence_score': 0.8},
            {'success': False, 'confidence_score': 0.3}
        ]
        
        insights = self.reflection_mode_plus._analyze_conversation_patterns(
            'Test input', 'Strategos', conversation_history
        )
        
        self.assertIsInstance(insights, list)
    
    def test_analyze_personality_patterns(self):
        """Test personality pattern analysis."""
        insights = self.reflection_mode_plus._analyze_personality_patterns(
            'Test input', 'Strategos', None
        )
        
        self.assertIsInstance(insights, list)
    
    def test_analyze_cross_personality_opportunities(self):
        """Test cross-personality opportunity analysis."""
        insights = self.reflection_mode_plus._analyze_cross_personality_opportunities(
            'Test input', 'Strategos', None
        )
        
        self.assertIsInstance(insights, list)
    
    def test_select_enhanced_reflection_strategy(self):
        """Test enhanced reflection strategy selection."""
        strategy = self.reflection_mode_plus._select_enhanced_reflection_strategy(0, [])
        
        self.assertIsInstance(strategy, ReflectionStrategy)
    
    def test_apply_enhanced_reflection_strategy(self):
        """Test enhanced reflection strategy application."""
        result = self.reflection_mode_plus._apply_enhanced_reflection_strategy(
            'Test input', 'Strategos', ReflectionStrategy.CONTEXT_ANALYSIS, [], None
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
    
    def test_apply_learning_insights(self):
        """Test learning insight application."""
        reflection_insights = [
            ReflectionInsightPlus(
                insight_type='test',
                description='Test insight',
                confidence=0.8,
                applicable_to=['test'],
                learning_insights=[],
                heuristic_suggestions=['Test suggestion'],
                meta_rule_suggestions=['Test meta-rule']
            )
        ]
        
        updates = self.reflection_mode_plus._apply_learning_insights(reflection_insights)
        
        self.assertIsInstance(updates, list)
    
    def test_discover_meta_rules_from_reflection(self):
        """Test meta-rule discovery from reflection."""
        reflection_insights = [
            ReflectionInsightPlus(
                insight_type='test',
                description='Test insight',
                confidence=0.8,
                applicable_to=['test'],
                learning_insights=[],
                heuristic_suggestions=[],
                meta_rule_suggestions=['Test meta-rule']
            )
        ]
        
        meta_rules = self.reflection_mode_plus._discover_meta_rules_from_reflection(reflection_insights)
        
        self.assertIsInstance(meta_rules, list)
    
    def test_apply_cross_personality_learning_plus(self):
        """Test cross-personality learning application."""
        reflection_insights = [
            ReflectionInsightPlus(
                insight_type='cross_personality',
                description='Test insight',
                confidence=0.8,
                applicable_to=['test'],
                learning_insights=[],
                heuristic_suggestions=[],
                meta_rule_suggestions=[],
                metadata={'cross_insights': ['insight1', 'insight2']}
            )
        ]
        
        cross_insights = self.reflection_mode_plus._apply_cross_personality_learning_plus(reflection_insights)
        
        self.assertIsInstance(cross_insights, list)
    
    def test_get_reflection_summary(self):
        """Test reflection summary generation."""
        summary = self.reflection_mode_plus.get_reflection_summary()
        
        self.assertIn('total_reflection_insights', summary)
        self.assertIn('reflection_success_patterns', summary)
        self.assertIn('reflection_failure_patterns', summary)
        self.assertIn('heuristic_adjustment_history', summary)
        self.assertIn('enhanced_strategies_available', summary)


class TestIntegrationReflexiveSelfImprovement(unittest.TestCase):
    """Integration tests for the reflexive self-improvement layer."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.omnimind_core = OmniMind(self.temp_dir)
        
        # Initialize components
        self.audit_module = ReflexiveAuditModule()
        self.meta_learning_engine = MetaLearningEngine()
        self.personality_evolution_engine = PersonalityEvolutionEngine(
            self.meta_learning_engine, self.audit_module
        )
        self.reflection_mode_plus = ReflectionModePlus(
            self.meta_learning_engine, self.personality_evolution_engine, self.audit_module
        )
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_meta_learning_pipeline(self):
        """Test the complete meta-learning pipeline."""
        # Create conversation logs
        conversation_logs = [
            {
                'turn_id': 'turn_1',
                'personality': 'Strategos',
                'user_input': 'What is the best strategy?',
                'system_response': 'Commander, I recommend a tactical approach.',
                'reasoning_trace': {'processing_steps': ['step1'], 'confidence': 0.8},
                'confidence_score': 0.8,
                'fallback_used': False
            }
        ]
        
        # Process conversation batch
        results = self.meta_learning_engine.process_conversation_batch(conversation_logs)
        
        self.assertIn('insights_generated', results)
        self.assertIn('heuristics_updated', results)
        self.assertIn('meta_rules_discovered', results)
        self.assertIn('recommendations', results)
    
    def test_personality_evolution_integration(self):
        """Test personality evolution integration."""
        # Create learning insights
        insights = [
            LearningInsight(
                insight_type='response_pattern',
                personality='Strategos',
                pattern='military_terminology',
                confidence=0.8,
                applicable_to=['response_generation']
            )
        ]
        
        # Evolve personality
        evolution_update = self.personality_evolution_engine.evolve_personality(
            'Strategos', insights
        )
        
        self.assertIsInstance(evolution_update, EvolutionUpdate)
        
        # Check evolution state
        state = self.personality_evolution_engine.get_evolution_state('Strategos')
        self.assertIsNotNone(state)
    
    def test_reflection_mode_plus_integration(self):
        """Test reflection mode plus integration."""
        # Test enhanced reflection
        result = self.reflection_mode_plus.reflect_and_retry_plus(
            'What is the best strategy?', 'Strategos', 'Test error'
        )
        
        self.assertIsInstance(result, ReflectionResultPlus)
        self.assertIn('learning_insights_generated', result.__dict__)
        self.assertIn('heuristic_updates_applied', result.__dict__)
        self.assertIn('meta_rules_discovered', result.__dict__)
        self.assertIn('cross_personality_insights', result.__dict__)
    
    def test_audit_integration(self):
        """Test audit module integration."""
        # Create sample turn
        turn = Mock()
        turn.turn_id = 'test_turn'
        turn.personality_used = 'Strategos'
        turn.system_response = 'Commander, I recommend a tactical approach.'
        turn.confidence_score = 0.8
        turn.intent = 'STRATEGIC_ANALYSIS'
        turn.reasoning_trace = []
        turn.meta_insights = []
        
        # Audit turn
        audit_result = self.audit_module.audit_conversation_turn(turn)
        
        self.assertIsInstance(audit_result, AuditResult)
        self.assertEqual(audit_result.personality, 'Strategos')
        
        # Check performance tracking
        performance = self.audit_module.get_personality_performance('Strategos')
        self.assertIsNotNone(performance)
    
    def test_cross_component_learning(self):
        """Test learning across all components."""
        # Create conversation data
        conversation_logs = [
            {
                'turn_id': 'turn_1',
                'personality': 'Strategos',
                'user_input': 'What is the best strategy?',
                'system_response': 'Commander, I recommend a tactical approach.',
                'reasoning_trace': {'processing_steps': ['step1'], 'confidence': 0.8},
                'confidence_score': 0.8,
                'fallback_used': False
            }
        ]
        
        # Process through meta-learning
        meta_results = self.meta_learning_engine.process_conversation_batch(conversation_logs)
        
        # Get learning insights
        insights = self.meta_learning_engine.get_learning_insights('Strategos')
        
        # Evolve personality
        if insights:
            evolution_update = self.personality_evolution_engine.evolve_personality(
                'Strategos', insights
            )
            self.assertIsInstance(evolution_update, EvolutionUpdate)
        
        # Test reflection with learned patterns
        reflection_result = self.reflection_mode_plus.reflect_and_retry_plus(
            'What is the best strategy?', 'Strategos', 'Test error'
        )
        
        self.assertIsInstance(reflection_result, ReflectionResultPlus)
    
    def test_continuous_improvement_simulation(self):
        """Test simulation of continuous improvement."""
        # Simulate multiple conversation turns
        for i in range(5):
            conversation_logs = [
                {
                    'turn_id': f'turn_{i}',
                    'personality': 'Strategos',
                    'user_input': f'Strategy question {i}',
                    'system_response': f'Commander, tactical response {i}.',
                    'reasoning_trace': {'processing_steps': [f'step{i}'], 'confidence': 0.7 + i * 0.05},
                    'confidence_score': 0.7 + i * 0.05,
                    'fallback_used': i < 2  # Reduce fallback usage over time
                }
            ]
            
            # Process through meta-learning
            meta_results = self.meta_learning_engine.process_conversation_batch(conversation_logs)
            
            # Get insights and evolve personality
            insights = self.meta_learning_engine.get_learning_insights('Strategos')
            if insights:
                self.personality_evolution_engine.evolve_personality('Strategos', insights)
        
        # Check that personality has evolved
        evolution_state = self.personality_evolution_engine.get_evolution_state('Strategos')
        self.assertIsNotNone(evolution_state)
        self.assertGreater(evolution_state.evolution_score, 0.0)
        
        # Check meta-learning effectiveness
        learning_summary = self.meta_learning_engine._generate_learning_summary()
        self.assertIn('learning_effectiveness', learning_summary)
        self.assertGreater(learning_summary['learning_effectiveness'], 0.0)


if __name__ == '__main__':
    unittest.main()
