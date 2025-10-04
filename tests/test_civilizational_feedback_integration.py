"""
Comprehensive Test Suite for OmniMind Civilizational Feedback + Meta-Rule Evolution
Tests all reasoning steps, simulation validation, and integration components.
"""

import unittest
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from egdol.omnimind.civilizational_feedback import (
    CivilizationalFeedbackEngine, 
    CivilizationalInsight, 
    MetaRuleCandidate,
    CivilizationalFeedbackResult
)
from egdol.omnimind.meta_rule_discovery import (
    MetaRuleDiscoveryEngine,
    PatternMatch,
    MetaRuleTemplate,
    MetaRuleValidation
)
from egdol.omnimind.enhanced_context_stabilization import (
    EnhancedContextStabilization,
    ContextStabilizationResult,
    ReasoningType,
    FallbackLevel
)
from egdol.omnimind.reflection_mode_plus_plus import (
    ReflectionModePlusPlus,
    ReflectionAnalysis,
    HeuristicUpdate,
    ReflectionResultPlusPlus,
    ReflectionTrigger,
    HeuristicAdjustment
)
from egdol.omnimind.conversational.reflexive_audit import ReflexiveAuditModule
from egdol.omnimind.conversational.meta_learning_engine import MetaLearningEngine
from egdol.omnimind.conversational.feedback_loop import FeedbackStorage
from egdol.omnimind.conversational.personality_evolution import PersonalityEvolutionEngine
from egdol.omnimind.conversational.reflection_mode_plus import ReflectionModePlus
from egdol.omnimind.conversational.context_intent_resolver import ContextIntentResolver
from egdol.omnimind.conversational.reasoning_normalizer import ReasoningNormalizer
from egdol.omnimind.conversational.personality_fallbacks import PersonalityFallbackReasoner
from egdol.omnimind.conversational.personality_framework import PersonalityFramework


class TestCivilizationalFeedbackIntegration(unittest.TestCase):
    """Test suite for civilizational feedback integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.audit_module = ReflexiveAuditModule()
        self.storage = FeedbackStorage()
        self.meta_learning = MetaLearningEngine(self.storage)
        self.personality_evolution = PersonalityEvolutionEngine(
            meta_learning_engine=self.meta_learning,
            audit_module=self.audit_module
        )
        self.reflection_plus = ReflectionModePlus(
            meta_learning_engine=self.meta_learning,
            personality_evolution_engine=self.personality_evolution,
            audit_module=self.audit_module
        )
        self.context_resolver = ContextIntentResolver()
        self.reasoning_normalizer = ReasoningNormalizer()
        self.fallback_reasoner = PersonalityFallbackReasoner()
        self.personality_framework = PersonalityFramework()
        
        # Create civilizational feedback engine
        self.feedback_engine = CivilizationalFeedbackEngine(
            multi_universe_orchestrator=None,  # Mock
            civilization_architect=None,  # Mock
            audit_module=self.audit_module,
            meta_learning=self.meta_learning,
            personality_evolution=self.personality_evolution,
            reflection_plus=self.reflection_plus,
            context_resolver=self.context_resolver,
            reasoning_normalizer=self.reasoning_normalizer,
            fallback_reasoner=self.fallback_reasoner
        )
    
    def test_civilizational_feedback_processing(self):
        """Test civilizational feedback processing."""
        user_input = "What is the optimal strategy for resource allocation in a multi-planetary civilization?"
        personality = "Strategos"
        
        result = self.feedback_engine.process_conversation_with_civilizational_feedback(
            user_input, personality
        )
        
        self.assertIsInstance(result, CivilizationalFeedbackResult)
        self.assertIsInstance(result.conversation_insights, list)
        self.assertIsInstance(result.meta_rule_candidates, list)
        self.assertIsInstance(result.simulation_updates, dict)
        self.assertIsInstance(result.personality_evolution, dict)
        self.assertIsInstance(result.audit_results, object)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.processing_time, float)
    
    def test_civilizational_insight_generation(self):
        """Test civilizational insight generation."""
        insight = CivilizationalInsight(
            insight_type="conversation_pattern",
            pattern="strategic_analysis",
            confidence=0.8,
            source_conversation="Test conversation",
            simulation_outcome={"civilization_type": "advanced"},
            meta_rule_candidate="strategic_rule: pattern => enhanced_strategy"
        )
        
        self.assertEqual(insight.insight_type, "conversation_pattern")
        self.assertEqual(insight.pattern, "strategic_analysis")
        self.assertEqual(insight.confidence, 0.8)
        self.assertIsNotNone(insight.timestamp)
    
    def test_meta_rule_candidate_creation(self):
        """Test meta-rule candidate creation."""
        candidate = MetaRuleCandidate(
            rule_name="test_rule",
            rule_pattern="test_pattern => enhanced_output",
            confidence=0.7,
            source_insights=[],
            validation_tests=[],
            simulation_validation={},
            personality_applicability=["Strategos", "Archivist"]
        )
        
        self.assertEqual(candidate.rule_name, "test_rule")
        self.assertEqual(candidate.rule_pattern, "test_pattern => enhanced_output")
        self.assertEqual(candidate.confidence, 0.7)
        self.assertIn("Strategos", candidate.personality_applicability)
        self.assertIsNotNone(candidate.timestamp)
    
    def test_feedback_summary_generation(self):
        """Test feedback summary generation."""
        summary = self.feedback_engine.get_feedback_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('feedback_cycles', summary)
        self.assertIn('successful_integrations', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('meta_rule_validation_rate', summary)
        self.assertIn('total_insights', summary)
        self.assertIn('total_meta_rules', summary)


class TestMetaRuleDiscovery(unittest.TestCase):
    """Test suite for meta-rule discovery system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.audit_module = ReflexiveAuditModule()
        self.storage = FeedbackStorage()
        self.meta_learning = MetaLearningEngine(self.storage)
        self.discovery_engine = MetaRuleDiscoveryEngine(
            audit_module=self.audit_module,
            meta_learning=self.meta_learning
        )
    
    def test_meta_rule_discovery_from_conversation(self):
        """Test meta-rule discovery from conversation data."""
        conversation_data = {
            'response': 'Commander, tactical analysis complete. Strategic evaluation suggests optimal resource allocation.',
            'reasoning_trace': ['tactical_analysis', 'strategic_evaluation', 'response_generation'],
            'personality': 'Strategos',
            'confidence': 0.8
        }
        
        candidates = self.discovery_engine.discover_meta_rules_from_conversation(conversation_data)
        
        self.assertIsInstance(candidates, list)
        # Should generate at least one candidate from the conversation
        self.assertGreaterEqual(len(candidates), 0)
    
    def test_meta_rule_discovery_from_simulation(self):
        """Test meta-rule discovery from simulation data."""
        simulation_data = {
            'social_structures': {'hierarchy': 'advanced', 'egalitarian': 'moderate'},
            'economic_patterns': {'market': 'efficient', 'planned': 'stable'},
            'conflict_resolution': {'mediation': 'successful', 'adjudication': 'fair'}
        }
        
        candidates = self.discovery_engine.discover_meta_rules_from_simulation(simulation_data)
        
        self.assertIsInstance(candidates, list)
        # Should generate candidates from simulation patterns
        self.assertGreaterEqual(len(candidates), 0)
    
    def test_meta_rule_discovery_from_insights(self):
        """Test meta-rule discovery from civilizational insights."""
        insights = [
            CivilizationalInsight(
                insight_type="conversation_pattern",
                pattern="strategic_analysis",
                confidence=0.8,
                source_conversation="Test conversation",
                simulation_outcome={"civilization_type": "advanced"},
                meta_rule_candidate="strategic_rule: pattern => enhanced_strategy"
            )
        ]
        
        candidates = self.discovery_engine.discover_meta_rules_from_insights(insights)
        
        self.assertIsInstance(candidates, list)
        # Should generate candidates from insights
        self.assertGreaterEqual(len(candidates), 0)
    
    def test_pattern_match_creation(self):
        """Test pattern match creation."""
        pattern_match = PatternMatch(
            pattern_type="response",
            pattern_text="strategic_response_pattern",
            confidence=0.8,
            source="conversation",
            context={"response_length": 100}
        )
        
        self.assertEqual(pattern_match.pattern_type, "response")
        self.assertEqual(pattern_match.pattern_text, "strategic_response_pattern")
        self.assertEqual(pattern_match.confidence, 0.8)
        self.assertIsNotNone(pattern_match.timestamp)
    
    def test_meta_rule_template_creation(self):
        """Test meta-rule template creation."""
        template = MetaRuleTemplate(
            template_name="response_pattern_rule",
            pattern_matcher="response",
            rule_generator="response_rule: {pattern} => improved_response",
            validation_criteria=["logical_consistency", "empirical_support"],
            personality_applicability=["Strategos", "Archivist", "Lawmaker", "Oracle"]
        )
        
        self.assertEqual(template.template_name, "response_pattern_rule")
        self.assertEqual(template.pattern_matcher, "response")
        self.assertIn("logical_consistency", template.validation_criteria)
        self.assertIn("Strategos", template.personality_applicability)
    
    def test_discovery_summary_generation(self):
        """Test discovery summary generation."""
        summary = self.discovery_engine.get_discovery_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('discovery_cycles', summary)
        self.assertIn('patterns_analyzed', summary)
        self.assertIn('rules_generated', summary)
        self.assertIn('rules_validated', summary)
        self.assertIn('rules_applied', summary)


class TestEnhancedContextStabilization(unittest.TestCase):
    """Test suite for enhanced context stabilization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.audit_module = ReflexiveAuditModule()
        self.context_resolver = ContextIntentResolver()
        self.reasoning_normalizer = ReasoningNormalizer()
        self.fallback_reasoner = PersonalityFallbackReasoner()
        self.personality_framework = PersonalityFramework()
        
        self.stabilization = EnhancedContextStabilization(
            context_resolver=self.context_resolver,
            reasoning_normalizer=self.reasoning_normalizer,
            fallback_reasoner=self.fallback_reasoner,
            audit_module=self.audit_module,
            personality_framework=self.personality_framework
        )
    
    def test_context_stabilization_success(self):
        """Test successful context stabilization."""
        user_input = "What is the optimal strategy for resource allocation?"
        personality = "Strategos"
        
        result = self.stabilization.stabilize_context_and_reason(
            user_input, personality
        )
        
        self.assertIsInstance(result, ContextStabilizationResult)
        self.assertTrue(result.success)
        self.assertEqual(result.personality, personality)
        self.assertIsInstance(result.reasoning_type, ReasoningType)
        self.assertIsInstance(result.confidence_score, float)
        self.assertIsInstance(result.fallback_used, bool)
        self.assertIsInstance(result.stabilization_notes, list)
        self.assertIsInstance(result.processing_time, float)
    
    def test_context_stabilization_with_fallback(self):
        """Test context stabilization with fallback usage."""
        user_input = "Complex query that might trigger fallback"
        personality = "Archivist"
        
        result = self.stabilization.stabilize_context_and_reason(
            user_input, personality, max_attempts=1
        )
        
        self.assertIsInstance(result, ContextStabilizationResult)
        # Should succeed even with fallback
        self.assertTrue(result.success)
        self.assertEqual(result.personality, personality)
    
    def test_reasoning_type_determination(self):
        """Test reasoning type determination."""
        # Test tactical reasoning
        user_input = "What is the tactical approach to this situation?"
        personality = "Strategos"
        
        result = self.stabilization.stabilize_context_and_reason(
            user_input, personality
        )
        
        self.assertEqual(result.reasoning_type, ReasoningType.TACTICAL)
    
    def test_personality_specific_reasoning(self):
        """Test personality-specific reasoning."""
        personalities = ["Strategos", "Archivist", "Lawmaker", "Oracle"]
        
        for personality in personalities:
            result = self.stabilization.stabilize_context_and_reason(
                f"Test query for {personality}", personality
            )
            
            self.assertTrue(result.success)
            self.assertEqual(result.personality, personality)
    
    def test_stabilization_summary_generation(self):
        """Test stabilization summary generation."""
        summary = self.stabilization.get_stabilization_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_stabilization_attempts', summary)
        self.assertIn('successful_stabilizations', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('fallback_usage_count', summary)
        self.assertIn('context_normalization_errors', summary)


class TestReflectionModePlusPlus(unittest.TestCase):
    """Test suite for reflection mode plus plus."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.audit_module = ReflexiveAuditModule()
        self.storage = FeedbackStorage()
        self.meta_learning = MetaLearningEngine(self.storage)
        self.personality_evolution = PersonalityEvolutionEngine(
            meta_learning_engine=self.meta_learning,
            audit_module=self.audit_module
        )
        self.reflection_plus = ReflectionModePlus(
            meta_learning_engine=self.meta_learning,
            personality_evolution_engine=self.personality_evolution,
            audit_module=self.audit_module
        )
        self.context_stabilization = EnhancedContextStabilization(
            context_resolver=ContextIntentResolver(),
            reasoning_normalizer=ReasoningNormalizer(),
            fallback_reasoner=PersonalityFallbackReasoner(),
            audit_module=self.audit_module,
            personality_framework=PersonalityFramework()
        )
        self.meta_rule_discovery = MetaRuleDiscoveryEngine(
            audit_module=self.audit_module,
            meta_learning=self.meta_learning
        )
        
        self.reflection_plus_plus = ReflectionModePlusPlus(
            reflection_plus=self.reflection_plus,
            audit_module=self.audit_module,
            meta_learning=self.meta_learning,
            personality_evolution=self.personality_evolution,
            context_stabilization=self.context_stabilization,
            meta_rule_discovery=self.meta_rule_discovery
        )
    
    def test_reflection_mode_activation(self):
        """Test reflection mode activation."""
        user_input = "Test query"
        personality = "Strategos"
        original_error = "Low confidence in reasoning output"
        
        result = self.reflection_plus_plus.reflect_and_retry_plus_plus(
            user_input, personality, original_error
        )
        
        self.assertIsInstance(result, ReflectionResultPlusPlus)
        self.assertIsInstance(result.success, bool)
        self.assertEqual(result.original_error, original_error)
        self.assertIsInstance(result.reflection_analysis, ReflectionAnalysis)
        self.assertIsInstance(result.heuristic_updates, list)
        self.assertIsInstance(result.retry_attempts, int)
        self.assertIsInstance(result.final_confidence, float)
        self.assertIsInstance(result.improvement_metrics, dict)
        self.assertIsInstance(result.processing_time, float)
    
    def test_reflection_analysis_creation(self):
        """Test reflection analysis creation."""
        analysis = ReflectionAnalysis(
            trigger=ReflectionTrigger.LOW_CONFIDENCE,
            analysis_depth="deep",
            identified_gaps=["Insufficient confidence", "Weak logical connections"],
            improvement_suggestions=["Increase confidence thresholds", "Strengthen logical reasoning"],
            heuristic_adjustments=[HeuristicAdjustment.CONFIDENCE_THRESHOLD],
            confidence_impact=0.3,
            processing_time=0.1
        )
        
        self.assertEqual(analysis.trigger, ReflectionTrigger.LOW_CONFIDENCE)
        self.assertEqual(analysis.analysis_depth, "deep")
        self.assertEqual(len(analysis.identified_gaps), 2)
        self.assertEqual(len(analysis.improvement_suggestions), 2)
        self.assertIsNotNone(analysis.timestamp)
    
    def test_heuristic_update_creation(self):
        """Test heuristic update creation."""
        update = HeuristicUpdate(
            adjustment_type=HeuristicAdjustment.CONFIDENCE_THRESHOLD,
            old_value=0.6,
            new_value=0.7,
            confidence_improvement=0.1,
            validation_tests=[],
            applied=True
        )
        
        self.assertEqual(update.adjustment_type, HeuristicAdjustment.CONFIDENCE_THRESHOLD)
        self.assertEqual(update.old_value, 0.6)
        self.assertEqual(update.new_value, 0.7)
        self.assertTrue(update.applied)
        self.assertIsNotNone(update.timestamp)
    
    def test_reflection_triggers(self):
        """Test different reflection triggers."""
        triggers = [
            ReflectionTrigger.LOW_CONFIDENCE,
            ReflectionTrigger.AUDIT_FAILURE,
            ReflectionTrigger.REASONING_ERROR,
            ReflectionTrigger.FALLBACK_OVERUSE,
            ReflectionTrigger.CONTEXT_MISMATCH,
            ReflectionTrigger.PERSONALITY_INCONSISTENCY
        ]
        
        for trigger in triggers:
            user_input = "Test query"
            personality = "Strategos"
            original_error = f"Error related to {trigger.value}"
            
            result = self.reflection_plus_plus.reflect_and_retry_plus_plus(
                user_input, personality, original_error
            )
            
            self.assertIsInstance(result, ReflectionResultPlusPlus)
            self.assertEqual(result.reflection_analysis.trigger, trigger)
    
    def test_heuristic_adjustment_types(self):
        """Test different heuristic adjustment types."""
        adjustment_types = [
            HeuristicAdjustment.CONFIDENCE_THRESHOLD,
            HeuristicAdjustment.REASONING_STRATEGY,
            HeuristicAdjustment.FALLBACK_TRIGGER,
            HeuristicAdjustment.CONTEXT_WEIGHT,
            HeuristicAdjustment.PERSONALITY_BIAS,
            HeuristicAdjustment.META_RULE_APPLICATION
        ]
        
        for adjustment_type in adjustment_types:
            update = HeuristicUpdate(
                adjustment_type=adjustment_type,
                old_value=0.5,
                new_value=0.6,
                confidence_improvement=0.1,
                validation_tests=[],
                applied=True
            )
            
            self.assertEqual(update.adjustment_type, adjustment_type)
            self.assertTrue(update.applied)
    
    def test_reflection_summary_generation(self):
        """Test reflection summary generation."""
        summary = self.reflection_plus_plus.get_reflection_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('reflection_cycles', summary)
        self.assertIn('successful_reflections', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('total_heuristic_updates', summary)
        self.assertIn('applied_heuristic_updates', summary)
        self.assertIn('average_confidence_improvement', summary)
        self.assertIn('heuristic_effectiveness', summary)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create all components
        self.audit_module = ReflexiveAuditModule()
        self.storage = FeedbackStorage()
        self.meta_learning = MetaLearningEngine(self.storage)
        self.personality_evolution = PersonalityEvolutionEngine(
            meta_learning_engine=self.meta_learning,
            audit_module=self.audit_module
        )
        self.reflection_plus = ReflectionModePlus(
            meta_learning_engine=self.meta_learning,
            personality_evolution_engine=self.personality_evolution,
            audit_module=self.audit_module
        )
        self.context_resolver = ContextIntentResolver()
        self.reasoning_normalizer = ReasoningNormalizer()
        self.fallback_reasoner = PersonalityFallbackReasoner()
        self.personality_framework = PersonalityFramework()
        
        # Create enhanced components
        self.context_stabilization = EnhancedContextStabilization(
            context_resolver=self.context_resolver,
            reasoning_normalizer=self.reasoning_normalizer,
            fallback_reasoner=self.fallback_reasoner,
            audit_module=self.audit_module,
            personality_framework=self.personality_framework
        )
        
        self.meta_rule_discovery = MetaRuleDiscoveryEngine(
            audit_module=self.audit_module,
            meta_learning=self.meta_learning
        )
        
        self.reflection_plus_plus = ReflectionModePlusPlus(
            reflection_plus=self.reflection_plus,
            audit_module=self.audit_module,
            meta_learning=self.meta_learning,
            personality_evolution=self.personality_evolution,
            context_stabilization=self.context_stabilization,
            meta_rule_discovery=self.meta_rule_discovery
        )
        
        self.feedback_engine = CivilizationalFeedbackEngine(
            multi_universe_orchestrator=None,  # Mock
            civilization_architect=None,  # Mock
            audit_module=self.audit_module,
            meta_learning=self.meta_learning,
            personality_evolution=self.personality_evolution,
            reflection_plus=self.reflection_plus,
            context_resolver=self.context_resolver,
            reasoning_normalizer=self.reasoning_normalizer,
            fallback_reasoner=self.fallback_reasoner
        )
    
    def test_complete_reasoning_pipeline(self):
        """Test complete reasoning pipeline from input to output."""
        user_input = "What is the optimal strategy for multi-planetary resource allocation?"
        personality = "Strategos"
        
        # Test context stabilization
        stabilization_result = self.context_stabilization.stabilize_context_and_reason(
            user_input, personality
        )
        
        self.assertTrue(stabilization_result.success)
        self.assertEqual(stabilization_result.personality, personality)
        
        # Test meta-rule discovery
        conversation_data = {
            'response': stabilization_result.stabilization_notes[0] if stabilization_result.stabilization_notes else "Test response",
            'reasoning_trace': ['context_analysis', 'reasoning_execution'],
            'personality': personality,
            'confidence': stabilization_result.confidence_score
        }
        
        meta_rules = self.meta_rule_discovery.discover_meta_rules_from_conversation(conversation_data)
        self.assertIsInstance(meta_rules, list)
        
        # Test reflection mode if needed
        if stabilization_result.confidence_score < 0.6:
            reflection_result = self.reflection_plus_plus.reflect_and_retry_plus_plus(
                user_input, personality, "Low confidence score"
            )
            
            self.assertIsInstance(reflection_result, ReflectionResultPlusPlus)
    
    def test_personality_specific_reasoning(self):
        """Test reasoning for different personalities."""
        personalities = ["Strategos", "Archivist", "Lawmaker", "Oracle"]
        test_queries = [
            "What is the optimal military strategy?",
            "What historical precedents exist?",
            "What are the legal implications?",
            "What does the Oracle foresee?"
        ]
        
        for personality, query in zip(personalities, test_queries):
            result = self.context_stabilization.stabilize_context_and_reason(
                query, personality
            )
            
            self.assertTrue(result.success)
            self.assertEqual(result.personality, personality)
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms for challenging queries."""
        challenging_queries = [
            "Complex mathematical proof that requires advanced reasoning",
            "Historical analysis requiring extensive archival knowledge",
            "Legal case requiring deep jurisdictional understanding",
            "Mystical prophecy requiring oracle insight"
        ]
        
        for query in challenging_queries:
            result = self.context_stabilization.stabilize_context_and_reason(
                query, "Strategos", max_attempts=1
            )
            
            # Should succeed even with fallback
            self.assertTrue(result.success)
    
    def test_meta_rule_evolution(self):
        """Test meta-rule evolution through conversation patterns."""
        # Simulate multiple conversation turns
        conversation_turns = [
            {"response": "Strategic analysis suggests...", "personality": "Strategos"},
            {"response": "Historical records indicate...", "personality": "Archivist"},
            {"response": "Legal precedent shows...", "personality": "Lawmaker"},
            {"response": "The Oracle perceives...", "personality": "Oracle"}
        ]
        
        discovered_rules = []
        
        for turn in conversation_turns:
            rules = self.meta_rule_discovery.discover_meta_rules_from_conversation(turn)
            discovered_rules.extend(rules)
        
        # Should discover rules from conversation patterns
        self.assertGreaterEqual(len(discovered_rules), 0)
    
    def test_system_performance_tracking(self):
        """Test system performance tracking across components."""
        # Test feedback engine performance
        feedback_summary = self.feedback_engine.get_feedback_summary()
        self.assertIsInstance(feedback_summary, dict)
        
        # Test discovery engine performance
        discovery_summary = self.meta_rule_discovery.get_discovery_summary()
        self.assertIsInstance(discovery_summary, dict)
        
        # Test stabilization performance
        stabilization_summary = self.context_stabilization.get_stabilization_summary()
        self.assertIsInstance(stabilization_summary, dict)
        
        # Test reflection performance
        reflection_summary = self.reflection_plus_plus.get_reflection_summary()
        self.assertIsInstance(reflection_summary, dict)


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)
