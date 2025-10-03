"""
Comprehensive Test Suite for Context-Intent Reasoning Stabilization Layer
Tests the stabilization layer components and personality fallback reasoning.
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
from egdol.omnimind.conversational.context_intent_resolver import ContextIntentResolver, ReasoningContext
from egdol.omnimind.conversational.reasoning_normalizer import ReasoningNormalizer, NormalizedReasoningInput
from egdol.omnimind.conversational.personality_fallbacks import PersonalityFallbackReasoner, FallbackResponse


class TestContextIntentResolver(unittest.TestCase):
    """Test the context intent resolver."""
    
    def setUp(self):
        self.resolver = ContextIntentResolver()
    
    def test_initialization(self):
        """Test resolver initialization."""
        self.assertIsNotNone(self.resolver.intent_parser)
        self.assertIsNotNone(self.resolver.fallback_patterns)
        self.assertIsNotNone(self.resolver.domain_mappings)
    
    def test_resolve_intent_and_context_basic(self):
        """Test resolving basic intent and context."""
        context = self.resolver.resolve_intent_and_context("Hello", "Strategos")
        
        self.assertIsInstance(context, ReasoningContext)
        self.assertIsNotNone(context.reasoning_type)
        self.assertIsNotNone(context.intent_type)
        self.assertEqual(context.personality, "Strategos")
        self.assertIsNotNone(context.domain)
        self.assertGreaterEqual(context.confidence, 0.0)
        self.assertLessEqual(context.confidence, 1.0)
    
    def test_resolve_intent_and_context_math(self):
        """Test resolving mathematical intent."""
        context = self.resolver.resolve_intent_and_context("What is 2 + 2?", "Strategos")
        
        # For Strategos personality, math questions might be routed to strategic reasoning
        self.assertIn(context.reasoning_type, ["basic_reasoning", "strategic_reasoning"])
        self.assertEqual(context.intent_type, "QUESTION")
        self.assertGreater(context.confidence, 0.0)
    
    def test_resolve_intent_and_context_logic(self):
        """Test resolving logical intent."""
        context = self.resolver.resolve_intent_and_context("If A then B", "Archivist")
        
        self.assertEqual(context.personality, "Archivist")
        self.assertIsNotNone(context.reasoning_type)
    
    def test_resolve_intent_and_context_civilizational(self):
        """Test resolving civilizational intent."""
        context = self.resolver.resolve_intent_and_context("Analyze civilization patterns", "Archivist")
        
        self.assertEqual(context.reasoning_type, "civilizational_reasoning")
        self.assertEqual(context.intent_type, "CIVILIZATIONAL_QUERY")
    
    def test_resolve_intent_and_context_strategic(self):
        """Test resolving strategic intent."""
        context = self.resolver.resolve_intent_and_context("What is the best military strategy?", "Strategos")
        
        self.assertEqual(context.reasoning_type, "strategic_reasoning")
        self.assertEqual(context.intent_type, "STRATEGIC_ANALYSIS")
    
    def test_resolve_intent_and_context_meta_rule(self):
        """Test resolving meta-rule intent."""
        context = self.resolver.resolve_intent_and_context("Discover meta-rules", "Lawmaker")
        
        self.assertEqual(context.reasoning_type, "meta_rule_reasoning")
        self.assertEqual(context.intent_type, "META_RULE_DISCOVERY")
    
    def test_resolve_intent_and_context_universe(self):
        """Test resolving universe comparison intent."""
        context = self.resolver.resolve_intent_and_context("Compare universes", "Oracle")
        
        self.assertEqual(context.reasoning_type, "universe_reasoning")
        self.assertEqual(context.intent_type, "UNIVERSE_COMPARISON")
    
    def test_normalize_reasoning_input(self):
        """Test normalizing reasoning input."""
        context = self.resolver.resolve_intent_and_context("Hello", "Strategos")
        normalized = self.resolver.normalize_reasoning_input(context, "Hello")
        
        self.assertIsInstance(normalized, dict)
        self.assertIn('user_input', normalized)
        self.assertIn('reasoning_type', normalized)
        self.assertIn('personality', normalized)
        self.assertIn('timestamp', normalized)
    
    def test_detect_fallback_requirement(self):
        """Test detecting fallback requirement."""
        context = ReasoningContext(
            reasoning_type="basic_reasoning",
            intent_type="QUESTION",
            personality="Strategos",
            domain="general",
            complexity_level=0.3,
            confidence=0.2
        )
        
        requires_fallback = self.resolver.detect_fallback_requirement(context)
        self.assertTrue(requires_fallback)
    
    def test_get_fallback_reasoning_type(self):
        """Test getting fallback reasoning type."""
        context = ReasoningContext(
            reasoning_type="basic_reasoning",
            intent_type="QUESTION",
            personality="Strategos",
            domain="general",
            complexity_level=0.5,
            confidence=0.5
        )
        
        fallback_type = self.resolver.get_fallback_reasoning_type(context)
        self.assertEqual(fallback_type, "tactical_fallback")
    
    def test_validate_context(self):
        """Test context validation."""
        context = ReasoningContext(
            reasoning_type="basic_reasoning",
            intent_type="QUESTION",
            personality="Strategos",
            domain="general",
            complexity_level=0.5,
            confidence=0.5
        )
        
        is_valid, errors = self.resolver.validate_context(context)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_context_invalid(self):
        """Test context validation with invalid data."""
        context = ReasoningContext(
            reasoning_type="",
            intent_type="",
            personality="",
            domain="",
            complexity_level=1.5,  # Invalid
            confidence=-0.1  # Invalid
        )
        
        is_valid, errors = self.resolver.validate_context(context)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_create_default_context(self):
        """Test creating default context."""
        context = self.resolver.create_default_context("Hello", "Strategos")
        
        self.assertIsInstance(context, ReasoningContext)
        self.assertEqual(context.personality, "Strategos")
        self.assertEqual(context.reasoning_type, "basic_reasoning")
        self.assertTrue(context.metadata.get('default_context', False))


class TestPersonalityFallbackReasoner(unittest.TestCase):
    """Test the personality fallback reasoner."""
    
    def setUp(self):
        self.fallback_reasoner = PersonalityFallbackReasoner()
    
    def test_initialization(self):
        """Test fallback reasoner initialization."""
        self.assertIsNotNone(self.fallback_reasoner.fallback_data)
        self.assertIsNotNone(self.fallback_reasoner.math_operations)
        self.assertIsNotNone(self.fallback_reasoner.logic_patterns)
    
    def test_get_fallback_response_strategos(self):
        """Test getting Strategos fallback response."""
        response = self.fallback_reasoner.get_fallback_response(
            "Hello", "Strategos", "basic_reasoning", {}
        )
        
        self.assertIsInstance(response, FallbackResponse)
        self.assertEqual(response.personality, "Strategos")
        self.assertIn("Commander", response.content)
        self.assertEqual(response.confidence, 0.7)
    
    def test_get_fallback_response_archivist(self):
        """Test getting Archivist fallback response."""
        response = self.fallback_reasoner.get_fallback_response(
            "Tell me about history", "Archivist", "basic_reasoning", {}
        )
        
        self.assertIsInstance(response, FallbackResponse)
        self.assertEqual(response.personality, "Archivist")
        self.assertIn("archives", response.content.lower())
    
    def test_get_fallback_response_lawmaker(self):
        """Test getting Lawmaker fallback response."""
        response = self.fallback_reasoner.get_fallback_response(
            "What are the rules?", "Lawmaker", "basic_reasoning", {}
        )
        
        self.assertIsInstance(response, FallbackResponse)
        self.assertEqual(response.personality, "Lawmaker")
        # Check for legal/governance related terms
        self.assertTrue(
            any(term in response.content.lower() for term in ["governance", "legal", "principles", "framework", "law"])
        )
    
    def test_get_fallback_response_oracle(self):
        """Test getting Oracle fallback response."""
        response = self.fallback_reasoner.get_fallback_response(
            "What are the cosmic patterns?", "Oracle", "basic_reasoning", {}
        )
        
        self.assertIsInstance(response, FallbackResponse)
        self.assertEqual(response.personality, "Oracle")
        # Check for mystical/cosmic related terms
        self.assertTrue(
            any(term in response.content.lower() for term in ["cosmic", "universal", "cosmos", "universe", "existence", "veil", "tapestry", "infinite"])
        )
    
    def test_handle_math_fallback(self):
        """Test handling mathematical fallback."""
        response = self.fallback_reasoner._handle_math_fallback("What is 2 + 2?", "Strategos")
        
        self.assertIsInstance(response, str)
        self.assertIn("2", response)
        self.assertIn("plus", response)
        self.assertIn("4", response)
    
    def test_handle_logic_fallback(self):
        """Test handling logical fallback."""
        response = self.fallback_reasoner._handle_logic_fallback("If A then B", "Archivist")
        
        self.assertIsInstance(response, str)
        self.assertIn("logical", response.lower())
    
    def test_should_use_fallback(self):
        """Test determining if fallback should be used."""
        # Low confidence should use fallback
        should_use = self.fallback_reasoner.should_use_fallback(0.3, "basic_reasoning", {})
        self.assertTrue(should_use)
        
        # High confidence should not use fallback
        should_use = self.fallback_reasoner.should_use_fallback(0.8, "strategic_reasoning", {})
        self.assertFalse(should_use)
    
    def test_determine_fallback_type(self):
        """Test determining fallback type."""
        fallback_type = self.fallback_reasoner._determine_fallback_type("What is 2 + 2?", "Strategos", "basic_reasoning")
        self.assertEqual(fallback_type, "mathematical")
        
        fallback_type = self.fallback_reasoner._determine_fallback_type("If A then B", "Archivist", "basic_reasoning")
        self.assertEqual(fallback_type, "logical")
        
        fallback_type = self.fallback_reasoner._determine_fallback_type("Hello", "Strategos", "basic_reasoning")
        self.assertEqual(fallback_type, "greeting")


class TestReasoningNormalizer(unittest.TestCase):
    """Test the reasoning normalizer."""
    
    def setUp(self):
        self.normalizer = ReasoningNormalizer()
    
    def test_initialization(self):
        """Test normalizer initialization."""
        self.assertIsNotNone(self.normalizer.fallback_reasoner)
        self.assertIsNotNone(self.normalizer.default_reasoning_types)
        self.assertIsNotNone(self.normalizer.default_domains)
    
    def test_normalize_input(self):
        """Test normalizing input."""
        normalized = self.normalizer.normalize_input("Hello", "Strategos")
        
        self.assertIsInstance(normalized, NormalizedReasoningInput)
        self.assertEqual(normalized.user_input, "Hello")
        self.assertEqual(normalized.personality, "Strategos")
        self.assertIsNotNone(normalized.reasoning_type)
        self.assertIsNotNone(normalized.intent_type)
    
    def test_normalize_input_with_context(self):
        """Test normalizing input with provided context."""
        context = ReasoningContext(
            reasoning_type="basic_reasoning",
            intent_type="QUESTION",
            personality="Strategos",
            domain="general",
            complexity_level=0.5,
            confidence=0.5
        )
        
        normalized = self.normalizer.normalize_input("Hello", "Strategos", context)
        
        self.assertEqual(normalized.reasoning_type, "basic_reasoning")
        self.assertEqual(normalized.intent_type, "QUESTION")
        self.assertEqual(normalized.personality, "Strategos")
    
    def test_validate_normalized_input(self):
        """Test validating normalized input."""
        normalized = NormalizedReasoningInput(
            user_input="Hello",
            reasoning_type="basic_reasoning",
            intent_type="QUESTION",
            personality="Strategos",
            domain="general",
            complexity_level=0.5,
            confidence=0.5
        )
        
        is_valid, errors = self.normalizer._validate_normalized_input(normalized)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_normalized_input_invalid(self):
        """Test validating invalid normalized input."""
        normalized = NormalizedReasoningInput(
            user_input="",
            reasoning_type="",
            intent_type="",
            personality="",
            domain="",
            complexity_level=1.5,  # Invalid
            confidence=-0.1  # Invalid
        )
        
        is_valid, errors = self.normalizer._validate_normalized_input(normalized)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_fix_normalization_issues(self):
        """Test fixing normalization issues."""
        normalized = NormalizedReasoningInput(
            user_input="",
            reasoning_type="",
            intent_type="",
            personality="",
            domain="",
            complexity_level=1.5,
            confidence=-0.1
        )
        
        errors = ["Missing user_input", "Missing reasoning_type", "Invalid complexity_level"]
        fixed = self.normalizer._fix_normalization_issues(normalized, errors)
        
        self.assertEqual(fixed.user_input, "Hello")
        self.assertEqual(fixed.reasoning_type, "basic_reasoning")
        self.assertEqual(fixed.complexity_level, 0.5)
    
    def test_get_fallback_response(self):
        """Test getting fallback response."""
        normalized = NormalizedReasoningInput(
            user_input="Hello",
            reasoning_type="basic_reasoning",
            intent_type="QUESTION",
            personality="Strategos",
            domain="general",
            complexity_level=0.5,
            confidence=0.5
        )
        
        fallback_response = self.normalizer.get_fallback_response(normalized)
        
        self.assertIsInstance(fallback_response, FallbackResponse)
        self.assertEqual(fallback_response.personality, "Strategos")
    
    def test_should_use_fallback(self):
        """Test determining if fallback should be used."""
        normalized = NormalizedReasoningInput(
            user_input="Hello",
            reasoning_type="basic_reasoning",
            intent_type="QUESTION",
            personality="Strategos",
            domain="general",
            complexity_level=0.5,
            confidence=0.3  # Low confidence
        )
        
        should_use = self.normalizer.should_use_fallback(normalized)
        self.assertTrue(should_use)
    
    def test_create_reasoning_engine_input(self):
        """Test creating reasoning engine input."""
        normalized = NormalizedReasoningInput(
            user_input="Hello",
            reasoning_type="basic_reasoning",
            intent_type="QUESTION",
            personality="Strategos",
            domain="general",
            complexity_level=0.5,
            confidence=0.5
        )
        
        engine_input = self.normalizer.create_reasoning_engine_input(normalized)
        
        self.assertIsInstance(engine_input, dict)
        self.assertIn('query', engine_input)
        self.assertIn('reasoning_type', engine_input)
        self.assertIn('personality', engine_input)
    
    def test_create_fallback_input(self):
        """Test creating fallback input."""
        normalized = NormalizedReasoningInput(
            user_input="Hello",
            reasoning_type="basic_reasoning",
            intent_type="QUESTION",
            personality="Strategos",
            domain="general",
            complexity_level=0.5,
            confidence=0.5
        )
        
        fallback_input = self.normalizer.create_fallback_input(normalized)
        
        self.assertIsInstance(fallback_input, dict)
        self.assertIn('user_input', fallback_input)
        self.assertIn('personality', fallback_input)
        self.assertIn('reasoning_type', fallback_input)


class TestStabilizedConversationalInterface(unittest.TestCase):
    """Test the stabilized conversational interface."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.omnimind_core = OmniMind(self.temp_dir)
        self.interface = ConversationalInterface(self.omnimind_core, self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test interface initialization with stabilization layer."""
        self.assertIsNotNone(self.interface.context_resolver)
        self.assertIsNotNone(self.interface.reasoning_normalizer)
        self.assertIsNotNone(self.interface.fallback_reasoner)
    
    def test_process_message_greeting(self):
        """Test processing greeting message."""
        self.interface.start_conversation()
        result = self.interface.process_message("Hello")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
        self.assertIn('fallback_used', result)
    
    def test_process_message_math(self):
        """Test processing mathematical message."""
        self.interface.start_conversation()
        result = self.interface.process_message("What is 2 + 2?")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
    
    def test_process_message_logic(self):
        """Test processing logical message."""
        self.interface.start_conversation()
        result = self.interface.process_message("If A then B")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
    
    def test_process_message_personality_switch(self):
        """Test processing personality switch message."""
        self.interface.start_conversation()
        result = self.interface.process_message("Switch to Archivist")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
    
    def test_process_message_civilizational(self):
        """Test processing civilizational message."""
        self.interface.start_conversation()
        result = self.interface.process_message("Analyze civilization patterns")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
    
    def test_process_message_strategic(self):
        """Test processing strategic message."""
        self.interface.start_conversation()
        result = self.interface.process_message("What is the best military strategy?")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
    
    def test_process_message_meta_rule(self):
        """Test processing meta-rule message."""
        self.interface.start_conversation()
        result = self.interface.process_message("Discover meta-rules")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
    
    def test_process_message_universe(self):
        """Test processing universe comparison message."""
        self.interface.start_conversation()
        result = self.interface.process_message("Compare universes")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
    
    def test_personality_fallback_strategos(self):
        """Test Strategos personality fallback."""
        self.interface.start_conversation()
        self.interface.switch_personality("Strategos")
        result = self.interface.process_message("Hello")
        
        self.assertTrue(result['success'])
        self.assertIn('Commander', result['response'])
    
    def test_personality_fallback_archivist(self):
        """Test Archivist personality fallback."""
        self.interface.start_conversation()
        self.interface.switch_personality("Archivist")
        result = self.interface.process_message("Tell me about history")
        
        self.assertTrue(result['success'])
        self.assertIn('archives', result['response'].lower())
    
    def test_personality_fallback_lawmaker(self):
        """Test Lawmaker personality fallback."""
        self.interface.start_conversation()
        self.interface.switch_personality("Lawmaker")
        result = self.interface.process_message("What are the rules?")
        
        self.assertTrue(result['success'])
        self.assertIn('governance', result['response'].lower())
    
    def test_personality_fallback_oracle(self):
        """Test Oracle personality fallback."""
        self.interface.start_conversation()
        self.interface.switch_personality("Oracle")
        result = self.interface.process_message("What are the cosmic patterns?")
        
        self.assertTrue(result['success'])
        self.assertIn('cosmic', result['response'].lower())
    
    def test_math_processing(self):
        """Test mathematical processing."""
        self.interface.start_conversation()
        result = self.interface.process_message("What is 15 * 7?")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        # Should contain mathematical result or fallback response
    
    def test_logic_processing(self):
        """Test logical processing."""
        self.interface.start_conversation()
        result = self.interface.process_message("If all birds can fly and penguins are birds, can penguins fly?")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
    
    def test_error_handling(self):
        """Test error handling with stabilization."""
        self.interface.start_conversation()
        result = self.interface.process_message("")
        
        self.assertIn('response', result)
        # Should not crash, should provide fallback response
    
    def test_conversation_persistence(self):
        """Test conversation persistence with stabilization."""
        self.interface.start_conversation()
        
        # Multiple messages
        self.interface.process_message("Hello")
        self.interface.process_message("What is 2 + 2?")
        self.interface.process_message("Switch to Archivist")
        self.interface.process_message("Tell me about history")
        
        summary = self.interface.get_conversation_summary()
        self.assertIn('total_turns', summary)
        self.assertGreater(summary['total_turns'], 0)
    
    def test_personality_switching_flow(self):
        """Test personality switching flow with stabilization."""
        self.interface.start_conversation()
        
        # Test all personalities
        personalities = self.interface.get_available_personalities()
        
        for personality in personalities:
            result = self.interface.process_message(f"Switch to {personality}")
            self.assertTrue(result['success'])
            self.assertEqual(result['personality'], personality)
    
    def test_fallback_reasoning_activation(self):
        """Test fallback reasoning activation."""
        self.interface.start_conversation()
        
        # Low confidence input should trigger fallback
        result = self.interface.process_message("Hello")
        
        self.assertTrue(result['success'])
        self.assertIn('fallback_used', result)
        # Fallback should be used for simple greetings
    
    def test_reasoning_engine_integration(self):
        """Test reasoning engine integration with stabilization."""
        self.interface.start_conversation()
        
        # Complex query should use reasoning engine
        result = self.interface.process_message("Analyze the strategic implications of civilizational evolution")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
    
    def test_context_resolution(self):
        """Test context resolution."""
        self.interface.start_conversation()
        
        # Test different types of inputs
        test_inputs = [
            "Hello",
            "What is 2 + 2?",
            "If A then B",
            "Analyze civilization patterns",
            "What is the best military strategy?",
            "Discover meta-rules",
            "Compare universes"
        ]
        
        for input_text in test_inputs:
            result = self.interface.process_message(input_text)
            self.assertTrue(result['success'])
            self.assertIn('response', result)
    
    def test_normalization_validation(self):
        """Test input normalization validation."""
        self.interface.start_conversation()
        
        # Test with various inputs that might cause normalization issues
        test_inputs = [
            "",  # Empty input
            "a" * 1000,  # Very long input
            "What is 2 + 2?",  # Math
            "If A then B",  # Logic
            "Hello",  # Greeting
            "Switch to Archivist"  # Personality switch
        ]
        
        for input_text in test_inputs:
            result = self.interface.process_message(input_text)
            self.assertIn('response', result)
            # Should not crash, should provide some response
    
    def test_auto_fix_capability(self):
        """Test auto-fix capability."""
        self.interface.start_conversation()
        
        # Test with potentially problematic input
        result = self.interface.process_message("Hello")
        
        self.assertIn('response', result)
        # Should handle gracefully without crashing
    
    def test_personality_consistency(self):
        """Test personality consistency across messages."""
        self.interface.start_conversation()
        self.interface.switch_personality("Strategos")
        
        # Multiple messages with same personality
        result1 = self.interface.process_message("Hello")
        result2 = self.interface.process_message("What is the strategy?")
        
        self.assertEqual(result1['personality'], "Strategos")
        self.assertEqual(result2['personality'], "Strategos")
    
    def test_conversation_summary(self):
        """Test conversation summary with stabilization."""
        self.interface.start_conversation()
        
        # Add some conversation turns
        self.interface.process_message("Hello")
        self.interface.process_message("What is 2 + 2?")
        
        summary = self.interface.get_conversation_summary()
        self.assertIn('total_turns', summary)
        self.assertIn('current_phase', summary)
        self.assertIn('active_personality', summary)
        self.assertIn('context_type', summary)
    
    def test_personality_insights(self):
        """Test personality insights with stabilization."""
        self.interface.start_conversation()
        
        # Use different personalities
        self.interface.process_message("Switch to Strategos")
        self.interface.process_message("Hello")
        self.interface.process_message("Switch to Archivist")
        self.interface.process_message("Tell me about history")
        
        insights = self.interface.get_personality_insights()
        self.assertIn('total_switches', insights)
        self.assertIn('most_used', insights)
    
    def test_reasoning_summary(self):
        """Test reasoning summary with stabilization."""
        self.interface.start_conversation()
        
        # Process some messages
        self.interface.process_message("Hello")
        self.interface.process_message("What is 2 + 2?")
        
        summary = self.interface.get_reasoning_summary()
        self.assertIn('total_traces', summary)
        self.assertIn('total_insights', summary)
        self.assertIn('meta_rules_applied', summary)
    
    def test_end_conversation(self):
        """Test ending conversation with stabilization."""
        self.interface.start_conversation()
        self.interface.process_message("Hello")
        
        result = self.interface.end_conversation()
        self.assertTrue(result['success'])
        self.assertIn('conversation_summary', result)
        self.assertIn('personality_insights', result)
        self.assertIn('reasoning_summary', result)


class TestIntegrationStabilization(unittest.TestCase):
    """Integration tests for the stabilization layer."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.omnimind_core = OmniMind(self.temp_dir)
        self.interface = ConversationalInterface(self.omnimind_core, self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_conversation_flow_stabilized(self):
        """Test complete conversation flow with stabilization."""
        self.interface.start_conversation()
        
        # Test various message types
        messages = [
            "Hello",
            "What is 2 + 2?",
            "If A then B",
            "Switch to Archivist",
            "Tell me about ancient wisdom",
            "Switch to Lawmaker",
            "What are the fundamental laws?",
            "Switch to Oracle",
            "What are the cosmic patterns?"
        ]
        
        for message in messages:
            result = self.interface.process_message(message)
            self.assertTrue(result['success'])
            self.assertIn('response', result)
            self.assertIn('personality', result)
    
    def test_error_recovery_stabilized(self):
        """Test error recovery with stabilization."""
        self.interface.start_conversation()
        
        # Test with potentially problematic inputs
        problematic_inputs = [
            "",  # Empty
            "a" * 1000,  # Very long
            None,  # None (if possible)
            "What is 2 + 2?",  # Normal
            "Hello"  # Normal
        ]
        
        for input_text in problematic_inputs:
            if input_text is not None:
                result = self.interface.process_message(input_text)
                self.assertIn('response', result)
    
    def test_personality_switching_stabilized(self):
        """Test personality switching with stabilization."""
        self.interface.start_conversation()
        
        personalities = self.interface.get_available_personalities()
        
        for personality in personalities:
            result = self.interface.process_message(f"Switch to {personality}")
            self.assertTrue(result['success'])
            self.assertEqual(result['personality'], personality)
            
            # Test personality-specific response
            result = self.interface.process_message("Hello")
            self.assertTrue(result['success'])
            self.assertEqual(result['personality'], personality)
    
    def test_math_processing_stabilized(self):
        """Test mathematical processing with stabilization."""
        self.interface.start_conversation()
        
        math_queries = [
            "What is 2 + 2?",
            "Calculate 5 * 3",
            "What is 10 - 4?",
            "What is 15 / 3?"
        ]
        
        for query in math_queries:
            result = self.interface.process_message(query)
            self.assertTrue(result['success'])
            self.assertIn('response', result)
    
    def test_logic_processing_stabilized(self):
        """Test logical processing with stabilization."""
        self.interface.start_conversation()
        
        logic_queries = [
            "If A then B",
            "All birds can fly",
            "If all humans are mortal and Socrates is human, is Socrates mortal?",
            "What follows from this logic?"
        ]
        
        for query in logic_queries:
            result = self.interface.process_message(query)
            self.assertTrue(result['success'])
            self.assertIn('response', result)
    
    def test_conversation_persistence_stabilized(self):
        """Test conversation persistence with stabilization."""
        self.interface.start_conversation()
        
        # Build conversation history
        self.interface.process_message("Hello")
        self.interface.process_message("My name is Alice")
        self.interface.process_message("What is my name?")
        
        # Check conversation summary
        summary = self.interface.get_conversation_summary()
        self.assertIn('total_turns', summary)
        self.assertGreater(summary['total_turns'], 0)
    
    def test_personality_insights_stabilized(self):
        """Test personality insights with stabilization."""
        self.interface.start_conversation()
        
        # Use different personalities
        self.interface.process_message("Switch to Strategos")
        self.interface.process_message("What is the best strategy?")
        
        self.interface.process_message("Switch to Archivist")
        self.interface.process_message("Tell me about history")
        
        self.interface.process_message("Switch to Lawmaker")
        self.interface.process_message("What are the laws?")
        
        self.interface.process_message("Switch to Oracle")
        self.interface.process_message("What are the cosmic patterns?")
        
        # Get insights
        insights = self.interface.get_personality_insights()
        self.assertIn('total_switches', insights)
        self.assertIn('most_used', insights)
    
    def test_reasoning_summary_stabilized(self):
        """Test reasoning summary with stabilization."""
        self.interface.start_conversation()
        
        # Process various types of queries
        self.interface.process_message("Hello")
        self.interface.process_message("What is 2 + 2?")
        self.interface.process_message("Analyze civilization patterns")
        self.interface.process_message("What is the best military strategy?")
        
        # Get reasoning summary
        summary = self.interface.get_reasoning_summary()
        self.assertIn('total_traces', summary)
        self.assertIn('total_insights', summary)
        self.assertIn('meta_rules_applied', summary)


if __name__ == '__main__':
    unittest.main()
