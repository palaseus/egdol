"""
Comprehensive Test Suite for Conversational Personality Layer
Tests all components of the conversational interface.
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
    ResponseGenerator, ResponseStyle, ConversationalReasoningEngine
)


class TestConversationalInterface(unittest.TestCase):
    """Test the main conversational interface."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.omnimind_core = OmniMind(self.temp_dir)
        self.interface = ConversationalInterface(self.omnimind_core, self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test conversational interface initialization."""
        self.assertIsNotNone(self.interface.intent_parser)
        self.assertIsNotNone(self.interface.personality_framework)
        self.assertIsNotNone(self.interface.reasoning_engine)
        self.assertIsNotNone(self.interface.response_generator)
        self.assertIsNone(self.interface.current_session)
    
    def test_start_conversation(self):
        """Test starting a new conversation."""
        session_id = self.interface.start_conversation()
        self.assertIsNotNone(session_id)
        self.assertIsNotNone(self.interface.current_session)
        self.assertEqual(self.interface.current_session.session_id, session_id)
    
    def test_process_message_basic(self):
        """Test processing a basic message."""
        self.interface.start_conversation()
        result = self.interface.process_message("Hello")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
        self.assertIn('session_id', result)
    
    def test_process_message_with_personality_switch(self):
        """Test processing message with personality switch."""
        self.interface.start_conversation()
        result = self.interface.process_message("Switch to Archivist")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('personality', result)
    
    def test_process_message_civilizational_query(self):
        """Test processing civilizational query."""
        self.interface.start_conversation()
        result = self.interface.process_message("Analyze the evolution of civilizations")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
        self.assertIn('reasoning_trace', result)
    
    def test_process_message_strategic_analysis(self):
        """Test processing strategic analysis."""
        self.interface.start_conversation()
        result = self.interface.process_message("What is the best military strategy?")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
    
    def test_process_message_meta_rule_discovery(self):
        """Test processing meta-rule discovery."""
        self.interface.start_conversation()
        result = self.interface.process_message("Discover the meta-rules governing this system")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
    
    def test_process_message_universe_comparison(self):
        """Test processing universe comparison."""
        self.interface.start_conversation()
        result = self.interface.process_message("Compare different universes")
        
        self.assertTrue(result['success'])
        self.assertIn('response', result)
    
    def test_error_handling(self):
        """Test error handling and auto-fix."""
        self.interface.start_conversation()
        
        # Test with invalid input that might cause errors
        result = self.interface.process_message("")
        self.assertIn('response', result)
    
    def test_conversation_summary(self):
        """Test getting conversation summary."""
        self.interface.start_conversation()
        self.interface.process_message("Hello")
        self.interface.process_message("How are you?")
        
        summary = self.interface.get_conversation_summary()
        self.assertIn('total_turns', summary)
        self.assertIn('current_phase', summary)
        self.assertIn('active_personality', summary)
    
    def test_personality_insights(self):
        """Test getting personality insights."""
        self.interface.start_conversation()
        self.interface.process_message("Hello")
        
        insights = self.interface.get_personality_insights()
        self.assertIsInstance(insights, dict)
    
    def test_reasoning_summary(self):
        """Test getting reasoning summary."""
        self.interface.start_conversation()
        self.interface.process_message("Analyze civilizational patterns")
        
        summary = self.interface.get_reasoning_summary()
        self.assertIsInstance(summary, dict)
    
    def test_switch_personality(self):
        """Test manual personality switching."""
        self.interface.start_conversation()
        
        # Switch to Archivist
        success = self.interface.switch_personality("Archivist")
        self.assertTrue(success)
        self.assertEqual(self.interface.current_session.active_personality, "Archivist")
    
    def test_get_available_personalities(self):
        """Test getting available personalities."""
        personalities = self.interface.get_available_personalities()
        self.assertIsInstance(personalities, list)
        self.assertIn("Strategos", personalities)
        self.assertIn("Archivist", personalities)
        self.assertIn("Lawmaker", personalities)
        self.assertIn("Oracle", personalities)
    
    def test_end_conversation(self):
        """Test ending conversation."""
        self.interface.start_conversation()
        self.interface.process_message("Hello")
        
        result = self.interface.end_conversation()
        self.assertTrue(result['success'])
        self.assertIn('conversation_summary', result)
        self.assertIsNone(self.interface.current_session)


class TestPersonalityFramework(unittest.TestCase):
    """Test the personality framework."""
    
    def setUp(self):
        self.framework = PersonalityFramework()
    
    def test_initialization(self):
        """Test personality framework initialization."""
        self.assertIsNotNone(self.framework.personalities)
        self.assertIn("Strategos", self.framework.personalities)
        self.assertIn("Archivist", self.framework.personalities)
        self.assertIn("Lawmaker", self.framework.personalities)
        self.assertIn("Oracle", self.framework.personalities)
        self.assertEqual(self.framework.active_personality, "Strategos")
    
    def test_get_personality(self):
        """Test getting personality by name."""
        strategos = self.framework.get_personality("Strategos")
        self.assertIsNotNone(strategos)
        self.assertEqual(strategos.name, "Strategos")
        self.assertEqual(strategos.personality_type, PersonalityType.STRATEGOS)
    
    def test_get_active_personality(self):
        """Test getting active personality."""
        active = self.framework.get_active_personality()
        self.assertIsNotNone(active)
        self.assertEqual(active.name, "Strategos")
    
    def test_switch_personality(self):
        """Test switching personality."""
        success = self.framework.switch_personality("Archivist")
        self.assertTrue(success)
        self.assertEqual(self.framework.active_personality, "Archivist")
        
        # Test switching to non-existent personality
        success = self.framework.switch_personality("NonExistent")
        self.assertFalse(success)
    
    def test_get_available_personalities(self):
        """Test getting available personalities."""
        personalities = self.framework.get_available_personalities()
        self.assertIn("Strategos", personalities)
        self.assertIn("Archivist", personalities)
        self.assertIn("Lawmaker", personalities)
        self.assertIn("Oracle", personalities)
    
    def test_get_personality_by_type(self):
        """Test getting personality by type."""
        strategos = self.framework.get_personality_by_type(PersonalityType.STRATEGOS)
        self.assertIsNotNone(strategos)
        self.assertEqual(strategos.name, "Strategos")
    
    def test_get_personality_recommendation(self):
        """Test getting personality recommendation."""
        context = {'context_type': 'strategic', 'domain': 'military'}
        recommendation = self.framework.get_personality_recommendation(context)
        self.assertEqual(recommendation, "Strategos")
        
        context = {'context_type': 'historical', 'domain': 'philosophy'}
        recommendation = self.framework.get_personality_recommendation(context)
        self.assertEqual(recommendation, "Archivist")
    
    def test_personality_usage_stats(self):
        """Test getting personality usage statistics."""
        self.framework.record_personality_usage("Strategos", time.time())
        self.framework.record_personality_usage("Archivist", time.time())
        
        stats = self.framework.get_personality_usage_stats()
        self.assertIn("Strategos", stats)
        self.assertIn("Archivist", stats)
    
    def test_personality_insights(self):
        """Test getting personality insights."""
        self.framework.record_personality_usage("Strategos", time.time())
        self.framework.record_personality_usage("Archivist", time.time())
        
        insights = self.framework.get_personality_insights()
        self.assertIn('total_switches', insights)
        self.assertIn('most_used', insights)
    
    def test_create_custom_personality(self):
        """Test creating custom personality."""
        custom = self.framework.create_custom_personality(
            name="Custom",
            description="A custom personality",
            archetype="Custom",
            epistemic_style="Custom",
            domain_expertise=["custom"],
            response_style={"tone": "custom"}
        )
        
        self.assertIsNotNone(custom)
        self.assertEqual(custom.name, "Custom")
        self.assertIn("Custom", self.framework.get_available_personalities())
    
    def test_evolve_personality(self):
        """Test evolving personality."""
        evolution_data = {
            'response_style_updates': {'tone': 'evolved'},
            'new_domains': ['evolved_domain'],
            'new_patterns': ['evolved_pattern']
        }
        
        success = self.framework.evolve_personality("Strategos", evolution_data)
        self.assertTrue(success)
    
    def test_personality_compatibility(self):
        """Test personality compatibility scoring."""
        context = {'domain': 'military', 'complexity_level': 0.8}
        compatibility = self.framework.get_personality_compatibility("Strategos", context)
        self.assertGreater(compatibility, 0.5)


class TestIntentParser(unittest.TestCase):
    """Test the intent parser."""
    
    def setUp(self):
        self.parser = IntentParser()
    
    def test_parse_question(self):
        """Test parsing question intent."""
        intent = self.parser.parse("What is the meaning of life?")
        self.assertEqual(intent.intent_type, IntentType.QUESTION)
        self.assertGreaterEqual(intent.confidence, 0.5)
        self.assertIn("meaning", intent.entities)
    
    def test_parse_command(self):
        """Test parsing command intent."""
        intent = self.parser.parse("Calculate 2 + 2")
        self.assertEqual(intent.intent_type, IntentType.COMMAND)
        self.assertGreaterEqual(intent.confidence, 0.5)
    
    def test_parse_statement(self):
        """Test parsing statement intent."""
        intent = self.parser.parse("I think this is correct")
        self.assertEqual(intent.intent_type, IntentType.STATEMENT)
        self.assertGreaterEqual(intent.confidence, 0.5)
    
    def test_parse_personality_switch(self):
        """Test parsing personality switch intent."""
        intent = self.parser.parse("Switch to Archivist")
        self.assertEqual(intent.intent_type, IntentType.PERSONALITY_SWITCH)
        self.assertGreaterEqual(intent.confidence, 0.5)
        self.assertEqual(intent.personality_hint, "archivist")
    
    def test_parse_civilizational_query(self):
        """Test parsing civilizational query intent."""
        intent = self.parser.parse("Analyze the evolution of civilizations")
        self.assertEqual(intent.intent_type, IntentType.CIVILIZATIONAL_QUERY)
        self.assertGreaterEqual(intent.confidence, 0.5)
    
    def test_parse_strategic_analysis(self):
        """Test parsing strategic analysis intent."""
        intent = self.parser.parse("What is the best military strategy?")
        self.assertEqual(intent.intent_type, IntentType.STRATEGIC_ANALYSIS)
        self.assertGreaterEqual(intent.confidence, 0.5)
    
    def test_parse_meta_rule_discovery(self):
        """Test parsing meta-rule discovery intent."""
        intent = self.parser.parse("Discover the meta-rules governing this system")
        self.assertEqual(intent.intent_type, IntentType.META_RULE_DISCOVERY)
        self.assertGreaterEqual(intent.confidence, 0.5)
    
    def test_parse_universe_comparison(self):
        """Test parsing universe comparison intent."""
        intent = self.parser.parse("Compare different universes")
        self.assertEqual(intent.intent_type, IntentType.UNIVERSE_COMPARISON)
        self.assertGreaterEqual(intent.confidence, 0.5)
    
    def test_extract_entities(self):
        """Test entity extraction."""
        entities = self.parser._extract_entities("Alice is 25 years old")
        self.assertIn("Alice", entities)
        self.assertIn("25", entities)
    
    def test_extract_context_clues(self):
        """Test context clue extraction."""
        clues = self.parser._extract_context_clues("Calculate the mathematical equation")
        self.assertIn("mathematical", clues)
    
    def test_determine_personality_hint(self):
        """Test personality hint determination."""
        hint = self.parser._determine_personality_hint("What is the best military strategy?")
        self.assertEqual(hint, "strategos")
        
        hint = self.parser._determine_personality_hint("Tell me about history")
        self.assertEqual(hint, "archivist")
    
    def test_calculate_complexity(self):
        """Test complexity calculation."""
        complexity = self.parser._calculate_complexity("Simple question")
        self.assertLessEqual(complexity, 0.5)
        
        complexity = self.parser._calculate_complexity("This is a very complex and sophisticated analysis")
        self.assertGreater(complexity, 0.5)
    
    def test_determine_domain(self):
        """Test domain determination."""
        domain = self.parser._determine_domain("Calculate 2 + 2")
        self.assertIn(domain, ["mathematical", None])  # May be None if no domain detected
        
        domain = self.parser._determine_domain("What is the meaning of life?")
        self.assertIn(domain, ["philosophical", None])  # May be None if no domain detected
    
    def test_calculate_urgency(self):
        """Test urgency calculation."""
        urgency = self.parser._calculate_urgency("This is urgent")
        self.assertGreater(urgency, 0.7)
        
        urgency = self.parser._calculate_urgency("This can wait")
        self.assertLessEqual(urgency, 0.5)
    
    def test_determine_emotional_tone(self):
        """Test emotional tone determination."""
        tone = self.parser._determine_emotional_tone("I am excited about this")
        self.assertEqual(tone, "excited")
        
        tone = self.parser._determine_emotional_tone("I am worried about this")
        self.assertEqual(tone, "concerned")


class TestConversationState(unittest.TestCase):
    """Test conversation state management."""
    
    def setUp(self):
        self.context = ConversationContext(
            context_type=ContextType.GENERAL,
            domain="general",
            complexity_level=0.5,
            emotional_tone="neutral"
        )
        self.state = ConversationState(
            session_id="test_session",
            created_at=datetime.now(),
            current_phase=ConversationPhase.GREETING,
            active_personality="Strategos",
            context=self.context
        )
    
    def test_initialization(self):
        """Test conversation state initialization."""
        self.assertEqual(self.state.session_id, "test_session")
        self.assertEqual(self.state.current_phase, ConversationPhase.GREETING)
        self.assertEqual(self.state.active_personality, "Strategos")
        self.assertEqual(self.state.context, self.context)
    
    def test_add_turn(self):
        """Test adding conversation turn."""
        from egdol.omnimind.conversational.conversation_state import ConversationTurn
        
        turn = ConversationTurn(
            turn_id="turn_1",
            timestamp=datetime.now(),
            user_input="Hello",
            system_response="Hi there!",
            intent="greeting",
            personality_used="Strategos"
        )
        
        self.state.add_turn(turn)
        self.assertEqual(len(self.state.conversation_history), 1)
        self.assertEqual(self.state.conversation_history[0], turn)
    
    def test_switch_personality(self):
        """Test switching personality."""
        self.state.switch_personality("Archivist")
        self.assertEqual(self.state.active_personality, "Archivist")
        self.assertEqual(len(self.state.personality_history), 1)
        self.assertEqual(self.state.personality_history[0][0], "Archivist")
    
    def test_update_context(self):
        """Test updating context."""
        new_context = ConversationContext(
            context_type=ContextType.STRATEGIC,
            domain="military",
            complexity_level=0.8,
            emotional_tone="focused"
        )
        
        self.state.update_context(new_context)
        self.assertEqual(self.state.context, new_context)
        self.assertEqual(len(self.state.context_transitions), 1)
    
    def test_get_recent_context(self):
        """Test getting recent context."""
        from egdol.omnimind.conversational.conversation_state import ConversationTurn
        
        # Add multiple turns
        for i in range(10):
            turn = ConversationTurn(
                turn_id=f"turn_{i}",
                timestamp=datetime.now(),
                user_input=f"Input {i}",
                system_response=f"Response {i}",
                intent="general",
                personality_used="Strategos"
            )
            self.state.add_turn(turn)
        
        recent = self.state.get_recent_context(5)
        self.assertEqual(len(recent), 5)
    
    def test_get_personality_usage(self):
        """Test getting personality usage statistics."""
        self.state.switch_personality("Archivist")
        self.state.switch_personality("Strategos")
        self.state.switch_personality("Archivist")
        
        usage = self.state.get_personality_usage()
        self.assertEqual(usage["Archivist"], 2)
        self.assertEqual(usage["Strategos"], 1)
    
    def test_get_context_evolution(self):
        """Test getting context evolution."""
        self.state.update_context(ConversationContext(
            context_type=ContextType.STRATEGIC,
            domain="military",
            complexity_level=0.8,
            emotional_tone="focused"
        ))
        
        evolution = self.state.get_context_evolution()
        self.assertEqual(len(evolution), 1)
        self.assertEqual(evolution[0][0], ContextType.STRATEGIC)
    
    def test_get_conversation_summary(self):
        """Test getting conversation summary."""
        summary = self.state.get_conversation_summary()
        self.assertIn('session_id', summary)
        self.assertIn('duration', summary)
        self.assertIn('total_turns', summary)
        self.assertIn('current_phase', summary)
        self.assertIn('active_personality', summary)
    
    def test_should_switch_personality(self):
        """Test personality switching logic."""
        personalities = ["Strategos", "Archivist", "Lawmaker", "Oracle"]
        
        # Test strategic context
        recommended = self.state.should_switch_personality("What is the best military strategy?", personalities)
        self.assertEqual(recommended, "strategos")
        
        # Test historical context
        recommended = self.state.should_switch_personality("Tell me about ancient history", personalities)
        self.assertEqual(recommended, "archivist")
    
    def test_evolve_phase(self):
        """Test conversation phase evolution."""
        # Test greeting phase
        phase = self.state.evolve_phase("Hello")
        self.assertEqual(phase, ConversationPhase.GREETING)
        
        # Test exploration phase
        phase = self.state.evolve_phase("Let's explore this topic")
        self.assertEqual(phase, ConversationPhase.EXPLORATION)
        
        # Test deep dive phase
        phase = self.state.evolve_phase("Let's analyze this in detail")
        self.assertEqual(phase, ConversationPhase.DEEP_DIVE)


class TestResponseGenerator(unittest.TestCase):
    """Test the response generator."""
    
    def setUp(self):
        self.generator = ResponseGenerator()
        self.personality = Personality(
            name="TestPersonality",
            personality_type=PersonalityType.STRATEGOS,
            description="Test personality",
            archetype="Test",
            epistemic_style="Test",
            response_style={"tone": "authoritative"}
        )
    
    def test_initialization(self):
        """Test response generator initialization."""
        self.assertIsNotNone(self.generator.response_templates)
        self.assertIsNotNone(self.generator.personality_phrases)
        self.assertIsNotNone(self.generator.reasoning_connectors)
    
    def test_generate_response_basic(self):
        """Test generating basic response."""
        response = self.generator.generate_response(
            personality=self.personality,
            context={'intent_type': 'general'}
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_generate_response_with_reasoning(self):
        """Test generating response with reasoning trace."""
        from egdol.omnimind.conversational.reasoning_engine import ReasoningTrace
        
        reasoning_trace = ReasoningTrace(
            step_id="test_step",
            timestamp=datetime.now(),
            reasoning_type="test",
            input_data={},
            processing_steps=["Step 1", "Step 2"],
            output_data={"result": "test_result"},
            confidence=0.8
        )
        
        response = self.generator.generate_response(
            personality=self.personality,
            reasoning_trace=reasoning_trace,
            context={'intent_type': 'civilizational_analysis'}
        )
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_generate_reasoning_explanation(self):
        """Test generating reasoning explanation."""
        from egdol.omnimind.conversational.reasoning_engine import ReasoningTrace
        
        reasoning_trace = ReasoningTrace(
            step_id="test_step",
            timestamp=datetime.now(),
            reasoning_type="test",
            input_data={},
            processing_steps=["Step 1", "Step 2"],
            output_data={"result": "test_result"},
            confidence=0.8,
            meta_insights=["Insight 1", "Insight 2"]
        )
        
        explanation = self.generator.generate_reasoning_explanation(reasoning_trace)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        self.assertIn("Step 1", explanation)
        self.assertIn("Step 2", explanation)
    
    def test_generate_personality_switch_response(self):
        """Test generating personality switch response."""
        response = self.generator.generate_personality_switch_response("Strategos", "Archivist")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_generate_error_response(self):
        """Test generating error response."""
        response = self.generator.generate_error_response("Test error", self.personality)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIn("Test error", response)


class TestConversationalReasoningEngine(unittest.TestCase):
    """Test the conversational reasoning engine."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.omnimind_core = OmniMind(self.temp_dir)
        self.reasoning_engine = ConversationalReasoningEngine(self.omnimind_core)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test reasoning engine initialization."""
        self.assertIsNotNone(self.reasoning_engine.omnimind_core)
        self.assertIsNotNone(self.reasoning_engine.reasoning_traces)
        self.assertIsNotNone(self.reasoning_engine.civilizational_insights)
        self.assertIsNotNone(self.reasoning_engine.meta_rules_applied)
    
    def test_process_civilizational_query(self):
        """Test processing civilizational query."""
        context = {'domain': 'civilization', 'complexity_level': 0.7}
        result = self.reasoning_engine.process_civilizational_query("Analyze civilization patterns", context)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('reasoning_trace', result)
        self.assertIn('processing_time', result)
    
    def test_process_strategic_analysis(self):
        """Test processing strategic analysis."""
        context = {'domain': 'military', 'complexity_level': 0.8}
        result = self.reasoning_engine.process_strategic_analysis("What is the best strategy?", context)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('reasoning_trace', result)
        self.assertIn('processing_time', result)
    
    def test_process_meta_rule_discovery(self):
        """Test processing meta-rule discovery."""
        context = {'domain': 'governance', 'complexity_level': 0.9}
        result = self.reasoning_engine.process_meta_rule_discovery("Discover meta-rules", context)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('reasoning_trace', result)
        self.assertIn('processing_time', result)
    
    def test_process_universe_comparison(self):
        """Test processing universe comparison."""
        context = {'domain': 'cosmic', 'complexity_level': 0.8}
        result = self.reasoning_engine.process_universe_comparison("Compare universes", context)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('reasoning_trace', result)
        self.assertIn('processing_time', result)
    
    def test_get_reasoning_summary(self):
        """Test getting reasoning summary."""
        # Process some queries to generate traces
        self.reasoning_engine.process_civilizational_query("Test query", {})
        
        summary = self.reasoning_engine.get_reasoning_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('total_traces', summary)
        self.assertIn('total_insights', summary)
        self.assertIn('meta_rules_applied', summary)


class TestIntegration(unittest.TestCase):
    """Integration tests for the conversational layer."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.omnimind_core = OmniMind(self.temp_dir)
        self.interface = ConversationalInterface(self.omnimind_core, self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_conversation_flow(self):
        """Test complete conversation flow."""
        # Start conversation
        session_id = self.interface.start_conversation()
        self.assertIsNotNone(session_id)
        
        # Process various message types
        messages = [
            "Hello",
            "What is the best military strategy?",
            "Switch to Archivist",
            "Tell me about ancient civilizations",
            "Discover the meta-rules governing this system",
            "Compare different universes"
        ]
        
        for message in messages:
            result = self.interface.process_message(message)
            self.assertTrue(result['success'])
            self.assertIn('response', result)
            self.assertIn('personality', result)
        
        # Get conversation summary
        summary = self.interface.get_conversation_summary()
        self.assertIn('total_turns', summary)
        self.assertGreater(summary['total_turns'], 0)
        
        # End conversation
        end_result = self.interface.end_conversation()
        self.assertTrue(end_result['success'])
    
    def test_personality_switching_flow(self):
        """Test personality switching throughout conversation."""
        self.interface.start_conversation()
        
        # Test all personalities
        personalities = self.interface.get_available_personalities()
        
        for personality in personalities:
            result = self.interface.process_message(f"Switch to {personality}")
            self.assertTrue(result['success'])
            self.assertEqual(result['personality'], personality)
    
    def test_reasoning_integration(self):
        """Test integration with reasoning engine."""
        self.interface.start_conversation()
        
        # Test civilizational query
        result = self.interface.process_message("Analyze the evolution of civilizations")
        self.assertTrue(result['success'])
        self.assertIn('reasoning_trace', result)
        
        # Test strategic analysis
        result = self.interface.process_message("What is the best military strategy?")
        self.assertTrue(result['success'])
        
        # Test meta-rule discovery
        result = self.interface.process_message("Discover the meta-rules governing this system")
        self.assertTrue(result['success'])
        
        # Test universe comparison
        result = self.interface.process_message("Compare different universes")
        self.assertTrue(result['success'])
    
    def test_error_recovery(self):
        """Test error recovery and auto-fix."""
        self.interface.start_conversation()
        
        # Test with potentially problematic input
        result = self.interface.process_message("")
        self.assertIn('response', result)
        
        # Test with very long input
        long_input = "A" * 1000
        result = self.interface.process_message(long_input)
        self.assertIn('response', result)
    
    def test_conversation_persistence(self):
        """Test conversation persistence across multiple interactions."""
        self.interface.start_conversation()
        
        # Build conversation history
        self.interface.process_message("Hello")
        self.interface.process_message("My name is Alice")
        self.interface.process_message("What is my name?")
        
        # Check conversation summary
        summary = self.interface.get_conversation_summary()
        self.assertGreater(summary['total_turns'], 0)
    
    def test_personality_insights(self):
        """Test personality insights generation."""
        self.interface.start_conversation()
        
        # Use different personalities
        self.interface.process_message("Switch to Strategos")
        self.interface.process_message("What is the best military strategy?")
        
        self.interface.process_message("Switch to Archivist")
        self.interface.process_message("Tell me about ancient history")
        
        # Get insights
        insights = self.interface.get_personality_insights()
        self.assertIsInstance(insights, dict)
        self.assertIn('total_switches', insights)
    
    def test_reasoning_summary(self):
        """Test reasoning summary generation."""
        self.interface.start_conversation()
        
        # Process various reasoning queries
        self.interface.process_message("Analyze civilization patterns")
        self.interface.process_message("What is the best strategy?")
        self.interface.process_message("Discover meta-rules")
        
        # Get reasoning summary
        summary = self.interface.get_reasoning_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('total_traces', summary)


if __name__ == '__main__':
    unittest.main()
