"""
Tests for OmniMind Core
Tests the complete OmniMind chatbot system.
"""

import unittest
import tempfile
import os
import time
from egdol.omnimind import OmniMind, NLUTranslator, ConversationMemory, SkillRouter
from egdol.omnimind.skills import MathSkill, LogicSkill, GeneralSkill, CodeSkill, FileSkill


class OmniMindCoreTests(unittest.TestCase):
    """Test the OmniMind core functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.omnimind = OmniMind(self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test OmniMind initialization."""
        self.assertIsNotNone(self.omnimind.engine)
        self.assertIsNotNone(self.omnimind.interpreter)
        self.assertIsNotNone(self.omnimind.dsl)
        self.assertIsNotNone(self.omnimind.nlu_translator)
        self.assertIsNotNone(self.omnimind.memory)
        self.assertIsNotNone(self.omnimind.router)
        
    def test_process_input_basic(self):
        """Test basic input processing."""
        response = self.omnimind.process_input("Hello")
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        self.assertIn('reasoning', response)
        
    def test_process_input_math(self):
        """Test math input processing."""
        response = self.omnimind.process_input("Calculate 2 + 2")
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        
    def test_process_input_logic(self):
        """Test logic input processing."""
        response = self.omnimind.process_input("Alice is a human")
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        
    def test_process_input_question(self):
        """Test question input processing."""
        response = self.omnimind.process_input("What is 5 + 3?")
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        
    def test_conversation_history(self):
        """Test conversation history tracking."""
        self.omnimind.process_input("Hello")
        self.omnimind.process_input("How are you?")
        
        history = self.omnimind.get_conversation_history()
        self.assertEqual(len(history), 4)  # 2 user inputs + 2 responses
        
    def test_verbose_mode(self):
        """Test verbose mode."""
        self.omnimind.set_verbose_mode(True)
        self.assertTrue(self.omnimind.verbose_mode)
        
        self.omnimind.set_verbose_mode(False)
        self.assertFalse(self.omnimind.verbose_mode)
        
    def test_explain_mode(self):
        """Test explain mode."""
        self.omnimind.set_explain_mode(True)
        self.assertTrue(self.omnimind.explain_mode)
        
        self.omnimind.set_explain_mode(False)
        self.assertFalse(self.omnimind.explain_mode)


class NLUTranslatorTests(unittest.TestCase):
    """Test the NLU translator."""
    
    def setUp(self):
        self.translator = NLUTranslator()
        
    def test_translate_question(self):
        """Test translating questions."""
        dsl = self.translator.translate("Who is Alice?")
        self.assertIsNotNone(dsl)
        self.assertIn("who is", dsl.lower())
        
    def test_translate_fact(self):
        """Test translating facts."""
        dsl = self.translator.translate("Alice is a human")
        self.assertIsNotNone(dsl)
        self.assertIn("alice", dsl.lower())
        
    def test_translate_rule(self):
        """Test translating rules."""
        dsl = self.translator.translate("If X is human then X is mortal")
        self.assertIsNotNone(dsl)
        self.assertIn("if", dsl.lower())
        
    def test_detect_intent(self):
        """Test intent detection."""
        intent = self.translator.detect_intent("Who is Alice?")
        self.assertEqual(intent, 'question')
        
        intent = self.translator.detect_intent("Alice is human")
        self.assertEqual(intent, 'fact_assertion')
        
    def test_extract_entities(self):
        """Test entity extraction."""
        entities = self.translator.extract_entities("Alice is 25 years old")
        self.assertGreater(len(entities), 0)
        
    def test_validate_dsl(self):
        """Test DSL validation."""
        valid_dsl = "who is alice?"
        invalid_dsl = "invalid dsl"
        
        self.assertTrue(self.translator.validate_dsl(valid_dsl))
        self.assertFalse(self.translator.validate_dsl(invalid_dsl))


class ConversationMemoryTests(unittest.TestCase):
    """Test conversation memory."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory = ConversationMemory(self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_store_input(self):
        """Test storing user input."""
        self.memory.store_input("Hello", "session1")
        memories = self.memory.get_recent_memories("session1")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]['content'], "Hello")
        
    def test_store_response(self):
        """Test storing assistant response."""
        self.memory.store_response("Hi there!", "session1")
        memories = self.memory.get_recent_memories("session1")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]['content'], "Hi there!")
        
    def test_store_fact(self):
        """Test storing facts."""
        fact_id = self.memory.store_fact("Alice is human", "session1")
        self.assertIsNotNone(fact_id)
        
        facts = self.memory.get_all_facts()
        self.assertGreater(len(facts), 0)
        
    def test_search_memories(self):
        """Test searching memories."""
        self.memory.store_input("Hello world", "session1")
        self.memory.store_input("Goodbye world", "session1")
        
        results = self.memory.search_memories("world")
        self.assertGreater(len(results), 0)
        
    def test_forget_pattern(self):
        """Test forgetting memories by pattern."""
        self.memory.store_input("Hello world", "session1")
        self.memory.store_input("Goodbye world", "session1")
        
        deleted_count = self.memory.forget_pattern("world")
        self.assertGreater(deleted_count, 0)
        
    def test_get_stats(self):
        """Test getting memory statistics."""
        self.memory.store_input("Hello", "session1")
        stats = self.memory.get_stats()
        self.assertIn('total_sessions', stats)
        self.assertIn('total_memories', stats)


class SkillRouterTests(unittest.TestCase):
    """Test the skill router."""
    
    def setUp(self):
        self.router = SkillRouter()
        
    def test_route_math(self):
        """Test routing math input."""
        response = self.router.route("Calculate 2 + 2", "math", {})
        self.assertIsInstance(response, dict)
        self.assertIn('handled', response)
        
    def test_route_logic(self):
        """Test routing logic input."""
        response = self.router.route("Alice is human", "fact_assertion", {})
        self.assertIsInstance(response, dict)
        self.assertIn('handled', response)
        
    def test_route_general(self):
        """Test routing general input."""
        response = self.router.route("Hello", "general", {})
        self.assertIsInstance(response, dict)
        self.assertIn('handled', response)
        
    def test_get_loaded_skills(self):
        """Test getting loaded skills."""
        skills = self.router.get_loaded_skills()
        self.assertIsInstance(skills, list)
        self.assertGreater(len(skills), 0)
        
    def test_get_skill_info(self):
        """Test getting skill information."""
        info = self.router.get_skill_info("math")
        self.assertIsNotNone(info)
        self.assertIn('name', info)
        self.assertIn('description', info)


class MathSkillTests(unittest.TestCase):
    """Test the math skill."""
    
    def setUp(self):
        self.skill = MathSkill()
        
    def test_can_handle_math(self):
        """Test math skill can handle math input."""
        self.assertTrue(self.skill.can_handle("Calculate 2 + 2", "math", {}))
        self.assertTrue(self.skill.can_handle("What is 5 * 3?", "question", {}))
        self.assertTrue(self.skill.can_handle("2 + 2", "general", {}))
        
    def test_can_handle_non_math(self):
        """Test math skill doesn't handle non-math input."""
        self.assertFalse(self.skill.can_handle("Hello world", "general", {}))
        self.assertFalse(self.skill.can_handle("Alice is human", "fact", {}))
        
    def test_handle_arithmetic(self):
        """Test handling arithmetic."""
        response = self.skill.handle("Calculate 2 + 2", "math", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        self.assertIn('reasoning', response)
        
    def test_handle_complex_math(self):
        """Test handling complex math."""
        response = self.skill.handle("What is 5 * 3 + 2?", "question", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)


class LogicSkillTests(unittest.TestCase):
    """Test the logic skill."""
    
    def setUp(self):
        self.skill = LogicSkill()
        
    def test_can_handle_logic(self):
        """Test logic skill can handle logic input."""
        self.assertTrue(self.skill.can_handle("Alice is human", "fact_assertion", {}))
        self.assertTrue(self.skill.can_handle("If X is human then X is mortal", "rule", {}))
        self.assertTrue(self.skill.can_handle("Why is Alice mortal?", "question", {}))
        
    def test_can_handle_non_logic(self):
        """Test logic skill doesn't handle non-logic input."""
        self.assertFalse(self.skill.can_handle("Calculate 2 + 2", "math", {}))
        self.assertFalse(self.skill.can_handle("Hello world", "general", {}))
        
    def test_handle_fact(self):
        """Test handling facts."""
        response = self.skill.handle("Alice is human", "fact_assertion", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        
    def test_handle_rule(self):
        """Test handling rules."""
        response = self.skill.handle("If X is human then X is mortal", "rule", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)


class GeneralSkillTests(unittest.TestCase):
    """Test the general skill."""
    
    def setUp(self):
        self.skill = GeneralSkill()
        
    def test_can_handle_always(self):
        """Test general skill can handle any input."""
        self.assertTrue(self.skill.can_handle("Hello", "general", {}))
        self.assertTrue(self.skill.can_handle("Calculate 2 + 2", "math", {}))
        self.assertTrue(self.skill.can_handle("Alice is human", "fact", {}))
        
    def test_handle_greeting(self):
        """Test handling greetings."""
        response = self.skill.handle("Hello", "general", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        self.assertIn('greeting', response['metadata']['type'])
        
    def test_handle_farewell(self):
        """Test handling farewells."""
        response = self.skill.handle("Goodbye", "general", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        self.assertIn('farewell', response['metadata']['type'])
        
    def test_handle_capabilities(self):
        """Test handling capability questions."""
        response = self.skill.handle("What can you do?", "question", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        self.assertIn('capabilities', response['metadata']['type'])


class CodeSkillTests(unittest.TestCase):
    """Test the code skill."""
    
    def setUp(self):
        self.skill = CodeSkill()
        
    def test_can_handle_code(self):
        """Test code skill can handle code input."""
        self.assertTrue(self.skill.can_handle("Write a function", "generation", {}))
        self.assertTrue(self.skill.can_handle("def hello():", "general", {}))
        self.assertTrue(self.skill.can_handle("Analyze this code", "analysis", {}))
        
    def test_can_handle_non_code(self):
        """Test code skill doesn't handle non-code input."""
        self.assertFalse(self.skill.can_handle("Hello world", "general", {}))
        self.assertFalse(self.skill.can_handle("Calculate 2 + 2", "math", {}))
        
    def test_handle_generation(self):
        """Test handling code generation."""
        response = self.skill.handle("Write a Python function", "generation", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        
    def test_handle_analysis(self):
        """Test handling code analysis."""
        response = self.skill.handle("Analyze this code: def hello(): pass", "analysis", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)


class FileSkillTests(unittest.TestCase):
    """Test the file skill."""
    
    def setUp(self):
        self.skill = FileSkill()
        
    def test_can_handle_file(self):
        """Test file skill can handle file input."""
        self.assertTrue(self.skill.can_handle("Read file.txt", "read", {}))
        self.assertTrue(self.skill.can_handle("Analyze document.md", "analysis", {}))
        self.assertTrue(self.skill.can_handle("List files", "list", {}))
        
    def test_can_handle_non_file(self):
        """Test file skill doesn't handle non-file input."""
        self.assertFalse(self.skill.can_handle("Hello world", "general", {}))
        self.assertFalse(self.skill.can_handle("Calculate 2 + 2", "math", {}))
        
    def test_handle_list(self):
        """Test handling file listing."""
        response = self.skill.handle("List files", "list", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)
        
    def test_handle_read(self):
        """Test handling file reading."""
        response = self.skill.handle("Read test.txt", "read", {})
        self.assertIsInstance(response, dict)
        self.assertIn('content', response)


class IntegrationTests(unittest.TestCase):
    """Integration tests for OmniMind."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.omnimind = OmniMind(self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_math_conversation(self):
        """Test math conversation flow."""
        response1 = self.omnimind.process_input("Calculate 2 + 2")
        self.assertIsInstance(response1, dict)
        
        response2 = self.omnimind.process_input("What is 5 * 3?")
        self.assertIsInstance(response2, dict)
        
    def test_logic_conversation(self):
        """Test logic conversation flow."""
        response1 = self.omnimind.process_input("Alice is human")
        self.assertIsInstance(response1, dict)
        
        response2 = self.omnimind.process_input("If X is human then X is mortal")
        self.assertIsInstance(response2, dict)
        
    def test_general_conversation(self):
        """Test general conversation flow."""
        response1 = self.omnimind.process_input("Hello")
        self.assertIsInstance(response1, dict)
        
        response2 = self.omnimind.process_input("What can you do?")
        self.assertIsInstance(response2, dict)
        
    def test_memory_persistence(self):
        """Test memory persistence across interactions."""
        self.omnimind.process_input("Remember that Alice is human")
        self.omnimind.process_input("What do you know about Alice?")
        
        # Memory should persist
        memories = self.omnimind.memory.get_all_facts()
        self.assertGreater(len(memories), 0)
        
    def test_skill_routing(self):
        """Test skill routing works correctly."""
        # Math input should go to math skill
        response = self.omnimind.process_input("Calculate 2 + 2")
        self.assertIsInstance(response, dict)
        
        # Logic input should go to logic skill
        response = self.omnimind.process_input("Alice is human")
        self.assertIsInstance(response, dict)
        
        # General input should go to general skill
        response = self.omnimind.process_input("Hello")
        self.assertIsInstance(response, dict)


if __name__ == '__main__':
    unittest.main()
