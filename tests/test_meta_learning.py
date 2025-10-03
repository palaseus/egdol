"""
Tests for OmniMind Meta-Learning System
Comprehensive testing of dynamic skill generation and validation.
"""

import unittest
import tempfile
import os
import time
from egdol.omnimind.meta_learning import SkillGenerator, SkillValidator, RuntimeLoader
from egdol.omnimind.knowledge_graph import KnowledgeGraph, Node, Edge, NodeType, EdgeType


class SkillGeneratorTests(unittest.TestCase):
    """Test the skill generator."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SkillGenerator(self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator.skills_dir)
        self.assertIsNotNone(self.generator.skill_templates)
        self.assertEqual(len(self.generator.generated_skills), 0)
        
    def test_analyze_conversation_patterns(self):
        """Test conversation pattern analysis."""
        conversation_history = [
            {'type': 'user', 'content': 'Convert binary to hex'},
            {'type': 'user', 'content': 'Convert decimal to binary'},
            {'type': 'user', 'content': 'Convert hex to decimal'},
            {'type': 'user', 'content': 'Convert binary to hex again'},
        ]
        
        patterns = self.generator.analyze_conversation_patterns(conversation_history)
        self.assertIsInstance(patterns, list)
        
        # Should find 'convert' pattern
        convert_patterns = [p for p in patterns if 'convert' in p['pattern']]
        self.assertGreater(len(convert_patterns), 0)
        
    def test_generate_skill_from_instruction(self):
        """Test skill generation from instruction."""
        instruction = "Teach yourself how to convert binary to hexadecimal"
        context = {}
        
        result = self.generator.generate_skill_from_instruction(instruction, context)
        
        self.assertIsNotNone(result)
        self.assertIn('skill_name', result)
        self.assertIn('file', result)
        self.assertIn('description', result)
        self.assertIn('capabilities', result)
        
        # Check if file was created
        self.assertTrue(os.path.exists(result['file']))
        
    def test_extract_skill_name(self):
        """Test skill name extraction."""
        instruction = "Teach yourself how to convert binary to hex"
        skill_name = self.generator._extract_skill_name(instruction)
        self.assertIsNotNone(skill_name)
        self.assertIn('convert', skill_name.lower())
        
    def test_extract_capabilities(self):
        """Test capability extraction."""
        instruction = "Create a skill to analyze data and convert formats"
        capabilities = self.generator._extract_capabilities(instruction)
        
        self.assertIn('analysis', capabilities)
        self.assertIn('conversion', capabilities)
        
    def test_determine_skill_type(self):
        """Test skill type determination."""
        convert_instruction = "Convert binary to hex"
        analyze_instruction = "Analyze this data"
        process_instruction = "Process the information"
        
        self.assertEqual(self.generator._determine_skill_type(convert_instruction), 'conversion')
        self.assertEqual(self.generator._determine_skill_type(analyze_instruction), 'analysis')
        self.assertEqual(self.generator._determine_skill_type(process_instruction), 'data_processing')
        
    def test_generate_skill_code(self):
        """Test skill code generation."""
        skill_name = "test_skill"
        description = "Test skill description"
        capabilities = ["testing", "validation"]
        patterns = ["test", "validate"]
        skill_type = "data_processing"
        
        code = self.generator._generate_skill_code(
            skill_name, description, capabilities, patterns, skill_type
        )
        
        self.assertIsInstance(code, str)
        self.assertIn('class', code)
        self.assertIn(skill_name, code)
        self.assertIn('BaseSkill', code)
        self.assertIn('can_handle', code)
        self.assertIn('handle', code)
        
    def test_remove_skill(self):
        """Test skill removal."""
        # Generate a skill first
        instruction = "Create a test skill"
        result = self.generator.generate_skill_from_instruction(instruction, {})
        
        if result:
            skill_name = result['skill_name']
            self.assertTrue(skill_name in self.generator.generated_skills)
            
            # Remove the skill
            success = self.generator.remove_skill(skill_name)
            self.assertTrue(success)
            self.assertFalse(skill_name in self.generator.generated_skills)


class SkillValidatorTests(unittest.TestCase):
    """Test the skill validator."""
    
    def setUp(self):
        self.validator = SkillValidator()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_validate_syntax_valid(self):
        """Test syntax validation with valid code."""
        valid_code = '''
class TestSkill(BaseSkill):
    def can_handle(self, input_str, intent, context):
        return True
        
    def handle(self, input_str, intent, context):
        return {"content": "test"}
'''
        valid, errors = self.validator._validate_syntax(valid_code)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        
    def test_validate_syntax_invalid(self):
        """Test syntax validation with invalid code."""
        invalid_code = '''
class TestSkill(BaseSkill):
    def can_handle(self, input_str, intent, context
        return True
'''
        valid, errors = self.validator._validate_syntax(invalid_code)
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
        
    def test_validate_imports_safe(self):
        """Test import validation with safe imports."""
        safe_code = '''
from typing import Dict, Any
from ..base import BaseSkill

class TestSkill(BaseSkill):
    pass
'''
        valid, errors = self.validator._validate_imports(safe_code)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        
    def test_validate_imports_dangerous(self):
        """Test import validation with dangerous imports."""
        dangerous_code = '''
import os
import subprocess
from ..base import BaseSkill

class TestSkill(BaseSkill):
    pass
'''
        valid, errors = self.validator._validate_imports(dangerous_code)
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
        
    def test_validate_interface_valid(self):
        """Test interface validation with valid interface."""
        valid_code = '''
from typing import Dict, Any
from ..base import BaseSkill

class TestSkill(BaseSkill):
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        return True
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"content": "test"}
'''
        valid, errors = self.validator._validate_interface(valid_code, "TestSkill")
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        
    def test_validate_interface_missing_method(self):
        """Test interface validation with missing methods."""
        invalid_code = '''
from typing import Dict, Any
from ..base import BaseSkill

class TestSkill(BaseSkill):
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        return True
    # Missing handle method
'''
        valid, errors = self.validator._validate_interface(invalid_code, "TestSkill")
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
        
    def test_validate_safety_safe(self):
        """Test safety validation with safe code."""
        safe_code = '''
from typing import Dict, Any
from ..base import BaseSkill

class TestSkill(BaseSkill):
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        return True
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"content": "test"}
'''
        valid, errors = self.validator._validate_safety(safe_code)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        
    def test_validate_safety_dangerous(self):
        """Test safety validation with dangerous code."""
        dangerous_code = '''
import os
from ..base import BaseSkill

class TestSkill(BaseSkill):
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        return True
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        os.system("rm -rf /")  # Dangerous!
        return {"content": "test"}
'''
        valid, errors = self.validator._validate_safety(dangerous_code)
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
        
    def test_validate_skill_complete(self):
        """Test complete skill validation."""
        # Create a valid skill file
        skill_file = os.path.join(self.temp_dir, "test_skill.py")
        valid_code = '''
from typing import Dict, Any
from ..base import BaseSkill

class TestSkill(BaseSkill):
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        return True
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"content": "test"}
'''
        with open(skill_file, 'w') as f:
            f.write(valid_code)
            
        result = self.validator.validate_skill(skill_file, "TestSkill")
        
        self.assertIsInstance(result, dict)
        self.assertIn('valid', result)
        self.assertIn('errors', result)
        self.assertIn('warnings', result)
        
    def test_get_validation_summary(self):
        """Test validation summary."""
        summary = self.validator.get_validation_summary()
        
        self.assertIn('total_skills', summary)
        self.assertIn('valid_skills', summary)
        self.assertIn('invalid_skills', summary)
        self.assertIn('validation_rate', summary)


class RuntimeLoaderTests(unittest.TestCase):
    """Test the runtime loader."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.loader = RuntimeLoader(self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test loader initialization."""
        self.assertIsNotNone(self.loader.skills_dir)
        self.assertIsNotNone(self.loader.loaded_skills)
        self.assertIsNotNone(self.loader.skill_manager)
        self.assertIsNotNone(self.loader.load_history)
        
    def test_discover_skills(self):
        """Test skill discovery."""
        # Create a test skill file
        skill_file = os.path.join(self.temp_dir, "test_skill.py")
        with open(skill_file, 'w') as f:
            f.write("# Test skill")
            
        skills = self.loader.discover_skills()
        self.assertIsInstance(skills, list)
        
    def test_load_skill_invalid(self):
        """Test loading invalid skill."""
        # Create invalid skill file
        skill_file = os.path.join(self.temp_dir, "invalid_skill.py")
        with open(skill_file, 'w') as f:
            f.write("invalid python code")
            
        result = self.loader.load_skill(skill_file)
        self.assertIsNotNone(result)
        self.assertFalse(result['success'])
        
    def test_get_loaded_skills(self):
        """Test getting loaded skills."""
        skills = self.loader.get_loaded_skills()
        self.assertIsInstance(skills, dict)
        
    def test_get_skill_info(self):
        """Test getting skill info."""
        # This will return None since no skills are loaded
        info = self.loader.get_skill_info("nonexistent")
        self.assertIsNone(info)
        
    def test_get_load_history(self):
        """Test getting load history."""
        history = self.loader.get_load_history()
        self.assertIsInstance(history, list)
        
    def test_get_skill_stats(self):
        """Test getting skill statistics."""
        stats = self.loader.get_skill_stats()
        
        self.assertIn('total_loaded', stats)
        self.assertIn('skill_names', stats)
        self.assertIn('load_history_count', stats)
        self.assertIn('recent_loads', stats)


class KnowledgeGraphTests(unittest.TestCase):
    """Test the knowledge graph."""
    
    def setUp(self):
        self.graph = KnowledgeGraph()
        
    def test_add_node(self):
        """Test adding nodes."""
        node_id = self.graph.add_node("Alice", NodeType.ENTITY, {"age": 30})
        
        self.assertIsNotNone(node_id)
        self.assertIn(node_id, self.graph.nodes)
        
        node = self.graph.get_node(node_id)
        self.assertEqual(node.name, "Alice")
        self.assertEqual(node.node_type, NodeType.ENTITY)
        self.assertEqual(node.properties["age"], 30)
        
    def test_add_edge(self):
        """Test adding edges."""
        # Add nodes first
        alice_id = self.graph.add_node("Alice", NodeType.ENTITY)
        human_id = self.graph.add_node("Human", NodeType.CONCEPT)
        
        # Add edge
        edge_id = self.graph.add_edge(alice_id, human_id, EdgeType.IS_A)
        
        self.assertIsNotNone(edge_id)
        self.assertIn(edge_id, self.graph.edges)
        
        edge = self.graph.get_edge(edge_id)
        self.assertEqual(edge.source_id, alice_id)
        self.assertEqual(edge.target_id, human_id)
        self.assertEqual(edge.edge_type, EdgeType.IS_A)
        
    def test_get_neighbors(self):
        """Test getting neighbors."""
        # Create a simple graph: Alice -> Human -> Mortal
        alice_id = self.graph.add_node("Alice", NodeType.ENTITY)
        human_id = self.graph.add_node("Human", NodeType.CONCEPT)
        mortal_id = self.graph.add_node("Mortal", NodeType.CONCEPT)
        
        self.graph.add_edge(alice_id, human_id, EdgeType.IS_A)
        self.graph.add_edge(human_id, mortal_id, EdgeType.IMPLIES)
        
        # Get neighbors of Alice
        neighbors = self.graph.get_neighbors(alice_id)
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].name, "Human")
        
        # Get neighbors of Human
        neighbors = self.graph.get_neighbors(human_id)
        self.assertEqual(len(neighbors), 2)  # Alice and Mortal
        
    def test_find_path(self):
        """Test finding paths between nodes."""
        # Create graph: Alice -> Human -> Mortal
        alice_id = self.graph.add_node("Alice", NodeType.ENTITY)
        human_id = self.graph.add_node("Human", NodeType.CONCEPT)
        mortal_id = self.graph.add_node("Mortal", NodeType.CONCEPT)
        
        self.graph.add_edge(alice_id, human_id, EdgeType.IS_A)
        self.graph.add_edge(human_id, mortal_id, EdgeType.IMPLIES)
        
        # Find path from Alice to Mortal
        paths = self.graph.find_path(alice_id, mortal_id)
        self.assertGreater(len(paths), 0)
        
        # Check path structure
        for path in paths:
            self.assertEqual(path[0], alice_id)
            self.assertEqual(path[-1], mortal_id)
            
    def test_get_subgraph(self):
        """Test getting subgraph."""
        # Create nodes
        alice_id = self.graph.add_node("Alice", NodeType.ENTITY)
        human_id = self.graph.add_node("Human", NodeType.CONCEPT)
        mortal_id = self.graph.add_node("Mortal", NodeType.CONCEPT)
        
        # Add edges
        self.graph.add_edge(alice_id, human_id, EdgeType.IS_A)
        self.graph.add_edge(human_id, mortal_id, EdgeType.IMPLIES)
        
        # Get subgraph
        subgraph = self.graph.get_subgraph([alice_id, human_id])
        
        self.assertEqual(len(subgraph.nodes), 2)
        self.assertEqual(len(subgraph.edges), 1)
        
    def test_remove_node(self):
        """Test removing nodes."""
        node_id = self.graph.add_node("Test", NodeType.ENTITY)
        
        self.assertTrue(node_id in self.graph.nodes)
        
        success = self.graph.remove_node(node_id)
        self.assertTrue(success)
        self.assertFalse(node_id in self.graph.nodes)
        
    def test_remove_edge(self):
        """Test removing edges."""
        alice_id = self.graph.add_node("Alice", NodeType.ENTITY)
        human_id = self.graph.add_node("Human", NodeType.CONCEPT)
        edge_id = self.graph.add_edge(alice_id, human_id, EdgeType.IS_A)
        
        self.assertTrue(edge_id in self.graph.edges)
        
        success = self.graph.remove_edge(edge_id)
        self.assertTrue(success)
        self.assertFalse(edge_id in self.graph.edges)
        
    def test_get_stats(self):
        """Test getting graph statistics."""
        # Add some nodes and edges
        alice_id = self.graph.add_node("Alice", NodeType.ENTITY)
        human_id = self.graph.add_node("Human", NodeType.CONCEPT)
        self.graph.add_edge(alice_id, human_id, EdgeType.IS_A)
        
        stats = self.graph.get_stats()
        
        self.assertIn('total_nodes', stats)
        self.assertIn('total_edges', stats)
        self.assertIn('node_types', stats)
        self.assertIn('edge_types', stats)
        self.assertIn('avg_confidence', stats)
        
        self.assertEqual(stats['total_nodes'], 2)
        self.assertEqual(stats['total_edges'], 1)
        
    def test_export_import(self):
        """Test graph export and import."""
        # Create a graph
        alice_id = self.graph.add_node("Alice", NodeType.ENTITY)
        human_id = self.graph.add_node("Human", NodeType.CONCEPT)
        edge_id = self.graph.add_edge(alice_id, human_id, EdgeType.IS_A)
        
        # Export
        exported = self.graph.export_graph()
        self.assertIn('nodes', exported)
        self.assertIn('edges', exported)
        
        # Create new graph and import
        new_graph = KnowledgeGraph()
        success = new_graph.import_graph(exported)
        self.assertTrue(success)
        
        # Check imported data
        self.assertEqual(len(new_graph.nodes), 2)
        self.assertEqual(len(new_graph.edges), 1)


class IntegrationTests(unittest.TestCase):
    """Integration tests for meta-learning system."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SkillGenerator(self.temp_dir)
        self.validator = SkillValidator()
        self.loader = RuntimeLoader(self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_full_skill_lifecycle(self):
        """Test complete skill generation, validation, and loading."""
        # 1. Generate a skill
        instruction = "Create a skill to convert binary to hexadecimal"
        result = self.generator.generate_skill_from_instruction(instruction, {})
        
        self.assertIsNotNone(result)
        self.assertTrue(result['success'])
        
        # 2. Validate the skill
        validation_result = self.validator.validate_skill(result['file'], result['skill_name'])
        
        # 3. Load the skill (if validation passed)
        if validation_result['valid']:
            load_result = self.loader.load_skill(result['file'])
            self.assertIsNotNone(load_result)
            
    def test_skill_generation_with_conversation_patterns(self):
        """Test skill generation based on conversation patterns."""
        conversation_history = [
            {'type': 'user', 'content': 'Convert 1010 to hex'},
            {'type': 'user', 'content': 'Convert 1100 to hex'},
            {'type': 'user', 'content': 'Convert 1111 to hex'},
        ]
        
        # Analyze patterns
        patterns = self.generator.analyze_conversation_patterns(conversation_history)
        
        # Should find conversion patterns
        self.assertGreater(len(patterns), 0)
        
        # Generate skill based on patterns
        if patterns:
            instruction = f"Create a skill for {patterns[0]['pattern']}"
            result = self.generator.generate_skill_from_instruction(instruction, {})
            self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
