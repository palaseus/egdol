"""
Tests for Planner and Persona Systems
Comprehensive testing of goal decomposition and persona management.
"""

import unittest
import time
from egdol.omnimind.planner import GoalPlanner, Task, Goal, TaskStatus, TaskPriority
from egdol.omnimind.planner.decomposer import GoalDecomposer
from egdol.omnimind.planner.executor import TaskExecutor, ExecutionResult
from egdol.omnimind.planner.scheduler import TaskScheduler
from egdol.omnimind.personas import PersonaManager, PersonaType
from egdol.omnimind.personas.domain_packs import LegalExpert, CodingAssistant, Historian, Strategist


class GoalDecomposerTests(unittest.TestCase):
    """Test the goal decomposer."""
    
    def setUp(self):
        self.decomposer = GoalDecomposer()
        
    def test_decompose_analysis_goal(self):
        """Test decomposing analysis goals."""
        goal_description = "Analyze this file for security issues"
        context = {'target': 'security_audit.py'}
        
        tasks = self.decomposer.decompose_goal(goal_description, context)
        
        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0)
        
        # Check that tasks have required fields
        for task in tasks:
            self.assertIsInstance(task, Task)
            self.assertIsNotNone(task.id)
            self.assertIsNotNone(task.name)
            self.assertIsNotNone(task.description)
            self.assertIsNotNone(task.skill_required)
            self.assertIsInstance(task.parameters, dict)
            self.assertIsInstance(task.dependencies, list)
            self.assertIsInstance(task.priority, TaskPriority)
            
    def test_decompose_summary_goal(self):
        """Test decomposing summary goals."""
        goal_description = "Summarize the main points of this document"
        context = {'target': 'research_paper.pdf'}
        
        tasks = self.decomposer.decompose_goal(goal_description, context)
        
        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0)
        
        # Check for summary-related tasks
        task_names = [task.name for task in tasks]
        self.assertTrue(any('summary' in name.lower() for name in task_names))
        
    def test_decompose_security_goal(self):
        """Test decomposing security analysis goals."""
        goal_description = "Analyze this system for security vulnerabilities"
        context = {'target': 'web_application'}
        
        tasks = self.decomposer.decompose_goal(goal_description, context)
        
        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0)
        
        # Check for security-related tasks
        task_names = [task.name for task in tasks]
        self.assertTrue(any('security' in name.lower() for name in task_names))
        
    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        goal_description = "Analyze security performance of the system"
        phrases = self.decomposer._extract_key_phrases(goal_description)
        
        self.assertIsInstance(phrases, list)
        self.assertIn('analyze', phrases)
        self.assertIn('security', phrases)
        self.assertIn('performance', phrases)
        
    def test_determine_strategy(self):
        """Test strategy determination."""
        goal_description = "Analyze security vulnerabilities"
        phrases = ['analyze', 'security']
        
        strategy = self.decomposer._determine_strategy(goal_description, phrases)
        self.assertEqual(strategy, 'security')


class TaskExecutorTests(unittest.TestCase):
    """Test the task executor."""
    
    def setUp(self):
        # Mock skill router
        class MockSkillRouter:
            def get_available_skills(self):
                return ['analysis', 'security', 'general']
                
            def execute_skill(self, skill_name, input_data, verbose=False):
                return {'result': f'Executed {skill_name}', 'success': True}
                
        self.skill_router = MockSkillRouter()
        self.executor = TaskExecutor(self.skill_router)
        
    def test_execute_task(self):
        """Test task execution."""
        task = Task(
            id="test_task",
            name="Test Task",
            description="Test task description",
            skill_required="analysis",
            parameters={'param1': 'value1'},
            dependencies=[],
            priority=TaskPriority.NORMAL
        )
        
        result = self.executor.execute_task(task, verbose=False)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.task_id, task.id)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.result)
        
    def test_execute_task_verbose(self):
        """Test task execution with verbose output."""
        task = Task(
            id="test_task_verbose",
            name="Test Task Verbose",
            description="Test task description",
            skill_required="analysis",
            parameters={'param1': 'value1'},
            dependencies=[],
            priority=TaskPriority.NORMAL
        )
        
        result = self.executor.execute_task(task, verbose=True)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertTrue(result.success)
        self.assertGreater(len(result.reasoning_trace), 0)
        
    def test_execute_task_failure(self):
        """Test task execution failure."""
        # Mock skill router that fails
        class FailingSkillRouter:
            def get_available_skills(self):
                return ['analysis']
                
            def execute_skill(self, skill_name, input_data, verbose=False):
                raise Exception("Skill execution failed")
                
        failing_router = FailingSkillRouter()
        executor = TaskExecutor(failing_router)
        
        task = Task(
            id="failing_task",
            name="Failing Task",
            description="Task that will fail",
            skill_required="analysis",
            parameters={},
            dependencies=[],
            priority=TaskPriority.NORMAL
        )
        
        result = executor.execute_task(task, verbose=False)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        
    def test_get_execution_stats(self):
        """Test execution statistics."""
        # Execute some tasks
        task1 = Task(
            id="task1",
            name="Task 1",
            description="First task",
            skill_required="analysis",
            parameters={},
            dependencies=[],
            priority=TaskPriority.NORMAL
        )
        
        task2 = Task(
            id="task2",
            name="Task 2",
            description="Second task",
            skill_required="security",
            parameters={},
            dependencies=[],
            priority=TaskPriority.HIGH
        )
        
        self.executor.execute_task(task1)
        self.executor.execute_task(task2)
        
        stats = self.executor.get_execution_stats()
        
        self.assertIn('total_executions', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('average_execution_time', stats)
        self.assertIn('skill_performance', stats)
        
        self.assertEqual(stats['total_executions'], 2)
        self.assertEqual(stats['success_rate'], 1.0)


class TaskSchedulerTests(unittest.TestCase):
    """Test the task scheduler."""
    
    def setUp(self):
        self.scheduler = TaskScheduler()
        
    def test_schedule_task(self):
        """Test task scheduling."""
        task = Task(
            id="test_task",
            name="Test Task",
            description="Test task description",
            skill_required="analysis",
            parameters={},
            dependencies=[],
            priority=TaskPriority.NORMAL
        )
        
        result = self.scheduler.schedule_task(task)
        self.assertTrue(result)
        
        # Check that task is scheduled
        self.assertIn(task.id, self.scheduler.scheduled_tasks)
        
    def test_schedule_tasks(self):
        """Test scheduling multiple tasks."""
        tasks = [
            Task(
                id=f"task_{i}",
                name=f"Task {i}",
                description=f"Task {i} description",
                skill_required="analysis",
                parameters={},
                dependencies=[],
                priority=TaskPriority.NORMAL
            )
            for i in range(3)
        ]
        
        results = self.scheduler.schedule_tasks(tasks)
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results))
        
        # Check that all tasks are scheduled
        for task in tasks:
            self.assertIn(task.id, self.scheduler.scheduled_tasks)
            
    def test_get_next_task(self):
        """Test getting next task to execute."""
        task = Task(
            id="test_task",
            name="Test Task",
            description="Test task description",
            skill_required="analysis",
            parameters={},
            dependencies=[],
            priority=TaskPriority.NORMAL
        )
        
        self.scheduler.schedule_task(task)
        next_task = self.scheduler.get_next_task()
        
        self.assertIsNotNone(next_task)
        self.assertEqual(next_task.id, task.id)
        
    def test_task_dependencies(self):
        """Test task dependencies."""
        # Create tasks with dependencies
        task1 = Task(
            id="task1",
            name="Task 1",
            description="First task",
            skill_required="analysis",
            parameters={},
            dependencies=[],
            priority=TaskPriority.NORMAL
        )
        
        task2 = Task(
            id="task2",
            name="Task 2",
            description="Second task",
            skill_required="security",
            parameters={},
            dependencies=["task1"],
            priority=TaskPriority.NORMAL
        )
        
        self.scheduler.schedule_task(task1)
        self.scheduler.schedule_task(task2)
        
        # Task1 should be ready, task2 should be blocked
        ready_tasks = self.scheduler.get_ready_tasks()
        blocked_tasks = self.scheduler.get_blocked_tasks()
        
        self.assertEqual(len(ready_tasks), 1)
        self.assertEqual(ready_tasks[0].id, "task1")
        
        self.assertEqual(len(blocked_tasks), 1)
        self.assertEqual(blocked_tasks[0].id, "task2")
        
        # Mark task1 as completed
        self.scheduler.mark_task_completed("task1")
        
        # Now task2 should be ready
        ready_tasks = self.scheduler.get_ready_tasks()
        self.assertEqual(len(ready_tasks), 1)
        self.assertEqual(ready_tasks[0].id, "task2")
        
    def test_get_scheduler_stats(self):
        """Test scheduler statistics."""
        # Schedule some tasks
        tasks = [
            Task(
                id=f"task_{i}",
                name=f"Task {i}",
                description=f"Task {i} description",
                skill_required="analysis",
                parameters={},
                dependencies=[],
                priority=TaskPriority.NORMAL
            )
            for i in range(3)
        ]
        
        self.scheduler.schedule_tasks(tasks)
        
        stats = self.scheduler.get_scheduler_stats()
        
        self.assertIn('total_tasks', stats)
        self.assertIn('pending_tasks', stats)
        self.assertIn('completed_tasks', stats)
        self.assertIn('failed_tasks', stats)
        
        self.assertEqual(stats['total_tasks'], 3)
        self.assertEqual(stats['pending_tasks'], 3)


class PersonaManagerTests(unittest.TestCase):
    """Test the persona manager."""
    
    def setUp(self):
        self.persona_manager = PersonaManager()
        
    def test_create_persona(self):
        """Test persona creation."""
        persona = self.persona_manager.create_persona(
            name="Test Persona",
            description="Test persona description",
            persona_type=PersonaType.GENERAL,
            skills=["analysis", "research"],
            knowledge_base={"domain": "testing"},
            response_style={"tone": "friendly"}
        )
        
        self.assertIsNotNone(persona)
        self.assertEqual(persona.name, "Test Persona")
        self.assertEqual(persona.persona_type, PersonaType.GENERAL)
        self.assertIn("analysis", persona.skills)
        self.assertIn("domain", persona.knowledge_base)
        
    def test_switch_persona(self):
        """Test persona switching."""
        # Create two personas
        persona1 = self.persona_manager.create_persona(
            name="Persona 1",
            description="First persona",
            persona_type=PersonaType.GENERAL,
            skills=["analysis"]
        )
        
        persona2 = self.persona_manager.create_persona(
            name="Persona 2",
            description="Second persona",
            persona_type=PersonaType.TECHNICAL,
            skills=["coding"]
        )
        
        # Switch to persona1
        result1 = self.persona_manager.switch_persona(persona1.id)
        self.assertTrue(result1)
        self.assertEqual(self.persona_manager.get_active_persona(), persona1)
        
        # Switch to persona2
        result2 = self.persona_manager.switch_persona(persona2.id)
        self.assertTrue(result2)
        self.assertEqual(self.persona_manager.get_active_persona(), persona2)
        
    def test_get_best_persona(self):
        """Test getting best persona for a query."""
        # Create specialized personas
        legal_persona = self.persona_manager.create_persona(
            name="Legal Expert",
            description="Legal analysis expert",
            persona_type=PersonaType.LEGAL,
            skills=["legal_analysis", "contract_review"],
            knowledge_base={"law": "legal knowledge"}
        )
        
        coding_persona = self.persona_manager.create_persona(
            name="Coding Assistant",
            description="Software development expert",
            persona_type=PersonaType.CODING,
            skills=["code_analysis", "debugging"],
            knowledge_base={"programming": "coding knowledge"}
        )
        
        # Test legal query
        legal_query = "Analyze this contract for legal issues"
        best_persona = self.persona_manager.get_best_persona(legal_query)
        self.assertEqual(best_persona, legal_persona)
        
        # Test coding query
        coding_query = "Review this code for bugs"
        best_persona = self.persona_manager.get_best_persona(coding_query)
        self.assertEqual(best_persona, coding_persona)
        
    def test_auto_switch_persona(self):
        """Test automatic persona switching."""
        # Create personas
        legal_persona = self.persona_manager.create_persona(
            name="Legal Expert",
            description="Legal analysis expert",
            persona_type=PersonaType.LEGAL,
            skills=["legal_analysis"],
            knowledge_base={"law": "legal knowledge"}
        )
        
        coding_persona = self.persona_manager.create_persona(
            name="Coding Assistant",
            description="Software development expert",
            persona_type=PersonaType.CODING,
            skills=["code_analysis"],
            knowledge_base={"programming": "coding knowledge"}
        )
        
        # Switch to legal persona
        self.persona_manager.switch_persona(legal_persona.id)
        
        # Auto-switch to coding persona for coding query
        coding_query = "Review this code for bugs"
        result = self.persona_manager.auto_switch_persona(coding_query)
        self.assertTrue(result)
        self.assertEqual(self.persona_manager.get_active_persona(), coding_persona)
        
    def test_delete_persona(self):
        """Test persona deletion."""
        persona = self.persona_manager.create_persona(
            name="Test Persona",
            description="Test persona",
            persona_type=PersonaType.GENERAL,
            skills=["analysis"]
        )
        
        # Delete persona
        result = self.persona_manager.delete_persona(persona.id)
        self.assertTrue(result)
        
        # Check that persona is deleted
        self.assertIsNone(self.persona_manager.get_persona(persona.id))
        
    def test_get_persona_stats(self):
        """Test persona statistics."""
        # Create some personas
        for i in range(3):
            self.persona_manager.create_persona(
                name=f"Persona {i}",
                description=f"Persona {i} description",
                persona_type=PersonaType.GENERAL,
                skills=["analysis"]
            )
            
        stats = self.persona_manager.get_persona_stats()
        
        self.assertIn('total_personas', stats)
        self.assertIn('active_persona', stats)
        self.assertIn('usage_stats', stats)
        
        self.assertEqual(stats['total_personas'], 3)


class DomainPackTests(unittest.TestCase):
    """Test domain packs."""
    
    def setUp(self):
        self.persona_manager = PersonaManager()
        
    def test_legal_expert_pack(self):
        """Test Legal Expert domain pack."""
        legal_pack = LegalExpert()
        persona = legal_pack.create_persona(self.persona_manager)
        
        self.assertEqual(persona.name, "Legal Expert")
        self.assertEqual(persona.persona_type, PersonaType.LEGAL)
        self.assertIn("legal_analysis", persona.skills)
        self.assertIn("contract_law", persona.knowledge_base)
        
    def test_coding_assistant_pack(self):
        """Test Coding Assistant domain pack."""
        coding_pack = CodingAssistant()
        persona = coding_pack.create_persona(self.persona_manager)
        
        self.assertEqual(persona.name, "Coding Assistant")
        self.assertEqual(persona.persona_type, PersonaType.CODING)
        self.assertIn("code_analysis", persona.skills)
        self.assertIn("programming_languages", persona.knowledge_base)
        
    def test_historian_pack(self):
        """Test Historian domain pack."""
        historian_pack = Historian()
        persona = historian_pack.create_persona(self.persona_manager)
        
        self.assertEqual(persona.name, "Historian")
        self.assertEqual(persona.persona_type, PersonaType.HISTORICAL)
        self.assertIn("historical_research", persona.skills)
        self.assertIn("ancient_history", persona.knowledge_base)
        
    def test_strategist_pack(self):
        """Test Strategist domain pack."""
        strategist_pack = Strategist()
        persona = strategist_pack.create_persona(self.persona_manager)
        
        self.assertEqual(persona.name, "Strategist")
        self.assertEqual(persona.persona_type, PersonaType.STRATEGIC)
        self.assertIn("strategic_planning", persona.skills)
        self.assertIn("strategy_frameworks", persona.knowledge_base)


class IntegrationTests(unittest.TestCase):
    """Integration tests for planner and persona systems."""
    
    def setUp(self):
        self.persona_manager = PersonaManager()
        self.decomposer = GoalDecomposer()
        
        # Mock skill router
        class MockSkillRouter:
            def get_available_skills(self):
                return ['analysis', 'security', 'general']
                
            def execute_skill(self, skill_name, input_data, verbose=False):
                return {'result': f'Executed {skill_name}', 'success': True}
                
        self.skill_router = MockSkillRouter()
        self.executor = TaskExecutor(self.skill_router)
        self.scheduler = TaskScheduler()
        
    def test_full_goal_execution(self):
        """Test full goal execution pipeline."""
        # Create a persona
        persona = self.persona_manager.create_persona(
            name="Test Persona",
            description="Test persona",
            persona_type=PersonaType.GENERAL,
            skills=["analysis", "security"]
        )
        
        # Decompose a goal
        goal_description = "Analyze this system for security issues"
        context = {'target': 'web_application'}
        tasks = self.decomposer.decompose_goal(goal_description, context)
        
        # Schedule tasks
        self.scheduler.schedule_tasks(tasks)
        
        # Execute tasks
        results = []
        while True:
            next_task = self.scheduler.get_next_task()
            if not next_task:
                break
                
            result = self.executor.execute_task(next_task, verbose=False)
            results.append(result)
            
            if result.success:
                self.scheduler.mark_task_completed(next_task.id)
            else:
                self.scheduler.mark_task_failed(next_task.id, result.error)
                
        # Check results
        self.assertGreater(len(results), 0)
        self.assertTrue(all(result.success for result in results))
        
    def test_persona_skill_matching(self):
        """Test persona skill matching."""
        # Create specialized personas
        legal_persona = self.persona_manager.create_persona(
            name="Legal Expert",
            description="Legal expert",
            persona_type=PersonaType.LEGAL,
            skills=["legal_analysis", "contract_review"],
            knowledge_base={"law": "legal knowledge"}
        )
        
        coding_persona = self.persona_manager.create_persona(
            name="Coding Assistant",
            description="Coding expert",
            persona_type=PersonaType.CODING,
            skills=["code_analysis", "debugging"],
            knowledge_base={"programming": "coding knowledge"}
        )
        
        # Test queries
        legal_query = "Review this contract for legal issues"
        coding_query = "Analyze this code for bugs"
        
        # Test persona selection
        legal_best = self.persona_manager.get_best_persona(legal_query)
        coding_best = self.persona_manager.get_best_persona(coding_query)
        
        self.assertEqual(legal_best, legal_persona)
        self.assertEqual(coding_best, coding_persona)
        
        # Test persona switching
        self.persona_manager.switch_persona(legal_persona.id)
        self.assertEqual(self.persona_manager.get_active_persona(), legal_persona)
        
        # Auto-switch for coding query
        self.persona_manager.auto_switch_persona(coding_query)
        self.assertEqual(self.persona_manager.get_active_persona(), coding_persona)


if __name__ == '__main__':
    unittest.main()
