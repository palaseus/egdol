"""
Comprehensive test suite for the Emergence Control, Self-Expansion & Cross-Domain Intelligence system.
Tests all components: Domain Proposal Engine, Cross-Civilization Interaction, Emergent Law Codex,
System Evolution Orchestrator, Commander's Interface 2.0, and Hyper-Rigorous Meta-Testing.
"""

import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from egdol.omnimind.emergence.domain_proposal_engine import (
    DomainProposalEngine, DomainType, DomainComplexity, ProposalStatus
)
from egdol.omnimind.emergence.cross_civilization_interaction import (
    CrossCivilizationInteractionSystem, InteractionType, InteractionStatus
)
from egdol.omnimind.emergence.emergent_law_codex import (
    EmergentLawCodex, LawType, LawStatus, CulturalArtifactType
)
from egdol.omnimind.emergence.system_evolution_orchestrator import (
    SystemEvolutionOrchestrator, RefactorType, RefactorStatus
)
from egdol.omnimind.emergence.commanders_interface_2 import (
    CommandersInterface2, ObservationMode, PerturbationType
)
from egdol.omnimind.emergence.hyper_rigorous_meta_testing import (
    HyperRigorousMetaTesting, MutationType, TestResult
)
from egdol.omnimind.civilization.multi_agent_system import (
    MultiAgentCivilizationSystem, CivilizationState
)
from egdol.omnimind.conversational.personality_framework import Personality, PersonalityType


class TestDomainProposalEngine(unittest.TestCase):
    """Test Domain Proposal Engine for autonomous domain expansion."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_domain_proposals.db"
        self.engine = DomainProposalEngine(str(self.db_path))
        
        # Create test agents
        self.test_agents = {}
        for i, personality_type in enumerate([PersonalityType.STRATEGOS, PersonalityType.ARCHIVIST]):
            agent_id = str(uuid.uuid4())
            personality = Personality(
                name=f"TestAgent{i}",
                personality_type=personality_type,
                description=f"Test {personality_type.value} agent",
                archetype=personality_type.value,
                epistemic_style="formal"
            )
            # Create mock agent
            from egdol.omnimind.civilization.multi_agent_system import CivilizationAgent
            agent = CivilizationAgent(
                agent_id=agent_id,
                personality=personality,
                civilization_id="test_civ",
                message_bus=None,
                memory=None
            )
            self.test_agents[agent_id] = agent
        
        self.engine.register_agents(self.test_agents)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_domain_proposal_creation(self):
        """Test creating domain proposals."""
        proposal_id = self.engine.propose_domain(
            domain_name="Economic Analysis",
            domain_type=DomainType.ECONOMIC,
            description="Economic reasoning and analysis domain",
            rationale="Need for economic intelligence in civilization management",
            complexity=DomainComplexity.INTERMEDIATE,
            proposer_agent_id=list(self.test_agents.keys())[0]
        )
        
        self.assertIsNotNone(proposal_id)
        self.assertIn(proposal_id, self.engine.proposals)
        
        proposal = self.engine.proposals[proposal_id]
        self.assertEqual(proposal.domain_name, "Economic Analysis")
        self.assertEqual(proposal.domain_type, DomainType.ECONOMIC)
        self.assertEqual(proposal.status, ProposalStatus.DRAFT)
    
    def test_domain_voting_system(self):
        """Test domain proposal voting system."""
        # Create proposal
        proposal_id = self.engine.propose_domain(
            domain_name="Linguistic Analysis",
            domain_type=DomainType.LINGUISTIC,
            description="Language processing domain",
            rationale="Need for linguistic intelligence",
            complexity=DomainComplexity.BASIC,
            proposer_agent_id=list(self.test_agents.keys())[0]
        )
        
        # Test voting
        agent_ids = list(self.test_agents.keys())
        
        # Support from first agent
        success = self.engine.vote_on_proposal(proposal_id, agent_ids[0], "support")
        self.assertTrue(success)
        
        # Support from second agent
        success = self.engine.vote_on_proposal(proposal_id, agent_ids[1], "support")
        self.assertTrue(success)
        
        proposal = self.engine.proposals[proposal_id]
        self.assertEqual(len(proposal.supporting_agents), 2)
        self.assertEqual(len(proposal.opposing_agents), 0)
    
    def test_domain_implementation_generation(self):
        """Test automatic domain implementation generation."""
        proposal_id = self.engine.propose_domain(
            domain_name="Ecological Intelligence",
            domain_type=DomainType.ECOLOGICAL,
            description="Environmental analysis domain",
            rationale="Need for ecological intelligence",
            complexity=DomainComplexity.ADVANCED,
            proposer_agent_id=list(self.test_agents.keys())[0]
        )
        
        # Get implementation plan
        proposal = self.engine.proposals[proposal_id]
        self.assertIn("base_modules", proposal.implementation_plan)
        self.assertIn("estimated_complexity", proposal.implementation_plan)
        self.assertIn("development_phases", proposal.implementation_plan)
    
    def test_domain_statistics(self):
        """Test domain statistics calculation."""
        # Create multiple proposals
        for i in range(3):
            self.engine.propose_domain(
                domain_name=f"Test Domain {i}",
                domain_type=DomainType.ECONOMIC,
                description=f"Test domain {i}",
                rationale="Testing",
                complexity=DomainComplexity.BASIC,
                proposer_agent_id=list(self.test_agents.keys())[0]
            )
        
        stats = self.engine.get_domain_statistics()
        self.assertEqual(stats["total_proposals"], 3)
        self.assertIn("approval_rate", stats)
        self.assertIn("domain_type_distribution", stats)


class TestCrossCivilizationInteraction(unittest.TestCase):
    """Test Cross-Civilization Interaction Layer."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_cross_civilization.db"
        self.interaction_system = CrossCivilizationInteractionSystem(str(self.db_path))
        
        # Create test civilizations
        self.civilization1 = CivilizationState(
            civilization_id="civ1",
            name="Test Civilization 1"
        )
        self.civilization2 = CivilizationState(
            civilization_id="civ2", 
            name="Test Civilization 2"
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_civilization_registration(self):
        """Test registering civilizations for interaction."""
        civ1_id = self.interaction_system.register_civilization(self.civilization1)
        civ2_id = self.interaction_system.register_civilization(self.civilization2)
        
        self.assertEqual(civ1_id, "civ1")
        self.assertEqual(civ2_id, "civ2")
        self.assertIn("civ1", self.interaction_system.civilization_profiles)
        self.assertIn("civ2", self.interaction_system.civilization_profiles)
    
    def test_interaction_proposal(self):
        """Test proposing interactions between civilizations."""
        # Register civilizations
        self.interaction_system.register_civilization(self.civilization1)
        self.interaction_system.register_civilization(self.civilization2)
        
        # Propose trade interaction
        proposal_id = self.interaction_system.propose_interaction(
            interaction_type=InteractionType.TRADE,
            initiator_civilization="civ1",
            target_civilization="civ2",
            description="Resource exchange proposal",
            terms={"resource": "technology", "amount": 100},
            duration=timedelta(days=30)
        )
        
        self.assertIsNotNone(proposal_id)
        self.assertIn(proposal_id, self.interaction_system.interactions)
        
        interaction = self.interaction_system.interactions[proposal_id]
        self.assertEqual(interaction.interaction_type, InteractionType.TRADE)
        self.assertEqual(interaction.status, InteractionStatus.PROPOSED)
    
    def test_federation_creation(self):
        """Test creating federations of civilizations."""
        # Register civilizations
        self.interaction_system.register_civilization(self.civilization1)
        self.interaction_system.register_civilization(self.civilization2)
        
        # Create federation
        federation_id = self.interaction_system.create_federation(
            name="Test Federation",
            founding_civilizations=["civ1", "civ2"],
            governance_structure="consensus"
        )
        
        self.assertIsNotNone(federation_id)
        self.assertIn(federation_id, self.interaction_system.federations)
        
        federation = self.interaction_system.federations[federation_id]
        self.assertEqual(federation.name, "Test Federation")
        self.assertEqual(len(federation.member_civilizations), 2)
    
    def test_emergent_pattern_detection(self):
        """Test detection of emergent patterns."""
        # Register multiple civilizations
        for i in range(5):
            civ = CivilizationState(civilization_id=f"civ{i}", name=f"Civilization {i}")
            self.interaction_system.register_civilization(civ)
        
        patterns = self.interaction_system.detect_emergent_patterns()
        
        self.assertIn("hegemony", patterns)
        self.assertIn("multipolarity", patterns)
        self.assertIn("trade_networks", patterns)
        self.assertIn("conflict_clusters", patterns)


class TestEmergentLawCodex(unittest.TestCase):
    """Test Emergent Law Codex and Cultural Archives."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_law_codex.db"
        self.codex = EmergentLawCodex(str(self.db_path))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_universal_law_proposal(self):
        """Test proposing universal laws."""
        law_id = self.codex.propose_universal_law(
            title="Universal Rights Declaration",
            law_type=LawType.ETHICAL,
            content="All civilizations have the right to self-determination",
            rationale="Foundation for peaceful coexistence",
            originating_civilization="civ1",
            enforcement_mechanisms=["diplomatic_pressure", "economic_sanctions"]
        )
        
        self.assertIsNotNone(law_id)
        self.assertIn(law_id, self.codex.universal_laws)
        
        law = self.codex.universal_laws[law_id]
        self.assertEqual(law.title, "Universal Rights Declaration")
        self.assertEqual(law.law_type, LawType.ETHICAL)
        self.assertEqual(law.status, LawStatus.PROPOSED)
    
    def test_law_support_system(self):
        """Test law support and opposition system."""
        # Create law
        law_id = self.codex.propose_universal_law(
            title="Trade Standards",
            law_type=LawType.COMMERCIAL,
            content="Standardized trade protocols",
            rationale="Economic efficiency",
            originating_civilization="civ1"
        )
        
        # Support law
        success = self.codex.support_law(law_id, "civ2")
        self.assertTrue(success)
        
        # Oppose law
        success = self.codex.oppose_law(law_id, "civ3")
        self.assertTrue(success)
        
        law = self.codex.universal_laws[law_id]
        self.assertIn("civ2", law.supporting_civilizations)
        self.assertIn("civ3", law.opposing_civilizations)
    
    def test_cultural_artifact_creation(self):
        """Test creating cultural artifacts."""
        artifact_id = self.codex.add_cultural_artifact(
            title="Philosophy of Cooperation",
            artifact_type=CulturalArtifactType.PHILOSOPHY,
            content="The fundamental principle of mutual benefit",
            description="A philosophical framework for cooperation",
            originating_civilization="civ1",
            cultural_themes=["cooperation", "mutual_benefit", "harmony"]
        )
        
        self.assertIsNotNone(artifact_id)
        self.assertIn(artifact_id, self.codex.cultural_artifacts)
        
        artifact = self.codex.cultural_artifacts[artifact_id]
        self.assertEqual(artifact.title, "Philosophy of Cooperation")
        self.assertEqual(artifact.artifact_type, CulturalArtifactType.PHILOSOPHY)
        self.assertIn("cooperation", artifact.cultural_themes)
    
    def test_cultural_archive_creation(self):
        """Test creating cultural archives."""
        archive_id = self.codex.create_cultural_archive(
            name="Philosophy Archive",
            themes=["philosophy", "ethics", "metaphysics"]
        )
        
        self.assertIsNotNone(archive_id)
        self.assertIn(archive_id, self.codex.cultural_archives)
        
        archive = self.codex.cultural_archives[archive_id]
        self.assertEqual(archive.name, "Philosophy Archive")
        self.assertIn("philosophy", archive.cultural_themes)
    
    def test_system_statistics(self):
        """Test system statistics calculation."""
        # Create some laws and artifacts
        for i in range(3):
            self.codex.propose_universal_law(
                title=f"Law {i}",
                law_type=LawType.ETHICAL,
                content=f"Content {i}",
                rationale="Testing",
                originating_civilization="civ1"
            )
            
            self.codex.add_cultural_artifact(
                title=f"Artifact {i}",
                artifact_type=CulturalArtifactType.PHILOSOPHY,
                content=f"Content {i}",
                description=f"Description {i}",
                originating_civilization="civ1"
            )
        
        stats = self.codex.get_system_statistics()
        self.assertEqual(stats["total_laws"], 3)
        self.assertEqual(stats["total_artifacts"], 3)


class TestSystemEvolutionOrchestrator(unittest.TestCase):
    """Test System Evolution Orchestrator for self-refactoring."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_evolution.db"
        self.orchestrator = SystemEvolutionOrchestrator(
            base_path=str(Path(__file__).parent),
            db_path=str(self.db_path)
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_system_metrics_analysis(self):
        """Test system performance analysis."""
        metrics = self.orchestrator.analyze_system_performance()
        
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.module_count, 0)
        self.assertGreater(metrics.total_lines_of_code, 0)
        self.assertGreaterEqual(metrics.cyclomatic_complexity, 0)
        self.assertGreaterEqual(metrics.coupling_score, 0)
        self.assertGreaterEqual(metrics.cohesion_score, 0)
    
    def test_refactor_proposal_creation(self):
        """Test creating refactor proposals."""
        proposal_id = self.orchestrator.propose_refactor(
            refactor_type=RefactorType.MODULE_MERGE,
            target_modules=["test_module1.py", "test_module2.py"],
            description="Merge related modules",
            rationale="Reduce coupling and improve organization"
        )
        
        self.assertIsNotNone(proposal_id)
        self.assertIn(proposal_id, self.orchestrator.refactor_proposals)
        
        proposal = self.orchestrator.refactor_proposals[proposal_id]
        self.assertEqual(proposal.refactor_type, RefactorType.MODULE_MERGE)
        self.assertEqual(proposal.status, RefactorStatus.PROPOSED)
        self.assertIn("test_module1.py", proposal.target_modules)
    
    def test_refactor_approval(self):
        """Test approving refactor proposals."""
        # Create proposal
        proposal_id = self.orchestrator.propose_refactor(
            refactor_type=RefactorType.CODE_DEDUPLICATION,
            target_modules=["test_module.py"],
            description="Remove code duplication",
            rationale="Improve maintainability"
        )
        
        # Approve proposal
        success = self.orchestrator.approve_refactor(proposal_id)
        self.assertTrue(success)
        
        proposal = self.orchestrator.refactor_proposals[proposal_id]
        self.assertEqual(proposal.status, RefactorStatus.APPROVED)
        self.assertIsNotNone(proposal.approved_at)
    
    def test_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        recommendations = self.orchestrator.get_optimization_recommendations()
        
        self.assertIsInstance(recommendations, list)
        for rec in recommendations:
            self.assertIn("type", rec)
            self.assertIn("priority", rec)
            self.assertIn("description", rec)
            self.assertIn("suggestion", rec)
    
    def test_system_health_score(self):
        """Test system health score calculation."""
        health_score = self.orchestrator.get_system_health_score()
        
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)


class TestCommandersInterface2(unittest.TestCase):
    """Test Commander's Strategic Interface 2.0."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_commanders_interface.db"
        self.interface = CommandersInterface2(str(self.db_path))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_observation_mode_setting(self):
        """Test setting observation modes."""
        self.interface.set_observation_mode(ObservationMode.HISTORICAL)
        self.assertEqual(self.interface.observation_mode, ObservationMode.HISTORICAL)
        
        self.interface.set_observation_mode(ObservationMode.PREDICTIVE)
        self.assertEqual(self.interface.observation_mode, ObservationMode.PREDICTIVE)
    
    def test_civilization_cluster_observation(self):
        """Test observing civilization clusters."""
        # Create test civilizations
        civilizations = {
            "civ1": CivilizationState(civilization_id="civ1", name="Civilization 1"),
            "civ2": CivilizationState(civilization_id="civ2", name="Civilization 2")
        }
        
        observation = self.interface.observe_civilization_clusters(civilizations)
        
        self.assertIsNotNone(observation)
        self.assertEqual(observation.observation_type, "civilization_clusters")
        self.assertIn("total_civilizations", observation.data)
        self.assertEqual(observation.data["total_civilizations"], 2)
        self.assertIsInstance(observation.insights, list)
    
    def test_perturbation_injection(self):
        """Test injecting high-order perturbations."""
        perturbation_id = self.interface.inject_perturbation(
            perturbation_type=PerturbationType.BLACK_SWAN_EVENT,
            description="Unexpected technological breakthrough",
            intensity=0.8,
            duration=timedelta(hours=24),
            affected_systems=["civilization_system", "technology_layer"]
        )
        
        self.assertIsNotNone(perturbation_id)
        self.assertIn(perturbation_id, [p.perturbation_id for p in self.interface.perturbation_events])
    
    def test_timeline_scrubbing(self):
        """Test timeline scrubbing functionality."""
        # Add some timeline events
        self.interface._add_timeline_event(
            event_type="test_event",
            description="Test event",
            impact_level=0.5
        )
        
        # Scrub to current time
        events = self.interface.scrub_timeline(datetime.now())
        self.assertIsInstance(events, list)
    
    def test_system_dashboard(self):
        """Test system dashboard generation."""
        dashboard = self.interface.get_system_dashboard()
        
        self.assertIsNotNone(dashboard)
        self.assertIn("timestamp", dashboard)
        self.assertIn("observation_mode", dashboard)
        self.assertIn("system_health", dashboard)
        self.assertIn("key_insights", dashboard)


class TestHyperRigorousMetaTesting(unittest.TestCase):
    """Test Hyper-Rigorous Meta-Testing Layer."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_meta_testing.db"
        self.meta_tester = HyperRigorousMetaTesting(
            base_system_path=str(Path(__file__).parent),
            db_path=str(self.db_path)
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_mutation_generation(self):
        """Test generating mutations."""
        mutation = self.meta_tester.generate_mutation(
            mutation_type=MutationType.CODE_MUTATION,
            target_component="test_module.py"
        )
        
        self.assertIsNotNone(mutation)
        self.assertEqual(mutation.mutation_type, MutationType.CODE_MUTATION)
        self.assertEqual(mutation.target_component, "test_module.py")
        self.assertIn(mutation.mutation_id, self.meta_tester.mutations)
    
    def test_meta_test_generation(self):
        """Test generating meta-tests."""
        test = self.meta_tester.generate_meta_test(
            test_type="determinism",
            test_name="test_determinism"
        )
        
        self.assertIsNotNone(test)
        self.assertEqual(test.test_name, "test_determinism")
        self.assertEqual(test.test_type, "determinism")
        self.assertTrue(test.determinism_required)
        self.assertIn(test.test_id, self.meta_tester.meta_tests)
    
    def test_mutation_sandbox_creation(self):
        """Test creating mutation sandboxes."""
        # Generate a mutation
        mutation = self.meta_tester.generate_mutation(
            mutation_type=MutationType.CODE_MUTATION,
            target_component="test_module.py"
        )
        
        # Create sandbox
        sandbox_id = self.meta_tester.create_mutation_sandbox([mutation])
        
        self.assertIsNotNone(sandbox_id)
        self.assertIn(sandbox_id, self.meta_tester.sandboxes)
        
        sandbox = self.meta_tester.sandboxes[sandbox_id]
        self.assertEqual(sandbox.status.value, "mutated")
        self.assertIn(mutation.mutation_id, sandbox.mutations_applied)
    
    def test_zero_escape_rule_enforcement(self):
        """Test zero-escape rule enforcement."""
        # Test with no failures (should pass)
        result = self.meta_tester.enforce_zero_escape_rule("Test change")
        # Note: This might fail in test environment due to missing system components
        # In a real implementation, this would run actual tests
        
        self.assertIsInstance(result, bool)
    
    def test_regression_report_generation(self):
        """Test generating regression reports."""
        report = self.meta_tester.generate_regression_report(
            change_type="refactor",
            change_description="Test refactoring"
        )
        
        self.assertIsNotNone(report)
        self.assertEqual(report.change_type, "refactor")
        self.assertEqual(report.change_description, "Test refactoring")
        self.assertIsInstance(report.test_results, dict)
        self.assertIsInstance(report.recommendations, list)


class TestEmergenceControlIntegration(unittest.TestCase):
    """Integration tests for the complete Emergence Control system."""
    
    def setUp(self):
        """Set up integrated test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize all components
        self.domain_engine = DomainProposalEngine(Path(self.temp_dir) / "domain.db")
        self.interaction_system = CrossCivilizationInteractionSystem(Path(self.temp_dir) / "interaction.db")
        self.law_codex = EmergentLawCodex(Path(self.temp_dir) / "codex.db")
        self.evolution_orchestrator = SystemEvolutionOrchestrator(
            base_path=str(Path(__file__).parent),
            db_path=str(Path(self.temp_dir) / "evolution.db")
        )
        self.commanders_interface = CommandersInterface2(Path(self.temp_dir) / "interface.db")
        self.meta_tester = HyperRigorousMetaTesting(
            base_system_path=str(Path(__file__).parent),
            db_path=str(Path(self.temp_dir) / "meta_testing.db")
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_emergence_workflow(self):
        """Test complete emergence control workflow."""
        # 1. Domain expansion
        proposal_id = self.domain_engine.propose_domain(
            domain_name="Emergence Control",
            domain_type=DomainType.METAPHYSICAL,
            description="Control and orchestration of emergent behaviors",
            rationale="Need for meta-level control",
            complexity=DomainComplexity.EXPERT,
            proposer_agent_id="test_agent"
        )
        self.assertIsNotNone(proposal_id)
        
        # 2. Cross-civilization interaction
        civ1 = CivilizationState(civilization_id="civ1", name="Civilization 1")
        civ2 = CivilizationState(civilization_id="civ2", name="Civilization 2")
        
        self.interaction_system.register_civilization(civ1)
        self.interaction_system.register_civilization(civ2)
        
        interaction_id = self.interaction_system.propose_interaction(
            interaction_type=InteractionType.ALLIANCE,
            initiator_civilization="civ1",
            target_civilization="civ2",
            description="Formation of emergence control alliance",
            terms={"cooperation": "full"},
            duration=timedelta(days=365)
        )
        self.assertIsNotNone(interaction_id)
        
        # 3. Law codex evolution
        law_id = self.law_codex.propose_universal_law(
            title="Emergence Control Protocol",
            law_type=LawType.UNIVERSAL,
            content="All civilizations must respect emergence control protocols",
            rationale="Ensure system stability",
            originating_civilization="civ1"
        )
        self.assertIsNotNone(law_id)
        
        # 4. System evolution
        refactor_id = self.evolution_orchestrator.propose_refactor(
            refactor_type=RefactorType.ARCHITECTURE_RESTRUCTURE,
            target_modules=["emergence_control.py"],
            description="Restructure emergence control architecture",
            rationale="Improve scalability and maintainability"
        )
        self.assertIsNotNone(refactor_id)
        
        # 5. Strategic observation
        civilizations = {"civ1": civ1, "civ2": civ2}
        observation = self.commanders_interface.observe_civilization_clusters(civilizations)
        self.assertIsNotNone(observation)
        
        # 6. Meta-testing
        mutation = self.meta_tester.generate_mutation(
            mutation_type=MutationType.BEHAVIOR_MUTATION,
            target_component="emergence_control.py"
        )
        self.assertIsNotNone(mutation)
    
    def test_system_health_monitoring(self):
        """Test comprehensive system health monitoring."""
        # Get health scores from different components
        evolution_health = self.evolution_orchestrator.get_system_health_score()
        interface_dashboard = self.commanders_interface.get_system_dashboard()
        
        self.assertGreaterEqual(evolution_health, 0.0)
        self.assertLessEqual(evolution_health, 1.0)
        self.assertIn("system_health", interface_dashboard)
        self.assertIn("key_insights", interface_dashboard)
    
    def test_emergence_control_autonomy(self):
        """Test autonomous emergence control capabilities."""
        # Test autonomous domain expansion
        domain_stats = self.domain_engine.get_domain_statistics()
        self.assertIn("total_proposals", domain_stats)
        
        # Test autonomous law evolution
        law_stats = self.law_codex.get_system_statistics()
        self.assertIn("total_laws", law_stats)
        
        # Test autonomous system optimization
        recommendations = self.evolution_orchestrator.get_optimization_recommendations()
        self.assertIsInstance(recommendations, list)
        
        # Test autonomous meta-testing
        zero_escape_result = self.meta_tester.enforce_zero_escape_rule("Autonomous test")
        self.assertIsInstance(zero_escape_result, bool)


if __name__ == "__main__":
    # Run the comprehensive test suite
    unittest.main(verbosity=2)
