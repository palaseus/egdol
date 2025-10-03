"""
Comprehensive tests for OmniMind Experimental Intelligence System
Tests all components with unit, integration, and property-based tests.
"""

import pytest
import uuid
import random
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Import experimental intelligence components
from egdol.omnimind.experimental import (
    HypothesisGenerator, Hypothesis, HypothesisType, HypothesisStatus,
    ExperimentExecutor, Experiment, ExperimentType, ExperimentStatus,
    ResultAnalyzer, ExperimentResult, ResultType, ConfidenceLevel,
    CreativeSynthesizer, CreativeOutput, SynthesisType, InnovationLevel,
    AutonomousResearcher, ResearchProject, ResearchPhase, ResearchStatus,
    KnowledgeExpander, KnowledgeItem, ExpansionStrategy, DiscoveryType, IntegrationStatus,
    ExperimentalCoordinator, ResearchCycle, CycleStatus
)


class TestHypothesisGenerator:
    """Test hypothesis generation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_network = Mock()
        self.mock_network.get_network_statistics.return_value = {
            'agent_count': 3,
            'message_count': 100,
            'collaboration_rate': 0.8
        }
        self.mock_network.get_all_agents.return_value = []
        self.mock_network.get_available_agents.return_value = []
        
        self.mock_memory = Mock()
        self.mock_knowledge_graph = Mock()
        
        self.hypothesis_generator = HypothesisGenerator(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph
        )
    
    def test_generate_hypotheses(self):
        """Test hypothesis generation."""
        # Mock network statistics
        self.mock_network.get_network_statistics.return_value = {
            'agent_count': 3,
            'message_count': 100,
            'collaboration_rate': 0.8
        }
        
        # Mock agent data
        mock_agent1 = Mock()
        mock_agent1.id = 'agent1'
        mock_agent1.skills = ['math', 'logic']
        mock_agent2 = Mock()
        mock_agent2.id = 'agent2'
        mock_agent2.skills = ['reasoning']
        mock_agent3 = Mock()
        mock_agent3.id = 'agent3'
        mock_agent3.skills = ['optimization']
        
        self.mock_network.get_all_agents.return_value = [mock_agent1, mock_agent2, mock_agent3]
        
        # Generate hypotheses
        hypotheses = self.hypothesis_generator.generate_hypotheses()
        
        assert len(hypotheses) > 0
        assert all(isinstance(h, Hypothesis) for h in hypotheses)
        assert all(h.id in self.hypothesis_generator.hypotheses for h in hypotheses)
    
    def test_hypothesis_types(self):
        """Test different hypothesis types."""
        hypotheses = self.hypothesis_generator.generate_hypotheses()
        
        types_found = set(h.type for h in hypotheses)
        expected_types = {
            HypothesisType.KNOWLEDGE_GAP,
            HypothesisType.EMERGENT_BEHAVIOR,
            HypothesisType.OPTIMIZATION,
            HypothesisType.CROSS_DOMAIN,
            HypothesisType.CREATIVE_SYNTHESIS
        }
        
        assert len(types_found.intersection(expected_types)) > 0
    
    def test_hypothesis_confidence(self):
        """Test hypothesis confidence scoring."""
        hypotheses = self.hypothesis_generator.generate_hypotheses()
        
        for hypothesis in hypotheses:
            assert 0.0 <= hypothesis.confidence <= 1.0
            assert 0.0 <= hypothesis.priority <= 1.0
            assert 0.0 <= hypothesis.complexity <= 1.0
    
    def test_update_hypothesis_status(self):
        """Test hypothesis status updates."""
        hypotheses = self.hypothesis_generator.generate_hypotheses()
        hypothesis = hypotheses[0]
        
        # Update status
        success = self.hypothesis_generator.update_hypothesis_status(
            hypothesis.id, HypothesisStatus.VALIDATED
        )
        
        assert success
        assert self.hypothesis_generator.hypotheses[hypothesis.id].status == HypothesisStatus.VALIDATED
    
    def test_get_hypothesis_statistics(self):
        """Test hypothesis statistics."""
        # Generate some hypotheses
        self.hypothesis_generator.generate_hypotheses()
        
        stats = self.hypothesis_generator.get_hypothesis_statistics()
        
        assert 'total_hypotheses' in stats
        assert 'status_distribution' in stats
        assert 'type_distribution' in stats
        assert 'average_confidence' in stats
        assert 'average_priority' in stats
        assert stats['total_hypotheses'] > 0
    
    def test_boost_creativity(self):
        """Test creativity boost functionality."""
        original_boost = self.hypothesis_generator.creativity_boost
        
        # Boost creativity
        self.hypothesis_generator.boost_creativity(2.0)
        assert self.hypothesis_generator.creativity_boost == 2.0
        
        # Test bounds
        self.hypothesis_generator.boost_creativity(5.0)
        assert self.hypothesis_generator.creativity_boost == 3.0  # Max bound
        
        self.hypothesis_generator.boost_creativity(0.05)
        assert self.hypothesis_generator.creativity_boost == 0.1  # Min bound


class TestExperimentExecutor:
    """Test experiment execution system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_network = Mock()
        self.mock_network.get_network_statistics.return_value = {
            'agent_count': 3,
            'message_count': 100,
            'collaboration_rate': 0.8
        }
        self.mock_network.get_all_agents.return_value = []
        self.mock_network.get_available_agents.return_value = []
        
        self.mock_memory = Mock()
        self.mock_knowledge_graph = Mock()
        
        self.experiment_executor = ExperimentExecutor(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph
        )
    
    def test_create_experiment(self):
        """Test experiment creation."""
        hypothesis_id = str(uuid.uuid4())
        experiment_type = ExperimentType.SIMULATION
        parameters = {
            'duration_minutes': 30,
            'resource_requirements': {'computational': 0.8},
            'success_criteria': ['accuracy > 0.8']
        }
        
        experiment = self.experiment_executor.create_experiment(
            hypothesis_id, experiment_type, parameters
        )
        
        assert isinstance(experiment, Experiment)
        assert experiment.hypothesis_id == hypothesis_id
        assert experiment.type == experiment_type
        assert experiment.id in self.experiment_executor.experiments
        assert experiment.id in self.experiment_executor.execution_queue
    
    def test_execute_simulation_experiment(self):
        """Test simulation experiment execution."""
        # Create experiment
        hypothesis_id = str(uuid.uuid4())
        experiment = self.experiment_executor.create_experiment(
            hypothesis_id, ExperimentType.SIMULATION, {}
        )
        
        # Execute experiment
        success = self.experiment_executor.execute_experiment(experiment.id)
        
        assert isinstance(success, bool)
        assert experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]
        assert experiment.started_at is not None
        assert experiment.completed_at is not None
    
    def test_execute_controlled_test(self):
        """Test controlled test experiment execution."""
        hypothesis_id = str(uuid.uuid4())
        experiment = self.experiment_executor.create_experiment(
            hypothesis_id, ExperimentType.CONTROLLED_TEST, {}
        )
        
        success = self.experiment_executor.execute_experiment(experiment.id)
        
        assert isinstance(success, bool)
        assert experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]
    
    def test_execute_multi_agent_experiment(self):
        """Test multi-agent experiment execution."""
        # Mock available agents
        mock_agent = Mock()
        mock_agent.id = 'agent1'
        self.mock_network.get_available_agents.return_value = [mock_agent]
        
        hypothesis_id = str(uuid.uuid4())
        experiment = self.experiment_executor.create_experiment(
            hypothesis_id, ExperimentType.MULTI_AGENT_COLLABORATION, {}
        )
        
        success = self.experiment_executor.execute_experiment(experiment.id)
        
        assert isinstance(success, bool)
        assert experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]
    
    def test_experiment_metrics(self):
        """Test experiment metrics collection."""
        hypothesis_id = str(uuid.uuid4())
        experiment = self.experiment_executor.create_experiment(
            hypothesis_id, ExperimentType.SIMULATION, {}
        )
        
        self.experiment_executor.execute_experiment(experiment.id)
        
        # Check that metrics were collected
        assert len(experiment.metrics) > 0
        assert 'simulation_accuracy' in experiment.metrics or 'test_accuracy' in experiment.metrics
    
    def test_cancel_experiment(self):
        """Test experiment cancellation."""
        hypothesis_id = str(uuid.uuid4())
        experiment = self.experiment_executor.create_experiment(
            hypothesis_id, ExperimentType.SIMULATION, {}
        )
        
        # Set experiment to RUNNING status first
        experiment.status = ExperimentStatus.RUNNING
        
        # Cancel experiment
        success = self.experiment_executor.cancel_experiment(experiment.id)
        
        assert success
        assert experiment.status == ExperimentStatus.CANCELLED
    
    def test_get_experiment_statistics(self):
        """Test experiment statistics."""
        # Create and execute some experiments
        for i in range(3):
            hypothesis_id = str(uuid.uuid4())
            experiment = self.experiment_executor.create_experiment(
                hypothesis_id, ExperimentType.SIMULATION, {}
            )
            self.experiment_executor.execute_experiment(experiment.id)
        
        stats = self.experiment_executor.get_experiment_statistics()
        
        assert 'total_experiments' in stats
        assert 'completed_experiments' in stats
        assert 'failed_experiments' in stats
        assert 'success_rate' in stats
        assert stats['total_experiments'] >= 3


class TestResultAnalyzer:
    """Test result analysis system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_network = Mock()
        self.mock_memory = Mock()
        self.mock_knowledge_graph = Mock()
        
        self.result_analyzer = ResultAnalyzer(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph
        )
    
    def test_analyze_experiment_result(self):
        """Test experiment result analysis."""
        experiment_id = str(uuid.uuid4())
        experiment_data = {
            'hypothesis_id': str(uuid.uuid4()),
            'metrics': {
                'accuracy': 0.85,
                'efficiency': 0.9,
                'significance': 0.8
            },
            'results': {
                'performance': 0.88,
                'improvement': 0.15
            },
            'errors': []
        }
        
        analysis = self.result_analyzer.analyze_experiment_result(experiment_id, experiment_data)
        
        assert isinstance(analysis, ExperimentResult)
        assert analysis.experiment_id == experiment_id
        assert analysis.result_type in [ResultType.SUCCESS, ResultType.PARTIAL_SUCCESS, 
                                       ResultType.FAILURE, ResultType.INCONCLUSIVE]
        assert 0.0 <= analysis.confidence_score <= 1.0
        assert len(analysis.insights) > 0
        assert len(analysis.implications) > 0
        assert len(analysis.recommendations) > 0
    
    def test_result_type_determination(self):
        """Test result type determination."""
        # Test success case
        success_data = {
            'metrics': {'accuracy': 0.9, 'efficiency': 0.95},
            'results': {'performance': 0.92},
            'errors': []
        }
        
        analysis = self.result_analyzer.analyze_experiment_result(
            str(uuid.uuid4()), success_data
        )
        
        assert analysis.result_type in [ResultType.SUCCESS, ResultType.PARTIAL_SUCCESS]
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        high_confidence_data = {
            'metrics': {'accuracy': 0.95, 'significance': 0.9, 'reproducibility': 0.9},
            'results': {'performance': 0.93},
            'errors': []
        }
        
        analysis = self.result_analyzer.analyze_experiment_result(
            str(uuid.uuid4()), high_confidence_data
        )
        
        assert analysis.confidence_score >= 0.7
    
    def test_insight_generation(self):
        """Test insight generation."""
        experiment_data = {
            'metrics': {'accuracy': 0.8, 'efficiency': 0.7},
            'results': {'performance': 0.75},
            'errors': []
        }
        
        analysis = self.result_analyzer.analyze_experiment_result(
            str(uuid.uuid4()), experiment_data
        )
        
        assert len(analysis.insights) > 0
        assert all(isinstance(insight, str) for insight in analysis.insights)
    
    def test_get_analysis_statistics(self):
        """Test analysis statistics."""
        # Analyze some results
        for i in range(3):
            experiment_data = {
                'metrics': {'accuracy': random.uniform(0.7, 0.9)},
                'results': {'performance': random.uniform(0.6, 0.9)},
                'errors': []
            }
            self.result_analyzer.analyze_experiment_result(str(uuid.uuid4()), experiment_data)
        
        stats = self.result_analyzer.get_analysis_statistics()
        
        assert 'total_analyses' in stats
        assert 'result_type_distribution' in stats
        assert 'confidence_distribution' in stats
        assert 'average_confidence' in stats
        assert stats['total_analyses'] >= 3


class TestCreativeSynthesizer:
    """Test creative synthesis system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_network = Mock()
        self.mock_memory = Mock()
        self.mock_knowledge_graph = Mock()
        
        self.creative_synthesizer = CreativeSynthesizer(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph
        )
    
    def test_synthesize_rule(self):
        """Test rule synthesis."""
        output = self.creative_synthesizer.synthesize_creative_output(
            SynthesisType.RULE_GENERATION
        )
        
        assert isinstance(output, CreativeOutput)
        assert output.type == SynthesisType.RULE_GENERATION
        assert 0.0 <= output.novelty_score <= 1.0
        assert 0.0 <= output.usefulness_score <= 1.0
        assert 0.0 <= output.feasibility_score <= 1.0
        assert output.id in self.creative_synthesizer.creative_outputs
    
    def test_synthesize_skill(self):
        """Test skill synthesis."""
        output = self.creative_synthesizer.synthesize_creative_output(
            SynthesisType.SKILL_CREATION
        )
        
        assert isinstance(output, CreativeOutput)
        assert output.type == SynthesisType.SKILL_CREATION
        assert output.innovation_level in [InnovationLevel.INCREMENTAL, InnovationLevel.MODERATE,
                                          InnovationLevel.BREAKTHROUGH, InnovationLevel.REVOLUTIONARY]
    
    def test_synthesize_strategy(self):
        """Test strategy synthesis."""
        output = self.creative_synthesizer.synthesize_creative_output(
            SynthesisType.STRATEGY_DEVELOPMENT
        )
        
        assert isinstance(output, CreativeOutput)
        assert output.type == SynthesisType.STRATEGY_DEVELOPMENT
        assert len(output.content) > 0
    
    def test_cross_domain_fusion(self):
        """Test cross-domain fusion synthesis."""
        output = self.creative_synthesizer.synthesize_creative_output(
            SynthesisType.CROSS_DOMAIN_FUSION
        )
        
        assert isinstance(output, CreativeOutput)
        assert output.type == SynthesisType.CROSS_DOMAIN_FUSION
        assert len(output.cross_domain_connections) > 0
    
    def test_novel_approach(self):
        """Test novel approach synthesis."""
        output = self.creative_synthesizer.synthesize_creative_output(
            SynthesisType.NOVEL_APPROACH
        )
        
        assert isinstance(output, CreativeOutput)
        assert output.type == SynthesisType.NOVEL_APPROACH
        assert output.novelty_score >= 0.6  # Should be high novelty
    
    def test_optimization_innovation(self):
        """Test optimization innovation synthesis."""
        output = self.creative_synthesizer.synthesize_creative_output(
            SynthesisType.OPTIMIZATION_INNOVATION
        )
        
        assert isinstance(output, CreativeOutput)
        assert output.type == SynthesisType.OPTIMIZATION_INNOVATION
        assert 'optimization' in output.title.lower()
    
    def test_boost_creativity(self):
        """Test creativity boost functionality."""
        original_boost = self.creative_synthesizer.creativity_boost
        
        # Boost creativity
        self.creative_synthesizer.boost_creativity(2.0)
        assert self.creative_synthesizer.creativity_boost == 2.0
        
        # Test bounds
        self.creative_synthesizer.boost_creativity(5.0)
        assert self.creative_synthesizer.creativity_boost == 3.0  # Max bound
    
    def test_get_creative_output_statistics(self):
        """Test creative output statistics."""
        # Generate some outputs
        for synthesis_type in SynthesisType:
            self.creative_synthesizer.synthesize_creative_output(synthesis_type)
        
        stats = self.creative_synthesizer.get_creative_output_statistics()
        
        assert 'total_outputs' in stats
        assert 'type_distribution' in stats
        assert 'innovation_distribution' in stats
        assert 'average_novelty' in stats
        assert 'average_usefulness' in stats
        assert 'average_feasibility' in stats
        assert stats['total_outputs'] >= len(SynthesisType)


class TestAutonomousResearcher:
    """Test autonomous research system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_network = Mock()
        self.mock_network.get_network_statistics.return_value = {
            'agent_count': 3,
            'message_count': 100,
            'collaboration_rate': 0.8
        }
        self.mock_network.get_all_agents.return_value = []
        self.mock_network.get_available_agents.return_value = []
        
        self.mock_memory = Mock()
        self.mock_knowledge_graph = Mock()
        self.mock_hypothesis_generator = Mock()
        self.mock_experiment_executor = Mock()
        
        self.autonomous_researcher = AutonomousResearcher(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph,
            self.mock_hypothesis_generator, self.mock_experiment_executor
        )
    
    def test_initiate_research_project(self):
        """Test research project initiation."""
        title = "Test Research Project"
        description = "A test research project"
        objectives = ["Objective 1", "Objective 2"]
        methodology = "Experimental"
        
        project = self.autonomous_researcher.initiate_research_project(
            title, description, objectives, methodology
        )
        
        assert isinstance(project, ResearchProject)
        assert project.title == title
        assert project.description == description
        assert project.objectives == objectives
        assert project.methodology == methodology
        assert project.id in self.autonomous_researcher.research_projects
        assert project.id in self.autonomous_researcher.research_queue
    
    def test_execute_planning_phase(self):
        """Test planning phase execution."""
        project = self.autonomous_researcher.initiate_research_project(
            "Test Project", "Description", ["Objective"], "Methodology"
        )
        
        success = self.autonomous_researcher.execute_research_phase(
            project.id, ResearchPhase.PLANNING
        )
        
        assert success
        assert project.current_phase == ResearchPhase.PLANNING
        assert project.progress > 0
        assert len(project.findings) > 0
    
    def test_execute_data_collection_phase(self):
        """Test data collection phase execution."""
        project = self.autonomous_researcher.initiate_research_project(
            "Test Project", "Description", ["Objective"], "Methodology"
        )
        
        success = self.autonomous_researcher.execute_research_phase(
            project.id, ResearchPhase.DATA_COLLECTION
        )
        
        assert success
        assert project.current_phase == ResearchPhase.DATA_COLLECTION
        assert project.progress > 0.1
        assert len(project.findings) > 0
    
    def test_execute_analysis_phase(self):
        """Test analysis phase execution."""
        project = self.autonomous_researcher.initiate_research_project(
            "Test Project", "Description", ["Objective"], "Methodology"
        )
        
        success = self.autonomous_researcher.execute_research_phase(
            project.id, ResearchPhase.ANALYSIS
        )
        
        assert success
        assert project.current_phase == ResearchPhase.ANALYSIS
        assert project.progress > 0.3
        assert len(project.findings) > 0
    
    def test_execute_synthesis_phase(self):
        """Test synthesis phase execution."""
        project = self.autonomous_researcher.initiate_research_project(
            "Test Project", "Description", ["Objective"], "Methodology"
        )
        
        success = self.autonomous_researcher.execute_research_phase(
            project.id, ResearchPhase.SYNTHESIS
        )
        
        assert success
        assert project.current_phase == ResearchPhase.SYNTHESIS
        assert project.progress > 0.6
        assert len(project.findings) > 0
    
    def test_execute_validation_phase(self):
        """Test validation phase execution."""
        project = self.autonomous_researcher.initiate_research_project(
            "Test Project", "Description", ["Objective"], "Methodology"
        )
        
        success = self.autonomous_researcher.execute_research_phase(
            project.id, ResearchPhase.VALIDATION
        )
        
        assert success
        assert project.current_phase == ResearchPhase.VALIDATION
        assert project.progress > 0.8
        assert len(project.findings) > 0
    
    def test_execute_integration_phase(self):
        """Test integration phase execution."""
        project = self.autonomous_researcher.initiate_research_project(
            "Test Project", "Description", ["Objective"], "Methodology"
        )
        
        success = self.autonomous_researcher.execute_research_phase(
            project.id, ResearchPhase.INTEGRATION
        )
        
        assert success
        assert project.current_phase == ResearchPhase.INTEGRATION
        assert project.progress == 1.0
        assert project.status == ResearchStatus.COMPLETED
        assert project.completed_at is not None
    
    def test_pause_resume_research_project(self):
        """Test pausing and resuming research projects."""
        project = self.autonomous_researcher.initiate_research_project(
            "Test Project", "Description", ["Objective"], "Methodology"
        )
        
        # Pause project
        success = self.autonomous_researcher.pause_research_project(project.id)
        assert success
        assert project.status == ResearchStatus.PAUSED
        
        # Resume project
        success = self.autonomous_researcher.resume_research_project(project.id)
        assert success
        assert project.status == ResearchStatus.ACTIVE
    
    def test_cancel_research_project(self):
        """Test canceling research projects."""
        project = self.autonomous_researcher.initiate_research_project(
            "Test Project", "Description", ["Objective"], "Methodology"
        )
        
        success = self.autonomous_researcher.cancel_research_project(project.id)
        assert success
        assert project.status == ResearchStatus.CANCELLED
    
    def test_get_research_statistics(self):
        """Test research statistics."""
        # Create some projects
        for i in range(3):
            self.autonomous_researcher.initiate_research_project(
                f"Project {i}", f"Description {i}", [f"Objective {i}"], "Methodology"
            )
        
        stats = self.autonomous_researcher.get_research_statistics()
        
        assert 'total_projects' in stats
        assert 'active_projects' in stats
        assert 'completed_projects' in stats
        assert 'phase_distribution' in stats
        assert 'average_progress' in stats
        assert stats['total_projects'] >= 3


class TestKnowledgeExpander:
    """Test knowledge expansion system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_network = Mock()
        self.mock_network.get_network_statistics.return_value = {
            'agent_count': 3,
            'message_count': 100,
            'collaboration_rate': 0.8
        }
        self.mock_network.get_all_agents.return_value = []
        self.mock_network.get_available_agents.return_value = []
        
        self.mock_memory = Mock()
        self.mock_knowledge_graph = Mock()
        self.mock_hypothesis_generator = Mock()
        
        self.knowledge_expander = KnowledgeExpander(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph,
            self.mock_hypothesis_generator
        )
    
    def test_discover_knowledge_deep_dive(self):
        """Test deep dive knowledge discovery."""
        discoveries = self.knowledge_expander.discover_knowledge(
            ExpansionStrategy.DEEP_DIVE
        )
        
        assert len(discoveries) > 0
        assert all(isinstance(d, KnowledgeItem) for d in discoveries)
        assert all(d.id in self.knowledge_expander.knowledge_items for d in discoveries)
        assert all(d.id in self.knowledge_expander.expansion_queue for d in discoveries)
    
    def test_discover_knowledge_breadth_exploration(self):
        """Test breadth exploration knowledge discovery."""
        discoveries = self.knowledge_expander.discover_knowledge(
            ExpansionStrategy.BREADTH_EXPLORATION
        )
        
        assert len(discoveries) > 0
        assert all(isinstance(d, KnowledgeItem) for d in discoveries)
        assert all(d.type == DiscoveryType.NEW_FACT for d in discoveries)
    
    def test_discover_knowledge_cross_domain_fusion(self):
        """Test cross-domain fusion knowledge discovery."""
        discoveries = self.knowledge_expander.discover_knowledge(
            ExpansionStrategy.CROSS_DOMAIN_FUSION
        )
        
        assert len(discoveries) > 0
        assert all(isinstance(d, KnowledgeItem) for d in discoveries)
        assert all(d.type == DiscoveryType.NEW_RELATIONSHIP for d in discoveries)
    
    def test_discover_knowledge_pattern_extension(self):
        """Test pattern extension knowledge discovery."""
        discoveries = self.knowledge_expander.discover_knowledge(
            ExpansionStrategy.PATTERN_EXTENSION
        )
        
        assert len(discoveries) > 0
        assert all(isinstance(d, KnowledgeItem) for d in discoveries)
        assert all(d.type == DiscoveryType.NEW_PATTERN for d in discoveries)
    
    def test_discover_knowledge_gap_filling(self):
        """Test gap filling knowledge discovery."""
        discoveries = self.knowledge_expander.discover_knowledge(
            ExpansionStrategy.GAP_FILLING
        )
        
        assert len(discoveries) > 0
        assert all(isinstance(d, KnowledgeItem) for d in discoveries)
        assert all(d.type == DiscoveryType.NEW_FACT for d in discoveries)
    
    def test_discover_knowledge_emergent_discovery(self):
        """Test emergent discovery knowledge discovery."""
        discoveries = self.knowledge_expander.discover_knowledge(
            ExpansionStrategy.EMERGENT_DISCOVERY
        )
        
        assert len(discoveries) > 0
        assert all(isinstance(d, KnowledgeItem) for d in discoveries)
        assert all(d.type == DiscoveryType.NEW_PATTERN for d in discoveries)
    
    def test_integrate_knowledge(self):
        """Test knowledge integration."""
        # First discover some knowledge
        discoveries = self.knowledge_expander.discover_knowledge(
            ExpansionStrategy.DEEP_DIVE
        )
        
        # Integrate first discovery
        if discoveries:
            success = self.knowledge_expander.integrate_knowledge(discoveries[0].id)
            assert isinstance(success, bool)
            
            # Check integration status
            knowledge_item = self.knowledge_expander.knowledge_items[discoveries[0].id]
            assert knowledge_item.integration_status in [
                IntegrationStatus.COMPLETED, IntegrationStatus.FAILED,
                IntegrationStatus.CONFLICTED
            ]
    
    def test_get_expansion_statistics(self):
        """Test expansion statistics."""
        # Discover some knowledge
        for strategy in ExpansionStrategy:
            self.knowledge_expander.discover_knowledge(strategy)
        
        stats = self.knowledge_expander.get_expansion_statistics()
        
        assert 'total_knowledge_items' in stats
        assert 'integrated_items' in stats
        assert 'pending_items' in stats
        assert 'type_distribution' in stats
        assert 'integration_status_distribution' in stats
        assert stats['total_knowledge_items'] > 0
    
    def test_get_items_by_type(self):
        """Test getting items by type."""
        # Discover knowledge
        self.knowledge_expander.discover_knowledge(ExpansionStrategy.DEEP_DIVE)
        
        # Get items by type
        pattern_items = self.knowledge_expander.get_items_by_type(DiscoveryType.NEW_PATTERN)
        fact_items = self.knowledge_expander.get_items_by_type(DiscoveryType.NEW_FACT)
        
        assert isinstance(pattern_items, list)
        assert isinstance(fact_items, list)
    
    def test_get_high_confidence_items(self):
        """Test getting high confidence items."""
        # Discover knowledge
        self.knowledge_expander.discover_knowledge(ExpansionStrategy.DEEP_DIVE)
        
        # Get high confidence items
        high_confidence_items = self.knowledge_expander.get_high_confidence_items(0.8)
        
        assert isinstance(high_confidence_items, list)
        assert all(item.confidence >= 0.8 for item in high_confidence_items)


class TestExperimentalCoordinator:
    """Test experimental coordination system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_network = Mock()
        self.mock_memory = Mock()
        self.mock_knowledge_graph = Mock()
        self.mock_hypothesis_generator = Mock()
        self.mock_experiment_executor = Mock()
        self.mock_result_analyzer = Mock()
        self.mock_creative_synthesizer = Mock()
        self.mock_autonomous_researcher = Mock()
        self.mock_knowledge_expander = Mock()
        
        self.experimental_coordinator = ExperimentalCoordinator(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph,
            self.mock_hypothesis_generator, self.mock_experiment_executor,
            self.mock_result_analyzer, self.mock_creative_synthesizer,
            self.mock_autonomous_researcher, self.mock_knowledge_expander
        )
    
    def test_initiate_research_cycle(self):
        """Test research cycle initiation."""
        title = "Test Research Cycle"
        description = "A test research cycle"
        objectives = ["Objective 1", "Objective 2"]
        
        cycle = self.experimental_coordinator.initiate_research_cycle(
            title, description, objectives
        )
        
        assert isinstance(cycle, ResearchCycle)
        assert cycle.title == title
        assert cycle.description == description
        assert cycle.objectives == objectives
        assert cycle.id in self.experimental_coordinator.research_cycles
        assert cycle.id in self.experimental_coordinator.active_cycles
    
    def test_execute_research_cycle(self):
        """Test research cycle execution."""
        # Mock the component methods
        self.mock_hypothesis_generator.generate_hypotheses.return_value = []
        self.mock_experiment_executor.create_experiment.return_value = Mock()
        self.mock_experiment_executor.execute_experiment.return_value = True
        self.mock_result_analyzer.analyze_experiment_result.return_value = Mock()
        self.mock_creative_synthesizer.synthesize_creative_output.return_value = Mock()
        self.mock_knowledge_expander.discover_knowledge.return_value = []
        self.mock_knowledge_expander.integrate_knowledge.return_value = True
        
        cycle = self.experimental_coordinator.initiate_research_cycle(
            "Test Cycle", "Description", ["Objective"]
        )
        
        success = self.experimental_coordinator.execute_research_cycle(cycle.id)
        
        assert isinstance(success, bool)
        if success:
            assert cycle.status == CycleStatus.COMPLETED
            assert cycle.progress == 1.0
            assert cycle.completed_at is not None
    
    def test_pause_resume_cycle(self):
        """Test pausing and resuming cycles."""
        cycle = self.experimental_coordinator.initiate_research_cycle(
            "Test Cycle", "Description", ["Objective"]
        )
        
        # Pause cycle
        success = self.experimental_coordinator.pause_cycle(cycle.id)
        assert isinstance(success, bool)
        
        # Resume cycle
        success = self.experimental_coordinator.resume_cycle(cycle.id)
        assert isinstance(success, bool)
    
    def test_cancel_cycle(self):
        """Test canceling cycles."""
        cycle = self.experimental_coordinator.initiate_research_cycle(
            "Test Cycle", "Description", ["Objective"]
        )
        
        success = self.experimental_coordinator.cancel_cycle(cycle.id)
        assert success
        assert cycle.status == CycleStatus.FAILED
        assert cycle.id not in self.experimental_coordinator.active_cycles
    
    def test_get_coordination_statistics(self):
        """Test coordination statistics."""
        # Create some cycles
        for i in range(3):
            self.experimental_coordinator.initiate_research_cycle(
                f"Cycle {i}", f"Description {i}", [f"Objective {i}"]
            )
        
        stats = self.experimental_coordinator.get_coordination_statistics()
        
        assert 'total_cycles' in stats
        assert 'active_cycles' in stats
        assert 'completed_cycles' in stats
        assert 'failed_cycles' in stats
        assert 'status_distribution' in stats
        assert 'average_progress' in stats
        assert stats['total_cycles'] >= 3
    
    def test_get_cycle_performance(self):
        """Test cycle performance metrics."""
        cycle = self.experimental_coordinator.initiate_research_cycle(
            "Test Cycle", "Description", ["Objective"]
        )
        
        performance = self.experimental_coordinator.get_cycle_performance(cycle.id)
        
        assert 'cycle_id' in performance
        assert 'title' in performance
        assert 'progress' in performance
        assert 'status' in performance
        assert performance['cycle_id'] == cycle.id


class TestIntegration:
    """Integration tests for experimental intelligence system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_network = Mock()
        self.mock_network.get_network_statistics.return_value = {
            'agent_count': 3,
            'message_count': 100,
            'collaboration_rate': 0.8
        }
        self.mock_network.get_all_agents.return_value = []
        self.mock_network.get_available_agents.return_value = []
        
        self.mock_memory = Mock()
        self.mock_knowledge_graph = Mock()
        
        # Create all components
        self.hypothesis_generator = HypothesisGenerator(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph
        )
        self.experiment_executor = ExperimentExecutor(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph
        )
        self.result_analyzer = ResultAnalyzer(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph
        )
        self.creative_synthesizer = CreativeSynthesizer(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph
        )
        self.autonomous_researcher = AutonomousResearcher(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph,
            self.hypothesis_generator, self.experiment_executor
        )
        self.knowledge_expander = KnowledgeExpander(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph,
            self.hypothesis_generator
        )
        self.experimental_coordinator = ExperimentalCoordinator(
            self.mock_network, self.mock_memory, self.mock_knowledge_graph,
            self.hypothesis_generator, self.experiment_executor,
            self.result_analyzer, self.creative_synthesizer,
            self.autonomous_researcher, self.knowledge_expander
        )
    
    def test_full_experimental_workflow(self):
        """Test complete experimental workflow."""
        # Mock network data
        self.mock_network.get_network_statistics.return_value = {
            'agent_count': 3, 'message_count': 100, 'collaboration_rate': 0.8
        }
        self.mock_network.get_all_agents.return_value = []
        self.mock_network.get_available_agents.return_value = []
        
        # Initiate research cycle
        cycle = self.experimental_coordinator.initiate_research_cycle(
            "Integration Test Cycle",
            "Testing complete experimental workflow",
            ["Test objective 1", "Test objective 2"]
        )
        
        # Execute cycle
        success = self.experimental_coordinator.execute_research_cycle(cycle.id)
        
        # Verify cycle completion
        assert isinstance(success, bool)
        if success:
            assert cycle.status == CycleStatus.COMPLETED
            assert cycle.progress == 1.0
            assert len(cycle.hypotheses) > 0
            assert len(cycle.experiments) > 0
            assert len(cycle.findings) > 0
    
    def test_experimental_intelligence_resilience(self):
        """Test system resilience under various conditions."""
        # Test with different network states
        network_states = [
            {'agent_count': 1, 'message_count': 10, 'collaboration_rate': 0.3},
            {'agent_count': 10, 'message_count': 1000, 'collaboration_rate': 0.9},
            {'agent_count': 0, 'message_count': 0, 'collaboration_rate': 0.0}
        ]
        
        for state in network_states:
            self.mock_network.get_network_statistics.return_value = state
            self.mock_network.get_all_agents.return_value = []
            self.mock_network.get_available_agents.return_value = []
            
            # Test hypothesis generation
            hypotheses = self.hypothesis_generator.generate_hypotheses()
            assert isinstance(hypotheses, list)
            
            # Test creative synthesis
            for synthesis_type in SynthesisType:
                output = self.creative_synthesizer.synthesize_creative_output(synthesis_type)
                assert isinstance(output, CreativeOutput)
            
            # Test knowledge expansion
            for strategy in ExpansionStrategy:
                discoveries = self.knowledge_expander.discover_knowledge(strategy)
                assert isinstance(discoveries, list)
    
    def test_experimental_intelligence_scalability(self):
        """Test system scalability with multiple concurrent operations."""
        # Mock network data
        self.mock_network.get_network_statistics.return_value = {
            'agent_count': 5, 'message_count': 500, 'collaboration_rate': 0.8
        }
        self.mock_network.get_all_agents.return_value = []
        self.mock_network.get_available_agents.return_value = []
        
        # Create multiple research cycles
        cycles = []
        for i in range(5):
            cycle = self.experimental_coordinator.initiate_research_cycle(
                f"Scalability Test Cycle {i}",
                f"Testing scalability with cycle {i}",
                [f"Objective {i}"]
            )
            cycles.append(cycle)
        
        # Execute all cycles
        results = []
        for cycle in cycles:
            success = self.experimental_coordinator.execute_research_cycle(cycle.id)
            results.append(success)
        
        # Verify results
        assert len(results) == 5
        assert all(isinstance(result, bool) for result in results)
        
        # Check statistics
        stats = self.experimental_coordinator.get_coordination_statistics()
        assert stats['total_cycles'] >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
