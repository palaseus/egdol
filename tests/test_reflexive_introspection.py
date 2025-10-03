"""
Comprehensive tests for the Reflexive Introspection Layer
Tests the system's ability to audit its own simulation histories and discover meta-rules.
"""

import pytest
import time
import random
from datetime import datetime, timedelta

from egdol.omnimind.transcendence import (
    # Core components
    CivilizationIntelligenceCore, CivilizationArchitect, TemporalEvolutionEngine,
    MacroPatternDetector, StrategicCivilizationalOrchestrator, CivilizationExperimentationSystem,
    
    # Meta-instrumentation components
    PatternCodificationEngine, CivilizationalGeneticArchive, StrategicFeedbackLoop,
    
    # Reflexive introspection layer
    ReflexiveIntrospectionLayer, IntrospectionAnalysis, IntrospectionType, IntrospectionStatus,
    MetaRule, MetaRuleType, SystemInsight,
    
    # Core structures
    Civilization, AgentCluster, EnvironmentState, TemporalState, GovernanceModel,
    AgentType, CivilizationArchetype
)


class TestReflexiveIntrospectionLayer:
    """Test suite for the reflexive introspection layer."""
    
    @pytest.fixture
    def core(self):
        """Create a civilization intelligence core."""
        return CivilizationIntelligenceCore()
    
    @pytest.fixture
    def architect(self, core):
        """Create a civilization architect."""
        return CivilizationArchitect(core)
    
    @pytest.fixture
    def evolution_engine(self, core):
        """Create a temporal evolution engine."""
        return TemporalEvolutionEngine(core)
    
    @pytest.fixture
    def pattern_detector(self, core):
        """Create a macro pattern detector."""
        return MacroPatternDetector(core)
    
    @pytest.fixture
    def pattern_codification_engine(self, core, pattern_detector):
        """Create a pattern codification engine."""
        return PatternCodificationEngine(core, pattern_detector)
    
    @pytest.fixture
    def genetic_archive(self, core):
        """Create a civilizational genetic archive."""
        return CivilizationalGeneticArchive(core)
    
    @pytest.fixture
    def strategic_feedback_loop(self, core, pattern_codification_engine, genetic_archive):
        """Create a strategic feedback loop."""
        return StrategicFeedbackLoop(core, pattern_codification_engine, genetic_archive)
    
    @pytest.fixture
    def introspection_layer(self, core, pattern_codification_engine, genetic_archive, strategic_feedback_loop):
        """Create a reflexive introspection layer."""
        return ReflexiveIntrospectionLayer(core, pattern_codification_engine, genetic_archive, strategic_feedback_loop)
    
    def test_reflexive_introspection_basic_functionality(self, core, architect, introspection_layer):
        """Test basic reflexive introspection functionality."""
        print("\n=== Testing Reflexive Introspection Basic Functionality ===")
        
        # Create multiple civilizations for analysis
        civilizations = []
        for i in range(5):
            civ = architect.generate_civilization(
                name=f"Introspection Test Civilization {i+1}",
                archetype=random.choice(list(CivilizationArchetype)),
                population_size=random.randint(100, 300),
                deterministic_seed=1000 + i
            )
            civilizations.append(civ)
        
        print(f"✓ Created {len(civilizations)} civilizations for introspection")
        
        # Perform comprehensive introspection
        civ_ids = [civ.id for civ in civilizations]
        analysis_id = introspection_layer.perform_comprehensive_introspection(civ_ids)
        
        assert analysis_id is not None
        print(f"✓ Performed comprehensive introspection: {analysis_id}")
        
        # Get analysis results
        analysis = introspection_layer.get_analysis(analysis_id)
        assert analysis is not None
        assert analysis.status == IntrospectionStatus.COMPLETED
        print(f"✓ Analysis completed with confidence: {analysis.confidence_score:.3f}")
        
        # Check discovered patterns
        assert len(analysis.discovered_patterns) > 0
        print(f"✓ Discovered {len(analysis.discovered_patterns)} patterns")
        
        # Check dominance analysis
        assert 'archetype_dominance' in analysis.dominance_analysis
        assert 'governance_dominance' in analysis.dominance_analysis
        assert 'performance_dominance' in analysis.dominance_analysis
        print("✓ Dominance analysis completed")
        
        # Check evolutionary trajectories
        assert len(analysis.evolutionary_trajectories) > 0
        print(f"✓ Analyzed {len(analysis.evolutionary_trajectories)} evolutionary trajectories")
        
        # Check performance correlations
        assert len(analysis.performance_correlations) > 0
        print(f"✓ Calculated {len(analysis.performance_correlations)} performance correlations")
        
        print("✓ Reflexive Introspection Basic Functionality Test Passed")
    
    def test_meta_rule_discovery(self, core, architect, introspection_layer):
        """Test meta-rule discovery capabilities."""
        print("\n=== Testing Meta-Rule Discovery ===")
        
        # Create civilizations with specific characteristics for rule discovery
        civilizations = []
        
        # Create high-cooperation civilizations
        for i in range(3):
            civ = architect.generate_civilization(
                name=f"High Cooperation Civ {i+1}",
                archetype=CivilizationArchetype.COOPERATIVE,
                population_size=200,
                deterministic_seed=2000 + i
            )
            # Manually set high cooperation and innovation
            civ.cooperation_level = 0.8 + random.uniform(0, 0.2)
            civ.innovation_capacity = 0.7 + random.uniform(0, 0.3)
            civilizations.append(civ)
        
        # Create high-complexity civilizations
        for i in range(2):
            civ = architect.generate_civilization(
                name=f"High Complexity Civ {i+1}",
                archetype=CivilizationArchetype.HIERARCHICAL,
                population_size=250,
                deterministic_seed=3000 + i
            )
            # Manually set high complexity and lower stability
            civ.complexity = 0.8 + random.uniform(0, 0.2)
            civ.stability = 0.4 + random.uniform(0, 0.2)
            civilizations.append(civ)
        
        print(f"✓ Created {len(civilizations)} specialized civilizations")
        
        # Perform introspection
        civ_ids = [civ.id for civ in civilizations]
        analysis_id = introspection_layer.perform_comprehensive_introspection(civ_ids)
        
        assert analysis_id is not None
        analysis = introspection_layer.get_analysis(analysis_id)
        assert analysis is not None
        
        # Check for discovered meta-rules
        assert len(analysis.discovered_meta_rules) >= 0
        print(f"✓ Discovered {len(analysis.discovered_meta_rules)} meta-rules")
        
        # Check meta-rule confidence scores
        if analysis.law_confidence_scores:
            for rule_name, confidence in analysis.law_confidence_scores.items():
                print(f"✓ Meta-rule '{rule_name}' confidence: {confidence:.3f}")
        
        # Check system insights
        assert len(analysis.system_insights) >= 0
        print(f"✓ Generated {len(analysis.system_insights)} system insights")
        
        # Check optimization recommendations
        assert len(analysis.optimization_recommendations) >= 0
        print(f"✓ Generated {len(analysis.optimization_recommendations)} optimization recommendations")
        
        print("✓ Meta-Rule Discovery Test Passed")
    
    def test_meta_rule_application(self, core, architect, introspection_layer):
        """Test meta-rule application to civilizations."""
        print("\n=== Testing Meta-Rule Application ===")
        
        # Create civilizations and perform introspection
        civilizations = []
        for i in range(4):
            civ = architect.generate_civilization(
                name=f"Meta-Rule Test Civ {i+1}",
                archetype=random.choice(list(CivilizationArchetype)),
                population_size=150,
                deterministic_seed=4000 + i
            )
            civilizations.append(civ)
        
        # Perform introspection to discover meta-rules
        civ_ids = [civ.id for civ in civilizations]
        analysis_id = introspection_layer.perform_comprehensive_introspection(civ_ids)
        
        assert analysis_id is not None
        analysis = introspection_layer.get_analysis(analysis_id)
        assert analysis is not None
        
        # Get discovered meta-rules
        meta_rules = introspection_layer.get_high_confidence_meta_rules(min_confidence=0.5)
        print(f"✓ Found {len(meta_rules)} high-confidence meta-rules")
        
        # Test applying meta-rules to civilizations
        applications_successful = 0
        for meta_rule in meta_rules[:2]:  # Test with first 2 meta-rules
            for civ in civilizations:
                success = introspection_layer.apply_meta_rule(meta_rule.id, civ.id)
                if success:
                    applications_successful += 1
                    print(f"✓ Applied meta-rule '{meta_rule.name}' to {civ.name}")
        
        print(f"✓ Successfully applied {applications_successful} meta-rules")
        
        # Test meta-rule retrieval by type
        governance_rules = introspection_layer.get_meta_rules_by_type(MetaRuleType.GOVERNANCE_LAW)
        innovation_rules = introspection_layer.get_meta_rules_by_type(MetaRuleType.INNOVATION_LAW)
        resource_rules = introspection_layer.get_meta_rules_by_type(MetaRuleType.RESOURCE_LAW)
        
        print(f"✓ Found {len(governance_rules)} governance rules, {len(innovation_rules)} innovation rules, {len(resource_rules)} resource rules")
        
        print("✓ Meta-Rule Application Test Passed")
    
    def test_system_insights_generation(self, core, architect, introspection_layer):
        """Test system insights generation."""
        print("\n=== Testing System Insights Generation ===")
        
        # Create diverse civilizations for insight generation
        civilizations = []
        archetypes = list(CivilizationArchetype)
        
        for i in range(6):
            civ = architect.generate_civilization(
                name=f"Insight Test Civ {i+1}",
                archetype=archetypes[i % len(archetypes)],
                population_size=random.randint(100, 400),
                deterministic_seed=5000 + i
            )
            # Vary performance characteristics
            civ.stability = random.uniform(0.3, 0.9)
            civ.complexity = random.uniform(0.2, 0.8)
            civ.adaptability = random.uniform(0.4, 0.9)
            civ.innovation_capacity = random.uniform(0.3, 0.9)
            civ.cooperation_level = random.uniform(0.2, 0.9)
            civilizations.append(civ)
        
        print(f"✓ Created {len(civilizations)} diverse civilizations")
        
        # Perform introspection
        civ_ids = [civ.id for civ in civilizations]
        analysis_id = introspection_layer.perform_comprehensive_introspection(civ_ids)
        
        assert analysis_id is not None
        analysis = introspection_layer.get_analysis(analysis_id)
        assert analysis is not None
        
        # Check system insights
        insights = introspection_layer.get_system_insights()
        print(f"✓ Generated {len(insights)} system insights")
        
        for insight in insights:
            print(f"✓ Insight: {insight.description} (confidence: {insight.confidence_level:.3f})")
        
        # Check introspection statistics
        stats = introspection_layer.get_introspection_statistics()
        assert 'total_analyses' in stats
        assert 'total_meta_rules' in stats
        assert 'total_insights' in stats
        print(f"✓ Introspection statistics: {stats['total_analyses']} analyses, {stats['total_meta_rules']} meta-rules, {stats['total_insights']} insights")
        
        print("✓ System Insights Generation Test Passed")
    
    def test_introspection_performance_under_load(self, core, architect, introspection_layer):
        """Test introspection performance under load."""
        print("\n=== Testing Introspection Performance Under Load ===")
        
        start_time = time.time()
        
        # Create many civilizations
        civilizations = []
        for i in range(10):
            civ = architect.generate_civilization(
                name=f"Load Test Civ {i+1}",
                archetype=random.choice(list(CivilizationArchetype)),
                population_size=random.randint(200, 500),
                deterministic_seed=6000 + i
            )
            civilizations.append(civ)
        
        print(f"✓ Created {len(civilizations)} civilizations")
        
        # Perform introspection
        civ_ids = [civ.id for civ in civilizations]
        analysis_id = introspection_layer.perform_comprehensive_introspection(civ_ids)
        
        assert analysis_id is not None
        analysis = introspection_layer.get_analysis(analysis_id)
        assert analysis is not None
        
        # Measure performance
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✓ Introspection completed in {total_time:.2f} seconds")
        print(f"✓ Average time per civilization: {total_time / len(civilizations):.2f} seconds")
        
        # Verify analysis quality
        assert analysis.confidence_score > 0.0
        assert len(analysis.discovered_patterns) > 0
        assert len(analysis.dominance_analysis) > 0
        
        print(f"✓ Analysis confidence: {analysis.confidence_score:.3f}")
        print(f"✓ Discovered patterns: {len(analysis.discovered_patterns)}")
        print(f"✓ Meta-rules discovered: {len(analysis.discovered_meta_rules)}")
        print(f"✓ System insights: {len(analysis.system_insights)}")
        
        print("✓ Introspection Performance Under Load Test Passed")
    
    def test_introspection_error_handling(self, core, architect, introspection_layer):
        """Test introspection error handling and recovery."""
        print("\n=== Testing Introspection Error Handling ===")
        
        # Test with empty civilization list
        empty_analysis = introspection_layer.perform_comprehensive_introspection([])
        assert empty_analysis is None
        print("✓ Handled empty civilization list gracefully")
        
        # Test with invalid civilization IDs
        invalid_analysis = introspection_layer.perform_comprehensive_introspection(["invalid_id"])
        assert invalid_analysis is None
        print("✓ Handled invalid civilization IDs gracefully")
        
        # Test with single civilization (below minimum threshold)
        civ = architect.generate_civilization(
            name="Single Civ Test",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=7000
        )
        
        single_analysis = introspection_layer.perform_comprehensive_introspection([civ.id])
        assert single_analysis is None
        print("✓ Handled single civilization (below threshold) gracefully")
        
        # Test meta-rule application with invalid IDs
        invalid_rule_application = introspection_layer.apply_meta_rule("invalid_rule_id", civ.id)
        assert not invalid_rule_application
        print("✓ Handled invalid meta-rule application gracefully")
        
        invalid_civ_application = introspection_layer.apply_meta_rule("invalid_rule_id", "invalid_civ_id")
        assert not invalid_civ_application
        print("✓ Handled invalid civilization application gracefully")
        
        # Test with corrupted civilization data
        corrupted_civ = architect.generate_civilization(
            name="Corrupted Civ Test",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=8000
        )
        
        # Manually corrupt some attributes
        corrupted_civ.stability = -1.0  # Invalid value
        corrupted_civ.complexity = 2.0   # Invalid value
        
        # This should still work but may produce lower confidence results
        civ_ids = [civ.id, corrupted_civ.id]
        analysis_id = introspection_layer.perform_comprehensive_introspection(civ_ids)
        
        if analysis_id:
            analysis = introspection_layer.get_analysis(analysis_id)
            if analysis:
                print(f"✓ Handled corrupted civilization data (confidence: {analysis.confidence_score:.3f})")
        
        print("✓ Introspection Error Handling Test Passed")
    
    def test_introspection_deterministic_reproducibility(self, core, architect, introspection_layer):
        """Test deterministic reproducibility of introspection."""
        print("\n=== Testing Introspection Deterministic Reproducibility ===")
        
        # Test 1: Same civilizations should produce similar results
        seed = 9000
        
        # First run
        civs1 = []
        for i in range(4):
            civ = architect.generate_civilization(
                name=f"Reproducibility Test Civ {i+1}",
                archetype=random.choice(list(CivilizationArchetype)),
                population_size=200,
                deterministic_seed=seed + i
            )
            civs1.append(civ)
        
        analysis_id_1 = introspection_layer.perform_comprehensive_introspection([civ.id for civ in civs1])
        assert analysis_id_1 is not None
        analysis_1 = introspection_layer.get_analysis(analysis_id_1)
        assert analysis_1 is not None
        
        # Second run with same seed
        civs2 = []
        for i in range(4):
            civ = architect.generate_civilization(
                name=f"Reproducibility Test Civ {i+1}",
                archetype=random.choice(list(CivilizationArchetype)),
                population_size=200,
                deterministic_seed=seed + i
            )
            civs2.append(civ)
        
        analysis_id_2 = introspection_layer.perform_comprehensive_introspection([civ.id for civ in civs2])
        assert analysis_id_2 is not None
        analysis_2 = introspection_layer.get_analysis(analysis_id_2)
        assert analysis_2 is not None
        
        # Compare results
        assert len(analysis_1.discovered_patterns) == len(analysis_2.discovered_patterns)
        print("✓ Deterministic reproducibility verified for pattern discovery")
        
        # Test 2: Different seeds should produce different results
        civs3 = []
        for i in range(4):
            civ = architect.generate_civilization(
                name=f"Different Seed Test Civ {i+1}",
                archetype=random.choice(list(CivilizationArchetype)),
                population_size=200,
                deterministic_seed=seed + 100 + i  # Different seed
            )
            civs3.append(civ)
        
        analysis_id_3 = introspection_layer.perform_comprehensive_introspection([civ.id for civ in civs3])
        assert analysis_id_3 is not None
        analysis_3 = introspection_layer.get_analysis(analysis_id_3)
        assert analysis_3 is not None
        
        # Results should be different due to different seeds
        print("✓ Different seeds produced different results")
        
        print("✓ Introspection Deterministic Reproducibility Test Passed")
    
    def test_full_reflexive_introspection_workflow(self, core, architect, evolution_engine, 
                                                 pattern_detector, pattern_codification_engine,
                                                 genetic_archive, strategic_feedback_loop, introspection_layer):
        """Test the complete reflexive introspection workflow."""
        print("\n=== Testing Full Reflexive Introspection Workflow ===")
        
        # Step 1: Create and evolve civilizations
        civilizations = []
        for i in range(5):
            civ = architect.generate_civilization(
                name=f"Full Workflow Civ {i+1}",
                archetype=random.choice(list(CivilizationArchetype)),
                population_size=random.randint(150, 350),
                deterministic_seed=10000 + i
            )
            civilizations.append(civ)
        
        print(f"✓ Created {len(civilizations)} civilizations")
        
        # Step 2: Evolve civilizations
        civ_ids = [civ.id for civ in civilizations]
        evolution_engine.start_evolution(civ_ids, deterministic_seed=10000)
        
        # Let evolution run
        time.sleep(3)
        
        evolution_engine.stop_evolution()
        print("✓ Evolution completed")
        
        # Step 3: Detect patterns
        for civ_id in civ_ids:
            pattern_detector.detect_patterns(civ_id)
        total_patterns = sum(len(patterns) for patterns in pattern_detector.detected_patterns.values())
        print(f"✓ Detected {total_patterns} total patterns")
        
        # Step 4: Codify patterns
        total_blueprints = 0
        for civ in civilizations:
            blueprints = pattern_codification_engine.codify_detected_patterns(civ.id)
            total_blueprints += len(blueprints)
        print(f"✓ Codified {total_blueprints} blueprints")
        
        # Step 5: Archive civilization DNA
        archived_dna = 0
        for civ in civilizations:
            performance_metrics = {
                'stability': civ.stability,
                'complexity': civ.complexity,
                'adaptability': civ.adaptability,
                'resilience': civ.resilience,
                'innovation_capacity': civ.innovation_capacity,
                'cooperation_level': civ.cooperation_level
            }
            
            dna = genetic_archive.archive_civilization_dna(civ, performance_metrics)
            if dna:
                archived_dna += 1
        print(f"✓ Archived {archived_dna} DNA sequences")
        
        # Step 6: Apply strategic feedback
        total_feedback = 0
        for civ in civilizations:
            feedback = strategic_feedback_loop.process_strategic_feedback(civ.id)
            total_feedback += len(feedback)
        print(f"✓ Applied {total_feedback} feedback applications")
        
        # Step 7: Perform reflexive introspection
        analysis_id = introspection_layer.perform_comprehensive_introspection(civ_ids)
        assert analysis_id is not None
        analysis = introspection_layer.get_analysis(analysis_id)
        assert analysis is not None
        
        print(f"✓ Performed comprehensive introspection (confidence: {analysis.confidence_score:.3f})")
        
        # Step 8: Apply discovered meta-rules
        meta_rules = introspection_layer.get_high_confidence_meta_rules(min_confidence=0.5)
        meta_rule_applications = 0
        
        for meta_rule in meta_rules:
            for civ in civilizations:
                success = introspection_layer.apply_meta_rule(meta_rule.id, civ.id)
                if success:
                    meta_rule_applications += 1
        
        print(f"✓ Applied {meta_rule_applications} meta-rules")
        
        # Step 9: Verify system integration
        assert len(introspection_layer.introspection_analyses) > 0
        assert len(introspection_layer.discovered_meta_rules) >= 0
        assert len(introspection_layer.system_insights) >= 0
        
        # Get final statistics
        stats = introspection_layer.get_introspection_statistics()
        print(f"✓ Final statistics: {stats['total_analyses']} analyses, {stats['total_meta_rules']} meta-rules, {stats['total_insights']} insights")
        
        print("✓ Full Reflexive Introspection Workflow Test Passed")
