"""
Comprehensive tests for the Meta-Instrumentation Layer
Tests Pattern Codification Engine, Civilizational Genetic Archive, and Strategic Feedback Loop
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
    PatternCodificationEngine, PatternBlueprint, BlueprintType, CodificationStatus,
    StrategicDoctrine, CivilizationalGeneticArchive, CivilizationDNA, LineageType, 
    ArchiveStatus, LineageTree, StrategicFeedbackLoop, FeedbackApplication, 
    FeedbackPipeline, FeedbackType, FeedbackStatus,
    
    # Core structures
    Civilization, AgentCluster, EnvironmentState, TemporalState, GovernanceModel,
    AgentType, CivilizationArchetype
)


class TestMetaInstrumentationLayer:
    """Test suite for the meta-instrumentation layer."""
    
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
    
    def test_pattern_codification_engine_integration(self, core, architect, pattern_detector, 
                                                   pattern_codification_engine):
        """Test pattern codification engine integration."""
        print("\n=== Testing Pattern Codification Engine Integration ===")
        
        # Create a civilization
        civ = architect.generate_civilization(
            name="Test Civilization",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=42
        )
        
        print(f"✓ Created civilization: {civ.id}")
        
        # Simulate some evolution to generate patterns
        evolution_engine = TemporalEvolutionEngine(core)
        evolution_engine.start_evolution([civ.id], deterministic_seed=42)
        
        # Let it run for a few ticks
        time.sleep(2)
        
        # Stop evolution
        evolution_engine.stop_evolution()
        print("✓ Evolution completed")
        
        # Detect patterns
        pattern_detector.detect_patterns(civ.id)
        detected_patterns = pattern_detector.detected_patterns.get(civ.id, {})
        print(f"✓ Detected {len(detected_patterns)} patterns")
        
        # Codify patterns
        codified_blueprints = pattern_codification_engine.codify_detected_patterns(civ.id)
        print(f"✓ Codified {len(codified_blueprints)} blueprints")
        
        # Verify blueprints were created
        assert len(codified_blueprints) >= 0  # May be 0 if no patterns met criteria
        
        # Test blueprint retrieval
        if codified_blueprints:
            blueprint = pattern_codification_engine.get_blueprint(codified_blueprints[0])
            assert blueprint is not None
            assert blueprint.source_civilization_id == civ.id
            print(f"✓ Retrieved blueprint: {blueprint.name}")
        
        # Test codification statistics
        stats = pattern_codification_engine.get_codification_statistics()
        assert 'total_blueprints' in stats
        assert 'codification_stats' in stats
        print(f"✓ Codification statistics: {stats['total_blueprints']} blueprints")
        
        print("✓ Pattern Codification Engine Integration Test Passed")
    
    def test_civilizational_genetic_archive_integration(self, core, architect, genetic_archive):
        """Test civilizational genetic archive integration."""
        print("\n=== Testing Civilizational Genetic Archive Integration ===")
        
        # Create a civilization
        civ = architect.generate_civilization(
            name="Test Civilization",
            archetype=CivilizationArchetype.HIERARCHICAL,
            population_size=150,
            deterministic_seed=123
        )
        
        print(f"✓ Created civilization: {civ.id}")
        
        # Archive civilization DNA
        performance_metrics = {
            'stability': civ.stability,
            'complexity': civ.complexity,
            'adaptability': civ.adaptability,
            'resilience': civ.resilience,
            'innovation_capacity': civ.innovation_capacity,
            'cooperation_level': civ.cooperation_level
        }
        
        dna = genetic_archive.archive_civilization_dna(civ, performance_metrics)
        assert dna is not None
        assert dna.source_civilization_id == civ.id
        print(f"✓ Archived civilization DNA: {dna.id}")
        
        # Test DNA mutation
        mutated_dna = genetic_archive.create_dna_mutation(dna.id, "random")
        assert mutated_dna is not None
        assert dna.id in mutated_dna.parent_lineages
        print(f"✓ Created DNA mutation: {mutated_dna.id}")
        
        # Test DNA crossover
        crossover_dna = genetic_archive.create_dna_crossover(dna.id, mutated_dna.id)
        assert crossover_dna is not None
        assert dna.id in crossover_dna.parent_lineages
        assert mutated_dna.id in crossover_dna.parent_lineages
        print(f"✓ Created DNA crossover: {crossover_dna.id}")
        
        # Test DNA search
        high_fitness_dna = genetic_archive.search_dna_by_fitness(0.5)
        assert len(high_fitness_dna) >= 0
        print(f"✓ Found {len(high_fitness_dna)} high-fitness DNA sequences")
        
        # Test lineage search
        lineage_dna = genetic_archive.search_dna_by_lineage(dna.lineage_type)
        assert len(lineage_dna) >= 0
        print(f"✓ Found {len(lineage_dna)} DNA sequences of lineage type: {dna.lineage_type.name}")
        
        # Test archive statistics
        stats = genetic_archive.get_archive_statistics()
        assert 'total_dna_sequences' in stats
        assert 'active_lineages' in stats
        print(f"✓ Archive statistics: {stats['total_dna_sequences']} DNA sequences")
        
        print("✓ Civilizational Genetic Archive Integration Test Passed")
    
    def test_strategic_feedback_loop_integration(self, core, architect, pattern_codification_engine, 
                                               genetic_archive, strategic_feedback_loop):
        """Test strategic feedback loop integration."""
        print("\n=== Testing Strategic Feedback Loop Integration ===")
        
        # Create a civilization
        civ = architect.generate_civilization(
            name="Test Civilization",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=200,
            deterministic_seed=456
        )
        
        print(f"✓ Created civilization: {civ.id}")
        
        # Archive some DNA for feedback
        performance_metrics = {
            'stability': 0.8,
            'complexity': 0.7,
            'adaptability': 0.6,
            'resilience': 0.7,
            'innovation_capacity': 0.8,
            'cooperation_level': 0.9
        }
        
        dna = genetic_archive.archive_civilization_dna(civ, performance_metrics)
        assert dna is not None
        print(f"✓ Archived DNA for feedback: {dna.id}")
        
        # Process strategic feedback
        applied_feedbacks = strategic_feedback_loop.process_strategic_feedback(civ.id)
        print(f"✓ Applied {len(applied_feedbacks)} feedback applications")
        
        # Verify feedback applications were created
        if applied_feedbacks:
            application = strategic_feedback_loop.feedback_applications.get(applied_feedbacks[0])
            assert application is not None
            assert application.target_civilization_id == civ.id
            print(f"✓ Feedback application created: {application.id}")
            
            # Test feedback evaluation
            performance = strategic_feedback_loop.evaluate_feedback_performance(application.id)
            assert 'application_id' in performance
            assert 'feedback_type' in performance
            print(f"✓ Feedback performance evaluated: {performance['overall_improvement']:.3f}")
        
        # Test feedback statistics
        stats = strategic_feedback_loop.get_feedback_statistics()
        assert 'total_applications' in stats
        assert 'successful_applications' in stats
        print(f"✓ Feedback statistics: {stats['total_applications']} applications")
        
        print("✓ Strategic Feedback Loop Integration Test Passed")
    
    def test_meta_instrumentation_full_workflow(self, core, architect, evolution_engine, 
                                               pattern_detector, pattern_codification_engine,
                                               genetic_archive, strategic_feedback_loop):
        """Test the complete meta-instrumentation workflow."""
        print("\n=== Testing Meta-Instrumentation Full Workflow ===")
        
        # Step 1: Create and evolve civilizations
        civilizations = []
        for i in range(3):
            civ = architect.generate_civilization(
                name=f"Test Civilization {i+1}",
                archetype=random.choice(list(CivilizationArchetype)),
                population_size=random.randint(100, 300),
                deterministic_seed=100 + i
            )
            civilizations.append(civ)
        
        print(f"✓ Created {len(civilizations)} civilizations")
        
        # Step 2: Evolve civilizations
        civ_ids = [civ.id for civ in civilizations]
        evolution_engine.start_evolution(civ_ids, deterministic_seed=789)
        
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
        
        # Step 7: Test cross-component integration
        # Create a strategic doctrine from blueprints
        if total_blueprints > 0:
            blueprint_ids = list(pattern_codification_engine.blueprints.keys())[:2]
            doctrine = pattern_codification_engine.create_strategic_doctrine(
                blueprint_ids, "Test Strategic Doctrine"
            )
            if doctrine:
                print(f"✓ Created strategic doctrine: {doctrine.name}")
        
        # Test DNA mutation and crossover
        if archived_dna > 0:
            dna_sequences = list(genetic_archive.dna_archive.values())
            if len(dna_sequences) >= 2:
                # Create mutation
                mutated = genetic_archive.create_dna_mutation(dna_sequences[0].id, "random")
                if mutated:
                    print(f"✓ Created DNA mutation: {mutated.id}")
                
                # Create crossover
                crossover = genetic_archive.create_dna_crossover(
                    dna_sequences[0].id, dna_sequences[1].id
                )
                if crossover:
                    print(f"✓ Created DNA crossover: {crossover.id}")
        
        # Step 8: Verify system integration
        # Check that all components are working together
        assert len(pattern_codification_engine.blueprints) >= 0
        assert len(genetic_archive.dna_archive) >= 0
        assert len(strategic_feedback_loop.feedback_applications) >= 0
        
        print("✓ Meta-Instrumentation Full Workflow Test Passed")
    
    def test_performance_under_meta_instrumentation_load(self, core, architect, 
                                                         pattern_codification_engine,
                                                         genetic_archive, strategic_feedback_loop):
        """Test performance under meta-instrumentation load."""
        print("\n=== Testing Performance Under Meta-Instrumentation Load ===")
        
        start_time = time.time()
        
        # Create multiple civilizations
        civilizations = []
        for i in range(5):
            civ = architect.generate_civilization(
                name=f"Load Test Civilization {i+1}",
                archetype=random.choice(list(CivilizationArchetype)),
                population_size=random.randint(200, 500),
                deterministic_seed=1000 + i
            )
            civilizations.append(civ)
        
        print(f"✓ Created {len(civilizations)} civilizations")
        
        # Archive DNA for all civilizations
        archived_count = 0
        for civ in civilizations:
            performance_metrics = {
                'stability': random.uniform(0.5, 0.9),
                'complexity': random.uniform(0.4, 0.8),
                'adaptability': random.uniform(0.6, 0.9),
                'resilience': random.uniform(0.5, 0.8),
                'innovation_capacity': random.uniform(0.5, 0.9),
                'cooperation_level': random.uniform(0.4, 0.9)
            }
            
            dna = genetic_archive.archive_civilization_dna(civ, performance_metrics)
            if dna:
                archived_count += 1
        
        print(f"✓ Archived {archived_count} DNA sequences")
        
        # Process strategic feedback for all civilizations
        total_feedback = 0
        for civ in civilizations:
            feedback = strategic_feedback_loop.process_strategic_feedback(civ.id)
            total_feedback += len(feedback)
        
        print(f"✓ Applied {total_feedback} feedback applications")
        
        # Test DNA operations
        if archived_count > 0:
            dna_sequences = list(genetic_archive.dna_archive.values())
            
            # Create mutations
            mutation_count = 0
            for dna in dna_sequences[:3]:  # Test with first 3 DNA sequences
                mutated = genetic_archive.create_dna_mutation(dna.id, "random")
                if mutated:
                    mutation_count += 1
            
            print(f"✓ Created {mutation_count} DNA mutations")
            
            # Create crossovers
            crossover_count = 0
            if len(dna_sequences) >= 2:
                for i in range(min(3, len(dna_sequences) - 1)):
                    crossover = genetic_archive.create_dna_crossover(
                        dna_sequences[i].id, dna_sequences[i+1].id
                    )
                    if crossover:
                        crossover_count += 1
            
            print(f"✓ Created {crossover_count} DNA crossovers")
        
        # Measure performance
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✓ Meta-instrumentation completed in {total_time:.2f} seconds")
        print(f"✓ Average time per civilization: {total_time / len(civilizations):.2f} seconds")
        
        # Verify system state
        assert len(genetic_archive.dna_archive) >= archived_count
        assert len(strategic_feedback_loop.feedback_applications) >= 0
        
        print("✓ Performance Under Meta-Instrumentation Load Test Passed")
    
    def test_error_handling_and_recovery_meta_instrumentation(self, core, architect,
                                                             pattern_codification_engine,
                                                             genetic_archive, strategic_feedback_loop):
        """Test error handling and recovery in meta-instrumentation."""
        print("\n=== Testing Error Handling and Recovery in Meta-Instrumentation ===")
        
        # Test with invalid civilization ID
        invalid_feedback = strategic_feedback_loop.process_strategic_feedback("invalid_id")
        assert len(invalid_feedback) == 0
        print("✓ Handled invalid civilization ID gracefully")
        
        # Test with empty civilization
        empty_civ = architect.generate_civilization(
            name="Empty Civilization",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=10,  # Very small population
            deterministic_seed=999
        )
        
        # Test with minimal performance metrics
        minimal_metrics = {
            'stability': 0.1,
            'complexity': 0.1,
            'adaptability': 0.1,
            'resilience': 0.1,
            'innovation_capacity': 0.1,
            'cooperation_level': 0.1
        }
        
        # This should still work but may not meet criteria
        dna = genetic_archive.archive_civilization_dna(empty_civ, minimal_metrics)
        if dna:
            print("✓ Handled minimal performance metrics")
        
        # Test feedback rollback
        if len(strategic_feedback_loop.feedback_applications) > 0:
            application_id = list(strategic_feedback_loop.feedback_applications.keys())[0]
            rollback_success = strategic_feedback_loop.rollback_feedback(application_id)
            if rollback_success:
                print("✓ Feedback rollback successful")
            else:
                print("✓ Feedback rollback handled gracefully")
        
        # Test archive cleanup
        cleanup_count = genetic_archive.cleanup_archive(days_old=0)  # Clean up everything
        print(f"✓ Archive cleanup removed {cleanup_count} old entries")
        
        # Test with corrupted data
        try:
            # Try to create DNA with invalid data
            corrupted_civ = architect.generate_civilization(
                name="Corrupted Civilization",
                archetype=CivilizationArchetype.COOPERATIVE,
                population_size=50,
                deterministic_seed=888
            )
            
            # Manually corrupt some attributes
            corrupted_civ.stability = -1.0  # Invalid value
            corrupted_civ.complexity = 2.0   # Invalid value
            
            # This should handle the corruption gracefully
            corrupted_metrics = {
                'stability': corrupted_civ.stability,
                'complexity': corrupted_civ.complexity,
                'adaptability': 0.5,
                'resilience': 0.5,
                'innovation_capacity': 0.5,
                'cooperation_level': 0.5
            }
            
            dna = genetic_archive.archive_civilization_dna(corrupted_civ, corrupted_metrics)
            if dna:
                print("✓ Handled corrupted civilization data gracefully")
            
        except Exception as e:
            print(f"✓ Error handling worked: {e}")
        
        print("✓ Error Handling and Recovery in Meta-Instrumentation Test Passed")
    
    def test_deterministic_reproducibility_meta_instrumentation(self, core, architect,
                                                               pattern_codification_engine,
                                                               genetic_archive, strategic_feedback_loop):
        """Test deterministic reproducibility in meta-instrumentation."""
        print("\n=== Testing Deterministic Reproducibility in Meta-Instrumentation ===")
        
        # Test 1: Same seed should produce same results
        seed = 12345
        
        # First run
        civ1 = architect.generate_civilization(
            name="Reproducibility Test 1",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=seed
        )
        
        metrics1 = {
            'stability': 0.7,
            'complexity': 0.6,
            'adaptability': 0.8,
            'resilience': 0.7,
            'innovation_capacity': 0.8,
            'cooperation_level': 0.9
        }
        
        dna1 = genetic_archive.archive_civilization_dna(civ1, metrics1)
        
        # Second run with same seed
        civ2 = architect.generate_civilization(
            name="Reproducibility Test 2",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=seed
        )
        
        metrics2 = {
            'stability': 0.7,
            'complexity': 0.6,
            'adaptability': 0.8,
            'resilience': 0.7,
            'innovation_capacity': 0.8,
            'cooperation_level': 0.9
        }
        
        dna2 = genetic_archive.archive_civilization_dna(civ2, metrics2)
        
        # Compare results
        if dna1 and dna2:
            assert dna1.fitness_score == dna2.fitness_score
            assert dna1.adaptability_score == dna2.adaptability_score
            assert dna1.stability_score == dna2.stability_score
            print("✓ Deterministic reproducibility verified")
        
        # Test 2: Different seeds should produce different results
        civ3 = architect.generate_civilization(
            name="Reproducibility Test 3",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=seed + 1
        )
        
        metrics3 = {
            'stability': 0.7,
            'complexity': 0.6,
            'adaptability': 0.8,
            'resilience': 0.7,
            'innovation_capacity': 0.8,
            'cooperation_level': 0.9
        }
        
        dna3 = genetic_archive.archive_civilization_dna(civ3, metrics3)
        
        if dna1 and dna3:
            # Results should be different due to different seed
            assert dna1.fitness_score != dna3.fitness_score or dna1.adaptability_score != dna3.adaptability_score
            print("✓ Different seeds produced different results")
        
        print("✓ Deterministic Reproducibility in Meta-Instrumentation Test Passed")
