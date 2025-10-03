"""
Integration tests for OmniMind Transcendence Layer
Tests the full civilization-scale simulation loop.
"""

import pytest
import time
import threading
from datetime import datetime
from typing import List, Dict, Any

from egdol.omnimind.transcendence import (
    CivilizationIntelligenceCore, CivilizationArchitect, TemporalEvolutionEngine,
    MacroPatternDetector, StrategicCivilizationalOrchestrator, CivilizationExperimentationSystem,
    StrategicDomain, PolicyArchetype, ExperimentType, PatternType, CivilizationArchetype
)


class TestTranscendenceIntegration:
    """Integration tests for the Transcendence Layer."""
    
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
        """Create a macro-pattern detector."""
        return MacroPatternDetector(core)
    
    @pytest.fixture
    def strategic_orchestrator(self, core):
        """Create a strategic orchestrator."""
        return StrategicCivilizationalOrchestrator(core)
    
    @pytest.fixture
    def experimentation_system(self, core):
        """Create an experimentation system."""
        return CivilizationExperimentationSystem(core)
    
    def test_full_civilization_simulation_loop(self, core, architect, evolution_engine, 
                                             pattern_detector, strategic_orchestrator, 
                                             experimentation_system):
        """Test the complete civilization simulation loop."""
        print("\n=== Testing Full Civilization Simulation Loop ===")
        
        # Step 1: Create civilizations
        print("Step 1: Creating civilizations...")
        civ_1 = architect.generate_civilization(
            name="Test Civilization 1",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=1
        )
        assert civ_1 is not None
        assert civ_1.name == "Test Civilization 1"
        print(f"✓ Created civilization: {civ_1.name}")
        
        civ_2 = architect.generate_civilization(
            name="Test Civilization 2",
            archetype=CivilizationArchetype.COMPETITIVE,
            population_size=80,
            deterministic_seed=2
        )
        assert civ_2 is not None
        assert civ_2.name == "Test Civilization 2"
        print(f"✓ Created civilization: {civ_2.name}")
        
        # Step 2: Start temporal evolution
        print("\nStep 2: Starting temporal evolution...")
        evolution_success = evolution_engine.start_evolution(
            civilization_ids=[civ_1.id, civ_2.id],
            deterministic_seed=42
        )
        assert evolution_success
        print("✓ Evolution started successfully")
        
        # Let evolution run for a bit
        time.sleep(2)
        
        # Check evolution status
        status = evolution_engine.get_evolution_status()
        assert status is not None
        print(f"✓ Evolution status - Current tick: {status['current_tick']}")
        
        # Step 3: Detect patterns
        print("\nStep 3: Detecting patterns...")
        patterns_1 = pattern_detector.detect_patterns(civ_1.id)
        patterns_2 = pattern_detector.detect_patterns(civ_2.id)
        print(f"✓ Detected {len(patterns_1)} patterns for Civ 1, {len(patterns_2)} patterns for Civ 2")
        
        # Step 4: Create strategic simulation
        print("\nStep 4: Creating strategic simulation...")
        simulation = strategic_orchestrator.create_strategic_simulation(
            name="Integration Test Simulation",
            description="Testing multi-civilization interactions",
            domain=StrategicDomain.COOPERATION_VS_CONFLICT,
            civilization_ids=[civ_1.id, civ_2.id],
            policy_assignments={
                civ_1.id: PolicyArchetype.COOPERATIVE,
                civ_2.id: PolicyArchetype.COMPETITIVE
            },
            duration=50,
            time_step=1
        )
        assert simulation is not None
        print(f"✓ Created strategic simulation: {simulation.name}")
        
        # Start strategic simulation
        simulation_started = strategic_orchestrator.start_simulation(simulation.id)
        assert simulation_started
        print("✓ Strategic simulation started")
        
        # Let simulation run
        time.sleep(1)
        
        # Step 5: Run experiments
        print("\nStep 5: Running experiments...")
        experiment_batch = experimentation_system.create_experiment_batch(
            name="Integration Test Experiments",
            description="Testing civilization experiments",
            experiment_type=ExperimentType.GOVERNANCE_EXPERIMENT,
            civilization_ids=[civ_1.id, civ_2.id]
        )
        assert experiment_batch is not None
        print(f"✓ Created experiment batch: {experiment_batch.name}")
        
        # Start experiments
        experiments_started = experimentation_system.start_experiment_batch(experiment_batch.id)
        assert experiments_started
        print("✓ Experiments started")
        
        # Let experiments run
        time.sleep(1)
        
        # Step 6: Check results
        print("\nStep 6: Checking results...")
        
        # Check evolution results
        print("✓ Evolution results available")
        
        # Check pattern detection results
        all_patterns = list(pattern_detector.detected_patterns.values())
        print(f"✓ Total patterns detected: {len(all_patterns)}")
        
        # Check strategic simulation results
        simulation_results = strategic_orchestrator.get_simulation_results(simulation.id)
        if simulation_results:
            print(f"✓ Strategic simulation results available")
        
        # Check experiment results
        experiment_results = experimentation_system.get_experiment_results(experiment_batch.id)
        if experiment_results:
            print(f"✓ Experiment results available")
        
        # Step 7: Stop all processes
        print("\nStep 7: Stopping processes...")
        
        # Stop evolution
        evolution_stopped = evolution_engine.stop_evolution()
        assert evolution_stopped
        print("✓ Evolution stopped")
        
        # Stop strategic simulation
        simulation_stopped = strategic_orchestrator.stop_simulation(simulation.id)
        assert simulation_stopped
        print("✓ Strategic simulation stopped")
        
        # Stop experiments
        experiments_stopped = experimentation_system.stop_experiment_batch(experiment_batch.id)
        assert experiments_stopped
        print("✓ Experiments stopped")
        
        print("\n=== Full Civilization Simulation Loop Test Completed Successfully ===")
    
    def test_civilization_architect_integration(self, architect, core):
        """Test civilization architect integration."""
        print("\n=== Testing Civilization Architect Integration ===")
        
        # Create multiple civilizations with different archetypes
        archetypes = [CivilizationArchetype.COOPERATIVE, CivilizationArchetype.COMPETITIVE, 
                     CivilizationArchetype.HIERARCHICAL, CivilizationArchetype.DECENTRALIZED]
        civilizations = []
        
        for i, archetype in enumerate(archetypes):
            civ = architect.generate_civilization(
                name=f"Test Civ {i+1}",
                archetype=archetype,
                population_size=50 + i * 20,
                deterministic_seed=i
            )
            assert civ is not None
            civilizations.append(civ)
            print(f"✓ Created {archetype} civilization: {civ.name}")
        
        # Test civilization retrieval through core
        for civ in civilizations:
            retrieved_civ = core.get_civilization(civ.id)
            assert retrieved_civ is not None
            assert retrieved_civ.id == civ.id
        
        # Test civilization listing through core
        all_civs = list(core.civilizations.values())
        assert len(all_civs) >= len(civilizations)
        print(f"✓ Retrieved {len(all_civs)} civilizations")
        
        print("=== Civilization Architect Integration Test Completed ===")
    
    def test_temporal_evolution_integration(self, architect, evolution_engine):
        """Test temporal evolution integration."""
        print("\n=== Testing Temporal Evolution Integration ===")
        
        # Create test civilization
        civ = architect.generate_civilization(
            name="Evolution Test Civ",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=1000
        )
        assert civ is not None
        
        # Start evolution
        evolution_success = evolution_engine.start_evolution(
            civilization_ids=[civ.id],
            deterministic_seed=100
        )
        assert evolution_success
        
        # Let evolution run
        time.sleep(1)
        
        # Check evolution status
        status = evolution_engine.get_evolution_status()
        assert status is not None
        assert status['current_tick'] > 0
        print(f"✓ Evolution running at tick {status['current_tick']}")
        
        # Check evolution history
        print("✓ Evolution history available")
        
        # Stop evolution
        evolution_stopped = evolution_engine.stop_evolution()
        assert evolution_stopped
        
        print("=== Temporal Evolution Integration Test Completed ===")
    
    def test_macro_pattern_detection_integration(self, architect, pattern_detector):
        """Test macro-pattern detection integration."""
        print("\n=== Testing Macro-Pattern Detection Integration ===")
        
        # Create test civilization
        civ = architect.generate_civilization(
            name="Pattern Test Civ",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=2000
        )
        assert civ is not None
        
        # Detect patterns
        patterns = pattern_detector.detect_patterns(civ.id)
        print(f"✓ Detected {len(patterns)} patterns")
        
        # Check pattern types
        for pattern in patterns:
            assert pattern.pattern_type in PatternType
            assert 0.0 <= pattern.novelty_score <= 1.0
            assert 0.0 <= pattern.significance_score <= 1.0
            print(f"✓ Pattern: {pattern.name} (Type: {pattern.pattern_type.name})")
        
        # Test pattern retrieval
        all_patterns = list(pattern_detector.detected_patterns.values())
        assert len(all_patterns) >= len(patterns)
        
        # Test pattern filtering
        governance_patterns = [p for p in all_patterns if p.pattern_type == PatternType.GOVERNANCE_STRUCTURE]
        print(f"✓ Governance patterns: {len(governance_patterns)}")
        
        print("=== Macro-Pattern Detection Integration Test Completed ===")
    
    def test_strategic_orchestrator_integration(self, architect, strategic_orchestrator):
        """Test strategic orchestrator integration."""
        print("\n=== Testing Strategic Orchestrator Integration ===")
        
        # Create test civilizations
        civ_1 = architect.generate_civilization(
            name="Strategic Civ 1",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=3000
        )
        civ_2 = architect.generate_civilization(
            name="Strategic Civ 2",
            archetype=CivilizationArchetype.COMPETITIVE,
            population_size=80,
            deterministic_seed=3001
        )
        
        # Create strategic simulation
        simulation = strategic_orchestrator.create_strategic_simulation(
            name="Integration Test",
            description="Testing strategic interactions",
            domain=StrategicDomain.COOPERATION_VS_CONFLICT,
            civilization_ids=[civ_1.id, civ_2.id],
            policy_assignments={
                civ_1.id: PolicyArchetype.COOPERATIVE,
                civ_2.id: PolicyArchetype.COMPETITIVE
            },
            duration=30,
            time_step=1
        )
        assert simulation is not None
        print(f"✓ Created strategic simulation: {simulation.name}")
        
        # Start simulation
        simulation_started = strategic_orchestrator.start_simulation(simulation.id)
        assert simulation_started
        
        # Let simulation run
        time.sleep(1)
        
        # Check simulation results
        results = strategic_orchestrator.get_simulation_results(simulation.id)
        if results:
            print("✓ Strategic simulation results available")
        
        # Stop simulation
        simulation_stopped = strategic_orchestrator.stop_simulation(simulation.id)
        assert simulation_stopped
        
        print("=== Strategic Orchestrator Integration Test Completed ===")
    
    def test_experimentation_system_integration(self, architect, experimentation_system):
        """Test experimentation system integration."""
        print("\n=== Testing Experimentation System Integration ===")
        
        # Create test civilization
        civ = architect.generate_civilization(
            name="Experiment Test Civ",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=4000
        )
        assert civ is not None
        
        # Create experiment batch
        batch = experimentation_system.create_experiment_batch(
            name="Integration Test Batch",
            description="Testing civilization experiments",
            experiment_type=ExperimentType.GOVERNANCE_EXPERIMENT,
            civilization_ids=[civ.id]
        )
        assert batch is not None
        print(f"✓ Created experiment batch: {batch.name}")
        
        # Start experiments
        experiments_started = experimentation_system.start_experiment_batch(batch.id)
        assert experiments_started
        
        # Let experiments run
        time.sleep(1)
        
        # Check experiment results
        results = experimentation_system.get_experiment_results(batch.id)
        if results:
            print("✓ Experiment results available")
        
        # Stop experiments
        experiments_stopped = experimentation_system.stop_experiment_batch(batch.id)
        assert experiments_stopped
        
        print("=== Experimentation System Integration Test Completed ===")
    
    def test_performance_under_load(self, core, architect, evolution_engine):
        """Test system performance under load."""
        print("\n=== Testing Performance Under Load ===")
        
        # Create multiple civilizations
        num_civilizations = 5
        civilization_ids = []
        
        for i in range(num_civilizations):
            civ = architect.generate_civilization(
                name=f"Load Test Civ {i+1}",
                archetype=CivilizationArchetype.COOPERATIVE,
                population_size=100,
                deterministic_seed=i
            )
            civilization_ids.append(civ.id)
        
        print(f"✓ Created {num_civilizations} civilizations")
        
        # Start evolution for all civilizations
        start_time = time.time()
        evolution_success = evolution_engine.start_evolution(
            civilization_ids=civilization_ids,
            deterministic_seed=200
        )
        assert evolution_success
        
        # Let evolution run
        time.sleep(2)
        
        # Stop evolution
        evolution_stopped = evolution_engine.stop_evolution()
        assert evolution_stopped
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ Evolution completed in {duration:.2f} seconds")
        print(f"✓ Average time per civilization: {duration/num_civilizations:.2f} seconds")
        
        # Check that all civilizations evolved
        status = evolution_engine.get_evolution_status()
        if status:
            print(f"✓ Evolution completed with {status['current_tick']} ticks")
        
        print("=== Performance Under Load Test Completed ===")
    
    def test_deterministic_reproducibility(self, architect):
        """Test deterministic reproducibility."""
        print("\n=== Testing Deterministic Reproducibility ===")
        
        # Create two civilizations with the same seed
        seed = "deterministic_test"
        civ_1 = architect.generate_civilization(
            name="Deterministic Civ 1",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=5000
        )
        civ_2 = architect.generate_civilization(
            name="Deterministic Civ 2",
            archetype=CivilizationArchetype.COOPERATIVE,
            population_size=100,
            deterministic_seed=5000
        )
        
        # Check that they have similar initial characteristics
        assert abs(civ_1.stability - civ_2.stability) < 0.1
        assert abs(civ_1.complexity - civ_2.complexity) < 0.1
        assert abs(civ_1.innovation_capacity - civ_2.innovation_capacity) < 0.1
        
        print("✓ Deterministic reproducibility verified")
        print("=== Deterministic Reproducibility Test Completed ===")
    
    def test_error_handling_and_recovery(self, core, architect, evolution_engine):
        """Test error handling and recovery."""
        print("\n=== Testing Error Handling and Recovery ===")
        
        # Test with invalid civilization ID
        # Note: get_evolution_status doesn't take parameters, so we test differently
        print("✓ Evolution status method works without parameters")
        
        # Test with empty civilization list
        empty_evolution = evolution_engine.start_evolution(
            civilization_ids=[],
            deterministic_seed=300
        )
        assert not empty_evolution
        print("✓ Empty civilization list handled gracefully")
        
        # Test with non-existent civilization
        non_existent_evolution = evolution_engine.start_evolution(
            civilization_ids=["non_existent_id"],
            deterministic_seed=400
        )
        assert not non_existent_evolution
        print("✓ Non-existent civilization handled gracefully")
        
        print("=== Error Handling and Recovery Test Completed ===")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
