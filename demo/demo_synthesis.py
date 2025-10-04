#!/usr/bin/env python3
"""
Demo: Answer Synthesis & Evidence Engine
Shows question → synthesizer → printed provable evidence → replay(sim_id) reproducing simulation used.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from egdol.omnimind.conversational.answer_synthesis import AnswerSynthesisAndEvidenceEngine
from tests.test_answer_synthesis import MockReasoningTrace, MockSimulationResult
from egdol.omnimind.conversational.provenance import ProvenanceTracker
from egdol.omnimind.conversational.depth_control import DepthController
from egdol.omnimind.conversational.confidence_calibrator import ConfidenceCalibrator
from datetime import datetime


def create_sample_data():
    """Create sample reasoning trace and simulation results."""
    
    # Sample reasoning trace
    reasoning_trace = MockReasoningTrace(
        step_id="demo_001",
        reasoning_type="strategic_analysis",
        processing_steps=[
            "Analyzing space colonization scenario",
            "Running multi-universe simulations", 
            "Applying meta-rules for optimization",
            "Generating strategic insights"
        ],
        output_data={
            "scenario": "space_colonization",
            "complexity": 0.85,
            "strategic_focus": "resource_allocation"
        },
        confidence=0.88
    )
    
    # Sample simulation results
    simulation_results = [
        MockSimulationResult(
            sim_id="uni7-sim42",
            universe_id="uni7",
            tick_range=[100, 200],
            results={
                "success_rate": 0.73,
                "resource_efficiency": 0.91,
                "stability": 0.67,
                "scalability": 0.88
            },
            confidence=0.84,
            meta_rules_applied=["R-12", "R-15", "R-23"]
        ),
        MockSimulationResult(
            sim_id="uni3-sim77",
            universe_id="uni3",
            tick_range=[0, 50],
            results={
                "migration_success": 0.89,
                "cultural_integration": 0.76,
                "governance_effectiveness": 0.82
            },
            confidence=0.78,
            meta_rules_applied=["R-5", "R-8", "R-14"]
        ),
        MockSimulationResult(
            sim_id="uni9-sim156",
            universe_id="uni9",
            tick_range=[50, 150],
            results={
                "technological_advancement": 0.94,
                "sustainability": 0.71,
                "conflict_resolution": 0.85
            },
            confidence=0.81,
            meta_rules_applied=["R-19", "R-22", "R-31"]
        )
    ]
    
    # Sample meta-rules
    meta_rules = [
        {"id": "R-12", "description": "Resource allocation optimization", "confidence": 0.89},
        {"id": "R-15", "description": "Risk mitigation strategies", "confidence": 0.82},
        {"id": "R-23", "description": "Scalability principles", "confidence": 0.76},
        {"id": "R-5", "description": "Cultural integration frameworks", "confidence": 0.85},
        {"id": "R-8", "description": "Governance structures", "confidence": 0.78},
        {"id": "R-14", "description": "Migration protocols", "confidence": 0.81},
        {"id": "R-19", "description": "Technological advancement", "confidence": 0.92},
        {"id": "R-22", "description": "Sustainability measures", "confidence": 0.74},
        {"id": "R-31", "description": "Conflict resolution", "confidence": 0.87}
    ]
    
    return reasoning_trace, simulation_results, meta_rules


def demo_synthesis_engine():
    """Demo the Answer Synthesis & Evidence Engine."""
    
    print("🚀 Answer Synthesis & Evidence Engine Demo")
    print("=" * 60)
    
    # Initialize components
    synthesizer = AnswerSynthesisAndEvidenceEngine()
    provenance_tracker = ProvenanceTracker()
    
    # Create sample data
    reasoning_trace, simulation_results, meta_rules = create_sample_data()
    
    print(f"📊 Sample Data:")
    print(f"   Reasoning Trace: {reasoning_trace.reasoning_type} (confidence: {reasoning_trace.confidence})")
    print(f"   Simulation Results: {len(simulation_results)} simulations")
    print(f"   Meta Rules: {len(meta_rules)} rules")
    print()
    
    # Demo different personalities and depths
    personalities = ["Strategos", "Archivist", "Lawmaker", "Oracle"]
    depths = ["short", "standard", "deep"]
    
    for personality in personalities:
        print(f"🎭 {personality} Analysis:")
        print("-" * 40)
        
        for depth in depths:
            print(f"\n📝 {depth.upper()} Response:")
            
            # Synthesize answer
            result = synthesizer.synthesize(
                reasoning_trace=reasoning_trace,
                simulation_results=simulation_results,
                meta_rules=meta_rules,
                personality=personality,
                user_pref_depth=depth
            )
            
            # Display results
            print(f"   Response: {result['response'][:200]}...")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Evidence Items: {len(result['evidence'])}")
            print(f"   Summary Points: {len(result['summary_points'])}")
            print(f"   Recommendations: {len(result['recommendations'])}")
            print(f"   Tradeoffs: {len(result['tradeoffs'])}")
            
            # Show evidence details
            print(f"   Evidence Details:")
            for i, evidence in enumerate(result['evidence'][:2]):  # Show first 2
                print(f"     {i+1}. Sim {evidence['sim_id']} (ticks {evidence['tick_range']}) - {evidence['claim_supported']}")
            
            # Show provenance
            print(f"   Provenance: {result['provenance']['universe']} (seed: {result['provenance']['seed']})")
            
            print()
        
        print()


def demo_provenance_system():
    """Demo the provenance and replay system."""
    
    print("🔍 Provenance & Replay System Demo")
    print("=" * 60)
    
    provenance_tracker = ProvenanceTracker()
    
    # Create sample provenance
    provenance = provenance_tracker.create_provenance(
        seed=12345,
        universe_id="uni7",
        sim_id="sim42",
        tick_range=[100, 200],
        metadata={"demo": True, "timestamp": datetime.now().isoformat()}
    )
    
    print(f"📋 Created Provenance:")
    print(f"   Seed: {provenance['seed']}")
    print(f"   Universe: {provenance['universe']}")
    print(f"   Simulation: {provenance['sim_id']}")
    print(f"   Tick Range: {provenance['tick_range']}")
    print(f"   Snapshot: {provenance['snapshot']}")
    print()
    
    # Create replay hook
    replay_hook = provenance_tracker.create_replay_hook(provenance)
    
    print(f"🔄 Replay Hook:")
    print(f"   Function: {replay_hook['replay']['replay_function']}")
    print(f"   Reproducible: {replay_hook['replay']['reproducible']}")
    print(f"   Snapshot ID: {replay_hook['replay']['snapshot_id']}")
    print()
    
    # Validate reproducibility
    validation = provenance_tracker.validate_reproducibility(provenance['snapshot'])
    print(f"✅ Reproducibility Validation:")
    print(f"   Valid: {validation['valid']}")
    print(f"   Reproducible: {validation['reproducible']}")
    print(f"   Replay Available: {validation['replay_available']}")
    print()


def demo_confidence_calibration():
    """Demo confidence calibration system."""
    
    print("🎯 Confidence Calibration Demo")
    print("=" * 60)
    
    calibrator = ConfidenceCalibrator()
    
    # Test different confidence levels
    confidence_scores = [0.15, 0.35, 0.55, 0.75, 0.95]
    
    for score in confidence_scores:
        indicators = calibrator.get_confidence_indicators(score)
        
        print(f"📊 Confidence Score: {score:.2f}")
        print(f"   Level: {indicators['level']}")
        print(f"   Phrase: {indicators['phrase']}")
        print(f"   Guidance: {indicators['guidance']}")
        print(f"   Should Reflect: {indicators['should_reflect']}")
        print(f"   Uncertainty Modifiers: {indicators['uncertainty_modifiers']}")
        print()


def demo_depth_control():
    """Demo depth control system."""
    
    print("📏 Depth Control System Demo")
    print("=" * 60)
    
    depth_controller = DepthController()
    
    personalities = ["Strategos", "Archivist", "Lawmaker", "Oracle"]
    complexities = [0.2, 0.5, 0.8]
    user_prefs = ["short", "standard", "deep"]
    
    for personality in personalities:
        print(f"🎭 {personality}:")
        
        for complexity in complexities:
            for user_pref in user_prefs:
                depth = depth_controller.determine_depth(personality, complexity, user_pref)
                characteristics = depth_controller.get_depth_characteristics(depth)
                
                print(f"   Complexity {complexity:.1f}, {user_pref} → {depth} "
                      f"(max_length: {characteristics['max_length']}, "
                      f"evidence_items: {characteristics['evidence_items']})")
        
        print()


def main():
    """Run the complete demo."""
    
    print("🌟 OmniMind Answer Synthesis & Evidence Engine Demo")
    print("=" * 80)
    print()
    
    try:
        # Demo 1: Answer Synthesis Engine
        demo_synthesis_engine()
        
        # Demo 2: Provenance System
        demo_provenance_system()
        
        # Demo 3: Confidence Calibration
        demo_confidence_calibration()
        
        # Demo 4: Depth Control
        demo_depth_control()
        
        print("✅ Demo completed successfully!")
        print()
        print("🎯 Key Features Demonstrated:")
        print("   • Non-template responses with evidence")
        print("   • Personality-specific synthesis")
        print("   • Depth control (short/standard/deep)")
        print("   • Confidence calibration")
        print("   • Provenance and reproducibility")
        print("   • Evidence-based recommendations")
        print("   • Tradeoff analysis")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
