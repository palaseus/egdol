"""
Test Answer Synthesis & Evidence Engine
Tests that force real synthesis, not template responses.
"""

import unittest
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from egdol.omnimind.conversational.answer_synthesis import AnswerSynthesisAndEvidenceEngine
from egdol.omnimind.conversational.provenance import ProvenanceTracker
from egdol.omnimind.conversational.depth_control import DepthController
from egdol.omnimind.conversational.confidence_calibrator import ConfidenceCalibrator


@dataclass
class MockReasoningTrace:
    """Mock reasoning trace for testing."""
    step_id: str
    reasoning_type: str
    processing_steps: List[str]
    output_data: Dict[str, Any]
    confidence: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MockSimulationResult:
    """Mock simulation result for testing."""
    sim_id: str
    universe_id: str
    tick_range: List[int]
    results: Dict[str, Any]
    confidence: float
    meta_rules_applied: List[str]


class TestAnswerSynthesisEngine(unittest.TestCase):
    """Test the Answer Synthesis & Evidence Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synthesizer = AnswerSynthesisAndEvidenceEngine()
        self.provenance_tracker = ProvenanceTracker()
        self.depth_controller = DepthController()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # Mock data for testing
        self.sample_reasoning_trace = MockReasoningTrace(
            step_id="test_001",
            reasoning_type="strategic_analysis",
            processing_steps=["Analyzing strategic scenario", "Running simulation", "Generating insights"],
            output_data={"scenario": "space_colonization", "complexity": 0.8},
            confidence=0.85
        )
        
        self.sample_simulation_results = [
            MockSimulationResult(
                sim_id="uni7-sim42",
                universe_id="uni7",
                tick_range=[100, 200],
                results={"success_rate": 0.73, "resource_efficiency": 0.91},
                confidence=0.84,
                meta_rules_applied=["R-12", "R-15"]
            ),
            MockSimulationResult(
                sim_id="uni3-sim77",
                universe_id="uni3", 
                tick_range=[0, 50],
                results={"stability": 0.67, "scalability": 0.88},
                confidence=0.78,
                meta_rules_applied=["R-5", "R-8"]
            )
        ]
        
        self.sample_meta_rules = [
            {"id": "R-12", "description": "Resource allocation optimization", "confidence": 0.89},
            {"id": "R-15", "description": "Risk mitigation strategies", "confidence": 0.82},
            {"id": "R-5", "description": "Scalability principles", "confidence": 0.76}
        ]

    def test_synthesis_must_not_echo_query(self):
        """CRITICAL: Responses must not echo the original query verbatim."""
        query = "What is the optimal strategy for space colonization?"
        
        result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        # Must not contain the original query
        self.assertNotIn(query.lower(), result["response"].lower())
        self.assertNotIn("what is the optimal strategy", result["response"].lower())
        
        # Must be substantive
        self.assertGreater(len(result["response"]), 100)
        
    def test_must_include_evidence_for_claims(self):
        """Every factual claim must have supporting evidence."""
        result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        # Must have evidence
        self.assertGreater(len(result["evidence"]), 0)
        
        # Each evidence item must have required fields
        for evidence in result["evidence"]:
            self.assertIn("sim_id", evidence)
            self.assertIn("tick_range", evidence)
            self.assertIn("rule_ids", evidence)
            self.assertIn("confidence", evidence)
            
    def test_must_include_provenance(self):
        """Every response must include provenance information."""
        result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        # Must have provenance
        self.assertIn("provenance", result)
        provenance = result["provenance"]
        self.assertIn("seed", provenance)
        self.assertIn("universe", provenance)
        self.assertIn("snapshot", provenance)
        
    def test_must_include_structured_output(self):
        """Response must include all required structured fields."""
        result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        # Required fields
        required_fields = [
            "response", "summary_points", "recommendations", 
            "tradeoffs", "evidence", "provenance", "confidence"
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
            
        # Validate field types and content
        self.assertIsInstance(result["summary_points"], list)
        self.assertGreaterEqual(len(result["summary_points"]), 3)
        
        self.assertIsInstance(result["recommendations"], list)
        self.assertGreaterEqual(len(result["recommendations"]), 2)
        
        self.assertIsInstance(result["tradeoffs"], list)
        self.assertGreater(len(result["tradeoffs"]), 0)
        
    def test_deterministic_output(self):
        """Same inputs must produce same outputs."""
        result1 = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        result2 = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        # Should be identical
        self.assertEqual(result1["response"], result2["response"])
        self.assertEqual(result1["summary_points"], result2["summary_points"])
        self.assertEqual(result1["recommendations"], result2["recommendations"])
        
    def test_personality_specific_synthesis(self):
        """Different personalities should produce different synthesis styles."""
        strategos_result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        archivist_result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Archivist",
            user_pref_depth="standard"
        )
        
        # Should be different
        self.assertNotEqual(strategos_result["response"], archivist_result["response"])
        
        # Strategos should focus on tactical elements
        self.assertIn("strategic", strategos_result["response"].lower())
        
        # Archivist should focus on historical elements
        self.assertIn("historical", archivist_result["response"].lower())
        
    def test_depth_control(self):
        """Different depth preferences should produce different response lengths."""
        short_result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="short"
        )
        
        deep_result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="deep"
        )
        
        # Deep should be longer
        self.assertGreater(len(deep_result["response"]), len(short_result["response"]))
        
    def test_confidence_calibration(self):
        """High confidence should produce assertive language."""
        high_confidence_trace = MockReasoningTrace(
            step_id="test_002",
            reasoning_type="strategic_analysis",
            processing_steps=["High confidence analysis"],
            output_data={"confidence": 0.95},
            confidence=0.95
        )
        
        result = self.synthesizer.synthesize(
            reasoning_trace=high_confidence_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        # Should reflect high confidence
        self.assertGreaterEqual(result["confidence"], 0.9)
        
    def test_evidence_quality(self):
        """Evidence must be relevant and well-formed."""
        result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        for evidence in result["evidence"]:
            # Must have valid simulation ID
            self.assertTrue(evidence["sim_id"].startswith("uni"))
            
            # Must have valid tick range
            self.assertIsInstance(evidence["tick_range"], list)
            self.assertEqual(len(evidence["tick_range"]), 2)
            self.assertLess(evidence["tick_range"][0], evidence["tick_range"][1])
            
            # Must have confidence
            self.assertGreaterEqual(evidence["confidence"], 0.0)
            self.assertLessEqual(evidence["confidence"], 1.0)
            
    def test_tradeoffs_analysis(self):
        """Must include meaningful tradeoffs."""
        result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        # Must have tradeoffs
        self.assertGreater(len(result["tradeoffs"]), 0)
        
        # Tradeoffs should be meaningful (not empty strings)
        for tradeoff in result["tradeoffs"]:
            self.assertGreater(len(tradeoff), 10)
            self.assertIn("vs", tradeoff.lower())  # Should contain comparison
            
    def test_recommendations_actionability(self):
        """Recommendations must be actionable."""
        result = self.synthesizer.synthesize(
            reasoning_trace=self.sample_reasoning_trace,
            simulation_results=self.sample_simulation_results,
            meta_rules=self.sample_meta_rules,
            personality="Strategos",
            user_pref_depth="standard"
        )
        
        # Must have recommendations
        self.assertGreater(len(result["recommendations"]), 0)
        
        # Recommendations should be actionable
        for rec in result["recommendations"]:
            self.assertGreater(len(rec), 10)
            # Should contain action words
            action_words = ["implement", "develop", "establish", "create", "build", "deploy"]
            self.assertTrue(any(word in rec.lower() for word in action_words))


class TestProvenanceSystem(unittest.TestCase):
    """Test provenance and snapshot system."""
    
    def setUp(self):
        self.provenance_tracker = ProvenanceTracker()
        
    def test_provenance_creation(self):
        """Must create reproducible provenance."""
        provenance = self.provenance_tracker.create_provenance(
            seed=12345,
            universe_id="uni7",
            sim_id="sim42",
            tick_range=[100, 200]
        )
        
        self.assertIn("seed", provenance)
        self.assertIn("universe", provenance)
        self.assertIn("snapshot", provenance)
        self.assertEqual(provenance["seed"], 12345)
        self.assertEqual(provenance["universe"], "uni7")
        
    def test_snapshot_replay(self):
        """Must be able to replay simulations from snapshots."""
        provenance = self.provenance_tracker.create_provenance(
            seed=12345,
            universe_id="uni7", 
            sim_id="sim42",
            tick_range=[100, 200]
        )
        
        # Should be able to create replay hook
        replay_hook = self.provenance_tracker.create_replay_hook(provenance)
        self.assertIsNotNone(replay_hook)
        self.assertIn("replay", replay_hook)


class TestDepthControl(unittest.TestCase):
    """Test depth control system."""
    
    def setUp(self):
        self.depth_controller = DepthController()
        
    def test_depth_mapping(self):
        """Must map personality + complexity to appropriate depth."""
        # Strategos with high complexity should get deep analysis
        depth = self.depth_controller.determine_depth(
            personality="Strategos",
            complexity=0.9,
            user_pref="standard"
        )
        
        self.assertIn(depth, ["short", "standard", "deep"])
        
    def test_personality_depth_preferences(self):
        """Different personalities should have different depth preferences."""
        strategos_depth = self.depth_controller.determine_depth(
            personality="Strategos",
            complexity=0.5,
            user_pref="standard"
        )
        
        oracle_depth = self.depth_controller.determine_depth(
            personality="Oracle",
            complexity=0.5,
            user_pref="standard"
        )
        
        # Should be different
        self.assertNotEqual(strategos_depth, oracle_depth)


class TestConfidenceCalibration(unittest.TestCase):
    """Test confidence calibration system."""
    
    def setUp(self):
        self.calibrator = ConfidenceCalibrator()
        
    def test_confidence_mapping(self):
        """Must map internal confidence to human-friendly phrases."""
        high_conf = self.calibrator.calibrate_confidence(0.9)
        medium_conf = self.calibrator.calibrate_confidence(0.6)
        low_conf = self.calibrator.calibrate_confidence(0.3)
        
        self.assertIn("high", high_conf.lower())
        self.assertIn("medium", medium_conf.lower())
        self.assertIn("low", low_conf.lower())
        
    def test_threshold_behavior(self):
        """Must trigger appropriate behavior based on confidence thresholds."""
        # High confidence should not trigger reflection
        should_reflect_high = self.calibrator.should_trigger_reflection(0.9)
        self.assertFalse(should_reflect_high)
        
        # Low confidence should trigger reflection
        should_reflect_low = self.calibrator.should_trigger_reflection(0.3)
        self.assertTrue(should_reflect_low)


if __name__ == "__main__":
    unittest.main()
