"""
Answer Synthesis & Evidence Engine
Transforms reasoning traces, simulation outputs, and meta-rules into 
human-grade answers with evidence and provenance.
"""

import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random

from .provenance import ProvenanceTracker
from .depth_control import DepthController
from .confidence_calibrator import ConfidenceCalibrator


@dataclass
class EvidenceItem:
    """Evidence item with provenance."""
    sim_id: str
    tick_range: List[int]
    rule_ids: List[str]
    confidence: float
    claim_supported: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sim_id": self.sim_id,
            "tick_range": self.tick_range,
            "rule_ids": self.rule_ids,
            "confidence": self.confidence,
            "claim_supported": self.claim_supported,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SynthesisResult:
    """Complete synthesis result."""
    response: str
    summary_points: List[str]
    recommendations: List[str]
    tradeoffs: List[str]
    evidence: List[EvidenceItem]
    provenance: Dict[str, Any]
    confidence: float
    personality: str
    depth_level: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response": self.response,
            "summary_points": self.summary_points,
            "recommendations": self.recommendations,
            "tradeoffs": self.tradeoffs,
            "evidence": [e.to_dict() for e in self.evidence],
            "provenance": self.provenance,
            "confidence": self.confidence,
            "personality": self.personality,
            "depth_level": self.depth_level,
            "timestamp": self.timestamp.isoformat()
        }


class AnswerSynthesisAndEvidenceEngine:
    """Main synthesizer that creates human-grade answers from reasoning traces."""
    
    def __init__(self):
        self.provenance_tracker = ProvenanceTracker()
        self.depth_controller = DepthController()
        self.confidence_calibrator = ConfidenceCalibrator()
        self._random_seed = None
        
    def synthesize(self, 
                  reasoning_trace: Any,
                  simulation_results: List[Any],
                  meta_rules: List[Dict[str, Any]],
                  personality: str,
                  user_pref_depth: str = "standard") -> Dict[str, Any]:
        """
        Synthesize a complete answer from reasoning trace and simulation results.
        
        Args:
            reasoning_trace: The reasoning trace from the reasoning engine
            simulation_results: List of simulation results
            meta_rules: List of meta-rules applied
            personality: Current personality (Strategos, Archivist, Lawmaker, Oracle)
            user_pref_depth: User depth preference (short, standard, deep)
            
        Returns:
            Complete synthesis result with evidence and provenance
        """
        # Set deterministic seed for reproducible results
        self._set_deterministic_seed(reasoning_trace, simulation_results)
        
        # Determine appropriate depth
        complexity = self._calculate_complexity(reasoning_trace, simulation_results)
        depth_level = self.depth_controller.determine_depth(
            personality=personality,
            complexity=complexity,
            user_pref=user_pref_depth
        )
        
        # Generate evidence from simulation results
        evidence_items = self._generate_evidence(simulation_results, meta_rules)
        
        # Create provenance
        provenance = self._create_provenance(simulation_results)
        
        # Synthesize response based on personality and depth
        response = self._synthesize_response(
            reasoning_trace, simulation_results, meta_rules, 
            personality, depth_level, evidence_items
        )
        
        # Extract structured components
        summary_points = self._extract_summary_points(response, personality)
        recommendations = self._extract_recommendations(response, personality)
        tradeoffs = self._extract_tradeoffs(response, personality)
        
        # Calculate final confidence
        confidence = self._calculate_final_confidence(
            reasoning_trace, simulation_results, evidence_items
        )
        
        # Create synthesis result
        result = SynthesisResult(
            response=response,
            summary_points=summary_points,
            recommendations=recommendations,
            tradeoffs=tradeoffs,
            evidence=evidence_items,
            provenance=provenance,
            confidence=confidence,
            personality=personality,
            depth_level=depth_level
        )
        
        return result.to_dict()
    
    def _set_deterministic_seed(self, reasoning_trace: Any, simulation_results: List[Any]):
        """Set deterministic seed for reproducible results."""
        # Create seed from trace and simulation data
        seed_data = f"{reasoning_trace.step_id}_{len(simulation_results)}"
        seed = int(hashlib.md5(seed_data.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        self._random_seed = seed
    
    def _calculate_complexity(self, reasoning_trace: Any, simulation_results: List[Any]) -> float:
        """Calculate complexity score from reasoning trace and simulations."""
        base_complexity = reasoning_trace.confidence
        
        # Add complexity from simulation results
        if simulation_results:
            sim_complexity = sum(sim.confidence for sim in simulation_results) / len(simulation_results)
            base_complexity = (base_complexity + sim_complexity) / 2
        
        return min(1.0, base_complexity)
    
    def _generate_evidence(self, simulation_results: List[Any], meta_rules: List[Dict[str, Any]]) -> List[EvidenceItem]:
        """Generate evidence items from simulation results."""
        evidence_items = []
        
        for i, sim_result in enumerate(simulation_results):
            # Create evidence for key claims
            claims = self._extract_claims_from_simulation(sim_result)
            
            for claim in claims:
                evidence_item = EvidenceItem(
                    sim_id=sim_result.sim_id,
                    tick_range=sim_result.tick_range,
                    rule_ids=sim_result.meta_rules_applied,
                    confidence=sim_result.confidence,
                    claim_supported=claim
                )
                evidence_items.append(evidence_item)
        
        return evidence_items
    
    def _extract_claims_from_simulation(self, sim_result: Any) -> List[str]:
        """Extract key claims from simulation results."""
        claims = []
        
        # Extract claims from simulation results
        if hasattr(sim_result, 'results'):
            for key, value in sim_result.results.items():
                if isinstance(value, (int, float)):
                    claims.append(f"{key}: {value:.2f}")
                else:
                    claims.append(f"{key}: {value}")
        
        return claims[:3]  # Limit to 3 claims per simulation
    
    def _create_provenance(self, simulation_results: List[Any]) -> Dict[str, Any]:
        """Create provenance information."""
        if not simulation_results:
            return self.provenance_tracker.create_provenance(
                seed=self._random_seed or 12345,
                universe_id="default",
                sim_id="none",
                tick_range=[0, 0]
            )
        
        # Use first simulation for provenance
        first_sim = simulation_results[0]
        return self.provenance_tracker.create_provenance(
            seed=self._random_seed or 12345,
            universe_id=first_sim.universe_id,
            sim_id=first_sim.sim_id,
            tick_range=first_sim.tick_range
        )
    
    def _synthesize_response(self, 
                            reasoning_trace: Any,
                            simulation_results: List[Any],
                            meta_rules: List[Dict[str, Any]],
                            personality: str,
                            depth_level: str,
                            evidence_items: List[EvidenceItem]) -> str:
        """Synthesize the main response text."""
        
        # Get personality-specific templates and style
        personality_style = self._get_personality_style(personality)
        
        # Extract key insights from reasoning trace
        insights = self._extract_insights(reasoning_trace, simulation_results)
        
        # Build response based on depth level
        if depth_level == "short":
            return self._build_short_response(insights, personality_style, evidence_items)
        elif depth_level == "deep":
            return self._build_deep_response(insights, personality_style, evidence_items, meta_rules)
        else:
            return self._build_standard_response(insights, personality_style, evidence_items)
    
    def _get_personality_style(self, personality: str) -> Dict[str, str]:
        """Get personality-specific style elements."""
        styles = {
            "Strategos": {
                "greeting": "Commander,",
                "focus": "tactical analysis and strategic planning",
                "tone": "authoritative and decisive",
                "keywords": ["strategy", "tactical", "military", "defense", "offense"]
            },
            "Archivist": {
                "greeting": "From the archives of knowledge,",
                "focus": "historical analysis and knowledge preservation",
                "tone": "scholarly and methodical",
                "keywords": ["historical", "precedent", "record", "documentation", "research"]
            },
            "Lawmaker": {
                "greeting": "According to the principles of governance,",
                "focus": "legal frameworks and governance principles",
                "tone": "precise and authoritative",
                "keywords": ["legal", "framework", "governance", "regulation", "jurisdiction"]
            },
            "Oracle": {
                "greeting": "In the cosmic dance of reality,",
                "focus": "mystical insights and cosmic wisdom",
                "tone": "mystical and profound",
                "keywords": ["cosmic", "destiny", "forces", "wisdom", "enlightenment"]
            }
        }
        
        return styles.get(personality, styles["Strategos"])
    
    def _extract_insights(self, reasoning_trace: Any, simulation_results: List[Any]) -> Dict[str, Any]:
        """Extract key insights from reasoning trace and simulations."""
        insights = {
            "primary_conclusion": "",
            "supporting_evidence": [],
            "key_metrics": {},
            "risks": [],
            "opportunities": []
        }
        
        # Extract from reasoning trace
        if hasattr(reasoning_trace, 'output_data'):
            insights["primary_conclusion"] = self._extract_primary_conclusion(reasoning_trace.output_data)
        
        # Extract from simulation results
        for sim_result in simulation_results:
            if hasattr(sim_result, 'results'):
                insights["key_metrics"].update(sim_result.results)
        
        return insights
    
    def _extract_primary_conclusion(self, output_data: Dict[str, Any]) -> str:
        """Extract primary conclusion from output data."""
        if "scenario" in output_data:
            return f"Analysis of {output_data['scenario']} scenario"
        return "Strategic analysis completed"
    
    def _build_short_response(self, insights: Dict[str, Any], style: Dict[str, str], evidence_items: List[EvidenceItem]) -> str:
        """Build short response (TL;DR style)."""
        response_parts = [
            f"{style['greeting']} {insights['primary_conclusion']}.",
            f"Key findings: {', '.join(insights['key_metrics'].keys()[:2])}",
            f"Recommendation: Implement systematic approach with clear milestones."
        ]
        
        return " ".join(response_parts)
    
    def _build_standard_response(self, insights: Dict[str, Any], style: Dict[str, str], evidence_items: List[EvidenceItem]) -> str:
        """Build standard response."""
        response_parts = [
            f"{style['greeting']} {insights['primary_conclusion']}.",
            f"I recommend a systematic approach focusing on {style['focus']}.",
            f"The key factors to consider are {', '.join(list(insights['key_metrics'].keys())[:3])}.",
            f"Evidence from {len(evidence_items)} simulation runs supports these conclusions.",
            f"We should develop a phased approach with clear milestones and contingency plans."
        ]
        
        return " ".join(response_parts)
    
    def _build_deep_response(self, insights: Dict[str, Any], style: Dict[str, str], evidence_items: List[EvidenceItem], meta_rules: List[Dict[str, Any]]) -> str:
        """Build deep response with comprehensive analysis."""
        response_parts = [
            f"{style['greeting']} {insights['primary_conclusion']}.",
            f"Comprehensive analysis reveals critical insights into {style['focus']}.",
            f"Simulation data from {len(evidence_items)} runs provides strong evidence for strategic recommendations.",
            f"Key metrics include: {', '.join(f'{k}: {v:.2f}' for k, v in list(insights['key_metrics'].items())[:3])}.",
            f"Meta-rules applied: {', '.join(rule['id'] for rule in meta_rules[:3])}.",
            f"Risk assessment indicates {len(insights.get('risks', []))} potential challenges requiring mitigation.",
            f"Implementation requires phased approach with clear milestones, resource allocation, and contingency planning."
        ]
        
        return " ".join(response_parts)
    
    def _extract_summary_points(self, response: str, personality: str) -> List[str]:
        """Extract summary points from response."""
        # Simple extraction - in real implementation, this would be more sophisticated
        points = [
            f"Strategic analysis completed with {personality} perspective",
            "Systematic approach recommended with clear milestones",
            "Evidence-based recommendations with simulation support"
        ]
        
        return points
    
    def _extract_recommendations(self, response: str, personality: str) -> List[str]:
        """Extract actionable recommendations."""
        recommendations = [
            "Implement systematic approach with clear milestones",
            "Develop contingency plans for risk mitigation",
            "Establish resource allocation framework"
        ]
        
        return recommendations
    
    def _extract_tradeoffs(self, response: str, personality: str) -> List[str]:
        """Extract tradeoffs and risks."""
        tradeoffs = [
            "Speed vs. thoroughness in implementation",
            "Resource allocation vs. risk mitigation",
            "Scalability vs. immediate effectiveness"
        ]
        
        return tradeoffs
    
    def _calculate_final_confidence(self, reasoning_trace: Any, simulation_results: List[Any], evidence_items: List[EvidenceItem]) -> float:
        """Calculate final confidence score."""
        base_confidence = reasoning_trace.confidence
        
        # If base confidence is very high, preserve it
        if base_confidence >= 0.9:
            return base_confidence
        
        # Boost confidence with evidence
        if evidence_items:
            evidence_confidence = sum(item.confidence for item in evidence_items) / len(evidence_items)
            base_confidence = (base_confidence + evidence_confidence) / 2
        
        # Boost confidence with simulation results
        if simulation_results:
            sim_confidence = sum(sim.confidence for sim in simulation_results) / len(simulation_results)
            base_confidence = (base_confidence + sim_confidence) / 2
        
        return min(1.0, base_confidence)
