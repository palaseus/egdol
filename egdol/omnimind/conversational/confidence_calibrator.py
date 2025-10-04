"""
Confidence Calibration System
Maps internal confidence scores to human-friendly guidance and thresholds.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class ConfidenceLevel(Enum):
    """Confidence levels."""
    VERY_LOW = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    VERY_HIGH = auto()


@dataclass
class ConfidenceMapping:
    """Confidence mapping configuration."""
    level: ConfidenceLevel
    min_score: float
    max_score: float
    human_phrase: str
    should_reflect: bool
    guidance: str


class ConfidenceCalibrator:
    """Calibrates internal confidence scores to human-friendly guidance."""
    
    def __init__(self):
        self.mappings = self._initialize_confidence_mappings()
        self.reflection_threshold = 0.6
    
    def _initialize_confidence_mappings(self) -> List[ConfidenceMapping]:
        """Initialize confidence mappings."""
        return [
            ConfidenceMapping(
                level=ConfidenceLevel.VERY_LOW,
                min_score=0.0,
                max_score=0.2,
                human_phrase="very low confidence",
                should_reflect=True,
                guidance="Requires additional simulation and analysis"
            ),
            ConfidenceMapping(
                level=ConfidenceLevel.LOW,
                min_score=0.2,
                max_score=0.4,
                human_phrase="low confidence",
                should_reflect=True,
                guidance="Consider running more simulations for validation"
            ),
            ConfidenceMapping(
                level=ConfidenceLevel.MEDIUM,
                min_score=0.4,
                max_score=0.6,
                human_phrase="medium confidence",
                should_reflect=False,
                guidance="Reasonable confidence, but could benefit from additional data"
            ),
            ConfidenceMapping(
                level=ConfidenceLevel.HIGH,
                min_score=0.6,
                max_score=0.8,
                human_phrase="high confidence",
                should_reflect=False,
                guidance="Strong evidence supports this conclusion"
            ),
            ConfidenceMapping(
                level=ConfidenceLevel.VERY_HIGH,
                min_score=0.8,
                max_score=1.0,
                human_phrase="very high confidence",
                should_reflect=False,
                guidance="Extremely strong evidence with high certainty"
            )
        ]
    
    def calibrate_confidence(self, internal_score: float) -> str:
        """
        Calibrate internal confidence score to human-friendly phrase.
        
        Args:
            internal_score: Internal confidence score (0.0 to 1.0)
            
        Returns:
            Human-friendly confidence phrase
        """
        for mapping in self.mappings:
            if mapping.min_score <= internal_score <= mapping.max_score:
                return f"{mapping.human_phrase} ({internal_score:.2f})"
        
        # Fallback
        return f"medium confidence ({internal_score:.2f})"
    
    def should_trigger_reflection(self, confidence_score: float) -> bool:
        """
        Determine if reflection should be triggered based on confidence.
        
        Args:
            confidence_score: Confidence score
            
        Returns:
            True if reflection should be triggered
        """
        return confidence_score < self.reflection_threshold
    
    def get_confidence_guidance(self, confidence_score: float) -> str:
        """
        Get guidance based on confidence level.
        
        Args:
            confidence_score: Confidence score
            
        Returns:
            Guidance string
        """
        for mapping in self.mappings:
            if mapping.min_score <= confidence_score <= mapping.max_score:
                return mapping.guidance
        
        return "Confidence level requires further analysis"
    
    def get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """
        Get confidence level enum from score.
        
        Args:
            confidence_score: Confidence score
            
        Returns:
            Confidence level enum
        """
        for mapping in self.mappings:
            if mapping.min_score <= confidence_score <= mapping.max_score:
                return mapping.level
        
        return ConfidenceLevel.MEDIUM
    
    def should_include_uncertainty_language(self, confidence_score: float) -> bool:
        """
        Determine if uncertainty language should be included.
        
        Args:
            confidence_score: Confidence score
            
        Returns:
            True if uncertainty language should be included
        """
        return confidence_score < 0.7
    
    def get_uncertainty_modifiers(self, confidence_score: float) -> List[str]:
        """
        Get appropriate uncertainty modifiers.
        
        Args:
            confidence_score: Confidence score
            
        Returns:
            List of uncertainty modifiers
        """
        if confidence_score >= 0.8:
            return ["confidently", "with certainty", "strongly suggest"]
        elif confidence_score >= 0.6:
            return ["recommend", "suggest", "indicate"]
        elif confidence_score >= 0.4:
            return ["tentatively suggest", "may", "possibly"]
        else:
            return ["uncertain", "requires further analysis", "preliminary"]
    
    def get_confidence_indicators(self, confidence_score: float) -> Dict[str, Any]:
        """
        Get comprehensive confidence indicators.
        
        Args:
            confidence_score: Confidence score
            
        Returns:
            Dictionary of confidence indicators
        """
        level = self.get_confidence_level(confidence_score)
        
        return {
            "score": confidence_score,
            "level": level.name,
            "phrase": self.calibrate_confidence(confidence_score),
            "guidance": self.get_confidence_guidance(confidence_score),
            "should_reflect": self.should_trigger_reflection(confidence_score),
            "uncertainty_modifiers": self.get_uncertainty_modifiers(confidence_score),
            "include_uncertainty": self.should_include_uncertainty_language(confidence_score)
        }
    
    def aggregate_confidence(self, confidence_scores: List[float]) -> float:
        """
        Aggregate multiple confidence scores.
        
        Args:
            confidence_scores: List of confidence scores
            
        Returns:
            Aggregated confidence score
        """
        if not confidence_scores:
            return 0.0
        
        # Weighted average with higher scores having more influence
        weights = [score ** 2 for score in confidence_scores]
        weighted_sum = sum(score * weight for score, weight in zip(confidence_scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_confidence_boost_suggestions(self, confidence_score: float) -> List[str]:
        """
        Get suggestions for boosting confidence.
        
        Args:
            confidence_score: Current confidence score
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        if confidence_score < 0.5:
            suggestions.extend([
                "Run additional simulations with different parameters",
                "Gather more evidence from multiple sources",
                "Apply more meta-rules for validation"
            ])
        elif confidence_score < 0.7:
            suggestions.extend([
                "Validate findings with cross-simulation analysis",
                "Apply additional meta-rules for confirmation"
            ])
        else:
            suggestions.append("Confidence is already high - consider final validation")
        
        return suggestions
