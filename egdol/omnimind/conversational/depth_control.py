"""
Depth Control System
Maps personality, user preference, and complexity to appropriate answer depth.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class DepthLevel(Enum):
    """Available depth levels."""
    SHORT = auto()
    STANDARD = auto()
    DEEP = auto()


@dataclass
class DepthPolicy:
    """Depth policy for a personality."""
    personality: str
    default_depth: DepthLevel
    complexity_thresholds: Dict[float, DepthLevel]
    user_preference_weights: Dict[str, float]


class DepthController:
    """Controls answer depth based on personality, complexity, and user preferences."""
    
    def __init__(self):
        self.policies = self._initialize_depth_policies()
    
    def _initialize_depth_policies(self) -> Dict[str, DepthPolicy]:
        """Initialize depth policies for each personality."""
        return {
            "Strategos": DepthPolicy(
                personality="Strategos",
                default_depth=DepthLevel.STANDARD,
                complexity_thresholds={
                    0.3: DepthLevel.SHORT,
                    0.7: DepthLevel.STANDARD,
                    0.9: DepthLevel.DEEP
                },
                user_preference_weights={
                    "short": 0.3,
                    "standard": 0.5,
                    "deep": 0.8
                }
            ),
            "Archivist": DepthPolicy(
                personality="Archivist",
                default_depth=DepthLevel.DEEP,
                complexity_thresholds={
                    0.2: DepthLevel.SHORT,
                    0.5: DepthLevel.STANDARD,
                    0.8: DepthLevel.DEEP
                },
                user_preference_weights={
                    "short": 0.2,
                    "standard": 0.4,
                    "deep": 0.9
                }
            ),
            "Lawmaker": DepthPolicy(
                personality="Lawmaker",
                default_depth=DepthLevel.STANDARD,
                complexity_thresholds={
                    0.4: DepthLevel.SHORT,
                    0.6: DepthLevel.STANDARD,
                    0.8: DepthLevel.DEEP
                },
                user_preference_weights={
                    "short": 0.3,
                    "standard": 0.6,
                    "deep": 0.7
                }
            ),
            "Oracle": DepthPolicy(
                personality="Oracle",
                default_depth=DepthLevel.DEEP,
                complexity_thresholds={
                    0.1: DepthLevel.SHORT,
                    0.2: DepthLevel.STANDARD,
                    0.4: DepthLevel.DEEP
                },
                user_preference_weights={
                    "short": 0.1,
                    "standard": 0.2,
                    "deep": 0.9
                }
            )
        }
    
    def determine_depth(self, 
                       personality: str,
                       complexity: float,
                       user_pref: str = "standard") -> str:
        """
        Determine appropriate depth level.
        
        Args:
            personality: Current personality
            complexity: Complexity score (0.0 to 1.0)
            user_pref: User depth preference
            
        Returns:
            Depth level string
        """
        policy = self.policies.get(personality, self.policies["Strategos"])
        
        # Determine depth based on complexity
        complexity_depth = self._get_complexity_depth(complexity, policy)
        
        # Adjust based on user preference
        final_depth = self._adjust_for_user_preference(
            complexity_depth, user_pref, policy
        )
        
        return final_depth.name.lower()
    
    def _get_complexity_depth(self, complexity: float, policy: DepthPolicy) -> DepthLevel:
        """Get depth level based on complexity."""
        for threshold, depth in sorted(policy.complexity_thresholds.items()):
            if complexity <= threshold:
                return depth
        
        # If complexity is above all thresholds, use deepest level
        return DepthLevel.DEEP
    
    def _adjust_for_user_preference(self, 
                                   complexity_depth: DepthLevel,
                                   user_pref: str,
                                   policy: DepthPolicy) -> DepthLevel:
        """Adjust depth based on user preference."""
        user_weight = policy.user_preference_weights.get(user_pref, 0.5)
        
        # Special case for Oracle - always prefer deep analysis
        if policy.personality == "Oracle":
            return DepthLevel.DEEP
        
        # If user prefers deeper analysis, upgrade depth
        if user_weight > 0.6 and complexity_depth == DepthLevel.STANDARD:
            return DepthLevel.DEEP
        elif user_weight > 0.6 and complexity_depth == DepthLevel.SHORT:
            return DepthLevel.STANDARD
        
        # If user prefers shorter analysis, downgrade depth
        elif user_weight < 0.4 and complexity_depth == DepthLevel.DEEP:
            return DepthLevel.STANDARD
        elif user_weight < 0.4 and complexity_depth == DepthLevel.STANDARD:
            return DepthLevel.SHORT
        
        return complexity_depth
    
    def get_depth_characteristics(self, depth_level: str) -> Dict[str, Any]:
        """Get characteristics of a depth level."""
        characteristics = {
            "short": {
                "max_length": 200,
                "summary_points": 2,
                "recommendations": 1,
                "tradeoffs": 1,
                "evidence_items": 1,
                "style": "concise and direct"
            },
            "standard": {
                "max_length": 500,
                "summary_points": 3,
                "recommendations": 2,
                "tradeoffs": 2,
                "evidence_items": 2,
                "style": "balanced and comprehensive"
            },
            "deep": {
                "max_length": 1000,
                "summary_points": 5,
                "recommendations": 3,
                "tradeoffs": 3,
                "evidence_items": 3,
                "style": "thorough and detailed"
            }
        }
        
        return characteristics.get(depth_level, characteristics["standard"])
    
    def should_include_technical_details(self, personality: str, depth_level: str) -> bool:
        """Determine if technical details should be included."""
        technical_personalities = ["Strategos", "Lawmaker"]
        deep_levels = ["deep"]
        
        return personality in technical_personalities or depth_level in deep_levels
    
    def should_include_historical_context(self, personality: str, depth_level: str) -> bool:
        """Determine if historical context should be included."""
        historical_personalities = ["Archivist", "Oracle"]
        standard_or_deep = ["standard", "deep"]
        
        return personality in historical_personalities or depth_level in standard_or_deep
    
    def should_include_risk_analysis(self, personality: str, depth_level: str) -> bool:
        """Determine if risk analysis should be included."""
        risk_personalities = ["Strategos", "Lawmaker"]
        standard_or_deep = ["standard", "deep"]
        
        return personality in risk_personalities or depth_level in standard_or_deep
