"""
Meta-Learning Engine
Analyzes feedback trends and adapts system behavior.
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from .feedback_loop import UserFeedback, LearningUpdate, FeedbackStorage, FeedbackType, FeedbackSentiment
from .depth_control import DepthController
from .confidence_calibrator import ConfidenceCalibrator


@dataclass
class LearningInsight:
    """Learning insight from feedback analysis."""
    insight_type: str
    confidence: float
    description: str
    personality: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "insight_type": self.insight_type,
            "confidence": self.confidence,
            "description": self.description,
            "personality": self.personality,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HeuristicUpdate:
    """Heuristic update from learning."""
    heuristic_name: str
    old_value: float
    new_value: float
    confidence: float
    personality: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heuristic_name": self.heuristic_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "confidence": self.confidence,
            "personality": self.personality,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PersonalityProfile:
    """Personality learning profile."""
    personality: str
    success_rate: float = 0.5
    preferred_depth: str = "standard"
    confidence_threshold: float = 0.6
    synthesis_style_weights: Dict[str, float] = field(default_factory=dict)
    user_satisfaction: float = 0.5
    adaptation_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "personality": self.personality,
            "success_rate": self.success_rate,
            "preferred_depth": self.preferred_depth,
            "confidence_threshold": self.confidence_threshold,
            "synthesis_style_weights": self.synthesis_style_weights,
            "user_satisfaction": self.user_satisfaction,
            "adaptation_rate": self.adaptation_rate
        }


@dataclass
class LearningMetrics:
    """Learning performance metrics."""
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    average_rating: float = 0.0
    personality_accuracy: Dict[str, float] = field(default_factory=dict)
    depth_alignment: Dict[str, float] = field(default_factory=dict)
    confidence_calibration: float = 0.0
    synthesis_quality: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_feedback": self.total_feedback,
            "positive_feedback": self.positive_feedback,
            "negative_feedback": self.negative_feedback,
            "average_rating": self.average_rating,
            "personality_accuracy": self.personality_accuracy,
            "depth_alignment": self.depth_alignment,
            "confidence_calibration": self.confidence_calibration,
            "synthesis_quality": self.synthesis_quality,
            "last_updated": self.last_updated.isoformat()
        }


class MetaLearningEngine:
    """Core meta-learning engine that adapts system behavior."""
    
    def __init__(self, storage: FeedbackStorage):
        self.storage = storage
        self.personality_profiles = self._initialize_personality_profiles()
        self.learning_metrics = LearningMetrics()
        self.depth_controller = DepthController()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.adaptation_threshold = 0.1  # Minimum change to trigger adaptation
        self.learning_rate = 0.05  # How fast to adapt
    
    def _initialize_personality_profiles(self) -> Dict[str, PersonalityProfile]:
        """Initialize personality learning profiles."""
        personalities = ["Strategos", "Archivist", "Lawmaker", "Oracle"]
        profiles = {}
        
        for personality in personalities:
            profiles[personality] = PersonalityProfile(
                personality=personality,
                synthesis_style_weights={
                    "tactical": 0.8 if personality == "Strategos" else 0.2,
                    "historical": 0.8 if personality == "Archivist" else 0.2,
                    "legal": 0.8 if personality == "Lawmaker" else 0.2,
                    "mystical": 0.8 if personality == "Oracle" else 0.2
                }
            )
        
        return profiles
    
    def analyze_feedback_trends(self, feedback_batch: List[UserFeedback]) -> Dict[str, Any]:
        """Analyze feedback trends and extract learning signals."""
        if not feedback_batch:
            return {"trends": {}, "recommendations": []}
        
        # Group feedback by personality
        personality_feedback = defaultdict(list)
        for feedback in feedback_batch:
            if feedback.personality:
                personality_feedback[feedback.personality].append(feedback)
        
        trends = {}
        recommendations = []
        
        # Analyze each personality
        for personality, feedback_list in personality_feedback.items():
            personality_trends = self._analyze_personality_trends(personality, feedback_list)
            trends[personality] = personality_trends
            
            # Generate recommendations
            if personality_trends["success_rate"] < 0.6:
                recommendations.append(f"Improve {personality} synthesis quality")
            if personality_trends["depth_misalignment"] > 0.3:
                recommendations.append(f"Adjust {personality} depth preferences")
            if personality_trends["confidence_miscalibration"] > 0.2:
                recommendations.append(f"Recalibrate {personality} confidence thresholds")
        
        return {
            "trends": trends,
            "recommendations": recommendations,
            "overall_health": self._calculate_overall_health(trends)
        }
    
    def _analyze_personality_trends(self, personality: str, feedback_list: List[UserFeedback]) -> Dict[str, Any]:
        """Analyze trends for a specific personality."""
        if not feedback_list:
            return {"success_rate": 0.5, "depth_misalignment": 0.0, "confidence_miscalibration": 0.0}
        
        # Calculate success rate
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        success_rate = sum(ratings) / len(ratings) / 5.0 if ratings else 0.5
        
        # Calculate depth alignment
        depth_feedback = [f.depth_preference for f in feedback_list if f.depth_preference]
        depth_misalignment = self._calculate_depth_misalignment(personality, depth_feedback)
        
        # Calculate confidence calibration
        confidence_feedback = [f.confidence_feedback for f in feedback_list if f.confidence_feedback is not None]
        confidence_miscalibration = self._calculate_confidence_miscalibration(confidence_feedback)
        
        return {
            "success_rate": success_rate,
            "depth_misalignment": depth_misalignment,
            "confidence_miscalibration": confidence_miscalibration,
            "feedback_count": len(feedback_list),
            "recent_trend": self._calculate_recent_trend(feedback_list)
        }
    
    def _calculate_depth_misalignment(self, personality: str, depth_feedback: List[str]) -> float:
        """Calculate how misaligned depth preferences are."""
        if not depth_feedback:
            return 0.0
        
        # Get current personality depth preferences
        profile = self.personality_profiles.get(personality)
        if not profile:
            return 0.0
        
        # Calculate misalignment
        preferred_depth = profile.preferred_depth
        misalignment = sum(1 for depth in depth_feedback if depth != preferred_depth) / len(depth_feedback)
        
        return misalignment
    
    def _calculate_confidence_miscalibration(self, confidence_feedback: List[float]) -> float:
        """Calculate confidence miscalibration."""
        if not confidence_feedback:
            return 0.0
        
        # Calculate variance from expected confidence
        expected_confidence = 0.7  # Target confidence
        miscalibration = sum(abs(cf - expected_confidence) for cf in confidence_feedback) / len(confidence_feedback)
        
        return miscalibration
    
    def _calculate_recent_trend(self, feedback_list: List[UserFeedback]) -> str:
        """Calculate recent trend direction."""
        if len(feedback_list) < 2:
            return "stable"
        
        # Sort by timestamp
        sorted_feedback = sorted(feedback_list, key=lambda f: f.timestamp)
        
        # Compare first half vs second half
        mid_point = len(sorted_feedback) // 2
        first_half = sorted_feedback[:mid_point]
        second_half = sorted_feedback[mid_point:]
        
        first_ratings = [f.rating for f in first_half if f.rating is not None]
        second_ratings = [f.rating for f in second_half if f.rating is not None]
        
        if not first_ratings or not second_ratings:
            return "stable"
        
        first_avg = sum(first_ratings) / len(first_ratings)
        second_avg = sum(second_ratings) / len(second_ratings)
        
        if second_avg > first_avg + 0.2:
            return "improving"
        elif second_avg < first_avg - 0.2:
            return "declining"
        else:
            return "stable"
    
    def _calculate_overall_health(self, trends: Dict[str, Any]) -> float:
        """Calculate overall system health."""
        if not trends:
            return 0.5
        
        health_scores = []
        for personality, trend in trends.items():
            success_rate = trend.get("success_rate", 0.5)
            depth_misalignment = trend.get("depth_misalignment", 0.0)
            confidence_miscalibration = trend.get("confidence_miscalibration", 0.0)
            
            # Calculate health score (higher is better)
            health_score = success_rate - (depth_misalignment + confidence_miscalibration) / 2
            health_scores.append(max(0.0, min(1.0, health_score)))
        
        return sum(health_scores) / len(health_scores) if health_scores else 0.5
    
    def generate_learning_update(self, feedback_batch: List[UserFeedback]) -> LearningUpdate:
        """Generate learning update based on feedback analysis."""
        trends = self.analyze_feedback_trends(feedback_batch)
        
        # Generate personality adjustments
        personality_adjustments = self._generate_personality_adjustments(trends["trends"])
        
        # Generate depth preference adjustments
        depth_adjustments = self._generate_depth_adjustments(trends["trends"])
        
        # Generate confidence calibration adjustments
        confidence_adjustments = self._generate_confidence_adjustments(trends["trends"])
        
        # Generate synthesis improvements
        synthesis_improvements = self._generate_synthesis_improvements(trends["trends"])
        
        # Create learning update
        update_id = self._generate_update_id()
        update = LearningUpdate(
            update_id=update_id,
            feedback_batch=feedback_batch,
            personality_adjustments=personality_adjustments,
            depth_preference_adjustments=depth_adjustments,
            confidence_calibration_adjustments=confidence_adjustments,
            synthesis_improvements=synthesis_improvements,
            snapshot_before=self._create_system_snapshot(),
            snapshot_after=""  # Will be filled after application
        )
        
        return update
    
    def _generate_personality_adjustments(self, trends: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Generate personality-specific adjustments."""
        adjustments = {}
        
        for personality, trend in trends.items():
            personality_adjustments = {}
            
            # Adjust success rate
            success_rate = trend.get("success_rate", 0.5)
            if success_rate < 0.6:
                personality_adjustments["success_rate_boost"] = 0.1
            elif success_rate > 0.8:
                personality_adjustments["success_rate_boost"] = -0.05
            
            # Adjust depth preferences
            depth_misalignment = trend.get("depth_misalignment", 0.0)
            if depth_misalignment > 0.3:
                personality_adjustments["depth_preference_adjustment"] = 0.1
            
            # Adjust confidence thresholds
            confidence_miscalibration = trend.get("confidence_miscalibration", 0.0)
            if confidence_miscalibration > 0.2:
                personality_adjustments["confidence_threshold_adjustment"] = 0.05
            
            adjustments[personality] = personality_adjustments
        
        return adjustments
    
    def _generate_depth_adjustments(self, trends: Dict[str, Any]) -> Dict[str, float]:
        """Generate depth preference adjustments."""
        adjustments = {}
        
        for personality, trend in trends.items():
            depth_misalignment = trend.get("depth_misalignment", 0.0)
            if depth_misalignment > 0.3:
                # Adjust depth controller for this personality
                adjustments[f"{personality}_depth_weight"] = 0.1
        
        return adjustments
    
    def _generate_confidence_adjustments(self, trends: Dict[str, Any]) -> Dict[str, float]:
        """Generate confidence calibration adjustments."""
        adjustments = {}
        
        for personality, trend in trends.items():
            confidence_miscalibration = trend.get("confidence_miscalibration", 0.0)
            if confidence_miscalibration > 0.2:
                # Adjust confidence calibration
                adjustments[f"{personality}_confidence_threshold"] = 0.05
        
        return adjustments
    
    def _generate_synthesis_improvements(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthesis improvements."""
        improvements = {}
        
        for personality, trend in trends.items():
            success_rate = trend.get("success_rate", 0.5)
            if success_rate < 0.6:
                # Improve synthesis quality
                improvements[f"{personality}_synthesis_quality"] = {
                    "style_weight_adjustment": 0.1,
                    "evidence_requirement_boost": 0.05,
                    "provenance_detail_increase": 0.1
                }
        
        return improvements
    
    def _generate_update_id(self) -> str:
        """Generate unique update ID."""
        data = f"update_{time.time()}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _create_system_snapshot(self) -> str:
        """Create system snapshot for rollback capability."""
        snapshot_data = {
            "personality_profiles": {k: v.to_dict() for k, v in self.personality_profiles.items()},
            "learning_metrics": self.learning_metrics.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        snapshot_id = f"snap_{int(time.time())}_{hashlib.sha256(json.dumps(snapshot_data).encode()).hexdigest()[:8]}"
        
        # Create snapshots directory if it doesn't exist
        import os
        os.makedirs("snapshots", exist_ok=True)
        
        # Store snapshot
        with open(f"snapshots/{snapshot_id}.json", "w") as f:
            json.dump(snapshot_data, f, indent=2)
        
        return snapshot_id
    
    def apply_learning_update(self, update: LearningUpdate) -> bool:
        """Apply learning update to system."""
        try:
            # Apply personality adjustments
            for personality, adjustments in update.personality_adjustments.items():
                if personality in self.personality_profiles:
                    profile = self.personality_profiles[personality]
                    
                    for adjustment_type, value in adjustments.items():
                        if adjustment_type == "success_rate_boost":
                            profile.success_rate = min(1.0, profile.success_rate + value)
                        elif adjustment_type == "depth_preference_adjustment":
                            profile.adaptation_rate = min(0.5, profile.adaptation_rate + value)
                        elif adjustment_type == "confidence_threshold_adjustment":
                            profile.confidence_threshold = max(0.1, min(0.9, profile.confidence_threshold + value))
            
            # Apply depth adjustments
            for adjustment_key, value in update.depth_preference_adjustments.items():
                # Update depth controller preferences
                pass  # Implementation depends on depth controller interface
            
            # Apply confidence adjustments
            for adjustment_key, value in update.confidence_calibration_adjustments.items():
                # Update confidence calibrator
                pass  # Implementation depends on confidence calibrator interface
            
            # Apply synthesis improvements
            for improvement_key, improvements in update.synthesis_improvements.items():
                # Update synthesis engine
                pass  # Implementation depends on synthesis engine interface
            
            # Update learning metrics
            self.learning_metrics.total_feedback += len(update.feedback_batch)
            
            # Calculate average rating from feedback
            ratings = [f.rating for f in update.feedback_batch if f.rating is not None]
            if ratings:
                self.learning_metrics.average_rating = sum(ratings) / len(ratings)
            
            # Update positive/negative feedback counts
            for feedback in update.feedback_batch:
                if feedback.sentiment == FeedbackSentiment.POSITIVE:
                    self.learning_metrics.positive_feedback += 1
                elif feedback.sentiment == FeedbackSentiment.NEGATIVE:
                    self.learning_metrics.negative_feedback += 1
            
            self.learning_metrics.last_updated = datetime.now()
            
            # Create after snapshot
            update.snapshot_after = self._create_system_snapshot()
            
            # Store update
            self.storage.store_learning_update(update)
            
            return True
            
        except Exception as e:
            print(f"Error applying learning update: {e}")
            return False
    
    def get_learning_metrics(self) -> LearningMetrics:
        """Get current learning metrics."""
        return self.learning_metrics
    
    def get_personality_profiles(self) -> Dict[str, PersonalityProfile]:
        """Get current personality profiles."""
        return self.personality_profiles
    
    def should_trigger_learning(self, feedback_count: int = 10) -> bool:
        """Determine if learning should be triggered."""
        recent_feedback = self.storage.get_feedback_batch(feedback_count)
        
        if len(recent_feedback) < 3:  # Minimum feedback for learning
            return False
        
        # Check if there's enough negative feedback to warrant learning
        negative_count = sum(1 for f in recent_feedback if f.sentiment == FeedbackSentiment.NEGATIVE)
        return negative_count >= 1  # At least 1 negative feedback item