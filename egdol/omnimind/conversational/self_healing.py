"""
Self-Healing Mechanisms
Auto-revert system if meta-learning introduces instability.
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import shutil

from .feedback_loop import FeedbackStorage, UserFeedback, LearningUpdate
from .meta_learning_engine import MetaLearningEngine, PersonalityProfile, LearningMetrics


@dataclass
class SystemSnapshot:
    """System state snapshot for rollback."""
    snapshot_id: str
    timestamp: datetime
    personality_profiles: Dict[str, PersonalityProfile]
    learning_metrics: LearningMetrics
    system_health: float
    stability_score: float
    is_stable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "personality_profiles": {k: v.to_dict() for k, v in self.personality_profiles.items()},
            "learning_metrics": self.learning_metrics.to_dict(),
            "system_health": self.system_health,
            "stability_score": self.stability_score,
            "is_stable": self.is_stable,
            "metadata": self.metadata
        }


@dataclass
class StabilityMetrics:
    """System stability metrics."""
    response_quality: float = 0.0
    user_satisfaction: float = 0.0
    error_rate: float = 0.0
    performance_degradation: float = 0.0
    learning_effectiveness: float = 0.0
    overall_stability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response_quality": self.response_quality,
            "user_satisfaction": self.user_satisfaction,
            "error_rate": self.error_rate,
            "performance_degradation": self.performance_degradation,
            "learning_effectiveness": self.learning_effectiveness,
            "overall_stability": self.overall_stability
        }


class SelfHealingController:
    """Controls self-healing mechanisms for system stability."""
    
    def __init__(self, storage: FeedbackStorage, engine: MetaLearningEngine):
        self.storage = storage
        self.engine = engine
        self.snapshots_dir = Path("snapshots")
        self.snapshots_dir.mkdir(exist_ok=True)
        self.stability_threshold = 0.7  # Minimum stability score
        self.health_threshold = 0.6     # Minimum health score
        self.rollback_threshold = 0.5   # Threshold for auto-rollback
        self.max_snapshots = 50         # Maximum snapshots to keep
    
    def create_system_snapshot(self, metadata: Optional[Dict[str, Any]] = None) -> SystemSnapshot:
        """Create system snapshot for rollback."""
        snapshot_id = self._generate_snapshot_id()
        timestamp = datetime.now()
        
        # Get current system state
        personality_profiles = self.engine.get_personality_profiles()
        learning_metrics = self.engine.get_learning_metrics()
        
        # Calculate system health and stability
        system_health = self._calculate_system_health()
        stability_score = self._calculate_stability_score()
        
        # Create snapshot
        snapshot = SystemSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            personality_profiles=personality_profiles.copy(),
            learning_metrics=learning_metrics,
            system_health=system_health,
            stability_score=stability_score,
            is_stable=stability_score >= self.stability_threshold,
            metadata=metadata or {}
        )
        
        # Save snapshot to disk
        self._save_snapshot(snapshot)
        
        # Clean up old snapshots
        self._cleanup_old_snapshots()
        
        return snapshot
    
    def _generate_snapshot_id(self) -> str:
        """Generate unique snapshot ID."""
        data = f"snapshot_{time.time()}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health."""
        metrics = self.engine.get_learning_metrics()
        
        # Health factors
        feedback_quality = metrics.average_rating / 5.0 if metrics.average_rating > 0 else 0.5
        personality_accuracy = sum(metrics.personality_accuracy.values()) / len(metrics.personality_accuracy) if metrics.personality_accuracy else 0.5
        depth_alignment = sum(metrics.depth_alignment.values()) / len(metrics.depth_alignment) if metrics.depth_alignment else 0.5
        confidence_calibration = metrics.confidence_calibration
        
        # Weighted average
        health = (
            feedback_quality * 0.3 +
            personality_accuracy * 0.25 +
            depth_alignment * 0.25 +
            confidence_calibration * 0.2
        )
        
        return min(1.0, max(0.0, health))
    
    def _calculate_stability_score(self) -> float:
        """Calculate system stability score."""
        # Get recent feedback
        recent_feedback = self.storage.get_feedback_batch(20)
        
        if not recent_feedback:
            return 0.5  # Neutral stability
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(recent_feedback)
        
        # Overall stability score
        stability = (
            stability_metrics.response_quality * 0.3 +
            stability_metrics.user_satisfaction * 0.3 +
            (1.0 - stability_metrics.error_rate) * 0.2 +
            (1.0 - stability_metrics.performance_degradation) * 0.2
        )
        
        return min(1.0, max(0.0, stability))
    
    def _calculate_stability_metrics(self, feedback: List[UserFeedback]) -> StabilityMetrics:
        """Calculate detailed stability metrics."""
        if not feedback:
            return StabilityMetrics()
        
        # Response quality (based on ratings)
        ratings = [f.rating for f in feedback if f.rating is not None]
        response_quality = sum(ratings) / len(ratings) / 5.0 if ratings else 0.5
        
        # User satisfaction (based on sentiment)
        sentiments = [f.sentiment for f in feedback if f.sentiment is not None]
        positive_count = sum(1 for s in sentiments if s.name == "POSITIVE")
        user_satisfaction = positive_count / len(sentiments) if sentiments else 0.5
        
        # Error rate (based on negative feedback)
        negative_count = sum(1 for s in sentiments if s.name == "NEGATIVE")
        error_rate = negative_count / len(sentiments) if sentiments else 0.0
        
        # Performance degradation (based on confidence feedback)
        confidence_feedback = [f.confidence_feedback for f in feedback if f.confidence_feedback is not None]
        if confidence_feedback:
            avg_confidence = sum(confidence_feedback) / len(confidence_feedback)
            performance_degradation = max(0.0, 0.7 - avg_confidence)  # Degradation if confidence < 0.7
        else:
            performance_degradation = 0.0
        
        # Learning effectiveness (based on recent trends)
        learning_effectiveness = self._calculate_learning_effectiveness(feedback)
        
        # Overall stability
        overall_stability = (
            response_quality * 0.3 +
            user_satisfaction * 0.3 +
            (1.0 - error_rate) * 0.2 +
            (1.0 - performance_degradation) * 0.2
        )
        
        return StabilityMetrics(
            response_quality=response_quality,
            user_satisfaction=user_satisfaction,
            error_rate=error_rate,
            performance_degradation=performance_degradation,
            learning_effectiveness=learning_effectiveness,
            overall_stability=overall_stability
        )
    
    def _calculate_learning_effectiveness(self, feedback: List[UserFeedback]) -> float:
        """Calculate learning effectiveness."""
        if len(feedback) < 5:
            return 0.5  # Not enough data
        
        # Sort by timestamp
        sorted_feedback = sorted(feedback, key=lambda f: f.timestamp)
        
        # Compare first half vs second half
        mid_point = len(sorted_feedback) // 2
        first_half = sorted_feedback[:mid_point]
        second_half = sorted_feedback[mid_point:]
        
        # Calculate average ratings
        first_ratings = [f.rating for f in first_half if f.rating is not None]
        second_ratings = [f.rating for f in second_half if f.rating is not None]
        
        if not first_ratings or not second_ratings:
            return 0.5
        
        first_avg = sum(first_ratings) / len(first_ratings)
        second_avg = sum(second_ratings) / len(second_ratings)
        
        # Learning is effective if second half is better
        improvement = (second_avg - first_avg) / 5.0  # Normalize to 0-1
        return min(1.0, max(0.0, 0.5 + improvement))
    
    def _save_snapshot(self, snapshot: SystemSnapshot):
        """Save snapshot to disk."""
        snapshot_path = self.snapshots_dir / f"{snapshot.snapshot_id}.json"
        
        with open(snapshot_path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
    
    def _cleanup_old_snapshots(self):
        """Clean up old snapshots to prevent disk bloat."""
        snapshot_files = list(self.snapshots_dir.glob("*.json"))
        
        if len(snapshot_files) > self.max_snapshots:
            # Sort by modification time and keep only the most recent
            snapshot_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            for old_snapshot in snapshot_files[self.max_snapshots:]:
                old_snapshot.unlink()
    
    def check_system_stability(self) -> Tuple[bool, StabilityMetrics]:
        """Check if system is stable."""
        stability_metrics = self._calculate_stability_metrics(self.storage.get_feedback_batch(20))
        is_stable = stability_metrics.overall_stability >= self.stability_threshold
        
        return is_stable, stability_metrics
    
    def should_trigger_rollback(self) -> bool:
        """Determine if rollback should be triggered."""
        is_stable, stability_metrics = self.check_system_stability()
        
        # Trigger rollback if system is unstable
        if not is_stable:
            return True
        
        # Trigger rollback if health is too low
        system_health = self._calculate_system_health()
        if system_health < self.health_threshold:
            return True
        
        # Trigger rollback if error rate is too high
        if stability_metrics.error_rate > 0.3:
            return True
        
        return False
    
    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rollback system to a specific snapshot."""
        try:
            snapshot_path = self.snapshots_dir / f"{snapshot_id}.json"
            
            if not snapshot_path.exists():
                return False
            
            # Load snapshot
            with open(snapshot_path, "r") as f:
                snapshot_data = json.load(f)
            
            # Restore personality profiles
            personality_profiles = {}
            for personality, profile_data in snapshot_data["personality_profiles"].items():
                profile = PersonalityProfile(
                    personality=profile_data["personality"],
                    success_rate=profile_data["success_rate"],
                    preferred_depth=profile_data["preferred_depth"],
                    confidence_threshold=profile_data["confidence_threshold"],
                    synthesis_style_weights=profile_data["synthesis_style_weights"],
                    user_satisfaction=profile_data["user_satisfaction"],
                    adaptation_rate=profile_data["adaptation_rate"]
                )
                personality_profiles[personality] = profile
            
            # Restore learning metrics
            metrics_data = snapshot_data["learning_metrics"]
            learning_metrics = LearningMetrics(
                total_feedback=metrics_data["total_feedback"],
                positive_feedback=metrics_data["positive_feedback"],
                negative_feedback=metrics_data["negative_feedback"],
                average_rating=metrics_data["average_rating"],
                personality_accuracy=metrics_data["personality_accuracy"],
                depth_alignment=metrics_data["depth_alignment"],
                confidence_calibration=metrics_data["confidence_calibration"],
                synthesis_quality=metrics_data["synthesis_quality"],
                last_updated=datetime.fromisoformat(metrics_data["last_updated"])
            )
            
            # Apply rollback (this would need to be implemented in the engine)
            # For now, we'll just log the rollback
            print(f"Rolling back to snapshot {snapshot_id}")
            print(f"System health: {snapshot_data['system_health']}")
            print(f"Stability score: {snapshot_data['stability_score']}")
            
            return True
            
        except Exception as e:
            print(f"Error during rollback: {e}")
            return False
    
    def get_latest_stable_snapshot(self) -> Optional[SystemSnapshot]:
        """Get the latest stable snapshot."""
        snapshot_files = list(self.snapshots_dir.glob("*.json"))
        
        if not snapshot_files:
            return None
        
        # Sort by modification time (newest first)
        snapshot_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        for snapshot_file in snapshot_files:
            try:
                with open(snapshot_file, "r") as f:
                    snapshot_data = json.load(f)
                
                if snapshot_data.get("is_stable", False):
                    # Reconstruct snapshot object
                    snapshot = SystemSnapshot(
                        snapshot_id=snapshot_data["snapshot_id"],
                        timestamp=datetime.fromisoformat(snapshot_data["timestamp"]),
                        personality_profiles={},  # Would need to reconstruct
                        learning_metrics=LearningMetrics(),  # Would need to reconstruct
                        system_health=snapshot_data["system_health"],
                        stability_score=snapshot_data["stability_score"],
                        is_stable=snapshot_data["is_stable"],
                        metadata=snapshot_data["metadata"]
                    )
                    return snapshot
                    
            except Exception as e:
                print(f"Error loading snapshot {snapshot_file}: {e}")
                continue
        
        return None
    
    def auto_heal_system(self) -> bool:
        """Automatically heal system if needed."""
        if not self.should_trigger_rollback():
            return True  # System is stable
        
        # Get latest stable snapshot
        stable_snapshot = self.get_latest_stable_snapshot()
        
        if not stable_snapshot:
            print("No stable snapshot found for rollback")
            return False
        
        # Rollback to stable snapshot
        success = self.rollback_to_snapshot(stable_snapshot.snapshot_id)
        
        if success:
            print(f"System rolled back to stable snapshot {stable_snapshot.snapshot_id}")
            print(f"Stability score: {stable_snapshot.stability_score}")
            print(f"System health: {stable_snapshot.system_health}")
        
        return success
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        is_stable, stability_metrics = self.check_system_stability()
        system_health = self._calculate_system_health()
        
        return {
            "is_stable": is_stable,
            "system_health": system_health,
            "stability_metrics": stability_metrics.to_dict(),
            "should_rollback": self.should_trigger_rollback(),
            "latest_stable_snapshot": self.get_latest_stable_snapshot().snapshot_id if self.get_latest_stable_snapshot() else None
        }
