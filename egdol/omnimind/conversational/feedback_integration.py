"""
Feedback Integration
Hooks feedback collection into the response pipeline.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .feedback_loop import FeedbackStorage, FeedbackCollector, UserFeedback, FeedbackType
from .meta_learning_engine import MetaLearningEngine
from .self_healing import SelfHealingController
from .answer_synthesis import AnswerSynthesisAndEvidenceEngine


@dataclass
class FeedbackPrompt:
    """Feedback prompt configuration."""
    prompt_text: str
    rating_scale: Tuple[int, int] = (1, 5)
    include_text_feedback: bool = True
    include_personality_feedback: bool = True
    include_depth_feedback: bool = True
    include_confidence_feedback: bool = True
    timeout_seconds: int = 30


class FeedbackIntegration:
    """Integrates feedback collection into the response pipeline."""
    
    def __init__(self, 
                 storage: FeedbackStorage,
                 collector: FeedbackCollector,
                 learning_engine: MetaLearningEngine,
                 healing_controller: SelfHealingController):
        self.storage = storage
        self.collector = collector
        self.learning_engine = learning_engine
        self.healing_controller = healing_controller
        self.feedback_prompt = self._create_default_feedback_prompt()
        self.learning_threshold = 5  # Minimum feedback for learning
        self.auto_heal_enabled = True
    
    def _create_default_feedback_prompt(self) -> FeedbackPrompt:
        """Create default feedback prompt."""
        return FeedbackPrompt(
            prompt_text="How was this response? Please rate and provide feedback:",
            rating_scale=(1, 5),
            include_text_feedback=True,
            include_personality_feedback=True,
            include_depth_feedback=True,
            include_confidence_feedback=True,
            timeout_seconds=30
        )
    
    def process_response_with_feedback(self, 
                                     response: Dict[str, Any],
                                     session_id: str,
                                     user_input: str,
                                     personality: str) -> Dict[str, Any]:
        """Process response and collect feedback."""
        
        # Generate response ID
        response_id = self._generate_response_id(session_id)
        
        # Add response metadata
        response["response_id"] = response_id
        response["session_id"] = session_id
        response["timestamp"] = datetime.now().isoformat()
        response["user_input"] = user_input
        response["personality"] = personality
        
        # Check if feedback should be collected
        if self._should_collect_feedback(session_id):
            response["feedback_prompt"] = self._generate_feedback_prompt()
            response["feedback_required"] = True
        else:
            response["feedback_required"] = False
        
        return response
    
    def _generate_response_id(self, session_id: str) -> str:
        """Generate unique response ID."""
        return f"{session_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    def _should_collect_feedback(self, session_id: str) -> bool:
        """Determine if feedback should be collected."""
        # Collect feedback every 3rd response or if system is unstable
        session_feedback = self.collector.get_session_feedback(session_id)
        
        # Check if system needs feedback for stability
        if self.auto_heal_enabled:
            system_status = self.healing_controller.get_system_status()
            if not system_status["is_stable"]:
                return True
        
        # Collect feedback periodically
        return len(session_feedback) % 3 == 0
    
    def _generate_feedback_prompt(self) -> Dict[str, Any]:
        """Generate feedback prompt."""
        prompt = {
            "text": self.feedback_prompt.prompt_text,
            "rating_scale": {
                "min": self.feedback_prompt.rating_scale[0],
                "max": self.feedback_prompt.rating_scale[1]
            },
            "fields": []
        }
        
        if self.feedback_prompt.include_text_feedback:
            prompt["fields"].append({
                "type": "text",
                "name": "text",
                "label": "Additional comments (optional)",
                "required": False
            })
        
        if self.feedback_prompt.include_personality_feedback:
            prompt["fields"].append({
                "type": "select",
                "name": "personality",
                "label": "Was the personality appropriate?",
                "options": ["Yes", "No", "Partially"],
                "required": False
            })
        
        if self.feedback_prompt.include_depth_feedback:
            prompt["fields"].append({
                "type": "select",
                "name": "depth_preference",
                "label": "Was the response depth appropriate?",
                "options": ["Too short", "Just right", "Too long"],
                "required": False
            })
        
        if self.feedback_prompt.include_confidence_feedback:
            prompt["fields"].append({
                "type": "slider",
                "name": "confidence_feedback",
                "label": "How confident should the system be?",
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "required": False
            })
        
        return prompt
    
    def collect_user_feedback(self, 
                            session_id: str,
                            response_id: str,
                            feedback_data: Dict[str, Any]) -> UserFeedback:
        """Collect and process user feedback."""
        
        # Collect feedback
        feedback = self.collector.collect_feedback(
            session_id=session_id,
            response_id=response_id,
            feedback_data=feedback_data
        )
        
        # Check if learning should be triggered
        if self._should_trigger_learning():
            self._trigger_learning_cycle()
        
        # Check if auto-healing is needed
        if self.auto_heal_enabled and self.healing_controller.should_trigger_rollback():
            self._trigger_auto_healing()
        
        return feedback
    
    def _should_trigger_learning(self) -> bool:
        """Determine if learning should be triggered."""
        # Check if enough feedback has been collected
        recent_feedback = self.storage.get_feedback_batch(self.learning_threshold)
        
        if len(recent_feedback) < self.learning_threshold:
            return False
        
        # Check if learning engine recommends learning
        return self.learning_engine.should_trigger_learning(len(recent_feedback))
    
    def _trigger_learning_cycle(self):
        """Trigger complete learning cycle."""
        try:
            # Get recent feedback
            feedback_batch = self.storage.get_feedback_batch(20)
            
            if not feedback_batch:
                return
            
            # Create system snapshot before learning
            snapshot = self.healing_controller.create_system_snapshot({
                "learning_trigger": "feedback_threshold",
                "feedback_count": len(feedback_batch)
            })
            
            # Generate learning update
            update = self.learning_engine.generate_learning_update(feedback_batch)
            update.snapshot_before = snapshot.snapshot_id
            
            # Apply learning update
            success = self.learning_engine.apply_learning_update(update)
            
            if success:
                # Create snapshot after learning
                after_snapshot = self.healing_controller.create_system_snapshot({
                    "learning_completed": True,
                    "update_id": update.update_id
                })
                update.snapshot_after = after_snapshot.snapshot_id
                
                print(f"Learning cycle completed. Update: {update.update_id}")
                print(f"Before snapshot: {snapshot.snapshot_id}")
                print(f"After snapshot: {after_snapshot.snapshot_id}")
            else:
                print("Learning cycle failed")
                
        except Exception as e:
            print(f"Error in learning cycle: {e}")
    
    def _trigger_auto_healing(self):
        """Trigger auto-healing if system is unstable."""
        try:
            success = self.healing_controller.auto_heal_system()
            
            if success:
                print("System auto-healed successfully")
            else:
                print("Auto-healing failed")
                
        except Exception as e:
            print(f"Error in auto-healing: {e}")
    
    def get_feedback_summary(self, session_id: str) -> Dict[str, Any]:
        """Get feedback summary for session."""
        return self.collector.get_feedback_summary(session_id)
    
    def get_system_learning_status(self) -> Dict[str, Any]:
        """Get system learning status."""
        learning_metrics = self.learning_engine.get_learning_metrics()
        system_status = self.healing_controller.get_system_status()
        
        return {
            "learning_metrics": learning_metrics.to_dict(),
            "system_status": system_status,
            "should_learn": self._should_trigger_learning(),
            "should_heal": self.healing_controller.should_trigger_rollback()
        }
    
    def update_feedback_prompt(self, prompt_config: Dict[str, Any]):
        """Update feedback prompt configuration."""
        if "prompt_text" in prompt_config:
            self.feedback_prompt.prompt_text = prompt_config["prompt_text"]
        
        if "rating_scale" in prompt_config:
            self.feedback_prompt.rating_scale = tuple(prompt_config["rating_scale"])
        
        if "include_text_feedback" in prompt_config:
            self.feedback_prompt.include_text_feedback = prompt_config["include_text_feedback"]
        
        if "include_personality_feedback" in prompt_config:
            self.feedback_prompt.include_personality_feedback = prompt_config["include_personality_feedback"]
        
        if "include_depth_feedback" in prompt_config:
            self.feedback_prompt.include_depth_feedback = prompt_config["include_depth_feedback"]
        
        if "include_confidence_feedback" in prompt_config:
            self.feedback_prompt.include_confidence_feedback = prompt_config["include_confidence_feedback"]
        
        if "timeout_seconds" in prompt_config:
            self.feedback_prompt.timeout_seconds = prompt_config["timeout_seconds"]
    
    def enable_auto_healing(self, enabled: bool = True):
        """Enable or disable auto-healing."""
        self.auto_heal_enabled = enabled
    
    def set_learning_threshold(self, threshold: int):
        """Set learning threshold."""
        self.learning_threshold = max(1, threshold)
    
    def get_learning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get learning history."""
        updates = self.storage.get_learning_history(limit)
        return [update.to_dict() for update in updates]
    
    def get_system_snapshots(self) -> List[Dict[str, Any]]:
        """Get system snapshots."""
        # This would need to be implemented in the healing controller
        return []
