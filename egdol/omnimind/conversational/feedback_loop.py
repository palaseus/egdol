"""
Feedback Loop & Meta-Learning System
Captures user feedback, analyzes trends, and adapts system behavior.
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import sqlite3
from pathlib import Path


class FeedbackType(Enum):
    """Types of feedback."""
    RATING = auto()
    TEXT = auto()
    BEHAVIORAL = auto()
    CORRECTION = auto()


class FeedbackSentiment(Enum):
    """Feedback sentiment."""
    POSITIVE = auto()
    NEUTRAL = auto()
    NEGATIVE = auto()


@dataclass
class UserFeedback:
    """Structured user feedback."""
    feedback_id: str
    session_id: str
    response_id: str
    feedback_type: FeedbackType
    rating: Optional[float] = None  # 1.0 to 5.0 scale
    text: Optional[str] = None
    sentiment: Optional[FeedbackSentiment] = None
    personality: Optional[str] = None
    depth_preference: Optional[str] = None
    confidence_feedback: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "session_id": self.session_id,
            "response_id": self.response_id,
            "feedback_type": self.feedback_type.name,
            "rating": self.rating,
            "text": self.text,
            "sentiment": self.sentiment.name if self.sentiment else None,
            "personality": self.personality,
            "depth_preference": self.depth_preference,
            "confidence_feedback": self.confidence_feedback,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class LearningUpdate:
    """Meta-learning update."""
    update_id: str
    feedback_batch: List[UserFeedback]
    personality_adjustments: Dict[str, Dict[str, float]]
    depth_preference_adjustments: Dict[str, float]
    confidence_calibration_adjustments: Dict[str, float]
    synthesis_improvements: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    snapshot_before: str = ""
    snapshot_after: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "update_id": self.update_id,
            "feedback_batch_size": len(self.feedback_batch),
            "personality_adjustments": self.personality_adjustments,
            "depth_preference_adjustments": self.depth_preference_adjustments,
            "confidence_calibration_adjustments": self.confidence_calibration_adjustments,
            "synthesis_improvements": self.synthesis_improvements,
            "timestamp": self.timestamp.isoformat(),
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after
        }


class FeedbackStorage:
    """Persistent storage for feedback and learning updates."""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    rating REAL,
                    text TEXT,
                    sentiment TEXT,
                    personality TEXT,
                    depth_preference TEXT,
                    confidence_feedback REAL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_updates (
                    update_id TEXT PRIMARY KEY,
                    feedback_batch TEXT NOT NULL,
                    personality_adjustments TEXT NOT NULL,
                    depth_preference_adjustments TEXT NOT NULL,
                    confidence_calibration_adjustments TEXT NOT NULL,
                    synthesis_improvements TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    snapshot_before TEXT,
                    snapshot_after TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    system_state TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    is_stable BOOLEAN DEFAULT TRUE
                )
            """)
    
    def store_feedback(self, feedback: UserFeedback):
        """Store user feedback."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feedback 
                (feedback_id, session_id, response_id, feedback_type, rating, text, 
                 sentiment, personality, depth_preference, confidence_feedback, 
                 timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_id,
                feedback.session_id,
                feedback.response_id,
                feedback.feedback_type.name,
                feedback.rating,
                feedback.text,
                feedback.sentiment.name if feedback.sentiment else None,
                feedback.personality,
                feedback.depth_preference,
                feedback.confidence_feedback,
                feedback.timestamp.isoformat(),
                json.dumps(feedback.metadata)
            ))
    
    def store_learning_update(self, update: LearningUpdate):
        """Store learning update."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO learning_updates 
                (update_id, feedback_batch, personality_adjustments, 
                 depth_preference_adjustments, confidence_calibration_adjustments,
                 synthesis_improvements, timestamp, snapshot_before, snapshot_after)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                update.update_id,
                json.dumps([f.to_dict() for f in update.feedback_batch]),
                json.dumps(update.personality_adjustments),
                json.dumps(update.depth_preference_adjustments),
                json.dumps(update.confidence_calibration_adjustments),
                json.dumps(update.synthesis_improvements),
                update.timestamp.isoformat(),
                update.snapshot_before,
                update.snapshot_after
            ))
    
    def get_feedback_batch(self, limit: int = 50) -> List[UserFeedback]:
        """Get recent feedback for learning."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            feedback_list = []
            for row in cursor.fetchall():
                feedback = UserFeedback(
                    feedback_id=row[0],
                    session_id=row[1],
                    response_id=row[2],
                    feedback_type=FeedbackType[row[3]],
                    rating=row[4],
                    text=row[5],
                    sentiment=FeedbackSentiment[row[6]] if row[6] else None,
                    personality=row[7],
                    depth_preference=row[8],
                    confidence_feedback=row[9],
                    timestamp=datetime.fromisoformat(row[10]),
                    metadata=json.loads(row[11]) if row[11] else {}
                )
                feedback_list.append(feedback)
            
            return feedback_list
    
    def get_learning_history(self, limit: int = 20) -> List[LearningUpdate]:
        """Get learning update history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM learning_updates 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            updates = []
            for row in cursor.fetchall():
                update = LearningUpdate(
                    update_id=row[0],
                    feedback_batch=[],  # Will be loaded separately
                    personality_adjustments=json.loads(row[2]),
                    depth_preference_adjustments=json.loads(row[3]),
                    confidence_calibration_adjustments=json.loads(row[4]),
                    synthesis_improvements=json.loads(row[5]),
                    timestamp=datetime.fromisoformat(row[6]),
                    snapshot_before=row[7],
                    snapshot_after=row[8]
                )
                updates.append(update)
            
            return updates


class FeedbackCollector:
    """Collects and processes user feedback."""
    
    def __init__(self, storage: FeedbackStorage):
        self.storage = storage
        self.session_feedback = {}  # In-memory session tracking
    
    def collect_feedback(self, 
                        session_id: str,
                        response_id: str,
                        feedback_data: Dict[str, Any]) -> UserFeedback:
        """Collect and store user feedback."""
        
        # Generate feedback ID
        feedback_id = self._generate_feedback_id(session_id, response_id)
        
        # Parse feedback data
        feedback_type = self._determine_feedback_type(feedback_data)
        sentiment = self._analyze_sentiment(feedback_data)
        
        # Create feedback object
        feedback = UserFeedback(
            feedback_id=feedback_id,
            session_id=session_id,
            response_id=response_id,
            feedback_type=feedback_type,
            rating=feedback_data.get('rating'),
            text=feedback_data.get('text'),
            sentiment=sentiment,
            personality=feedback_data.get('personality'),
            depth_preference=feedback_data.get('depth_preference'),
            confidence_feedback=feedback_data.get('confidence_feedback'),
            metadata=feedback_data.get('metadata', {})
        )
        
        # Store feedback
        self.storage.store_feedback(feedback)
        
        # Track in session
        if session_id not in self.session_feedback:
            self.session_feedback[session_id] = []
        self.session_feedback[session_id].append(feedback)
        
        return feedback
    
    def _generate_feedback_id(self, session_id: str, response_id: str) -> str:
        """Generate unique feedback ID."""
        data = f"{session_id}_{response_id}_{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _determine_feedback_type(self, feedback_data: Dict[str, Any]) -> FeedbackType:
        """Determine feedback type from data."""
        if 'rating' in feedback_data:
            return FeedbackType.RATING
        elif 'text' in feedback_data:
            return FeedbackType.TEXT
        elif 'correction' in feedback_data:
            return FeedbackType.CORRECTION
        else:
            return FeedbackType.BEHAVIORAL
    
    def _analyze_sentiment(self, feedback_data: Dict[str, Any]) -> Optional[FeedbackSentiment]:
        """Analyze sentiment from feedback."""
        rating = feedback_data.get('rating')
        text = feedback_data.get('text', '').lower()
        
        if rating is not None:
            if rating >= 4.0:
                return FeedbackSentiment.POSITIVE
            elif rating <= 2.0:
                return FeedbackSentiment.NEGATIVE
            else:
                return FeedbackSentiment.NEUTRAL
        
        # Analyze text sentiment
        positive_words = ['good', 'great', 'excellent', 'helpful', 'accurate', 'clear']
        negative_words = ['bad', 'wrong', 'incorrect', 'confusing', 'unclear', 'poor', 'shallow']
        
        # Check for negative words first (more specific)
        if any(word in text for word in negative_words):
            return FeedbackSentiment.NEGATIVE
        elif any(word in text for word in positive_words):
            return FeedbackSentiment.POSITIVE
        else:
            return FeedbackSentiment.NEUTRAL
    
    def get_session_feedback(self, session_id: str) -> List[UserFeedback]:
        """Get feedback for a specific session."""
        return self.session_feedback.get(session_id, [])
    
    def get_feedback_summary(self, session_id: str) -> Dict[str, Any]:
        """Get feedback summary for session."""
        feedback_list = self.get_session_feedback(session_id)
        
        if not feedback_list:
            return {"total": 0, "average_rating": 0.0, "sentiment_distribution": {}}
        
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        sentiments = [f.sentiment for f in feedback_list if f.sentiment is not None]
        
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment.name] = sentiment_counts.get(sentiment.name, 0) + 1
        
        return {
            "total": len(feedback_list),
            "average_rating": sum(ratings) / len(ratings) if ratings else 0.0,
            "sentiment_distribution": sentiment_counts,
            "recent_feedback": [f.to_dict() for f in feedback_list[-5:]]  # Last 5
        }
