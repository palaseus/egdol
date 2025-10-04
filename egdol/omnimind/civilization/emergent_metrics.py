"""
Emergent Intelligence Metrics
Real-time metrics dashboards tracking personality divergence, law formation, and strategic dominance.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import sqlite3
from collections import defaultdict, deque
import statistics
import numpy as np

from ..conversational.personality_framework import Personality, PersonalityType
from .multi_agent_system import CivilizationAgent, AgentMemory
from .world_persistence import WorldEvent, EventType, EventImportance


class MetricType(Enum):
    """Metric type enumeration."""
    PERSONALITY_DIVERGENCE = auto()
    LAW_FORMATION = auto()
    STRATEGIC_DOMINANCE = auto()
    TOOL_ADOPTION = auto()
    RESPONSE_SOPHISTICATION = auto()
    CULTURAL_EVOLUTION = auto()
    TECHNOLOGICAL_PROGRESS = auto()
    POLITICAL_STABILITY = auto()
    ECONOMIC_HEALTH = auto()
    ENVIRONMENTAL_IMPACT = auto()


class MetricTrend(Enum):
    """Metric trend enumeration."""
    RISING = auto()
    FALLING = auto()
    STABLE = auto()
    VOLATILE = auto()
    UNKNOWN = auto()


@dataclass
class MetricDataPoint:
    """Single metric data point."""
    metric_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_id": metric_id,
            "metric_type": metric_type.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class MetricTrendAnalysis:
    """Metric trend analysis."""
    metric_type: MetricType
    current_value: float
    previous_value: float
    trend: MetricTrend
    change_rate: float
    volatility: float
    confidence: float
    prediction: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.name,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "trend": self.trend.name,
            "change_rate": self.change_rate,
            "volatility": self.volatility,
            "confidence": self.confidence,
            "prediction": self.prediction,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EmergentPattern:
    """Emergent pattern in the system."""
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    participants: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    first_observed: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)
    frequency: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": self.confidence,
            "participants": self.participants,
            "evidence": self.evidence,
            "first_observed": self.first_observed.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "frequency": self.frequency
        }


class EmergentMetrics:
    """Emergent intelligence metrics system."""
    
    def __init__(self, db_path: str = "emergent_metrics.db"):
        self.db_path = db_path
        self.metrics: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.trends: Dict[MetricType, MetricTrendAnalysis] = {}
        self.patterns: Dict[str, EmergentPattern] = {}
        self.agents: Dict[str, CivilizationAgent] = {}
        self.world_events: List[WorldEvent] = []
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id TEXT PRIMARY KEY,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Trends table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trends (
                    metric_type TEXT PRIMARY KEY,
                    current_value REAL,
                    previous_value REAL,
                    trend TEXT,
                    change_rate REAL,
                    volatility REAL,
                    confidence REAL,
                    prediction REAL,
                    timestamp TEXT
                )
            """)
            
            # Patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    description TEXT,
                    confidence REAL,
                    participants TEXT,
                    evidence TEXT,
                    first_observed TEXT,
                    last_observed TEXT,
                    frequency INTEGER
                )
            """)
    
    def register_agents(self, agents: Dict[str, CivilizationAgent]):
        """Register agents for metrics tracking."""
        self.agents.update(agents)
    
    def add_world_events(self, events: List[WorldEvent]):
        """Add world events for metrics calculation."""
        self.world_events.extend(events)
    
    def calculate_personality_divergence(self) -> float:
        """Calculate personality divergence metric."""
        if len(self.agents) < 2:
            return 0.0
        
        # Get personality traits for all agents
        personality_vectors = []
        for agent in self.agents.values():
            # Use personality type and other attributes to create vector
            personality_type = agent.personality.personality_type.value
            archetype = len(agent.personality.archetype) / 10.0  # Normalize archetype length
            epistemic_style = len(agent.personality.epistemic_style) / 10.0  # Normalize style length
            
            vector = [
                personality_type / 10.0,  # Normalize personality type
                archetype,
                epistemic_style,
                0.5,  # Placeholder for missing traits
                0.5   # Placeholder for missing traits
            ]
            personality_vectors.append(vector)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(personality_vectors)):
            for j in range(i + 1, len(personality_vectors)):
                distance = np.linalg.norm(
                    np.array(personality_vectors[i]) - np.array(personality_vectors[j])
                )
                distances.append(distance)
        
        # Return average distance as divergence metric
        return statistics.mean(distances) if distances else 0.0
    
    def calculate_law_formation_rate(self) -> float:
        """Calculate law formation rate."""
        # Count law-related events in recent time window
        recent_events = [
            e for e in self.world_events
            if e.event_type == EventType.LAW_ENACTMENT
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return len(recent_events) / 24.0  # Events per hour
    
    def calculate_strategic_dominance(self, agent_id: str) -> float:
        """Calculate strategic dominance for specific agent."""
        if agent_id not in self.agents:
            return 0.0
        
        agent = self.agents[agent_id]
        memory = agent.memory
        
        # Calculate dominance based on relationships and achievements
        relationship_score = statistics.mean(memory.relationships.values()) if memory.relationships else 0.0
        achievement_score = len(memory.achievements) / 10.0  # Normalize
        expertise_score = statistics.mean(memory.expertise.values()) if memory.expertise else 0.0
        
        # Weighted combination
        dominance = (
            relationship_score * 0.4 +
            achievement_score * 0.3 +
            expertise_score * 0.3
        )
        
        return min(1.0, dominance)
    
    def calculate_tool_adoption_trend(self, tool_id: str, time_window: int = 24) -> float:
        """Calculate tool adoption trend."""
        # This would integrate with the Toolforge system
        # For now, return a placeholder value
        return 0.5
    
    def calculate_response_sophistication(self, agent_id: str) -> float:
        """Calculate response sophistication for agent."""
        if agent_id not in self.agents:
            return 0.0
        
        agent = self.agents[agent_id]
        
        # Calculate sophistication based on agent's communication patterns
        # This is a simplified metric - could be enhanced with NLP analysis
        sophistication = 0.5  # Base sophistication
        
        # Adjust based on agent's expertise
        if agent.memory.expertise:
            avg_expertise = statistics.mean(agent.memory.expertise.values())
            sophistication = min(1.0, sophistication + avg_expertise * 0.5)
        
        # Adjust based on agent's achievements
        achievement_bonus = len(agent.memory.achievements) * 0.1
        sophistication = min(1.0, sophistication + achievement_bonus)
        
        return sophistication
    
    def calculate_cultural_evolution(self) -> float:
        """Calculate cultural evolution metric."""
        # Count cultural events in recent time window
        recent_cultural_events = [
            e for e in self.world_events
            if e.event_type == EventType.CULTURAL_EVOLUTION
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return len(recent_cultural_events) / 24.0  # Events per hour
    
    def calculate_technological_progress(self) -> float:
        """Calculate technological progress metric."""
        # Count technology events in recent time window
        recent_tech_events = [
            e for e in self.world_events
            if e.event_type == EventType.TECHNOLOGICAL_BREAKTHROUGH
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return len(recent_tech_events) / 24.0  # Events per hour
    
    def calculate_political_stability(self) -> float:
        """Calculate political stability metric."""
        # Count conflict events vs. positive events
        recent_conflicts = [
            e for e in self.world_events
            if e.event_type == EventType.CONFLICT
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        recent_positive_events = [
            e for e in self.world_events
            if e.event_type in [EventType.LAW_ENACTMENT, EventType.TREATY_SIGNING]
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        conflict_rate = len(recent_conflicts) / 24.0
        positive_rate = len(recent_positive_events) / 24.0
        
        # Stability is inverse of conflict rate, boosted by positive events
        stability = max(0.0, 1.0 - conflict_rate + positive_rate * 0.5)
        return min(1.0, stability)
    
    def calculate_economic_health(self) -> float:
        """Calculate economic health metric."""
        # Count economic events
        recent_economic_events = [
            e for e in self.world_events
            if e.event_type == EventType.ECONOMIC_SHIFT
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        # Economic health based on frequency of economic activity
        economic_activity = len(recent_economic_events) / 24.0
        return min(1.0, economic_activity)
    
    def calculate_environmental_impact(self) -> float:
        """Calculate environmental impact metric."""
        # Count environmental events
        recent_env_events = [
            e for e in self.world_events
            if e.event_type == EventType.ENVIRONMENTAL_CHANGE
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        # Environmental impact based on frequency of environmental events
        env_activity = len(recent_env_events) / 24.0
        return min(1.0, env_activity)
    
    def update_metrics(self):
        """Update all metrics."""
        current_time = datetime.now()
        
        # Calculate all metrics
        metrics_data = {
            MetricType.PERSONALITY_DIVERGENCE: self.calculate_personality_divergence(),
            MetricType.LAW_FORMATION: self.calculate_law_formation_rate(),
            MetricType.CULTURAL_EVOLUTION: self.calculate_cultural_evolution(),
            MetricType.TECHNOLOGICAL_PROGRESS: self.calculate_technological_progress(),
            MetricType.POLITICAL_STABILITY: self.calculate_political_stability(),
            MetricType.ECONOMIC_HEALTH: self.calculate_economic_health(),
            MetricType.ENVIRONMENTAL_IMPACT: self.calculate_environmental_impact()
        }
        
        # Add agent-specific metrics
        for agent_id, agent in self.agents.items():
            strategic_dominance = self.calculate_strategic_dominance(agent_id)
            response_sophistication = self.calculate_response_sophistication(agent_id)
            
            # Store agent-specific metrics
            self._store_metric(
                MetricType.STRATEGIC_DOMINANCE,
                strategic_dominance,
                {"agent_id": agent_id}
            )
            self._store_metric(
                MetricType.RESPONSE_SOPHISTICATION,
                response_sophistication,
                {"agent_id": agent_id}
            )
        
        # Store general metrics
        for metric_type, value in metrics_data.items():
            self._store_metric(metric_type, value)
        
        # Calculate trends
        self._calculate_trends()
        
        # Detect patterns
        self._detect_patterns()
    
    def _store_metric(self, metric_type: MetricType, value: float, metadata: Dict[str, Any] = None):
        """Store metric data point."""
        metric_id = str(uuid.uuid4())
        data_point = MetricDataPoint(
            metric_id=metric_id,
            metric_type=metric_type,
            value=value,
            metadata=metadata or {}
        )
        
        # Store in memory
        self.metrics[metric_type].append(data_point)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics (metric_id, metric_type, value, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric_id,
                metric_type.name,
                value,
                data_point.timestamp.isoformat(),
                json.dumps(metadata or {})
            ))
    
    def _calculate_trends(self):
        """Calculate trends for all metrics."""
        for metric_type, data_points in self.metrics.items():
            if len(data_points) < 2:
                continue
            
            # Get recent data points
            recent_points = list(data_points)[-10:]  # Last 10 points
            values = [dp.value for dp in recent_points]
            
            # Calculate trend
            current_value = values[-1]
            previous_value = values[-2] if len(values) > 1 else current_value
            change_rate = (current_value - previous_value) / previous_value if previous_value != 0 else 0.0
            
            # Determine trend direction
            if change_rate > 0.05:
                trend = MetricTrend.RISING
            elif change_rate < -0.05:
                trend = MetricTrend.FALLING
            else:
                trend = MetricTrend.STABLE
            
            # Calculate volatility
            volatility = statistics.stdev(values) if len(values) > 1 else 0.0
            
            # Calculate confidence
            confidence = min(1.0, len(recent_points) / 10.0)
            
            # Simple prediction (linear extrapolation)
            prediction = None
            if len(values) >= 3:
                prediction = current_value + change_rate * current_value
            
            # Create trend object
            trend_obj = MetricTrendAnalysis(
                metric_type=metric_type,
                current_value=current_value,
                previous_value=previous_value,
                trend=trend,
                change_rate=change_rate,
                volatility=volatility,
                confidence=confidence,
                prediction=prediction
            )
            
            self.trends[metric_type] = trend_obj
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trends 
                    (metric_type, current_value, previous_value, trend, change_rate, 
                     volatility, confidence, prediction, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric_type.name,
                    current_value,
                    previous_value,
                    trend.name,
                    change_rate,
                    volatility,
                    confidence,
                    prediction,
                    trend_obj.timestamp.isoformat()
                ))
    
    def _detect_patterns(self):
        """Detect emergent patterns in the system."""
        # Simple pattern detection - could be enhanced with ML
        patterns_detected = []
        
        # Detect collaboration patterns
        collaboration_events = [
            e for e in self.world_events
            if e.event_type == EventType.TREATY_SIGNING
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        if len(collaboration_events) >= 3:
            pattern = EmergentPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="collaboration_spike",
                description="Increased collaboration activity detected",
                confidence=0.8,
                participants=[p for e in collaboration_events for p in e.participants],
                evidence=[e.event_id for e in collaboration_events],
                frequency=len(collaboration_events)
            )
            patterns_detected.append(pattern)
        
        # Detect conflict patterns
        conflict_events = [
            e for e in self.world_events
            if e.event_type == EventType.CONFLICT
            and e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        if len(conflict_events) >= 2:
            pattern = EmergentPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="conflict_escalation",
                description="Escalating conflict activity detected",
                confidence=0.7,
                participants=[p for e in conflict_events for p in e.participants],
                evidence=[e.event_id for e in conflict_events],
                frequency=len(conflict_events)
            )
            patterns_detected.append(pattern)
        
        # Store patterns
        for pattern in patterns_detected:
            self.patterns[pattern.pattern_id] = pattern
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO patterns 
                    (pattern_id, pattern_type, description, confidence, participants, 
                     evidence, first_observed, last_observed, frequency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    pattern.description,
                    pattern.confidence,
                    json.dumps(pattern.participants),
                    json.dumps(pattern.evidence),
                    pattern.first_observed.isoformat(),
                    pattern.last_observed.isoformat(),
                    pattern.frequency
                ))
    
    def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive metrics dashboard."""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "trends": {},
            "patterns": [],
            "summary": {}
        }
        
        # Get current metric values
        for metric_type, data_points in self.metrics.items():
            if data_points:
                latest_value = data_points[-1].value
                dashboard["metrics"][metric_type.name] = {
                    "current_value": latest_value,
                    "data_points": len(data_points),
                    "last_updated": data_points[-1].timestamp.isoformat()
                }
        
        # Get trends
        for metric_type, trend in self.trends.items():
            dashboard["trends"][metric_type.name] = trend.to_dict()
        
        # Get patterns
        dashboard["patterns"] = [pattern.to_dict() for pattern in self.patterns.values()]
        
        # Generate summary
        dashboard["summary"] = self._generate_summary()
        
        return dashboard
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of system state."""
        summary = {
            "overall_health": 0.0,
            "key_insights": [],
            "recommendations": []
        }
        
        # Calculate overall health
        health_metrics = [
            MetricType.POLITICAL_STABILITY,
            MetricType.ECONOMIC_HEALTH,
            MetricType.CULTURAL_EVOLUTION,
            MetricType.TECHNOLOGICAL_PROGRESS
        ]
        
        health_values = []
        for metric_type in health_metrics:
            if metric_type in self.metrics and self.metrics[metric_type]:
                health_values.append(self.metrics[metric_type][-1].value)
        
        if health_values:
            summary["overall_health"] = statistics.mean(health_values)
        
        # Generate insights
        if summary["overall_health"] > 0.8:
            summary["key_insights"].append("System is in excellent health")
        elif summary["overall_health"] > 0.6:
            summary["key_insights"].append("System is in good health")
        else:
            summary["key_insights"].append("System health needs attention")
        
        # Generate recommendations
        if summary["overall_health"] < 0.6:
            summary["recommendations"].append("Consider intervention to improve system health")
        
        return summary
