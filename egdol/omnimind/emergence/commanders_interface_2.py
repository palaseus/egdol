"""
Commander's Strategic Interface 2.0
Upgraded console from "control panel" to strategic observatory.
Real-time visualization of civilization clusters, agent debates, tool evolution, and codex changes.
Interactive timeline scrubbing and live injection of high-order perturbations.
"""

import uuid
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import sqlite3
from collections import defaultdict, deque
import statistics
import numpy as np

from ..civilization.multi_agent_system import CivilizationState, CivilizationAgent
from ..civilization.world_persistence import WorldEvent, EventType, EventImportance
from .domain_proposal_engine import DomainProposal, ProposalStatus
from .cross_civilization_interaction import InteractionProposal, InteractionStatus
from .emergent_law_codex import UniversalLaw, LawStatus
from .system_evolution_orchestrator import RefactorProposal, RefactorStatus


class ObservationMode(Enum):
    """Modes of strategic observation."""
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    PREDICTIVE = "predictive"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"


class PerturbationType(Enum):
    """Types of high-order perturbations."""
    BLACK_SWAN_EVENT = "black_swan_event"
    PARADIGM_SHIFT = "paradigm_shift"
    RESOURCE_SHOCK = "resource_shock"
    TECHNOLOGICAL_BREAKTHROUGH = "technological_breakthrough"
    CULTURAL_REVOLUTION = "cultural_revolution"
    POLITICAL_UPHEAVAL = "political_upheaval"
    ENVIRONMENTAL_CRISIS = "environmental_crisis"
    EXTERNAL_THREAT = "external_threat"


class VisualizationType(Enum):
    """Types of visualizations."""
    CIVILIZATION_CLUSTERS = "civilization_clusters"
    AGENT_DEBATES = "agent_debates"
    TOOL_EVOLUTION = "tool_evolution"
    CODEX_CHANGES = "codex_changes"
    INTERACTION_NETWORKS = "interaction_networks"
    PERFORMANCE_METRICS = "performance_metrics"
    EMERGENT_PATTERNS = "emergent_patterns"


@dataclass
class StrategicObservation:
    """A strategic observation of the system."""
    observation_id: str
    observation_type: str
    timestamp: datetime
    data: Dict[str, Any]
    insights: List[str] = field(default_factory=list)
    significance: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "observation_id": self.observation_id,
            "observation_type": self.observation_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "insights": self.insights,
            "significance": self.significance,
            "confidence": self.confidence
        }


@dataclass
class TimelineEvent:
    """An event in the system timeline."""
    event_id: str
    timestamp: datetime
    event_type: str
    description: str
    participants: List[str] = field(default_factory=list)
    impact_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "participants": self.participants,
            "impact_level": self.impact_level,
            "metadata": self.metadata
        }


@dataclass
class PerturbationEvent:
    """A high-order perturbation event."""
    perturbation_id: str
    perturbation_type: PerturbationType
    description: str
    intensity: float  # 0.0 to 1.0
    duration: timedelta
    affected_systems: List[str] = field(default_factory=list)
    expected_impact: Dict[str, float] = field(default_factory=dict)
    injection_timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "perturbation_id": self.perturbation_id,
            "perturbation_type": self.perturbation_type.value,
            "description": self.description,
            "intensity": self.intensity,
            "duration": str(self.duration),
            "affected_systems": self.affected_systems,
            "expected_impact": self.expected_impact,
            "injection_timestamp": self.injection_timestamp.isoformat(),
            "status": self.status
        }


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    viz_type: VisualizationType
    data_sources: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    time_range: Optional[Tuple[datetime, datetime]] = None
    update_frequency: timedelta = timedelta(seconds=5)
    display_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "viz_type": self.viz_type.value,
            "data_sources": self.data_sources,
            "filters": self.filters,
            "time_range": [self.time_range[0].isoformat(), self.time_range[1].isoformat()] if self.time_range else None,
            "update_frequency": str(self.update_frequency),
            "display_options": self.display_options
        }


class CommandersInterface2:
    """Commander's Strategic Interface 2.0 - Strategic Observatory."""
    
    def __init__(self, db_path: str = "commanders_interface_2.db"):
        self.db_path = db_path
        self.observations: List[StrategicObservation] = []
        self.timeline_events: List[TimelineEvent] = []
        self.perturbation_events: List[PerturbationEvent] = []
        self.visualization_configs: Dict[str, VisualizationConfig] = {}
        self.active_observations: Set[str] = set()
        self.observation_mode: ObservationMode = ObservationMode.REAL_TIME
        self.current_timeline_position: Optional[datetime] = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create observations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategic_observations (
                observation_id TEXT PRIMARY KEY,
                observation_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                insights TEXT,
                significance REAL,
                confidence REAL
            )
        ''')
        
        # Create timeline events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timeline_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                participants TEXT,
                impact_level REAL,
                metadata TEXT
            )
        ''')
        
        # Create perturbation events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS perturbation_events (
                perturbation_id TEXT PRIMARY KEY,
                perturbation_type TEXT NOT NULL,
                description TEXT NOT NULL,
                intensity REAL NOT NULL,
                duration TEXT NOT NULL,
                affected_systems TEXT,
                expected_impact TEXT,
                injection_timestamp TEXT NOT NULL,
                status TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def observe_civilization_clusters(self, civilizations: Dict[str, CivilizationState]) -> StrategicObservation:
        """Observe civilization clusters and their interactions."""
        observation_id = str(uuid.uuid4())
        
        # Analyze civilization clusters
        cluster_data = self._analyze_civilization_clusters(civilizations)
        
        # Generate insights
        insights = self._generate_cluster_insights(cluster_data)
        
        observation = StrategicObservation(
            observation_id=observation_id,
            observation_type="civilization_clusters",
            timestamp=datetime.now(),
            data=cluster_data,
            insights=insights,
            significance=self._calculate_cluster_significance(cluster_data),
            confidence=0.8
        )
        
        self.observations.append(observation)
        self._save_observation(observation)
        
        return observation
    
    def _analyze_civilization_clusters(self, civilizations: Dict[str, CivilizationState]) -> Dict[str, Any]:
        """Analyze civilization clusters."""
        total_civilizations = len(civilizations)
        
        # Calculate cluster metrics
        cluster_metrics = {
            "total_civilizations": total_civilizations,
            "average_agent_count": statistics.mean([len(civ.agents) for civ in civilizations.values()]) if civilizations else 0,
            "total_agents": sum(len(civ.agents) for civ in civilizations.values()),
            "law_count": sum(len(civ.laws) for civ in civilizations.values()),
            "technology_count": sum(len(civ.technologies) for civ in civilizations.values()),
            "cultural_traits": sum(len(civ.culture) for civ in civilizations.values())
        }
        
        # Identify dominant civilizations
        agent_counts = [(civ_id, len(civ.agents)) for civ_id, civ in civilizations.items()]
        agent_counts.sort(key=lambda x: x[1], reverse=True)
        
        cluster_metrics["dominant_civilizations"] = [
            {"civilization_id": civ_id, "agent_count": count}
            for civ_id, count in agent_counts[:3]
        ]
        
        # Calculate diversity metrics
        cluster_metrics["diversity_score"] = self._calculate_diversity_score(civilizations)
        
        return cluster_metrics
    
    def _calculate_diversity_score(self, civilizations: Dict[str, CivilizationState]) -> float:
        """Calculate diversity score of civilizations."""
        if not civilizations:
            return 0.0

        # Calculate diversity based on different attributes
        agent_counts = [len(civ.agents) for civ in civilizations.values()]
        law_counts = [len(civ.laws) for civ in civilizations.values()]
        tech_counts = [len(civ.technologies) for civ in civilizations.values()]

        # Calculate coefficient of variation (standard deviation / mean)
        # Need at least 2 data points for variance calculation
        agent_diversity = 0.0
        if len(agent_counts) >= 2 and statistics.mean(agent_counts) > 0:
            try:
                agent_diversity = statistics.stdev(agent_counts) / statistics.mean(agent_counts)
            except statistics.StatisticsError:
                agent_diversity = 0.0
        
        law_diversity = 0.0
        if len(law_counts) >= 2 and statistics.mean(law_counts) > 0:
            try:
                law_diversity = statistics.stdev(law_counts) / statistics.mean(law_counts)
            except statistics.StatisticsError:
                law_diversity = 0.0
        
        tech_diversity = 0.0
        if len(tech_counts) >= 2 and statistics.mean(tech_counts) > 0:
            try:
                tech_diversity = statistics.stdev(tech_counts) / statistics.mean(tech_counts)
            except statistics.StatisticsError:
                tech_diversity = 0.0

        return (agent_diversity + law_diversity + tech_diversity) / 3.0
    
    def _generate_cluster_insights(self, cluster_data: Dict[str, Any]) -> List[str]:
        """Generate insights from cluster analysis."""
        insights = []
        
        if cluster_data["total_civilizations"] > 5:
            insights.append("Multiple civilization clusters detected - potential for inter-civilization dynamics")
        
        if cluster_data["diversity_score"] > 0.5:
            insights.append("High diversity in civilization characteristics - rich ecosystem")
        elif cluster_data["diversity_score"] < 0.2:
            insights.append("Low diversity - civilizations may be converging")
        
        dominant_civs = cluster_data["dominant_civilizations"]
        if len(dominant_civs) > 0:
            top_civ = dominant_civs[0]
            if top_civ["agent_count"] > cluster_data["average_agent_count"] * 2:
                insights.append(f"Dominant civilization detected with {top_civ['agent_count']} agents")
        
        return insights
    
    def _calculate_cluster_significance(self, cluster_data: Dict[str, Any]) -> float:
        """Calculate significance of cluster observation."""
        # Factors: number of civilizations, diversity, dominance
        civ_factor = min(1.0, cluster_data["total_civilizations"] / 10.0)
        diversity_factor = cluster_data["diversity_score"]
        dominance_factor = len(cluster_data["dominant_civilizations"]) / max(1, cluster_data["total_civilizations"])
        
        return (civ_factor + diversity_factor + dominance_factor) / 3.0
    
    def observe_agent_debates(self, agents: Dict[str, CivilizationAgent]) -> StrategicObservation:
        """Observe agent debates and interactions."""
        observation_id = str(uuid.uuid4())
        
        # Analyze agent interactions
        debate_data = self._analyze_agent_debates(agents)
        
        # Generate insights
        insights = self._generate_debate_insights(debate_data)
        
        observation = StrategicObservation(
            observation_id=observation_id,
            observation_type="agent_debates",
            timestamp=datetime.now(),
            data=debate_data,
            insights=insights,
            significance=self._calculate_debate_significance(debate_data),
            confidence=0.7
        )
        
        self.observations.append(observation)
        self._save_observation(observation)
        
        return observation
    
    def _analyze_agent_debates(self, agents: Dict[str, CivilizationAgent]) -> Dict[str, Any]:
        """Analyze agent debates and interactions."""
        total_agents = len(agents)
        
        # Analyze personality distribution
        personality_counts = defaultdict(int)
        for agent in agents.values():
            personality_counts[agent.personality.personality_type.value] += 1
        
        # Calculate debate metrics
        debate_metrics = {
            "total_agents": total_agents,
            "personality_distribution": dict(personality_counts),
            "debate_intensity": self._calculate_debate_intensity(agents),
            "consensus_level": self._calculate_consensus_level(agents),
            "conflict_level": self._calculate_conflict_level(agents)
        }
        
        return debate_metrics
    
    def _calculate_debate_intensity(self, agents: Dict[str, CivilizationAgent]) -> float:
        """Calculate intensity of agent debates."""
        # Simplified calculation based on agent count and personality diversity
        total_agents = len(agents)
        personality_types = len(set(agent.personality.personality_type for agent in agents.values()))
        
        return min(1.0, (total_agents * personality_types) / 20.0)
    
    def _calculate_consensus_level(self, agents: Dict[str, CivilizationAgent]) -> float:
        """Calculate consensus level among agents."""
        # Simplified: based on personality type distribution
        personality_counts = defaultdict(int)
        for agent in agents.values():
            personality_counts[agent.personality.personality_type.value] += 1
        
        if not personality_counts:
            return 0.0
        
        # Calculate entropy (higher entropy = lower consensus)
        total = sum(personality_counts.values())
        entropy = 0.0
        for count in personality_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Convert entropy to consensus (0 = no consensus, 1 = full consensus)
        max_entropy = np.log2(len(personality_counts))
        consensus = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        return consensus
    
    def _calculate_conflict_level(self, agents: Dict[str, CivilizationAgent]) -> float:
        """Calculate conflict level among agents."""
        # Simplified: based on personality type diversity
        personality_types = set(agent.personality.personality_type for agent in agents.values())
        
        # More diverse personality types = higher potential for conflict
        return min(1.0, len(personality_types) / 4.0)
    
    def _generate_debate_insights(self, debate_data: Dict[str, Any]) -> List[str]:
        """Generate insights from debate analysis."""
        insights = []
        
        if debate_data["debate_intensity"] > 0.7:
            insights.append("High debate intensity detected - active intellectual discourse")
        
        if debate_data["consensus_level"] > 0.8:
            insights.append("High consensus level - agents are converging on solutions")
        elif debate_data["consensus_level"] < 0.3:
            insights.append("Low consensus level - agents have divergent views")
        
        if debate_data["conflict_level"] > 0.6:
            insights.append("High conflict level - potential for resolution needed")
        
        personality_dist = debate_data["personality_distribution"]
        if len(personality_dist) > 1:
            insights.append(f"Multi-personality debate with {len(personality_dist)} different types")
        
        return insights
    
    def _calculate_debate_significance(self, debate_data: Dict[str, Any]) -> float:
        """Calculate significance of debate observation."""
        intensity_factor = debate_data["debate_intensity"]
        consensus_factor = 1.0 - debate_data["consensus_level"]  # Higher conflict = more significant
        conflict_factor = debate_data["conflict_level"]
        
        return (intensity_factor + consensus_factor + conflict_factor) / 3.0
    
    def observe_tool_evolution(self, tool_data: Dict[str, Any]) -> StrategicObservation:
        """Observe tool evolution and development."""
        observation_id = str(uuid.uuid4())
        
        # Analyze tool evolution
        evolution_data = self._analyze_tool_evolution(tool_data)
        
        # Generate insights
        insights = self._generate_tool_insights(evolution_data)
        
        observation = StrategicObservation(
            observation_id=observation_id,
            observation_type="tool_evolution",
            timestamp=datetime.now(),
            data=evolution_data,
            insights=insights,
            significance=self._calculate_tool_significance(evolution_data),
            confidence=0.6
        )
        
        self.observations.append(observation)
        self._save_observation(observation)
        
        return observation
    
    def _analyze_tool_evolution(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tool evolution patterns."""
        return {
            "total_tools": tool_data.get("total_tools", 0),
            "active_tools": tool_data.get("active_tools", 0),
            "development_rate": tool_data.get("development_rate", 0.0),
            "adoption_rate": tool_data.get("adoption_rate", 0.0),
            "tool_categories": tool_data.get("categories", {}),
            "evolution_trends": tool_data.get("trends", [])
        }
    
    def _generate_tool_insights(self, evolution_data: Dict[str, Any]) -> List[str]:
        """Generate insights from tool evolution."""
        insights = []
        
        if evolution_data["development_rate"] > 0.5:
            insights.append("High tool development rate - rapid innovation")
        
        if evolution_data["adoption_rate"] > 0.8:
            insights.append("High tool adoption rate - effective tool development")
        elif evolution_data["adoption_rate"] < 0.3:
            insights.append("Low tool adoption rate - tools may not meet needs")
        
        if evolution_data["total_tools"] > 10:
            insights.append("Rich tool ecosystem - diverse capabilities")
        
        return insights
    
    def _calculate_tool_significance(self, evolution_data: Dict[str, Any]) -> float:
        """Calculate significance of tool evolution."""
        development_factor = evolution_data["development_rate"]
        adoption_factor = evolution_data["adoption_rate"]
        ecosystem_factor = min(1.0, evolution_data["total_tools"] / 20.0)
        
        return (development_factor + adoption_factor + ecosystem_factor) / 3.0
    
    def observe_codex_changes(self, codex_data: Dict[str, Any]) -> StrategicObservation:
        """Observe changes in the universal codex."""
        observation_id = str(uuid.uuid4())
        
        # Analyze codex changes
        change_data = self._analyze_codex_changes(codex_data)
        
        # Generate insights
        insights = self._generate_codex_insights(change_data)
        
        observation = StrategicObservation(
            observation_id=observation_id,
            observation_type="codex_changes",
            timestamp=datetime.now(),
            data=change_data,
            insights=insights,
            significance=self._calculate_codex_significance(change_data),
            confidence=0.9
        )
        
        self.observations.append(observation)
        self._save_observation(observation)
        
        return observation
    
    def _analyze_codex_changes(self, codex_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze codex changes."""
        return {
            "total_laws": codex_data.get("total_laws", 0),
            "active_laws": codex_data.get("active_laws", 0),
            "recent_changes": codex_data.get("recent_changes", []),
            "law_types": codex_data.get("law_types", {}),
            "effectiveness_score": codex_data.get("effectiveness_score", 0.0),
            "adoption_rate": codex_data.get("adoption_rate", 0.0)
        }
    
    def _generate_codex_insights(self, change_data: Dict[str, Any]) -> List[str]:
        """Generate insights from codex changes."""
        insights = []
        
        if change_data["effectiveness_score"] > 0.8:
            insights.append("High codex effectiveness - laws are working well")
        elif change_data["effectiveness_score"] < 0.4:
            insights.append("Low codex effectiveness - laws may need revision")
        
        if change_data["adoption_rate"] > 0.7:
            insights.append("High law adoption rate - universal acceptance")
        elif change_data["adoption_rate"] < 0.3:
            insights.append("Low law adoption rate - resistance to universal laws")
        
        if len(change_data["recent_changes"]) > 5:
            insights.append("Active law evolution - rapid legal development")
        
        return insights
    
    def _calculate_codex_significance(self, change_data: Dict[str, Any]) -> float:
        """Calculate significance of codex changes."""
        effectiveness_factor = change_data["effectiveness_score"]
        adoption_factor = change_data["adoption_rate"]
        activity_factor = min(1.0, len(change_data["recent_changes"]) / 10.0)
        
        return (effectiveness_factor + adoption_factor + activity_factor) / 3.0
    
    def inject_perturbation(self,
                          perturbation_type: PerturbationType,
                          description: str,
                          intensity: float,
                          duration: timedelta,
                          affected_systems: List[str] = None) -> str:
        """Inject a high-order perturbation into the system."""
        perturbation_id = str(uuid.uuid4())
        
        # Calculate expected impact
        expected_impact = self._calculate_perturbation_impact(perturbation_type, intensity, affected_systems or [])
        
        perturbation = PerturbationEvent(
            perturbation_id=perturbation_id,
            perturbation_type=perturbation_type,
            description=description,
            intensity=intensity,
            duration=duration,
            affected_systems=affected_systems or [],
            expected_impact=expected_impact
        )
        
        self.perturbation_events.append(perturbation)
        self._save_perturbation(perturbation)
        
        # Add to timeline
        self._add_timeline_event(
            event_type="perturbation_injection",
            description=f"Injected {perturbation_type.value}: {description}",
            participants=affected_systems or [],
            impact_level=intensity
        )
        
        return perturbation_id
    
    def _calculate_perturbation_impact(self, perturbation_type: PerturbationType, intensity: float, affected_systems: List[str]) -> Dict[str, float]:
        """Calculate expected impact of perturbation."""
        base_impact = {
            "system_stability": -intensity * 0.8,
            "innovation_rate": intensity * 0.6,
            "adaptation_pressure": intensity * 0.9,
            "cooperation_level": -intensity * 0.4,
            "conflict_level": intensity * 0.7
        }
        
        # Adjust based on perturbation type
        if perturbation_type == PerturbationType.BLACK_SWAN_EVENT:
            base_impact["system_stability"] *= 1.5
            base_impact["adaptation_pressure"] *= 1.3
        elif perturbation_type == PerturbationType.PARADIGM_SHIFT:
            base_impact["innovation_rate"] *= 1.4
            base_impact["adaptation_pressure"] *= 1.2
        elif perturbation_type == PerturbationType.TECHNOLOGICAL_BREAKTHROUGH:
            base_impact["innovation_rate"] *= 1.6
            base_impact["system_stability"] *= 0.8
        
        return base_impact
    
    def scrub_timeline(self, target_time: datetime) -> List[TimelineEvent]:
        """Scrub timeline to a specific point in time."""
        self.current_timeline_position = target_time
        
        # Filter events up to target time
        timeline_events = [event for event in self.timeline_events if event.timestamp <= target_time]
        
        # Sort by timestamp
        timeline_events.sort(key=lambda x: x.timestamp)
        
        return timeline_events
    
    def get_timeline_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get summary of timeline events in a time range."""
        events_in_range = [
            event for event in self.timeline_events
            if start_time <= event.timestamp <= end_time
        ]
        
        # Categorize events
        event_types = defaultdict(int)
        for event in events_in_range:
            event_types[event.event_type] += 1
        
        # Calculate impact
        total_impact = sum(event.impact_level for event in events_in_range)
        average_impact = total_impact / len(events_in_range) if events_in_range else 0.0
        
        return {
            "total_events": len(events_in_range),
            "event_types": dict(event_types),
            "total_impact": total_impact,
            "average_impact": average_impact,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration": str(end_time - start_time)
            }
        }
    
    def set_observation_mode(self, mode: ObservationMode):
        """Set the observation mode."""
        self.observation_mode = mode
        print(f"🔍 Observation mode set to: {mode.value}")
    
    def create_visualization(self, viz_type: VisualizationType, config: VisualizationConfig) -> str:
        """Create a new visualization."""
        viz_id = str(uuid.uuid4())
        self.visualization_configs[viz_id] = config
        
        print(f"📊 Created visualization: {viz_type.value}")
        print(f"   Data sources: {config.data_sources}")
        print(f"   Update frequency: {config.update_frequency}")
        
        return viz_id
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard."""
        recent_observations = [obs for obs in self.observations if (datetime.now() - obs.timestamp).seconds < 300]  # Last 5 minutes
        
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "observation_mode": self.observation_mode.value,
            "timeline_position": self.current_timeline_position.isoformat() if self.current_timeline_position else None,
            "recent_observations": len(recent_observations),
            "total_observations": len(self.observations),
            "active_perturbations": len([p for p in self.perturbation_events if p.status == "active"]),
            "timeline_events": len(self.timeline_events),
            "visualizations": len(self.visualization_configs),
            "system_health": self._calculate_system_health(),
            "key_insights": self._extract_key_insights()
        }
        
        return dashboard
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health."""
        if not self.observations:
            return 0.5
        
        # Calculate health based on recent observations
        recent_obs = [obs for obs in self.observations if (datetime.now() - obs.timestamp).seconds < 600]  # Last 10 minutes
        
        if not recent_obs:
            return 0.5
        
        # Factors: observation frequency, significance, confidence
        frequency_score = min(1.0, len(recent_obs) / 10.0)
        significance_score = statistics.mean([obs.significance for obs in recent_obs])
        confidence_score = statistics.mean([obs.confidence for obs in recent_obs])
        
        return (frequency_score + significance_score + confidence_score) / 3.0
    
    def _extract_key_insights(self) -> List[str]:
        """Extract key insights from recent observations."""
        recent_obs = [obs for obs in self.observations if (datetime.now() - obs.timestamp).seconds < 600]  # Last 10 minutes
        
        if not recent_obs:
            return ["No recent observations"]
        
        # Get insights from most significant observations
        significant_obs = [obs for obs in recent_obs if obs.significance > 0.7]
        
        if not significant_obs:
            return ["No significant recent observations"]
        
        # Extract unique insights
        all_insights = []
        for obs in significant_obs:
            all_insights.extend(obs.insights)
        
        # Return top insights (simplified)
        return all_insights[:5]
    
    def _add_timeline_event(self, event_type: str, description: str, participants: List[str] = None, impact_level: float = 0.0):
        """Add event to timeline."""
        event = TimelineEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            description=description,
            participants=participants or [],
            impact_level=impact_level
        )
        
        self.timeline_events.append(event)
        self._save_timeline_event(event)
    
    def _save_observation(self, observation: StrategicObservation):
        """Save observation to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO strategic_observations 
            (observation_id, observation_type, timestamp, data, insights, significance, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            observation.observation_id, observation.observation_type,
            observation.timestamp.isoformat(), json.dumps(observation.data),
            json.dumps(observation.insights), observation.significance, observation.confidence
        ))
        
        conn.commit()
        conn.close()
    
    def _save_timeline_event(self, event: TimelineEvent):
        """Save timeline event to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO timeline_events 
            (event_id, timestamp, event_type, description, participants, impact_level, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id, event.timestamp.isoformat(), event.event_type,
            event.description, json.dumps(event.participants), event.impact_level,
            json.dumps(event.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_perturbation(self, perturbation: PerturbationEvent):
        """Save perturbation to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO perturbation_events 
            (perturbation_id, perturbation_type, description, intensity, duration,
             affected_systems, expected_impact, injection_timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            perturbation.perturbation_id, perturbation.perturbation_type.value,
            perturbation.description, perturbation.intensity, str(perturbation.duration),
            json.dumps(perturbation.affected_systems), json.dumps(perturbation.expected_impact),
            perturbation.injection_timestamp.isoformat(), perturbation.status
        ))
        
        conn.commit()
        conn.close()
