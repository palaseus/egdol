"""
World Persistence & Narrative Memory
Long-lived world models with civilizational histories, laws, and technologies.
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
import sqlite3
from collections import defaultdict, deque

from ..conversational.personality_framework import Personality, PersonalityType


class EventType(Enum):
    """Event type enumeration."""
    DISCOVERY = auto()
    INVENTION = auto()
    LAW_ENACTMENT = auto()
    TREATY_SIGNING = auto()
    CONFLICT = auto()
    CULTURAL_EVOLUTION = auto()
    TECHNOLOGICAL_BREAKTHROUGH = auto()
    POLITICAL_CHANGE = auto()
    ECONOMIC_SHIFT = auto()
    ENVIRONMENTAL_CHANGE = auto()


class EventImportance(Enum):
    """Event importance level."""
    MINOR = auto()
    MODERATE = auto()
    MAJOR = auto()
    CRITICAL = auto()
    HISTORICAL = auto()


@dataclass
class WorldEvent:
    """World event in civilizational history."""
    event_id: str
    event_type: EventType
    importance: EventImportance
    title: str
    description: str
    participants: List[str] = field(default_factory=list)  # agent_ids
    location: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    consequences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "importance": self.importance.name,
            "title": self.title,
            "description": self.description,
            "participants": self.participants,
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
            "consequences": self.consequences,
            "metadata": self.metadata
        }


@dataclass
class CivilizationalLaw:
    """Civilizational law or rule."""
    law_id: str
    title: str
    description: str
    proposer_agent_id: str
    enacted_by: List[str] = field(default_factory=list)  # agent_ids who voted for
    opposed_by: List[str] = field(default_factory=list)  # agent_ids who voted against
    enactment_date: datetime = field(default_factory=datetime.now)
    amendments: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"  # active, repealed, suspended
    domain: str = "general"  # governance, technology, culture, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "law_id": self.law_id,
            "title": self.title,
            "description": self.description,
            "proposer_agent_id": self.proposer_agent_id,
            "enacted_by": self.enacted_by,
            "opposed_by": self.opposed_by,
            "enactment_date": self.enactment_date.isoformat(),
            "amendments": self.amendments,
            "status": self.status,
            "domain": self.domain
        }


@dataclass
class Technology:
    """Civilizational technology."""
    tech_id: str
    name: str
    description: str
    inventor_agent_id: str
    discovery_date: datetime = field(default_factory=datetime.now)
    prerequisites: List[str] = field(default_factory=list)  # tech_ids
    applications: List[str] = field(default_factory=list)
    adoption_rate: float = 0.0  # 0.0 to 1.0
    impact_level: int = 1  # 1-10 scale
    domain: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tech_id": self.tech_id,
            "name": self.name,
            "description": self.description,
            "inventor_agent_id": self.inventor_agent_id,
            "discovery_date": self.discovery_date.isoformat(),
            "prerequisites": self.prerequisites,
            "applications": self.applications,
            "adoption_rate": self.adoption_rate,
            "impact_level": self.impact_level,
            "domain": self.domain
        }


@dataclass
class CulturalTrait:
    """Cultural trait or practice."""
    trait_id: str
    name: str
    description: str
    origin_agent_id: str
    adoption_date: datetime = field(default_factory=datetime.now)
    prevalence: float = 0.0  # 0.0 to 1.0
    influence: float = 0.0  # 0.0 to 1.0
    related_traits: List[str] = field(default_factory=list)
    domain: str = "cultural"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trait_id": self.trait_id,
            "name": self.name,
            "description": self.description,
            "origin_agent_id": self.origin_agent_id,
            "adoption_date": self.adoption_date.isoformat(),
            "prevalence": self.prevalence,
            "influence": self.influence,
            "related_traits": self.related_traits,
            "domain": self.domain
        }


@dataclass
class WorldState:
    """Current world state."""
    world_id: str
    name: str
    current_era: str = "foundation"
    population: int = 0
    technology_level: float = 0.0
    cultural_diversity: float = 0.0
    political_stability: float = 0.0
    economic_health: float = 0.0
    environmental_health: float = 0.0
    laws: List[str] = field(default_factory=list)  # law_ids
    technologies: List[str] = field(default_factory=list)  # tech_ids
    cultural_traits: List[str] = field(default_factory=list)  # trait_ids
    active_conflicts: List[str] = field(default_factory=list)
    active_treaties: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "world_id": self.world_id,
            "name": self.name,
            "current_era": self.current_era,
            "population": self.population,
            "technology_level": self.technology_level,
            "cultural_diversity": self.cultural_diversity,
            "political_stability": self.political_stability,
            "economic_health": self.economic_health,
            "environmental_health": self.environmental_health,
            "laws": self.laws,
            "technologies": self.technologies,
            "cultural_traits": self.cultural_traits,
            "active_conflicts": self.active_conflicts,
            "active_treaties": self.active_treaties,
            "last_updated": self.last_updated.isoformat()
        }


class WorldPersistence:
    """World persistence and narrative memory system."""
    
    def __init__(self, db_path: str = "world_persistence.db"):
        self.db_path = db_path
        self.world_states: Dict[str, WorldState] = {}
        self.events: Dict[str, WorldEvent] = {}
        self.laws: Dict[str, CivilizationalLaw] = {}
        self.technologies: Dict[str, Technology] = {}
        self.cultural_traits: Dict[str, CulturalTrait] = {}
        self.narrative_memory: deque = deque(maxlen=10000)  # Last 10k events
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # World states table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS world_states (
                    world_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    current_era TEXT,
                    population INTEGER,
                    technology_level REAL,
                    cultural_diversity REAL,
                    political_stability REAL,
                    economic_health REAL,
                    environmental_health REAL,
                    laws TEXT,
                    technologies TEXT,
                    cultural_traits TEXT,
                    active_conflicts TEXT,
                    active_treaties TEXT,
                    last_updated TEXT
                )
            """)
            
            # Events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    importance TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    participants TEXT,
                    location TEXT,
                    timestamp TEXT NOT NULL,
                    consequences TEXT,
                    metadata TEXT
                )
            """)
            
            # Laws table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS laws (
                    law_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    proposer_agent_id TEXT,
                    enacted_by TEXT,
                    opposed_by TEXT,
                    enactment_date TEXT,
                    amendments TEXT,
                    status TEXT,
                    domain TEXT
                )
            """)
            
            # Technologies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS technologies (
                    tech_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    inventor_agent_id TEXT,
                    discovery_date TEXT,
                    prerequisites TEXT,
                    applications TEXT,
                    adoption_rate REAL,
                    impact_level INTEGER,
                    domain TEXT
                )
            """)
            
            # Cultural traits table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cultural_traits (
                    trait_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    origin_agent_id TEXT,
                    adoption_date TEXT,
                    prevalence REAL,
                    influence REAL,
                    related_traits TEXT,
                    domain TEXT
                )
            """)
    
    def create_world(self, name: str) -> str:
        """Create a new world."""
        world_id = str(uuid.uuid4())
        
        world_state = WorldState(
            world_id=world_id,
            name=name
        )
        
        self.world_states[world_id] = world_state
        self._save_world_state(world_state)
        
        return world_id
    
    def add_event(self, 
                 world_id: str,
                 event_type: EventType,
                 importance: EventImportance,
                 title: str,
                 description: str,
                 participants: List[str] = None,
                 location: str = None,
                 consequences: List[str] = None,
                 metadata: Dict[str, Any] = None) -> str:
        """Add event to world history."""
        event_id = str(uuid.uuid4())
        
        event = WorldEvent(
            event_id=event_id,
            event_type=event_type,
            importance=importance,
            title=title,
            description=description,
            participants=participants or [],
            location=location,
            consequences=consequences or [],
            metadata=(metadata or {}).copy()
        )
        
        # Add world_id to metadata
        event.metadata["world_id"] = world_id
        
        self.events[event_id] = event
        self.narrative_memory.append(event)
        self._save_event(event)
        
        # Update world state based on event
        self._update_world_state_from_event(world_id, event)
        
        return event_id
    
    def enact_law(self,
                 world_id: str,
                 title: str,
                 description: str,
                 proposer_agent_id: str,
                 enacted_by: List[str],
                 opposed_by: List[str] = None,
                 domain: str = "general") -> str:
        """Enact a new law."""
        law_id = str(uuid.uuid4())
        
        law = CivilizationalLaw(
            law_id=law_id,
            title=title,
            description=description,
            proposer_agent_id=proposer_agent_id,
            enacted_by=enacted_by,
            opposed_by=opposed_by or [],
            domain=domain
        )
        
        self.laws[law_id] = law
        self._save_law(law)
        
        # Add to world state
        if world_id in self.world_states:
            self.world_states[world_id].laws.append(law_id)
            self._save_world_state(self.world_states[world_id])
        
        # Create event for law enactment
        self.add_event(
            world_id=world_id,
            event_type=EventType.LAW_ENACTMENT,
            importance=EventImportance.MODERATE,
            title=f"Law Enacted: {title}",
            description=description,
            participants=enacted_by,
            metadata={"law_id": law_id, "domain": domain}
        )
        
        return law_id
    
    def discover_technology(self,
                          world_id: str,
                          name: str,
                          description: str,
                          inventor_agent_id: str,
                          prerequisites: List[str] = None,
                          applications: List[str] = None,
                          domain: str = "general") -> str:
        """Discover new technology."""
        tech_id = str(uuid.uuid4())
        
        technology = Technology(
            tech_id=tech_id,
            name=name,
            description=description,
            inventor_agent_id=inventor_agent_id,
            prerequisites=prerequisites or [],
            applications=applications or [],
            domain=domain
        )
        
        self.technologies[tech_id] = technology
        self._save_technology(technology)
        
        # Add to world state
        if world_id in self.world_states:
            self.world_states[world_id].technologies.append(tech_id)
            self._save_world_state(self.world_states[world_id])
        
        # Create event for technology discovery
        self.add_event(
            world_id=world_id,
            event_type=EventType.TECHNOLOGICAL_BREAKTHROUGH,
            importance=EventImportance.MAJOR,
            title=f"Technology Discovered: {name}",
            description=description,
            participants=[inventor_agent_id],
            metadata={"tech_id": tech_id, "domain": domain}
        )
        
        return tech_id
    
    def adopt_cultural_trait(self,
                           world_id: str,
                           name: str,
                           description: str,
                           origin_agent_id: str,
                           related_traits: List[str] = None,
                           domain: str = "cultural") -> str:
        """Adopt new cultural trait."""
        trait_id = str(uuid.uuid4())
        
        trait = CulturalTrait(
            trait_id=trait_id,
            name=name,
            description=description,
            origin_agent_id=origin_agent_id,
            related_traits=related_traits or [],
            domain=domain
        )
        
        self.cultural_traits[trait_id] = trait
        self._save_cultural_trait(trait)
        
        # Add to world state
        if world_id in self.world_states:
            self.world_states[world_id].cultural_traits.append(trait_id)
            self._save_world_state(self.world_states[world_id])
        
        # Create event for cultural evolution
        self.add_event(
            world_id=world_id,
            event_type=EventType.CULTURAL_EVOLUTION,
            importance=EventImportance.MODERATE,
            title=f"Cultural Trait Adopted: {name}",
            description=description,
            participants=[origin_agent_id],
            metadata={"trait_id": trait_id, "domain": domain}
        )
        
        return trait_id
    
    def _update_world_state_from_event(self, world_id: str, event: WorldEvent):
        """Update world state based on event."""
        if world_id not in self.world_states:
            return
        
        world_state = self.world_states[world_id]
        
        # Update based on event type
        if event.event_type == EventType.TECHNOLOGICAL_BREAKTHROUGH:
            world_state.technology_level = min(1.0, world_state.technology_level + 0.1)
        elif event.event_type == EventType.CULTURAL_EVOLUTION:
            world_state.cultural_diversity = min(1.0, world_state.cultural_diversity + 0.05)
        elif event.event_type == EventType.LAW_ENACTMENT:
            world_state.political_stability = min(1.0, world_state.political_stability + 0.02)
        elif event.event_type == EventType.CONFLICT:
            world_state.political_stability = max(0.0, world_state.political_stability - 0.1)
        elif event.event_type == EventType.ECONOMIC_SHIFT:
            world_state.economic_health = min(1.0, world_state.economic_health + 0.05)
        elif event.event_type == EventType.ENVIRONMENTAL_CHANGE:
            world_state.environmental_health = min(1.0, world_state.environmental_health + 0.03)
        
        # Update population based on events
        if event.importance == EventImportance.HISTORICAL:
            world_state.population = max(0, world_state.population + 100)
        elif event.importance == EventImportance.CRITICAL:
            world_state.population = max(0, world_state.population + 50)
        elif event.importance == EventImportance.MAJOR:
            world_state.population = max(0, world_state.population + 25)
        
        world_state.last_updated = datetime.now()
        self._save_world_state(world_state)
    
    def get_world_history(self, world_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get world history."""
        world_events = [e for e in self.events.values() if e.metadata.get("world_id") == world_id]
        world_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return [event.to_dict() for event in world_events[:limit]]
    
    def get_world_state(self, world_id: str) -> Dict[str, Any]:
        """Get current world state."""
        if world_id not in self.world_states:
            return {}
        
        return self.world_states[world_id].to_dict()
    
    def get_civilizational_memory(self, world_id: str, query: str = None) -> List[Dict[str, Any]]:
        """Get civilizational memory based on query."""
        # Get all events for world
        world_events = [e for e in self.events.values() if e.metadata.get("world_id") == world_id]
        
        if query:
            # Simple keyword matching - could be enhanced with semantic search
            query_lower = query.lower()
            relevant_events = [
                e for e in world_events
                if (query_lower in e.title.lower() or 
                    query_lower in e.description.lower() or
                    query_lower in e.event_type.name.lower())
            ]
        else:
            relevant_events = world_events
        
        # Sort by importance and timestamp
        relevant_events.sort(key=lambda e: (e.importance.value, e.timestamp), reverse=True)
        
        return [event.to_dict() for event in relevant_events]
    
    def get_historical_context(self, world_id: str, time_period: str = "recent") -> Dict[str, Any]:
        """Get historical context for time period."""
        world_events = [e for e in self.events.values() if e.metadata.get("world_id") == world_id]
        
        if time_period == "recent":
            # Last 100 events
            recent_events = sorted(world_events, key=lambda e: e.timestamp, reverse=True)[:100]
        elif time_period == "era":
            # Events from current era
            world_state = self.world_states.get(world_id)
            if world_state:
                era_events = [e for e in world_events if e.metadata.get("era") == world_state.current_era]
            else:
                era_events = []
        else:
            # All events
            recent_events = world_events
        
        # Analyze events
        event_types = defaultdict(int)
        importance_levels = defaultdict(int)
        participants = set()
        
        for event in recent_events:
            event_types[event.event_type.name] += 1
            importance_levels[event.importance.name] += 1
            participants.update(event.participants)
        
        return {
            "time_period": time_period,
            "total_events": len(recent_events),
            "event_types": dict(event_types),
            "importance_distribution": dict(importance_levels),
            "active_participants": len(participants),
            "events": [event.to_dict() for event in recent_events[:50]]  # Top 50 events
        }
    
    def _save_world_state(self, world_state: WorldState):
        """Save world state to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO world_states 
                (world_id, name, current_era, population, technology_level, 
                 cultural_diversity, political_stability, economic_health, 
                 environmental_health, laws, technologies, cultural_traits, 
                 active_conflicts, active_treaties, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                world_state.world_id,
                world_state.name,
                world_state.current_era,
                world_state.population,
                world_state.technology_level,
                world_state.cultural_diversity,
                world_state.political_stability,
                world_state.economic_health,
                world_state.environmental_health,
                json.dumps(world_state.laws),
                json.dumps(world_state.technologies),
                json.dumps(world_state.cultural_traits),
                json.dumps(world_state.active_conflicts),
                json.dumps(world_state.active_treaties),
                world_state.last_updated.isoformat()
            ))
    
    def _save_event(self, event: WorldEvent):
        """Save event to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO events 
                (event_id, event_type, importance, title, description, 
                 participants, location, timestamp, consequences, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.name,
                event.importance.name,
                event.title,
                event.description,
                json.dumps(event.participants),
                event.location,
                event.timestamp.isoformat(),
                json.dumps(event.consequences),
                json.dumps(event.metadata)
            ))
    
    def _save_law(self, law: CivilizationalLaw):
        """Save law to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO laws 
                (law_id, title, description, proposer_agent_id, enacted_by, 
                 opposed_by, enactment_date, amendments, status, domain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                law.law_id,
                law.title,
                law.description,
                law.proposer_agent_id,
                json.dumps(law.enacted_by),
                json.dumps(law.opposed_by),
                law.enactment_date.isoformat(),
                json.dumps(law.amendments),
                law.status,
                law.domain
            ))
    
    def _save_technology(self, technology: Technology):
        """Save technology to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO technologies 
                (tech_id, name, description, inventor_agent_id, discovery_date, 
                 prerequisites, applications, adoption_rate, impact_level, domain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                technology.tech_id,
                technology.name,
                technology.description,
                technology.inventor_agent_id,
                technology.discovery_date.isoformat(),
                json.dumps(technology.prerequisites),
                json.dumps(technology.applications),
                technology.adoption_rate,
                technology.impact_level,
                technology.domain
            ))
    
    def _save_cultural_trait(self, trait: CulturalTrait):
        """Save cultural trait to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cultural_traits 
                (trait_id, name, description, origin_agent_id, adoption_date, 
                 prevalence, influence, related_traits, domain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trait.trait_id,
                trait.name,
                trait.description,
                trait.origin_agent_id,
                trait.adoption_date.isoformat(),
                trait.prevalence,
                trait.influence,
                json.dumps(trait.related_traits),
                trait.domain
            ))
