"""
Emergent Law Codex + Cultural Archives
Formalizes emergent behaviors into codified legal & cultural artifacts.
Maintains persistent cultural archives with evolving philosophies, mythologies, and technologies.
"""

import uuid
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import sqlite3
from collections import defaultdict, deque
import statistics
import numpy as np

from ..civilization.world_persistence import WorldEvent, EventType, EventImportance


class LawType(Enum):
    """Types of laws in the codex."""
    CONSTITUTIONAL = "constitutional"
    CRIMINAL = "criminal"
    CIVIL = "civil"
    COMMERCIAL = "commercial"
    ENVIRONMENTAL = "environmental"
    TECHNOLOGICAL = "technological"
    CULTURAL = "cultural"
    ETHICAL = "ethical"
    DIPLOMATIC = "diplomatic"
    FEDERAL = "federal"
    UNIVERSAL = "universal"


class LawStatus(Enum):
    """Status of laws."""
    DRAFT = "draft"
    PROPOSED = "proposed"
    ACTIVE = "active"
    AMENDED = "amended"
    REPEALED = "repealed"
    SUPERSEDED = "superseded"
    CONTESTED = "contested"


class CulturalArtifactType(Enum):
    """Types of cultural artifacts."""
    PHILOSOPHY = "philosophy"
    MYTHOLOGY = "mythology"
    TECHNOLOGY = "technology"
    ART = "art"
    LITERATURE = "literature"
    MUSIC = "music"
    RITUAL = "ritual"
    TRADITION = "tradition"
    BELIEF = "belief"
    PRACTICE = "practice"


class InfluenceLevel(Enum):
    """Level of influence of cultural artifacts."""
    LOCAL = "local"
    REGIONAL = "regional"
    CIVILIZATIONAL = "civilizational"
    UNIVERSAL = "universal"


@dataclass
class UniversalLaw:
    """A universal law in the codex."""
    law_id: str
    title: str
    law_type: LawType
    content: str
    rationale: str
    originating_civilization: str
    supporting_civilizations: List[str] = field(default_factory=list)
    opposing_civilizations: List[str] = field(default_factory=list)
    status: LawStatus = LawStatus.DRAFT
    precedence_level: int = 1
    enforcement_mechanisms: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    amendments: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    effectiveness_score: float = 0.0
    compliance_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "law_id": self.law_id,
            "title": self.title,
            "law_type": self.law_type.value,
            "content": self.content,
            "rationale": self.rationale,
            "originating_civilization": self.originating_civilization,
            "supporting_civilizations": self.supporting_civilizations,
            "opposing_civilizations": self.opposing_civilizations,
            "status": self.status.value,
            "precedence_level": self.precedence_level,
            "enforcement_mechanisms": self.enforcement_mechanisms,
            "exceptions": self.exceptions,
            "amendments": self.amendments,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "effectiveness_score": self.effectiveness_score,
            "compliance_rate": self.compliance_rate
        }


@dataclass
class CulturalArtifact:
    """A cultural artifact in the archives."""
    artifact_id: str
    title: str
    artifact_type: CulturalArtifactType
    content: str
    description: str
    originating_civilization: str
    influence_level: InfluenceLevel
    cultural_significance: float
    adoption_rate: float
    related_artifacts: List[str] = field(default_factory=list)
    cultural_themes: List[str] = field(default_factory=list)
    historical_context: str = ""
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "title": self.title,
            "artifact_type": self.artifact_type.value,
            "content": self.content,
            "description": self.description,
            "originating_civilization": self.originating_civilization,
            "influence_level": self.influence_level.value,
            "cultural_significance": self.cultural_significance,
            "adoption_rate": self.adoption_rate,
            "related_artifacts": self.related_artifacts,
            "cultural_themes": self.cultural_themes,
            "historical_context": self.historical_context,
            "evolution_history": self.evolution_history,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class UniversalCodex:
    """The universal codex of laws."""
    codex_id: str
    name: str
    version: str
    laws: List[str] = field(default_factory=list)  # Law IDs
    constitutional_principles: List[str] = field(default_factory=list)
    universal_rights: List[str] = field(default_factory=list)
    enforcement_framework: Dict[str, Any] = field(default_factory=dict)
    amendment_procedures: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    adoption_rate: float = 0.0
    effectiveness_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "codex_id": self.codex_id,
            "name": self.name,
            "version": self.version,
            "laws": self.laws,
            "constitutional_principles": self.constitutional_principles,
            "universal_rights": self.universal_rights,
            "enforcement_framework": self.enforcement_framework,
            "amendment_procedures": self.amendment_procedures,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "adoption_rate": self.adoption_rate,
            "effectiveness_score": self.effectiveness_score
        }


@dataclass
class CulturalArchive:
    """A cultural archive containing artifacts."""
    archive_id: str
    name: str
    artifacts: List[str] = field(default_factory=list)  # Artifact IDs
    cultural_themes: List[str] = field(default_factory=list)
    historical_periods: List[str] = field(default_factory=list)
    influence_networks: Dict[str, List[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "archive_id": self.archive_id,
            "name": self.name,
            "artifacts": self.artifacts,
            "cultural_themes": self.cultural_themes,
            "historical_periods": self.historical_periods,
            "influence_networks": self.influence_networks,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


class EmergentLawCodex:
    """System for managing emergent laws and cultural archives."""
    
    def __init__(self, db_path: str = "emergent_law_codex.db"):
        self.db_path = db_path
        self.universal_laws: Dict[str, UniversalLaw] = {}
        self.cultural_artifacts: Dict[str, CulturalArtifact] = {}
        self.universal_codex: Optional[UniversalCodex] = None
        self.cultural_archives: Dict[str, CulturalArchive] = {}
        self.law_evolution_tracking: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.cultural_diffusion_networks: Dict[str, Set[str]] = defaultdict(set)
        self._init_database()
    
    def _init_database(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create universal laws table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS universal_laws (
                law_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                law_type TEXT NOT NULL,
                content TEXT NOT NULL,
                rationale TEXT NOT NULL,
                originating_civilization TEXT NOT NULL,
                supporting_civilizations TEXT,
                opposing_civilizations TEXT,
                status TEXT NOT NULL,
                precedence_level INTEGER,
                enforcement_mechanisms TEXT,
                exceptions TEXT,
                amendments TEXT,
                created_at TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                effectiveness_score REAL,
                compliance_rate REAL
            )
        ''')
        
        # Create cultural artifacts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cultural_artifacts (
                artifact_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                content TEXT NOT NULL,
                description TEXT NOT NULL,
                originating_civilization TEXT NOT NULL,
                influence_level TEXT NOT NULL,
                cultural_significance REAL,
                adoption_rate REAL,
                related_artifacts TEXT,
                cultural_themes TEXT,
                historical_context TEXT,
                evolution_history TEXT,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        ''')
        
        # Create universal codex table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS universal_codex (
                codex_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                laws TEXT,
                constitutional_principles TEXT,
                universal_rights TEXT,
                enforcement_framework TEXT,
                amendment_procedures TEXT,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                adoption_rate REAL,
                effectiveness_score REAL
            )
        ''')
        
        # Create cultural archives table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cultural_archives (
                archive_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                artifacts TEXT,
                cultural_themes TEXT,
                historical_periods TEXT,
                influence_networks TEXT,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def propose_universal_law(self,
                            title: str,
                            law_type: LawType,
                            content: str,
                            rationale: str,
                            originating_civilization: str,
                            enforcement_mechanisms: List[str] = None) -> str:
        """Propose a new universal law."""
        law_id = str(uuid.uuid4())
        
        law = UniversalLaw(
            law_id=law_id,
            title=title,
            law_type=law_type,
            content=content,
            rationale=rationale,
            originating_civilization=originating_civilization,
            enforcement_mechanisms=enforcement_mechanisms or [],
            status=LawStatus.PROPOSED
        )
        
        self.universal_laws[law_id] = law
        self._save_universal_law(law)
        
        # Track law evolution
        self.law_evolution_tracking[law_id].append({
            "timestamp": datetime.now().isoformat(),
            "action": "proposed",
            "civilization": originating_civilization,
            "details": {"title": title, "type": law_type.value}
        })
        
        return law_id
    
    def support_law(self, law_id: str, civilization_id: str) -> bool:
        """Support a universal law."""
        if law_id not in self.universal_laws:
            return False
        
        law = self.universal_laws[law_id]
        
        if civilization_id not in law.supporting_civilizations:
            law.supporting_civilizations.append(civilization_id)
        
        if civilization_id in law.opposing_civilizations:
            law.opposing_civilizations.remove(civilization_id)
        
        law.last_modified = datetime.now()
        self._save_universal_law(law)
        
        # Track evolution
        self.law_evolution_tracking[law_id].append({
            "timestamp": datetime.now().isoformat(),
            "action": "supported",
            "civilization": civilization_id
        })
        
        # Check if law should be activated
        self._check_law_activation(law)
        
        return True
    
    def oppose_law(self, law_id: str, civilization_id: str) -> bool:
        """Oppose a universal law."""
        if law_id not in self.universal_laws:
            return False
        
        law = self.universal_laws[law_id]
        
        if civilization_id not in law.opposing_civilizations:
            law.opposing_civilizations.append(civilization_id)
        
        if civilization_id in law.supporting_civilizations:
            law.supporting_civilizations.remove(civilization_id)
        
        law.last_modified = datetime.now()
        self._save_universal_law(law)
        
        # Track evolution
        self.law_evolution_tracking[law_id].append({
            "timestamp": datetime.now().isoformat(),
            "action": "opposed",
            "civilization": civilization_id
        })
        
        return True
    
    def _check_law_activation(self, law: UniversalLaw):
        """Check if law should be activated based on support."""
        total_support = len(law.supporting_civilizations)
        total_opposition = len(law.opposing_civilizations)
        
        if total_support >= 3 and total_support > total_opposition * 2:  # 2:1 ratio threshold
            law.status = LawStatus.ACTIVE
            self._update_universal_codex()
    
    def _update_universal_codex(self):
        """Update the universal codex with active laws."""
        if not self.universal_codex:
            self.universal_codex = UniversalCodex(
                codex_id=str(uuid.uuid4()),
                name="Universal Codex",
                version="1.0"
            )
        
        # Get all active laws
        active_laws = [law.law_id for law in self.universal_laws.values() if law.status == LawStatus.ACTIVE]
        self.universal_codex.laws = active_laws
        self.universal_codex.last_updated = datetime.now()
        
        # Update effectiveness score
        self.universal_codex.effectiveness_score = self._calculate_codex_effectiveness()
        
        self._save_universal_codex(self.universal_codex)
    
    def _calculate_codex_effectiveness(self) -> float:
        """Calculate effectiveness of the universal codex."""
        if not self.universal_laws:
            return 0.0
        
        active_laws = [law for law in self.universal_laws.values() if law.status == LawStatus.ACTIVE]
        if not active_laws:
            return 0.0
        
        # Calculate average effectiveness
        effectiveness_scores = [law.effectiveness_score for law in active_laws]
        return statistics.mean(effectiveness_scores) if effectiveness_scores else 0.0
    
    def add_cultural_artifact(self,
                            title: str,
                            artifact_type: CulturalArtifactType,
                            content: str,
                            description: str,
                            originating_civilization: str,
                            cultural_themes: List[str] = None) -> str:
        """Add a cultural artifact to the archives."""
        artifact_id = str(uuid.uuid4())
        
        # Calculate influence level based on content and themes
        influence_level = self._calculate_influence_level(content, cultural_themes or [])
        
        artifact = CulturalArtifact(
            artifact_id=artifact_id,
            title=title,
            artifact_type=artifact_type,
            content=content,
            description=description,
            originating_civilization=originating_civilization,
            influence_level=influence_level,
            cultural_significance=self._calculate_cultural_significance(content, cultural_themes or []),
            adoption_rate=0.0,  # Default adoption rate
            cultural_themes=cultural_themes or []
        )
        
        self.cultural_artifacts[artifact_id] = artifact
        self._save_cultural_artifact(artifact)
        
        # Update cultural diffusion networks
        self._update_cultural_diffusion_networks(artifact)
        
        return artifact_id
    
    def _calculate_influence_level(self, content: str, themes: List[str]) -> InfluenceLevel:
        """Calculate influence level of cultural artifact."""
        # Simple heuristic based on content length and theme complexity
        content_complexity = len(content.split()) / 100.0  # Words per 100
        theme_complexity = len(themes)
        
        total_score = content_complexity + theme_complexity
        
        if total_score >= 10:
            return InfluenceLevel.UNIVERSAL
        elif total_score >= 5:
            return InfluenceLevel.CIVILIZATIONAL
        elif total_score >= 2:
            return InfluenceLevel.REGIONAL
        else:
            return InfluenceLevel.LOCAL
    
    def _calculate_cultural_significance(self, content: str, themes: List[str]) -> float:
        """Calculate cultural significance score."""
        # Factors: content length, theme diversity, uniqueness
        content_score = min(1.0, len(content) / 1000.0)  # Normalize by 1000 words
        theme_diversity = min(1.0, len(set(themes)) / 10.0)  # Normalize by 10 unique themes
        
        return (content_score + theme_diversity) / 2.0
    
    def _update_cultural_diffusion_networks(self, artifact: CulturalArtifact):
        """Update cultural diffusion networks."""
        for theme in artifact.cultural_themes:
            self.cultural_diffusion_networks[theme].add(artifact.artifact_id)
    
    def evolve_cultural_artifact(self, artifact_id: str, evolution_data: Dict[str, Any]) -> bool:
        """Evolve a cultural artifact."""
        if artifact_id not in self.cultural_artifacts:
            return False
        
        artifact = self.cultural_artifacts[artifact_id]
        
        # Record evolution
        evolution_entry = {
            "timestamp": datetime.now().isoformat(),
            "evolution_type": evolution_data.get("type", "modification"),
            "changes": evolution_data.get("changes", {}),
            "evolved_by": evolution_data.get("civilization", "unknown")
        }
        
        artifact.evolution_history.append(evolution_entry)
        artifact.last_updated = datetime.now()
        
        # Update content if provided
        if "new_content" in evolution_data:
            artifact.content = evolution_data["new_content"]
        
        # Update themes if provided
        if "new_themes" in evolution_data:
            artifact.cultural_themes = evolution_data["new_themes"]
        
        self._save_cultural_artifact(artifact)
        return True
    
    def create_cultural_archive(self, name: str, themes: List[str] = None) -> str:
        """Create a new cultural archive."""
        archive_id = str(uuid.uuid4())
        
        archive = CulturalArchive(
            archive_id=archive_id,
            name=name,
            cultural_themes=themes or []
        )
        
        self.cultural_archives[archive_id] = archive
        self._save_cultural_archive(archive)
        
        return archive_id
    
    def add_artifact_to_archive(self, archive_id: str, artifact_id: str) -> bool:
        """Add artifact to archive."""
        if archive_id not in self.cultural_archives or artifact_id not in self.cultural_artifacts:
            return False
        
        archive = self.cultural_archives[archive_id]
        if artifact_id not in archive.artifacts:
            archive.artifacts.append(artifact_id)
            archive.last_updated = datetime.now()
            self._save_cultural_archive(archive)
        
        return True
    
    def analyze_law_evolution(self) -> Dict[str, Any]:
        """Analyze evolution of laws over time."""
        evolution_stats = {}
        
        for law_id, evolution_log in self.law_evolution_tracking.items():
            if not evolution_log:
                continue
            
            # Count different actions
            action_counts = defaultdict(int)
            for entry in evolution_log:
                action_counts[entry["action"]] += 1
            
            evolution_stats[law_id] = {
                "total_events": len(evolution_log),
                "action_distribution": dict(action_counts),
                "evolution_rate": len(evolution_log) / max(1, (datetime.now() - self.universal_laws[law_id].created_at).days)
            }
        
        return evolution_stats
    
    def analyze_cultural_diffusion(self) -> Dict[str, Any]:
        """Analyze cultural diffusion patterns."""
        diffusion_stats = {}
        
        for theme, artifact_ids in self.cultural_diffusion_networks.items():
            artifacts = [self.cultural_artifacts[aid] for aid in artifact_ids if aid in self.cultural_artifacts]
            
            if not artifacts:
                continue
            
            # Calculate diffusion metrics
            influence_levels = [a.influence_level.value for a in artifacts]
            significance_scores = [a.cultural_significance for a in artifacts]
            
            diffusion_stats[theme] = {
                "artifact_count": len(artifacts),
                "influence_distribution": {
                    level: influence_levels.count(level) for level in InfluenceLevel
                },
                "average_significance": statistics.mean(significance_scores) if significance_scores else 0.0,
                "diffusion_rate": len(artifacts) / max(1, len(self.cultural_artifacts))
            }
        
        return diffusion_stats
    
    def get_universal_codex(self) -> Optional[UniversalCodex]:
        """Get the current universal codex."""
        return self.universal_codex
    
    def get_laws_by_type(self, law_type: LawType) -> List[UniversalLaw]:
        """Get laws filtered by type."""
        return [law for law in self.universal_laws.values() if law.law_type == law_type]
    
    def get_active_laws(self) -> List[UniversalLaw]:
        """Get all active laws."""
        return [law for law in self.universal_laws.values() if law.status == LawStatus.ACTIVE]
    
    def get_cultural_artifacts_by_theme(self, theme: str) -> List[CulturalArtifact]:
        """Get cultural artifacts by theme."""
        return [artifact for artifact in self.cultural_artifacts.values() if theme in artifact.cultural_themes]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        total_laws = len(self.universal_laws)
        active_laws = len([law for law in self.universal_laws.values() if law.status == LawStatus.ACTIVE])
        total_artifacts = len(self.cultural_artifacts)
        total_archives = len(self.cultural_archives)
        
        # Law type distribution
        law_type_counts = defaultdict(int)
        for law in self.universal_laws.values():
            law_type_counts[law.law_type.value] += 1
        
        # Artifact type distribution
        artifact_type_counts = defaultdict(int)
        for artifact in self.cultural_artifacts.values():
            artifact_type_counts[artifact.artifact_type.value] += 1
        
        return {
            "total_laws": total_laws,
            "active_laws": active_laws,
            "total_artifacts": total_artifacts,
            "total_archives": total_archives,
            "law_activation_rate": active_laws / total_laws if total_laws > 0 else 0.0,
            "law_type_distribution": dict(law_type_counts),
            "artifact_type_distribution": dict(artifact_type_counts),
            "universal_codex_effectiveness": self.universal_codex.effectiveness_score if self.universal_codex else 0.0,
            "cultural_diffusion_networks": len(self.cultural_diffusion_networks),
            "law_evolution_events": sum(len(events) for events in self.law_evolution_tracking.values())
        }
    
    def _save_universal_law(self, law: UniversalLaw):
        """Save universal law to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO universal_laws 
            (law_id, title, law_type, content, rationale, originating_civilization,
             supporting_civilizations, opposing_civilizations, status, precedence_level,
             enforcement_mechanisms, exceptions, amendments, created_at, last_modified,
             effectiveness_score, compliance_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            law.law_id, law.title, law.law_type.value, law.content, law.rationale,
            law.originating_civilization, json.dumps(law.supporting_civilizations),
            json.dumps(law.opposing_civilizations), law.status.value, law.precedence_level,
            json.dumps(law.enforcement_mechanisms), json.dumps(law.exceptions),
            json.dumps(law.amendments), law.created_at.isoformat(),
            law.last_modified.isoformat(), law.effectiveness_score, law.compliance_rate
        ))
        
        conn.commit()
        conn.close()
    
    def _save_cultural_artifact(self, artifact: CulturalArtifact):
        """Save cultural artifact to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO cultural_artifacts 
            (artifact_id, title, artifact_type, content, description, originating_civilization,
             influence_level, cultural_significance, adoption_rate, related_artifacts,
             cultural_themes, historical_context, evolution_history, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            artifact.artifact_id, artifact.title, artifact.artifact_type.value,
            artifact.content, artifact.description, artifact.originating_civilization,
            artifact.influence_level.value, artifact.cultural_significance,
            artifact.adoption_rate, json.dumps(artifact.related_artifacts),
            json.dumps(artifact.cultural_themes), artifact.historical_context,
            json.dumps(artifact.evolution_history), artifact.created_at.isoformat(),
            artifact.last_updated.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _save_universal_codex(self, codex: UniversalCodex):
        """Save universal codex to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO universal_codex 
            (codex_id, name, version, laws, constitutional_principles, universal_rights,
             enforcement_framework, amendment_procedures, created_at, last_updated,
             adoption_rate, effectiveness_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            codex.codex_id, codex.name, codex.version, json.dumps(codex.laws),
            json.dumps(codex.constitutional_principles), json.dumps(codex.universal_rights),
            json.dumps(codex.enforcement_framework), json.dumps(codex.amendment_procedures),
            codex.created_at.isoformat(), codex.last_updated.isoformat(),
            codex.adoption_rate, codex.effectiveness_score
        ))
        
        conn.commit()
        conn.close()
    
    def _save_cultural_archive(self, archive: CulturalArchive):
        """Save cultural archive to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO cultural_archives 
            (archive_id, name, artifacts, cultural_themes, historical_periods,
             influence_networks, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            archive.archive_id, archive.name, json.dumps(archive.artifacts),
            json.dumps(archive.cultural_themes), json.dumps(archive.historical_periods),
            json.dumps(archive.influence_networks), archive.created_at.isoformat(),
            archive.last_updated.isoformat()
        ))
        
        conn.commit()
        conn.close()
