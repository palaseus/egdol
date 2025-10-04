"""
Cross-Civilization Interaction Layer
Enables civilizations to interact through trade, conflict, treaties, and cultural exchange.
Supports federation, alliances, and collapse under external pressure.
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

from ..civilization.multi_agent_system import CivilizationState, CivilizationAgent
from ..civilization.world_persistence import WorldEvent, EventType, EventImportance


class InteractionType(Enum):
    """Types of inter-civilization interactions."""
    TRADE = "trade"
    DIPLOMACY = "diplomacy"
    CONFLICT = "conflict"
    CULTURAL_EXCHANGE = "cultural_exchange"
    TECHNOLOGY_TRANSFER = "technology_transfer"
    RESOURCE_SHARING = "resource_sharing"
    TREATY = "treaty"
    ALLIANCE = "alliance"
    FEDERATION = "federation"
    WAR = "war"
    PEACE = "peace"
    EMBARGO = "embargo"
    MIGRATION = "migration"
    KNOWLEDGE_EXCHANGE = "knowledge_exchange"


class InteractionStatus(Enum):
    """Status of interactions."""
    PROPOSED = "proposed"
    NEGOTIATING = "negotiating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class CivilizationRelation(Enum):
    """Types of civilization relationships."""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    ALLIED = "allied"
    HOSTILE = "hostile"
    AT_WAR = "at_war"
    FEDERATED = "federated"
    ISOLATED = "isolated"


@dataclass
class CivilizationProfile:
    """Profile of a civilization for interaction purposes."""
    civilization_id: str
    name: str
    power_level: float
    technology_level: float
    cultural_cohesion: float
    economic_strength: float
    military_capability: float
    diplomatic_standing: float
    resources: Dict[str, float] = field(default_factory=dict)
    technologies: List[str] = field(default_factory=list)
    cultural_traits: List[str] = field(default_factory=list)
    political_system: str = "democracy"
    population_size: float = 1.0
    territorial_claims: List[str] = field(default_factory=list)
    alliances: List[str] = field(default_factory=list)
    enemies: List[str] = field(default_factory=list)
    trade_partners: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "civilization_id": self.civilization_id,
            "name": self.name,
            "power_level": self.power_level,
            "technology_level": self.technology_level,
            "cultural_cohesion": self.cultural_cohesion,
            "economic_strength": self.economic_strength,
            "military_capability": self.military_capability,
            "diplomatic_standing": self.diplomatic_standing,
            "resources": self.resources,
            "technologies": self.technologies,
            "cultural_traits": self.cultural_traits,
            "political_system": self.political_system,
            "population_size": self.population_size,
            "territorial_claims": self.territorial_claims,
            "alliances": self.alliances,
            "enemies": self.enemies,
            "trade_partners": self.trade_partners,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class InteractionProposal:
    """Proposal for inter-civilization interaction."""
    proposal_id: str
    interaction_type: InteractionType
    initiator_civilization: str
    target_civilization: str
    description: str
    terms: Dict[str, Any]
    benefits: List[str]
    costs: List[str]
    duration: timedelta
    status: InteractionStatus = InteractionStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    negotiations: List[Dict[str, Any]] = field(default_factory=list)
    final_agreement: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "interaction_type": self.interaction_type.value,
            "initiator_civilization": self.initiator_civilization,
            "target_civilization": self.target_civilization,
            "description": self.description,
            "terms": self.terms,
            "benefits": self.benefits,
            "costs": self.costs,
            "duration": str(self.duration),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "negotiations": self.negotiations,
            "final_agreement": self.final_agreement
        }


@dataclass
class Federation:
    """A federation of civilizations."""
    federation_id: str
    name: str
    founding_civilizations: List[str]
    member_civilizations: List[str]
    governance_structure: str
    shared_resources: Dict[str, float] = field(default_factory=dict)
    common_laws: List[str] = field(default_factory=list)
    federation_technologies: List[str] = field(default_factory=list)
    decision_making_process: str = "consensus"
    created_at: datetime = field(default_factory=datetime.now)
    last_meeting: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "federation_id": self.federation_id,
            "name": self.name,
            "founding_civilizations": self.founding_civilizations,
            "member_civilizations": self.member_civilizations,
            "governance_structure": self.governance_structure,
            "shared_resources": self.shared_resources,
            "common_laws": self.common_laws,
            "federation_technologies": self.federation_technologies,
            "decision_making_process": self.decision_making_process,
            "created_at": self.created_at.isoformat(),
            "last_meeting": self.last_meeting.isoformat() if self.last_meeting else None
        }


class CrossCivilizationInteractionSystem:
    """System for managing inter-civilization interactions."""
    
    def __init__(self, db_path: str = "cross_civilization.db"):
        self.db_path = db_path
        self.civilization_profiles: Dict[str, CivilizationProfile] = {}
        self.interactions: Dict[str, InteractionProposal] = {}
        self.federations: Dict[str, Federation] = {}
        self.relationship_matrix: Dict[Tuple[str, str], CivilizationRelation] = {}
        self.trade_routes: Dict[str, List[str]] = defaultdict(list)
        self.conflict_zones: Set[str] = set()
        self.peace_treaties: Dict[str, Dict[str, Any]] = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create civilization profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS civilization_profiles (
                civilization_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                power_level REAL NOT NULL,
                technology_level REAL NOT NULL,
                cultural_cohesion REAL NOT NULL,
                economic_strength REAL NOT NULL,
                military_capability REAL NOT NULL,
                diplomatic_standing REAL NOT NULL,
                resources TEXT,
                technologies TEXT,
                cultural_traits TEXT,
                political_system TEXT,
                population_size REAL,
                territorial_claims TEXT,
                alliances TEXT,
                enemies TEXT,
                trade_partners TEXT,
                last_updated TEXT NOT NULL
            )
        ''')
        
        # Create interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                proposal_id TEXT PRIMARY KEY,
                interaction_type TEXT NOT NULL,
                initiator_civilization TEXT NOT NULL,
                target_civilization TEXT NOT NULL,
                description TEXT NOT NULL,
                terms TEXT,
                benefits TEXT,
                costs TEXT,
                duration TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                negotiations TEXT,
                final_agreement TEXT
            )
        ''')
        
        # Create federations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS federations (
                federation_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                founding_civilizations TEXT,
                member_civilizations TEXT,
                governance_structure TEXT,
                shared_resources TEXT,
                common_laws TEXT,
                federation_technologies TEXT,
                decision_making_process TEXT,
                created_at TEXT NOT NULL,
                last_meeting TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_civilization(self, civilization: CivilizationState) -> str:
        """Register a civilization for interaction."""
        profile = CivilizationProfile(
            civilization_id=civilization.civilization_id,
            name=civilization.name,
            power_level=self._calculate_power_level(civilization),
            technology_level=self._calculate_technology_level(civilization),
            cultural_cohesion=self._calculate_cultural_cohesion(civilization),
            economic_strength=self._calculate_economic_strength(civilization),
            military_capability=self._calculate_military_capability(civilization),
            diplomatic_standing=0.5  # Neutral starting point
        )
        
        self.civilization_profiles[civilization.civilization_id] = profile
        self._save_civilization_profile(profile)
        
        # Initialize relationships with existing civilizations
        for existing_id in self.civilization_profiles:
            if existing_id != civilization.civilization_id:
                self.relationship_matrix[(civilization.civilization_id, existing_id)] = CivilizationRelation.NEUTRAL
                self.relationship_matrix[(existing_id, civilization.civilization_id)] = CivilizationRelation.NEUTRAL
        
        return civilization.civilization_id
    
    def _calculate_power_level(self, civilization: CivilizationState) -> float:
        """Calculate civilization power level."""
        # Base power from agent count
        agent_power = len(civilization.agents) * 0.1
        
        # Technology bonus
        tech_bonus = len(civilization.technologies) * 0.05
        
        # Law stability bonus
        law_bonus = len(civilization.laws) * 0.02
        
        return min(1.0, agent_power + tech_bonus + law_bonus)
    
    def _calculate_technology_level(self, civilization: CivilizationState) -> float:
        """Calculate civilization technology level."""
        return min(1.0, len(civilization.technologies) * 0.1)
    
    def _calculate_cultural_cohesion(self, civilization: CivilizationState) -> float:
        """Calculate civilization cultural cohesion."""
        # Based on cultural traits and laws
        cultural_factors = len(civilization.culture) + len(civilization.laws)
        return min(1.0, cultural_factors * 0.1)
    
    def _calculate_economic_strength(self, civilization: CivilizationState) -> float:
        """Calculate civilization economic strength."""
        # Simplified economic calculation
        return min(1.0, len(civilization.technologies) * 0.15)
    
    def _calculate_military_capability(self, civilization: CivilizationState) -> float:
        """Calculate civilization military capability."""
        # Based on agent count and technology
        agent_military = len(civilization.agents) * 0.08
        tech_military = len(civilization.technologies) * 0.06
        return min(1.0, agent_military + tech_military)
    
    def propose_interaction(self,
                          interaction_type: InteractionType,
                          initiator_civilization: str,
                          target_civilization: str,
                          description: str,
                          terms: Dict[str, Any],
                          duration: timedelta) -> str:
        """Propose an inter-civilization interaction."""
        proposal_id = str(uuid.uuid4())
        
        # Generate benefits and costs
        benefits = self._generate_interaction_benefits(interaction_type, initiator_civilization, target_civilization)
        costs = self._generate_interaction_costs(interaction_type, initiator_civilization, target_civilization)
        
        proposal = InteractionProposal(
            proposal_id=proposal_id,
            interaction_type=interaction_type,
            initiator_civilization=initiator_civilization,
            target_civilization=target_civilization,
            description=description,
            terms=terms,
            benefits=benefits,
            costs=costs,
            duration=duration,
            expires_at=datetime.now() + timedelta(days=7)  # 7-day expiration
        )
        
        self.interactions[proposal_id] = proposal
        self._save_interaction(proposal)
        
        # Update relationship based on interaction type
        self._update_relationship(initiator_civilization, target_civilization, interaction_type)
        
        return proposal_id
    
    def _generate_interaction_benefits(self, interaction_type: InteractionType, 
                                     initiator: str, target: str) -> List[str]:
        """Generate benefits for interaction."""
        benefits_map = {
            InteractionType.TRADE: [
                "Resource exchange",
                "Economic growth",
                "Technology transfer potential"
            ],
            InteractionType.DIPLOMACY: [
                "Improved relations",
                "Information sharing",
                "Conflict prevention"
            ],
            InteractionType.CULTURAL_EXCHANGE: [
                "Cultural enrichment",
                "Knowledge sharing",
                "Social cohesion"
            ],
            InteractionType.TECHNOLOGY_TRANSFER: [
                "Technological advancement",
                "Innovation boost",
                "Competitive advantage"
            ],
            InteractionType.ALLIANCE: [
                "Mutual defense",
                "Resource sharing",
                "Political support"
            ],
            InteractionType.FEDERATION: [
                "Unified governance",
                "Shared resources",
                "Collective security"
            ]
        }
        
        return benefits_map.get(interaction_type, ["General cooperation"])
    
    def _generate_interaction_costs(self, interaction_type: InteractionType,
                                  initiator: str, target: str) -> List[str]:
        """Generate costs for interaction."""
        costs_map = {
            InteractionType.TRADE: [
                "Resource expenditure",
                "Transportation costs",
                "Negotiation time"
            ],
            InteractionType.DIPLOMACY: [
                "Diplomatic resources",
                "Information disclosure",
                "Commitment obligations"
            ],
            InteractionType.CONFLICT: [
                "Resource depletion",
                "Population loss",
                "Infrastructure damage"
            ],
            InteractionType.ALLIANCE: [
                "Autonomy reduction",
                "Defense obligations",
                "Resource commitments"
            ],
            InteractionType.FEDERATION: [
                "Sovereignty loss",
                "Governance complexity",
                "Resource pooling"
            ]
        }
        
        return costs_map.get(interaction_type, ["General costs"])
    
    def _update_relationship(self, civilization1: str, civilization2: str, interaction_type: InteractionType):
        """Update relationship between civilizations."""
        if interaction_type in [InteractionType.TRADE, InteractionType.DIPLOMACY, InteractionType.CULTURAL_EXCHANGE]:
            new_relation = CivilizationRelation.FRIENDLY
        elif interaction_type in [InteractionType.ALLIANCE, InteractionType.FEDERATION]:
            new_relation = CivilizationRelation.ALLIED
        elif interaction_type in [InteractionType.CONFLICT, InteractionType.WAR]:
            new_relation = CivilizationRelation.AT_WAR
        else:
            new_relation = CivilizationRelation.NEUTRAL
        
        self.relationship_matrix[(civilization1, civilization2)] = new_relation
        self.relationship_matrix[(civilization2, civilization1)] = new_relation
    
    def respond_to_proposal(self, proposal_id: str, response: str, counter_terms: Optional[Dict[str, Any]] = None) -> bool:
        """Respond to an interaction proposal."""
        if proposal_id not in self.interactions:
            return False
        
        proposal = self.interactions[proposal_id]
        
        if response == "accept":
            proposal.status = InteractionStatus.ACCEPTED
            proposal.final_agreement = proposal.terms.copy()
            if counter_terms:
                proposal.final_agreement.update(counter_terms)
        elif response == "reject":
            proposal.status = InteractionStatus.REJECTED
        elif response == "negotiate":
            proposal.status = InteractionStatus.NEGOTIATING
            if counter_terms:
                proposal.negotiations.append({
                    "timestamp": datetime.now().isoformat(),
                    "counter_terms": counter_terms
                })
        
        proposal.updated_at = datetime.now()
        self._save_interaction(proposal)
        
        return True
    
    def create_federation(self, name: str, founding_civilizations: List[str], governance_structure: str) -> str:
        """Create a federation of civilizations."""
        federation_id = str(uuid.uuid4())
        
        federation = Federation(
            federation_id=federation_id,
            name=name,
            founding_civilizations=founding_civilizations,
            member_civilizations=founding_civilizations.copy(),
            governance_structure=governance_structure
        )
        
        self.federations[federation_id] = federation
        self._save_federation(federation)
        
        # Update relationships between federation members
        for civ1 in founding_civilizations:
            for civ2 in founding_civilizations:
                if civ1 != civ2:
                    self.relationship_matrix[(civ1, civ2)] = CivilizationRelation.FEDERATED
        
        return federation_id
    
    def join_federation(self, federation_id: str, civilization_id: str) -> bool:
        """Add civilization to federation."""
        if federation_id not in self.federations:
            return False
        
        federation = self.federations[federation_id]
        if civilization_id not in federation.member_civilizations:
            federation.member_civilizations.append(civilization_id)
            federation.last_meeting = datetime.now()
            
            # Update relationships
            for member in federation.member_civilizations:
                if member != civilization_id:
                    self.relationship_matrix[(civilization_id, member)] = CivilizationRelation.FEDERATED
                    self.relationship_matrix[(member, civilization_id)] = CivilizationRelation.FEDERATED
            
            self._save_federation(federation)
            return True
        
        return False
    
    def detect_emergent_patterns(self) -> Dict[str, Any]:
        """Detect emergent macro-patterns in civilization interactions."""
        patterns = {
            "hegemony": self._detect_hegemony(),
            "multipolarity": self._detect_multipolarity(),
            "universal_law_evolution": self._detect_universal_law_evolution(),
            "trade_networks": self._analyze_trade_networks(),
            "conflict_clusters": self._analyze_conflict_clusters(),
            "cultural_diffusion": self._analyze_cultural_diffusion()
        }
        
        return patterns
    
    def _detect_hegemony(self) -> Dict[str, Any]:
        """Detect hegemony patterns."""
        if not self.civilization_profiles:
            return {"detected": False, "hegemon": None, "power_ratio": 0.0}
        
        # Find civilization with highest power level
        max_power = max(profile.power_level for profile in self.civilization_profiles.values())
        hegemon = next(
            (civ_id for civ_id, profile in self.civilization_profiles.items() 
             if profile.power_level == max_power), None
        )
        
        # Calculate power ratio
        total_power = sum(profile.power_level for profile in self.civilization_profiles.values())
        power_ratio = max_power / total_power if total_power > 0 else 0.0
        
        return {
            "detected": power_ratio > 0.6,  # 60% threshold for hegemony
            "hegemon": hegemon,
            "power_ratio": power_ratio
        }
    
    def _detect_multipolarity(self) -> Dict[str, Any]:
        """Detect multipolar power distribution."""
        if len(self.civilization_profiles) < 3:
            return {"detected": False, "poles": 0, "balance_ratio": 0.0}
        
        power_levels = [profile.power_level for profile in self.civilization_profiles.values()]
        power_levels.sort(reverse=True)
        
        # Count significant powers (within 20% of top power)
        top_power = power_levels[0]
        significant_powers = sum(1 for power in power_levels if power >= top_power * 0.8)
        
        # Calculate balance ratio
        if len(power_levels) > 1:
            balance_ratio = power_levels[1] / power_levels[0] if power_levels[0] > 0 else 0.0
        else:
            balance_ratio = 0.0
        
        return {
            "detected": significant_powers >= 3 and balance_ratio > 0.7,
            "poles": significant_powers,
            "balance_ratio": balance_ratio
        }
    
    def _detect_universal_law_evolution(self) -> Dict[str, Any]:
        """Detect evolution of universal laws across civilizations."""
        # This would analyze common laws across civilizations
        # For now, return a placeholder
        return {
            "detected": False,
            "common_laws": [],
            "evolution_rate": 0.0
        }
    
    def _analyze_trade_networks(self) -> Dict[str, Any]:
        """Analyze trade network patterns."""
        trade_connections = 0
        for profile in self.civilization_profiles.values():
            trade_connections += len(profile.trade_partners)
        
        total_possible_connections = len(self.civilization_profiles) * (len(self.civilization_profiles) - 1)
        network_density = trade_connections / total_possible_connections if total_possible_connections > 0 else 0.0
        
        return {
            "total_connections": trade_connections,
            "network_density": network_density,
            "hub_civilizations": self._find_trade_hubs()
        }
    
    def _analyze_conflict_clusters(self) -> Dict[str, Any]:
        """Analyze conflict patterns."""
        conflicts = 0
        for (civ1, civ2), relation in self.relationship_matrix.items():
            if relation in [CivilizationRelation.AT_WAR, CivilizationRelation.HOSTILE]:
                conflicts += 1
        
        return {
            "total_conflicts": conflicts,
            "conflict_rate": conflicts / len(self.relationship_matrix) if self.relationship_matrix else 0.0,
            "warring_civilizations": self._find_warring_civilizations()
        }
    
    def _analyze_cultural_diffusion(self) -> Dict[str, Any]:
        """Analyze cultural diffusion patterns."""
        # Analyze shared cultural traits across civilizations
        all_traits = set()
        for profile in self.civilization_profiles.values():
            all_traits.update(profile.cultural_traits)
        
        shared_traits = 0
        for trait in all_traits:
            trait_count = sum(1 for profile in self.civilization_profiles.values() 
                            if trait in profile.cultural_traits)
            if trait_count > 1:
                shared_traits += 1
        
        return {
            "total_traits": len(all_traits),
            "shared_traits": shared_traits,
            "diffusion_rate": shared_traits / len(all_traits) if all_traits else 0.0
        }
    
    def _find_trade_hubs(self) -> List[str]:
        """Find civilizations that are major trade hubs."""
        trade_counts = {}
        for profile in self.civilization_profiles.values():
            trade_counts[profile.civilization_id] = len(profile.trade_partners)
        
        if not trade_counts:
            return []
        
        max_trades = max(trade_counts.values())
        return [civ_id for civ_id, count in trade_counts.items() if count >= max_trades * 0.8]
    
    def _find_warring_civilizations(self) -> List[str]:
        """Find civilizations currently at war."""
        warring = set()
        for (civ1, civ2), relation in self.relationship_matrix.items():
            if relation == CivilizationRelation.AT_WAR:
                warring.add(civ1)
                warring.add(civ2)
        return list(warring)
    
    def get_civilization_relationships(self, civilization_id: str) -> Dict[str, CivilizationRelation]:
        """Get all relationships for a civilization."""
        relationships = {}
        for (civ1, civ2), relation in self.relationship_matrix.items():
            if civ1 == civilization_id:
                relationships[civ2] = relation
        return relationships
    
    def get_interaction_history(self, civilization_id: str) -> List[InteractionProposal]:
        """Get interaction history for a civilization."""
        history = []
        for interaction in self.interactions.values():
            if (interaction.initiator_civilization == civilization_id or 
                interaction.target_civilization == civilization_id):
                history.append(interaction)
        return history
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        total_civilizations = len(self.civilization_profiles)
        total_interactions = len(self.interactions)
        total_federations = len(self.federations)
        
        active_interactions = len([i for i in self.interactions.values() if i.status == InteractionStatus.ACTIVE])
        completed_interactions = len([i for i in self.interactions.values() if i.status == InteractionStatus.COMPLETED])
        
        return {
            "total_civilizations": total_civilizations,
            "total_interactions": total_interactions,
            "active_interactions": active_interactions,
            "completed_interactions": completed_interactions,
            "total_federations": total_federations,
            "average_power_level": statistics.mean([p.power_level for p in self.civilization_profiles.values()]) if self.civilization_profiles else 0.0,
            "emergent_patterns": self.detect_emergent_patterns()
        }
    
    def _save_civilization_profile(self, profile: CivilizationProfile):
        """Save civilization profile to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO civilization_profiles 
            (civilization_id, name, power_level, technology_level, cultural_cohesion,
             economic_strength, military_capability, diplomatic_standing, resources,
             technologies, cultural_traits, political_system, population_size,
             territorial_claims, alliances, enemies, trade_partners, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.civilization_id, profile.name, profile.power_level,
            profile.technology_level, profile.cultural_cohesion,
            profile.economic_strength, profile.military_capability,
            profile.diplomatic_standing, json.dumps(profile.resources),
            json.dumps(profile.technologies), json.dumps(profile.cultural_traits),
            profile.political_system, profile.population_size,
            json.dumps(profile.territorial_claims), json.dumps(profile.alliances),
            json.dumps(profile.enemies), json.dumps(profile.trade_partners),
            profile.last_updated.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _save_interaction(self, interaction: InteractionProposal):
        """Save interaction to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO interactions 
            (proposal_id, interaction_type, initiator_civilization, target_civilization,
             description, terms, benefits, costs, duration, status, created_at,
             expires_at, negotiations, final_agreement)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            interaction.proposal_id, interaction.interaction_type.value,
            interaction.initiator_civilization, interaction.target_civilization,
            interaction.description, json.dumps(interaction.terms),
            json.dumps(interaction.benefits), json.dumps(interaction.costs),
            str(interaction.duration), interaction.status.value,
            interaction.created_at.isoformat(),
            interaction.expires_at.isoformat() if interaction.expires_at else None,
            json.dumps(interaction.negotiations), json.dumps(interaction.final_agreement)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_federation(self, federation: Federation):
        """Save federation to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO federations 
            (federation_id, name, founding_civilizations, member_civilizations,
             governance_structure, shared_resources, common_laws,
             federation_technologies, decision_making_process, created_at, last_meeting)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            federation.federation_id, federation.name,
            json.dumps(federation.founding_civilizations),
            json.dumps(federation.member_civilizations),
            federation.governance_structure, json.dumps(federation.shared_resources),
            json.dumps(federation.common_laws), json.dumps(federation.federation_technologies),
            federation.decision_making_process, federation.created_at.isoformat(),
            federation.last_meeting.isoformat() if federation.last_meeting else None
        ))
        
        conn.commit()
        conn.close()
