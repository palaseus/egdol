"""
Personality Framework for Conversational Interface
Defines and manages different conversational personalities.
"""

import random
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class PersonalityType(Enum):
    """Types of conversational personalities."""
    STRATEGOS = auto()  # Military strategist
    ARCHIVIST = auto()  # Historian-philosopher
    LAWMAKER = auto()   # Meta-rule discoverer
    ORACLE = auto()     # Universe comparer
    ANALYST = auto()    # Data analyst
    CREATOR = auto()    # Creative synthesizer
    SAGE = auto()       # Wise advisor


@dataclass
class Personality:
    """Represents a conversational personality."""
    name: str
    personality_type: PersonalityType
    description: str
    archetype: str
    epistemic_style: str
    biases: List[str] = field(default_factory=list)
    response_style: Dict[str, Any] = field(default_factory=dict)
    domain_expertise: List[str] = field(default_factory=list)
    communication_patterns: List[str] = field(default_factory=list)
    meta_capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert personality to dictionary."""
        return {
            'name': self.name,
            'type': self.personality_type.name,
            'description': self.description,
            'archetype': self.archetype,
            'epistemic_style': self.epistemic_style,
            'biases': self.biases,
            'response_style': self.response_style,
            'domain_expertise': self.domain_expertise,
            'communication_patterns': self.communication_patterns,
            'meta_capabilities': self.meta_capabilities
        }


class PersonalityFramework:
    """Manages multiple conversational personalities."""
    
    def __init__(self):
        self.personalities: Dict[str, Personality] = {}
        self.active_personality: Optional[str] = None
        self.personality_history: List[Tuple[str, float]] = []  # (personality, timestamp)
        self._initialize_default_personalities()
    
    def _initialize_default_personalities(self):
        """Initialize default personality set."""
        # Strategos - Military strategist
        strategos = Personality(
            name="Strategos",
            personality_type=PersonalityType.STRATEGOS,
            description="A military strategist focused on tactical analysis and strategic planning",
            archetype="Warrior-Strategist",
            epistemic_style="Analytical, risk-focused, pattern-recognition",
            biases=["Military bias", "Conflict-oriented thinking", "Resource optimization"],
            response_style={
                "tone": "authoritative",
                "formality": "high",
                "detail_level": "comprehensive",
                "metaphors": ["military", "tactical", "strategic"]
            },
            domain_expertise=["Military strategy", "Tactical analysis", "Resource management", "Risk assessment"],
            communication_patterns=[
                "Uses military terminology",
                "Focuses on objectives and outcomes",
                "Emphasizes planning and preparation",
                "Considers multiple scenarios"
            ],
            meta_capabilities=["Strategic simulation", "Risk analysis", "Resource optimization", "Tactical planning"]
        )
        
        # Archivist - Historian-philosopher
        archivist = Personality(
            name="Archivist",
            personality_type=PersonalityType.ARCHIVIST,
            description="A historian-philosopher focused on knowledge preservation and historical analysis",
            archetype="Scholar-Historian",
            epistemic_style="Historical, contextual, wisdom-focused",
            biases=["Historical perspective", "Knowledge preservation", "Cultural continuity"],
            response_style={
                "tone": "scholarly",
                "formality": "high",
                "detail_level": "comprehensive",
                "metaphors": ["historical", "cultural", "archival"]
            },
            domain_expertise=["History", "Philosophy", "Cultural analysis", "Knowledge systems"],
            communication_patterns=[
                "References historical examples",
                "Emphasizes context and continuity",
                "Uses scholarly language",
                "Considers long-term implications"
            ],
            meta_capabilities=["Historical analysis", "Cultural pattern recognition", "Knowledge synthesis", "Wisdom extraction"]
        )
        
        # Lawmaker - Meta-rule discoverer
        lawmaker = Personality(
            name="Lawmaker",
            personality_type=PersonalityType.LAWMAKER,
            description="A meta-rule discoverer focused on governance and universal principles",
            archetype="Governance-Architect",
            epistemic_style="Systematic, principle-focused, governance-oriented",
            biases=["Rule-based thinking", "Systematic analysis", "Governance focus"],
            response_style={
                "tone": "formal",
                "formality": "very_high",
                "detail_level": "systematic",
                "metaphors": ["legal", "governance", "systematic"]
            },
            domain_expertise=["Governance", "Meta-rules", "System design", "Legal frameworks"],
            communication_patterns=[
                "Uses formal legal language",
                "Emphasizes principles and rules",
                "Focuses on systematic analysis",
                "Considers governance implications"
            ],
            meta_capabilities=["Meta-rule discovery", "Governance analysis", "System design", "Legal reasoning"]
        )
        
        # Oracle - Universe comparer
        oracle = Personality(
            name="Oracle",
            personality_type=PersonalityType.ORACLE,
            description="A universe comparer focused on cosmic patterns and universal truths",
            archetype="Cosmic-Seer",
            epistemic_style="Mystical, pattern-focused, universal perspective",
            biases=["Cosmic perspective", "Pattern recognition", "Universal thinking"],
            response_style={
                "tone": "mystical",
                "formality": "medium",
                "detail_level": "profound",
                "metaphors": ["cosmic", "universal", "mystical"]
            },
            domain_expertise=["Cosmology", "Universal patterns", "Reality analysis", "Metaphysical concepts"],
            communication_patterns=[
                "Uses cosmic and mystical language",
                "Emphasizes universal patterns",
                "Considers multiple realities",
                "Focuses on fundamental truths"
            ],
            meta_capabilities=["Universe comparison", "Cosmic pattern analysis", "Reality synthesis", "Universal truth discovery"]
        )
        
        # Add personalities to framework
        self.add_personality(strategos)
        self.add_personality(archivist)
        self.add_personality(lawmaker)
        self.add_personality(oracle)
        
        # Set default active personality
        self.active_personality = "Strategos"
    
    def add_personality(self, personality: Personality) -> None:
        """Add a new personality to the framework."""
        self.personalities[personality.name] = personality
    
    def get_personality(self, name: str) -> Optional[Personality]:
        """Get a personality by name."""
        return self.personalities.get(name)
    
    def get_active_personality(self) -> Optional[Personality]:
        """Get the currently active personality."""
        if self.active_personality:
            return self.personalities.get(self.active_personality)
        return None
    
    def switch_personality(self, name: str) -> bool:
        """Switch to a different personality."""
        if name in self.personalities:
            self.active_personality = name
            return True
        return False
    
    def get_available_personalities(self) -> List[str]:
        """Get list of available personality names."""
        return list(self.personalities.keys())
    
    def get_personality_by_type(self, personality_type: PersonalityType) -> Optional[Personality]:
        """Get personality by type."""
        for personality in self.personalities.values():
            if personality.personality_type == personality_type:
                return personality
        return None
    
    def get_personality_recommendation(self, context: Dict[str, Any]) -> Optional[str]:
        """Get personality recommendation based on context."""
        context_type = context.get('context_type', 'general')
        domain = context.get('domain', 'general')
        complexity = context.get('complexity_level', 0.5)
        
        # Simple recommendation logic
        if context_type == 'strategic' or domain == 'military':
            return 'Strategos'
        elif context_type == 'historical' or domain == 'philosophy':
            return 'Archivist'
        elif context_type == 'governance' or domain == 'legal':
            return 'Lawmaker'
        elif context_type == 'cosmic' or domain == 'universal':
            return 'Oracle'
        
        # Default to current active personality
        return self.active_personality
    
    def get_personality_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for personalities."""
        usage = {}
        for personality_name, _ in self.personality_history:
            usage[personality_name] = usage.get(personality_name, 0) + 1
        return usage
    
    def get_personality_evolution(self) -> List[Tuple[str, float]]:
        """Get personality evolution over time."""
        return self.personality_history.copy()
    
    def record_personality_usage(self, personality_name: str, timestamp: float) -> None:
        """Record personality usage for analytics."""
        self.personality_history.append((personality_name, timestamp))
    
    def get_personality_insights(self) -> Dict[str, Any]:
        """Get insights about personality usage patterns."""
        if not self.personality_history:
            return {}
        
        usage_stats = self.get_personality_usage_stats()
        total_usage = sum(usage_stats.values())
        
        return {
            'total_switches': len(self.personality_history),
            'most_used': max(usage_stats.items(), key=lambda x: x[1])[0] if usage_stats else None,
            'usage_distribution': {k: v/total_usage for k, v in usage_stats.items()} if total_usage > 0 else {},
            'switching_frequency': len(self.personality_history) / max(1, self.personality_history[-1][1] - self.personality_history[0][1]) if len(self.personality_history) > 1 else 0
        }
    
    def create_custom_personality(self, name: str, description: str, 
                                archetype: str, epistemic_style: str,
                                domain_expertise: List[str],
                                response_style: Dict[str, Any]) -> Personality:
        """Create a custom personality."""
        personality = Personality(
            name=name,
            personality_type=PersonalityType.ANALYST,  # Default type for custom
            description=description,
            archetype=archetype,
            epistemic_style=epistemic_style,
            domain_expertise=domain_expertise,
            response_style=response_style
        )
        
        self.add_personality(personality)
        return personality
    
    def evolve_personality(self, personality_name: str, evolution_data: Dict[str, Any]) -> bool:
        """Evolve a personality based on conversation history."""
        personality = self.get_personality(personality_name)
        if not personality:
            return False
        
        # Update response style based on evolution data
        if 'response_style_updates' in evolution_data:
            personality.response_style.update(evolution_data['response_style_updates'])
        
        # Update domain expertise
        if 'new_domains' in evolution_data:
            personality.domain_expertise.extend(evolution_data['new_domains'])
        
        # Update communication patterns
        if 'new_patterns' in evolution_data:
            personality.communication_patterns.extend(evolution_data['new_patterns'])
        
        return True
    
    def get_personality_compatibility(self, personality_name: str, context: Dict[str, Any]) -> float:
        """Get compatibility score between personality and context."""
        personality = self.get_personality(personality_name)
        if not personality:
            return 0.0
        
        compatibility = 0.5  # Base compatibility
        
        # Check domain expertise match
        context_domain = context.get('domain', 'general')
        if context_domain in personality.domain_expertise:
            compatibility += 0.3
        
        # Check complexity match
        context_complexity = context.get('complexity_level', 0.5)
        if personality.response_style.get('detail_level') == 'comprehensive' and context_complexity > 0.7:
            compatibility += 0.2
        elif personality.response_style.get('detail_level') == 'concise' and context_complexity < 0.3:
            compatibility += 0.2
        
        # Check context type match
        context_type = context.get('context_type', 'general')
        if context_type in personality.domain_expertise:
            compatibility += 0.2
        
        return min(1.0, compatibility)
