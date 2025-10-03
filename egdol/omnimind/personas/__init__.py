"""
Persona/Domain Pack System for OmniMind
Manages specialized personas with different skills and knowledge.
"""

from .persona import Persona, PersonaManager, PersonaType
from .domain_packs import DomainPack, LegalExpert, CodingAssistant, Historian, Strategist
from .persona_commands import PersonaCommands

__all__ = ['Persona', 'PersonaManager', 'PersonaType', 'DomainPack', 'LegalExpert', 'CodingAssistant', 'Historian', 'Strategist', 'PersonaCommands']
