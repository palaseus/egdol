"""
Meta Layer for Egdol.
Provides introspection, reflection, and self-modification capabilities.
"""

from .inspector import MemoryInspector, RuleInspector
from .scorer import RuleScorer, ConfidenceTracker
from .modifier import SelfModifier, RuleModifier

__all__ = [
    'MemoryInspector', 'RuleInspector',
    'RuleScorer', 'ConfidenceTracker', 
    'SelfModifier', 'RuleModifier'
]
