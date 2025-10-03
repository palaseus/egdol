"""
Skill System for OmniMind
Modular skills that handle different domains.
"""

from .base import BaseSkill, SkillManager
from .math_skill import MathSkill
from .logic_skill import LogicSkill
from .general_skill import GeneralSkill
from .code_skill import CodeSkill
from .file_skill import FileSkill

__all__ = [
    'BaseSkill', 'SkillManager',
    'MathSkill', 'LogicSkill', 'GeneralSkill', 
    'CodeSkill', 'FileSkill'
]
