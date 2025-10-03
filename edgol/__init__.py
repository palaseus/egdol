"""Compatibility shim: `edgol` package that re-exports the real implementation from `egdol`.

This allows older imports like `from edgol.rules_engine import RulesEngine`
to continue working after renaming the real package to `egdol`.
"""
from importlib import import_module
_mod = import_module('egdol')
# copy public attributes from egdol into this package namespace
for _name in dir(_mod):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_mod, _name)

try:
    __all__ = _mod.__all__
except Exception:
    __all__ = [n for n in globals().keys() if not n.startswith('_')]
"""egdol package initializer"""

from .lexer import Token, Lexer, LexerError

__all__ = ["Token", "Lexer", "LexerError"]
