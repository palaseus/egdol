"""
String operations plugin for Egdol DSL.
Provides string manipulation and comparison operations.
"""

from typing import List, Any
from . import Plugin


class StringOpsPlugin(Plugin):
    """Plugin providing string operations."""
    
    def __init__(self):
        super().__init__("string_ops")
        self._register_predicates()
        self._register_functions()
        
    def _register_predicates(self):
        """Register string predicates."""
        self.register_predicate("contains", self._contains)
        self.register_predicate("starts_with", self._starts_with)
        self.register_predicate("ends_with", self._ends_with)
        self.register_predicate("matches", self._matches)
        self.register_predicate("is_empty", self._is_empty)
        self.register_predicate("is_numeric", self._is_numeric)
        self.register_predicate("is_alpha", self._is_alpha)
        self.register_predicate("is_alnum", self._is_alnum)
        
    def _register_functions(self):
        """Register string functions."""
        self.register_function("length", self._length)
        self.register_function("upper", self._upper)
        self.register_function("lower", self._lower)
        self.register_function("strip", self._strip)
        self.register_function("replace", self._replace)
        self.register_function("split", self._split)
        self.register_function("join", self._join)
        self.register_function("substring", self._substring)
        self.register_function("reverse", self._reverse)
        
    def _contains(self, text: str, substring: str) -> bool:
        """Check if text contains substring."""
        return substring in str(text)
        
    def _starts_with(self, text: str, prefix: str) -> bool:
        """Check if text starts with prefix."""
        return str(text).startswith(str(prefix))
        
    def _ends_with(self, text: str, suffix: str) -> bool:
        """Check if text ends with suffix."""
        return str(text).endswith(str(suffix))
        
    def _matches(self, text: str, pattern: str) -> bool:
        """Check if text matches pattern (simple string matching)."""
        return str(text) == str(pattern)
        
    def _is_empty(self, text: str) -> bool:
        """Check if text is empty."""
        return len(str(text).strip()) == 0
        
    def _is_numeric(self, text: str) -> bool:
        """Check if text is numeric."""
        try:
            float(text)
            return True
        except (ValueError, TypeError):
            return False
            
    def _is_alpha(self, text: str) -> bool:
        """Check if text contains only alphabetic characters."""
        return str(text).isalpha()
        
    def _is_alnum(self, text: str) -> bool:
        """Check if text contains only alphanumeric characters."""
        return str(text).isalnum()
        
    def _length(self, text: str) -> int:
        """Get length of text."""
        return len(str(text))
        
    def _upper(self, text: str) -> str:
        """Convert text to uppercase."""
        return str(text).upper()
        
    def _lower(self, text: str) -> str:
        """Convert text to lowercase."""
        return str(text).lower()
        
    def _strip(self, text: str) -> str:
        """Strip whitespace from text."""
        return str(text).strip()
        
    def _replace(self, text: str, old: str, new: str) -> str:
        """Replace old with new in text."""
        return str(text).replace(str(old), str(new))
        
    def _split(self, text: str, delimiter: str = " ") -> List[str]:
        """Split text by delimiter."""
        return str(text).split(str(delimiter))
        
    def _join(self, parts: List[str], delimiter: str = " ") -> str:
        """Join parts with delimiter."""
        if isinstance(parts, str):
            return parts
        return str(delimiter).join(str(part) for part in parts)
        
    def _substring(self, text: str, start: int, end: int = None) -> str:
        """Get substring from start to end."""
        text_str = str(text)
        if end is None:
            return text_str[start:]
        return text_str[start:end]
        
    def _reverse(self, text: str) -> str:
        """Reverse text."""
        return str(text)[::-1]
