"""
Time operations plugin for Egdol DSL.
Provides date and time operations.
"""

from typing import List, Any
from datetime import datetime, timedelta
from . import Plugin


class TimePlugin(Plugin):
    """Plugin providing time operations."""
    
    def __init__(self):
        super().__init__("time")
        self._register_predicates()
        self._register_functions()
        
    def _register_predicates(self):
        """Register time predicates."""
        self.register_predicate("is_today", self._is_today)
        self.register_predicate("is_yesterday", self._is_yesterday)
        self.register_predicate("is_tomorrow", self._is_tomorrow)
        self.register_predicate("is_weekend", self._is_weekend)
        self.register_predicate("is_weekday", self._is_weekday)
        self.register_predicate("is_before", self._is_before)
        self.register_predicate("is_after", self._is_after)
        self.register_predicate("is_same_day", self._is_same_day)
        
    def _register_functions(self):
        """Register time functions."""
        self.register_function("now", self._now)
        self.register_function("today", self._today)
        self.register_function("yesterday", self._yesterday)
        self.register_function("tomorrow", self._tomorrow)
        self.register_function("add_days", self._add_days)
        self.register_function("subtract_days", self._subtract_days)
        self.register_function("format_date", self._format_date)
        self.register_function("parse_date", self._parse_date)
        self.register_function("day_of_week", self._day_of_week)
        self.register_function("days_between", self._days_between)
        
    def _is_today(self, date_str: str) -> bool:
        """Check if date is today."""
        try:
            date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
            return date == datetime.now().date()
        except (ValueError, TypeError):
            return False
            
    def _is_yesterday(self, date_str: str) -> bool:
        """Check if date is yesterday."""
        try:
            date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
            return date == (datetime.now().date() - timedelta(days=1))
        except (ValueError, TypeError):
            return False
            
    def _is_tomorrow(self, date_str: str) -> bool:
        """Check if date is tomorrow."""
        try:
            date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
            return date == (datetime.now().date() + timedelta(days=1))
        except (ValueError, TypeError):
            return False
            
    def _is_weekend(self, date_str: str) -> bool:
        """Check if date is weekend."""
        try:
            date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
            return date.weekday() >= 5  # Saturday = 5, Sunday = 6
        except (ValueError, TypeError):
            return False
            
    def _is_weekday(self, date_str: str) -> bool:
        """Check if date is weekday."""
        try:
            date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
            return date.weekday() < 5  # Monday = 0, Friday = 4
        except (ValueError, TypeError):
            return False
            
    def _is_before(self, date1_str: str, date2_str: str) -> bool:
        """Check if date1 is before date2."""
        try:
            date1 = datetime.strptime(str(date1_str), "%Y-%m-%d").date()
            date2 = datetime.strptime(str(date2_str), "%Y-%m-%d").date()
            return date1 < date2
        except (ValueError, TypeError):
            return False
            
    def _is_after(self, date1_str: str, date2_str: str) -> bool:
        """Check if date1 is after date2."""
        try:
            date1 = datetime.strptime(str(date1_str), "%Y-%m-%d").date()
            date2 = datetime.strptime(str(date2_str), "%Y-%m-%d").date()
            return date1 > date2
        except (ValueError, TypeError):
            return False
            
    def _is_same_day(self, date1_str: str, date2_str: str) -> bool:
        """Check if dates are the same day."""
        try:
            date1 = datetime.strptime(str(date1_str), "%Y-%m-%d").date()
            date2 = datetime.strptime(str(date2_str), "%Y-%m-%d").date()
            return date1 == date2
        except (ValueError, TypeError):
            return False
            
    def _now(self) -> str:
        """Get current datetime."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def _today(self) -> str:
        """Get today's date."""
        return datetime.now().strftime("%Y-%m-%d")
        
    def _yesterday(self) -> str:
        """Get yesterday's date."""
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
    def _tomorrow(self) -> str:
        """Get tomorrow's date."""
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
    def _add_days(self, date_str: str, days: int) -> str:
        """Add days to date."""
        try:
            date = datetime.strptime(str(date_str), "%Y-%m-%d")
            new_date = date + timedelta(days=int(days))
            return new_date.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return str(date_str)
            
    def _subtract_days(self, date_str: str, days: int) -> str:
        """Subtract days from date."""
        try:
            date = datetime.strptime(str(date_str), "%Y-%m-%d")
            new_date = date - timedelta(days=int(days))
            return new_date.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return str(date_str)
            
    def _format_date(self, date_str: str, format_str: str = "%Y-%m-%d") -> str:
        """Format date string."""
        try:
            date = datetime.strptime(str(date_str), "%Y-%m-%d")
            return date.strftime(format_str)
        except (ValueError, TypeError):
            return str(date_str)
            
    def _parse_date(self, date_str: str, format_str: str = "%Y-%m-%d") -> str:
        """Parse date string."""
        try:
            date = datetime.strptime(str(date_str), format_str)
            return date.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return str(date_str)
            
    def _day_of_week(self, date_str: str) -> str:
        """Get day of week for date."""
        try:
            date = datetime.strptime(str(date_str), "%Y-%m-%d")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            return days[date.weekday()]
        except (ValueError, TypeError):
            return "Unknown"
            
    def _days_between(self, date1_str: str, date2_str: str) -> int:
        """Get days between two dates."""
        try:
            date1 = datetime.strptime(str(date1_str), "%Y-%m-%d").date()
            date2 = datetime.strptime(str(date2_str), "%Y-%m-%d").date()
            return abs((date2 - date1).days)
        except (ValueError, TypeError):
            return 0
