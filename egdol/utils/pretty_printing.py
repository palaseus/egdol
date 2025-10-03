"""
Pretty Printing Utilities for OmniMind
Enhanced visualization and formatting for complex data structures.
"""

import json
import pprint
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class PrettyPrintStyle(Enum):
    """Pretty printing style options."""
    COMPACT = "compact"
    DETAILED = "detailed"
    COLORFUL = "colorful"
    JSON = "json"
    TABLE = "table"


@dataclass
class PrettyPrintConfig:
    """Configuration for pretty printing."""
    style: PrettyPrintStyle = PrettyPrintStyle.DETAILED
    max_depth: int = 3
    max_items: int = 10
    show_types: bool = True
    show_metadata: bool = True
    colorize: bool = True
    indent: int = 2


class PrettyPrinter:
    """Enhanced pretty printer for OmniMind data structures."""
    
    def __init__(self, config: Optional[PrettyPrintConfig] = None):
        self.config = config or PrettyPrintConfig()
        self.colors = {
            'header': '\033[1;36m',      # Cyan
            'key': '\033[1;34m',         # Blue
            'value': '\033[1;32m',       # Green
            'type': '\033[1;35m',        # Magenta
            'number': '\033[1;33m',      # Yellow
            'string': '\033[1;37m',      # White
            'boolean': '\033[1;31m',     # Red
            'none': '\033[1;30m',        # Gray
            'reset': '\033[0m'           # Reset
        }
    
    def print(self, obj: Any, title: Optional[str] = None, **kwargs) -> str:
        """Pretty print any object with enhanced formatting."""
        if title:
            self._print_header(title)
        
        if self.config.style == PrettyPrintStyle.JSON:
            return self._print_json(obj)
        elif self.config.style == PrettyPrintStyle.TABLE:
            return self._print_table(obj)
        else:
            return self._print_structured(obj, **kwargs)
    
    def _print_header(self, title: str):
        """Print a formatted header."""
        if self.config.colorize:
            print(f"{self.colors['header']}{'='*60}")
            print(f"{title:^60}")
            print(f"{'='*60}{self.colors['reset']}")
        else:
            print(f"{'='*60}")
            print(f"{title:^60}")
            print(f"{'='*60}")
    
    def _print_json(self, obj: Any) -> str:
        """Print object as formatted JSON."""
        try:
            if hasattr(obj, '__dict__'):
                obj_dict = asdict(obj) if hasattr(obj, '__dataclass__') else obj.__dict__
            else:
                obj_dict = obj
            
            formatted = json.dumps(obj_dict, indent=self.config.indent, default=str)
            print(formatted)
            return formatted
        except Exception as e:
            return f"JSON formatting error: {e}"
    
    def _print_table(self, obj: Any) -> str:
        """Print object as a table."""
        if isinstance(obj, (list, tuple)):
            return self._print_list_table(obj)
        elif isinstance(obj, dict):
            return self._print_dict_table(obj)
        else:
            return self._print_structured(obj)
    
    def _print_list_table(self, items: List[Any]) -> str:
        """Print list as table."""
        if not items:
            return "Empty list"
        
        # Get headers from first item
        if hasattr(items[0], '__dict__'):
            headers = list(items[0].__dict__.keys())
        elif isinstance(items[0], dict):
            headers = list(items[0].keys())
        else:
            return self._print_structured(items)
        
        # Print table
        if self.config.colorize:
            header_str = " | ".join(f"{self.colors['header']}{h:^15}{self.colors['reset']}" for h in headers)
        else:
            header_str = " | ".join(f"{h:^15}" for h in headers)
        
        print(header_str)
        print("-" * len(header_str))
        
        for item in items[:self.config.max_items]:
            if hasattr(item, '__dict__'):
                values = [str(getattr(item, h, ''))[:15] for h in headers]
            elif isinstance(item, dict):
                values = [str(item.get(h, ''))[:15] for h in headers]
            else:
                values = [str(item)[:15]]
            
            if self.config.colorize:
                row_str = " | ".join(f"{self.colors['value']}{v:^15}{self.colors['reset']}" for v in values)
            else:
                row_str = " | ".join(f"{v:^15}" for v in values)
            print(row_str)
        
        if len(items) > self.config.max_items:
            print(f"... and {len(items) - self.config.max_items} more items")
        
        return "Table printed"
    
    def _print_dict_table(self, data: Dict[str, Any]) -> str:
        """Print dictionary as table."""
        if not data:
            return "Empty dictionary"
        
        max_key_len = max(len(str(k)) for k in data.keys())
        max_value_len = 50
        
        for key, value in list(data.items())[:self.config.max_items]:
            key_str = str(key).ljust(max_key_len)
            value_str = str(value)[:max_value_len]
            
            if self.config.colorize:
                print(f"{self.colors['key']}{key_str}{self.colors['reset']} : {self.colors['value']}{value_str}{self.colors['reset']}")
            else:
                print(f"{key_str} : {value_str}")
        
        if len(data) > self.config.max_items:
            print(f"... and {len(data) - self.config.max_items} more items")
        
        return "Dictionary table printed"
    
    def _print_structured(self, obj: Any, depth: int = 0, **kwargs) -> str:
        """Print object with structured formatting."""
        indent = " " * (depth * self.config.indent)
        
        if depth > self.config.max_depth:
            return f"{indent}... (max depth reached)"
        
        if obj is None:
            return self._colorize(f"{indent}None", 'none')
        elif isinstance(obj, bool):
            return self._colorize(f"{indent}{obj}", 'boolean')
        elif isinstance(obj, (int, float, np.number)):
            return self._colorize(f"{indent}{obj}", 'number')
        elif isinstance(obj, str):
            return self._colorize(f"{indent}'{obj}'", 'string')
        elif isinstance(obj, (list, tuple)):
            return self._print_sequence(obj, depth, **kwargs)
        elif isinstance(obj, dict):
            return self._print_mapping(obj, depth, **kwargs)
        elif isinstance(obj, Enum):
            return self._colorize(f"{indent}{obj.__class__.__name__}.{obj.name}", 'type')
        elif hasattr(obj, '__dict__'):
            return self._print_object(obj, depth, **kwargs)
        else:
            return self._colorize(f"{indent}{type(obj).__name__}: {obj}", 'type')
    
    def _print_sequence(self, seq: Union[List, tuple], depth: int, **kwargs) -> str:
        """Print sequence (list/tuple) with formatting."""
        indent = " " * (depth * self.config.indent)
        seq_type = "List" if isinstance(seq, list) else "Tuple"
        
        if not seq:
            return self._colorize(f"{indent}{seq_type}[]", 'type')
        
        result = [self._colorize(f"{indent}{seq_type}[", 'type')]
        
        for i, item in enumerate(seq[:self.config.max_items]):
            item_str = self._print_structured(item, depth + 1, **kwargs)
            result.append(item_str)
        
        if len(seq) > self.config.max_items:
            result.append(f"{indent}  ... and {len(seq) - self.config.max_items} more items")
        
        result.append(self._colorize(f"{indent}]", 'type'))
        return "\n".join(result)
    
    def _print_mapping(self, mapping: Dict[str, Any], depth: int, **kwargs) -> str:
        """Print mapping (dict) with formatting."""
        indent = " " * (depth * self.config.indent)
        
        if not mapping:
            return self._colorize(f"{indent}Dict{{}}", 'type')
        
        result = [self._colorize(f"{indent}Dict{{", 'type')]
        
        for i, (key, value) in enumerate(list(mapping.items())[:self.config.max_items]):
            key_str = self._colorize(f"'{key}'", 'key')
            value_str = self._print_structured(value, depth + 1, **kwargs)
            result.append(f"{indent}  {key_str}: {value_str}")
        
        if len(mapping) > self.config.max_items:
            result.append(f"{indent}  ... and {len(mapping) - self.config.max_items} more items")
        
        result.append(self._colorize(f"{indent}}}", 'type'))
        return "\n".join(result)
    
    def _print_object(self, obj: Any, depth: int, **kwargs) -> str:
        """Print object with attributes."""
        indent = " " * (depth * self.config.indent)
        class_name = obj.__class__.__name__
        
        result = [self._colorize(f"{indent}{class_name}(", 'type')]
        
        # Get attributes
        attrs = {}
        if hasattr(obj, '__dict__'):
            attrs = obj.__dict__
        elif hasattr(obj, '__dataclass_fields__'):
            attrs = {field: getattr(obj, field) for field in obj.__dataclass_fields__}
        
        for i, (key, value) in enumerate(list(attrs.items())[:self.config.max_items]):
            key_str = self._colorize(f"'{key}'", 'key')
            value_str = self._print_structured(value, depth + 1, **kwargs)
            result.append(f"{indent}  {key_str}={value_str}")
        
        if len(attrs) > self.config.max_items:
            result.append(f"{indent}  ... and {len(attrs) - self.config.max_items} more attributes")
        
        result.append(self._colorize(f"{indent})", 'type'))
        return "\n".join(result)
    
    def _colorize(self, text: str, color_type: str) -> str:
        """Apply color formatting to text."""
        if not self.config.colorize:
            return text
        
        color = self.colors.get(color_type, self.colors['reset'])
        return f"{color}{text}{self.colors['reset']}"


# Global pretty printer instance
pp = PrettyPrinter()


def pretty_print(obj: Any, title: Optional[str] = None, style: PrettyPrintStyle = PrettyPrintStyle.DETAILED, **kwargs):
    """Convenience function for pretty printing."""
    config = PrettyPrintConfig(style=style, **kwargs)
    printer = PrettyPrinter(config)
    return printer.print(obj, title, **kwargs)


def print_civilization(civilization, title: str = "Civilization Overview"):
    """Pretty print a civilization with specialized formatting."""
    pp.print({
        'id': civilization.id,
        'name': civilization.name,
        'archetype': civilization.archetype.name if hasattr(civilization.archetype, 'name') else str(civilization.archetype),
        'population': len(civilization.agent_clusters),
        'governance': civilization.governance_model.name if hasattr(civilization.governance_model, 'name') else str(civilization.governance_model),
        'cooperation_level': civilization.cooperation_level,
        'environment': {
            'climate_stability': civilization.environment.climate_stability,
            'natural_hazards': civilization.environment.natural_hazards,
            'technological_opportunities': civilization.environment.technological_opportunities
        },
        'temporal_state': {
            'current_tick': civilization.temporal_state.current_tick,
            'simulation_time': civilization.temporal_state.simulation_time,
            'major_events': len(civilization.temporal_state.major_events)
        }
    }, title)


def print_evolution_metrics(metrics, title: str = "Evolution Metrics"):
    """Pretty print evolution metrics."""
    pp.print({
        'complexity': metrics.complexity,
        'stability': metrics.stability,
        'adaptability': metrics.adaptability,
        'innovation_rate': metrics.innovation_rate,
        'cooperation_level': metrics.cooperation_level,
        'resource_efficiency': metrics.resource_efficiency,
        'governance_effectiveness': metrics.governance_effectiveness,
        'cultural_cohesion': metrics.cultural_cohesion,
        'technological_advancement': metrics.technological_advancement,
        'environmental_harmony': metrics.environmental_harmony
    }, title)


def print_pattern_analysis(patterns, title: str = "Pattern Analysis"):
    """Pretty print pattern analysis results."""
    if not patterns:
        print("No patterns detected")
        return
    
    pattern_data = {}
    for pattern_id, pattern in patterns.items():
        pattern_data[pattern_id] = {
            'type': pattern.pattern_type.name if hasattr(pattern.pattern_type, 'name') else str(pattern.pattern_type),
            'significance': pattern.significance.name if hasattr(pattern.significance, 'name') else str(pattern.significance),
            'confidence': pattern.confidence,
            'description': pattern.description[:100] + "..." if len(pattern.description) > 100 else pattern.description
        }
    
    pp.print(pattern_data, title)


def print_experiment_results(results, title: str = "Experiment Results"):
    """Pretty print experiment results."""
    if isinstance(results, list):
        for i, result in enumerate(results):
            pp.print({
                'experiment_id': result.experiment_id,
                'status': result.status.name if hasattr(result.status, 'name') else str(result.status),
                'success': result.success,
                'duration': result.duration,
                'metrics': {
                    'complexity_change': result.metrics.get('complexity_change', 0),
                    'stability_change': result.metrics.get('stability_change', 0),
                    'innovation_change': result.metrics.get('innovation_change', 0)
                }
            }, f"{title} #{i+1}")
    else:
        pp.print({
            'experiment_id': results.experiment_id,
            'status': results.status.name if hasattr(results.status, 'name') else str(results.status),
            'success': results.success,
            'duration': results.duration,
            'metrics': results.metrics
        }, title)


def print_universe_status(universe, title: str = "Universe Status"):
    """Pretty print universe status."""
    pp.print({
        'universe_id': universe.universe_id,
        'status': universe.status.name if hasattr(universe.status, 'name') else str(universe.status),
        'universe_type': universe.universe_type.name if hasattr(universe.universe_type, 'name') else str(universe.universe_type),
        'civilizations': len(universe.civilizations),
        'simulation_time': universe.simulation_time,
        'performance_metrics': universe.performance_metrics
    }, title)
