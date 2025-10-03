"""
Memory Serializer for Egdol.
Handles serialization of Egdol terms, rules, and facts for persistent storage.
"""

import json
from typing import Any, Dict, List, Union
from ..parser import Term, Variable, Constant, Rule, Fact


class MemorySerializer:
    """Serializes Egdol objects for persistent storage."""
    
    @staticmethod
    def serialize_term(term: Term) -> Dict[str, Any]:
        """Serialize a Term to dictionary."""
        return {
            'type': 'Term',
            'name': term.name,
            'args': [MemorySerializer.serialize_term(arg) if isinstance(arg, Term) 
                     else MemorySerializer.serialize_constant(arg) for arg in term.args]
        }
        
    @staticmethod
    def serialize_constant(constant: Union[Constant, str, int, float]) -> Dict[str, Any]:
        """Serialize a Constant to dictionary."""
        if isinstance(constant, Constant):
            return {
                'type': 'Constant',
                'value': constant.name
            }
        else:
            return {
                'type': 'Constant',
                'value': str(constant)
            }
            
    @staticmethod
    def serialize_variable(variable: Variable) -> Dict[str, Any]:
        """Serialize a Variable to dictionary."""
        return {
            'type': 'Variable',
            'name': variable.name
        }
        
    @staticmethod
    def serialize_rule(rule: Rule) -> Dict[str, Any]:
        """Serialize a Rule to dictionary."""
        return {
            'type': 'Rule',
            'head': MemorySerializer.serialize_term(rule.head),
            'body': [MemorySerializer.serialize_term(term) for term in rule.body]
        }
        
    @staticmethod
    def serialize_fact(fact: Fact) -> Dict[str, Any]:
        """Serialize a Fact to dictionary."""
        return {
            'type': 'Fact',
            'term': MemorySerializer.serialize_term(fact.term)
        }
        
    @staticmethod
    def serialize_any(obj: Any) -> Dict[str, Any]:
        """Serialize any Egdol object to dictionary."""
        if isinstance(obj, Term):
            return MemorySerializer.serialize_term(obj)
        elif isinstance(obj, Constant):
            return MemorySerializer.serialize_constant(obj)
        elif isinstance(obj, Variable):
            return MemorySerializer.serialize_variable(obj)
        elif isinstance(obj, Rule):
            return MemorySerializer.serialize_rule(obj)
        elif isinstance(obj, Fact):
            return MemorySerializer.serialize_fact(obj)
        else:
            return {
                'type': 'Unknown',
                'value': str(obj)
            }
            
    @staticmethod
    def deserialize_term(data: Dict[str, Any]) -> Term:
        """Deserialize a Term from dictionary."""
        if data['type'] != 'Term':
            raise ValueError(f"Expected Term, got {data['type']}")
            
        args = []
        for arg_data in data['args']:
            if arg_data['type'] == 'Term':
                args.append(MemorySerializer.deserialize_term(arg_data))
            elif arg_data['type'] == 'Constant':
                args.append(Constant(arg_data['value']))
            elif arg_data['type'] == 'Variable':
                args.append(Variable(arg_data['name']))
            else:
                args.append(Constant(str(arg_data.get('value', ''))))
                
        return Term(data['name'], args)
        
    @staticmethod
    def deserialize_constant(data: Dict[str, Any]) -> Constant:
        """Deserialize a Constant from dictionary."""
        if data['type'] != 'Constant':
            raise ValueError(f"Expected Constant, got {data['type']}")
        return Constant(data['value'])
        
    @staticmethod
    def deserialize_variable(data: Dict[str, Any]) -> Variable:
        """Deserialize a Variable from dictionary."""
        if data['type'] != 'Variable':
            raise ValueError(f"Expected Variable, got {data['type']}")
        return Variable(data['name'])
        
    @staticmethod
    def deserialize_rule(data: Dict[str, Any]) -> Rule:
        """Deserialize a Rule from dictionary."""
        if data['type'] != 'Rule':
            raise ValueError(f"Expected Rule, got {data['type']}")
            
        head = MemorySerializer.deserialize_term(data['head'])
        body = [MemorySerializer.deserialize_term(term_data) for term_data in data['body']]
        return Rule(head, body)
        
    @staticmethod
    def deserialize_fact(data: Dict[str, Any]) -> Fact:
        """Deserialize a Fact from dictionary."""
        if data['type'] != 'Fact':
            raise ValueError(f"Expected Fact, got {data['type']}")
            
        term = MemorySerializer.deserialize_term(data['term'])
        return Fact(term)
        
    @staticmethod
    def deserialize_any(data: Dict[str, Any]) -> Any:
        """Deserialize any Egdol object from dictionary."""
        obj_type = data.get('type', 'Unknown')
        
        if obj_type == 'Term':
            return MemorySerializer.deserialize_term(data)
        elif obj_type == 'Constant':
            return MemorySerializer.deserialize_constant(data)
        elif obj_type == 'Variable':
            return MemorySerializer.deserialize_variable(data)
        elif obj_type == 'Rule':
            return MemorySerializer.deserialize_rule(data)
        elif obj_type == 'Fact':
            return MemorySerializer.deserialize_fact(data)
        else:
            return data.get('value', str(data))
            
    @staticmethod
    def to_json(obj: Any) -> str:
        """Convert Egdol object to JSON string."""
        return json.dumps(MemorySerializer.serialize_any(obj), indent=2)
        
    @staticmethod
    def from_json(json_str: str) -> Any:
        """Convert JSON string to Egdol object."""
        data = json.loads(json_str)
        return MemorySerializer.deserialize_any(data)
