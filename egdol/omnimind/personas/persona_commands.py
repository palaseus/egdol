"""
Persona Commands for OmniMind
Command-line interface for persona management.
"""

from typing import Dict, Any, List, Optional
from .persona import PersonaManager, PersonaType
from .domain_packs import get_all_domain_packs


class PersonaCommands:
    """Command-line interface for persona management."""
    
    def __init__(self, persona_manager: PersonaManager):
        self.persona_manager = persona_manager
        self.domain_packs = get_all_domain_packs()
        
    def create_persona(self, name: str, description: str, persona_type: str,
                      skills: List[str], **kwargs) -> Dict[str, Any]:
        """Create a new persona."""
        try:
            # Convert string to PersonaType
            persona_type_enum = PersonaType[persona_type.upper()]
            
            persona = self.persona_manager.create_persona(
                name=name,
                description=description,
                persona_type=persona_type_enum,
                skills=skills,
                **kwargs
            )
            
            return {
                'success': True,
                'persona_id': persona.id,
                'message': f"Created persona '{name}'"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def create_domain_persona(self, domain_pack_name: str) -> Dict[str, Any]:
        """Create a persona from a domain pack."""
        try:
            # Find domain pack
            domain_pack = None
            for pack in self.domain_packs:
                if pack.name.lower() == domain_pack_name.lower():
                    domain_pack = pack
                    break
                    
            if not domain_pack:
                return {
                    'success': False,
                    'error': f"Domain pack '{domain_pack_name}' not found"
                }
                
            # Create persona from domain pack
            persona = domain_pack.create_persona(self.persona_manager)
            
            return {
                'success': True,
                'persona_id': persona.id,
                'message': f"Created {domain_pack.name} persona"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def switch_persona(self, persona_identifier: str) -> Dict[str, Any]:
        """Switch to a different persona."""
        try:
            # Try by ID first
            if self.persona_manager.get_persona(persona_identifier):
                result = self.persona_manager.switch_persona(persona_identifier)
            else:
                # Try by name
                persona = self.persona_manager.get_persona_by_name(persona_identifier)
                if persona:
                    result = self.persona_manager.switch_persona(persona.id)
                else:
                    return {
                        'success': False,
                        'error': f"Persona '{persona_identifier}' not found"
                    }
                    
            if result:
                active_persona = self.persona_manager.get_active_persona()
                return {
                    'success': True,
                    'message': f"Switched to persona '{active_persona.name}'"
                }
            else:
                return {
                    'success': False,
                    'error': "Failed to switch persona"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def list_personas(self) -> Dict[str, Any]:
        """List all personas."""
        try:
            personas = self.persona_manager.list_personas()
            active_persona = self.persona_manager.get_active_persona()
            
            return {
                'success': True,
                'personas': personas,
                'active_persona': active_persona.name if active_persona else None,
                'total_count': len(personas)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def get_persona_info(self, persona_identifier: str) -> Dict[str, Any]:
        """Get information about a specific persona."""
        try:
            # Try by ID first
            persona = self.persona_manager.get_persona(persona_identifier)
            if not persona:
                # Try by name
                persona = self.persona_manager.get_persona_by_name(persona_identifier)
                
            if not persona:
                return {
                    'success': False,
                    'error': f"Persona '{persona_identifier}' not found"
                }
                
            return {
                'success': True,
                'persona': persona.to_dict()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def delete_persona(self, persona_identifier: str) -> Dict[str, Any]:
        """Delete a persona."""
        try:
            # Try by ID first
            persona = self.persona_manager.get_persona(persona_identifier)
            if not persona:
                # Try by name
                persona = self.persona_manager.get_persona_by_name(persona_identifier)
                
            if not persona:
                return {
                    'success': False,
                    'error': f"Persona '{persona_identifier}' not found"
                }
                
            result = self.persona_manager.delete_persona(persona.id)
            
            if result:
                return {
                    'success': True,
                    'message': f"Deleted persona '{persona.name}'"
                }
            else:
                return {
                    'success': False,
                    'error': "Failed to delete persona (may be active)"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def update_persona(self, persona_identifier: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a persona."""
        try:
            # Try by ID first
            persona = self.persona_manager.get_persona(persona_identifier)
            if not persona:
                # Try by name
                persona = self.persona_manager.get_persona_by_name(persona_identifier)
                
            if not persona:
                return {
                    'success': False,
                    'error': f"Persona '{persona_identifier}' not found"
                }
                
            result = self.persona_manager.update_persona(persona.id, updates)
            
            if result:
                return {
                    'success': True,
                    'message': f"Updated persona '{persona.name}'"
                }
            else:
                return {
                    'success': False,
                    'error': "Failed to update persona"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def get_best_persona(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get the best persona for a query."""
        try:
            best_persona = self.persona_manager.get_best_persona(query, context)
            
            if best_persona:
                return {
                    'success': True,
                    'persona': best_persona.to_dict(),
                    'message': f"Best persona for query: {best_persona.name}"
                }
            else:
                return {
                    'success': False,
                    'error': "No suitable persona found"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def auto_switch_persona(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Automatically switch to the best persona for a query."""
        try:
            result = self.persona_manager.auto_switch_persona(query, context)
            
            if result:
                active_persona = self.persona_manager.get_active_persona()
                return {
                    'success': True,
                    'message': f"Auto-switched to persona '{active_persona.name}'"
                }
            else:
                return {
                    'success': False,
                    'message': "No persona switch needed"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def get_stats(self) -> Dict[str, Any]:
        """Get persona statistics."""
        try:
            stats = self.persona_manager.get_persona_stats()
            return {
                'success': True,
                'stats': stats
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def get_available_domain_packs(self) -> Dict[str, Any]:
        """Get available domain packs."""
        try:
            pack_info = []
            for pack in self.domain_packs:
                pack_info.append({
                    'name': pack.name,
                    'description': pack.description
                })
                
            return {
                'success': True,
                'domain_packs': pack_info
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
