"""
Commander's Console
Control layer for strategic directives to civilizations and personalities.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import asyncio
from collections import defaultdict, deque

from .multi_agent_system import MultiAgentCivilizationSystem, AgentMessage, MessageType
from .toolforge import Toolforge, ToolType, ToolStatus
from .world_persistence import WorldPersistence, EventType, EventImportance
from .emergent_metrics import EmergentMetrics, MetricType
from ..conversational.personality_framework import Personality, PersonalityType


class DirectiveType(Enum):
    """Directive type enumeration."""
    STRATEGIC = auto()
    TACTICAL = auto()
    DIPLOMATIC = auto()
    TECHNOLOGICAL = auto()
    CULTURAL = auto()
    ECONOMIC = auto()
    ENVIRONMENTAL = auto()
    EMERGENCY = auto()


class DirectivePriority(Enum):
    """Directive priority enumeration."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass
class CommanderDirective:
    """Commander's directive to civilization."""
    directive_id: str
    directive_type: DirectiveType
    priority: DirectivePriority
    title: str
    description: str
    target_civilization: str
    target_agents: List[str] = field(default_factory=list)  # Empty = all agents
    parameters: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    success_criteria: List[str] = field(default_factory=list)
    issued_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, in_progress, completed, failed, cancelled
    progress: float = 0.0  # 0.0 to 1.0
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "directive_id": self.directive_id,
            "directive_type": self.directive_type.name,
            "priority": self.priority.name,
            "title": self.title,
            "description": self.description,
            "target_civilization": self.target_civilization,
            "target_agents": self.target_agents,
            "parameters": self.parameters,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "success_criteria": self.success_criteria,
            "issued_at": self.issued_at.isoformat(),
            "status": self.status,
            "progress": self.progress,
            "results": self.results
        }


@dataclass
class EventInjection:
    """Event injection for simulation."""
    event_id: str
    event_type: EventType
    importance: EventImportance
    title: str
    description: str
    target_world: str
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    consequences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scheduled_time: Optional[datetime] = None
    executed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "importance": self.importance.name,
            "title": self.title,
            "description": self.description,
            "target_world": self.target_world,
            "participants": self.participants,
            "location": self.location,
            "consequences": self.consequences,
            "metadata": self.metadata,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "executed": self.executed
        }


@dataclass
class EvolutionGuide:
    """Guide for emergent evolution."""
    guide_id: str
    target_personality: str
    evolution_direction: str  # "more_aggressive", "more_collaborative", etc.
    intensity: float  # 0.0 to 1.0
    duration: timedelta
    success_metrics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "guide_id": self.guide_id,
            "target_personality": self.target_personality,
            "evolution_direction": self.evolution_direction,
            "intensity": self.intensity,
            "duration": str(self.duration),
            "success_metrics": self.success_metrics,
            "created_at": self.created_at.isoformat(),
            "active": self.active
        }


class CommandConsole:
    """Commander's console for strategic control."""
    
    def __init__(self):
        self.multi_agent_system = MultiAgentCivilizationSystem()
        self.toolforge = Toolforge()
        self.world_persistence = WorldPersistence()
        self.emergent_metrics = EmergentMetrics()
        
        self.directives: Dict[str, CommanderDirective] = {}
        self.event_injections: Dict[str, EventInjection] = {}
        self.evolution_guides: Dict[str, EvolutionGuide] = {}
        self.observation_logs: deque = deque(maxlen=10000)
        self.running = False
        self.console_thread = None
    
    def start_console(self):
        """Start the command console."""
        self.running = True
        self.console_thread = asyncio.create_task(self._console_loop())
    
    def stop_console(self):
        """Stop the command console."""
        self.running = False
        if self.console_thread:
            self.console_thread.cancel()
    
    async def _console_loop(self):
        """Main console loop."""
        while self.running:
            try:
                # Process pending directives
                await self._process_directives()
                
                # Execute scheduled event injections
                await self._execute_event_injections()
                
                # Apply evolution guides
                await self._apply_evolution_guides()
                
                # Update metrics
                self.emergent_metrics.update_metrics()
                
                # Small delay
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"Error in console loop: {e}")
                await asyncio.sleep(1.0)
    
    def issue_directive(self,
                       directive_type: DirectiveType,
                       priority: DirectivePriority,
                       title: str,
                       description: str,
                       target_civilization: str,
                       target_agents: List[str] = None,
                       parameters: Dict[str, Any] = None,
                       deadline: Optional[datetime] = None,
                       success_criteria: List[str] = None) -> str:
        """Issue a new directive."""
        directive_id = str(uuid.uuid4())
        
        directive = CommanderDirective(
            directive_id=directive_id,
            directive_type=directive_type,
            priority=priority,
            title=title,
            description=description,
            target_civilization=target_civilization,
            target_agents=target_agents or [],
            parameters=parameters or {},
            deadline=deadline,
            success_criteria=success_criteria or []
        )
        
        self.directives[directive_id] = directive
        
        # Log directive
        self._log_observation(f"Directive issued: {title}", {
            "directive_id": directive_id,
            "type": directive_type.name,
            "priority": priority.name
        })
        
        return directive_id
    
    def inject_event(self,
                    event_type: EventType,
                    importance: EventImportance,
                    title: str,
                    description: str,
                    target_world: str,
                    participants: List[str] = None,
                    location: str = None,
                    consequences: List[str] = None,
                    scheduled_time: Optional[datetime] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """Inject an event into the simulation."""
        event_id = str(uuid.uuid4())
        
        event_injection = EventInjection(
            event_id=event_id,
            event_type=event_type,
            importance=importance,
            title=title,
            description=description,
            target_world=target_world,
            participants=participants or [],
            location=location,
            consequences=consequences or [],
            metadata=metadata or {},
            scheduled_time=scheduled_time
        )
        
        self.event_injections[event_id] = event_injection
        
        # Log injection
        self._log_observation(f"Event injected: {title}", {
            "event_id": event_id,
            "type": event_type.name,
            "importance": importance.name
        })
        
        return event_id
    
    def guide_evolution(self,
                       target_personality: str,
                       evolution_direction: str,
                       intensity: float,
                       duration: timedelta,
                       success_metrics: List[str] = None) -> str:
        """Guide the evolution of a personality."""
        guide_id = str(uuid.uuid4())
        
        guide = EvolutionGuide(
            guide_id=guide_id,
            target_personality=target_personality,
            evolution_direction=evolution_direction,
            intensity=intensity,
            duration=duration,
            success_metrics=success_metrics or []
        )
        
        self.evolution_guides[guide_id] = guide
        
        # Log guide
        self._log_observation(f"Evolution guide created: {evolution_direction}", {
            "guide_id": guide_id,
            "target": target_personality,
            "direction": evolution_direction,
            "intensity": intensity
        })
        
        return guide_id
    
    async def _process_directives(self):
        """Process pending directives."""
        for directive in self.directives.values():
            if directive.status == "pending":
                await self._execute_directive(directive)
            elif directive.status == "in_progress":
                await self._update_directive_progress(directive)
    
    async def _execute_directive(self, directive: CommanderDirective):
        """Execute a directive."""
        directive.status = "in_progress"
        
        # Get target civilization
        if directive.target_civilization not in self.multi_agent_system.civilizations:
            directive.status = "failed"
            directive.results.append({"error": "Civilization not found"})
            return
        
        civilization = self.multi_agent_system.civilizations[directive.target_civilization]
        
        # Create directive message
        message_content = f"DIRECTIVE: {directive.title}\n{directive.description}"
        if directive.parameters:
            message_content += f"\nParameters: {json.dumps(directive.parameters)}"
        
        # Send to target agents
        target_agents = directive.target_agents if directive.target_agents else list(civilization.agents.keys())
        
        for agent_id in target_agents:
            if agent_id in civilization.agents:
                message = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id="commander",
                    receiver_id=agent_id,
                    message_type=MessageType.PROPOSAL,
                    content=message_content,
                    priority=directive.priority.value,
                    metadata={
                        "directive_id": directive.directive_id,
                        "directive_type": directive.directive_type.name
                    }
                )
                
                self.multi_agent_system.message_bus.publish(message)
        
        # Log execution
        self._log_observation(f"Directive executed: {directive.title}", {
            "directive_id": directive.directive_id,
            "target_agents": len(target_agents)
        })
    
    async def _update_directive_progress(self, directive: CommanderDirective):
        """Update directive progress."""
        # Simple progress calculation based on time elapsed
        time_elapsed = datetime.now() - directive.issued_at
        max_duration = timedelta(hours=1)  # Default max duration
        
        if directive.deadline:
            max_duration = directive.deadline - directive.issued_at
        
        progress = min(1.0, time_elapsed.total_seconds() / max_duration.total_seconds())
        directive.progress = progress
        
        # Check if completed
        if progress >= 1.0:
            directive.status = "completed"
            directive.results.append({"status": "completed", "timestamp": datetime.now().isoformat()})
            
            self._log_observation(f"Directive completed: {directive.title}", {
                "directive_id": directive.directive_id,
                "progress": progress
            })
    
    async def _execute_event_injections(self):
        """Execute scheduled event injections."""
        current_time = datetime.now()
        
        for event_injection in self.event_injections.values():
            if not event_injection.executed:
                should_execute = True
                
                if event_injection.scheduled_time:
                    should_execute = current_time >= event_injection.scheduled_time
                
                if should_execute:
                    # Inject event into world
                    event_id = self.world_persistence.add_event(
                        world_id=event_injection.target_world,
                        event_type=event_injection.event_type,
                        importance=event_injection.importance,
                        title=event_injection.title,
                        description=event_injection.description,
                        participants=event_injection.participants,
                        location=event_injection.location,
                        consequences=event_injection.consequences,
                        metadata=event_injection.metadata
                    )
                    
                    event_injection.executed = True
                    
                    self._log_observation(f"Event injected: {event_injection.title}", {
                        "event_id": event_id,
                        "injection_id": event_injection.event_id
                    })
    
    async def _apply_evolution_guides(self):
        """Apply evolution guides to personalities."""
        current_time = datetime.now()
        
        for guide in self.evolution_guides.values():
            if guide.active:
                # Check if guide has expired
                if current_time - guide.created_at > guide.duration:
                    guide.active = False
                    continue
                
                # Apply evolution to target personality
                await self._apply_personality_evolution(guide)
    
    async def _apply_personality_evolution(self, guide: EvolutionGuide):
        """Apply personality evolution based on guide."""
        # This is a simplified implementation
        # In a real system, this would modify personality traits over time
        
        evolution_strength = guide.intensity * 0.1  # Small incremental changes
        
        # Log evolution application
        self._log_observation(f"Evolution applied: {guide.evolution_direction}", {
            "guide_id": guide.guide_id,
            "target": guide.target_personality,
            "strength": evolution_strength
        })
    
    def observe_agent_debates(self, civilization_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Observe agent debates."""
        if civilization_id not in self.multi_agent_system.civilizations:
            return []
        
        # Get recent messages
        messages = self.multi_agent_system.get_message_history(limit)
        
        # Filter for debate messages
        debate_messages = [
            msg for msg in messages
            if msg.get("message_type") == "DEBATE"
        ]
        
        return debate_messages
    
    def observe_consensus_building(self, civilization_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Observe consensus building processes."""
        if civilization_id not in self.multi_agent_system.civilizations:
            return []
        
        # Get recent messages
        messages = self.multi_agent_system.get_message_history(limit)
        
        # Filter for consensus messages
        consensus_messages = [
            msg for msg in messages
            if msg.get("message_type") in ["CONSENSUS", "TREATY", "COLLABORATION"]
        ]
        
        return consensus_messages
    
    def observe_conflict_resolution(self, civilization_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Observe conflict resolution processes."""
        if civilization_id not in self.multi_agent_system.civilizations:
            return []
        
        # Get recent messages
        messages = self.multi_agent_system.get_message_history(limit)
        
        # Filter for conflict-related messages
        conflict_messages = [
            msg for msg in messages
            if msg.get("message_type") in ["COMPETITION", "CHALLENGE"]
        ]
        
        return conflict_messages
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "civilizations": len(self.multi_agent_system.civilizations),
            "active_directives": len([d for d in self.directives.values() if d.status == "in_progress"]),
            "pending_directives": len([d for d in self.directives.values() if d.status == "pending"]),
            "scheduled_events": len([e for e in self.event_injections.values() if not e.executed]),
            "active_guides": len([g for g in self.evolution_guides.values() if g.active]),
            "total_observations": len(self.observation_logs),
            "metrics_dashboard": self.emergent_metrics.get_metrics_dashboard()
        }
    
    def get_directive_status(self, directive_id: str) -> Dict[str, Any]:
        """Get status of a specific directive."""
        if directive_id not in self.directives:
            return {"error": "Directive not found"}
        
        return self.directives[directive_id].to_dict()
    
    def cancel_directive(self, directive_id: str) -> bool:
        """Cancel a directive."""
        if directive_id not in self.directives:
            return False
        
        directive = self.directives[directive_id]
        if directive.status in ["completed", "failed"]:
            return False
        
        directive.status = "cancelled"
        directive.results.append({"status": "cancelled", "timestamp": datetime.now().isoformat()})
        
        self._log_observation(f"Directive cancelled: {directive.title}", {
            "directive_id": directive_id
        })
        
        return True
    
    def get_observation_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get observation log."""
        recent_observations = list(self.observation_logs)[-limit:]
        return [obs for obs in recent_observations]
    
    def _log_observation(self, message: str, metadata: Dict[str, Any] = None):
        """Log an observation."""
        observation = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "metadata": metadata or {}
        }
        
        self.observation_logs.append(observation)
    
    def emergency_intervention(self, civilization_id: str, intervention_type: str, parameters: Dict[str, Any] = None) -> str:
        """Emergency intervention in civilization."""
        intervention_id = str(uuid.uuid4())
        
        # Create emergency directive
        directive_id = self.issue_directive(
            directive_type=DirectiveType.EMERGENCY,
            priority=DirectivePriority.EMERGENCY,
            title=f"Emergency Intervention: {intervention_type}",
            description=f"Emergency intervention of type {intervention_type}",
            target_civilization=civilization_id,
            parameters=parameters or {}
        )
        
        # Log emergency
        self._log_observation(f"Emergency intervention: {intervention_type}", {
            "intervention_id": intervention_id,
            "civilization_id": civilization_id,
            "type": intervention_type,
            "directive_id": directive_id
        })
        
        return intervention_id
    
    def create_civilization_snapshot(self, civilization_id: str) -> str:
        """Create snapshot of civilization state."""
        snapshot_id = str(uuid.uuid4())
        
        if civilization_id not in self.multi_agent_system.civilizations:
            return None
        
        civilization = self.multi_agent_system.civilizations[civilization_id]
        
        # Create snapshot data
        snapshot_data = {
            "snapshot_id": snapshot_id,
            "civilization_id": civilization_id,
            "timestamp": datetime.now().isoformat(),
            "civilization_state": civilization.to_dict(),
            "agent_memories": {
                agent_id: agent.memory.to_dict()
                for agent_id, agent in civilization.agents.items()
            },
            "message_history": self.multi_agent_system.get_message_history(100),
            "active_directives": [
                d.to_dict() for d in self.directives.values()
                if d.target_civilization == civilization_id
            ]
        }
        
        # Store snapshot (in a real system, this would be persisted)
        self._log_observation(f"Civilization snapshot created: {civilization_id}", {
            "snapshot_id": snapshot_id,
            "civilization_id": civilization_id
        })
        
        return snapshot_id
