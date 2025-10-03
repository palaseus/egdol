"""
Communication system for Egdol agents.
Provides message passing and coordination between agents.
"""

import time
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto


class MessageType(Enum):
    """Types of messages between agents."""
    TEXT = auto()
    QUERY = auto()
    RESPONSE = auto()
    FACT = auto()
    RULE = auto()
    COMMAND = auto()
    NOTIFICATION = auto()


@dataclass
class Message:
    """Message between agents."""
    sender: str
    recipient: str
    content: str
    message_type: str = "text"
    timestamp: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'content': self.content,
            'message_type': self.message_type,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            sender=data['sender'],
            recipient=data['recipient'],
            content=data['content'],
            message_type=data.get('message_type', 'text'),
            timestamp=data.get('timestamp', time.time()),
            metadata=data.get('metadata', {})
        )


class MessageBus:
    """Message bus for agent communication."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
    def send(self, message: Message):
        """Send a message."""
        self.messages.append(message)
        
        # Notify subscribers
        if message.recipient in self.subscribers:
            for handler in self.subscribers[message.recipient]:
                try:
                    handler(message)
                except Exception as e:
                    print(f"Error in message handler: {e}")
                    
        # Handle message type
        if message.message_type in self.message_handlers:
            try:
                self.message_handlers[message.message_type](message)
            except Exception as e:
                print(f"Error in message type handler: {e}")
                
    def subscribe(self, agent_name: str, handler: Callable):
        """Subscribe an agent to receive messages."""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(handler)
        
    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
        
    def get_messages_for(self, agent_name: str, limit: int = 100) -> List[Message]:
        """Get messages for a specific agent."""
        messages = [m for m in self.messages if m.recipient == agent_name]
        return messages[-limit:] if limit else messages
        
    def get_messages_from(self, agent_name: str, limit: int = 100) -> List[Message]:
        """Get messages from a specific agent."""
        messages = [m for m in self.messages if m.sender == agent_name]
        return messages[-limit:] if limit else messages
        
    def get_conversation(self, agent1: str, agent2: str, limit: int = 100) -> List[Message]:
        """Get conversation between two agents."""
        messages = []
        for message in self.messages:
            if ((message.sender == agent1 and message.recipient == agent2) or
                (message.sender == agent2 and message.recipient == agent1)):
                messages.append(message)
                
        return messages[-limit:] if limit else messages
        
    def clear_messages(self, older_than: float = None):
        """Clear messages, optionally only those older than a timestamp."""
        if older_than is None:
            self.messages.clear()
        else:
            self.messages = [m for m in self.messages if m.timestamp > older_than]
            
    def get_message_stats(self) -> Dict[str, Any]:
        """Get message statistics."""
        if not self.messages:
            return {'total_messages': 0}
            
        # Count by type
        by_type = {}
        for message in self.messages:
            msg_type = message.message_type
            by_type[msg_type] = by_type.get(msg_type, 0) + 1
            
        # Count by sender
        by_sender = {}
        for message in self.messages:
            sender = message.sender
            by_sender[sender] = by_sender.get(sender, 0) + 1
            
        # Recent activity
        recent_messages = len([m for m in self.messages if m.timestamp > time.time() - 3600])
        
        return {
            'total_messages': len(self.messages),
            'by_type': by_type,
            'by_sender': by_sender,
            'recent_messages': recent_messages,
            'active_agents': len(set(m.sender for m in self.messages))
        }
        
    def export_messages(self, file_path: str) -> bool:
        """Export messages to a file."""
        try:
            data = {
                'export_timestamp': time.time(),
                'total_messages': len(self.messages),
                'messages': [message.to_dict() for message in self.messages]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False
            
    def import_messages(self, file_path: str) -> int:
        """Import messages from a file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            imported_count = 0
            for message_data in data.get('messages', []):
                message = Message.from_dict(message_data)
                self.messages.append(message)
                imported_count += 1
                
            return imported_count
        except Exception:
            return 0


class AgentCoordinator:
    """Coordinates communication between multiple agents."""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, Any] = {}
        self.coordination_rules: List[Dict[str, Any]] = []
        
    def register_agent(self, agent: Any):
        """Register an agent with the coordinator."""
        self.agents[agent.profile.name] = agent
        
        # Subscribe to messages
        self.message_bus.subscribe(agent.profile.name, agent.receive_message)
        
    def unregister_agent(self, agent_name: str):
        """Unregister an agent."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            
    def broadcast(self, sender: str, content: str, message_type: str = "notification"):
        """Broadcast a message to all agents."""
        for agent_name in self.agents:
            if agent_name != sender:
                message = Message(
                    sender=sender,
                    recipient=agent_name,
                    content=content,
                    message_type=message_type
                )
                self.message_bus.send(message)
                
    def create_agent_network(self, agents: List[Any]) -> Dict[str, Any]:
        """Create a network of agents."""
        network = {}
        
        for agent in agents:
            self.register_agent(agent)
            network[agent.profile.name] = {
                'agent': agent,
                'connections': []
            }
            
        # Create connections based on expertise
        for agent1 in agents:
            for agent2 in agents:
                if agent1 != agent2:
                    # Check if agents have complementary expertise
                    if self._are_complementary(agent1, agent2):
                        network[agent1.profile.name]['connections'].append(agent2.profile.name)
                        
        return network
        
    def _are_complementary(self, agent1: Any, agent2: Any) -> bool:
        """Check if two agents have complementary expertise."""
        expertise1 = set(agent1.profile.expertise)
        expertise2 = set(agent2.profile.expertise)
        
        # Agents are complementary if they have different expertise
        return len(expertise1.intersection(expertise2)) == 0 and len(expertise1) > 0 and len(expertise2) > 0
        
    def coordinate_task(self, task: str, required_expertise: List[str]) -> List[str]:
        """Coordinate a task among agents with required expertise."""
        suitable_agents = []
        
        for agent_name, agent_info in self.agents.items():
            agent = agent_info
            if hasattr(agent, 'profile'):
                agent_expertise = set(agent.profile.expertise)
                required_set = set(required_expertise)
                
                if required_set.intersection(agent_expertise):
                    suitable_agents.append(agent_name)
                    
        return suitable_agents
        
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return {
            'total_agents': len(self.agents),
            'message_stats': self.message_bus.get_message_stats(),
            'coordination_rules': len(self.coordination_rules)
        }
