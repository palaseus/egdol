"""
Communication System for OmniMind Network
Handles inter-agent messaging and coordination.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque


class MessageType(Enum):
    """Types of messages in the network."""
    QUERY = auto()
    RESPONSE = auto()
    TASK_DELEGATION = auto()
    KNOWLEDGE_SHARE = auto()
    SKILL_REQUEST = auto()
    COORDINATION = auto()
    EMERGENCY = auto()
    HEARTBEAT = auto()
    LEARNING_UPDATE = auto()
    OPTIMIZATION_REQUEST = auto()


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class Message:
    """A message in the network."""
    id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: float
    expires_at: Optional[float] = None
    requires_response: bool = False
    response_to: Optional[str] = None
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type.name,
            'priority': self.priority.name,
            'content': self.content,
            'timestamp': self.timestamp,
            'expires_at': self.expires_at,
            'requires_response': self.requires_response,
            'response_to': self.response_to,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }


class MessageBus:
    """Message bus for inter-agent communication."""
    
    def __init__(self, network_id: str):
        self.network_id = network_id
        self.messages: Dict[str, Message] = {}
        self.message_queues: Dict[str, deque] = defaultdict(deque)
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.response_handlers: Dict[str, Callable] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.delivery_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
    def send_message(self, sender_id: str, recipient_id: str, 
                    message_type: MessageType, content: Dict[str, Any],
                    priority: MessagePriority = MessagePriority.NORMAL,
                    expires_in: Optional[float] = None,
                    requires_response: bool = False,
                    response_to: Optional[str] = None,
                    confidence_score: float = 1.0) -> str:
        """Send a message between agents."""
        message_id = str(uuid.uuid4())
        
        message = Message(
            id=message_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            priority=priority,
            content=content,
            timestamp=time.time(),
            expires_at=time.time() + expires_in if expires_in else None,
            requires_response=requires_response,
            response_to=response_to,
            confidence_score=confidence_score
        )
        
        # Store message
        self.messages[message_id] = message
        
        # Add to recipient's queue (priority-based)
        self._add_to_queue(recipient_id, message)
        
        # Log message
        self._log_message(message)
        
        # Update delivery stats
        self.delivery_stats[sender_id]['sent'] += 1
        self.delivery_stats[recipient_id]['received'] += 1
        
        return message_id
        
    def broadcast_message(self, sender_id: str, recipient_ids: List[str],
                         message_type: MessageType, content: Dict[str, Any],
                         priority: MessagePriority = MessagePriority.NORMAL,
                         expires_in: Optional[float] = None,
                         requires_response: bool = False,
                         confidence_score: float = 1.0) -> List[str]:
        """Broadcast a message to multiple agents."""
        message_ids = []
        
        for recipient_id in recipient_ids:
            message_id = self.send_message(
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=message_type,
                content=content,
                priority=priority,
                expires_in=expires_in,
                requires_response=requires_response,
                confidence_score=confidence_score
            )
            message_ids.append(message_id)
            
        return message_ids
        
    def get_messages_for_agent(self, agent_id: str, 
                              message_type: Optional[MessageType] = None,
                              priority: Optional[MessagePriority] = None) -> List[Message]:
        """Get messages for an agent."""
        messages = []
        
        # Get from queue
        if agent_id in self.message_queues:
            queue = self.message_queues[agent_id]
            
            # Process messages in priority order
            while queue:
                message = queue.popleft()
                
                # Check if expired
                if message.is_expired():
                    continue
                    
                # Filter by type and priority
                if message_type and message.message_type != message_type:
                    continue
                if priority and message.priority != priority:
                    continue
                    
                messages.append(message)
                
        return messages
        
    def mark_message_delivered(self, message_id: str, agent_id: str) -> bool:
        """Mark a message as delivered."""
        if message_id not in self.messages:
            return False
            
        message = self.messages[message_id]
        if message.recipient_id != agent_id:
            return False
            
        # Update delivery stats
        self.delivery_stats[agent_id]['delivered'] += 1
        
        return True
        
    def send_response(self, original_message_id: str, sender_id: str,
                     content: Dict[str, Any], confidence_score: float = 1.0) -> str:
        """Send a response to a message."""
        if original_message_id not in self.messages:
            return None
            
        original_message = self.messages[original_message_id]
        
        response_id = self.send_message(
            sender_id=sender_id,
            recipient_id=original_message.sender_id,
            message_type=MessageType.RESPONSE,
            content=content,
            priority=original_message.priority,
            requires_response=False,
            response_to=original_message_id,
            confidence_score=confidence_score
        )
        
        return response_id
        
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler."""
        self.message_handlers[message_type].append(handler)
        
    def register_response_handler(self, message_id: str, handler: Callable):
        """Register a response handler for a specific message."""
        self.response_handlers[message_id] = handler
        
    def process_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Process messages for an agent."""
        messages = self.get_messages_for_agent(agent_id)
        results = []
        
        for message in messages:
            # Mark as delivered
            self.mark_message_delivered(message.id, agent_id)
            
            # Process message
            result = self._process_message(message)
            results.append(result)
            
        return results
        
    def _process_message(self, message: Message) -> Dict[str, Any]:
        """Process a single message."""
        result = {
            'message_id': message.id,
            'processed': False,
            'response_sent': False,
            'error': None
        }
        
        try:
            # Get handlers for message type
            handlers = self.message_handlers.get(message.message_type, [])
            
            # Process with handlers
            for handler in handlers:
                handler_result = handler(message)
                if handler_result:
                    result['processed'] = True
                    break
                    
            # Handle response if required
            if message.requires_response and message.response_to:
                response_handler = self.response_handlers.get(message.response_to)
                if response_handler:
                    response_result = response_handler(message)
                    if response_result:
                        result['response_sent'] = True
                        
        except Exception as e:
            result['error'] = str(e)
            
        return result
        
    def _add_to_queue(self, agent_id: str, message: Message):
        """Add message to agent's queue with priority ordering."""
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = deque()
            
        queue = self.message_queues[agent_id]
        
        # Insert based on priority
        priority_order = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3
        }
        
        message_priority = priority_order[message.priority]
        
        # Find insertion point
        insert_index = 0
        for i, existing_message in enumerate(queue):
            existing_priority = priority_order[existing_message.priority]
            if message_priority < existing_priority:
                insert_index = i
                break
            insert_index = i + 1
            
        # Insert message
        queue.insert(insert_index, message)
        
    def _log_message(self, message: Message):
        """Log a message."""
        log_entry = {
            'timestamp': time.time(),
            'message': message.to_dict()
        }
        self.message_history.append(log_entry)
        
    def get_message_statistics(self) -> Dict[str, Any]:
        """Get message statistics."""
        total_messages = len(self.messages)
        total_delivered = sum(stats['delivered'] for stats in self.delivery_stats.values())
        total_sent = sum(stats['sent'] for stats in self.delivery_stats.values())
        total_received = sum(stats['received'] for stats in self.delivery_stats.values())
        
        # Calculate delivery rate
        delivery_rate = total_delivered / total_sent if total_sent > 0 else 0
        
        # Calculate message type distribution
        type_distribution = defaultdict(int)
        for message in self.messages.values():
            type_distribution[message.message_type.name] += 1
            
        # Calculate priority distribution
        priority_distribution = defaultdict(int)
        for message in self.messages.values():
            priority_distribution[message.priority.name] += 1
            
        return {
            'total_messages': total_messages,
            'total_delivered': total_delivered,
            'total_sent': total_sent,
            'total_received': total_received,
            'delivery_rate': delivery_rate,
            'type_distribution': dict(type_distribution),
            'priority_distribution': dict(priority_distribution),
            'agent_stats': dict(self.delivery_stats)
        }
        
    def get_message_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get message history."""
        return list(self.message_history[-limit:])
        
    def clear_message_history(self):
        """Clear message history."""
        self.message_history.clear()
        
    def get_network_communication_graph(self) -> Dict[str, Any]:
        """Get communication graph between agents."""
        communication_graph = defaultdict(lambda: defaultdict(int))
        
        for message in self.messages.values():
            sender = message.sender_id
            recipient = message.recipient_id
            communication_graph[sender][recipient] += 1
            
        return dict(communication_graph)
        
    def detect_communication_patterns(self) -> List[Dict[str, Any]]:
        """Detect communication patterns in the network."""
        patterns = []
        
        # Analyze message flow
        communication_graph = self.get_network_communication_graph()
        
        # Find highly connected agents
        for sender, recipients in communication_graph.items():
            total_communications = sum(recipients.values())
            if total_communications > 10:  # Threshold for high communication
                patterns.append({
                    'type': 'high_communication_agent',
                    'agent_id': sender,
                    'communication_count': total_communications,
                    'description': f'Agent {sender} has high communication activity'
                })
                
        # Find communication bottlenecks
        for sender, recipients in communication_graph.items():
            if len(recipients) == 1:  # Only communicates with one agent
                patterns.append({
                    'type': 'communication_bottleneck',
                    'agent_id': sender,
                    'description': f'Agent {sender} has limited communication'
                })
                
        # Find communication clusters
        cluster_sizes = {}
        for sender, recipients in communication_graph.items():
            cluster_size = len(recipients)
            cluster_sizes[cluster_size] = cluster_sizes.get(cluster_size, 0) + 1
            
        for size, count in cluster_sizes.items():
            if count > 1:  # Multiple agents with same cluster size
                patterns.append({
                    'type': 'communication_cluster',
                    'cluster_size': size,
                    'agent_count': count,
                    'description': f'{count} agents form communication clusters of size {size}'
                })
                
        return patterns
        
    def optimize_message_routing(self) -> Dict[str, Any]:
        """Optimize message routing in the network."""
        optimizations = []
        
        # Analyze message delivery rates
        stats = self.get_message_statistics()
        delivery_rate = stats['delivery_rate']
        
        if delivery_rate < 0.8:  # Low delivery rate
            optimizations.append({
                'type': 'improve_delivery_rate',
                'description': f'Message delivery rate is {delivery_rate:.2f}, needs improvement',
                'priority': 'high'
            })
            
        # Analyze message type distribution
        type_dist = stats['type_distribution']
        if type_dist.get('EMERGENCY', 0) > 5:  # Too many emergency messages
            optimizations.append({
                'type': 'reduce_emergency_messages',
                'description': 'High number of emergency messages indicates system instability',
                'priority': 'high'
            })
            
        # Analyze priority distribution
        priority_dist = stats['priority_distribution']
        if priority_dist.get('CRITICAL', 0) > 10:  # Too many critical messages
            optimizations.append({
                'type': 'reduce_critical_messages',
                'description': 'High number of critical messages may indicate system overload',
                'priority': 'medium'
            })
            
        return {
            'optimizations': optimizations,
            'total_optimizations': len(optimizations),
            'high_priority': len([opt for opt in optimizations if opt['priority'] == 'high']),
            'medium_priority': len([opt for opt in optimizations if opt['priority'] == 'medium'])
        }

