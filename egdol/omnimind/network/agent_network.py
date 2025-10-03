"""
Agent Network for OmniMind
Manages multiple OmniMind agents in a networked ecosystem.
"""

import time
import uuid
import threading
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque


class NetworkTopology(Enum):
    """Network topology types."""
    STAR = auto()
    MESH = auto()
    RING = auto()
    HIERARCHICAL = auto()
    AD_HOC = auto()


class AgentStatus(Enum):
    """Agent status in the network."""
    ACTIVE = auto()
    IDLE = auto()
    BUSY = auto()
    OFFLINE = auto()
    LEARNING = auto()
    OPTIMIZING = auto()


@dataclass
class NetworkAgent:
    """A networked OmniMind agent."""
    id: str
    name: str
    persona_type: str
    skills: List[str]
    memory_manager: Any
    skill_router: Any
    planner: Any
    introspector: Any
    status: AgentStatus = AgentStatus.IDLE
    created_at: float = None
    last_activity: float = None
    performance_metrics: Dict[str, Any] = None
    knowledge_graph: Dict[str, Any] = None
    communication_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_activity is None:
            self.last_activity = time.time()
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.knowledge_graph is None:
            self.knowledge_graph = {}
        if self.communication_history is None:
            self.communication_history = []
            
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
        
    def update_performance(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        self.performance_metrics.update(metrics)
        
    def add_communication(self, message: Dict[str, Any]):
        """Add communication to history."""
        self.communication_history.append({
            'timestamp': time.time(),
            'message': message
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'persona_type': self.persona_type,
            'skills': self.skills,
            'status': self.status.name,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'performance_metrics': self.performance_metrics,
            'knowledge_graph_size': len(self.knowledge_graph),
            'communication_count': len(self.communication_history)
        }


class AgentNetwork:
    """Manages a network of OmniMind agents."""
    
    def __init__(self, network_id: str = None, topology: NetworkTopology = NetworkTopology.MESH):
        self.network_id = network_id or str(uuid.uuid4())
        self.topology = topology
        self.agents: Dict[str, NetworkAgent] = {}
        self.connections: Dict[str, Set[str]] = defaultdict(set)
        self.message_queue: deque = deque()
        self.network_events: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        
    def add_agent(self, name: str, persona_type: str, skills: List[str],
                  memory_manager=None, skill_router=None, planner=None,
                  introspector=None) -> NetworkAgent:
        """Add a new agent to the network."""
        with self.lock:
            agent_id = str(uuid.uuid4())
            
            agent = NetworkAgent(
                id=agent_id,
                name=name,
                persona_type=persona_type,
                skills=skills,
                memory_manager=memory_manager,
                skill_router=skill_router,
                planner=planner,
                introspector=introspector
            )
            
            self.agents[agent_id] = agent
            
            # Establish connections based on topology
            self._establish_connections(agent_id)
            
            # Log network event
            self._log_network_event('agent_added', {
                'agent_id': agent_id,
                'name': name,
                'persona_type': persona_type,
                'skills_count': len(skills)
            })
            
            return agent
            
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the network."""
        with self.lock:
            if agent_id not in self.agents:
                return False
                
            # Remove connections
            if agent_id in self.connections:
                del self.connections[agent_id]
                
            # Remove from other agents' connections
            for other_agent_id in self.agents:
                if agent_id in self.connections[other_agent_id]:
                    self.connections[other_agent_id].remove(agent_id)
                    
            # Remove agent
            del self.agents[agent_id]
            
            # Log network event
            self._log_network_event('agent_removed', {
                'agent_id': agent_id
            })
            
            return True
            
    def get_agent(self, agent_id: str) -> Optional[NetworkAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
        
    def get_agents_by_persona(self, persona_type: str) -> List[NetworkAgent]:
        """Get all agents with a specific persona type."""
        return [agent for agent in self.agents.values() 
                if agent.persona_type == persona_type]
        
    def get_agents_by_skill(self, skill: str) -> List[NetworkAgent]:
        """Get all agents with a specific skill."""
        return [agent for agent in self.agents.values() 
                if skill in agent.skills]
        
    def get_connected_agents(self, agent_id: str) -> List[NetworkAgent]:
        """Get agents connected to a specific agent."""
        if agent_id not in self.connections:
            return []
            
        connected_ids = self.connections[agent_id]
        return [self.agents[aid] for aid in connected_ids if aid in self.agents]
        
    def connect_agents(self, agent1_id: str, agent2_id: str) -> bool:
        """Connect two agents."""
        with self.lock:
            if agent1_id not in self.agents or agent2_id not in self.agents:
                return False
                
            self.connections[agent1_id].add(agent2_id)
            self.connections[agent2_id].add(agent1_id)
            
            # Log network event
            self._log_network_event('agents_connected', {
                'agent1_id': agent1_id,
                'agent2_id': agent2_id
            })
            
            return True
            
    def disconnect_agents(self, agent1_id: str, agent2_id: str) -> bool:
        """Disconnect two agents."""
        with self.lock:
            if agent1_id not in self.connections or agent2_id not in self.connections:
                return False
                
            self.connections[agent1_id].discard(agent2_id)
            self.connections[agent2_id].discard(agent1_id)
            
            # Log network event
            self._log_network_event('agents_disconnected', {
                'agent1_id': agent1_id,
                'agent2_id': agent2_id
            })
            
            return True
            
    def broadcast_message(self, sender_id: str, message: Dict[str, Any], 
                         target_agents: List[str] = None) -> int:
        """Broadcast a message to agents."""
        with self.lock:
            if sender_id not in self.agents:
                return 0
                
            # Determine target agents
            if target_agents is None:
                target_agents = list(self.agents.keys())
            else:
                target_agents = [aid for aid in target_agents if aid in self.agents]
                
            # Add message to queue
            message_entry = {
                'id': str(uuid.uuid4()),
                'sender_id': sender_id,
                'target_agents': target_agents,
                'message': message,
                'timestamp': time.time(),
                'delivered': False
            }
            
            self.message_queue.append(message_entry)
            
            # Log network event
            self._log_network_event('message_broadcast', {
                'sender_id': sender_id,
                'target_count': len(target_agents),
                'message_type': message.get('type', 'unknown')
            })
            
            return len(target_agents)
            
    def get_messages_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get pending messages for an agent."""
        with self.lock:
            messages = []
            
            for message_entry in self.message_queue:
                if (agent_id in message_entry['target_agents'] and 
                    not message_entry['delivered']):
                    messages.append({
                        'id': message_entry['id'],
                        'sender_id': message_entry['sender_id'],
                        'message': message_entry['message'],
                        'timestamp': message_entry['timestamp']
                    })
                    
            return messages
            
    def mark_message_delivered(self, message_id: str, agent_id: str) -> bool:
        """Mark a message as delivered to an agent."""
        with self.lock:
            for message_entry in self.message_queue:
                if (message_entry['id'] == message_id and 
                    agent_id in message_entry['target_agents']):
                    message_entry['delivered'] = True
                    return True
            return False
            
    def get_network_topology(self) -> Dict[str, Any]:
        """Get network topology information."""
        with self.lock:
            return {
                'network_id': self.network_id,
                'topology': self.topology.name,
                'agent_count': len(self.agents),
                'connection_count': sum(len(connections) for connections in self.connections.values()) // 2,
                'agents': [agent.to_dict() for agent in self.agents.values()],
                'connections': dict(self.connections)
            }
            
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        with self.lock:
            if not self.agents:
                return {
                    'total_agents': 0,
                    'active_agents': 0,
                    'total_skills': 0,
                    'unique_skills': 0,
                    'total_communications': 0,
                    'average_performance': 0
                }
                
            total_agents = len(self.agents)
            active_agents = sum(1 for agent in self.agents.values() 
                              if agent.status == AgentStatus.ACTIVE)
            
            all_skills = []
            for agent in self.agents.values():
                all_skills.extend(agent.skills)
                
            unique_skills = len(set(all_skills))
            total_communications = sum(len(agent.communication_history) 
                                     for agent in self.agents.values())
            
            # Calculate average performance
            performance_scores = []
            for agent in self.agents.values():
                if agent.performance_metrics:
                    score = agent.performance_metrics.get('overall_score', 0)
                    performance_scores.append(score)
                    
            average_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
            
            return {
                'total_agents': total_agents,
                'active_agents': active_agents,
                'total_skills': len(all_skills),
                'unique_skills': unique_skills,
                'total_communications': total_communications,
                'average_performance': average_performance,
                'network_efficiency': self._calculate_network_efficiency()
            }
            
    def _establish_connections(self, agent_id: str):
        """Establish connections for a new agent based on topology."""
        if self.topology == NetworkTopology.STAR:
            # Connect to first agent (hub)
            if len(self.agents) > 1:
                first_agent_id = next(iter(self.agents.keys()))
                if first_agent_id != agent_id:
                    self.connections[agent_id].add(first_agent_id)
                    self.connections[first_agent_id].add(agent_id)
                    
        elif self.topology == NetworkTopology.MESH:
            # Connect to all existing agents
            for existing_id in self.agents:
                if existing_id != agent_id:
                    self.connections[agent_id].add(existing_id)
                    self.connections[existing_id].add(agent_id)
                    
        elif self.topology == NetworkTopology.RING:
            # Connect to previous and next agents in ring
            agent_ids = list(self.agents.keys())
            if len(agent_ids) > 1:
                current_index = agent_ids.index(agent_id)
                prev_index = (current_index - 1) % len(agent_ids)
                next_index = (current_index + 1) % len(agent_ids)
                
                prev_id = agent_ids[prev_index]
                next_id = agent_ids[next_index]
                
                self.connections[agent_id].add(prev_id)
                self.connections[prev_id].add(agent_id)
                self.connections[agent_id].add(next_id)
                self.connections[next_id].add(agent_id)
                
    def _calculate_network_efficiency(self) -> float:
        """Calculate network efficiency score."""
        if len(self.agents) < 2:
            return 1.0
            
        # Calculate connectivity ratio
        total_possible_connections = len(self.agents) * (len(self.agents) - 1) // 2
        actual_connections = sum(len(connections) for connections in self.connections.values()) // 2
        
        connectivity_ratio = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
        # Calculate skill diversity
        all_skills = set()
        for agent in self.agents.values():
            all_skills.update(agent.skills)
            
        skill_diversity = len(all_skills) / (len(self.agents) * 5) if self.agents else 0  # Assume 5 skills per agent
        
        # Calculate communication efficiency
        total_communications = sum(len(agent.communication_history) for agent in self.agents.values())
        communication_efficiency = min(total_communications / (len(self.agents) * 10), 1.0)  # Normalize to 1.0
        
        # Combine metrics
        efficiency = (connectivity_ratio * 0.4 + skill_diversity * 0.3 + communication_efficiency * 0.3)
        return min(efficiency, 1.0)
        
    def _log_network_event(self, event_type: str, data: Dict[str, Any]):
        """Log a network event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.network_events.append(event)
        
    def get_network_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent network events."""
        return list(self.network_events[-limit:])
        
    def clear_network_events(self):
        """Clear network events history."""
        self.network_events.clear()
        
    def get_agent_performance_ranking(self) -> List[Dict[str, Any]]:
        """Get agents ranked by performance."""
        agent_scores = []
        
        for agent in self.agents.values():
            score = agent.performance_metrics.get('overall_score', 0)
            agent_scores.append({
                'agent_id': agent.id,
                'name': agent.name,
                'score': score,
                'status': agent.status.name
            })
            
        return sorted(agent_scores, key=lambda x: x['score'], reverse=True)
        
    def detect_network_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect network bottlenecks."""
        bottlenecks = []
        
        # Check for overloaded agents
        for agent in self.agents.values():
            if agent.status == AgentStatus.BUSY:
                # Check if agent has been busy for too long
                if time.time() - agent.last_activity > 300:  # 5 minutes
                    bottlenecks.append({
                        'type': 'overloaded_agent',
                        'agent_id': agent.id,
                        'name': agent.name,
                        'severity': 'high',
                        'description': f'Agent {agent.name} has been busy for over 5 minutes'
                    })
                    
        # Check for isolated agents
        for agent_id, connections in self.connections.items():
            if len(connections) == 0 and len(self.agents) > 1 and agent_id in self.agents:
                bottlenecks.append({
                    'type': 'isolated_agent',
                    'agent_id': agent_id,
                    'name': self.agents[agent_id].name,
                    'severity': 'medium',
                    'description': f'Agent {self.agents[agent_id].name} is isolated from the network'
                })
                
        # Check for skill gaps
        all_skills = set()
        for agent in self.agents.values():
            all_skills.update(agent.skills)
            
        if len(all_skills) < len(self.agents) * 3:  # Expect at least 3 skills per agent
            bottlenecks.append({
                'type': 'skill_gap',
                'severity': 'low',
                'description': 'Network has limited skill diversity'
            })
            
        return bottlenecks

