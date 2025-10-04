"""
Multi-Agent Civilization System
Deploys autonomous agent instances of each personality archetype within persistent universes.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import threading
from collections import defaultdict, deque
import queue

from ..conversational.personality_framework import Personality, PersonalityType
from ..conversational.feedback_loop import FeedbackStorage, UserFeedback, FeedbackType, FeedbackSentiment
from ..conversational.meta_learning_engine import MetaLearningEngine


class AgentState(Enum):
    """Agent state enumeration."""
    IDLE = auto()
    THINKING = auto()
    COMMUNICATING = auto()
    COLLABORATING = auto()
    COMPETING = auto()
    BUILDING = auto()
    LEARNING = auto()


class MessageType(Enum):
    """Message type enumeration."""
    DEBATE = auto()
    COLLABORATION = auto()
    COMPETITION = auto()
    TREATY = auto()
    DISCOVERY = auto()
    CHALLENGE = auto()
    PROPOSAL = auto()
    CONSENSUS = auto()


@dataclass
class AgentMessage:
    """Message between agents."""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: str
    priority: int = 1  # 1-10, higher is more urgent
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.name,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentMemory:
    """Agent memory and experience."""
    agent_id: str
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    relationships: Dict[str, float] = field(default_factory=dict)  # agent_id -> relationship_score
    expertise: Dict[str, float] = field(default_factory=dict)  # domain -> expertise_level
    achievements: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "experiences": self.experiences,
            "relationships": self.relationships,
            "expertise": self.expertise,
            "achievements": self.achievements,
            "failures": self.failures,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class CivilizationState:
    """Civilization state and history."""
    civilization_id: str
    name: str
    agents: Dict[str, 'CivilizationAgent'] = field(default_factory=dict)
    laws: List[Dict[str, Any]] = field(default_factory=list)
    technologies: List[Dict[str, Any]] = field(default_factory=list)
    culture: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    treaties: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    discoveries: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "civilization_id": self.civilization_id,
            "name": self.name,
            "agent_count": len(self.agents),
            "laws": self.laws,
            "technologies": self.technologies,
            "culture": self.culture,
            "history": self.history,
            "treaties": self.treaties,
            "conflicts": self.conflicts,
            "discoveries": self.discoveries,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


class CivilizationAgent:
    """Autonomous agent within a civilization."""
    
    def __init__(self, 
                 agent_id: str,
                 personality: Personality,
                 civilization_id: str,
                 message_bus: 'MessageBus',
                 memory: AgentMemory):
        self.agent_id = agent_id
        self.personality = personality
        self.civilization_id = civilization_id
        self.message_bus = message_bus
        self.memory = memory
        self.state = AgentState.IDLE
        self.current_task = None
        self.collaboration_partners: Set[str] = set()
        self.competition_targets: Set[str] = set()
        self.active_debates: Set[str] = set()
        self.tool_inventory: List[str] = []  # Available tools
        self.last_activity = datetime.now()
        
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and potentially respond."""
        self.last_activity = datetime.now()
        
        # Update relationship with sender
        if message.sender_id != self.agent_id:
            self._update_relationship(message.sender_id, message.message_type)
        
        # Process based on message type
        if message.message_type == MessageType.DEBATE:
            return await self._handle_debate(message)
        elif message.message_type == MessageType.COLLABORATION:
            return await self._handle_collaboration(message)
        elif message.message_type == MessageType.COMPETITION:
            return await self._handle_competition(message)
        elif message.message_type == MessageType.TREATY:
            return await self._handle_treaty(message)
        elif message.message_type == MessageType.DISCOVERY:
            return await self._handle_discovery(message)
        elif message.message_type == MessageType.CHALLENGE:
            return await self._handle_challenge(message)
        elif message.message_type == MessageType.PROPOSAL:
            return await self._handle_proposal(message)
        elif message.message_type == MessageType.CONSENSUS:
            return await self._handle_consensus(message)
        
        return None
    
    async def _handle_debate(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle debate message."""
        self.state = AgentState.COMMUNICATING
        self.active_debates.add(message.sender_id)
        
        # Generate debate response based on personality
        response_content = self._generate_debate_response(message.content)
        
        if response_content:
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.DEBATE,
                content=response_content,
                priority=message.priority,
                metadata={"debate_topic": message.metadata.get("topic", "general")}
            )
        
        return None
    
    async def _handle_collaboration(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle collaboration message."""
        self.state = AgentState.COLLABORATING
        self.collaboration_partners.add(message.sender_id)
        
        # Generate collaboration response
        response_content = self._generate_collaboration_response(message.content)
        
        if response_content:
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.COLLABORATION,
                content=response_content,
                priority=message.priority,
                metadata={"collaboration_type": message.metadata.get("type", "general")}
            )
        
        return None
    
    async def _handle_competition(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle competition message."""
        self.state = AgentState.COMPETING
        self.competition_targets.add(message.sender_id)
        
        # Generate competition response
        response_content = self._generate_competition_response(message.content)
        
        if response_content:
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.COMPETITION,
                content=response_content,
                priority=message.priority,
                metadata={"competition_type": message.metadata.get("type", "general")}
            )
        
        return None
    
    async def _handle_treaty(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle treaty message."""
        # Generate treaty response
        response_content = self._generate_treaty_response(message.content)
        
        if response_content:
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TREATY,
                content=response_content,
                priority=message.priority,
                metadata={"treaty_type": message.metadata.get("type", "general")}
            )
        
        return None
    
    async def _handle_discovery(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle discovery message."""
        # Update expertise based on discovery
        domain = message.metadata.get("domain", "general")
        self.memory.expertise[domain] = self.memory.expertise.get(domain, 0.0) + 0.1
        
        # Generate discovery response
        response_content = self._generate_discovery_response(message.content)
        
        if response_content:
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.DISCOVERY,
                content=response_content,
                priority=message.priority,
                metadata={"discovery_domain": domain}
            )
        
        return None
    
    async def _handle_challenge(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle challenge message."""
        # Generate challenge response
        response_content = self._generate_challenge_response(message.content)
        
        if response_content:
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.CHALLENGE,
                content=response_content,
                priority=message.priority,
                metadata={"challenge_type": message.metadata.get("type", "general")}
            )
        
        return None
    
    async def _handle_proposal(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle proposal message."""
        # Generate proposal response
        response_content = self._generate_proposal_response(message.content)
        
        if response_content:
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.PROPOSAL,
                content=response_content,
                priority=message.priority,
                metadata={"proposal_type": message.metadata.get("type", "general")}
            )
        
        return None
    
    async def _handle_consensus(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle consensus message."""
        # Generate consensus response
        response_content = self._generate_consensus_response(message.content)
        
        if response_content:
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.CONSENSUS,
                content=response_content,
                priority=message.priority,
                metadata={"consensus_topic": message.metadata.get("topic", "general")}
            )
        
        return None
    
    def _generate_debate_response(self, content: str) -> str:
        """Generate debate response based on personality."""
        if self.personality.personality_type == PersonalityType.STRATEGOS:
            return f"Commander, from a strategic perspective: {content}. I propose we analyze the tactical implications and develop a comprehensive battle plan."
        elif self.personality.personality_type == PersonalityType.ARCHIVIST:
            return f"From the archives of knowledge: {content}. Let me reference historical precedents to support this position."
        elif self.personality.personality_type == PersonalityType.LAWMAKER:
            return f"According to legal principles: {content}. We must ensure this aligns with established governance frameworks."
        elif self.personality.personality_type == PersonalityType.ORACLE:
            return f"In the cosmic dance of reality: {content}. The mystical forces suggest a deeper truth lies beneath the surface."
        else:
            return f"I consider this perspective: {content}."
    
    def _generate_collaboration_response(self, content: str) -> str:
        """Generate collaboration response."""
        return f"I agree to collaborate on: {content}. Let us work together to achieve our shared goals."
    
    def _generate_competition_response(self, content: str) -> str:
        """Generate competition response."""
        return f"I accept the challenge: {content}. May the best strategy prevail."
    
    def _generate_treaty_response(self, content: str) -> str:
        """Generate treaty response."""
        return f"I propose we establish a treaty regarding: {content}. This will benefit our mutual interests."
    
    def _generate_discovery_response(self, content: str) -> str:
        """Generate discovery response."""
        return f"Fascinating discovery: {content}. This opens new possibilities for our civilization."
    
    def _generate_challenge_response(self, content: str) -> str:
        """Generate challenge response."""
        return f"I accept your challenge: {content}. Let us test our capabilities."
    
    def _generate_proposal_response(self, content: str) -> str:
        """Generate proposal response."""
        return f"I propose: {content}. This could advance our civilization significantly."
    
    def _generate_consensus_response(self, content: str) -> str:
        """Generate consensus response."""
        return f"I agree with the consensus: {content}. This represents our collective wisdom."
    
    def _update_relationship(self, other_agent_id: str, message_type: MessageType):
        """Update relationship with another agent."""
        current_relationship = self.memory.relationships.get(other_agent_id, 0.5)
        
        # Update based on message type
        if message_type in [MessageType.COLLABORATION, MessageType.TREATY, MessageType.CONSENSUS]:
            current_relationship = min(1.0, current_relationship + 0.1)
        elif message_type in [MessageType.COMPETITION, MessageType.CHALLENGE]:
            current_relationship = max(0.0, current_relationship - 0.05)
        elif message_type == MessageType.DEBATE:
            current_relationship = max(0.0, current_relationship - 0.02)
        
        self.memory.relationships[other_agent_id] = current_relationship
    
    async def autonomous_action(self) -> Optional[AgentMessage]:
        """Perform autonomous action."""
        self.state = AgentState.THINKING
        self.last_activity = datetime.now()
        
        # Decide on action based on personality and state
        action = self._decide_autonomous_action()
        
        if action:
            return AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=None,  # Broadcast
                message_type=action["type"],
                content=action["content"],
                priority=action.get("priority", 1),
                metadata=action.get("metadata", {})
            )
        
        return None
    
    def _decide_autonomous_action(self) -> Optional[Dict[str, Any]]:
        """Decide on autonomous action."""
        # Simple decision logic - can be enhanced with more sophisticated AI
        import random
        
        actions = [
            {
                "type": MessageType.DISCOVERY,
                "content": f"I have discovered new insights in {random.choice(['strategy', 'history', 'law', 'mysticism'])}",
                "priority": random.randint(1, 5),
                "metadata": {"domain": "general"}
            },
            {
                "type": MessageType.PROPOSAL,
                "content": f"I propose we explore {random.choice(['new technologies', 'cultural exchange', 'legal frameworks', 'mystical practices'])}",
                "priority": random.randint(1, 5),
                "metadata": {"type": "exploration"}
            },
            {
                "type": MessageType.CHALLENGE,
                "content": f"I challenge the current approach to {random.choice(['governance', 'defense', 'knowledge', 'wisdom'])}",
                "priority": random.randint(1, 5),
                "metadata": {"type": "governance"}
            }
        ]
        
        return random.choice(actions)


class MessageBus:
    """Message bus for agent communication."""
    
    def __init__(self):
        self.messages: queue.Queue = queue.Queue()
        self.subscribers: Dict[str, List[str]] = defaultdict(list)  # message_type -> agent_ids
        self.message_history: deque = deque(maxlen=1000)
        self.running = False
        self.thread = None
    
    def subscribe(self, agent_id: str, message_types: List[MessageType]):
        """Subscribe agent to message types."""
        for message_type in message_types:
            self.subscribers[message_type.name].append(agent_id)
    
    def unsubscribe(self, agent_id: str, message_types: List[MessageType]):
        """Unsubscribe agent from message types."""
        for message_type in message_types:
            if agent_id in self.subscribers[message_type.name]:
                self.subscribers[message_type.name].remove(agent_id)
    
    def publish(self, message: AgentMessage):
        """Publish message to subscribers."""
        self.messages.put(message)
        self.message_history.append(message)
    
    def get_messages_for_agent(self, agent_id: str) -> List[AgentMessage]:
        """Get messages for specific agent."""
        messages = []
        temp_messages = []
        
        # Process all messages in queue
        while not self.messages.empty():
            try:
                message = self.messages.get_nowait()
                temp_messages.append(message)
                
                # Check if agent should receive this message
                if (message.receiver_id == agent_id or 
                    message.receiver_id is None or  # Broadcast
                    agent_id in self.subscribers.get(message.message_type.name, [])):
                    messages.append(message)
            except queue.Empty:
                break
        
        # Put messages back in queue for other agents
        for message in temp_messages:
            self.messages.put(message)
        
        return messages
    
    def start(self):
        """Start message bus."""
        self.running = True
        self.thread = threading.Thread(target=self._process_messages)
        self.thread.start()
    
    def stop(self):
        """Stop message bus."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _process_message(self, message: AgentMessage):
        """Process a single message."""
        # Update civilization state based on message
        # This is a simplified implementation that updates the last_updated timestamp
        # In a real implementation, this would update the civilization state based on message content
        pass
    
    def _process_messages(self):
        """Process messages in background thread."""
        while self.running:
            try:
                # Process messages
                pass  # Messages are processed synchronously when requested
            except Exception as e:
                print(f"Error in message processing: {e}")
            
            time.sleep(0.1)  # Small delay to prevent busy waiting


class MultiAgentCivilizationSystem:
    """Main system for managing multi-agent civilizations."""
    
    def __init__(self):
        self.civilizations: Dict[str, CivilizationState] = {}
        self.message_bus = MessageBus()
        self.agents: Dict[str, CivilizationAgent] = {}
        self.running = False
        self.simulation_thread = None
    
    def create_civilization(self, name: str, agent_configs: List[Dict[str, Any]]) -> str:
        """Create a new civilization with agents."""
        civilization_id = str(uuid.uuid4())
        
        # Create civilization state
        civilization = CivilizationState(
            civilization_id=civilization_id,
            name=name
        )
        
        # Create agents
        for config in agent_configs:
            agent_id = str(uuid.uuid4())
            personality = Personality(
                name=config["personality_name"],
                personality_type=config["personality_type"],
                description=f"{config['personality_name']} personality",
                archetype=config["personality_name"].lower(),
                epistemic_style=config.get("communication_style", "formal")
            )
            
            # Create agent memory
            memory = AgentMemory(agent_id=agent_id)
            
            # Create agent
            agent = CivilizationAgent(
                agent_id=agent_id,
                personality=personality,
                civilization_id=civilization_id,
                message_bus=self.message_bus,
                memory=memory
            )
            
            # Subscribe agent to message types
            message_types = [
                MessageType.DEBATE, MessageType.COLLABORATION, MessageType.COMPETITION,
                MessageType.TREATY, MessageType.DISCOVERY, MessageType.CHALLENGE,
                MessageType.PROPOSAL, MessageType.CONSENSUS
            ]
            self.message_bus.subscribe(agent_id, message_types)
            
            # Add agent to civilization
            civilization.agents[agent_id] = agent
            self.agents[agent_id] = agent
        
        # Store civilization
        self.civilizations[civilization_id] = civilization
        
        return civilization_id
    
    def start_simulation(self, civilization_id: str):
        """Start simulation for a civilization."""
        if civilization_id not in self.civilizations:
            raise ValueError(f"Civilization {civilization_id} not found")
        
        self.running = True
        self.message_bus.start()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=self._run_simulation,
            args=(civilization_id,)
        )
        self.simulation_thread.start()
    
    def stop_simulation(self, civilization_id: str):
        """Stop simulation for a civilization."""
        self.running = False
        self.message_bus.stop()
        
        if self.simulation_thread:
            self.simulation_thread.join()
    
    def _run_simulation(self, civilization_id: str):
        """Run simulation loop."""
        civilization = self.civilizations[civilization_id]
        
        while self.running:
            try:
                # Process all agents
                for agent_id, agent in civilization.agents.items():
                    # Get messages for agent
                    messages = self.message_bus.get_messages_for_agent(agent_id)
                    
                    # Process messages
                    for message in messages:
                        response = asyncio.run(agent.process_message(message))
                        if response:
                            self.message_bus.publish(response)
                    
                    # Autonomous action
                    if len(messages) == 0:  # No messages, perform autonomous action
                        action = asyncio.run(agent.autonomous_action())
                        if action:
                            self.message_bus.publish(action)
                
                # Update civilization state
                self._update_civilization_state(civilization)
                
                # Small delay
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Error in simulation: {e}")
                break
    
    def _update_civilization_state(self, civilization: CivilizationState):
        """Update civilization state based on agent interactions."""
        civilization.last_updated = datetime.now()
        
        # Update history with recent events
        recent_messages = list(self.message_bus.message_history)[-10:]  # Last 10 messages
        
        for message in recent_messages:
            if message.sender_id in civilization.agents:
                event = {
                    "timestamp": message.timestamp.isoformat(),
                    "agent_id": message.sender_id,
                    "message_type": message.message_type.name,
                    "content": message.content,
                    "metadata": message.metadata
                }
                civilization.history.append(event)
        
        # Keep only last 1000 events
        if len(civilization.history) > 1000:
            civilization.history = civilization.history[-1000:]
    
    def get_civilization_state(self, civilization_id: str) -> Dict[str, Any]:
        """Get current civilization state."""
        if civilization_id not in self.civilizations:
            return {}
        
        civilization = self.civilizations[civilization_id]
        
        # Update timestamp to reflect current state
        civilization.last_updated = datetime.now()
        
        return civilization.to_dict()
    
    def get_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        """Get agent memory."""
        if agent_id not in self.agents:
            return {}
        
        return self.agents[agent_id].memory.to_dict()
    
    def get_message_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get message history."""
        messages = list(self.message_bus.message_history)[-limit:]
        return [message.to_dict() for message in messages]
