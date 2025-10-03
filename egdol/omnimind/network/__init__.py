"""
Multi-Agent Network Layer for OmniMind
Enables networked multi-agent ecosystem with emergent collaborative reasoning.
"""

from .agent_network import AgentNetwork, NetworkTopology, AgentStatus
from .communication import MessageBus, Message, MessageType, MessagePriority
from .coordination import TaskCoordinator, GoalNegotiator, ResourceManager, TaskStatus, ResourceType
from .learning import NetworkLearning, KnowledgePropagation, SkillSharing, LearningType, LearningStatus
from .monitoring import NetworkMonitor, PerformanceAnalyzer, ConflictDetector, MonitorType, AlertLevel
from .emergent import EmergentBehavior, PatternDetector, CollaborationEngine, EmergentType, PatternStatus

__all__ = [
    'AgentNetwork', 'NetworkTopology', 'AgentStatus',
    'MessageBus', 'Message', 'MessageType', 'MessagePriority',
    'TaskCoordinator', 'GoalNegotiator', 'ResourceManager', 'TaskStatus', 'ResourceType',
    'NetworkLearning', 'KnowledgePropagation', 'SkillSharing', 'LearningType', 'LearningStatus',
    'NetworkMonitor', 'PerformanceAnalyzer', 'ConflictDetector', 'MonitorType', 'AlertLevel',
    'EmergentBehavior', 'PatternDetector', 'CollaborationEngine', 'EmergentType', 'PatternStatus'
]

