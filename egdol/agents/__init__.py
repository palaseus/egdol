"""
Multi-Agent System for Egdol.
Provides agent management, communication, and orchestration.
"""

from .manager import AgentManager
from .agent import Agent, AgentProfile
from .communication import MessageBus, Message

__all__ = ['AgentManager', 'Agent', 'AgentProfile', 'MessageBus', 'Message']
