"""
Agent Manager for Egdol.
Manages multiple agents, their profiles, and interactions.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from .agent import Agent, AgentProfile
from .communication import MessageBus, AgentCoordinator


class AgentManager:
    """Manages multiple agents and their interactions."""
    
    def __init__(self, data_dir: str = "agents_data"):
        self.data_dir = data_dir
        self.agents: Dict[str, Agent] = {}
        self.coordinator = AgentCoordinator()
        self.profiles_file = os.path.join(data_dir, "profiles.json")
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing profiles
        self._load_profiles()
        
    def create_agent(self, name: str, description: str = "", 
                     expertise: List[str] = None, personality: Dict[str, Any] = None) -> Agent:
        """Create a new agent."""
        if name in self.agents:
            raise ValueError(f"Agent '{name}' already exists")
            
        profile = AgentProfile(
            name=name,
            description=description,
            expertise=expertise or [],
            personality=personality or {}
        )
        
        agent = Agent(profile, self.data_dir)
        self.agents[name] = agent
        
        # Register with coordinator
        self.coordinator.register_agent(agent)
        
        # Save profile
        self._save_profiles()
        
        return agent
        
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)
        
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents."""
        return [
            {
                'name': agent.profile.name,
                'description': agent.profile.description,
                'expertise': agent.profile.expertise,
                'is_active': agent.is_active,
                'last_active': agent.profile.last_active
            }
            for agent in self.agents.values()
        ]
        
    def delete_agent(self, name: str) -> bool:
        """Delete an agent."""
        if name not in self.agents:
            return False
            
        agent = self.agents[name]
        
        # Unregister from coordinator
        self.coordinator.unregister_agent(name)
        
        # Remove agent
        del self.agents[name]
        
        # Save profiles
        self._save_profiles()
        
        return True
        
    def switch_agent(self, name: str) -> Optional[Agent]:
        """Switch to a specific agent."""
        if name not in self.agents:
            return None
            
        agent = self.agents[name]
        agent.profile.last_active = time.time()
        return agent
        
    def get_active_agent(self) -> Optional[Agent]:
        """Get the most recently active agent."""
        if not self.agents:
            return None
            
        # Find most recently active agent
        most_recent = max(self.agents.values(), key=lambda a: a.profile.last_active)
        return most_recent
        
    def broadcast_to_agents(self, sender: str, message: str, message_type: str = "notification"):
        """Broadcast a message to all agents."""
        self.coordinator.broadcast(sender, message, message_type)
        
    def coordinate_task(self, task: str, required_expertise: List[str]) -> List[str]:
        """Find agents suitable for a task."""
        return self.coordinator.coordinate_task(task, required_expertise)
        
    def get_agent_network(self) -> Dict[str, Any]:
        """Get the agent network structure."""
        return self.coordinator.create_agent_network(list(self.agents.values()))
        
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return self.coordinator.get_coordination_stats()
        
    def export_agent_data(self, agent_name: str, file_path: str) -> bool:
        """Export an agent's data."""
        if agent_name not in self.agents:
            return False
            
        agent = self.agents[agent_name]
        return agent.export_knowledge(file_path)
        
    def import_agent_data(self, agent_name: str, file_path: str) -> int:
        """Import data for an agent."""
        if agent_name not in self.agents:
            return 0
            
        agent = self.agents[agent_name]
        return agent.import_knowledge(file_path)
        
    def save_all_agents(self) -> bool:
        """Save all agents' states."""
        success = True
        for agent in self.agents.values():
            if not agent.save_state():
                success = False
        return success
        
    def load_all_agents(self) -> bool:
        """Load all agents' states."""
        success = True
        for agent in self.agents.values():
            if not agent.load_state():
                success = False
        return success
        
    def _load_profiles(self):
        """Load agent profiles from disk."""
        if not os.path.exists(self.profiles_file):
            return
            
        try:
            with open(self.profiles_file, 'r') as f:
                profiles_data = json.load(f)
                
            for profile_data in profiles_data.get('profiles', []):
                profile = AgentProfile.from_dict(profile_data)
                agent = Agent(profile, self.data_dir)
                self.agents[profile.name] = agent
                self.coordinator.register_agent(agent)
                
        except Exception as e:
            print(f"Error loading profiles: {e}")
            
    def _save_profiles(self):
        """Save agent profiles to disk."""
        try:
            profiles_data = {
                'profiles': [agent.profile.to_dict() for agent in self.agents.values()],
                'last_updated': time.time()
            }
            
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving profiles: {e}")
            
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            'total_agents': len(self.agents),
            'active_agents': len([a for a in self.agents.values() if a.is_active]),
            'data_directory': self.data_dir,
            'communication_stats': self.get_communication_stats()
        }
