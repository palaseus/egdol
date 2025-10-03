"""
Coordination System for OmniMind Network
Handles task coordination, goal negotiation, and resource management.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque


class TaskStatus(Enum):
    """Task status in coordination."""
    PENDING = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class ResourceType(Enum):
    """Types of resources in the network."""
    COMPUTATIONAL = auto()
    MEMORY = auto()
    SKILL = auto()
    KNOWLEDGE = auto()
    NETWORK = auto()


@dataclass
class CoordinatedTask:
    """A task coordinated across multiple agents."""
    id: str
    goal: str
    description: str
    required_skills: List[str]
    required_resources: Dict[ResourceType, float]
    assigned_agents: List[str]
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    priority: int = 1
    estimated_duration: float = 0.0
    actual_duration: Optional[float] = None
    dependencies: List[str] = None
    results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.results is None:
            self.results = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'id': self.id,
            'goal': self.goal,
            'description': self.description,
            'required_skills': self.required_skills,
            'required_resources': {k.name: v for k, v in self.required_resources.items()},
            'assigned_agents': self.assigned_agents,
            'status': self.status.name,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'priority': self.priority,
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration,
            'dependencies': self.dependencies,
            'results': self.results
        }


@dataclass
class ResourceAllocation:
    """Resource allocation for an agent."""
    agent_id: str
    resource_type: ResourceType
    allocated_amount: float
    available_amount: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert allocation to dictionary."""
        return {
            'agent_id': self.agent_id,
            'resource_type': self.resource_type.name,
            'allocated_amount': self.allocated_amount,
            'available_amount': self.available_amount,
            'timestamp': self.timestamp
        }


class TaskCoordinator:
    """Coordinates tasks across multiple agents."""
    
    def __init__(self, network):
        self.network = network
        self.tasks: Dict[str, CoordinatedTask] = {}
        self.task_queue: deque = deque()
        self.agent_workloads: Dict[str, int] = defaultdict(int)
        self.resource_allocations: Dict[str, List[ResourceAllocation]] = defaultdict(list)
        self.coordination_history: List[Dict[str, Any]] = []
        
    def create_task(self, goal: str, description: str, required_skills: List[str],
                   required_resources: Dict[ResourceType, float] = None,
                   priority: int = 1, estimated_duration: float = 0.0,
                   dependencies: List[str] = None) -> str:
        """Create a new coordinated task."""
        task_id = str(uuid.uuid4())
        
        if required_resources is None:
            required_resources = {}
        if dependencies is None:
            dependencies = []
            
        task = CoordinatedTask(
            id=task_id,
            goal=goal,
            description=description,
            required_skills=required_skills,
            required_resources=required_resources,
            assigned_agents=[],
            status=TaskStatus.PENDING,
            created_at=time.time(),
            priority=priority,
            estimated_duration=estimated_duration,
            dependencies=dependencies
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        # Log coordination event
        self._log_coordination_event('task_created', {
            'task_id': task_id,
            'goal': goal,
            'required_skills': required_skills,
            'priority': priority
        })
        
        return task_id
        
    def assign_task(self, task_id: str, agent_ids: List[str]) -> bool:
        """Assign a task to agents."""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        # Check if agents have required skills
        for agent_id in agent_ids:
            agent = self.network.get_agent(agent_id)
            if not agent:
                return False
                
            # Check skills
            if not all(skill in agent.skills for skill in task.required_skills):
                return False
                
        # Assign task
        task.assigned_agents = agent_ids
        task.status = TaskStatus.ASSIGNED
        
        # Update agent workloads
        for agent_id in agent_ids:
            self.agent_workloads[agent_id] += 1
            
        # Log coordination event
        self._log_coordination_event('task_assigned', {
            'task_id': task_id,
            'assigned_agents': agent_ids
        })
        
        return True
        
    def start_task(self, task_id: str) -> bool:
        """Start a task execution."""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        if task.status != TaskStatus.ASSIGNED:
            return False
            
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        
        # Log coordination event
        self._log_coordination_event('task_started', {
            'task_id': task_id,
            'started_at': task.started_at
        })
        
        return True
        
    def complete_task(self, task_id: str, results: Dict[str, Any] = None) -> bool:
        """Complete a task."""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        if task.status != TaskStatus.IN_PROGRESS:
            return False
            
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.actual_duration = task.completed_at - task.started_at if task.started_at else 0
        task.results = results or {}
        
        # Update agent workloads
        for agent_id in task.assigned_agents:
            self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] - 1)
            
        # Log coordination event
        self._log_coordination_event('task_completed', {
            'task_id': task_id,
            'completed_at': task.completed_at,
            'actual_duration': task.actual_duration,
            'results': results
        })
        
        return True
        
    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed."""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        
        # Update agent workloads
        for agent_id in task.assigned_agents:
            self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] - 1)
            
        # Log coordination event
        self._log_coordination_event('task_failed', {
            'task_id': task_id,
            'error': error,
            'failed_at': task.completed_at
        })
        
        return True
        
    def get_available_agents(self, required_skills: List[str]) -> List[str]:
        """Get agents available for a task."""
        available_agents = []
        
        for agent_id, agent in self.network.agents.items():
            # Check if agent has required skills
            if not all(skill in agent.skills for skill in required_skills):
                continue
                
            # Check if agent is not overloaded
            if self.agent_workloads[agent_id] > 3:  # Threshold for overload
                continue
                
            # Check if agent is active (ACTIVE status is 1)
            if hasattr(agent.status, 'value') and agent.status.value != 1:
                continue
                
            available_agents.append(agent_id)
            
        return available_agents
        
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        if task_id not in self.tasks:
            return None
            
        return self.tasks[task_id].to_dict()
        
    def get_agent_workload(self, agent_id: str) -> int:
        """Get agent workload."""
        return self.agent_workloads[agent_id]
        
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for task in self.tasks.values() 
                            if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in self.tasks.values() 
                          if task.status == TaskStatus.FAILED)
        in_progress_tasks = sum(1 for task in self.tasks.values() 
                              if task.status == TaskStatus.IN_PROGRESS)
        
        # Calculate success rate
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        # Calculate average task duration
        completed_durations = [task.actual_duration for task in self.tasks.values() 
                             if task.actual_duration is not None]
        average_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0
        
        # Calculate agent workload distribution
        workload_distribution = dict(self.agent_workloads)
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'in_progress_tasks': in_progress_tasks,
            'success_rate': success_rate,
            'average_duration': average_duration,
            'workload_distribution': workload_distribution
        }
        
    def _log_coordination_event(self, event_type: str, data: Dict[str, Any]):
        """Log a coordination event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.coordination_history.append(event)
        
    def get_coordination_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get coordination history."""
        return list(self.coordination_history[-limit:])
        
    def detect_coordination_issues(self) -> List[Dict[str, Any]]:
        """Detect coordination issues."""
        issues = []
        
        # Check for overloaded agents
        for agent_id, workload in self.agent_workloads.items():
            if workload > 5:  # Threshold for overload
                issues.append({
                    'type': 'agent_overload',
                    'agent_id': agent_id,
                    'workload': workload,
                    'severity': 'high',
                    'description': f'Agent {agent_id} is overloaded with {workload} tasks'
                })
                
        # Check for stuck tasks
        current_time = time.time()
        for task in self.tasks.values():
            if task.status == TaskStatus.IN_PROGRESS:
                if task.started_at and current_time - task.started_at > 300:  # 5 minutes
                    issues.append({
                        'type': 'stuck_task',
                        'task_id': task.id,
                        'duration': current_time - task.started_at,
                        'severity': 'medium',
                        'description': f'Task {task.id} has been running for {current_time - task.started_at:.1f} seconds'
                    })
                    
        # Check for failed tasks
        failed_count = sum(1 for task in self.tasks.values() 
                          if task.status == TaskStatus.FAILED)
        if failed_count > 3:  # Threshold for high failure rate
            issues.append({
                'type': 'high_failure_rate',
                'failed_count': failed_count,
                'severity': 'high',
                'description': f'High task failure rate: {failed_count} failed tasks'
            })
            
        return issues


class GoalNegotiator:
    """Negotiates goals between agents."""
    
    def __init__(self, network):
        self.network = network
        self.negotiations: Dict[str, Dict[str, Any]] = {}
        self.negotiation_history: List[Dict[str, Any]] = []
        
    def start_negotiation(self, goal: str, initiator_id: str, 
                         target_agents: List[str]) -> str:
        """Start a goal negotiation."""
        negotiation_id = str(uuid.uuid4())
        
        negotiation = {
            'id': negotiation_id,
            'goal': goal,
            'initiator_id': initiator_id,
            'target_agents': target_agents,
            'status': 'active',
            'created_at': time.time(),
            'proposals': {},
            'votes': {},
            'consensus': None
        }
        
        self.negotiations[negotiation_id] = negotiation
        
        # Log negotiation event
        self._log_negotiation_event('negotiation_started', {
            'negotiation_id': negotiation_id,
            'goal': goal,
            'initiator_id': initiator_id,
            'target_agents': target_agents
        })
        
        return negotiation_id
        
    def submit_proposal(self, negotiation_id: str, agent_id: str, 
                       proposal: Dict[str, Any]) -> bool:
        """Submit a proposal for a negotiation."""
        if negotiation_id not in self.negotiations:
            return False
            
        negotiation = self.negotiations[negotiation_id]
        
        if agent_id not in negotiation['target_agents']:
            return False
            
        negotiation['proposals'][agent_id] = {
            'proposal': proposal,
            'timestamp': time.time(),
            'confidence': proposal.get('confidence', 0.5)
        }
        
        # Log negotiation event
        self._log_negotiation_event('proposal_submitted', {
            'negotiation_id': negotiation_id,
            'agent_id': agent_id,
            'proposal': proposal
        })
        
        return True
        
    def vote_on_proposal(self, negotiation_id: str, agent_id: str, 
                        proposal_agent_id: str, vote: bool) -> bool:
        """Vote on a proposal."""
        if negotiation_id not in self.negotiations:
            return False
            
        negotiation = self.negotiations[negotiation_id]
        
        if agent_id not in negotiation['target_agents']:
            return False
            
        if proposal_agent_id not in negotiation['proposals']:
            return False
            
        negotiation['votes'][agent_id] = {
            'proposal_agent_id': proposal_agent_id,
            'vote': vote,
            'timestamp': time.time()
        }
        
        # Check for consensus
        self._check_consensus(negotiation_id)
        
        # Log negotiation event
        self._log_negotiation_event('vote_cast', {
            'negotiation_id': negotiation_id,
            'agent_id': agent_id,
            'proposal_agent_id': proposal_agent_id,
            'vote': vote
        })
        
        return True
        
    def _check_consensus(self, negotiation_id: str):
        """Check for consensus in a negotiation."""
        if negotiation_id not in self.negotiations:
            return
            
        negotiation = self.negotiations[negotiation_id]
        
        # Count votes for each proposal
        proposal_votes = defaultdict(int)
        for vote_data in negotiation['votes'].values():
            if vote_data['vote']:
                proposal_votes[vote_data['proposal_agent_id']] += 1
                
        # Check if any proposal has majority
        total_agents = len(negotiation['target_agents'])
        majority_threshold = total_agents // 2 + 1
        
        for proposal_agent_id, vote_count in proposal_votes.items():
            if vote_count >= majority_threshold:
                negotiation['consensus'] = {
                    'proposal_agent_id': proposal_agent_id,
                    'vote_count': vote_count,
                    'timestamp': time.time()
                }
                negotiation['status'] = 'consensus_reached'
                
                # Log consensus event
                self._log_negotiation_event('consensus_reached', {
                    'negotiation_id': negotiation_id,
                    'proposal_agent_id': proposal_agent_id,
                    'vote_count': vote_count
                })
                break
                
    def get_negotiation_status(self, negotiation_id: str) -> Optional[Dict[str, Any]]:
        """Get negotiation status."""
        if negotiation_id not in self.negotiations:
            return None
            
        return self.negotiations[negotiation_id].copy()
        
    def get_active_negotiations(self) -> List[Dict[str, Any]]:
        """Get active negotiations."""
        return [negotiation for negotiation in self.negotiations.values() 
                if negotiation['status'] == 'active']
                
    def _log_negotiation_event(self, event_type: str, data: Dict[str, Any]):
        """Log a negotiation event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.negotiation_history.append(event)
        
    def get_negotiation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get negotiation history."""
        return list(self.negotiation_history[-limit:])


class ResourceManager:
    """Manages resources across the network."""
    
    def __init__(self, network):
        self.network = network
        self.resource_pools: Dict[ResourceType, float] = defaultdict(float)
        self.allocations: Dict[str, List[ResourceAllocation]] = defaultdict(list)
        self.resource_history: List[Dict[str, Any]] = []
        
    def allocate_resource(self, agent_id: str, resource_type: ResourceType, 
                         amount: float) -> bool:
        """Allocate a resource to an agent."""
        if amount <= 0:
            return False
            
        # Check if resource is available
        if self.resource_pools[resource_type] < amount:
            return False
            
        # Allocate resource
        self.resource_pools[resource_type] -= amount
        
        allocation = ResourceAllocation(
            agent_id=agent_id,
            resource_type=resource_type,
            allocated_amount=amount,
            available_amount=self.resource_pools[resource_type],
            timestamp=time.time()
        )
        
        self.allocations[agent_id].append(allocation)
        
        # Log resource event
        self._log_resource_event('resource_allocated', {
            'agent_id': agent_id,
            'resource_type': resource_type.name,
            'amount': amount,
            'remaining': self.resource_pools[resource_type]
        })
        
        return True
        
    def deallocate_resource(self, agent_id: str, resource_type: ResourceType, 
                           amount: float) -> bool:
        """Deallocate a resource from an agent."""
        if amount <= 0:
            return False
            
        # Find allocation to deallocate
        for allocation in self.allocations[agent_id]:
            if allocation.resource_type == resource_type and allocation.allocated_amount >= amount:
                allocation.allocated_amount -= amount
                self.resource_pools[resource_type] += amount
                
                # Log resource event
                self._log_resource_event('resource_deallocated', {
                    'agent_id': agent_id,
                    'resource_type': resource_type.name,
                    'amount': amount,
                    'remaining': self.resource_pools[resource_type]
                })
                
                return True
                
        return False
        
    def get_resource_availability(self, resource_type: ResourceType) -> float:
        """Get available amount of a resource."""
        return self.resource_pools[resource_type]
        
    def get_agent_allocations(self, agent_id: str) -> List[ResourceAllocation]:
        """Get resource allocations for an agent."""
        return self.allocations[agent_id]
        
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get resource statistics."""
        total_allocations = sum(len(allocations) for allocations in self.allocations.values())
        
        resource_usage = {}
        for resource_type in ResourceType:
            resource_usage[resource_type.name] = {
                'available': self.resource_pools[resource_type],
                'allocated': sum(
                    allocation.allocated_amount 
                    for allocations in self.allocations.values() 
                    for allocation in allocations 
                    if allocation.resource_type == resource_type
                )
            }
            
        return {
            'total_allocations': total_allocations,
            'resource_usage': resource_usage,
            'agent_allocations': {
                agent_id: len(allocations) 
                for agent_id, allocations in self.allocations.items()
            }
        }
        
    def _log_resource_event(self, event_type: str, data: Dict[str, Any]):
        """Log a resource event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.resource_history.append(event)
        
    def get_resource_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get resource history."""
        return list(self.resource_history[-limit:])

