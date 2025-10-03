"""
Goal Planner for OmniMind
Handles complex goal decomposition and task planning.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    """Priority of a task."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class Task:
    """A task in the planning system."""
    id: str
    name: str
    description: str
    skill_required: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = None
    started_at: float = None
    completed_at: float = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
            
    def start(self):
        """Start the task."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
        
    def complete(self, result: Dict[str, Any]):
        """Complete the task."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result
        
    def fail(self, error: str):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()
        self.error = error
        self.retry_count += 1
        
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'skill_required': self.skill_required,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'priority': self.priority.name,
            'status': self.status.name,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


@dataclass
class Goal:
    """A goal in the planning system."""
    id: str
    description: str
    tasks: List[Task]
    created_at: float = None
    completed_at: float = None
    status: TaskStatus = TaskStatus.PENDING
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
            
    def is_completed(self) -> bool:
        """Check if goal is completed."""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)
        
    def is_failed(self) -> bool:
        """Check if goal has failed tasks."""
        return any(task.status == TaskStatus.FAILED for task in self.tasks)
        
    def get_progress(self) -> float:
        """Get completion progress (0.0 to 1.0)."""
        if not self.tasks:
            return 1.0
        completed = sum(1 for task in self.tasks if task.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary."""
        return {
            'id': self.id,
            'description': self.description,
            'tasks': [task.to_dict() for task in self.tasks],
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'status': self.status.name,
            'progress': self.get_progress()
        }


class GoalPlanner:
    """Main planner for complex goal decomposition and execution."""
    
    def __init__(self, decomposer, executor, scheduler):
        self.decomposer = decomposer
        self.executor = executor
        self.scheduler = scheduler
        self.active_goals: Dict[str, Goal] = {}
        self.completed_goals: Dict[str, Goal] = {}
        self.planning_history: List[Dict[str, Any]] = []
        
    def plan_goal(self, goal_description: str, context: Dict[str, Any] = None) -> Goal:
        """Plan a complex goal by decomposing it into tasks."""
        goal_id = str(uuid.uuid4())
        
        # Decompose goal into tasks
        tasks = self.decomposer.decompose_goal(goal_description, context or {})
        
        # Create goal
        goal = Goal(
            id=goal_id,
            description=goal_description,
            tasks=tasks
        )
        
        # Store goal
        self.active_goals[goal_id] = goal
        
        # Record planning
        self.planning_history.append({
            'goal_id': goal_id,
            'description': goal_description,
            'tasks_count': len(tasks),
            'planned_at': time.time()
        })
        
        return goal
        
    def execute_goal(self, goal_id: str, verbose: bool = False) -> Dict[str, Any]:
        """Execute a planned goal."""
        if goal_id not in self.active_goals:
            return {'success': False, 'error': 'Goal not found'}
            
        goal = self.active_goals[goal_id]
        execution_result = {
            'goal_id': goal_id,
            'description': goal.description,
            'tasks': [],
            'success': False,
            'started_at': time.time(),
            'completed_at': None,
            'total_tasks': len(goal.tasks),
            'completed_tasks': 0,
            'failed_tasks': 0,
            'reasoning_trace': []
        }
        
        if verbose:
            execution_result['reasoning_trace'].append(f"Starting execution of goal: {goal.description}")
            
        # Execute tasks in order
        for task in goal.tasks:
            if verbose:
                execution_result['reasoning_trace'].append(f"Executing task: {task.name}")
                
            task_result = self.executor.execute_task(task, verbose)
            execution_result['tasks'].append(task_result)
            
            if task_result['success']:
                execution_result['completed_tasks'] += 1
                if verbose:
                    execution_result['reasoning_trace'].append(f"Task completed: {task.name}")
            else:
                execution_result['failed_tasks'] += 1
                if verbose:
                    execution_result['reasoning_trace'].append(f"Task failed: {task.name} - {task_result.get('error', 'Unknown error')}")
                    
                # Handle task failure
                if task.can_retry():
                    if verbose:
                        execution_result['reasoning_trace'].append(f"Retrying task: {task.name}")
                    task.status = TaskStatus.PENDING
                    task.retry_count += 1
                else:
                    if verbose:
                        execution_result['reasoning_trace'].append(f"Task failed permanently: {task.name}")
                    break
                    
        # Check if goal is completed
        if goal.is_completed():
            execution_result['success'] = True
            goal.status = TaskStatus.COMPLETED
            goal.completed_at = time.time()
            self.completed_goals[goal_id] = goal
            del self.active_goals[goal_id]
            
            if verbose:
                execution_result['reasoning_trace'].append("Goal completed successfully")
        else:
            if verbose:
                execution_result['reasoning_trace'].append("Goal execution failed")
                
        execution_result['completed_at'] = time.time()
        return execution_result
        
    def get_goal_status(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a goal."""
        if goal_id in self.active_goals:
            goal = self.active_goals[goal_id]
            return {
                'goal_id': goal_id,
                'description': goal.description,
                'status': goal.status.name,
                'progress': goal.get_progress(),
                'tasks': [task.to_dict() for task in goal.tasks]
            }
        elif goal_id in self.completed_goals:
            goal = self.completed_goals[goal_id]
            return {
                'goal_id': goal_id,
                'description': goal.description,
                'status': goal.status.name,
                'progress': goal.get_progress(),
                'tasks': [task.to_dict() for task in goal.tasks]
            }
        return None
        
    def get_all_goals(self) -> Dict[str, Any]:
        """Get all goals (active and completed)."""
        return {
            'active_goals': [goal.to_dict() for goal in self.active_goals.values()],
            'completed_goals': [goal.to_dict() for goal in self.completed_goals.values()],
            'planning_history': self.planning_history
        }
        
    def cancel_goal(self, goal_id: str) -> bool:
        """Cancel an active goal."""
        if goal_id in self.active_goals:
            goal = self.active_goals[goal_id]
            goal.status = TaskStatus.CANCELLED
            
            # Cancel all pending tasks
            for task in goal.tasks:
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    
            return True
        return False
        
    def get_planner_stats(self) -> Dict[str, Any]:
        """Get planner statistics."""
        total_goals = len(self.active_goals) + len(self.completed_goals)
        completed_goals = len(self.completed_goals)
        
        return {
            'total_goals': total_goals,
            'active_goals': len(self.active_goals),
            'completed_goals': completed_goals,
            'completion_rate': completed_goals / total_goals if total_goals > 0 else 0,
            'planning_history_count': len(self.planning_history)
        }
