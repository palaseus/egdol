"""
Task Scheduler for OmniMind
Manages task scheduling and execution order.
"""

import time
import heapq
from typing import Dict, Any, List, Optional, Tuple
from .planner import Task, TaskStatus, TaskPriority


class TaskScheduler:
    """Manages task scheduling and execution order."""
    
    def __init__(self):
        self.task_queue = []
        self.scheduled_tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []
        self.dependency_graph: Dict[str, List[str]] = {}
        
    def schedule_task(self, task: Task) -> bool:
        """Schedule a task for execution."""
        if task.id in self.scheduled_tasks:
            return False
            
        # Add to dependency graph
        self.dependency_graph[task.id] = task.dependencies.copy()
        
        # Schedule task based on priority
        priority_score = self._calculate_priority_score(task)
        heapq.heappush(self.task_queue, (priority_score, time.time(), task.id))
        
        self.scheduled_tasks[task.id] = task
        return True
        
    def schedule_tasks(self, tasks: List[Task]) -> List[bool]:
        """Schedule multiple tasks."""
        results = []
        for task in tasks:
            result = self.schedule_task(task)
            results.append(result)
        return results
        
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute."""
        # First, try to get a ready task from the queue
        if self.task_queue:
            # Find next available task
            attempts = 0
            max_attempts = len(self.task_queue)
            
            while self.task_queue and attempts < max_attempts:
                priority_score, timestamp, task_id = heapq.heappop(self.task_queue)
                attempts += 1
                
                if task_id not in self.scheduled_tasks:
                    continue
                    
                task = self.scheduled_tasks[task_id]
                
                # Check if task is ready (dependencies satisfied)
                if self._is_task_ready(task):
                    return task
                else:
                    # Re-schedule task for later
                    heapq.heappush(self.task_queue, (priority_score, timestamp, task_id))
                    
        # If no task from queue, try to get from ready tasks
        ready_tasks = self.get_ready_tasks()
        if ready_tasks:
            # Return the first ready task
            return ready_tasks[0]
            
        return None
        
    def mark_task_completed(self, task_id: str):
        """Mark a task as completed."""
        if task_id in self.scheduled_tasks:
            task = self.scheduled_tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            # Remove from dependency graph
            if task_id in self.dependency_graph:
                del self.dependency_graph[task_id]
                
            # Update execution order
            self.execution_order.append(task_id)
            
    def mark_task_failed(self, task_id: str, error: str):
        """Mark a task as failed."""
        if task_id in self.scheduled_tasks:
            task = self.scheduled_tasks[task_id]
            task.status = TaskStatus.FAILED
            task.error = error
            
    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute."""
        ready_tasks = []
        
        for task_id, task in self.scheduled_tasks.items():
            if task.status == TaskStatus.PENDING and self._is_task_ready(task):
                ready_tasks.append(task)
                
        return ready_tasks
        
    def get_blocked_tasks(self) -> List[Task]:
        """Get all tasks that are blocked by dependencies."""
        blocked_tasks = []
        
        for task_id, task in self.scheduled_tasks.items():
            if task.status == TaskStatus.PENDING and not self._is_task_ready(task):
                blocked_tasks.append(task)
                
        return blocked_tasks
        
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        total_tasks = len(self.scheduled_tasks)
        pending_tasks = sum(1 for task in self.scheduled_tasks.values() if task.status == TaskStatus.PENDING)
        running_tasks = sum(1 for task in self.scheduled_tasks.values() if task.status == TaskStatus.RUNNING)
        completed_tasks = sum(1 for task in self.scheduled_tasks.values() if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in self.scheduled_tasks.values() if task.status == TaskStatus.FAILED)
        
        return {
            'total_tasks': total_tasks,
            'pending_tasks': pending_tasks,
            'running_tasks': running_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'queue_size': len(self.task_queue),
            'execution_order_length': len(self.execution_order)
        }
        
    def _calculate_priority_score(self, task: Task) -> float:
        """Calculate priority score for task scheduling."""
        # Base priority score
        priority_scores = {
            TaskPriority.LOW: 1.0,
            TaskPriority.NORMAL: 2.0,
            TaskPriority.HIGH: 3.0,
            TaskPriority.CRITICAL: 4.0
        }
        
        base_score = priority_scores.get(task.priority, 2.0)
        
        # Adjust for dependencies (fewer dependencies = higher priority)
        dependency_penalty = len(task.dependencies) * 0.1
        
        # Adjust for retry count (more retries = lower priority)
        retry_penalty = task.retry_count * 0.2
        
        return base_score - dependency_penalty - retry_penalty
        
    def _is_task_ready(self, task: Task) -> bool:
        """Check if a task is ready to execute."""
        if task.status != TaskStatus.PENDING:
            return False
            
        # If no dependencies, task is ready
        if not task.dependencies:
            return True
            
        # Check if all dependencies are completed
        for dep_id in task.dependencies:
            if dep_id not in self.scheduled_tasks:
                return False
                
            dep_task = self.scheduled_tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
                
        return True
        
    def get_task_dependencies(self, task_id: str) -> List[str]:
        """Get dependencies for a task."""
        return self.dependency_graph.get(task_id, [])
        
    def get_dependent_tasks(self, task_id: str) -> List[str]:
        """Get tasks that depend on the given task."""
        dependent_tasks = []
        
        for other_task_id, dependencies in self.dependency_graph.items():
            if task_id in dependencies:
                dependent_tasks.append(other_task_id)
                
        return dependent_tasks
        
    def clear_scheduler(self):
        """Clear all scheduled tasks."""
        self.task_queue.clear()
        self.scheduled_tasks.clear()
        self.execution_order.clear()
        self.dependency_graph.clear()
