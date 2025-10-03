"""
Behavior Scheduler for Egdol.
Provides scheduled reasoning and autonomous behaviors.
"""

import time
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto


class TaskType(Enum):
    """Types of scheduled tasks."""
    PERIODIC = auto()
    ONCE = auto()
    CONDITIONAL = auto()


@dataclass
class ScheduledTask:
    """A scheduled task for autonomous behavior."""
    name: str
    task_type: TaskType
    function: Callable
    interval: float = None  # For periodic tasks
    condition: Callable = None  # For conditional tasks
    last_run: float = None
    next_run: float = None
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_run is None:
            self.last_run = 0.0
        if self.next_run is None and self.interval:
            self.next_run = time.time() + self.interval


class BehaviorScheduler:
    """Schedules and manages autonomous behaviors."""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
    def add_task(self, name: str, function: Callable, 
                 interval: float = None, condition: Callable = None,
                 task_type: TaskType = TaskType.PERIODIC,
                 metadata: Dict[str, Any] = None) -> bool:
        """Add a scheduled task."""
        if name in self.tasks:
            return False
            
        task = ScheduledTask(
            name=name,
            task_type=task_type,
            function=function,
            interval=interval,
            condition=condition,
            metadata=metadata or {}
        )
        
        self.tasks[name] = task
        return True
        
    def remove_task(self, name: str) -> bool:
        """Remove a scheduled task."""
        if name in self.tasks:
            del self.tasks[name]
            return True
        return False
        
    def enable_task(self, name: str) -> bool:
        """Enable a task."""
        if name in self.tasks:
            self.tasks[name].enabled = True
            return True
        return False
        
    def disable_task(self, name: str) -> bool:
        """Disable a task."""
        if name in self.tasks:
            self.tasks[name].enabled = False
            return True
        return False
        
    def start(self):
        """Start the scheduler."""
        if self.running:
            return
            
        self.running = True
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        self.stop_event.set()
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1.0)
            
    def _run_scheduler(self):
        """Main scheduler loop."""
        while self.running and not self.stop_event.is_set():
            current_time = time.time()
            
            for task in self.tasks.values():
                if not task.enabled:
                    continue
                    
                should_run = False
                
                if task.task_type == TaskType.PERIODIC:
                    if task.next_run and current_time >= task.next_run:
                        should_run = True
                        task.next_run = current_time + task.interval
                        
                elif task.task_type == TaskType.ONCE:
                    if task.next_run and current_time >= task.next_run:
                        should_run = True
                        task.enabled = False  # Disable after running once
                        
                elif task.task_type == TaskType.CONDITIONAL:
                    if task.condition and task.condition():
                        should_run = True
                        
                if should_run:
                    try:
                        task.function()
                        task.last_run = current_time
                    except Exception as e:
                        print(f"Error in task '{task.name}': {e}")
                        
            # Sleep for a short interval
            time.sleep(0.1)
            
    def get_task_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        if name not in self.tasks:
            return None
            
        task = self.tasks[name]
        return {
            'name': task.name,
            'type': task.task_type.name,
            'enabled': task.enabled,
            'last_run': task.last_run,
            'next_run': task.next_run,
            'interval': task.interval,
            'metadata': task.metadata
        }
        
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks."""
        return [self.get_task_status(name) for name in self.tasks.keys()]
        
    def run_task_now(self, name: str) -> bool:
        """Run a task immediately."""
        if name not in self.tasks:
            return False
            
        task = self.tasks[name]
        if not task.enabled:
            return False
            
        try:
            task.function()
            task.last_run = time.time()
            return True
        except Exception as e:
            print(f"Error running task '{name}': {e}")
            return False
            
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        total_tasks = len(self.tasks)
        enabled_tasks = len([t for t in self.tasks.values() if t.enabled])
        
        task_types = {}
        for task in self.tasks.values():
            task_type = task.task_type.name
            task_types[task_type] = task_types.get(task_type, 0) + 1
            
        return {
            'total_tasks': total_tasks,
            'enabled_tasks': enabled_tasks,
            'running': self.running,
            'task_types': task_types
        }
