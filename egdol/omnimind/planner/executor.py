"""
Task Executor for OmniMind
Executes individual tasks and manages skill routing.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .planner import Task, TaskStatus


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    skill_used: Optional[str] = None
    reasoning_trace: List[str] = None
    
    def __post_init__(self):
        if self.reasoning_trace is None:
            self.reasoning_trace = []
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'task_id': self.task_id,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'skill_used': self.skill_used,
            'reasoning_trace': self.reasoning_trace
        }


class TaskExecutor:
    """Executes individual tasks and manages skill routing."""
    
    def __init__(self, skill_router, dsl_compiler=None):
        self.skill_router = skill_router
        self.dsl_compiler = dsl_compiler
        self.execution_history: List[ExecutionResult] = []
        self.skill_performance: Dict[str, List[float]] = {}
        
    def execute_task(self, task: Task, verbose: bool = False) -> ExecutionResult:
        """Execute a single task."""
        start_time = time.time()
        
        result = ExecutionResult(
            task_id=task.id,
            success=False,
            reasoning_trace=[]
        )
        
        if verbose:
            result.reasoning_trace.append(f"Starting execution of task: {task.name}")
            result.reasoning_trace.append(f"Task description: {task.description}")
            result.reasoning_trace.append(f"Required skill: {task.skill_required}")
            
        try:
            # Route task to appropriate skill
            skill_result = self._route_to_skill(task, verbose)
            
            if skill_result['success']:
                result.success = True
                result.result = skill_result['result']
                result.skill_used = task.skill_required
                
                if verbose:
                    result.reasoning_trace.append(f"Task completed successfully using {task.skill_required}")
                    result.reasoning_trace.append(f"Result: {skill_result['result']}")
            else:
                result.error = skill_result.get('error', 'Unknown error')
                
                if verbose:
                    result.reasoning_trace.append(f"Task failed: {result.error}")
                    
        except Exception as e:
            result.error = str(e)
            if verbose:
                result.reasoning_trace.append(f"Task execution error: {str(e)}")
                
        # Record execution time
        result.execution_time = time.time() - start_time
        
        # Update skill performance
        if task.skill_required not in self.skill_performance:
            self.skill_performance[task.skill_required] = []
        self.skill_performance[task.skill_required].append(result.execution_time)
        
        # Store execution history
        self.execution_history.append(result)
        
        return result
        
    def _route_to_skill(self, task: Task, verbose: bool = False) -> Dict[str, Any]:
        """Route task to appropriate skill."""
        if verbose:
            print(f"Routing task '{task.name}' to skill '{task.skill_required}'")
            
        # Get available skills
        available_skills = self.skill_router.get_available_skills()
        
        if task.skill_required not in available_skills:
            return {
                'success': False,
                'error': f"Skill '{task.skill_required}' not available"
            }
            
        # Prepare input for skill
        skill_input = {
            'task_name': task.name,
            'task_description': task.description,
            'parameters': task.parameters,
            'context': {}
        }
        
        # Add DSL context if available
        if self.dsl_compiler:
            skill_input['context']['dsl_compiler'] = self.dsl_compiler
            
        # Execute skill
        try:
            skill_result = self.skill_router.execute_skill(
                task.skill_required,
                skill_input,
                verbose
            )
            
            return {
                'success': True,
                'result': skill_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Skill execution error: {str(e)}"
            }
            
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'skill_performance': {}
            }
            
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for result in self.execution_history if result.success)
        success_rate = successful_executions / total_executions
        
        total_time = sum(result.execution_time for result in self.execution_history)
        average_time = total_time / total_executions
        
        # Calculate skill performance
        skill_performance = {}
        for skill, times in self.skill_performance.items():
            skill_performance[skill] = {
                'execution_count': len(times),
                'average_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
            
        return {
            'total_executions': total_executions,
            'success_rate': success_rate,
            'average_execution_time': average_time,
            'skill_performance': skill_performance
        }
        
    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution results."""
        recent = self.execution_history[-limit:]
        return [result.to_dict() for result in recent]
        
    def clear_history(self):
        """Clear execution history."""
        self.execution_history.clear()
        self.skill_performance.clear()
