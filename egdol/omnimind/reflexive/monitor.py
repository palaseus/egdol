"""
Performance Monitor for OmniMind
Tracks performance metrics and detects bottlenecks.
"""

import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque
import statistics


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    execution_time: float
    task_count: int
    active_goals: int
    memory_entries: int
    skill_executions: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'memory_available': self.memory_available,
            'execution_time': self.execution_time,
            'task_count': self.task_count,
            'active_goals': self.active_goals,
            'memory_entries': self.memory_entries,
            'skill_executions': self.skill_executions
        }


class PerformanceMonitor:
    """Monitors system performance and detects bottlenecks."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.bottleneck_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'execution_time': 10.0,
            'task_count': 50
        }
        self.alert_history: List[Dict[str, Any]] = []
        
    def capture_metrics(self, planner=None, memory_manager=None, 
                       skill_router=None) -> PerformanceMetrics:
        """Capture current performance metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        
        # Application metrics
        execution_time = 0.0
        task_count = 0
        active_goals = 0
        memory_entries = 0
        skill_executions = 0
        
        if planner:
            planner_stats = planner.get_planner_stats()
            execution_stats = planner.get_execution_statistics()
            execution_time = execution_stats.get('average_execution_time', 0)
            active_goals = planner_stats.get('active_goals', 0)
            task_count = sum(len(goal.get('tasks', [])) for goal in planner.get_all_goals().get('active_goals', []))
            
        if memory_manager:
            memory_stats = memory_manager.get_memory_statistics()
            memory_entries = memory_stats.get('total_entries', 0)
            
        if skill_router:
            skill_stats = skill_router.get_skill_statistics()
            skill_executions = sum(stats.get('execution_count', 0) for stats in skill_stats.values())
            
        # Create metrics
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            execution_time=execution_time,
            task_count=task_count,
            active_goals=active_goals,
            memory_entries=memory_entries,
            skill_executions=skill_executions
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Check for bottlenecks
        self._check_bottlenecks(metrics)
        
        return metrics
        
    def _check_bottlenecks(self, metrics: PerformanceMetrics):
        """Check for performance bottlenecks."""
        alerts = []
        
        # Check CPU usage
        if metrics.cpu_usage > self.bottleneck_thresholds['cpu_usage']:
            alerts.append({
                'type': 'high_cpu_usage',
                'value': metrics.cpu_usage,
                'threshold': self.bottleneck_thresholds['cpu_usage'],
                'severity': 'high'
            })
            
        # Check memory usage
        if metrics.memory_usage > self.bottleneck_thresholds['memory_usage']:
            alerts.append({
                'type': 'high_memory_usage',
                'value': metrics.memory_usage,
                'threshold': self.bottleneck_thresholds['memory_usage'],
                'severity': 'high'
            })
            
        # Check execution time
        if metrics.execution_time > self.bottleneck_thresholds['execution_time']:
            alerts.append({
                'type': 'slow_execution',
                'value': metrics.execution_time,
                'threshold': self.bottleneck_thresholds['execution_time'],
                'severity': 'medium'
            })
            
        # Check task count
        if metrics.task_count > self.bottleneck_thresholds['task_count']:
            alerts.append({
                'type': 'high_task_count',
                'value': metrics.task_count,
                'threshold': self.bottleneck_thresholds['task_count'],
                'severity': 'medium'
            })
            
        # Store alerts
        for alert in alerts:
            alert['timestamp'] = metrics.timestamp
            self.alert_history.append(alert)
            
    def get_performance_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance trends over time."""
        if not self.metrics_history:
            return {}
            
        # Filter metrics by time range
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
            
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        execution_trend = self._calculate_trend([m.execution_time for m in recent_metrics])
        
        return {
            'time_range_hours': hours,
            'data_points': len(recent_metrics),
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'execution_trend': execution_trend,
            'average_cpu': statistics.mean([m.cpu_usage for m in recent_metrics]),
            'average_memory': statistics.mean([m.memory_usage for m in recent_metrics]),
            'average_execution_time': statistics.mean([m.execution_time for m in recent_metrics])
        }
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return 'insufficient_data'
            
        # Simple linear trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg * 1.1:
            return 'increasing'
        elif second_avg < first_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'
            
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Analyze current bottlenecks."""
        if not self.metrics_history:
            return {}
            
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        bottlenecks = []
        
        # Analyze CPU usage
        cpu_values = [m.cpu_usage for m in recent_metrics]
        if any(cpu > self.bottleneck_thresholds['cpu_usage'] for cpu in cpu_values):
            bottlenecks.append({
                'type': 'cpu_bottleneck',
                'severity': 'high',
                'description': 'High CPU usage detected',
                'recommendation': 'Optimize computation or reduce task complexity'
            })
            
        # Analyze memory usage
        memory_values = [m.memory_usage for m in recent_metrics]
        if any(mem > self.bottleneck_thresholds['memory_usage'] for mem in memory_values):
            bottlenecks.append({
                'type': 'memory_bottleneck',
                'severity': 'high',
                'description': 'High memory usage detected',
                'recommendation': 'Clean up memory or optimize data structures'
            })
            
        # Analyze execution time
        execution_values = [m.execution_time for m in recent_metrics]
        if any(exec > self.bottleneck_thresholds['execution_time'] for exec in execution_values):
            bottlenecks.append({
                'type': 'execution_bottleneck',
                'severity': 'medium',
                'description': 'Slow execution detected',
                'recommendation': 'Optimize algorithms or reduce task complexity'
            })
            
        return {
            'bottlenecks': bottlenecks,
            'total_bottlenecks': len(bottlenecks),
            'high_severity': len([b for b in bottlenecks if b['severity'] == 'high']),
            'medium_severity': len([b for b in bottlenecks if b['severity'] == 'medium'])
        }
        
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on performance data."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
            
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements
        
        # CPU optimization recommendations
        cpu_values = [m.cpu_usage for m in recent_metrics]
        if statistics.mean(cpu_values) > 70:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'high',
                'description': 'Optimize CPU usage',
                'suggestions': [
                    'Reduce task complexity',
                    'Implement task batching',
                    'Optimize algorithms'
                ]
            })
            
        # Memory optimization recommendations
        memory_values = [m.memory_usage for m in recent_metrics]
        if statistics.mean(memory_values) > 75:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'description': 'Optimize memory usage',
                'suggestions': [
                    'Clean up unused data',
                    'Implement memory pooling',
                    'Optimize data structures'
                ]
            })
            
        # Execution time optimization recommendations
        execution_values = [m.execution_time for m in recent_metrics]
        if statistics.mean(execution_values) > 5:
            recommendations.append({
                'type': 'execution_optimization',
                'priority': 'medium',
                'description': 'Optimize execution time',
                'suggestions': [
                    'Parallelize tasks',
                    'Cache results',
                    'Optimize algorithms'
                ]
            })
            
        return recommendations
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {
                'total_measurements': 0,
                'average_cpu': 0,
                'average_memory': 0,
                'average_execution_time': 0,
                'total_alerts': 0
            }
            
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 measurements
        
        return {
            'total_measurements': len(self.metrics_history),
            'recent_measurements': len(recent_metrics),
            'average_cpu': statistics.mean([m.cpu_usage for m in recent_metrics]),
            'average_memory': statistics.mean([m.memory_usage for m in recent_metrics]),
            'average_execution_time': statistics.mean([m.execution_time for m in recent_metrics]),
            'max_cpu': max(m.cpu_usage for m in recent_metrics),
            'max_memory': max(m.memory_usage for m in recent_metrics),
            'max_execution_time': max(m.execution_time for m in recent_metrics),
            'total_alerts': len(self.alert_history),
            'recent_alerts': len([a for a in self.alert_history if a['timestamp'] > time.time() - 3600])
        }
        
    def set_bottleneck_thresholds(self, thresholds: Dict[str, float]):
        """Set custom bottleneck thresholds."""
        self.bottleneck_thresholds.update(thresholds)
        
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert['timestamp'] >= cutoff_time]
        
    def clear_history(self):
        """Clear performance history."""
        self.metrics_history.clear()
        self.alert_history.clear()
