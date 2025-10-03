"""
Network Monitoring System for OmniMind
Monitors network performance, detects conflicts, and analyzes emergent behavior.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque
import statistics


class MonitorType(Enum):
    """Types of monitoring."""
    PERFORMANCE = auto()
    CONFLICT = auto()
    BOTTLENECK = auto()
    EMERGENT_BEHAVIOR = auto()
    LEARNING_EFFICIENCY = auto()
    COMMUNICATION_PATTERN = auto()


class AlertLevel(Enum):
    """Alert levels for monitoring."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class NetworkAlert:
    """A network alert."""
    id: str
    alert_type: str
    level: AlertLevel
    message: str
    agent_id: Optional[str]
    timestamp: float
    resolved: bool = False
    resolution_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resolution_data is None:
            self.resolution_data = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'alert_type': self.alert_type,
            'level': self.level.name,
            'message': self.message,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp,
            'resolved': self.resolved,
            'resolution_data': self.resolution_data
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for the network."""
    timestamp: float
    total_agents: int
    active_agents: int
    total_communications: int
    average_response_time: float
    network_efficiency: float
    learning_rate: float
    conflict_count: int
    bottleneck_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp,
            'total_agents': self.total_agents,
            'active_agents': self.active_agents,
            'total_communications': self.total_communications,
            'average_response_time': self.average_response_time,
            'network_efficiency': self.network_efficiency,
            'learning_rate': self.learning_rate,
            'conflict_count': self.conflict_count,
            'bottleneck_count': self.bottleneck_count
        }


class NetworkMonitor:
    """Monitors the network for issues and performance."""
    
    def __init__(self, network):
        self.network = network
        self.alerts: Dict[str, NetworkAlert] = {}
        self.performance_history: List[PerformanceMetrics] = []
        self.monitoring_rules: Dict[str, Dict[str, Any]] = {}
        self.monitoring_history: List[Dict[str, Any]] = []
        
    def add_monitoring_rule(self, rule_id: str, rule_type: str, 
                           conditions: Dict[str, Any], alert_level: AlertLevel) -> bool:
        """Add a monitoring rule."""
        rule = {
            'id': rule_id,
            'rule_type': rule_type,
            'conditions': conditions,
            'alert_level': alert_level,
            'active': True,
            'created_at': time.time()
        }
        
        self.monitoring_rules[rule_id] = rule
        
        # Log monitoring event
        self._log_monitoring_event('rule_added', {
            'rule_id': rule_id,
            'rule_type': rule_type,
            'conditions': conditions,
            'alert_level': alert_level.name
        })
        
        return True
        
    def check_network_health(self) -> List[NetworkAlert]:
        """Check network health and generate alerts."""
        alerts = []
        
        # Check agent status
        for agent_id, agent in self.network.agents.items():
            # Check for idle agents
            if agent.status.value == 1:  # IDLE
                if time.time() - agent.last_activity > 300:  # 5 minutes
                    alert = NetworkAlert(
                        id=str(uuid.uuid4()),
                        alert_type='idle_agent',
                        level=AlertLevel.WARNING,
                        message=f'Agent {agent.name} has been idle for over 5 minutes',
                        agent_id=agent_id,
                        timestamp=time.time()
                    )
                    alerts.append(alert)
                    
            # Check for overloaded agents
            if hasattr(agent, 'performance_metrics'):
                cpu_usage = agent.performance_metrics.get('cpu_usage', 0)
                if cpu_usage > 90:
                    alert = NetworkAlert(
                        id=str(uuid.uuid4()),
                        alert_type='overloaded_agent',
                        level=AlertLevel.ERROR,
                        message=f'Agent {agent.name} has high CPU usage: {cpu_usage}%',
                        agent_id=agent_id,
                        timestamp=time.time()
                    )
                    alerts.append(alert)
                    
        # Check network connectivity
        isolated_agents = self._detect_isolated_agents()
        for agent_id in isolated_agents:
            alert = NetworkAlert(
                id=str(uuid.uuid4()),
                alert_type='isolated_agent',
                level=AlertLevel.WARNING,
                message=f'Agent {agent_id} is isolated from the network',
                agent_id=agent_id,
                timestamp=time.time()
            )
            alerts.append(alert)
            
        # Check communication patterns
        communication_issues = self._detect_communication_issues()
        for issue in communication_issues:
            alert = NetworkAlert(
                id=str(uuid.uuid4()),
                alert_type='communication_issue',
                level=AlertLevel.WARNING,
                message=issue['description'],
                agent_id=issue.get('agent_id'),
                timestamp=time.time()
            )
            alerts.append(alert)
            
        # Store alerts
        for alert in alerts:
            self.alerts[alert.id] = alert
            
        return alerts
        
    def capture_performance_metrics(self) -> PerformanceMetrics:
        """Capture current performance metrics."""
        total_agents = len(self.network.agents)
        active_agents = sum(1 for agent in self.network.agents.values() 
                           if agent.status.value == 0)  # ACTIVE
        
        # Calculate communication metrics
        total_communications = sum(len(agent.communication_history) 
                                for agent in self.network.agents.values())
        
        # Calculate average response time
        response_times = []
        for agent in self.network.agents.values():
            if hasattr(agent, 'performance_metrics'):
                response_time = agent.performance_metrics.get('average_response_time', 0)
                if response_time > 0:
                    response_times.append(response_time)
                    
        average_response_time = statistics.mean(response_times) if response_times else 0
        
        # Calculate network efficiency
        network_efficiency = self._calculate_network_efficiency()
        
        # Calculate learning rate
        learning_rate = self._calculate_learning_rate()
        
        # Count conflicts and bottlenecks
        conflict_count = len([alert for alert in self.alerts.values() 
                            if alert.alert_type == 'conflict'])
        bottleneck_count = len([alert for alert in self.alerts.values() 
                               if alert.alert_type == 'bottleneck'])
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            total_agents=total_agents,
            active_agents=active_agents,
            total_communications=total_communications,
            average_response_time=average_response_time,
            network_efficiency=network_efficiency,
            learning_rate=learning_rate,
            conflict_count=conflict_count,
            bottleneck_count=bottleneck_count
        )
        
        self.performance_history.append(metrics)
        
        return metrics
        
    def _detect_isolated_agents(self) -> List[str]:
        """Detect isolated agents in the network."""
        isolated_agents = []
        
        for agent_id, agent in self.network.agents.items():
            # Check if agent has connections
            if agent_id not in self.network.connections:
                isolated_agents.append(agent_id)
                continue
                
            # Check if agent has active connections
            active_connections = 0
            for connected_id in self.network.connections[agent_id]:
                if connected_id in self.network.agents:
                    connected_agent = self.network.agents[connected_id]
                    if connected_agent.status.value == 0:  # ACTIVE
                        active_connections += 1
                        
            if active_connections == 0:
                isolated_agents.append(agent_id)
                
        return isolated_agents
        
    def _detect_communication_issues(self) -> List[Dict[str, Any]]:
        """Detect communication issues in the network."""
        issues = []
        
        # Check for communication bottlenecks
        communication_counts = defaultdict(int)
        for agent in self.network.agents.values():
            communication_counts[agent.id] = len(agent.communication_history)
            
        if communication_counts:
            max_communications = max(communication_counts.values())
            min_communications = min(communication_counts.values())
            
            # Check for significant imbalance
            if max_communications > min_communications * 5:
                issues.append({
                    'type': 'communication_imbalance',
                    'description': 'Significant communication imbalance detected',
                    'max_communications': max_communications,
                    'min_communications': min_communications
                })
                
        # Check for communication patterns
        for agent_id, agent in self.network.agents.items():
            if len(agent.communication_history) > 100:  # High communication threshold
                issues.append({
                    'type': 'high_communication',
                    'agent_id': agent_id,
                    'description': f'Agent {agent.name} has high communication activity',
                    'communication_count': len(agent.communication_history)
                })
                
        return issues
        
    def _calculate_network_efficiency(self) -> float:
        """Calculate network efficiency score."""
        if len(self.network.agents) < 2:
            return 1.0
            
        # Calculate connectivity ratio
        total_possible_connections = len(self.network.agents) * (len(self.network.agents) - 1) // 2
        actual_connections = sum(len(connections) for connections in self.network.connections.values()) // 2
        
        connectivity_ratio = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
        # Calculate communication efficiency
        total_communications = sum(len(agent.communication_history) 
                                for agent in self.network.agents.values())
        communication_efficiency = min(total_communications / (len(self.network.agents) * 10), 1.0)
        
        # Calculate skill diversity
        all_skills = set()
        for agent in self.network.agents.values():
            all_skills.update(agent.skills)
            
        skill_diversity = len(all_skills) / (len(self.network.agents) * 5) if self.network.agents else 0
        
        # Combine metrics
        efficiency = (connectivity_ratio * 0.4 + communication_efficiency * 0.3 + skill_diversity * 0.3)
        return min(efficiency, 1.0)
        
    def _calculate_learning_rate(self) -> float:
        """Calculate learning rate in the network."""
        # This would need to be implemented based on actual learning events
        # For now, return a placeholder
        return 0.5
        
    def get_performance_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance trends over time."""
        if not self.performance_history:
            return {}
            
        # Filter metrics by time range
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
            
        # Calculate trends
        network_efficiency_trend = self._calculate_trend([m.network_efficiency for m in recent_metrics])
        learning_rate_trend = self._calculate_trend([m.learning_rate for m in recent_metrics])
        conflict_trend = self._calculate_trend([m.conflict_count for m in recent_metrics])
        
        return {
            'time_range_hours': hours,
            'data_points': len(recent_metrics),
            'network_efficiency_trend': network_efficiency_trend,
            'learning_rate_trend': learning_rate_trend,
            'conflict_trend': conflict_trend,
            'average_efficiency': statistics.mean([m.network_efficiency for m in recent_metrics]),
            'average_learning_rate': statistics.mean([m.learning_rate for m in recent_metrics]),
            'average_conflicts': statistics.mean([m.conflict_count for m in recent_metrics])
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
            
    def get_active_alerts(self) -> List[NetworkAlert]:
        """Get active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
        
    def resolve_alert(self, alert_id: str, resolution_data: Dict[str, Any] = None) -> bool:
        """Resolve an alert."""
        if alert_id not in self.alerts:
            return False
            
        alert = self.alerts[alert_id]
        alert.resolved = True
        alert.resolution_data = resolution_data or {}
        
        # Log monitoring event
        self._log_monitoring_event('alert_resolved', {
            'alert_id': alert_id,
            'resolution_data': resolution_data
        })
        
        return True
        
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        total_alerts = len(self.alerts)
        resolved_alerts = sum(1 for alert in self.alerts.values() if alert.resolved)
        active_alerts = total_alerts - resolved_alerts
        
        # Calculate alert distribution by level
        alert_distribution = defaultdict(int)
        for alert in self.alerts.values():
            alert_distribution[alert.level.name] += 1
            
        # Calculate alert distribution by type
        type_distribution = defaultdict(int)
        for alert in self.alerts.values():
            type_distribution[alert.alert_type] += 1
            
        return {
            'total_alerts': total_alerts,
            'resolved_alerts': resolved_alerts,
            'active_alerts': active_alerts,
            'resolution_rate': resolved_alerts / total_alerts if total_alerts > 0 else 0,
            'alert_distribution': dict(alert_distribution),
            'type_distribution': dict(type_distribution)
        }
        
    def _log_monitoring_event(self, event_type: str, data: Dict[str, Any]):
        """Log a monitoring event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.monitoring_history.append(event)
        
    def get_monitoring_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get monitoring history."""
        return list(self.monitoring_history[-limit:])


class PerformanceAnalyzer:
    """Analyzes network performance and provides insights."""
    
    def __init__(self, network):
        self.network = network
        self.analysis_history: List[Dict[str, Any]] = []
        
    def analyze_network_performance(self) -> Dict[str, Any]:
        """Analyze overall network performance."""
        analysis = {
            'timestamp': time.time(),
            'network_size': len(self.network.agents),
            'connectivity_score': self._calculate_connectivity_score(),
            'communication_efficiency': self._calculate_communication_efficiency(),
            'skill_diversity': self._calculate_skill_diversity(),
            'learning_efficiency': self._calculate_learning_efficiency(),
            'bottlenecks': self._identify_bottlenecks(),
            'recommendations': self._generate_recommendations()
        }
        
        # Log analysis
        self._log_analysis('network_performance_analysis', analysis)
        
        return analysis
        
    def _calculate_connectivity_score(self) -> float:
        """Calculate network connectivity score."""
        if len(self.network.agents) < 2:
            return 1.0
            
        # Calculate connectivity ratio
        total_possible_connections = len(self.network.agents) * (len(self.network.agents) - 1) // 2
        actual_connections = sum(len(connections) for connections in self.network.connections.values()) // 2
        
        return actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
    def _calculate_communication_efficiency(self) -> float:
        """Calculate communication efficiency score."""
        total_communications = sum(len(agent.communication_history) 
                                for agent in self.network.agents.values())
        
        if not total_communications:
            return 0.0
            
        # Normalize by network size
        normalized_communications = total_communications / len(self.network.agents)
        
        # Score based on communication activity
        if normalized_communications < 5:
            return 0.3
        elif normalized_communications < 20:
            return 0.7
        else:
            return 1.0
            
    def _calculate_skill_diversity(self) -> float:
        """Calculate skill diversity score."""
        all_skills = set()
        for agent in self.network.agents.values():
            all_skills.update(agent.skills)
            
        if not all_skills:
            return 0.0
            
        # Calculate diversity ratio
        total_skills = sum(len(agent.skills) for agent in self.network.agents.values())
        diversity_ratio = len(all_skills) / total_skills if total_skills > 0 else 0
        
        return min(diversity_ratio, 1.0)
        
    def _calculate_learning_efficiency(self) -> float:
        """Calculate learning efficiency score."""
        # This would need to be implemented based on actual learning events
        # For now, return a placeholder
        return 0.5
        
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify network bottlenecks."""
        bottlenecks = []
        
        # Check for overloaded agents
        for agent_id, agent in self.network.agents.items():
            if hasattr(agent, 'performance_metrics'):
                cpu_usage = agent.performance_metrics.get('cpu_usage', 0)
                if cpu_usage > 80:
                    bottlenecks.append({
                        'type': 'agent_overload',
                        'agent_id': agent_id,
                        'severity': 'high',
                        'description': f'Agent {agent.name} is overloaded (CPU: {cpu_usage}%)'
                    })
                    
        # Check for communication bottlenecks
        communication_counts = defaultdict(int)
        for agent in self.network.agents.values():
            communication_counts[agent.id] = len(agent.communication_history)
            
        if communication_counts:
            max_communications = max(communication_counts.values())
            if max_communications > 50:  # High communication threshold
                bottlenecks.append({
                    'type': 'communication_bottleneck',
                    'severity': 'medium',
                    'description': 'High communication activity detected'
                })
                
        return bottlenecks
        
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check network size
        if len(self.network.agents) < 3:
            recommendations.append({
                'type': 'expand_network',
                'priority': 'medium',
                'description': 'Consider adding more agents to improve network resilience'
            })
            
        # Check connectivity
        connectivity_score = self._calculate_connectivity_score()
        if connectivity_score < 0.5:
            recommendations.append({
                'type': 'improve_connectivity',
                'priority': 'high',
                'description': 'Network connectivity is low, consider adding more connections'
            })
            
        # Check skill diversity
        skill_diversity = self._calculate_skill_diversity()
        if skill_diversity < 0.3:
            recommendations.append({
                'type': 'improve_skill_diversity',
                'priority': 'medium',
                'description': 'Network has low skill diversity, consider adding agents with different skills'
            })
            
        return recommendations
        
    def _log_analysis(self, analysis_type: str, data: Dict[str, Any]):
        """Log an analysis."""
        analysis = {
            'id': str(uuid.uuid4()),
            'type': analysis_type,
            'timestamp': time.time(),
            'data': data
        }
        self.analysis_history.append(analysis)
        
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return list(self.analysis_history[-limit:])


class ConflictDetector:
    """Detects and resolves conflicts in the network."""
    
    def __init__(self, network):
        self.network = network
        self.conflicts: Dict[str, Dict[str, Any]] = {}
        self.conflict_history: List[Dict[str, Any]] = []
        
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicts in the network."""
        conflicts = []
        
        # Check for conflicting knowledge
        knowledge_conflicts = self._detect_knowledge_conflicts()
        conflicts.extend(knowledge_conflicts)
        
        # Check for conflicting skills
        skill_conflicts = self._detect_skill_conflicts()
        conflicts.extend(skill_conflicts)
        
        # Check for conflicting goals
        goal_conflicts = self._detect_goal_conflicts()
        conflicts.extend(goal_conflicts)
        
        # Store conflicts
        for conflict in conflicts:
            conflict_id = str(uuid.uuid4())
            self.conflicts[conflict_id] = conflict
            
        return conflicts
        
    def _detect_knowledge_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicting knowledge between agents."""
        conflicts = []
        
        # This would need to be implemented based on actual knowledge comparison
        # For now, return empty list
        return conflicts
        
    def _detect_skill_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicting skills between agents."""
        conflicts = []
        
        # Check for duplicate skills with different implementations
        skill_implementations = defaultdict(list)
        
        for agent_id, agent in self.network.agents.items():
            for skill in agent.skills:
                skill_implementations[skill].append(agent_id)
                
        # Find skills with multiple implementations
        for skill, agent_ids in skill_implementations.items():
            if len(agent_ids) > 1:
                conflicts.append({
                    'type': 'skill_conflict',
                    'skill_name': skill,
                    'conflicting_agents': agent_ids,
                    'severity': 'medium',
                    'description': f'Multiple agents have conflicting implementations of skill {skill}'
                })
                
        return conflicts
        
    def _detect_goal_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicting goals between agents."""
        conflicts = []
        
        # This would need to be implemented based on actual goal comparison
        # For now, return empty list
        return conflicts
        
    def resolve_conflict(self, conflict_id: str, resolution_method: str,
                        resolution_data: Dict[str, Any] = None) -> bool:
        """Resolve a conflict."""
        if conflict_id not in self.conflicts:
            return False
            
        conflict = self.conflicts[conflict_id]
        conflict['resolved'] = True
        conflict['resolution_method'] = resolution_method
        conflict['resolution_data'] = resolution_data or {}
        conflict['resolved_at'] = time.time()
        
        # Log conflict resolution
        self._log_conflict_event('conflict_resolved', {
            'conflict_id': conflict_id,
            'resolution_method': resolution_method,
            'resolution_data': resolution_data
        })
        
        return True
        
    def get_conflict_statistics(self) -> Dict[str, Any]:
        """Get conflict statistics."""
        total_conflicts = len(self.conflicts)
        resolved_conflicts = sum(1 for conflict in self.conflicts.values() 
                               if conflict.get('resolved', False))
        
        # Calculate conflict distribution by type
        type_distribution = defaultdict(int)
        for conflict in self.conflicts.values():
            type_distribution[conflict['type']] += 1
            
        return {
            'total_conflicts': total_conflicts,
            'resolved_conflicts': resolved_conflicts,
            'active_conflicts': total_conflicts - resolved_conflicts,
            'resolution_rate': resolved_conflicts / total_conflicts if total_conflicts > 0 else 0,
            'type_distribution': dict(type_distribution)
        }
        
    def _log_conflict_event(self, event_type: str, data: Dict[str, Any]):
        """Log a conflict event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.conflict_history.append(event)
        
    def get_conflict_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get conflict history."""
        return list(self.conflict_history[-limit:])

