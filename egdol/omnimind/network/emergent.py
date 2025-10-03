"""
Emergent Behavior System for OmniMind Network
Detects and manages emergent behaviors and collaborative patterns.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque
import statistics


class EmergentType(Enum):
    """Types of emergent behaviors."""
    COLLABORATIVE_PATTERN = auto()
    SELF_ORGANIZATION = auto()
    COLLECTIVE_INTELLIGENCE = auto()
    EMERGENT_LEARNING = auto()
    SWARM_BEHAVIOR = auto()
    ADAPTIVE_COORDINATION = auto()


class PatternStatus(Enum):
    """Status of emergent patterns."""
    EMERGING = auto()
    STABLE = auto()
    DECLINING = auto()
    EXTINCT = auto()


@dataclass
class EmergentPattern:
    """An emergent pattern in the network."""
    id: str
    pattern_type: EmergentType
    description: str
    participating_agents: List[str]
    strength: float
    stability: float
    first_observed: float
    last_observed: float
    frequency: int
    status: PatternStatus
    characteristics: Dict[str, Any] = None
    evolution_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = {}
        if self.evolution_history is None:
            self.evolution_history = []
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            'id': self.id,
            'pattern_type': self.pattern_type.name,
            'description': self.description,
            'participating_agents': self.participating_agents,
            'strength': self.strength,
            'stability': self.stability,
            'first_observed': self.first_observed,
            'last_observed': self.last_observed,
            'frequency': self.frequency,
            'status': self.status.name,
            'characteristics': self.characteristics,
            'evolution_history': self.evolution_history
        }


@dataclass
class CollaborationEvent:
    """A collaboration event in the network."""
    id: str
    event_type: str
    participating_agents: List[str]
    description: str
    timestamp: float
    duration: float
    success: bool
    outcome: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.outcome is None:
            self.outcome = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'participating_agents': self.participating_agents,
            'description': self.description,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'success': self.success,
            'outcome': self.outcome
        }


class EmergentBehavior:
    """Detects and manages emergent behaviors in the network."""
    
    def __init__(self, network):
        self.network = network
        self.patterns: Dict[str, EmergentPattern] = {}
        self.collaboration_events: List[CollaborationEvent] = []
        self.behavior_history: List[Dict[str, Any]] = []
        self.pattern_detectors: Dict[str, callable] = {}
        
    def detect_emergent_patterns(self) -> List[EmergentPattern]:
        """Detect emergent patterns in the network."""
        patterns = []
        
        # Detect collaborative patterns
        collaborative_patterns = self._detect_collaborative_patterns()
        patterns.extend(collaborative_patterns)
        
        # Detect self-organization patterns
        self_org_patterns = self._detect_self_organization_patterns()
        patterns.extend(self_org_patterns)
        
        # Detect collective intelligence patterns
        collective_patterns = self._detect_collective_intelligence_patterns()
        patterns.extend(collective_patterns)
        
        # Detect emergent learning patterns
        learning_patterns = self._detect_emergent_learning_patterns()
        patterns.extend(learning_patterns)
        
        # Store patterns
        for pattern in patterns:
            self.patterns[pattern.id] = pattern
            
        return patterns
        
    def _detect_collaborative_patterns(self) -> List[EmergentPattern]:
        """Detect collaborative patterns between agents."""
        patterns = []
        
        # Analyze communication patterns
        communication_graph = self._build_communication_graph()
        
        # Find highly connected groups
        connected_groups = self._find_connected_groups(communication_graph)
        
        for group in connected_groups:
            if len(group) >= 2:  # Minimum group size
                pattern = EmergentPattern(
                    id=str(uuid.uuid4()),
                    pattern_type=EmergentType.COLLABORATIVE_PATTERN,
                    description=f'Collaborative group of {len(group)} agents',
                    participating_agents=group,
                    strength=self._calculate_group_strength(group),
                    stability=self._calculate_group_stability(group),
                    first_observed=time.time(),
                    last_observed=time.time(),
                    frequency=1,
                    status=PatternStatus.EMERGING
                )
                patterns.append(pattern)
                
        return patterns
        
    def _detect_self_organization_patterns(self) -> List[EmergentPattern]:
        """Detect self-organization patterns."""
        patterns = []
        
        # Analyze agent behavior patterns
        behavior_patterns = self._analyze_agent_behaviors()
        
        for pattern_data in behavior_patterns:
            if pattern_data['type'] == 'self_organization':
                pattern = EmergentPattern(
                    id=str(uuid.uuid4()),
                    pattern_type=EmergentType.SELF_ORGANIZATION,
                    description=pattern_data['description'],
                    participating_agents=pattern_data['agents'],
                    strength=pattern_data['strength'],
                    stability=pattern_data['stability'],
                    first_observed=time.time(),
                    last_observed=time.time(),
                    frequency=1,
                    status=PatternStatus.EMERGING
                )
                patterns.append(pattern)
                
        return patterns
        
    def _detect_collective_intelligence_patterns(self) -> List[EmergentPattern]:
        """Detect collective intelligence patterns."""
        patterns = []
        
        # Analyze collective decision-making
        decision_patterns = self._analyze_collective_decisions()
        
        for pattern_data in decision_patterns:
            if pattern_data['type'] == 'collective_intelligence':
                pattern = EmergentPattern(
                    id=str(uuid.uuid4()),
                    pattern_type=EmergentType.COLLECTIVE_INTELLIGENCE,
                    description=pattern_data['description'],
                    participating_agents=pattern_data['agents'],
                    strength=pattern_data['strength'],
                    stability=pattern_data['stability'],
                    first_observed=time.time(),
                    last_observed=time.time(),
                    frequency=1,
                    status=PatternStatus.EMERGING
                )
                patterns.append(pattern)
                
        return patterns
        
    def _detect_emergent_learning_patterns(self) -> List[EmergentPattern]:
        """Detect emergent learning patterns."""
        patterns = []
        
        # Analyze learning propagation
        learning_patterns = self._analyze_learning_propagation()
        
        for pattern_data in learning_patterns:
            if pattern_data['type'] == 'emergent_learning':
                pattern = EmergentPattern(
                    id=str(uuid.uuid4()),
                    pattern_type=EmergentType.EMERGENT_LEARNING,
                    description=pattern_data['description'],
                    participating_agents=pattern_data['agents'],
                    strength=pattern_data['strength'],
                    stability=pattern_data['stability'],
                    first_observed=time.time(),
                    last_observed=time.time(),
                    frequency=1,
                    status=PatternStatus.EMERGING
                )
                patterns.append(pattern)
                
        return patterns
        
    def _build_communication_graph(self) -> Dict[str, List[str]]:
        """Build communication graph from agent interactions."""
        graph = defaultdict(list)
        
        for agent_id, agent in self.network.agents.items():
            # Analyze communication history
            for comm in agent.communication_history:
                if 'target_agent' in comm:
                    target = comm['target_agent']
                    if target != agent_id:
                        graph[agent_id].append(target)
                        
        return dict(graph)
        
    def _find_connected_groups(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find connected groups in the communication graph."""
        visited = set()
        groups = []
        
        def dfs(node, group):
            if node in visited:
                return
            visited.add(node)
            group.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, group)
                
        for node in graph:
            if node not in visited:
                group = []
                dfs(node, group)
                if len(group) > 1:
                    groups.append(group)
                    
        return groups
        
    def _calculate_group_strength(self, group: List[str]) -> float:
        """Calculate strength of a group."""
        if len(group) < 2:
            return 0.0
            
        # Calculate average communication frequency
        total_communications = 0
        for agent_id in group:
            agent = self.network.get_agent(agent_id)
            if agent:
                total_communications += len(agent.communication_history)
                
        average_communications = total_communications / len(group)
        
        # Normalize to 0-1 scale
        strength = min(average_communications / 10, 1.0)
        return strength
        
    def _calculate_group_stability(self, group: List[str]) -> float:
        """Calculate stability of a group."""
        if len(group) < 2:
            return 0.0
            
        # Calculate consistency of interactions
        interaction_consistency = 0.0
        
        for agent_id in group:
            agent = self.network.get_agent(agent_id)
            if agent and agent.communication_history:
                # Calculate time consistency
                timestamps = [comm.get('timestamp', 0) for comm in agent.communication_history]
                if len(timestamps) > 1:
                    time_intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                    if len(time_intervals) > 1:  # Need at least 2 intervals for stdev
                        try:
                            consistency = 1.0 / (1.0 + statistics.stdev(time_intervals))
                            interaction_consistency += consistency
                        except statistics.StatisticsError:
                            # If stdev fails, use a default consistency
                            interaction_consistency += 0.5
                    else:
                        interaction_consistency += 0.5
                        
        stability = interaction_consistency / len(group)
        return min(stability, 1.0)
        
    def _analyze_agent_behaviors(self) -> List[Dict[str, Any]]:
        """Analyze agent behaviors for patterns."""
        patterns = []
        
        # This would need to be implemented based on actual behavior analysis
        # For now, return empty list
        return patterns
        
    def _analyze_collective_decisions(self) -> List[Dict[str, Any]]:
        """Analyze collective decision-making patterns."""
        patterns = []
        
        # This would need to be implemented based on actual decision analysis
        # For now, return empty list
        return patterns
        
    def _analyze_learning_propagation(self) -> List[Dict[str, Any]]:
        """Analyze learning propagation patterns."""
        patterns = []
        
        # This would need to be implemented based on actual learning analysis
        # For now, return empty list
        return patterns
        
    def record_collaboration_event(self, event_type: str, participating_agents: List[str],
                                 description: str, duration: float, success: bool,
                                 outcome: Dict[str, Any] = None) -> str:
        """Record a collaboration event."""
        event_id = str(uuid.uuid4())
        
        event = CollaborationEvent(
            id=event_id,
            event_type=event_type,
            participating_agents=participating_agents,
            description=description,
            timestamp=time.time(),
            duration=duration,
            success=success,
            outcome=outcome
        )
        
        self.collaboration_events.append(event)
        
        # Log behavior event
        self._log_behavior_event('collaboration_recorded', {
            'event_id': event_id,
            'event_type': event_type,
            'participating_agents': participating_agents,
            'description': description,
            'duration': duration,
            'success': success
        })
        
        return event_id
        
    def update_pattern_status(self, pattern_id: str, new_status: PatternStatus) -> bool:
        """Update the status of an emergent pattern."""
        if pattern_id not in self.patterns:
            return False
            
        pattern = self.patterns[pattern_id]
        old_status = pattern.status
        pattern.status = new_status
        pattern.last_observed = time.time()
        
        # Record evolution
        evolution_entry = {
            'timestamp': time.time(),
            'old_status': old_status.name,
            'new_status': new_status.name,
            'reason': 'status_update'
        }
        pattern.evolution_history.append(evolution_entry)
        
        # Log behavior event
        self._log_behavior_event('pattern_status_updated', {
            'pattern_id': pattern_id,
            'old_status': old_status.name,
            'new_status': new_status.name
        })
        
        return True
        
    def get_emergent_statistics(self) -> Dict[str, Any]:
        """Get emergent behavior statistics."""
        total_patterns = len(self.patterns)
        active_patterns = sum(1 for pattern in self.patterns.values() 
                           if pattern.status in [PatternStatus.EMERGING, PatternStatus.STABLE])
        
        # Calculate pattern distribution by type
        type_distribution = defaultdict(int)
        for pattern in self.patterns.values():
            type_distribution[pattern.pattern_type.name] += 1
            
        # Calculate pattern distribution by status
        status_distribution = defaultdict(int)
        for pattern in self.patterns.values():
            status_distribution[pattern.status.name] += 1
            
        # Calculate collaboration statistics
        total_collaborations = len(self.collaboration_events)
        successful_collaborations = sum(1 for event in self.collaboration_events if event.success)
        success_rate = successful_collaborations / total_collaborations if total_collaborations > 0 else 0
        
        # Calculate average collaboration duration
        durations = [event.duration for event in self.collaboration_events]
        average_duration = statistics.mean(durations) if durations else 0
        
        return {
            'total_patterns': total_patterns,
            'active_patterns': active_patterns,
            'type_distribution': dict(type_distribution),
            'status_distribution': dict(status_distribution),
            'total_collaborations': total_collaborations,
            'successful_collaborations': successful_collaborations,
            'collaboration_success_rate': success_rate,
            'average_collaboration_duration': average_duration
        }
        
    def _log_behavior_event(self, event_type: str, data: Dict[str, Any]):
        """Log a behavior event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.behavior_history.append(event)
        
    def get_behavior_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get behavior history."""
        return list(self.behavior_history[-limit:])


class PatternDetector:
    """Detects specific patterns in the network."""
    
    def __init__(self, network):
        self.network = network
        self.detection_rules: Dict[str, Dict[str, Any]] = {}
        self.detection_history: List[Dict[str, Any]] = []
        
    def add_detection_rule(self, rule_id: str, pattern_type: str,
                         conditions: Dict[str, Any], threshold: float = 0.5) -> bool:
        """Add a pattern detection rule."""
        rule = {
            'id': rule_id,
            'pattern_type': pattern_type,
            'conditions': conditions,
            'threshold': threshold,
            'active': True,
            'created_at': time.time()
        }
        
        self.detection_rules[rule_id] = rule
        
        # Log detection event
        self._log_detection_event('rule_added', {
            'rule_id': rule_id,
            'pattern_type': pattern_type,
            'conditions': conditions,
            'threshold': threshold
        })
        
        return True
        
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns using configured rules."""
        detected_patterns = []
        
        for rule_id, rule in self.detection_rules.items():
            if not rule['active']:
                continue
                
            # Apply detection rule
            pattern = self._apply_detection_rule(rule)
            if pattern:
                detected_patterns.append(pattern)
                
        return detected_patterns
        
    def _apply_detection_rule(self, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply a detection rule."""
        pattern_type = rule['pattern_type']
        conditions = rule['conditions']
        threshold = rule['threshold']
        
        # Calculate pattern score
        score = self._calculate_pattern_score(pattern_type, conditions)
        
        if score >= threshold:
            return {
                'rule_id': rule['id'],
                'pattern_type': pattern_type,
                'score': score,
                'threshold': threshold,
                'detected_at': time.time(),
                'confidence': min(score / threshold, 1.0)
            }
            
        return None
        
    def _calculate_pattern_score(self, pattern_type: str, conditions: Dict[str, Any]) -> float:
        """Calculate pattern score based on conditions."""
        score = 0.0
        
        if pattern_type == 'collaboration':
            # Check for high communication between agents
            communication_score = self._calculate_communication_score(conditions)
            score += communication_score * 0.5
            
            # Check for shared goals
            goal_score = self._calculate_goal_score(conditions)
            score += goal_score * 0.3
            
            # Check for skill complementarity
            skill_score = self._calculate_skill_score(conditions)
            score += skill_score * 0.2
            
        elif pattern_type == 'learning':
            # Check for knowledge sharing
            knowledge_score = self._calculate_knowledge_score(conditions)
            score += knowledge_score * 0.6
            
            # Check for skill propagation
            propagation_score = self._calculate_propagation_score(conditions)
            score += propagation_score * 0.4
            
        elif pattern_type == 'coordination':
            # Check for task coordination
            coordination_score = self._calculate_coordination_score(conditions)
            score += coordination_score * 0.7
            
            # Check for resource sharing
            resource_score = self._calculate_resource_score(conditions)
            score += resource_score * 0.3
            
        return min(score, 1.0)
        
    def _calculate_communication_score(self, conditions: Dict[str, Any]) -> float:
        """Calculate communication score."""
        # This would need to be implemented based on actual communication analysis
        return 0.5
        
    def _calculate_goal_score(self, conditions: Dict[str, Any]) -> float:
        """Calculate goal alignment score."""
        # This would need to be implemented based on actual goal analysis
        return 0.5
        
    def _calculate_skill_score(self, conditions: Dict[str, Any]) -> float:
        """Calculate skill complementarity score."""
        # This would need to be implemented based on actual skill analysis
        return 0.5
        
    def _calculate_knowledge_score(self, conditions: Dict[str, Any]) -> float:
        """Calculate knowledge sharing score."""
        # This would need to be implemented based on actual knowledge analysis
        return 0.5
        
    def _calculate_propagation_score(self, conditions: Dict[str, Any]) -> float:
        """Calculate propagation score."""
        # This would need to be implemented based on actual propagation analysis
        return 0.5
        
    def _calculate_coordination_score(self, conditions: Dict[str, Any]) -> float:
        """Calculate coordination score."""
        # This would need to be implemented based on actual coordination analysis
        return 0.5
        
    def _calculate_resource_score(self, conditions: Dict[str, Any]) -> float:
        """Calculate resource sharing score."""
        # This would need to be implemented based on actual resource analysis
        return 0.5
        
    def _log_detection_event(self, event_type: str, data: Dict[str, Any]):
        """Log a detection event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.detection_history.append(event)
        
    def get_detection_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get detection history."""
        return list(self.detection_history[-limit:])


class CollaborationEngine:
    """Engine for managing collaborative behaviors."""
    
    def __init__(self, network):
        self.network = network
        self.collaboration_sessions: Dict[str, Dict[str, Any]] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
    def start_collaboration_session(self, session_id: str, participating_agents: List[str],
                                  goal: str, description: str) -> bool:
        """Start a collaboration session."""
        session = {
            'id': session_id,
            'participating_agents': participating_agents,
            'goal': goal,
            'description': description,
            'started_at': time.time(),
            'status': 'active',
            'activities': [],
            'outcomes': []
        }
        
        self.collaboration_sessions[session_id] = session
        
        # Log collaboration event
        self._log_collaboration_event('session_started', {
            'session_id': session_id,
            'participating_agents': participating_agents,
            'goal': goal,
            'description': description
        })
        
        return True
        
    def add_collaboration_activity(self, session_id: str, agent_id: str,
                                 activity_type: str, description: str,
                                 data: Dict[str, Any] = None) -> bool:
        """Add an activity to a collaboration session."""
        if session_id not in self.collaboration_sessions:
            return False
            
        session = self.collaboration_sessions[session_id]
        
        activity = {
            'id': str(uuid.uuid4()),
            'agent_id': agent_id,
            'activity_type': activity_type,
            'description': description,
            'data': data or {},
            'timestamp': time.time()
        }
        
        session['activities'].append(activity)
        
        # Log collaboration event
        self._log_collaboration_event('activity_added', {
            'session_id': session_id,
            'agent_id': agent_id,
            'activity_type': activity_type,
            'description': description
        })
        
        return True
        
    def add_collaboration_outcome(self, session_id: str, outcome_type: str,
                                description: str, data: Dict[str, Any] = None) -> bool:
        """Add an outcome to a collaboration session."""
        if session_id not in self.collaboration_sessions:
            return False
            
        session = self.collaboration_sessions[session_id]
        
        outcome = {
            'id': str(uuid.uuid4()),
            'outcome_type': outcome_type,
            'description': description,
            'data': data or {},
            'timestamp': time.time()
        }
        
        session['outcomes'].append(outcome)
        
        # Log collaboration event
        self._log_collaboration_event('outcome_added', {
            'session_id': session_id,
            'outcome_type': outcome_type,
            'description': description
        })
        
        return True
        
    def end_collaboration_session(self, session_id: str, success: bool,
                                summary: str = None) -> bool:
        """End a collaboration session."""
        if session_id not in self.collaboration_sessions:
            return False
            
        session = self.collaboration_sessions[session_id]
        session['status'] = 'completed'
        session['ended_at'] = time.time()
        session['success'] = success
        session['summary'] = summary
        
        # Log collaboration event
        self._log_collaboration_event('session_ended', {
            'session_id': session_id,
            'success': success,
            'summary': summary,
            'duration': session['ended_at'] - session['started_at']
        })
        
        return True
        
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Get collaboration statistics."""
        total_sessions = len(self.collaboration_sessions)
        active_sessions = sum(1 for session in self.collaboration_sessions.values() 
                            if session['status'] == 'active')
        completed_sessions = sum(1 for session in self.collaboration_sessions.values() 
                               if session['status'] == 'completed')
        
        # Calculate success rate
        successful_sessions = sum(1 for session in self.collaboration_sessions.values() 
                               if session.get('success', False))
        success_rate = successful_sessions / completed_sessions if completed_sessions > 0 else 0
        
        # Calculate average session duration
        durations = []
        for session in self.collaboration_sessions.values():
            if 'ended_at' in session:
                duration = session['ended_at'] - session['started_at']
                durations.append(duration)
                
        average_duration = statistics.mean(durations) if durations else 0
        
        # Calculate activity statistics
        total_activities = sum(len(session['activities']) 
                              for session in self.collaboration_sessions.values())
        total_outcomes = sum(len(session['outcomes']) 
                           for session in self.collaboration_sessions.values())
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'completed_sessions': completed_sessions,
            'successful_sessions': successful_sessions,
            'success_rate': success_rate,
            'average_duration': average_duration,
            'total_activities': total_activities,
            'total_outcomes': total_outcomes
        }
        
    def _log_collaboration_event(self, event_type: str, data: Dict[str, Any]):
        """Log a collaboration event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.collaboration_history.append(event)
        
    def get_collaboration_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get collaboration history."""
        return list(self.collaboration_history[-limit:])

