"""
Network Learning System for OmniMind
Handles knowledge propagation, skill sharing, and emergent learning.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque
import json


class LearningType(Enum):
    """Types of learning in the network."""
    SKILL_ACQUISITION = auto()
    KNOWLEDGE_PROPAGATION = auto()
    PATTERN_RECOGNITION = auto()
    COLLABORATIVE_LEARNING = auto()
    EMERGENT_BEHAVIOR = auto()


class LearningStatus(Enum):
    """Status of learning processes."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    PROPAGATED = auto()


@dataclass
class LearningEvent:
    """A learning event in the network."""
    id: str
    learning_type: LearningType
    source_agent_id: str
    target_agent_ids: List[str]
    content: Dict[str, Any]
    status: LearningStatus
    created_at: float
    completed_at: Optional[float] = None
    confidence_score: float = 0.5
    propagation_path: List[str] = None
    validation_results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.propagation_path is None:
            self.propagation_path = []
        if self.validation_results is None:
            self.validation_results = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'id': self.id,
            'learning_type': self.learning_type.name,
            'source_agent_id': self.source_agent_id,
            'target_agent_ids': self.target_agent_ids,
            'content': self.content,
            'status': self.status.name,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'confidence_score': self.confidence_score,
            'propagation_path': self.propagation_path,
            'validation_results': self.validation_results
        }


@dataclass
class KnowledgePattern:
    """A knowledge pattern discovered in the network."""
    id: str
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    discovered_by: str
    discovered_at: float
    applications: List[str] = None
    validation_score: float = 0.0
    
    def __post_init__(self):
        if self.applications is None:
            self.applications = []
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            'id': self.id,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'frequency': self.frequency,
            'confidence': self.confidence,
            'discovered_by': self.discovered_by,
            'discovered_at': self.discovered_at,
            'applications': self.applications,
            'validation_score': self.validation_score
        }


class NetworkLearning:
    """Manages learning across the network."""
    
    def __init__(self, network):
        self.network = network
        self.learning_events: Dict[str, LearningEvent] = {}
        self.knowledge_patterns: Dict[str, KnowledgePattern] = {}
        self.skill_sharing: Dict[str, List[str]] = defaultdict(list)
        self.learning_history: List[Dict[str, Any]] = []
        self.learning_statistics: Dict[str, Any] = defaultdict(int)
        
    def initiate_learning(self, learning_type: LearningType, source_agent_id: str,
                        target_agent_ids: List[str], content: Dict[str, Any],
                        confidence_score: float = 0.5) -> str:
        """Initiate a learning process."""
        learning_id = str(uuid.uuid4())
        
        learning_event = LearningEvent(
            id=learning_id,
            learning_type=learning_type,
            source_agent_id=source_agent_id,
            target_agent_ids=target_agent_ids,
            content=content,
            status=LearningStatus.PENDING,
            created_at=time.time(),
            confidence_score=confidence_score
        )
        
        self.learning_events[learning_id] = learning_event
        
        # Log learning event
        self._log_learning_event('learning_initiated', {
            'learning_id': learning_id,
            'learning_type': learning_type.name,
            'source_agent_id': source_agent_id,
            'target_agent_ids': target_agent_ids,
            'confidence_score': confidence_score
        })
        
        return learning_id
        
    def propagate_knowledge(self, learning_id: str, target_agent_id: str,
                           validation_result: Dict[str, Any]) -> bool:
        """Propagate knowledge to a target agent."""
        if learning_id not in self.learning_events:
            return False
            
        learning_event = self.learning_events[learning_id]
        
        if target_agent_id not in learning_event.target_agent_ids:
            return False
            
        # Update propagation path
        learning_event.propagation_path.append(target_agent_id)
        
        # Update validation results
        learning_event.validation_results[target_agent_id] = validation_result
        
        # Check if all targets have been reached
        if len(learning_event.propagation_path) == len(learning_event.target_agent_ids):
            learning_event.status = LearningStatus.COMPLETED
            learning_event.completed_at = time.time()
            
        # Log learning event
        self._log_learning_event('knowledge_propagated', {
            'learning_id': learning_id,
            'target_agent_id': target_agent_id,
            'validation_result': validation_result
        })
        
        return True
        
    def share_skill(self, source_agent_id: str, target_agent_id: str, 
                   skill_name: str, skill_definition: Dict[str, Any]) -> bool:
        """Share a skill between agents."""
        if source_agent_id not in self.network.agents:
            return False
        if target_agent_id not in self.network.agents:
            return False
            
        # Add skill to target agent
        target_agent = self.network.agents[target_agent_id]
        if skill_name not in target_agent.skills:
            target_agent.skills.append(skill_name)
            
        # Record skill sharing
        self.skill_sharing[source_agent_id].append(skill_name)
        
        # Log learning event
        self._log_learning_event('skill_shared', {
            'source_agent_id': source_agent_id,
            'target_agent_id': target_agent_id,
            'skill_name': skill_name,
            'skill_definition': skill_definition
        })
        
        return True
        
    def discover_pattern(self, agent_id: str, pattern_type: str, 
                        description: str, frequency: int = 1,
                        confidence: float = 0.5) -> str:
        """Discover a new pattern in the network."""
        pattern_id = str(uuid.uuid4())
        
        pattern = KnowledgePattern(
            id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            frequency=frequency,
            confidence=confidence,
            discovered_by=agent_id,
            discovered_at=time.time()
        )
        
        self.knowledge_patterns[pattern_id] = pattern
        
        # Log learning event
        self._log_learning_event('pattern_discovered', {
            'pattern_id': pattern_id,
            'agent_id': agent_id,
            'pattern_type': pattern_type,
            'description': description,
            'frequency': frequency,
            'confidence': confidence
        })
        
        return pattern_id
        
    def validate_learning(self, learning_id: str, agent_id: str,
                        validation_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a learning event."""
        if learning_id not in self.learning_events:
            return {'valid': False, 'error': 'Learning event not found'}
            
        learning_event = self.learning_events[learning_id]
        
        # Perform validation
        validation_result = {
            'valid': True,
            'score': 0.0,
            'criteria_met': [],
            'criteria_failed': [],
            'recommendations': []
        }
        
        # Check confidence threshold
        if learning_event.confidence_score < validation_criteria.get('min_confidence', 0.3):
            validation_result['valid'] = False
            validation_result['criteria_failed'].append('confidence_threshold')
            validation_result['recommendations'].append('Increase confidence score')
        else:
            validation_result['criteria_met'].append('confidence_threshold')
            validation_result['score'] += 0.3
            
        # Check content quality
        content = learning_event.content
        if len(content.get('description', '')) < validation_criteria.get('min_description_length', 10):
            validation_result['valid'] = False
            validation_result['criteria_failed'].append('content_quality')
            validation_result['recommendations'].append('Improve content description')
        else:
            validation_result['criteria_met'].append('content_quality')
            validation_result['score'] += 0.3
            
        # Check source agent credibility
        source_agent = self.network.get_agent(learning_event.source_agent_id)
        if source_agent and source_agent.performance_metrics.get('overall_score', 0) < validation_criteria.get('min_source_score', 0.5):
            validation_result['valid'] = False
            validation_result['criteria_failed'].append('source_credibility')
            validation_result['recommendations'].append('Source agent needs higher performance score')
        else:
            validation_result['criteria_met'].append('source_credibility')
            validation_result['score'] += 0.4
            
        return validation_result
        
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        total_events = len(self.learning_events)
        completed_events = sum(1 for event in self.learning_events.values() 
                              if event.status == LearningStatus.COMPLETED)
        failed_events = sum(1 for event in self.learning_events.values() 
                           if event.status == LearningStatus.FAILED)
        
        # Calculate success rate
        success_rate = completed_events / total_events if total_events > 0 else 0
        
        # Calculate average confidence
        confidence_scores = [event.confidence_score for event in self.learning_events.values()]
        average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Calculate pattern discovery rate
        patterns_discovered = len(self.knowledge_patterns)
        pattern_discovery_rate = patterns_discovered / total_events if total_events > 0 else 0
        
        # Calculate skill sharing statistics
        total_skills_shared = sum(len(skills) for skills in self.skill_sharing.values())
        unique_skills_shared = len(set(skill for skills in self.skill_sharing.values() for skill in skills))
        
        return {
            'total_learning_events': total_events,
            'completed_events': completed_events,
            'failed_events': failed_events,
            'success_rate': success_rate,
            'average_confidence': average_confidence,
            'patterns_discovered': patterns_discovered,
            'pattern_discovery_rate': pattern_discovery_rate,
            'total_skills_shared': total_skills_shared,
            'unique_skills_shared': unique_skills_shared,
            'learning_statistics': dict(self.learning_statistics)
        }
        
    def get_learning_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get learning history."""
        return list(self.learning_history[-limit:])
        
    def _log_learning_event(self, event_type: str, data: Dict[str, Any]):
        """Log a learning event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.learning_history.append(event)
        
        # Update statistics
        self.learning_statistics[event_type] += 1


class KnowledgePropagation:
    """Handles knowledge propagation across the network."""
    
    def __init__(self, network):
        self.network = network
        self.propagation_rules: Dict[str, Dict[str, Any]] = {}
        self.propagation_history: List[Dict[str, Any]] = []
        
    def add_propagation_rule(self, rule_id: str, source_conditions: Dict[str, Any],
                           target_conditions: Dict[str, Any], 
                           propagation_method: str) -> bool:
        """Add a knowledge propagation rule."""
        rule = {
            'id': rule_id,
            'source_conditions': source_conditions,
            'target_conditions': target_conditions,
            'propagation_method': propagation_method,
            'created_at': time.time(),
            'active': True
        }
        
        self.propagation_rules[rule_id] = rule
        
        # Log propagation event
        self._log_propagation_event('rule_added', {
            'rule_id': rule_id,
            'source_conditions': source_conditions,
            'target_conditions': target_conditions,
            'propagation_method': propagation_method
        })
        
        return True
        
    def propagate_knowledge(self, source_agent_id: str, knowledge: Dict[str, Any],
                           target_agents: List[str] = None) -> List[str]:
        """Propagate knowledge from source to target agents."""
        if source_agent_id not in self.network.agents:
            return []
            
        # Determine target agents
        if target_agents is None:
            target_agents = list(self.network.agents.keys())
            
        propagated_agents = []
        
        for target_agent_id in target_agents:
            if target_agent_id == source_agent_id:
                continue
                
            # Check propagation rules
            if self._should_propagate(source_agent_id, target_agent_id, knowledge):
                # Propagate knowledge
                self._apply_propagation(source_agent_id, target_agent_id, knowledge)
                propagated_agents.append(target_agent_id)
                
                # Log propagation event
                self._log_propagation_event('knowledge_propagated', {
                    'source_agent_id': source_agent_id,
                    'target_agent_id': target_agent_id,
                    'knowledge': knowledge
                })
                
        return propagated_agents
        
    def _should_propagate(self, source_agent_id: str, target_agent_id: str,
                         knowledge: Dict[str, Any]) -> bool:
        """Check if knowledge should be propagated."""
        for rule in self.propagation_rules.values():
            if not rule['active']:
                continue
                
            # Check source conditions
            source_agent = self.network.get_agent(source_agent_id)
            if not self._check_conditions(source_agent, rule['source_conditions']):
                continue
                
            # Check target conditions
            target_agent = self.network.get_agent(target_agent_id)
            if not self._check_conditions(target_agent, rule['target_conditions']):
                continue
                
            return True
            
        return False
        
    def _check_conditions(self, agent: Any, conditions: Dict[str, Any]) -> bool:
        """Check if agent meets conditions."""
        if not agent:
            return False
            
        # Check persona type
        if 'persona_type' in conditions:
            if agent.persona_type != conditions['persona_type']:
                return False
                
        # Check skills
        if 'required_skills' in conditions:
            if not all(skill in agent.skills for skill in conditions['required_skills']):
                return False
                
        # Check performance metrics
        if 'min_performance' in conditions:
            if agent.performance_metrics.get('overall_score', 0) < conditions['min_performance']:
                return False
                
        return True
        
    def _apply_propagation(self, source_agent_id: str, target_agent_id: str,
                          knowledge: Dict[str, Any]):
        """Apply knowledge propagation."""
        target_agent = self.network.get_agent(target_agent_id)
        if not target_agent:
            return
            
        # Update target agent's knowledge
        if 'facts' in knowledge:
            for fact in knowledge['facts']:
                if hasattr(target_agent.memory_manager, 'add_fact'):
                    target_agent.memory_manager.add_fact(fact)
                    
        if 'rules' in knowledge:
            for rule in knowledge['rules']:
                if hasattr(target_agent.memory_manager, 'add_rule'):
                    target_agent.memory_manager.add_rule(rule)
                    
        if 'skills' in knowledge:
            for skill in knowledge['skills']:
                if skill not in target_agent.skills:
                    target_agent.skills.append(skill)
                    
    def _log_propagation_event(self, event_type: str, data: Dict[str, Any]):
        """Log a propagation event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.propagation_history.append(event)
        
    def get_propagation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get propagation history."""
        return list(self.propagation_history[-limit:])


class SkillSharing:
    """Handles skill sharing between agents."""
    
    def __init__(self, network):
        self.network = network
        self.skill_registry: Dict[str, Dict[str, Any]] = {}
        self.sharing_history: List[Dict[str, Any]] = []
        
    def register_skill(self, agent_id: str, skill_name: str, 
                      skill_definition: Dict[str, Any]) -> bool:
        """Register a skill for sharing."""
        skill_key = f"{agent_id}:{skill_name}"
        
        skill_entry = {
            'agent_id': agent_id,
            'skill_name': skill_name,
            'skill_definition': skill_definition,
            'registered_at': time.time(),
            'usage_count': 0,
            'success_rate': 1.0
        }
        
        self.skill_registry[skill_key] = skill_entry
        
        # Log sharing event
        self._log_sharing_event('skill_registered', {
            'agent_id': agent_id,
            'skill_name': skill_name,
            'skill_definition': skill_definition
        })
        
        return True
        
    def request_skill(self, requester_id: str, skill_name: str) -> Optional[Dict[str, Any]]:
        """Request a skill from the registry."""
        # Find best skill provider
        best_skill = None
        best_score = 0
        
        for skill_key, skill_entry in self.skill_registry.items():
            if skill_entry['skill_name'] == skill_name:
                # Calculate score based on success rate and usage
                score = skill_entry['success_rate'] * (1 + skill_entry['usage_count'])
                if score > best_score:
                    best_score = score
                    best_skill = skill_entry
                    
        if best_skill:
            # Update usage count
            best_skill['usage_count'] += 1
            
            # Log sharing event
            self._log_sharing_event('skill_requested', {
                'requester_id': requester_id,
                'skill_name': skill_name,
                'provider_id': best_skill['agent_id'],
                'score': best_score
            })
            
            return best_skill['skill_definition']
            
        return None
        
    def update_skill_performance(self, agent_id: str, skill_name: str, 
                               success: bool) -> bool:
        """Update skill performance."""
        skill_key = f"{agent_id}:{skill_name}"
        
        if skill_key not in self.skill_registry:
            return False
            
        skill_entry = self.skill_registry[skill_key]
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        if success:
            skill_entry['success_rate'] = (1 - alpha) * skill_entry['success_rate'] + alpha * 1.0
        else:
            skill_entry['success_rate'] = (1 - alpha) * skill_entry['success_rate'] + alpha * 0.0
            
        # Log sharing event
        self._log_sharing_event('skill_performance_updated', {
            'agent_id': agent_id,
            'skill_name': skill_name,
            'success': success,
            'new_success_rate': skill_entry['success_rate']
        })
        
        return True
        
    def get_skill_statistics(self) -> Dict[str, Any]:
        """Get skill sharing statistics."""
        total_skills = len(self.skill_registry)
        total_usage = sum(entry['usage_count'] for entry in self.skill_registry.values())
        average_success_rate = sum(entry['success_rate'] for entry in self.skill_registry.values()) / total_skills if total_skills > 0 else 0
        
        # Calculate skill popularity
        skill_popularity = defaultdict(int)
        for entry in self.skill_registry.values():
            skill_popularity[entry['skill_name']] += entry['usage_count']
            
        most_popular_skill = max(skill_popularity.items(), key=lambda x: x[1]) if skill_popularity else None
        
        return {
            'total_skills': total_skills,
            'total_usage': total_usage,
            'average_success_rate': average_success_rate,
            'skill_popularity': dict(skill_popularity),
            'most_popular_skill': most_popular_skill
        }
        
    def _log_sharing_event(self, event_type: str, data: Dict[str, Any]):
        """Log a sharing event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.sharing_history.append(event)
        
    def get_sharing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sharing history."""
        return list(self.sharing_history[-limit:])

