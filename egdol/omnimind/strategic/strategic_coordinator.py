"""
Strategic Coordinator for OmniMind
Coordinates all strategic autonomy components for autonomous decision-making.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque
import statistics


class DecisionType(Enum):
    """Types of strategic decisions."""
    GOAL_APPROVAL = auto()
    POLICY_CHANGE = auto()
    RESOURCE_ALLOCATION = auto()
    RISK_MITIGATION = auto()
    OPTIMIZATION_EXECUTION = auto()
    KNOWLEDGE_ACTION = auto()
    FORECAST_RESPONSE = auto()


@dataclass
class StrategicDecision:
    """A strategic decision made by the system."""
    id: str
    decision_type: DecisionType
    description: str
    rationale: str
    confidence: float
    impact_assessment: Dict[str, Any]
    created_at: float
    executed_at: Optional[float] = None
    status: str = "pending"
    outcome: Dict[str, Any] = None
    rollback_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.outcome is None:
            self.outcome = {}
        if self.rollback_data is None:
            self.rollback_data = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            'id': self.id,
            'decision_type': self.decision_type.name,
            'description': self.description,
            'rationale': self.rationale,
            'confidence': self.confidence,
            'impact_assessment': self.impact_assessment,
            'created_at': self.created_at,
            'executed_at': self.executed_at,
            'status': self.status,
            'outcome': self.outcome,
            'rollback_data': self.rollback_data
        }


class StrategicCoordinator:
    """Coordinates all strategic autonomy components for autonomous decision-making."""
    
    def __init__(self, network, goal_generator, scenario_simulator, policy_evolver,
                 risk_assessor, autonomous_optimizer, knowledge_lifecycle, performance_forecaster):
        self.network = network
        self.goal_generator = goal_generator
        self.scenario_simulator = scenario_simulator
        self.policy_evolver = policy_evolver
        self.risk_assessor = risk_assessor
        self.autonomous_optimizer = autonomous_optimizer
        self.knowledge_lifecycle = knowledge_lifecycle
        self.performance_forecaster = performance_forecaster
        
        self.decisions: Dict[str, StrategicDecision] = {}
        self.decision_history: List[Dict[str, Any]] = []
        self.strategic_cycles: List[Dict[str, Any]] = []
        
    def execute_strategic_cycle(self) -> Dict[str, Any]:
        """Execute a complete strategic autonomy cycle."""
        cycle_id = str(uuid.uuid4())
        cycle_start = time.time()
        
        cycle_results = {
            'cycle_id': cycle_id,
            'start_time': cycle_start,
            'goals_generated': 0,
            'simulations_run': 0,
            'policies_evolved': 0,
            'risks_assessed': 0,
            'optimizations_executed': 0,
            'knowledge_actions': 0,
            'forecasts_made': 0,
            'decisions_made': 0,
            'success_rate': 0.0
        }
        
        try:
            # 1. Generate strategic goals
            goals = self.goal_generator.generate_strategic_goals(max_goals=3)
            cycle_results['goals_generated'] = len(goals)
            
            # 2. Simulate goal scenarios
            simulations = []
            for goal in goals:
                simulation = self.scenario_simulator.simulate_goal_execution(
                    goal.id, goal.to_dict()
                )
                simulations.append(simulation)
            cycle_results['simulations_run'] = len(simulations)
            
            # 3. Evolve policies
            policies = self.policy_evolver.evolve_policies(max_evolutions=2)
            cycle_results['policies_evolved'] = len(policies)
            
            # 4. Assess risks
            risks = self.risk_assessor.assess_network_risks()
            cycle_results['risks_assessed'] = len(risks)
            
            # 5. Analyze optimization opportunities
            optimizations = self.autonomous_optimizer.analyze_optimization_opportunities()
            cycle_results['optimizations_executed'] = len(optimizations)
            
            # 6. Analyze knowledge lifecycle
            knowledge_items = self.knowledge_lifecycle.analyze_knowledge_lifecycle()
            cycle_results['knowledge_actions'] = len(knowledge_items)
            
            # 7. Forecast performance
            forecasts = self.performance_forecaster.forecast_network_performance()
            cycle_results['forecasts_made'] = len(forecasts)
            
            # 8. Make strategic decisions
            decisions = self._make_strategic_decisions(goals, simulations, policies, 
                                                    risks, optimizations, knowledge_items, forecasts)
            cycle_results['decisions_made'] = len(decisions)
            
            # 9. Execute decisions
            executed_decisions = self._execute_decisions(decisions)
            cycle_results['success_rate'] = len(executed_decisions) / len(decisions) if decisions else 0
            
            # Log strategic cycle
            cycle_results['end_time'] = time.time()
            cycle_results['duration'] = cycle_results['end_time'] - cycle_start
            self.strategic_cycles.append(cycle_results)
            
            # Store cycle results
            self.strategic_cycles.append(cycle_results)
            
            self._log_strategic_event('strategic_cycle_completed', cycle_results)
            
        except Exception as e:
            cycle_results['error'] = str(e)
            cycle_results['end_time'] = time.time()
            cycle_results['duration'] = cycle_results['end_time'] - cycle_start
            self._log_strategic_event('strategic_cycle_failed', cycle_results)
            
        return cycle_results
        
    def _make_strategic_decisions(self, goals: List, simulations: List, policies: List,
                                risks: List, optimizations: List, knowledge_items: List,
                                forecasts: List) -> List[StrategicDecision]:
        """Make strategic decisions based on analysis results."""
        decisions = []
        
        # Decision 1: Approve high-confidence goals
        for goal, simulation in zip(goals, simulations):
            if simulation.success_probability > 0.7 and simulation.confidence_score > 0.6:
                decision = StrategicDecision(
                    id=str(uuid.uuid4()),
                    decision_type=DecisionType.GOAL_APPROVAL,
                    description=f"Approve goal: {goal.title}",
                    rationale=f"High success probability ({simulation.success_probability:.2f}) and confidence ({simulation.confidence_score:.2f})",
                    confidence=simulation.confidence_score,
                    impact_assessment=simulation.expected_outcome,
                    created_at=time.time()
                )
                decisions.append(decision)
                
        # Decision 2: Activate effective policies
        for policy in policies:
            if policy.effectiveness_score > 0.7:
                decision = StrategicDecision(
                    id=str(uuid.uuid4()),
                    decision_type=DecisionType.POLICY_CHANGE,
                    description=f"Activate policy: {policy.name}",
                    rationale=f"High effectiveness score ({policy.effectiveness_score:.2f})",
                    confidence=policy.effectiveness_score,
                    impact_assessment={'policy_type': policy.policy_type.name},
                    created_at=time.time()
                )
                decisions.append(decision)
                
        # Decision 3: Mitigate high-risk issues
        for risk in risks:
            if risk.level.value >= 3:  # HIGH or CRITICAL
                decision = StrategicDecision(
                    id=str(uuid.uuid4()),
                    decision_type=DecisionType.RISK_MITIGATION,
                    description=f"Mitigate risk: {risk.description}",
                    rationale=f"High risk level ({risk.level.name}) with probability {risk.probability:.2f}",
                    confidence=risk.probability,
                    impact_assessment={'risk_type': risk.risk_type.name, 'impact': risk.impact},
                    created_at=time.time()
                )
                decisions.append(decision)
                
        # Decision 4: Execute high-impact optimizations
        for optimization in optimizations:
            if optimization.improvement_score > 0.6:
                decision = StrategicDecision(
                    id=str(uuid.uuid4()),
                    decision_type=DecisionType.OPTIMIZATION_EXECUTION,
                    description=f"Execute optimization: {optimization.description}",
                    rationale=f"High improvement potential ({optimization.improvement_score:.2f})",
                    confidence=optimization.improvement_score,
                    impact_assessment={'strategy': optimization.strategy.name},
                    created_at=time.time()
                )
                decisions.append(decision)
                
        # Decision 5: Apply knowledge lifecycle actions
        for item in knowledge_items:
            if item.state.value >= 3:  # REDUNDANT or OUTDATED
                decision = StrategicDecision(
                    id=str(uuid.uuid4()),
                    decision_type=DecisionType.KNOWLEDGE_ACTION,
                    description=f"Apply knowledge action: {item.content[:50]}...",
                    rationale=f"Knowledge state: {item.state.name}",
                    confidence=0.8,
                    impact_assessment={'knowledge_type': item.knowledge_type},
                    created_at=time.time()
                )
                decisions.append(decision)
                
        # Decision 6: Respond to forecasts
        for forecast in forecasts:
            if forecast.confidence > 0.7 and forecast.trend_type.value <= 2:  # IMPROVING or DECLINING
                decision = StrategicDecision(
                    id=str(uuid.uuid4()),
                    decision_type=DecisionType.FORECAST_RESPONSE,
                    description=f"Respond to forecast: {forecast.metric_name}",
                    rationale=f"High confidence forecast ({forecast.confidence:.2f}) with trend {forecast.trend_type.name}",
                    confidence=forecast.confidence,
                    impact_assessment={'metric': forecast.metric_name, 'trend': forecast.trend_type.name},
                    created_at=time.time()
                )
                decisions.append(decision)
                
        return decisions
        
    def _execute_decisions(self, decisions: List[StrategicDecision]) -> List[StrategicDecision]:
        """Execute strategic decisions."""
        executed_decisions = []
        
        for decision in decisions:
            try:
                success = self._execute_decision(decision)
                if success:
                    decision.status = "executed"
                    decision.executed_at = time.time()
                    executed_decisions.append(decision)
                    
                    # Log decision execution
                    self._log_strategic_event('decision_executed', {
                        'decision_id': decision.id,
                        'decision_type': decision.decision_type.name,
                        'description': decision.description
                    })
                else:
                    decision.status = "failed"
                    
            except Exception as e:
                decision.status = "failed"
                decision.outcome = {'error': str(e)}
                
        return executed_decisions
        
    def _execute_decision(self, decision: StrategicDecision) -> bool:
        """Execute a specific strategic decision."""
        if decision.decision_type == DecisionType.GOAL_APPROVAL:
            return self._execute_goal_approval(decision)
        elif decision.decision_type == DecisionType.POLICY_CHANGE:
            return self._execute_policy_change(decision)
        elif decision.decision_type == DecisionType.RISK_MITIGATION:
            return self._execute_risk_mitigation(decision)
        elif decision.decision_type == DecisionType.OPTIMIZATION_EXECUTION:
            return self._execute_optimization(decision)
        elif decision.decision_type == DecisionType.KNOWLEDGE_ACTION:
            return self._execute_knowledge_action(decision)
        elif decision.decision_type == DecisionType.FORECAST_RESPONSE:
            return self._execute_forecast_response(decision)
        else:
            return False
            
    def _execute_goal_approval(self, decision: StrategicDecision) -> bool:
        """Execute goal approval decision."""
        # Extract goal ID from decision description
        # This would need to be implemented based on the actual goal structure
        return True
        
    def _execute_policy_change(self, decision: StrategicDecision) -> bool:
        """Execute policy change decision."""
        # Extract policy ID from decision description
        # This would need to be implemented based on the actual policy structure
        return True
        
    def _execute_risk_mitigation(self, decision: StrategicDecision) -> bool:
        """Execute risk mitigation decision."""
        # Extract risk ID from decision description
        # This would need to be implemented based on the actual risk structure
        return True
        
    def _execute_optimization(self, decision: StrategicDecision) -> bool:
        """Execute optimization decision."""
        # Extract optimization ID from decision description
        # This would need to be implemented based on the actual optimization structure
        return True
        
    def _execute_knowledge_action(self, decision: StrategicDecision) -> bool:
        """Execute knowledge action decision."""
        # Extract knowledge item ID from decision description
        # This would need to be implemented based on the actual knowledge structure
        return True
        
    def _execute_forecast_response(self, decision: StrategicDecision) -> bool:
        """Execute forecast response decision."""
        # Extract forecast ID from decision description
        # This would need to be implemented based on the actual forecast structure
        return True
        
    def get_strategic_statistics(self) -> Dict[str, Any]:
        """Get strategic coordination statistics."""
        total_decisions = len(self.decisions)
        executed_decisions = sum(1 for d in self.decisions.values() if d.status == "executed")
        failed_decisions = sum(1 for d in self.decisions.values() if d.status == "failed")
        
        # Calculate decision type distribution
        type_distribution = defaultdict(int)
        for decision in self.decisions.values():
            type_distribution[decision.decision_type.name] += 1
            
        # Calculate average confidence
        confidence_scores = [d.confidence for d in self.decisions.values()]
        average_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        # Calculate strategic cycle statistics
        total_cycles = len(self.strategic_cycles)
        successful_cycles = sum(1 for c in self.strategic_cycles if c.get('success_rate', 0) > 0.5)
        
        return {
            'total_decisions': total_decisions,
            'executed_decisions': executed_decisions,
            'failed_decisions': failed_decisions,
            'execution_rate': executed_decisions / total_decisions if total_decisions > 0 else 0,
            'type_distribution': dict(type_distribution),
            'average_confidence': average_confidence,
            'total_cycles': total_cycles,
            'successful_cycles': successful_cycles,
            'cycle_success_rate': successful_cycles / total_cycles if total_cycles > 0 else 0
        }
        
    def _log_strategic_event(self, event_type: str, data: Dict[str, Any]):
        """Log a strategic event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.decision_history.append(event)
        
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get decision history."""
        return list(self.decision_history[-limit:])
