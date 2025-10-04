"""
OmniMind Civilizational Feedback System
Integrates conversation insights into multi-universe simulations and meta-rule evolution.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

from .transcendence.multi_universe_orchestration import MultiUniverseOrchestrator
from .transcendence.civilization_architect import CivilizationArchitect
from .conversational.reflexive_audit import ReflexiveAuditModule, AuditResult
from .conversational.meta_learning_engine import MetaLearningEngine, LearningInsight
from .conversational.personality_evolution import PersonalityEvolutionEngine
from .conversational.reflection_mode_plus import ReflectionModePlus
from .conversational.context_intent_resolver import ContextIntentResolver
from .conversational.reasoning_normalizer import ReasoningNormalizer
from .conversational.personality_fallbacks import PersonalityFallbackReasoner


@dataclass
class CivilizationalInsight:
    """Insight extracted from conversation-simulation integration."""
    insight_type: str
    pattern: str
    confidence: float
    source_conversation: str
    simulation_outcome: Dict[str, Any]
    meta_rule_candidate: Optional[str] = None
    validation_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetaRuleCandidate:
    """Candidate meta-rule for validation and application."""
    rule_name: str
    rule_pattern: str
    confidence: float
    source_insights: List[CivilizationalInsight]
    validation_tests: List[Dict[str, Any]]
    simulation_validation: Dict[str, Any]
    personality_applicability: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CivilizationalFeedbackResult:
    """Result of civilizational feedback processing."""
    conversation_insights: List[CivilizationalInsight]
    meta_rule_candidates: List[MetaRuleCandidate]
    simulation_updates: Dict[str, Any]
    personality_evolution: Dict[str, Any]
    audit_results: AuditResult
    success: bool
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class CivilizationalFeedbackEngine:
    """
    Core engine for integrating conversation insights into civilizational simulations
    and generating meta-rule evolution through deterministic feedback loops.
    """
    
    def __init__(self, 
                 multi_universe_orchestrator: MultiUniverseOrchestrator,
                 civilization_architect: CivilizationArchitect,
                 audit_module: ReflexiveAuditModule,
                 meta_learning: MetaLearningEngine,
                 personality_evolution: PersonalityEvolutionEngine,
                 reflection_plus: ReflectionModePlus,
                 context_resolver: ContextIntentResolver,
                 reasoning_normalizer: ReasoningNormalizer,
                 fallback_reasoner: PersonalityFallbackReasoner):
        """Initialize the civilizational feedback engine."""
        self.multi_universe_orchestrator = multi_universe_orchestrator
        self.civilization_architect = civilization_architect
        self.audit_module = audit_module
        self.meta_learning = meta_learning
        self.personality_evolution = personality_evolution
        self.reflection_plus = reflection_plus
        self.context_resolver = context_resolver
        self.reasoning_normalizer = reasoning_normalizer
        self.fallback_reasoner = fallback_reasoner
        
        # Feedback loop state
        self.conversation_history: List[Dict[str, Any]] = []
        self.civilizational_insights: List[CivilizationalInsight] = []
        self.meta_rule_candidates: List[MetaRuleCandidate] = []
        self.applied_meta_rules: List[MetaRuleCandidate] = []
        
        # Performance tracking
        self.feedback_cycles: int = 0
        self.successful_integrations: int = 0
        self.meta_rule_validation_rate: float = 0.0
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def process_conversation_with_civilizational_feedback(self, 
                                                        user_input: str,
                                                        personality: str = "Strategos",
                                                        context: Optional[Dict[str, Any]] = None) -> CivilizationalFeedbackResult:
        """
        Process conversation input through full civilizational feedback loop.
        
        Args:
            user_input: User input to process
            personality: Target personality for reasoning
            context: Optional context for reasoning
            
        Returns:
            Complete feedback result with insights, meta-rules, and evolution
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Context Resolution and Normalization
            normalized_input = self._normalize_input(user_input, personality, context)
            
            # Step 2: Civilizational Pattern Integration
            simulation_insights = self._integrate_with_simulations(normalized_input)
            
            # Step 3: Reasoning Engine with Fallbacks
            reasoning_result = self._execute_reasoning_with_fallbacks(normalized_input, simulation_insights)
            
            # Step 4: Meta-Learning and Personality Evolution
            learning_insights = self._extract_learning_insights(reasoning_result, simulation_insights)
            evolution_result = self._apply_personality_evolution(learning_insights)
            
            # Step 5: Meta-Rule Discovery and Validation
            meta_rule_candidates = self._discover_meta_rules(learning_insights, simulation_insights)
            validated_rules = self._validate_meta_rules(meta_rule_candidates)
            
            # Step 6: Reflection and Retry if Needed
            if reasoning_result.get('audit_score', 0) < 0.6:
                reflection_result = self._apply_reflection_mode_plus(
                    user_input, personality, reasoning_result, learning_insights
                )
                if reflection_result.success:
                    reasoning_result = reflection_result
            
            # Step 7: Update Civilizational Simulations
            simulation_updates = self._update_simulations_with_insights(learning_insights, validated_rules)
            
            # Step 8: Audit and Performance Tracking
            audit_result = self._audit_feedback_cycle(reasoning_result, learning_insights, validated_rules)
            
            # Create comprehensive result
            result = CivilizationalFeedbackResult(
                conversation_insights=learning_insights,
                meta_rule_candidates=validated_rules,
                simulation_updates=simulation_updates,
                personality_evolution=evolution_result,
                audit_results=audit_result,
                success=True,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Update internal state
            self._update_feedback_state(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in civilizational feedback processing: {e}")
            return CivilizationalFeedbackResult(
                conversation_insights=[],
                meta_rule_candidates=[],
                simulation_updates={},
                personality_evolution={},
                audit_results=AuditResult(
                    turn_id="error",
                    personality=personality,
                    metrics={},
                    gaps_identified=[str(e)],
                    improvement_suggestions=["Error handling improvement needed"],
                    confidence_score=0.0
                ),
                success=False,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _normalize_input(self, user_input: str, personality: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize input through context resolution and reasoning normalization."""
        try:
            # Resolve context and intent
            context_result = self.context_resolver.resolve_intent_and_context(
                user_input, personality
            )
            
            # Normalize reasoning input
            normalized = self.reasoning_normalizer.normalize_input(
                user_input, personality
            )
            
            return {
                'success': True,
                'user_input': user_input,
                'personality': personality,
                'context': context_result,
                'normalized': normalized,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Context normalization error: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_input': user_input,
                'personality': personality
            }
    
    def _integrate_with_simulations(self, normalized_input: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate conversation input with multi-universe simulations."""
        try:
            # Extract patterns from conversation
            conversation_patterns = self._extract_conversation_patterns(normalized_input)
            
            # Feed patterns into simulations
            context_dict = normalized_input['context']
            if hasattr(context_dict, 'to_dict'):
                context_dict = context_dict.to_dict()
            elif hasattr(context_dict, '__dict__'):
                context_dict = context_dict.__dict__
            
            simulation_input = {
                'conversation_patterns': conversation_patterns,
                'personality_context': normalized_input['personality'],
                'reasoning_context': context_dict
            }
            
            # Run civilization simulations
            if self.multi_universe_orchestrator is not None:
                simulation_results = self.multi_universe_orchestrator.simulate_civilizational_patterns(
                    simulation_input
                )
            else:
                # Mock simulation results for testing
                simulation_results = {
                    'social_structures': {'hierarchy': 'advanced'},
                    'economic_patterns': {'market': 'efficient'},
                    'conflict_resolution': {'mediation': 'successful'},
                    'resource_management': {'allocation': 'optimal'}
                }
            
            # Extract meta-patterns from simulation outcomes
            meta_patterns = self._extract_meta_patterns(simulation_results)
            
            return {
                'conversation_patterns': conversation_patterns,
                'simulation_results': simulation_results,
                'meta_patterns': meta_patterns,
                'integration_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulation integration: {e}")
            return {
                'conversation_patterns': {},
                'simulation_results': {},
                'meta_patterns': {},
                'integration_success': False,
                'error': str(e)
            }
    
    def _execute_reasoning_with_fallbacks(self, normalized_input: Dict[str, Any], 
                                        simulation_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reasoning with deterministic fallback mechanisms."""
        try:
            # Attempt primary reasoning
            reasoning_result = self._attempt_primary_reasoning(normalized_input, simulation_insights)
            
            if reasoning_result.get('success', False):
                return reasoning_result
            
            # Trigger personality-specific fallback
            fallback_result = self._trigger_personality_fallback(normalized_input, simulation_insights)
            
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"Error in reasoning execution: {e}")
            return {
                'success': False,
                'response': f"Reasoning error: {str(e)}",
                'fallback_used': True,
                'error': str(e)
            }
    
    def _attempt_primary_reasoning(self, normalized_input: Dict[str, Any], 
                                 simulation_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt primary reasoning without fallbacks."""
        # This would integrate with the conversational reasoning engine
        # For now, return a placeholder structure
        return {
            'success': True,
            'response': f"Primary reasoning for {normalized_input['personality']}",
            'reasoning_trace': ['context_analysis', 'pattern_matching', 'response_generation'],
            'fallback_used': False,
            'confidence': 0.8
        }
    
    def _trigger_personality_fallback(self, normalized_input: Dict[str, Any], 
                                    simulation_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger personality-specific fallback reasoning."""
        personality = normalized_input['personality']
        user_input = normalized_input['user_input']
        
        # Use personality fallback reasoner
        fallback_response = self.fallback_reasoner.get_fallback_response(
            user_input, personality
        )
        
        return {
            'success': True,
            'response': fallback_response,
            'reasoning_trace': ['fallback_triggered', 'personality_specific_reasoning'],
            'fallback_used': True,
            'confidence': 0.6
        }
    
    def _extract_learning_insights(self, reasoning_result: Dict[str, Any], 
                                 simulation_insights: Dict[str, Any]) -> List[CivilizationalInsight]:
        """Extract learning insights from reasoning and simulation results."""
        insights = []
        
        # Extract conversation patterns
        conversation_pattern = self._analyze_conversation_pattern(reasoning_result)
        if conversation_pattern:
            insight = CivilizationalInsight(
                insight_type='conversation_pattern',
                pattern=conversation_pattern,
                confidence=0.7,
                source_conversation=reasoning_result.get('response', ''),
                simulation_outcome=simulation_insights.get('simulation_results', {}),
                meta_rule_candidate=self._generate_meta_rule_candidate(conversation_pattern)
            )
            insights.append(insight)
        
        # Extract simulation patterns
        simulation_pattern = self._analyze_simulation_pattern(simulation_insights)
        if simulation_pattern:
            insight = CivilizationalInsight(
                insight_type='simulation_pattern',
                pattern=simulation_pattern,
                confidence=0.8,
                source_conversation=reasoning_result.get('response', ''),
                simulation_outcome=simulation_insights.get('simulation_results', {}),
                meta_rule_candidate=self._generate_meta_rule_candidate(simulation_pattern)
            )
            insights.append(insight)
        
        return insights
    
    def _apply_personality_evolution(self, learning_insights: List[CivilizationalInsight]) -> Dict[str, Any]:
        """Apply personality evolution based on learning insights."""
        evolution_results = {}
        
        for insight in learning_insights:
            # Convert civilizational insights to learning insights format
            learning_insight = LearningInsight(
                insight_type=insight.insight_type,
                personality="Strategos",  # Default, should be determined from context
                pattern=insight.pattern,
                confidence=insight.confidence,
                applicable_to=['reasoning_engine'],
                metadata={'civilizational_insight': True}
            )
            
            # Apply evolution
            evolution_result = self.personality_evolution.evolve_personality(
                personality="Strategos",
                learning_insights=[learning_insight]
            )
            
            evolution_results[insight.insight_type] = evolution_result
        
        return evolution_results
    
    def _discover_meta_rules(self, learning_insights: List[CivilizationalInsight], 
                           simulation_insights: Dict[str, Any]) -> List[MetaRuleCandidate]:
        """Discover candidate meta-rules from insights and simulations."""
        candidates = []
        
        for insight in learning_insights:
            if insight.meta_rule_candidate:
                candidate = MetaRuleCandidate(
                    rule_name=f"rule_{len(candidates)}",
                    rule_pattern=insight.meta_rule_candidate,
                    confidence=insight.confidence,
                    source_insights=[insight],
                    validation_tests=[],
                    simulation_validation={},
                    personality_applicability=['Strategos', 'Archivist', 'Lawmaker', 'Oracle']
                )
                candidates.append(candidate)
        
        return candidates
    
    def _validate_meta_rules(self, candidates: List[MetaRuleCandidate]) -> List[MetaRuleCandidate]:
        """Validate meta-rule candidates through simulation testing."""
        validated_rules = []
        
        for candidate in candidates:
            # Run validation tests
            validation_score = self._run_meta_rule_validation(candidate)
            candidate.validation_score = validation_score
            
            if validation_score >= 0.7:  # Threshold for acceptance
                validated_rules.append(candidate)
        
        return validated_rules
    
    def _run_meta_rule_validation(self, candidate: MetaRuleCandidate) -> float:
        """Run validation tests for a meta-rule candidate."""
        # This would implement actual validation logic
        # For now, return a placeholder score
        return 0.8
    
    def _apply_reflection_mode_plus(self, user_input: str, personality: str, 
                                  reasoning_result: Dict[str, Any], 
                                  learning_insights: List[CivilizationalInsight]) -> Any:
        """Apply enhanced reflection mode with retry mechanisms."""
        return self.reflection_plus.reflect_and_retry_plus(
            user_input=user_input,
            personality=personality,
            original_error="Low audit score",
            context={"learning_insights": learning_insights}
        )
    
    def _update_simulations_with_insights(self, learning_insights: List[CivilizationalInsight], 
                                        validated_rules: List[MetaRuleCandidate]) -> Dict[str, Any]:
        """Update civilizational simulations with new insights and meta-rules."""
        updates = {
            'insights_applied': len(learning_insights),
            'meta_rules_applied': len(validated_rules),
            'simulation_updates': {},
            'success': True
        }
        
        # Apply insights to simulations
        for insight in learning_insights:
            self._apply_insight_to_simulation(insight)
        
        # Apply meta-rules to simulations
        for rule in validated_rules:
            self._apply_meta_rule_to_simulation(rule)
        
        return updates
    
    def _audit_feedback_cycle(self, reasoning_result: Dict[str, Any], 
                            learning_insights: List[CivilizationalInsight], 
                            validated_rules: List[MetaRuleCandidate]) -> AuditResult:
        """Audit the complete feedback cycle for performance and quality."""
        # Create a conversation turn for auditing
        from .conversational.conversation_state import ConversationTurn
        
        turn = ConversationTurn(
            turn_id=f"feedback_cycle_{self.feedback_cycles}",
            timestamp=datetime.now(),
            user_input=reasoning_result.get('user_input', ''),
            system_response=reasoning_result.get('response', ''),
            intent="civilizational_feedback",
            personality_used=reasoning_result.get('personality', 'Strategos'),
            reasoning_trace=reasoning_result.get('reasoning_trace', []),
            confidence_score=reasoning_result.get('confidence', 0.0)
        )
        
        return self.audit_module.audit_conversation_turn(turn)
    
    def _update_feedback_state(self, result: CivilizationalFeedbackResult):
        """Update internal feedback state with results."""
        self.feedback_cycles += 1
        
        if result.success:
            self.successful_integrations += 1
        
        # Update meta-rule validation rate
        if result.meta_rule_candidates:
            validated_count = len([r for r in result.meta_rule_candidates if r.validation_score >= 0.7])
            self.meta_rule_validation_rate = validated_count / len(result.meta_rule_candidates)
        
        # Store results
        self.civilizational_insights.extend(result.conversation_insights)
        self.meta_rule_candidates.extend(result.meta_rule_candidates)
        
        # Apply validated meta-rules
        for rule in result.meta_rule_candidates:
            if rule.validation_score >= 0.7:
                self.applied_meta_rules.append(rule)
    
    def _extract_conversation_patterns(self, normalized_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patterns from conversation input."""
        context = normalized_input['context']
        if hasattr(context, 'to_dict'):
            context = context.to_dict()
        elif hasattr(context, '__dict__'):
            context = context.__dict__
        
        return {
            'intent_type': context.get('intent', 'unknown') if isinstance(context, dict) else 'unknown',
            'reasoning_type': context.get('reasoning_type', 'general') if isinstance(context, dict) else 'general',
            'personality': normalized_input['personality'],
            'complexity': len(normalized_input['user_input'].split()),
            'timestamp': normalized_input['timestamp']
        }
    
    def _extract_meta_patterns(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meta-patterns from simulation results."""
        return {
            'social_structures': simulation_results.get('social_structures', {}),
            'economic_patterns': simulation_results.get('economic_patterns', {}),
            'conflict_resolution': simulation_results.get('conflict_resolution', {}),
            'resource_management': simulation_results.get('resource_management', {})
        }
    
    def _analyze_conversation_pattern(self, reasoning_result: Dict[str, Any]) -> Optional[str]:
        """Analyze conversation pattern for insights."""
        response = reasoning_result.get('response', '')
        if len(response) > 50:
            return f"complex_response_pattern_{len(response)}"
        return None
    
    def _analyze_simulation_pattern(self, simulation_insights: Dict[str, Any]) -> Optional[str]:
        """Analyze simulation pattern for insights."""
        meta_patterns = simulation_insights.get('meta_patterns', {})
        if meta_patterns:
            return f"simulation_pattern_{len(meta_patterns)}"
        return None
    
    def _generate_meta_rule_candidate(self, pattern: str) -> Optional[str]:
        """Generate meta-rule candidate from pattern."""
        if 'complex_response' in pattern:
            return f"complex_response_rule: {pattern}"
        elif 'simulation_pattern' in pattern:
            return f"simulation_rule: {pattern}"
        return None
    
    def _apply_insight_to_simulation(self, insight: CivilizationalInsight):
        """Apply insight to civilizational simulation."""
        # This would integrate with the simulation system
        pass
    
    def _apply_meta_rule_to_simulation(self, rule: MetaRuleCandidate):
        """Apply meta-rule to civilizational simulation."""
        # This would integrate with the simulation system
        pass
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of civilizational feedback processing."""
        return {
            'feedback_cycles': self.feedback_cycles,
            'successful_integrations': self.successful_integrations,
            'success_rate': self.successful_integrations / max(1, self.feedback_cycles),
            'meta_rule_validation_rate': self.meta_rule_validation_rate,
            'total_insights': len(self.civilizational_insights),
            'total_meta_rules': len(self.applied_meta_rules),
            'recent_insights': self.civilizational_insights[-10:] if self.civilizational_insights else [],
            'recent_meta_rules': self.applied_meta_rules[-5:] if self.applied_meta_rules else []
        }
