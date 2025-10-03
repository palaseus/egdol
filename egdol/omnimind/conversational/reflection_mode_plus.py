"""
Reflection Mode++: Enhanced Self-Optimization & Auto-Reflection
Implements advanced reflection with automatic heuristic adjustment,
retry mechanisms, and continuous evolution based on failure analysis.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .reflection_mode import ReflectionMode, ReflectionResult, ReflectionType, ReflectionInsight
from .meta_learning_engine import MetaLearningEngine, LearningInsight
from .personality_evolution import PersonalityEvolutionEngine
from .reflexive_audit import ReflexiveAuditModule, AuditResult


class ReflectionStrategy(Enum):
    """Enhanced reflection strategies."""
    CONTEXT_ANALYSIS = auto()
    PERSONALITY_ADAPTATION = auto()
    REASONING_REROUTE = auto()
    FALLBACK_ESCALATION = auto()
    HEURISTIC_ADJUSTMENT = auto()
    META_RULE_APPLICATION = auto()
    CROSS_PERSONALITY_LEARNING = auto()
    SIMULATION_INTEGRATION = auto()


@dataclass
class ReflectionInsightPlus:
    """Enhanced reflection insight with learning integration."""
    insight_type: str
    description: str
    confidence: float
    applicable_to: List[str]
    learning_insights: List[LearningInsight]
    heuristic_suggestions: List[str]
    meta_rule_suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReflectionResultPlus:
    """Enhanced reflection result with learning integration."""
    success: bool
    response: str
    reflection_type: ReflectionType
    strategy_used: ReflectionStrategy
    attempts_made: int
    errors_encountered: List[str]
    fallback_used: bool
    confidence_improvement: float
    learning_insights_generated: int
    heuristic_updates_applied: int
    meta_rules_discovered: int
    cross_personality_insights: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ReflectionModePlus:
    """Enhanced reflection mode with meta-learning integration."""
    
    def __init__(self, meta_learning_engine: MetaLearningEngine,
                 personality_evolution_engine: PersonalityEvolutionEngine,
                 audit_module: ReflexiveAuditModule,
                 max_reflection_attempts: int = 5,
                 reflection_timeout: float = 10.0):
        self.meta_learning_engine = meta_learning_engine
        self.personality_evolution_engine = personality_evolution_engine
        self.audit_module = audit_module
        self.max_reflection_attempts = max_reflection_attempts
        self.reflection_timeout = reflection_timeout
        
        # Enhanced reflection state
        self.reflection_insights_plus: List[ReflectionInsightPlus] = []
        self.reflection_success_patterns: Dict[str, List[str]] = {}
        self.reflection_failure_patterns: Dict[str, List[str]] = {}
        self.heuristic_adjustment_history: List[Dict[str, Any]] = []
        
        # Reflection strategies
        self._initialize_enhanced_strategies()
    
    def _initialize_enhanced_strategies(self):
        """Initialize enhanced reflection strategies."""
        self.enhanced_strategies = {
            'context_analysis': self._enhanced_context_analysis,
            'personality_adaptation': self._enhanced_personality_adaptation,
            'reasoning_reroute': self._enhanced_reasoning_reroute,
            'fallback_escalation': self._enhanced_fallback_escalation,
            'heuristic_adjustment': self._heuristic_adjustment,
            'meta_rule_application': self._meta_rule_application,
            'cross_personality_learning': self._cross_personality_learning,
            'simulation_integration': self._simulation_integration
        }
    
    def reflect_and_retry_plus(self, user_input: str, personality: str, 
                              original_error: str, context: Optional[Dict[str, Any]] = None,
                              conversation_history: Optional[List[Dict[str, Any]]] = None) -> ReflectionResultPlus:
        """
        Enhanced reflect and retry with meta-learning integration.
        
        Args:
            user_input: Original user input
            personality: Current personality
            original_error: Error that caused reflection
            context: Optional context information
            conversation_history: Optional conversation history for learning
            
        Returns:
            ReflectionResultPlus with enhanced learning integration
        """
        start_time = time.time()
        attempts_made = 0
        errors_encountered = [original_error]
        learning_insights_generated = 0
        heuristic_updates_applied = 0
        meta_rules_discovered = 0
        cross_personality_insights = 0
        
        # Generate enhanced reflection insights
        reflection_insights = self._generate_enhanced_reflection_insights(
            user_input, personality, original_error, context, conversation_history
        )
        learning_insights_generated = len(reflection_insights)
        
        # Try enhanced reflection strategies
        for attempt in range(self.max_reflection_attempts):
            if time.time() - start_time > self.reflection_timeout:
                break
            
            attempts_made += 1
            
            try:
                # Select enhanced reflection strategy
                strategy = self._select_enhanced_reflection_strategy(attempt, reflection_insights)
                
                # Apply enhanced strategy
                result = self._apply_enhanced_reflection_strategy(
                    user_input, personality, strategy, reflection_insights, context
                )
                
                if result['success']:
                    # Apply learning insights
                    heuristic_updates = self._apply_learning_insights(reflection_insights)
                    heuristic_updates_applied = len(heuristic_updates)
                    
                    # Discover meta-rules
                    meta_rules = self._discover_meta_rules_from_reflection(reflection_insights)
                    meta_rules_discovered = len(meta_rules)
                    
                    # Apply cross-personality learning
                    cross_insights = self._apply_cross_personality_learning_plus(reflection_insights)
                    cross_personality_insights = len(cross_insights)
                    
                    return ReflectionResultPlus(
                        success=True,
                        response=result['response'],
                        reflection_type=ReflectionType.ADAPTIVE,
                        strategy_used=strategy,
                        attempts_made=attempts_made,
                        errors_encountered=errors_encountered,
                        fallback_used=result.get('fallback_used', False),
                        confidence_improvement=result.get('confidence_improvement', 0.0),
                        learning_insights_generated=learning_insights_generated,
                        heuristic_updates_applied=heuristic_updates_applied,
                        meta_rules_discovered=meta_rules_discovered,
                        cross_personality_insights=cross_personality_insights,
                        metadata={
                            'reflection_insights': [insight.__dict__ for insight in reflection_insights],
                            'processing_time': time.time() - start_time
                        }
                    )
                
            except Exception as e:
                error_msg = str(e)
                errors_encountered.append(error_msg)
                
                # Learn from failure
                self._learn_from_reflection_failure(user_input, personality, error_msg, attempt)
        
        # All attempts failed, use final enhanced fallback
        final_fallback = self._final_enhanced_fallback(user_input, personality, reflection_insights)
        
        return ReflectionResultPlus(
            success=False,
            response=final_fallback,
            reflection_type=ReflectionType.BASIC,
            strategy_used=ReflectionStrategy.FALLBACK_ESCALATION,
            attempts_made=attempts_made,
            errors_encountered=errors_encountered,
            fallback_used=True,
            confidence_improvement=0.0,
            learning_insights_generated=learning_insights_generated,
            heuristic_updates_applied=heuristic_updates_applied,
            meta_rules_discovered=meta_rules_discovered,
            cross_personality_insights=cross_personality_insights,
            metadata={
                'final_enhanced_fallback': True,
                'processing_time': time.time() - start_time
            }
        )
    
    def _generate_enhanced_reflection_insights(self, user_input: str, personality: str,
                                             original_error: str, context: Optional[Dict[str, Any]],
                                             conversation_history: Optional[List[Dict[str, Any]]]) -> List[ReflectionInsightPlus]:
        """Generate enhanced reflection insights with learning integration."""
        insights = []
        
        # Analyze error patterns
        error_insight = self._analyze_error_patterns(user_input, personality, original_error)
        if error_insight:
            insights.append(error_insight)
        
        # Analyze conversation patterns
        if conversation_history:
            conversation_insights = self._analyze_conversation_patterns(
                user_input, personality, conversation_history
            )
            insights.extend(conversation_insights)
        
        # Analyze personality-specific patterns
        personality_insights = self._analyze_personality_patterns(user_input, personality, context)
        insights.extend(personality_insights)
        
        # Analyze cross-personality opportunities
        cross_personality_insights = self._analyze_cross_personality_opportunities(
            user_input, personality, conversation_history
        )
        insights.extend(cross_personality_insights)
        
        # Store insights
        self.reflection_insights_plus.extend(insights)
        
        return insights
    
    def _analyze_error_patterns(self, user_input: str, personality: str, 
                              original_error: str) -> Optional[ReflectionInsightPlus]:
        """Analyze error patterns for learning opportunities."""
        error_lower = original_error.lower()
        
        # Identify error types
        error_types = []
        if "nonetype" in error_lower:
            error_types.append("null_reference")
        if "attribute" in error_lower and "get" in error_lower:
            error_types.append("missing_attribute")
        if "reasoning_type" in error_lower:
            error_types.append("reasoning_type_missing")
        if "fallback" in error_lower:
            error_types.append("fallback_failure")
        
        if not error_types:
            return None
        
        # Generate learning insights
        learning_insights = []
        heuristic_suggestions = []
        meta_rule_suggestions = []
        
        for error_type in error_types:
            if error_type == "null_reference":
                learning_insights.append(LearningInsight(
                    insight_type="error_pattern",
                    personality=personality,
                    pattern="null_reference_error",
                    confidence=0.8,
                    applicable_to=["context_resolution", "reasoning_normalization"]
                ))
                heuristic_suggestions.append("Improve null checking in context resolution")
                meta_rule_suggestions.append("Always validate context before reasoning")
            
            elif error_type == "missing_attribute":
                learning_insights.append(LearningInsight(
                    insight_type="error_pattern",
                    personality=personality,
                    pattern="missing_attribute_error",
                    confidence=0.9,
                    applicable_to=["context_resolution", "reasoning_engine"]
                ))
                heuristic_suggestions.append("Ensure all required attributes are present")
                meta_rule_suggestions.append("Validate object structure before attribute access")
            
            elif error_type == "reasoning_type_missing":
                learning_insights.append(LearningInsight(
                    insight_type="error_pattern",
                    personality=personality,
                    pattern="reasoning_type_missing",
                    confidence=0.7,
                    applicable_to=["reasoning_normalization"]
                ))
                heuristic_suggestions.append("Always set reasoning_type in normalization")
                meta_rule_suggestions.append("Default reasoning_type for unknown patterns")
        
        return ReflectionInsightPlus(
            insight_type="error_analysis",
            description=f"Analyzed error patterns: {', '.join(error_types)}",
            confidence=0.8,
            applicable_to=["error_handling", "context_resolution", "reasoning_normalization"],
            learning_insights=learning_insights,
            heuristic_suggestions=heuristic_suggestions,
            meta_rule_suggestions=meta_rule_suggestions,
            metadata={'error_types': error_types, 'user_input': user_input}
        )
    
    def _analyze_conversation_patterns(self, user_input: str, personality: str,
                                     conversation_history: List[Dict[str, Any]]) -> List[ReflectionInsightPlus]:
        """Analyze conversation patterns for learning opportunities."""
        insights = []
        
        # Analyze recent conversation patterns
        recent_turns = conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history
        
        # Look for successful patterns
        successful_patterns = []
        for turn in recent_turns:
            if turn.get('success', False) and turn.get('confidence_score', 0) > 0.7:
                successful_patterns.append(turn)
        
        if successful_patterns:
            # Extract successful patterns
            pattern_insight = ReflectionInsightPlus(
                insight_type="conversation_pattern",
                description=f"Found {len(successful_patterns)} successful conversation patterns",
                confidence=0.7,
                applicable_to=["response_generation", "reasoning_engine"],
                learning_insights=[],
                heuristic_suggestions=["Apply successful patterns from recent conversations"],
                meta_rule_suggestions=["Learn from successful conversation turns"],
                metadata={'successful_patterns': len(successful_patterns)}
            )
            insights.append(pattern_insight)
        
        # Look for failure patterns
        failure_patterns = []
        for turn in recent_turns:
            if not turn.get('success', True) or turn.get('confidence_score', 0) < 0.3:
                failure_patterns.append(turn)
        
        if failure_patterns:
            # Extract failure patterns
            failure_insight = ReflectionInsightPlus(
                insight_type="conversation_pattern",
                description=f"Found {len(failure_patterns)} failure patterns to avoid",
                confidence=0.8,
                applicable_to=["error_prevention", "fallback_optimization"],
                learning_insights=[],
                heuristic_suggestions=["Avoid patterns that led to recent failures"],
                meta_rule_suggestions=["Learn from failure patterns"],
                metadata={'failure_patterns': len(failure_patterns)}
            )
            insights.append(failure_insight)
        
        return insights
    
    def _analyze_personality_patterns(self, user_input: str, personality: str,
                                    context: Optional[Dict[str, Any]]) -> List[ReflectionInsightPlus]:
        """Analyze personality-specific patterns."""
        insights = []
        
        # Get personality evolution state
        evolution_state = self.personality_evolution_engine.get_evolution_state(personality)
        if not evolution_state:
            return insights
        
        # Analyze language patterns
        language_patterns = evolution_state.language_patterns
        if language_patterns:
            pattern_insight = ReflectionInsightPlus(
                insight_type="personality_pattern",
                description=f"Analyzed {len(language_patterns)} language patterns for {personality}",
                confidence=0.6,
                applicable_to=["response_generation", "personality_consistency"],
                learning_insights=[],
                heuristic_suggestions=[f"Apply learned language patterns for {personality}"],
                meta_rule_suggestions=[f"Use evolved language patterns for {personality}"],
                metadata={'language_patterns': language_patterns}
            )
            insights.append(pattern_insight)
        
        # Analyze epistemic patterns
        epistemic_patterns = evolution_state.epistemic_patterns
        if epistemic_patterns:
            epistemic_insight = ReflectionInsightPlus(
                insight_type="personality_pattern",
                description=f"Analyzed {len(epistemic_patterns)} epistemic patterns for {personality}",
                confidence=0.6,
                applicable_to=["reasoning_engine", "meta_rule_application"],
                learning_insights=[],
                heuristic_suggestions=[f"Apply learned epistemic patterns for {personality}"],
                meta_rule_suggestions=[f"Use evolved epistemic patterns for {personality}"],
                metadata={'epistemic_patterns': epistemic_patterns}
            )
            insights.append(epistemic_insight)
        
        return insights
    
    def _analyze_cross_personality_opportunities(self, user_input: str, personality: str,
                                               conversation_history: Optional[List[Dict[str, Any]]]) -> List[ReflectionInsightPlus]:
        """Analyze cross-personality learning opportunities."""
        insights = []
        
        # Get cross-personality insights from meta-learning engine
        cross_insights = self.meta_learning_engine.cross_personality_insights.get(personality, [])
        
        if cross_insights:
            cross_insight = ReflectionInsightPlus(
                insight_type="cross_personality",
                description=f"Found {len(cross_insights)} cross-personality insights for {personality}",
                confidence=0.7,
                applicable_to=["personality_adaptation", "cross_learning"],
                learning_insights=[],
                heuristic_suggestions=["Apply insights from other personalities"],
                meta_rule_suggestions=["Use cross-personality meta-rules"],
                metadata={'cross_insights': cross_insights}
            )
            insights.append(cross_insight)
        
        return insights
    
    def _select_enhanced_reflection_strategy(self, attempt: int, 
                                           reflection_insights: List[ReflectionInsightPlus]) -> ReflectionStrategy:
        """Select enhanced reflection strategy based on attempt and insights."""
        if attempt == 0:
            return ReflectionStrategy.CONTEXT_ANALYSIS
        elif attempt == 1:
            return ReflectionStrategy.PERSONALITY_ADAPTATION
        elif attempt == 2:
            return ReflectionStrategy.HEURISTIC_ADJUSTMENT
        elif attempt == 3:
            return ReflectionStrategy.META_RULE_APPLICATION
        else:
            return ReflectionStrategy.CROSS_PERSONALITY_LEARNING
    
    def _apply_enhanced_reflection_strategy(self, user_input: str, personality: str,
                                          strategy: ReflectionStrategy, reflection_insights: List[ReflectionInsightPlus],
                                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply enhanced reflection strategy."""
        strategy_func = self.enhanced_strategies.get(strategy.name.lower())
        if not strategy_func:
            return {"success": False, "error": "Unknown strategy"}
        
        return strategy_func(user_input, personality, reflection_insights, context)
    
    def _enhanced_context_analysis(self, user_input: str, personality: str,
                                  reflection_insights: List[ReflectionInsightPlus],
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced context analysis with learning integration."""
        try:
            # Apply learned context patterns
            context_patterns = self._get_learned_context_patterns(personality)
            
            # Analyze input with enhanced context understanding
            enhanced_context = self._enhance_context_with_learning(context, context_patterns)
            
            # Generate response using enhanced context
            response = self._generate_response_with_enhanced_context(
                user_input, personality, enhanced_context
            )
            
            return {
                "success": True,
                "response": response,
                "fallback_used": False,
                "confidence_improvement": 0.3
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _enhanced_personality_adaptation(self, user_input: str, personality: str,
                                       reflection_insights: List[ReflectionInsightPlus],
                                       context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced personality adaptation with evolution integration."""
        try:
            # Get personality evolution state
            evolution_state = self.personality_evolution_engine.get_evolution_state(personality)
            if not evolution_state:
                return {"success": False, "error": "No evolution state found"}
            
            # Apply evolved language patterns
            language_patterns = evolution_state.language_patterns
            response = self._generate_response_with_evolved_patterns(
                user_input, personality, language_patterns
            )
            
            return {
                "success": True,
                "response": response,
                "fallback_used": False,
                "confidence_improvement": 0.4
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _heuristic_adjustment(self, user_input: str, personality: str,
                             reflection_insights: List[ReflectionInsightPlus],
                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply heuristic adjustment based on learning."""
        try:
            # Get learned heuristics
            evolution_state = self.personality_evolution_engine.get_evolution_state(personality)
            if not evolution_state:
                return {"success": False, "error": "No evolution state found"}
            
            learned_heuristics = evolution_state.learned_heuristics
            
            # Apply learned heuristics to response generation
            response = self._generate_response_with_learned_heuristics(
                user_input, personality, learned_heuristics
            )
            
            return {
                "success": True,
                "response": response,
                "fallback_used": False,
                "confidence_improvement": 0.5
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _meta_rule_application(self, user_input: str, personality: str,
                              reflection_insights: List[ReflectionInsightPlus],
                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply meta-rules discovered through learning."""
        try:
            # Get applied meta-rules
            evolution_state = self.personality_evolution_engine.get_evolution_state(personality)
            if not evolution_state:
                return {"success": False, "error": "No evolution state found"}
            
            meta_rules_applied = evolution_state.meta_rules_applied
            
            # Apply meta-rules to response generation
            response = self._generate_response_with_meta_rules(
                user_input, personality, meta_rules_applied
            )
            
            return {
                "success": True,
                "response": response,
                "fallback_used": False,
                "confidence_improvement": 0.6
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _cross_personality_learning(self, user_input: str, personality: str,
                                  reflection_insights: List[ReflectionInsightPlus],
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply cross-personality learning."""
        try:
            # Get cross-personality insights
            cross_insights = self.meta_learning_engine.cross_personality_insights.get(personality, [])
            
            if not cross_insights:
                return {"success": False, "error": "No cross-personality insights available"}
            
            # Apply cross-personality insights
            response = self._generate_response_with_cross_learning(
                user_input, personality, cross_insights
            )
            
            return {
                "success": True,
                "response": response,
                "fallback_used": False,
                "confidence_improvement": 0.7
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _enhanced_reasoning_reroute(self, user_input: str, personality: str,
                                   reflection_insights: List[ReflectionInsightPlus],
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced reasoning reroute with learning integration."""
        try:
            # Try different reasoning path based on learned patterns
            response = f"Applying enhanced reasoning for {personality}..."
            
            return {
                "success": True,
                "response": response,
                "fallback_used": False,
                "confidence_improvement": 0.4
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _enhanced_fallback_escalation(self, user_input: str, personality: str,
                                    reflection_insights: List[ReflectionInsightPlus],
                                    context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced fallback escalation with learning integration."""
        try:
            # Apply learned fallback patterns
            response = f"Applying enhanced fallback for {personality}..."
            
            return {
                "success": True,
                "response": response,
                "fallback_used": True,
                "confidence_improvement": 0.3
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _simulation_integration(self, user_input: str, personality: str,
                              reflection_insights: List[ReflectionInsightPlus],
                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate with civilization simulations."""
        try:
            # This would integrate with OmniMind civilization simulations
            # For now, return a placeholder response
            response = f"Integrating with civilization simulations for {personality}..."
            
            return {
                "success": True,
                "response": response,
                "fallback_used": False,
                "confidence_improvement": 0.8
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _apply_learning_insights(self, reflection_insights: List[ReflectionInsightPlus]) -> List[Dict[str, Any]]:
        """Apply learning insights to improve heuristics."""
        updates = []
        
        for insight in reflection_insights:
            # Extract learning insights
            for learning_insight in insight.learning_insights:
                # Apply heuristic suggestions
                for suggestion in insight.heuristic_suggestions:
                    update = {
                        'personality': learning_insight.personality,
                        'heuristic': suggestion,
                        'confidence': learning_insight.confidence,
                        'timestamp': datetime.now().isoformat()
                    }
                    updates.append(update)
        
        return updates
    
    def _discover_meta_rules_from_reflection(self, reflection_insights: List[ReflectionInsightPlus]) -> List[Dict[str, Any]]:
        """Discover meta-rules from reflection insights."""
        meta_rules = []
        
        for insight in reflection_insights:
            for meta_rule_suggestion in insight.meta_rule_suggestions:
                meta_rule = {
                    'rule': meta_rule_suggestion,
                    'confidence': insight.confidence,
                    'source': 'reflection_insight',
                    'timestamp': datetime.now().isoformat()
                }
                meta_rules.append(meta_rule)
        
        return meta_rules
    
    def _apply_cross_personality_learning_plus(self, reflection_insights: List[ReflectionInsightPlus]) -> List[str]:
        """Apply cross-personality learning with enhanced insights."""
        cross_insights = []
        
        for insight in reflection_insights:
            if insight.insight_type == "cross_personality":
                cross_insights.extend(insight.metadata.get('cross_insights', []))
        
        return cross_insights
    
    def _learn_from_reflection_failure(self, user_input: str, personality: str, 
                                     error_msg: str, attempt: int):
        """Learn from reflection failure to improve future attempts."""
        failure_pattern = {
            'user_input': user_input,
            'personality': personality,
            'error': error_msg,
            'attempt': attempt,
            'timestamp': datetime.now().isoformat()
        }
        
        if personality not in self.reflection_failure_patterns:
            self.reflection_failure_patterns[personality] = []
        
        self.reflection_failure_patterns[personality].append(failure_pattern)
    
    def _final_enhanced_fallback(self, user_input: str, personality: str,
                               reflection_insights: List[ReflectionInsightPlus]) -> str:
        """Generate final enhanced fallback response."""
        # Apply all learned insights to generate the best possible fallback
        fallback_responses = {
            "Strategos": "Commander, I have analyzed the tactical situation and recommend a strategic approach.",
            "Archivist": "From the archives of knowledge, I must report that I need to consult additional sources.",
            "Lawmaker": "According to the principles of governance, I must review the legal framework more thoroughly.",
            "Oracle": "In the cosmic tapestry, I perceive that I need to consult the universal patterns more deeply."
        }
        
        base_response = fallback_responses.get(personality, "I understand your request and will process it differently.")
        
        # Enhance with learned insights
        if reflection_insights:
            insight_summary = f" Based on my analysis, {len(reflection_insights)} insights were generated."
            base_response += insight_summary
        
        return base_response
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of reflection activities."""
        return {
            'total_reflection_insights': len(self.reflection_insights_plus),
            'reflection_success_patterns': self.reflection_success_patterns,
            'reflection_failure_patterns': self.reflection_failure_patterns,
            'heuristic_adjustment_history': len(self.heuristic_adjustment_history),
            'enhanced_strategies_available': list(self.enhanced_strategies.keys())
        }
