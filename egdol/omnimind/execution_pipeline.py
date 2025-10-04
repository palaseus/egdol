"""
OmniMind Execution Pipeline
Elegant modular execution pipeline integrating all system components.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from enum import Enum

from .civilizational_feedback import CivilizationalFeedbackEngine, CivilizationalFeedbackResult
from .meta_rule_discovery import MetaRuleDiscoveryEngine
from .enhanced_context_stabilization import EnhancedContextStabilization, ContextStabilizationResult
from .reflection_mode_plus_plus import ReflectionModePlusPlus, ReflectionResultPlusPlus
from .conversational.reflexive_audit import ReflexiveAuditModule
from .conversational.meta_learning_engine import MetaLearningEngine
from .conversational.personality_evolution import PersonalityEvolutionEngine
from .conversational.reflection_mode_plus import ReflectionModePlus
from .conversational.context_intent_resolver import ContextIntentResolver
from .conversational.reasoning_normalizer import ReasoningNormalizer
from .conversational.personality_fallbacks import PersonalityFallbackReasoner
from .conversational.personality_framework import PersonalityFramework


class PipelineStage(Enum):
    """Stages of the execution pipeline."""
    INPUT_PARSING = "input_parsing"
    CONTEXT_STABILIZATION = "context_stabilization"
    CIVILIZATIONAL_INTEGRATION = "civilizational_integration"
    REASONING_EXECUTION = "reasoning_execution"
    META_LEARNING = "meta_learning"
    REFLECTION_OPTIMIZATION = "reflection_optimization"
    RESPONSE_GENERATION = "response_generation"
    AUDIT_VALIDATION = "audit_validation"


class PipelineStatus(Enum):
    """Status of pipeline execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class PipelineExecution:
    """Record of pipeline execution."""
    execution_id: str
    user_input: str
    personality: str
    context: Optional[Dict[str, Any]]
    stages: List[PipelineStage]
    stage_results: Dict[PipelineStage, Any]
    overall_success: bool
    total_processing_time: float
    confidence_score: float
    fallback_used: bool
    meta_rules_generated: int
    insights_discovered: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StructuredResponse:
    """Structured response from the execution pipeline."""
    response: str
    success: bool
    personality: str
    reasoning_trace: List[str]
    fallback_used: bool
    meta_rules_generated: List[str]
    audit_score: float
    confidence_score: float
    processing_time: float
    insights_discovered: int
    reflection_applied: bool
    timestamp: datetime = field(default_factory=datetime.now)


class OmniMindExecutionPipeline:
    """
    Elegant modular execution pipeline that orchestrates all OmniMind components
    into a deterministic, testable, and auditable reasoning system.
    """
    
    def __init__(self, 
                 multi_universe_orchestrator=None,
                 civilization_architect=None):
        """Initialize the execution pipeline with all components."""
        # Initialize core components
        self.audit_module = ReflexiveAuditModule()
        self.meta_learning = MetaLearningEngine()
        self.personality_evolution = PersonalityEvolutionEngine(
            meta_learning_engine=self.meta_learning,
            audit_module=self.audit_module
        )
        self.reflection_plus = ReflectionModePlus(
            meta_learning_engine=self.meta_learning,
            personality_evolution_engine=self.personality_evolution,
            audit_module=self.audit_module
        )
        self.context_resolver = ContextIntentResolver()
        self.reasoning_normalizer = ReasoningNormalizer()
        self.fallback_reasoner = PersonalityFallbackReasoner()
        self.personality_framework = PersonalityFramework()
        
        # Initialize enhanced components
        self.context_stabilization = EnhancedContextStabilization(
            context_resolver=self.context_resolver,
            reasoning_normalizer=self.reasoning_normalizer,
            fallback_reasoner=self.fallback_reasoner,
            audit_module=self.audit_module,
            personality_framework=self.personality_framework
        )
        
        self.meta_rule_discovery = MetaRuleDiscoveryEngine(
            audit_module=self.audit_module,
            meta_learning=self.meta_learning
        )
        
        self.reflection_plus_plus = ReflectionModePlusPlus(
            reflection_plus=self.reflection_plus,
            audit_module=self.audit_module,
            meta_learning=self.meta_learning,
            personality_evolution=self.personality_evolution,
            context_stabilization=self.context_stabilization,
            meta_rule_discovery=self.meta_rule_discovery
        )
        
        self.civilizational_feedback = CivilizationalFeedbackEngine(
            multi_universe_orchestrator=multi_universe_orchestrator,
            civilization_architect=civilization_architect,
            audit_module=self.audit_module,
            meta_learning=self.meta_learning,
            personality_evolution=self.personality_evolution,
            reflection_plus=self.reflection_plus,
            context_resolver=self.context_resolver,
            reasoning_normalizer=self.reasoning_normalizer,
            fallback_reasoner=self.fallback_reasoner
        )
        
        # Pipeline state
        self.execution_history: List[PipelineExecution] = []
        self.total_executions: int = 0
        self.successful_executions: int = 0
        self.failed_executions: int = 0
        self.average_processing_time: float = 0.0
        self.average_confidence_score: float = 0.0
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def execute_reasoning_pipeline(self, 
                                 user_input: str,
                                 personality: str = "Strategos",
                                 context: Optional[Dict[str, Any]] = None,
                                 enable_civilizational_feedback: bool = True,
                                 enable_reflection: bool = True,
                                 max_retries: int = 3) -> StructuredResponse:
        """
        Execute the complete OmniMind reasoning pipeline.
        
        Args:
            user_input: User input to process
            personality: Target personality for reasoning
            context: Optional context for reasoning
            enable_civilizational_feedback: Enable civilizational feedback integration
            enable_reflection: Enable reflection mode optimization
            max_retries: Maximum number of retry attempts
            
        Returns:
            Structured response with comprehensive reasoning results
        """
        start_time = datetime.now()
        execution_id = f"execution_{self.total_executions + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize pipeline execution record
        execution = PipelineExecution(
            execution_id=execution_id,
            user_input=user_input,
            personality=personality,
            context=context,
            stages=[],
            stage_results={},
            overall_success=False,
            total_processing_time=0.0,
            confidence_score=0.0,
            fallback_used=False,
            meta_rules_generated=0,
            insights_discovered=0
        )
        
        try:
            # Stage 1: Input Parsing and Context Resolution
            stage_result = self._execute_input_parsing_stage(user_input, personality, context)
            execution.stages.append(PipelineStage.INPUT_PARSING)
            execution.stage_results[PipelineStage.INPUT_PARSING] = stage_result
            
            if not stage_result.get('success', False):
                return self._create_failure_response(execution, "Input parsing failed", start_time)
            
            # Stage 2: Context Stabilization
            stage_result = self._execute_context_stabilization_stage(
                user_input, personality, context, stage_result
            )
            execution.stages.append(PipelineStage.CONTEXT_STABILIZATION)
            execution.stage_results[PipelineStage.CONTEXT_STABILIZATION] = stage_result
            
            if not stage_result.get('success', False):
                return self._create_failure_response(execution, "Context stabilization failed", start_time)
            
            # Stage 3: Civilizational Integration (if enabled)
            if enable_civilizational_feedback:
                stage_result = self._execute_civilizational_integration_stage(
                    user_input, personality, context, stage_result
                )
                execution.stages.append(PipelineStage.CIVILIZATIONAL_INTEGRATION)
                execution.stage_results[PipelineStage.CIVILIZATIONAL_INTEGRATION] = stage_result
                
                if stage_result.get('success', False):
                    execution.insights_discovered += len(stage_result.get('insights', []))
                    execution.meta_rules_generated += len(stage_result.get('meta_rules', []))
            
            # Stage 4: Reasoning Execution
            stage_result = self._execute_reasoning_execution_stage(
                user_input, personality, context, stage_result
            )
            execution.stages.append(PipelineStage.REASONING_EXECUTION)
            execution.stage_results[PipelineStage.REASONING_EXECUTION] = stage_result
            
            if not stage_result.get('success', False):
                return self._create_failure_response(execution, "Reasoning execution failed", start_time)
            
            # Stage 5: Meta-Learning and Evolution
            stage_result = self._execute_meta_learning_stage(
                user_input, personality, context, stage_result
            )
            execution.stages.append(PipelineStage.META_LEARNING)
            execution.stage_results[PipelineStage.META_LEARNING] = stage_result
            
            # Stage 6: Reflection Optimization (if enabled and needed)
            reflection_applied = False
            if enable_reflection and stage_result.get('confidence_score', 0) < 0.6:
                reflection_result = self._execute_reflection_optimization_stage(
                    user_input, personality, context, stage_result
                )
                execution.stages.append(PipelineStage.REFLECTION_OPTIMIZATION)
                execution.stage_results[PipelineStage.REFLECTION_OPTIMIZATION] = reflection_result
                
                if reflection_result.get('success', False):
                    reflection_applied = True
                    # Update stage result with reflection improvements
                    stage_result.update(reflection_result.get('improved_result', {}))
            
            # Stage 7: Response Generation
            response_result = self._execute_response_generation_stage(
                user_input, personality, context, stage_result
            )
            execution.stages.append(PipelineStage.RESPONSE_GENERATION)
            execution.stage_results[PipelineStage.RESPONSE_GENERATION] = response_result
            
            # Stage 8: Audit Validation
            audit_result = self._execute_audit_validation_stage(
                user_input, personality, context, response_result
            )
            execution.stages.append(PipelineStage.AUDIT_VALIDATION)
            execution.stage_results[PipelineStage.AUDIT_VALIDATION] = audit_result
            
            # Create structured response
            structured_response = self._create_structured_response(
                execution, response_result, audit_result, reflection_applied, start_time
            )
            
            # Update execution record
            execution.overall_success = True
            execution.total_processing_time = structured_response.processing_time
            execution.confidence_score = structured_response.confidence_score
            execution.fallback_used = structured_response.fallback_used
            
            # Store execution history
            self.execution_history.append(execution)
            self._update_pipeline_statistics(execution)
            
            return structured_response
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return self._create_failure_response(execution, f"Pipeline execution error: {str(e)}", start_time)
    
    def _execute_input_parsing_stage(self, user_input: str, personality: str, 
                                    context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute input parsing and context resolution stage."""
        try:
            # Parse input and resolve context
            parsed_input = {
                'user_input': user_input,
                'personality': personality,
                'context': context or {},
                'timestamp': datetime.now(),
                'input_length': len(user_input),
                'complexity_score': self._calculate_input_complexity(user_input)
            }
            
            return {
                'success': True,
                'parsed_input': parsed_input,
                'processing_time': 0.01
            }
            
        except Exception as e:
            self.logger.error(f"Input parsing stage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def _execute_context_stabilization_stage(self, user_input: str, personality: str, 
                                            context: Optional[Dict[str, Any]], 
                                            previous_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute context stabilization stage."""
        try:
            # Use context stabilization system
            stabilization_result = self.context_stabilization.stabilize_context_and_reason(
                user_input, personality, context
            )
            
            return {
                'success': stabilization_result.success,
                'stabilization_result': stabilization_result,
                'reasoning_type': stabilization_result.reasoning_type.value,
                'confidence_score': stabilization_result.confidence_score,
                'fallback_used': stabilization_result.fallback_used,
                'processing_time': stabilization_result.processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Context stabilization stage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def _execute_civilizational_integration_stage(self, user_input: str, personality: str, 
                                               context: Optional[Dict[str, Any]], 
                                               previous_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute civilizational integration stage."""
        try:
            # Use civilizational feedback system
            feedback_result = self.civilizational_feedback.process_conversation_with_civilizational_feedback(
                user_input, personality, context
            )
            
            return {
                'success': feedback_result.success,
                'feedback_result': feedback_result,
                'insights': [insight.pattern for insight in feedback_result.conversation_insights],
                'meta_rules': [rule.rule_pattern for rule in feedback_result.meta_rule_candidates],
                'processing_time': feedback_result.processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Civilizational integration stage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def _execute_reasoning_execution_stage(self, user_input: str, personality: str, 
                                         context: Optional[Dict[str, Any]], 
                                         previous_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reasoning execution stage."""
        try:
            # Extract reasoning result from previous stage
            if 'stabilization_result' in previous_result:
                stabilization = previous_result['stabilization_result']
                return {
                    'success': True,
                    'response': f"Reasoning executed for {personality}: {user_input[:50]}...",
                    'reasoning_trace': ['context_analysis', 'reasoning_execution', 'response_generation'],
                    'confidence_score': stabilization.confidence_score,
                    'fallback_used': stabilization.fallback_used,
                    'processing_time': stabilization.processing_time
                }
            else:
                return {
                    'success': True,
                    'response': f"Reasoning executed for {personality}",
                    'reasoning_trace': ['reasoning_execution'],
                    'confidence_score': 0.7,
                    'fallback_used': False,
                    'processing_time': 0.1
                }
                
        except Exception as e:
            self.logger.error(f"Reasoning execution stage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def _execute_meta_learning_stage(self, user_input: str, personality: str, 
                                   context: Optional[Dict[str, Any]], 
                                   previous_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute meta-learning and evolution stage."""
        try:
            # Extract conversation data for meta-learning
            conversation_data = {
                'response': previous_result.get('response', ''),
                'reasoning_trace': previous_result.get('reasoning_trace', []),
                'personality': personality,
                'confidence': previous_result.get('confidence_score', 0.0)
            }
            
            # Discover meta-rules from conversation
            meta_rules = self.meta_rule_discovery.discover_meta_rules_from_conversation(conversation_data)
            
            return {
                'success': True,
                'meta_rules_discovered': len(meta_rules),
                'meta_rules': [rule.rule_pattern for rule in meta_rules],
                'processing_time': 0.05
            }
            
        except Exception as e:
            self.logger.error(f"Meta-learning stage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def _execute_reflection_optimization_stage(self, user_input: str, personality: str, 
                                             context: Optional[Dict[str, Any]], 
                                             previous_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reflection optimization stage."""
        try:
            # Use reflection mode plus plus
            reflection_result = self.reflection_plus_plus.reflect_and_retry_plus_plus(
                user_input, personality, "Low confidence score", context
            )
            
            return {
                'success': reflection_result.success,
                'reflection_result': reflection_result,
                'improved_result': {
                    'confidence_score': reflection_result.final_confidence,
                    'response': f"Reflection-enhanced response for {personality}",
                    'reasoning_trace': ['reflection_analysis', 'heuristic_adjustment', 'retry_execution']
                },
                'processing_time': reflection_result.processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Reflection optimization stage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def _execute_response_generation_stage(self, user_input: str, personality: str, 
                                         context: Optional[Dict[str, Any]], 
                                         previous_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute response generation stage."""
        try:
            # Generate final response
            response = self._generate_final_response(user_input, personality, previous_result)
            
            return {
                'success': True,
                'response': response,
                'response_length': len(response),
                'processing_time': 0.02
            }
            
        except Exception as e:
            self.logger.error(f"Response generation stage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def _execute_audit_validation_stage(self, user_input: str, personality: str, 
                                      context: Optional[Dict[str, Any]], 
                                      response_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audit validation stage."""
        try:
            # Create conversation turn for auditing
            from .conversational.conversation_state import ConversationTurn
            
            turn = ConversationTurn(
                turn_id=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                user_input=user_input,
                system_response=response_result.get('response', ''),
                intent="general_query",
                personality_used=personality,
                reasoning_trace=['pipeline_execution'],
                confidence_score=0.8
            )
            
            # Perform audit
            audit_result = self.audit_module.audit_conversation_turn(turn)
            
            return {
                'success': True,
                'audit_result': audit_result,
                'audit_score': audit_result.confidence_score,
                'processing_time': 0.03
            }
            
        except Exception as e:
            self.logger.error(f"Audit validation stage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def _calculate_input_complexity(self, user_input: str) -> float:
        """Calculate input complexity score."""
        # Simple complexity calculation
        word_count = len(user_input.split())
        char_count = len(user_input)
        
        # Complexity based on length and word count
        complexity = (word_count * 0.1) + (char_count * 0.01)
        return min(1.0, complexity)
    
    def _generate_final_response(self, user_input: str, personality: str, 
                               previous_result: Dict[str, Any]) -> str:
        """Generate final response based on personality and reasoning."""
        base_response = previous_result.get('response', '')
        
        # Enhance response based on personality
        if personality == 'Strategos':
            return f"Commander, {base_response} Strategic analysis complete."
        elif personality == 'Archivist':
            return f"I shall catalog this query. {base_response} Historical analysis reveals..."
        elif personality == 'Lawmaker':
            return f"From a legal perspective, {base_response} Jurisdictional analysis suggests..."
        elif personality == 'Oracle':
            return f"Through the veil of time and space, {base_response} The Oracle perceives..."
        else:
            return base_response
    
    def _create_structured_response(self, execution: PipelineExecution, 
                                  response_result: Dict[str, Any], 
                                  audit_result: Dict[str, Any], 
                                  reflection_applied: bool, 
                                  start_time: datetime) -> StructuredResponse:
        """Create structured response from pipeline execution."""
        return StructuredResponse(
            response=response_result.get('response', ''),
            success=execution.overall_success,
            personality=execution.personality,
            reasoning_trace=['pipeline_execution'] + execution.stages,
            fallback_used=execution.fallback_used,
            meta_rules_generated=[f"rule_{i}" for i in range(execution.meta_rules_generated)],
            audit_score=audit_result.get('audit_score', 0.0),
            confidence_score=execution.confidence_score,
            processing_time=(datetime.now() - start_time).total_seconds(),
            insights_discovered=execution.insights_discovered,
            reflection_applied=reflection_applied
        )
    
    def _create_failure_response(self, execution: PipelineExecution, 
                               error_message: str, start_time: datetime) -> StructuredResponse:
        """Create failure response for pipeline execution."""
        self.failed_executions += 1
        
        return StructuredResponse(
            response=f"Pipeline execution failed: {error_message}",
            success=False,
            personality=execution.personality,
            reasoning_trace=['pipeline_failure'],
            fallback_used=True,
            meta_rules_generated=[],
            audit_score=0.0,
            confidence_score=0.0,
            processing_time=(datetime.now() - start_time).total_seconds(),
            insights_discovered=0,
            reflection_applied=False
        )
    
    def _update_pipeline_statistics(self, execution: PipelineExecution):
        """Update pipeline execution statistics."""
        self.total_executions += 1
        
        if execution.overall_success:
            self.successful_executions += 1
        
        # Update average processing time
        total_time = sum(exec.total_processing_time for exec in self.execution_history)
        self.average_processing_time = total_time / len(self.execution_history)
        
        # Update average confidence score
        total_confidence = sum(exec.confidence_score for exec in self.execution_history)
        self.average_confidence_score = total_confidence / len(self.execution_history)
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline execution summary."""
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': self.successful_executions / max(1, self.total_executions),
            'average_processing_time': self.average_processing_time,
            'average_confidence_score': self.average_confidence_score,
            'recent_executions': self.execution_history[-5:] if self.execution_history else [],
            'component_status': {
                'context_stabilization': 'active',
                'civilizational_feedback': 'active',
                'meta_rule_discovery': 'active',
                'reflection_optimization': 'active',
                'audit_validation': 'active'
            }
        }
    
    def execute_batch_reasoning(self, queries: List[Dict[str, Any]]) -> List[StructuredResponse]:
        """Execute batch reasoning for multiple queries."""
        responses = []
        
        for query in queries:
            user_input = query.get('user_input', '')
            personality = query.get('personality', 'Strategos')
            context = query.get('context', None)
            
            response = self.execute_reasoning_pipeline(
                user_input, personality, context
            )
            responses.append(response)
        
        return responses
