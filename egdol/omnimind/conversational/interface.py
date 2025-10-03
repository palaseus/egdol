"""
Main Conversational Interface
Orchestrates the complete conversational personality layer.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from ..core import OmniMind
from .conversation_state import ConversationState, ConversationContext, ConversationPhase, ContextType
from .intent_parser import IntentParser, Intent, IntentType
from .personality_framework import PersonalityFramework, Personality
from .reasoning_engine import ConversationalReasoningEngine
from .response_generator import ResponseGenerator
from .context_intent_resolver import ContextIntentResolver, ReasoningContext
from .reasoning_normalizer import ReasoningNormalizer, NormalizedReasoningInput
from .personality_fallbacks import PersonalityFallbackReasoner, FallbackResponse
from .reflection_mode import ReflectionMode, ReflectionResult, ReflectionType


class ConversationalInterface:
    """Main conversational interface that orchestrates all components."""
    
    def __init__(self, omnimind_core: OmniMind, data_dir: str = "conversational_data"):
        self.omnimind_core = omnimind_core
        self.data_dir = data_dir
        
        # Initialize components
        self.intent_parser = IntentParser()
        self.personality_framework = PersonalityFramework()
        self.reasoning_engine = ConversationalReasoningEngine(omnimind_core)
        self.response_generator = ResponseGenerator()
        
        # Initialize stabilization layer components
        self.context_resolver = ContextIntentResolver()
        self.reasoning_normalizer = ReasoningNormalizer()
        self.fallback_reasoner = PersonalityFallbackReasoner()
        self.reflection_mode = ReflectionMode()
        
        # Conversation state
        self.current_session: Optional[ConversationState] = None
        self.conversation_history: List[ConversationState] = []
        
        # Logging
        self.logger = self._setup_logging()
        
        # Error handling
        self.error_count = 0
        self.max_errors = 5
        
        self.logger.info("ConversationalInterface initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for conversational interface."""
        logger = logging.getLogger('conversational_interface')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(f"{self.data_dir}/conversational.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def start_conversation(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session."""
        if not session_id:
            session_id = f"conv_{int(time.time())}"
        
        # Create conversation context
        context = ConversationContext(
            context_type=ContextType.GENERAL,
            domain="general",
            complexity_level=0.5,
            emotional_tone="neutral"
        )
        
        # Create conversation state
        self.current_session = ConversationState(
            session_id=session_id,
            created_at=datetime.now(),
            current_phase=ConversationPhase.GREETING,
            active_personality=self.personality_framework.active_personality,
            context=context
        )
        
        self.conversation_history.append(self.current_session)
        
        self.logger.info(f"Started conversation session: {session_id}")
        return session_id
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user message and generate response using stabilization layer."""
        try:
            # Ensure we have an active session
            if not self.current_session or (session_id and self.current_session.session_id != session_id):
                self.start_conversation(session_id)
            
            # Step 1: Resolve intent and context
            self.logger.info(f"Processing message: {message[:50]}...")
            context = self.context_resolver.resolve_intent_and_context(message, self.current_session.active_personality)
            self.logger.info(f"Resolved context: {context.reasoning_type} (confidence: {context.confidence})")
            
            # Step 2: Handle personality switch
            if context.intent_type == 'PERSONALITY_SWITCH':
                return self._handle_personality_switch_stabilized(message, context)
            
            # Step 3: Normalize reasoning input
            normalized = self.reasoning_normalizer.normalize_input(message, self.current_session.active_personality, context)
            self.logger.info(f"Normalized input: {normalized.reasoning_type} (fallback: {normalized.requires_fallback})")
            
            # Step 4: Determine processing path
            if normalized.requires_fallback or self.reasoning_normalizer.should_use_fallback(normalized):
                # Use fallback reasoning
                self.logger.info("Using fallback reasoning")
                fallback_response = self.fallback_reasoner.get_fallback_response(
                    message, self.current_session.active_personality, 
                    normalized.reasoning_type, normalized.metadata
                )
                response = fallback_response.content
                reasoning_trace = None
            else:
                # Use reasoning engine
                self.logger.info("Using reasoning engine")
                reasoning_result = self._process_with_reasoning_stabilized(message, normalized)
                response = reasoning_result.get('response', 'I understand your request.')
                reasoning_trace = reasoning_result.get('reasoning_trace')
            
            # Step 5: Generate final response
            final_response = self._generate_stabilized_response(message, response, context, reasoning_trace)
            
            # Step 6: Create conversation turn
            turn = self._create_conversation_turn_stabilized(message, final_response, context, reasoning_trace)
            self.current_session.add_turn(turn)
            
            # Step 7: Log conversation
            self._log_conversation_turn(turn)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return {
                'success': True,
                'response': final_response,
                'personality': self.current_session.active_personality,
                'reasoning_trace': reasoning_trace,
                'session_id': self.current_session.session_id,
                'conversation_summary': self.current_session.get_conversation_summary(),
                'fallback_used': normalized.requires_fallback
            }
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing message: {str(e)}")
            
            # Auto-fix attempt with reflection mode
            if self.error_count <= self.max_errors:
                return self._attempt_auto_fix_with_reflection(message, str(e))
            else:
                return self._generate_error_response(str(e))
    
    def _handle_personality_switch(self, message: str, intent: Intent) -> Dict[str, Any]:
        """Handle personality switch request."""
        # Extract personality name from message
        personality_name = self._extract_personality_name(message)
        
        if not personality_name:
            return {
                'success': False,
                'error': 'Could not determine which personality to switch to',
                'available_personalities': self.personality_framework.get_available_personalities()
            }
        
        # Switch personality
        if self.personality_framework.switch_personality(personality_name):
            self.current_session.switch_personality(personality_name)
            
            # Generate switch response
            personality = self.personality_framework.get_personality(personality_name)
            response = self.response_generator.generate_personality_switch_response(
                self.current_session.active_personality, personality_name
            )
            
            return {
                'success': True,
                'response': response,
                'personality': personality_name,
                'personality_info': personality.to_dict() if personality else None
            }
        else:
            return {
                'success': False,
                'error': f'Personality "{personality_name}" not found',
                'available_personalities': self.personality_framework.get_available_personalities()
            }
    
    def _recommend_personality(self, intent: Intent) -> Optional[str]:
        """Recommend personality based on intent."""
        if intent.personality_hint:
            return intent.personality_hint
        
        # Use personality framework recommendation
        context = {
            'context_type': intent.intent_type.name.lower(),
            'domain': intent.domain,
            'complexity_level': intent.complexity_level
        }
        
        return self.personality_framework.get_personality_recommendation(context)
    
    def _switch_personality(self, personality_name: str) -> None:
        """Switch to a different personality."""
        if self.personality_framework.switch_personality(personality_name):
            self.current_session.switch_personality(personality_name)
            self.logger.info(f"Switched to personality: {personality_name}")
    
    def _update_conversation_context(self, intent: Intent) -> None:
        """Update conversation context based on intent."""
        # Determine context type
        context_type = self._determine_context_type(intent)
        
        # Update context
        new_context = ConversationContext(
            context_type=context_type,
            domain=intent.domain or "general",
            complexity_level=intent.complexity_level,
            emotional_tone=intent.emotional_tone,
            focus_areas=intent.context_clues,
            constraints=[],
            preferences={}
        )
        
        self.current_session.update_context(new_context)
        
        # Update conversation phase
        new_phase = self.current_session.evolve_phase(intent.context_clues[0] if intent.context_clues else "")
        self.current_session.current_phase = new_phase
    
    def _determine_context_type(self, intent: Intent) -> ContextType:
        """Determine context type from intent."""
        if intent.intent_type == IntentType.CIVILIZATIONAL_QUERY:
            return ContextType.CIVILIZATIONAL
        elif intent.intent_type == IntentType.STRATEGIC_ANALYSIS:
            return ContextType.STRATEGIC
        elif intent.intent_type == IntentType.META_RULE_DISCOVERY:
            return ContextType.PHILOSOPHICAL
        elif intent.intent_type == IntentType.UNIVERSE_COMPARISON:
            return ContextType.PHILOSOPHICAL
        else:
            return ContextType.GENERAL
    
    def _process_with_reasoning(self, message: str, intent: Intent) -> Dict[str, Any]:
        """Process message with reasoning engine."""
        context = {
            'domain': intent.domain,
            'complexity_level': intent.complexity_level,
            'urgency': intent.urgency,
            'emotional_tone': intent.emotional_tone
        }
        
        # Route to appropriate reasoning method
        if intent.intent_type == IntentType.CIVILIZATIONAL_QUERY:
            return self.reasoning_engine.process_civilizational_query(message, context)
        elif intent.intent_type == IntentType.STRATEGIC_ANALYSIS:
            return self.reasoning_engine.process_strategic_analysis(message, context)
        elif intent.intent_type == IntentType.META_RULE_DISCOVERY:
            return self.reasoning_engine.process_meta_rule_discovery(message, context)
        elif intent.intent_type == IntentType.UNIVERSE_COMPARISON:
            return self.reasoning_engine.process_universe_comparison(message, context)
        else:
            # Fallback to basic OmniMind processing
            return self._fallback_to_omnimind(message, context)
    
    def _fallback_to_omnimind(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic OmniMind processing."""
        try:
            response = self.omnimind_core.process_input(message)
            return {
                'success': True,
                'response': response['content'],
                'reasoning_trace': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'reasoning_trace': None
            }
    
    def _generate_response(self, message: str, intent: Intent, reasoning_result: Dict[str, Any]) -> str:
        """Generate response using response generator."""
        personality = self.personality_framework.get_active_personality()
        if not personality:
            return "I'm here to help with your inquiry."
        
        # Extract reasoning trace and insights
        reasoning_trace = reasoning_result.get('reasoning_trace')
        civilizational_insights = reasoning_result.get('civilizational_insights', [])
        
        # Generate response
        response = self.response_generator.generate_response(
            personality=personality,
            reasoning_trace=reasoning_trace,
            civilizational_insights=civilizational_insights,
            context={
                'intent_type': intent.intent_type.name,
                'domain': intent.domain,
                'complexity_level': intent.complexity_level
            }
        )
        
        return response
    
    def _create_conversation_turn(self, message: str, response: str, intent: Intent, reasoning_result: Dict[str, Any]) -> 'ConversationTurn':
        """Create a conversation turn."""
        from .conversation_state import ConversationTurn
        
        return ConversationTurn(
            turn_id=f"turn_{int(time.time())}",
            timestamp=datetime.now(),
            user_input=message,
            system_response=response,
            intent=intent.intent_type.name,
            personality_used=self.current_session.active_personality,
            reasoning_trace=reasoning_result.get('reasoning_trace', {}).get('processing_steps', []),
            confidence_score=intent.confidence,
            context_shift=reasoning_result.get('context_shift', False),
            meta_insights=reasoning_result.get('meta_insights', [])
        )
    
    def _log_conversation_turn(self, turn: 'ConversationTurn') -> None:
        """Log conversation turn."""
        self.logger.info(f"Turn {turn.turn_id}: {turn.intent} -> {turn.personality_used} (confidence: {turn.confidence_score})")
    
    def _extract_personality_name(self, message: str) -> Optional[str]:
        """Extract personality name from message."""
        message_lower = message.lower()
        
        for personality_name in self.personality_framework.get_available_personalities():
            if personality_name.lower() in message_lower:
                return personality_name
        
        return None
    
    def _attempt_auto_fix(self, message: str, error: str) -> Dict[str, Any]:
        """Attempt to auto-fix the error."""
        self.logger.info(f"Attempting auto-fix for error: {error}")
        
        # Simple auto-fix strategies
        if "personality" in error.lower():
            # Try switching to default personality
            self.personality_framework.switch_personality("Strategos")
            return self.process_message(message)
        
        elif "reasoning" in error.lower():
            # Try fallback to basic OmniMind
            return self._fallback_to_omnimind(message, {})
        
        else:
            # Generate error response
            return self._generate_error_response(error)
    
    def _generate_error_response(self, error: str) -> Dict[str, Any]:
        """Generate error response."""
        personality = self.personality_framework.get_active_personality()
        if personality:
            response = self.response_generator.generate_error_response(error, personality)
        else:
            response = f"I encountered an error: {error}. Let me try a different approach."
        
        return {
            'success': False,
            'response': response,
            'error': error,
            'personality': self.current_session.active_personality if self.current_session else None
        }
    
    # New stabilized methods
    def _handle_personality_switch_stabilized(self, message: str, context: ReasoningContext) -> Dict[str, Any]:
        """Handle personality switch with stabilization."""
        personality_name = self._extract_personality_name(message)
        
        if not personality_name:
            return {
                'success': False,
                'error': 'Could not determine which personality to switch to',
                'available_personalities': self.personality_framework.get_available_personalities()
            }
        
        if self.personality_framework.switch_personality(personality_name):
            self.current_session.switch_personality(personality_name)
            personality = self.personality_framework.get_personality(personality_name)
            response = self.response_generator.generate_personality_switch_response(
                self.current_session.active_personality, personality_name
            )
            
            return {
                'success': True,
                'response': response,
                'personality': personality_name,
                'personality_info': personality.to_dict() if personality else None
            }
        else:
            return {
                'success': False,
                'error': f'Personality "{personality_name}" not found',
                'available_personalities': self.personality_framework.get_available_personalities()
            }
    
    def _process_with_reasoning_stabilized(self, message: str, normalized: NormalizedReasoningInput) -> Dict[str, Any]:
        """Process with reasoning engine using stabilized input."""
        try:
            # Create proper input for reasoning engine
            reasoning_input = self.reasoning_normalizer.create_reasoning_engine_input(normalized)
            
            # Route to appropriate reasoning method
            if normalized.reasoning_type == 'civilizational_reasoning':
                return self.reasoning_engine.process_civilizational_query(message, reasoning_input)
            elif normalized.reasoning_type == 'strategic_reasoning':
                return self.reasoning_engine.process_strategic_analysis(message, reasoning_input)
            elif normalized.reasoning_type == 'meta_rule_reasoning':
                return self.reasoning_engine.process_meta_rule_discovery(message, reasoning_input)
            elif normalized.reasoning_type == 'universe_reasoning':
                return self.reasoning_engine.process_universe_comparison(message, reasoning_input)
            else:
                # Fallback to basic OmniMind processing
                return self._fallback_to_omnimind_stabilized(message, reasoning_input)
                
        except Exception as e:
            self.logger.error(f"Reasoning engine error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'response': 'I encountered an issue with my reasoning process.',
                'reasoning_trace': None
            }
    
    def _fallback_to_omnimind_stabilized(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic OmniMind processing with stabilization."""
        try:
            response = self.omnimind_core.process_input(message)
            return {
                'success': True,
                'response': response['content'],
                'reasoning_trace': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': 'I understand your request but need to process it differently.',
                'reasoning_trace': None
            }
    
    def _generate_stabilized_response(self, message: str, response: str, context: ReasoningContext, reasoning_trace: Optional[Dict[str, Any]]) -> str:
        """Generate response using stabilization layer."""
        personality = self.personality_framework.get_personality(context.personality)
        if not personality:
            return response
        
        # Apply personality-specific styling
        if context.personality == 'Strategos':
            if not response.startswith('Commander'):
                response = f"Commander, {response.lower()}"
        elif context.personality == 'Archivist':
            if not response.startswith('From the archives'):
                response = f"From the archives of knowledge, {response.lower()}"
        elif context.personality == 'Lawmaker':
            if not response.startswith('According to'):
                response = f"According to the principles of governance, {response.lower()}"
        elif context.personality == 'Oracle':
            if not response.startswith('In the cosmic'):
                response = f"In the cosmic dance of reality, {response.lower()}"
        
        return response
    
    def _create_conversation_turn_stabilized(self, message: str, response: str, context: ReasoningContext, reasoning_trace: Optional[Dict[str, Any]]) -> 'ConversationTurn':
        """Create conversation turn with stabilized context."""
        from .conversation_state import ConversationTurn
        
        return ConversationTurn(
            turn_id=f"turn_{int(time.time())}",
            timestamp=datetime.now(),
            user_input=message,
            system_response=response,
            intent=context.intent_type,
            personality_used=context.personality,
            reasoning_trace=reasoning_trace.get('processing_steps', []) if reasoning_trace else [],
            confidence_score=context.confidence,
            context_shift=context.metadata.get('context_shift', False),
            meta_insights=reasoning_trace.get('meta_insights', []) if reasoning_trace else []
        )
    
    def _attempt_auto_fix_stabilized(self, message: str, error: str) -> Dict[str, Any]:
        """Attempt auto-fix with stabilization layer."""
        self.logger.info(f"Attempting auto-fix for error: {error}")
        
        # Try fallback reasoning
        try:
            fallback_response = self.fallback_reasoner.get_fallback_response(
                message, self.current_session.active_personality, 'basic_reasoning', {}
            )
            return {
                'success': True,
                'response': fallback_response.content,
                'personality': self.current_session.active_personality,
                'fallback_used': True
            }
        except Exception as e:
            self.logger.error(f"Auto-fix failed: {str(e)}")
            return self._generate_error_response(error)
    
    def _attempt_auto_fix_with_reflection(self, message: str, error: str) -> Dict[str, Any]:
        """Attempt auto-fix with reflection mode."""
        self.logger.info(f"Attempting auto-fix with reflection for error: {error}")
        
        try:
            # Use reflection mode to analyze and retry
            reflection_result = self.reflection_mode.reflect_and_retry(
                user_input=message,
                personality=self.current_session.active_personality,
                original_error=error,
                context=None
            )
            
            if reflection_result.success:
                self.logger.info(f"Reflection successful after {reflection_result.attempts_made} attempts")
                return {
                    'success': True,
                    'response': reflection_result.response,
                    'personality': self.current_session.active_personality,
                    'fallback_used': reflection_result.fallback_used,
                    'reflection_used': True,
                    'confidence_improvement': reflection_result.confidence_improvement
                }
            else:
                self.logger.warning("Reflection failed, using final fallback")
                return {
                    'success': True,
                    'response': reflection_result.response,
                    'personality': self.current_session.active_personality,
                    'fallback_used': True,
                    'reflection_used': True,
                    'confidence_improvement': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Reflection mode failed: {str(e)}")
            # Fallback to basic auto-fix
            return self._attempt_auto_fix_stabilized(message, error)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        if not self.current_session:
            return {}
        
        return self.current_session.get_conversation_summary()
    
    def get_personality_insights(self) -> Dict[str, Any]:
        """Get insights about personality usage."""
        return self.personality_framework.get_personality_insights()
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of reasoning activity."""
        return self.reasoning_engine.get_reasoning_summary()
    
    def switch_personality(self, personality_name: str) -> bool:
        """Manually switch personality."""
        if self.personality_framework.switch_personality(personality_name):
            if self.current_session:
                self.current_session.switch_personality(personality_name)
            return True
        return False
    
    def get_available_personalities(self) -> List[str]:
        """Get list of available personalities."""
        return self.personality_framework.get_available_personalities()
    
    def get_personality_insights(self) -> Dict[str, Any]:
        """Get insights about personality usage."""
        if not self.current_session:
            return {}
        
        # Count personality switches
        personality_switches = 0
        personality_usage = {}
        
        for turn in self.current_session.conversation_history:
            if turn.personality_used:
                personality_usage[turn.personality_used] = personality_usage.get(turn.personality_used, 0) + 1
        
        # Count switches (when personality changes between turns)
        for i in range(1, len(self.current_session.conversation_history)):
            if (self.current_session.conversation_history[i].personality_used != 
                self.current_session.conversation_history[i-1].personality_used):
                personality_switches += 1
        
        # Find most used personality
        most_used = max(personality_usage.items(), key=lambda x: x[1])[0] if personality_usage else None
        
        return {
            'total_switches': personality_switches,
            'personality_usage': personality_usage,
            'most_used': most_used,
            'total_turns': len(self.current_session.conversation_history),
            'current_personality': self.current_session.active_personality
        }
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of reasoning activity."""
        if not self.current_session:
            return {}
        
        return {
            'total_traces': len([t for t in self.current_session.conversation_history if t.reasoning_trace]),
            'total_insights': len([t for t in self.current_session.conversation_history if t.meta_insights]),
            'meta_rules_applied': len([t for t in self.current_session.conversation_history if t.meta_insights]),
            'reasoning_types_used': list(set([t.intent for t in self.current_session.conversation_history if t.intent])),
            'average_confidence': sum([t.confidence_score for t in self.current_session.conversation_history if t.confidence_score]) / max(1, len(self.current_session.conversation_history))
        }
    
    def end_conversation(self) -> Dict[str, Any]:
        """End current conversation and return summary."""
        if not self.current_session:
            return {'success': False, 'error': 'No active conversation'}
        
        summary = self.current_session.get_conversation_summary()
        self.current_session = None
        
        return {
            'success': True,
            'conversation_summary': summary,
            'personality_insights': self.get_personality_insights(),
            'reasoning_summary': self.get_reasoning_summary()
        }
