"""
OmniMind Core - Main conversation loop and coordination
"""

import time
import re
from typing import Dict, Any, List, Optional, Tuple
from ..rules_engine import RulesEngine
from ..interpreter import Interpreter
from ..dsl.simple_dsl import SimpleDSL
from .nlu_translator import NLUTranslator
from .memory import ConversationMemory
from .router import SkillRouter


class OmniMind:
    """Main OmniMind chatbot core."""
    
    def __init__(self, data_dir: str = "omnimind_data"):
        self.data_dir = data_dir
        
        # Initialize core components
        self.engine = RulesEngine()
        self.interpreter = Interpreter(self.engine)
        self.dsl = SimpleDSL(self.engine)
        
        # Initialize OmniMind components
        self.nlu_translator = NLUTranslator()
        self.memory = ConversationMemory(data_dir)
        self.router = SkillRouter()
        
        # Conversation state
        self.session_id = f"session_{int(time.time())}"
        self.conversation_history: List[Dict[str, Any]] = []
        self.verbose_mode = False
        self.explain_mode = False
        
        # Load existing knowledge
        self._load_persistent_knowledge()
        
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate response."""
        # Store input in conversation history
        self.conversation_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': time.time()
        })
        
        # Store in memory
        self.memory.store_input(user_input, self.session_id)
        
        # Detect intent and route
        intent = self._detect_intent(user_input)
        response = self._route_and_process(user_input, intent)
        
        # Store response in conversation history
        self.conversation_history.append({
            'type': 'assistant',
            'content': response['content'],
            'reasoning': response.get('reasoning', []),
            'timestamp': time.time()
        })
        
        # Store in memory
        self.memory.store_response(response['content'], self.session_id)
        
        return response
        
    def _detect_intent(self, user_input: str) -> str:
        """Detect user intent from input."""
        user_input_lower = user_input.lower().strip()
        
        # Command patterns
        if user_input_lower.startswith(':'):
            return 'command'
        elif user_input_lower.startswith('explain') or user_input_lower.startswith('why'):
            return 'explain'
        elif user_input_lower.startswith('teach') or user_input_lower.startswith('learn'):
            return 'learning'
        elif user_input_lower.startswith('remember'):
            return 'memory'
        elif user_input_lower.startswith('forget'):
            return 'forget'
        elif user_input_lower.startswith('what do you know'):
            return 'knowledge_query'
        elif user_input_lower.startswith('who is') or user_input_lower.startswith('what is'):
            return 'factual_query'
        elif user_input_lower.startswith('if') and 'then' in user_input_lower:
            return 'rule_definition'
        elif user_input_lower.startswith('calculate') or any(op in user_input_lower for op in ['+', '-', '*', '/', '=']):
            return 'math'
        elif user_input_lower.startswith('write') or user_input_lower.startswith('code'):
            return 'coding'
        elif user_input_lower.startswith('analyze') or user_input_lower.startswith('read'):
            return 'file_analysis'
        elif '?' in user_input:
            return 'question'
        else:
            return 'general'
            
    def _route_and_process(self, user_input: str, intent: str) -> Dict[str, Any]:
        """Route input to appropriate handler and process."""
        reasoning_steps = []
        
        if self.verbose_mode:
            reasoning_steps.append(f"Detected intent: {intent}")
            
        # Handle special intents
        if intent == 'command':
            return self._handle_command(user_input)
        elif intent == 'explain':
            return self._handle_explain_request(user_input)
        elif intent == 'learning':
            return self._handle_learning_request(user_input)
        elif intent == 'memory':
            return self._handle_memory_request(user_input)
        elif intent == 'forget':
            return self._handle_forget_request(user_input)
            
        # Route to skill system
        skill_response = self.router.route(user_input, intent, self._get_context())
        
        if skill_response['handled']:
            if self.verbose_mode:
                reasoning_steps.extend(skill_response.get('reasoning', []))
            return {
                'content': skill_response['response'],
                'reasoning': reasoning_steps,
                'skill_used': skill_response.get('skill', 'unknown')
            }
            
        # Fallback to Egdol reasoning
        return self._fallback_to_egdol(user_input, reasoning_steps)
        
    def _handle_command(self, user_input: str) -> Dict[str, Any]:
        """Handle system commands."""
        command = user_input[1:].strip().lower()
        
        if command == 'verbose':
            self.verbose_mode = not self.verbose_mode
            return {
                'content': f"Verbose mode {'enabled' if self.verbose_mode else 'disabled'}",
                'reasoning': []
            }
        elif command == 'explain':
            self.explain_mode = not self.explain_mode
            return {
                'content': f"Explain mode {'enabled' if self.explain_mode else 'disabled'}",
                'reasoning': []
            }
        elif command == 'memory':
            memories = self.memory.get_recent_memories(self.session_id, limit=10)
            content = "Recent memories:\n" + "\n".join(f"- {m['content']}" for m in memories)
            return {'content': content, 'reasoning': []}
        elif command == 'stats':
            stats = self._get_system_stats()
            content = f"System stats: {stats}"
            return {'content': content, 'reasoning': []}
        elif command == 'reset':
            self._reset_session()
            return {'content': "Session reset", 'reasoning': []}
        else:
            return {'content': f"Unknown command: {command}", 'reasoning': []}
            
    def _handle_explain_request(self, user_input: str) -> Dict[str, Any]:
        """Handle explain/why requests."""
        # Get the last response to explain
        if len(self.conversation_history) >= 2:
            last_response = self.conversation_history[-2]
            if last_response['type'] == 'assistant':
                reasoning = last_response.get('reasoning', [])
                if reasoning:
                    content = "Here's how I arrived at that conclusion:\n" + "\n".join(f"- {step}" for step in reasoning)
                else:
                    content = "I don't have detailed reasoning steps for that response."
            else:
                content = "No previous response to explain."
        else:
            content = "No previous response to explain."
            
        return {'content': content, 'reasoning': []}
        
    def _handle_learning_request(self, user_input: str) -> Dict[str, Any]:
        """Handle learning/teaching requests."""
        # Extract what to learn
        if 'teach yourself' in user_input.lower():
            skill_name = user_input.lower().replace('teach yourself', '').replace('how to', '').strip()
            return self._create_dynamic_skill(skill_name)
        else:
            return {'content': "I can learn new skills. Try: 'Teach yourself how to [skill]'", 'reasoning': []}
            
    def _handle_memory_request(self, user_input: str) -> Dict[str, Any]:
        """Handle memory requests."""
        content = user_input.replace('remember', '').strip()
        if content:
            self.memory.store_fact(content, self.session_id)
            return {'content': f"I'll remember: {content}", 'reasoning': []}
        else:
            return {'content': "What would you like me to remember?", 'reasoning': []}
            
    def _handle_forget_request(self, user_input: str) -> Dict[str, Any]:
        """Handle forget requests."""
        pattern = user_input.replace('forget', '').strip()
        if pattern:
            count = self.memory.forget_pattern(pattern, self.session_id)
            return {'content': f"Forgot {count} memories matching '{pattern}'", 'reasoning': []}
        else:
            return {'content': "What would you like me to forget?", 'reasoning': []}
            
    def _create_dynamic_skill(self, skill_name: str) -> Dict[str, Any]:
        """Create a new skill dynamically."""
        # This is a placeholder - in practice, you'd generate actual skill code
        skill_code = f"""
# Auto-generated skill for: {skill_name}
def can_handle(query):
    return '{skill_name}' in query.lower()
    
def handle(query, context):
    return "I'm learning about {skill_name}. Please provide more details."
"""
        
        # Save skill (placeholder)
        return {'content': f"Created new skill for: {skill_name}", 'reasoning': []}
        
    def _fallback_to_egdol(self, user_input: str, reasoning_steps: List[str]) -> Dict[str, Any]:
        """Fallback to Egdol reasoning when no skill handles the input."""
        if self.verbose_mode:
            reasoning_steps.append("No skill could handle input, trying Egdol reasoning")
            
        try:
            # Try to translate to DSL
            dsl_query = self.nlu_translator.translate(user_input)
            if dsl_query:
                result = self.dsl.execute(dsl_query)
                
                if result.get('type') == 'query' and result.get('results'):
                    content = f"Based on my knowledge: {self._format_egdol_results(result['results'])}"
                elif result.get('type') == 'fact':
                    content = f"I've learned: {result['description']}"
                elif result.get('type') == 'rule':
                    content = f"I've added the rule: {result['description']}"
                else:
                    content = "I don't have enough information to answer that. Could you provide more details?"
                    
                if self.verbose_mode:
                    reasoning_steps.append(f"Egdol reasoning: {dsl_query}")
                    
                return {'content': content, 'reasoning': reasoning_steps}
            else:
                return {'content': "I don't understand. Could you rephrase or provide more context?", 'reasoning': reasoning_steps}
                
        except Exception as e:
            if self.verbose_mode:
                reasoning_steps.append(f"Egdol reasoning failed: {e}")
            return {'content': "I encountered an error processing that. Could you try again?", 'reasoning': reasoning_steps}
            
    def _format_egdol_results(self, results: List[Dict[str, Any]]) -> str:
        """Format Egdol query results for display."""
        if not results:
            return "No results found"
            
        formatted = []
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                parts = []
                for var, value in result.items():
                    parts.append(f"{var}={value}")
                formatted.append(f"{i}. {', '.join(parts)}")
            else:
                formatted.append(f"{i}. {result}")
                
        return "\n".join(formatted)
        
    def _get_context(self) -> Dict[str, Any]:
        """Get current conversation context."""
        return {
            'session_id': self.session_id,
            'conversation_history': self.conversation_history[-5:],  # Last 5 exchanges
            'engine_stats': self.engine.stats(),
            'memory_stats': self.memory.get_stats()
        }
        
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'session_id': self.session_id,
            'conversation_length': len(self.conversation_history),
            'engine_stats': self.engine.stats(),
            'memory_stats': self.memory.get_stats(),
            'skills_loaded': len(self.router.get_loaded_skills())
        }
        
    def _reset_session(self):
        """Reset current session."""
        self.conversation_history = []
        self.session_id = f"session_{int(time.time())}"
        
    def _load_persistent_knowledge(self):
        """Load persistent knowledge from memory."""
        # Load facts and rules from memory
        memories = self.memory.get_all_facts()
        for memory in memories:
            try:
                self.dsl.execute(memory['content'])
            except Exception:
                pass  # Skip invalid memories
                
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history
        
    def set_verbose_mode(self, enabled: bool):
        """Set verbose mode."""
        self.verbose_mode = enabled
        
    def set_explain_mode(self, enabled: bool):
        """Set explain mode."""
        self.explain_mode = enabled
