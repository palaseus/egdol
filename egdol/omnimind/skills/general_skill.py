"""
General Skill for OmniMind
Handles general knowledge and fallback responses.
"""

import re
from typing import Dict, Any
from .base import BaseSkill


class GeneralSkill(BaseSkill):
    """Handles general knowledge and fallback responses."""
    
    def __init__(self):
        super().__init__()
        self.description = "Handles general knowledge and fallback responses"
        self.capabilities = [
            "general knowledge",
            "conversation",
            "fallback responses",
            "context awareness"
        ]
        
        # General knowledge base
        self.knowledge_base = {
            'greetings': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'
            ],
            'farewells': [
                'goodbye', 'bye', 'see you', 'farewell', 'take care'
            ],
            'questions': [
                'how are you', 'what can you do', 'who are you', 'what is your name'
            ]
        }
        
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        """Check if this skill can handle the input."""
        # This is a fallback skill, so it can handle most inputs
        # but with lower priority
        return True
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general input."""
        user_input_lower = user_input.lower().strip()
        
        # Handle greetings
        if any(greeting in user_input_lower for greeting in self.knowledge_base['greetings']):
            return {
                'content': "Hello! I'm OmniMind, your local AI assistant. How can I help you today?",
                'reasoning': ['Detected greeting'],
                'metadata': {'skill': 'general', 'type': 'greeting'}
            }
            
        # Handle farewells
        if any(farewell in user_input_lower for farewell in self.knowledge_base['farewells']):
            return {
                'content': "Goodbye! It was nice talking with you. Feel free to come back anytime!",
                'reasoning': ['Detected farewell'],
                'metadata': {'skill': 'general', 'type': 'farewell'}
            }
            
        # Handle questions about capabilities
        if 'what can you do' in user_input_lower or 'what are your capabilities' in user_input_lower:
            return {
                'content': "I can help you with:\n- Mathematical calculations\n- Logical reasoning\n- General questions\n- Code analysis\n- File operations\n- And much more! Just ask me anything.",
                'reasoning': ['Capability question'],
                'metadata': {'skill': 'general', 'type': 'capabilities'}
            }
            
        # Handle questions about identity
        if 'who are you' in user_input_lower or 'what is your name' in user_input_lower:
            return {
                'content': "I'm OmniMind, a local AI assistant powered by Egdol reasoning engine. I can help you with various tasks and answer questions using logical reasoning.",
                'reasoning': ['Identity question'],
                'metadata': {'skill': 'general', 'type': 'identity'}
            }
            
        # Handle how are you questions
        if 'how are you' in user_input_lower:
            return {
                'content': "I'm doing well, thank you for asking! I'm ready to help you with any questions or tasks you have.",
                'reasoning': ['Status question'],
                'metadata': {'skill': 'general', 'type': 'status'}
            }
            
        # Handle general questions
        if user_input_lower.endswith('?'):
            return {
                'content': "That's an interesting question. I'd need more specific information to give you a good answer. Could you provide more details?",
                'reasoning': ['General question'],
                'metadata': {'skill': 'general', 'type': 'question'}
            }
            
        # Handle statements
        if not user_input_lower.endswith('?'):
            return {
                'content': "I understand. Is there anything specific you'd like me to help you with regarding that?",
                'reasoning': ['General statement'],
                'metadata': {'skill': 'general', 'type': 'statement'}
            }
            
        # Fallback
        return {
            'content': "I'm not sure how to help with that. Could you rephrase your request or provide more context?",
            'reasoning': ['Fallback response'],
            'metadata': {'skill': 'general', 'type': 'fallback'}
        }
