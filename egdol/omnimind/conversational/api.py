"""
Clean Python API for Conversational Personality Layer
Provides simple, clean interface for external use.
"""

from typing import Dict, List, Any, Optional
from .interface import ConversationalInterface
from .personality_framework import PersonalityType


class OmniMindChat:
    """
    Clean Python API for OmniMind Conversational Personality Layer.
    
    This class provides a simple, clean interface for interacting with
    the conversational personality layer without exposing internal complexity.
    """
    
    def __init__(self, data_dir: str = "omnimind_chat_data"):
        """
        Initialize OmniMind Chat interface.
        
        Args:
            data_dir: Directory for storing conversation data
        """
        from ..core import OmniMind
        
        self.omnimind_core = OmniMind(data_dir)
        self.interface = ConversationalInterface(self.omnimind_core, data_dir)
        self.data_dir = data_dir
    
    def chat(self, message: str, personality: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a message and get a response.
        
        Args:
            message: User message
            personality: Optional personality to use (Strategos, Archivist, Lawmaker, Oracle)
            
        Returns:
            Dictionary containing response and metadata
        """
        # Start conversation if not already started
        if not self.interface.current_session:
            self.interface.start_conversation()
        
        # Switch personality if requested
        if personality and personality != self.interface.current_session.active_personality:
            self.interface.switch_personality(personality)
        
        # Process message
        result = self.interface.process_message(message)
        
        return {
            'response': result.get('response', 'I apologize, but I encountered an error.'),
            'personality': result.get('personality', self.interface.current_session.active_personality),
            'success': result.get('success', False),
            'reasoning_available': 'reasoning_trace' in result,
            'session_id': result.get('session_id')
        }
    
    def switch_personality(self, personality: str) -> bool:
        """
        Switch to a different personality.
        
        Args:
            personality: Name of personality to switch to
            
        Returns:
            True if successful, False otherwise
        """
        return self.interface.switch_personality(personality)
    
    def get_available_personalities(self) -> List[str]:
        """
        Get list of available personalities.
        
        Returns:
            List of personality names
        """
        return self.interface.get_available_personalities()
    
    def get_current_personality(self) -> str:
        """
        Get current active personality.
        
        Returns:
            Current personality name
        """
        if self.interface.current_session:
            return self.interface.current_session.active_personality
        return "None"
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        return self.interface.get_conversation_summary()
    
    def get_personality_insights(self) -> Dict[str, Any]:
        """
        Get insights about personality usage.
        
        Returns:
            Dictionary with personality usage statistics
        """
        return self.interface.get_personality_insights()
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """
        Get summary of reasoning activity.
        
        Returns:
            Dictionary with reasoning statistics
        """
        return self.interface.get_reasoning_summary()
    
    def end_conversation(self) -> Dict[str, Any]:
        """
        End current conversation and get final summary.
        
        Returns:
            Dictionary with conversation summary and insights
        """
        return self.interface.end_conversation()
    
    def start_new_conversation(self) -> str:
        """
        Start a new conversation session.
        
        Returns:
            New session ID
        """
        return self.interface.start_conversation()


# Convenience functions for quick access
def create_chat(data_dir: str = "omnimind_chat_data") -> OmniMindChat:
    """
    Create a new OmniMind chat instance.
    
    Args:
        data_dir: Directory for storing conversation data
        
    Returns:
        OmniMindChat instance
    """
    return OmniMindChat(data_dir)


def quick_chat(message: str, personality: Optional[str] = None, data_dir: str = "omnimind_chat_data") -> str:
    """
    Quick chat function for simple interactions.
    
    Args:
        message: User message
        personality: Optional personality to use
        data_dir: Directory for storing conversation data
        
    Returns:
        Response string
    """
    chat = OmniMindChat(data_dir)
    result = chat.chat(message, personality)
    return result['response']


# Personality-specific convenience functions
def chat_with_strategos(message: str, data_dir: str = "omnimind_chat_data") -> str:
    """Chat with Strategos personality (military strategist)."""
    return quick_chat(message, "Strategos", data_dir)


def chat_with_archivist(message: str, data_dir: str = "omnimind_chat_data") -> str:
    """Chat with Archivist personality (historian-philosopher)."""
    return quick_chat(message, "Archivist", data_dir)


def chat_with_lawmaker(message: str, data_dir: str = "omnimind_chat_data") -> str:
    """Chat with Lawmaker personality (meta-rule discoverer)."""
    return quick_chat(message, "Lawmaker", data_dir)


def chat_with_oracle(message: str, data_dir: str = "omnimind_chat_data") -> str:
    """Chat with Oracle personality (universe comparer)."""
    return quick_chat(message, "Oracle", data_dir)


# Advanced API for power users
class OmniMindChatAdvanced(OmniMindChat):
    """
    Advanced API with access to internal components.
    
    This class provides access to internal components for advanced users
    who need more control over the conversational interface.
    """
    
    def __init__(self, data_dir: str = "omnimind_chat_data"):
        super().__init__(data_dir)
    
    def get_interface(self) -> ConversationalInterface:
        """Get the underlying conversational interface."""
        return self.interface
    
    def get_omnimind_core(self):
        """Get the underlying OmniMind core."""
        return self.omnimind_core
    
    def get_personality_framework(self):
        """Get the personality framework."""
        return self.interface.personality_framework
    
    def get_reasoning_engine(self):
        """Get the reasoning engine."""
        return self.interface.reasoning_engine
    
    def get_intent_parser(self):
        """Get the intent parser."""
        return self.interface.intent_parser
    
    def get_response_generator(self):
        """Get the response generator."""
        return self.interface.response_generator
    
    def process_with_reasoning(self, message: str, personality: Optional[str] = None) -> Dict[str, Any]:
        """
        Process message with full reasoning trace.
        
        Args:
            message: User message
            personality: Optional personality to use
            
        Returns:
            Dictionary with response and full reasoning trace
        """
        # Start conversation if not already started
        if not self.interface.current_session:
            self.interface.start_conversation()
        
        # Switch personality if requested
        if personality and personality != self.interface.current_session.active_personality:
            self.interface.switch_personality(personality)
        
        # Process message
        result = self.interface.process_message(message)
        
        return {
            'response': result.get('response', 'I apologize, but I encountered an error.'),
            'personality': result.get('personality', self.interface.current_session.active_personality),
            'success': result.get('success', False),
            'reasoning_trace': result.get('reasoning_trace'),
            'session_id': result.get('session_id'),
            'conversation_summary': result.get('conversation_summary')
        }
    
    def analyze_intent(self, message: str) -> Dict[str, Any]:
        """
        Analyze the intent of a message without processing it.
        
        Args:
            message: Message to analyze
            
        Returns:
            Dictionary with intent analysis
        """
        intent = self.interface.intent_parser.parse(message)
        return intent.to_dict()
    
    def get_conversation_state(self):
        """Get the current conversation state."""
        return self.interface.current_session
    
    def set_verbose_mode(self, verbose: bool = True):
        """Set verbose mode for detailed output."""
        # This would need to be implemented in the interface
        pass
    
    def set_explain_mode(self, explain: bool = True):
        """Set explain mode for reasoning explanations."""
        # This would need to be implemented in the interface
        pass
