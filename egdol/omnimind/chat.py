"""
OmniMind Chat Interface
Interactive chatbot interface for OmniMind.
"""

import readline
import sys
import os
import time
from typing import Dict, Any, Optional
from egdol.omnimind.core import OmniMind


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class OmniMindChat:
    """Interactive chat interface for OmniMind."""
    
    def __init__(self, data_dir: str = "omnimind_data"):
        self.omnimind = OmniMind(data_dir)
        self.history_file = os.path.expanduser('~/.omnimind_history')
        self.verbose_mode = False
        self.explain_mode = False
        
        # Load history
        self._load_history()
        
        # Set up readline
        readline.set_completer(self._completer)
        readline.parse_and_bind('tab: complete')
        
    def run(self):
        """Run the interactive chat."""
        self._show_welcome()
        
        while True:
            try:
                # Get user input
                prompt = self._get_prompt()
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                    
                # Handle exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print(f"{Colors.YELLOW}Goodbye! 👋{Colors.END}")
                    break
                    
                # Process input
                response = self.omnimind.process_input(user_input)
                self._display_response(response)
                
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Use 'quit' to exit{Colors.END}")
                continue
            except EOFError:
                print(f"\n{Colors.YELLOW}Goodbye! 👋{Colors.END}")
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}")
                
    def _show_welcome(self):
        """Show welcome message."""
        print(f"{Colors.CYAN}{Colors.BOLD}🧠 OmniMind Core - Local AI Assistant{Colors.END}")
        print(f"{Colors.YELLOW}Powered by Egdol Reasoning Engine{Colors.END}")
        print(f"{Colors.YELLOW}Type 'help' for commands, 'quit' to exit{Colors.END}")
        print()
        
    def _get_prompt(self) -> str:
        """Get input prompt."""
        if self.verbose_mode:
            return f"{Colors.GREEN}omnimind[verbose]> {Colors.END}"
        else:
            return f"{Colors.GREEN}omnimind> {Colors.END}"
            
    def _display_response(self, response: Dict[str, Any]):
        """Display response to user."""
        content = response.get('content', 'No response')
        reasoning = response.get('reasoning', [])
        metadata = response.get('metadata', {})
        
        # Display main response
        print(f"{Colors.WHITE}{content}{Colors.END}")
        
        # Display reasoning if verbose mode is on
        if self.verbose_mode and reasoning:
            print(f"{Colors.CYAN}💭 Reasoning:{Colors.END}")
            for step in reasoning:
                print(f"  {Colors.CYAN}• {step}{Colors.END}")
                
        # Display metadata if verbose mode is on
        if self.verbose_mode and metadata:
            skill = metadata.get('skill', 'unknown')
            print(f"{Colors.MAGENTA}🔧 Skill: {skill}{Colors.END}")
            
        print()
        
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for readline."""
        commands = [
            'help', 'verbose', 'explain', 'memory', 'stats', 'reset',
            'quit', 'exit', 'bye', 'goodbye'
        ]
        
        matches = [cmd for cmd in commands if cmd.startswith(text.lower())]
        
        if state < len(matches):
            return matches[state]
        return None
        
    def _load_history(self):
        """Load command history."""
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass
            
    def _save_history(self):
        """Save command history."""
        try:
            readline.write_history_file(self.history_file)
        except Exception:
            pass


def main():
    """Main entry point for the chat interface."""
    chat = OmniMindChat()
    try:
        chat.run()
    finally:
        chat._save_history()


if __name__ == '__main__':
    main()
