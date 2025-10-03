"""
Interactive REPL for Egdol DSL.
Provides a command-line interface with history, color highlighting, and commands.
"""

import readline
import sys
import os
from typing import List, Dict, Any, Optional
from .translator import DSLExecutor
from ..rules_engine import RulesEngine
from ..interpreter import Interpreter


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


class DSLREPL:
    """Interactive REPL for the Egdol DSL."""
    
    def __init__(self):
        self.engine = RulesEngine()
        self.executor = DSLExecutor(self.engine)
        self.interpreter = Interpreter(self.engine)
        self.history_file = os.path.expanduser('~/.egdol_history')
        self.session_facts = []
        self.session_rules = []
        self.session_queries = []
        
        # Load history
        self._load_history()
        
        # Set up readline
        readline.set_completer(self._completer)
        readline.parse_and_bind('tab: complete')
        
    def run(self):
        """Run the interactive REPL."""
        print(f"{Colors.CYAN}{Colors.BOLD}🤖 Egdol Interactive DSL Assistant{Colors.END}")
        print(f"{Colors.YELLOW}Type 'help' or ':help' for commands, 'quit' or ':quit' to exit{Colors.END}")
        print()
        
        while True:
            try:
                # Get input with prompt
                prompt = f"{Colors.GREEN}egdol> {Colors.END}"
                line = input(prompt).strip()
                
                if not line:
                    continue
                    
                # Handle exit commands
                if line.lower() in ['quit', 'exit', ':quit', ':exit']:
                    print(f"{Colors.YELLOW}Goodbye! 👋{Colors.END}")
                    break
                    
                # Execute the line
                self._execute_line(line)
                
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Use 'quit' or ':quit' to exit{Colors.END}")
                continue
            except EOFError:
                print(f"\n{Colors.YELLOW}Goodbye! 👋{Colors.END}")
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}")
                
    def _execute_line(self, line: str):
        """Execute a single line of input."""
        # Handle commands
        if line.startswith(':'):
            self._handle_command(line)
        else:
            # Handle DSL statements
            self._handle_dsl_statement(line)
            
    def _handle_command(self, line: str):
        """Handle command statements."""
        parts = line[1:].split()
        if not parts:
            return
            
        command = parts[0].lower()
        args = parts[1:]
        
        if command == 'help':
            self._show_help()
        elif command == 'facts':
            self._show_facts()
        elif command == 'rules':
            self._show_rules()
        elif command == 'reset':
            self._reset_session()
        elif command == 'save':
            self._save_session(args[0] if args else 'session.egdol')
        elif command == 'load':
            self._load_session(args[0] if args else 'session.egdol')
        elif command == 'trace':
            self._trace_query(' '.join(args))
        elif command == 'stats':
            self._show_stats()
        elif command == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
        else:
            print(f"{Colors.RED}Unknown command: {command}{Colors.END}")
            print(f"{Colors.YELLOW}Type ':help' for available commands{Colors.END}")
            
    def _handle_dsl_statement(self, line: str):
        """Handle DSL statements."""
        try:
            results = self.executor.execute(line)
            
            # Process results
            if results['facts']:
                for fact_result in results['facts']:
                    self.session_facts.append(fact_result)
                    print(f"{Colors.GREEN}✓ Added fact: {fact_result['description']}{Colors.END}")
                    
            if results['rules']:
                for rule_result in results['rules']:
                    self.session_rules.append(rule_result)
                    print(f"{Colors.BLUE}✓ Added rule: {rule_result['description']}{Colors.END}")
                    
            if results['queries']:
                for query_result in results['queries']:
                    self.session_queries.append(query_result)
                    self._display_query_results(query_result)
                    
            if results['commands']:
                for cmd_result in results['commands']:
                    print(f"{Colors.MAGENTA}Executed: {cmd_result['description']}{Colors.END}")
                    
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
            
    def _display_query_results(self, query_result: Dict[str, Any]):
        """Display query results in a formatted way."""
        results = query_result['results']
        description = query_result['description']
        
        if not results:
            print(f"{Colors.YELLOW}No results found for: {description}{Colors.END}")
        else:
            print(f"{Colors.CYAN}Results for: {description}{Colors.END}")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {self._format_result(result)}")
                
    def _format_result(self, result: Dict[str, Any]) -> str:
        """Format a single result for display."""
        if not result:
            return "No bindings"
            
        parts = []
        for var, value in result.items():
            if isinstance(value, str):
                parts.append(f"{var}={value}")
            else:
                parts.append(f"{var}={value}")
                
        return ", ".join(parts)
        
    def _show_help(self):
        """Show help information."""
        print(f"{Colors.BOLD}Egdol DSL Commands:{Colors.END}")
        print(f"  {Colors.GREEN}:help{Colors.END}     - Show this help")
        print(f"  {Colors.GREEN}:facts{Colors.END}    - Show all facts")
        print(f"  {Colors.GREEN}:rules{Colors.END}    - Show all rules")
        print(f"  {Colors.GREEN}:reset{Colors.END}    - Clear session")
        print(f"  {Colors.GREEN}:save{Colors.END}    - Save session to file")
        print(f"  {Colors.GREEN}:load{Colors.END}    - Load session from file")
        print(f"  {Colors.GREEN}:trace{Colors.END}    - Trace query execution")
        print(f"  {Colors.GREEN}:stats{Colors.END}    - Show session statistics")
        print(f"  {Colors.GREEN}:clear{Colors.END}    - Clear screen")
        print(f"  {Colors.GREEN}:quit{Colors.END}     - Exit")
        print()
        print(f"{Colors.BOLD}DSL Examples:{Colors.END}")
        print(f"  {Colors.CYAN}Person Alice is a human.{Colors.END}")
        print(f"  {Colors.CYAN}Human X is mortal.{Colors.END}")
        print(f"  {Colors.CYAN}Who is mortal?{Colors.END}")
        print(f"  {Colors.CYAN}Is Alice mortal?{Colors.END}")
        print(f"  {Colors.CYAN}If X is a human then X is mortal.{Colors.END}")
        
    def _show_facts(self):
        """Show all facts."""
        if not self.session_facts:
            print(f"{Colors.YELLOW}No facts in session{Colors.END}")
            return
            
        print(f"{Colors.BOLD}Session Facts:{Colors.END}")
        for i, fact in enumerate(self.session_facts, 1):
            print(f"  {i}. {fact['description']}")
            
    def _show_rules(self):
        """Show all rules."""
        if not self.session_rules:
            print(f"{Colors.YELLOW}No rules in session{Colors.END}")
            return
            
        print(f"{Colors.BOLD}Session Rules:{Colors.END}")
        for i, rule in enumerate(self.session_rules, 1):
            print(f"  {i}. {rule['description']}")
            
    def _reset_session(self):
        """Reset the session."""
        self.engine = RulesEngine()
        self.executor = DSLExecutor(self.engine)
        self.interpreter = Interpreter(self.engine)
        self.session_facts = []
        self.session_rules = []
        self.session_queries = []
        print(f"{Colors.GREEN}Session reset{Colors.END}")
        
    def _save_session(self, filename: str):
        """Save session to file."""
        try:
            with open(filename, 'w') as f:
                f.write("# Egdol Session\n\n")
                f.write("# Facts\n")
                for fact in self.session_facts:
                    f.write(f"{fact['description']}\n")
                f.write("\n# Rules\n")
                for rule in self.session_rules:
                    f.write(f"{rule['description']}\n")
            print(f"{Colors.GREEN}Session saved to {filename}{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Error saving session: {e}{Colors.END}")
            
    def _load_session(self, filename: str):
        """Load session from file."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                
            print(f"{Colors.GREEN}Loading session from {filename}...{Colors.END}")
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    self._handle_dsl_statement(line)
                    
        except FileNotFoundError:
            print(f"{Colors.RED}File not found: {filename}{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Error loading session: {e}{Colors.END}")
            
    def _trace_query(self, query_text: str):
        """Trace query execution."""
        if not query_text:
            print(f"{Colors.RED}Please provide a query to trace{Colors.END}")
            return
            
        print(f"{Colors.CYAN}Tracing query: {query_text}{Colors.END}")
        # This would implement detailed tracing
        # For now, just execute the query
        self._handle_dsl_statement(query_text)
        
    def _show_stats(self):
        """Show session statistics."""
        stats = self.engine.stats()
        print(f"{Colors.BOLD}Session Statistics:{Colors.END}")
        print(f"  Facts: {stats['num_facts']}")
        print(f"  Rules: {stats['num_rules']}")
        print(f"  Session Facts: {len(self.session_facts)}")
        print(f"  Session Rules: {len(self.session_rules)}")
        print(f"  Session Queries: {len(self.session_queries)}")
        
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for readline."""
        commands = [
            'help', 'facts', 'rules', 'reset', 'save', 'load', 
            'trace', 'stats', 'clear', 'quit'
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
    """Main entry point for the REPL."""
    repl = DSLREPL()
    try:
        repl.run()
    finally:
        repl._save_history()


if __name__ == '__main__':
    main()
