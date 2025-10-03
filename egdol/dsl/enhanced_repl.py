"""
Enhanced REPL for Egdol with Persistent Memory and Autonomous Behaviors.
Provides a comprehensive interface with memory, introspection, and multi-agent support.
"""

import readline
import sys
import os
import time
from typing import List, Dict, Any, Optional
from .simple_dsl import SimpleDSL
from ..rules_engine import RulesEngine
from ..interpreter import Interpreter
from ..memory import MemoryStore, MemoryItem
from ..meta import MemoryInspector, RuleInspector, RuleScorer, ConfidenceTracker
from ..agents import AgentManager, AgentProfile
from ..autonomous import BehaviorScheduler, WatcherManager, ActionManager


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


class EnhancedREPL:
    """Enhanced REPL with persistent memory and autonomous behaviors."""
    
    def __init__(self, data_dir: str = "egdol_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize core components
        self.engine = RulesEngine()
        self.interpreter = Interpreter(self.engine)
        self.memory_store = MemoryStore(os.path.join(data_dir, "memory.db"))
        self.dsl = SimpleDSL(self.engine)
        
        # Initialize meta components
        self.memory_inspector = MemoryInspector(self.memory_store)
        self.rule_inspector = RuleInspector(self.engine, self.memory_store)
        self.rule_scorer = RuleScorer(self.memory_store)
        self.confidence_tracker = ConfidenceTracker(self.memory_store)
        
        # Initialize autonomous components
        self.scheduler = BehaviorScheduler()
        self.watcher_manager = WatcherManager()
        self.action_manager = ActionManager()
        
        # Initialize agent manager
        self.agent_manager = AgentManager(os.path.join(data_dir, "agents"))
        
        # Session state
        self.current_agent = None
        self.session_facts = []
        self.session_rules = []
        self.session_queries = []
        
        # Load history
        self.history_file = os.path.expanduser('~/.egdol_enhanced_history')
        self._load_history()
        
        # Set up readline
        readline.set_completer(self._completer)
        readline.parse_and_bind('tab: complete')
        
        # Start autonomous behaviors
        self._setup_autonomous_behaviors()
        self.scheduler.start()
        
    def run(self):
        """Run the enhanced REPL."""
        print(f"{Colors.CYAN}{Colors.BOLD}🧠 Egdol Enhanced AI Assistant{Colors.END}")
        print(f"{Colors.YELLOW}Persistent Memory • Autonomous Behaviors • Multi-Agent Support{Colors.END}")
        print(f"{Colors.YELLOW}Type 'help' or ':help' for commands, 'quit' or ':quit' to exit{Colors.END}")
        print()
        
        while True:
            try:
                # Get input with prompt
                agent_name = self.current_agent.profile.name if self.current_agent else "egdol"
                prompt = f"{Colors.GREEN}{agent_name}> {Colors.END}"
                line = input(prompt).strip()
                
                if not line:
                    continue
                    
                # Handle exit commands
                if line.lower() in ['quit', 'exit', ':quit', ':exit']:
                    print(f"{Colors.YELLOW}Goodbye! 👋{Colors.END}")
                    self._cleanup()
                    break
                    
                # Execute the line
                self._execute_line(line)
                
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Use 'quit' or ':quit' to exit{Colors.END}")
                continue
            except EOFError:
                print(f"\n{Colors.YELLOW}Goodbye! 👋{Colors.END}")
                self._cleanup()
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
        elif command == 'remember':
            self._remember(' '.join(args))
        elif command == 'forget':
            self._forget(' '.join(args))
        elif command == 'memories':
            self._list_memories()
        elif command == 'introspect':
            self._introspect()
        elif command == 'agents':
            self._list_agents()
        elif command == 'switch':
            self._switch_agent(args[0] if args else None)
        elif command == 'create':
            self._create_agent(args)
        elif command == 'watchers':
            self._list_watchers()
        elif command == 'tasks':
            self._list_tasks()
        else:
            print(f"{Colors.RED}Unknown command: {command}{Colors.END}")
            print(f"{Colors.YELLOW}Type ':help' for available commands{Colors.END}")
            
    def _handle_dsl_statement(self, line: str):
        """Handle DSL statements."""
        try:
            # Store input in memory
            self.memory_store.store(
                content=line,
                item_type='input',
                source='user',
                confidence=1.0
            )
            
            # Execute with DSL
            result = self.dsl.execute(line)
            
            # Store result in memory
            if result.get('type') in ['fact', 'rule', 'query']:
                self.memory_store.store(
                    content=result,
                    item_type='response',
                    source='agent',
                    confidence=0.8
                )
            
            # Process results
            if result.get('facts'):
                for fact_result in result['facts']:
                    self.session_facts.append(fact_result)
                    print(f"{Colors.GREEN}✓ Added fact: {fact_result['description']}{Colors.END}")
                    
            if result.get('rules'):
                for rule_result in result['rules']:
                    self.session_rules.append(rule_result)
                    print(f"{Colors.BLUE}✓ Added rule: {rule_result['description']}{Colors.END}")
                    
            if result.get('queries'):
                for query_result in result['queries']:
                    self.session_queries.append(query_result)
                    self._display_query_results(query_result)
                    
            if result.get('commands'):
                for cmd_result in result['commands']:
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
        print(f"{Colors.BOLD}Egdol Enhanced Commands:{Colors.END}")
        print(f"  {Colors.GREEN}:help{Colors.END}     - Show this help")
        print(f"  {Colors.GREEN}:facts{Colors.END}    - Show all facts")
        print(f"  {Colors.GREEN}:rules{Colors.END}    - Show all rules")
        print(f"  {Colors.GREEN}:reset{Colors.END}    - Clear session")
        print(f"  {Colors.GREEN}:save{Colors.END}    - Save session to file")
        print(f"  {Colors.GREEN}:load{Colors.END}    - Load session from file")
        print(f"  {Colors.GREEN}:trace{Colors.END}    - Trace query execution")
        print(f"  {Colors.GREEN}:stats{Colors.END}    - Show session statistics")
        print(f"  {Colors.GREEN}:clear{Colors.END}    - Clear screen")
        print(f"  {Colors.GREEN}:remember{Colors.END} - Remember something explicitly")
        print(f"  {Colors.GREEN}:forget{Colors.END}  - Forget memories matching pattern")
        print(f"  {Colors.GREEN}:memories{Colors.END} - List all memories")
        print(f"  {Colors.GREEN}:introspect{Colors.END} - Introspect system state")
        print(f"  {Colors.GREEN}:agents{Colors.END}  - List all agents")
        print(f"  {Colors.GREEN}:switch{Colors.END}  - Switch to agent")
        print(f"  {Colors.GREEN}:create{Colors.END}  - Create new agent")
        print(f"  {Colors.GREEN}:watchers{Colors.END} - List watchers")
        print(f"  {Colors.GREEN}:tasks{Colors.END}   - List scheduled tasks")
        print(f"  {Colors.GREEN}:quit{Colors.END}    - Exit")
        print()
        print(f"{Colors.BOLD}DSL Examples:{Colors.END}")
        print(f"  {Colors.CYAN}Alice is a human{Colors.END}")
        print(f"  {Colors.CYAN}If X is a human then X is mortal{Colors.END}")
        print(f"  {Colors.CYAN}Who is mortal?{Colors.END}")
        print(f"  {Colors.CYAN}Is Alice mortal?{Colors.END}")
        
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
        self.interpreter = Interpreter(self.engine)
        self.dsl = SimpleDSL(self.engine)
        self.session_facts = []
        self.session_rules = []
        self.session_queries = []
        print(f"{Colors.GREEN}Session reset{Colors.END}")
        
    def _save_session(self, filename: str):
        """Save session to file."""
        try:
            with open(filename, 'w') as f:
                f.write("# Egdol Enhanced Session\n\n")
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
        memory_stats = self.memory_store.get_stats()
        
        print(f"{Colors.BOLD}Session Statistics:{Colors.END}")
        print(f"  Facts: {stats['num_facts']}")
        print(f"  Rules: {stats['num_rules']}")
        print(f"  Session Facts: {len(self.session_facts)}")
        print(f"  Session Rules: {len(self.session_rules)}")
        print(f"  Session Queries: {len(self.session_queries)}")
        print(f"  Memory Items: {memory_stats['total_items']}")
        print(f"  Recent Items: {memory_stats['recent_items']}")
        
    def _remember(self, content: str):
        """Remember something explicitly."""
        if not content:
            print(f"{Colors.RED}Please provide something to remember{Colors.END}")
            return
            
        memory_id = self.memory_store.store(
            content=content,
            item_type='explicit_memory',
            source='user',
            confidence=1.0
        )
        print(f"{Colors.GREEN}✓ Remembered: {content} (ID: {memory_id}){Colors.END}")
        
    def _forget(self, pattern: str):
        """Forget memories matching pattern."""
        if not pattern:
            print(f"{Colors.RED}Please provide a pattern to forget{Colors.END}")
            return
            
        deleted_count = self.memory_store.forget(pattern=pattern)
        print(f"{Colors.GREEN}✓ Forgot {deleted_count} memories matching '{pattern}'{Colors.END}")
        
    def _list_memories(self):
        """List all memories."""
        memories = self.memory_store.search(limit=50)
        
        if not memories:
            print(f"{Colors.YELLOW}No memories found{Colors.END}")
            return
            
        print(f"{Colors.BOLD}Recent Memories:{Colors.END}")
        for memory in memories[:10]:  # Show last 10
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(memory.timestamp))
            print(f"  {memory.id}: {memory.content} ({memory.item_type}) - {timestamp}")
            
    def _introspect(self):
        """Introspect system state."""
        print(f"{Colors.BOLD}System Introspection:{Colors.END}")
        
        # Memory analysis
        memory_analysis = self.memory_inspector.analyze_memory_patterns()
        print(f"  Memory Items: {memory_analysis['total_memories']}")
        print(f"  By Type: {memory_analysis['by_type']}")
        print(f"  Avg Confidence: {memory_analysis['avg_confidence']:.2f}")
        
        # Rule analysis
        rule_usage = self.rule_inspector.analyze_rule_usage()
        print(f"  Active Rules: {len(rule_usage)}")
        
        # Agent info
        if self.current_agent:
            agent_info = self.current_agent.introspect()
            print(f"  Current Agent: {agent_info['name']}")
            print(f"  Thinking Mode: {agent_info['thinking_mode']}")
            
    def _list_agents(self):
        """List all agents."""
        agents = self.agent_manager.list_agents()
        
        if not agents:
            print(f"{Colors.YELLOW}No agents found{Colors.END}")
            return
            
        print(f"{Colors.BOLD}Available Agents:{Colors.END}")
        for agent in agents:
            status = "Active" if agent['is_active'] else "Inactive"
            print(f"  {agent['name']}: {agent['description']} ({status})")
            
    def _switch_agent(self, agent_name: str):
        """Switch to a specific agent."""
        if not agent_name:
            print(f"{Colors.RED}Please provide an agent name{Colors.END}")
            return
            
        agent = self.agent_manager.switch_agent(agent_name)
        if agent:
            self.current_agent = agent
            print(f"{Colors.GREEN}Switched to agent: {agent_name}{Colors.END}")
        else:
            print(f"{Colors.RED}Agent not found: {agent_name}{Colors.END}")
            
    def _create_agent(self, args: List[str]):
        """Create a new agent."""
        if len(args) < 1:
            print(f"{Colors.RED}Usage: :create <name> [description]{Colors.END}")
            return
            
        name = args[0]
        description = ' '.join(args[1:]) if len(args) > 1 else ""
        
        try:
            agent = self.agent_manager.create_agent(name, description)
            self.current_agent = agent
            print(f"{Colors.GREEN}Created agent: {name}{Colors.END}")
        except ValueError as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
            
    def _list_watchers(self):
        """List all watchers."""
        watchers = self.watcher_manager.get_all_watchers()
        
        if not watchers:
            print(f"{Colors.YELLOW}No watchers found{Colors.END}")
            return
            
        print(f"{Colors.BOLD}Active Watchers:{Colors.END}")
        for watcher in watchers:
            status = "Enabled" if watcher['enabled'] else "Disabled"
            print(f"  {watcher['name']}: {status} (Priority: {watcher['priority']})")
            
    def _list_tasks(self):
        """List all scheduled tasks."""
        tasks = self.scheduler.get_all_tasks()
        
        if not tasks:
            print(f"{Colors.YELLOW}No tasks found{Colors.END}")
            return
            
        print(f"{Colors.BOLD}Scheduled Tasks:{Colors.END}")
        for task in tasks:
            status = "Enabled" if task['enabled'] else "Disabled"
            print(f"  {task['name']}: {status} ({task['type']})")
            
    def _setup_autonomous_behaviors(self):
        """Set up autonomous behaviors."""
        # Add a periodic memory cleanup task
        def cleanup_memories():
            # Clean up old low-confidence memories
            cutoff_time = time.time() - 86400 * 7  # 7 days ago
            deleted = self.memory_store.forget(older_than=cutoff_time)
            if deleted > 0:
                print(f"{Colors.CYAN}Cleaned up {deleted} old memories{Colors.END}")
                
        self.scheduler.add_task(
            name="memory_cleanup",
            function=cleanup_memories,
            interval=3600,  # Every hour
            metadata={'type': 'maintenance'}
        )
        
        # Add a confidence monitoring watcher
        def check_confidence():
            # This would check for low-confidence memories
            pass
            
        self.watcher_manager.add_watcher(
            name="confidence_monitor",
            condition=check_confidence,
            action=lambda: print(f"{Colors.YELLOW}Low confidence detected{Colors.END}"),
            priority=1
        )
        
    def _cleanup(self):
        """Cleanup before exit."""
        self.scheduler.stop()
        self.agent_manager.save_all_agents()
        
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for readline."""
        commands = [
            'help', 'facts', 'rules', 'reset', 'save', 'load', 
            'trace', 'stats', 'clear', 'remember', 'forget', 'memories',
            'introspect', 'agents', 'switch', 'create', 'watchers', 'tasks', 'quit'
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
    """Main entry point for the enhanced REPL."""
    repl = EnhancedREPL()
    try:
        repl.run()
    finally:
        repl._save_history()


if __name__ == '__main__':
    main()
