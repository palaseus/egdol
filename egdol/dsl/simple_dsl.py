"""
Simplified DSL for Egdol Interactive Assistant.
A working implementation that focuses on core functionality.
"""

import re
from typing import List, Dict, Any, Optional
from ..rules_engine import RulesEngine
from ..parser import Term, Variable, Constant, Rule, Fact
from ..interpreter import Interpreter


class SimpleDSL:
    """Simplified DSL parser and executor."""
    
    def __init__(self, engine: RulesEngine = None):
        self.engine = engine or RulesEngine()
        self.interpreter = Interpreter(self.engine)
        self.context = {}
        
    def execute(self, text: str) -> Dict[str, Any]:
        """Execute DSL text."""
        text = text.strip()
        
        # Handle commands
        if text.startswith(':'):
            return self._handle_command(text)
            
        # Handle queries
        if text.endswith('?'):
            return self._handle_query(text)
            
        # Handle rules
        if 'if' in text.lower() and 'then' in text.lower():
            return self._handle_rule(text)
            
        # Handle facts
        return self._handle_fact(text)
        
    def _handle_command(self, text: str) -> Dict[str, Any]:
        """Handle command statements."""
        parts = text[1:].split()
        command = parts[0].lower()
        args = parts[1:]
        
        if command == 'facts':
            return self._show_facts()
        elif command == 'rules':
            return self._show_rules()
        elif command == 'reset':
            return self._reset()
        elif command == 'stats':
            return self._show_stats()
        elif command == 'help':
            return self._show_help()
        else:
            return {'error': f'Unknown command: {command}'}
            
    def _handle_query(self, text: str) -> Dict[str, Any]:
        """Handle query statements."""
        text = text.lower().rstrip('?')
        
        if text.startswith('who is'):
            predicate = text[6:].strip()
            return self._query_who(predicate)
        elif text.startswith('what is'):
            subject = text[7:].strip()
            return self._query_what(subject)
        elif text.startswith('is '):
            parts = text[3:].split()
            if len(parts) >= 2:
                subject = parts[0]
                predicate = ' '.join(parts[1:])
                return self._query_is(subject, predicate)
        elif text.startswith('does '):
            parts = text[5:].split()
            if len(parts) >= 3 and parts[1] == 'have':
                subject = parts[0]
                property_ = ' '.join(parts[2:])
                return self._query_does_have(subject, property_)
                
        return {'error': f'Unknown query: {text}'}
        
    def _handle_rule(self, text: str) -> Dict[str, Any]:
        """Handle rule statements."""
        # Simple rule parsing: "if X is Y then Z is W"
        if 'if' in text and 'then' in text:
            parts = text.split('then')
            if len(parts) == 2:
                condition = parts[0].replace('if', '').strip()
                conclusion = parts[1].strip()
                
                # Parse condition and conclusion
                cond_fact = self._parse_fact(condition)
                concl_fact = self._parse_fact(conclusion)
                
                if cond_fact and concl_fact:
                    # Create rule
                    head = self._fact_to_term(concl_fact)
                    body = [self._fact_to_term(cond_fact)]
                    
                    rule = Rule(head, body)
                    self.engine.add_rule(rule)
                    
                    return {
                        'type': 'rule',
                        'rule': rule,
                        'description': f'if {condition} then {conclusion}'
                    }
                    
        return {'error': 'Could not parse rule'}
        
    def _handle_fact(self, text: str) -> Dict[str, Any]:
        """Handle fact statements."""
        fact = self._parse_fact(text)
        if fact:
            term = self._fact_to_term(fact)
            self.engine.add_fact(term)
            
            return {
                'type': 'fact',
                'fact': term,
                'description': text
            }
        else:
            return {'error': 'Could not parse fact'}
            
    def _parse_fact(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a fact from text."""
        text = text.strip()
        
        # Pattern: "X is Y"
        if ' is ' in text:
            parts = text.split(' is ', 1)
            if len(parts) == 2:
                subject = parts[0].strip()
                predicate = parts[1].strip()
                
                # Remove "a" or "an" from predicate
                if predicate.startswith('a '):
                    predicate = predicate[2:]
                elif predicate.startswith('an '):
                    predicate = predicate[3:]
                    
                return {
                    'type': 'is',
                    'subject': subject,
                    'predicate': predicate
                }
                
        # Pattern: "X has Y"
        elif ' has ' in text:
            parts = text.split(' has ', 1)
            if len(parts) == 2:
                subject = parts[0].strip()
                property_ = parts[1].strip()
                
                return {
                    'type': 'has',
                    'subject': subject,
                    'property': property_
                }
                
        return None
        
    def _fact_to_term(self, fact: Dict[str, Any]) -> Term:
        """Convert fact dict to Egdol term."""
        if fact['type'] == 'is':
            subject_term = Constant(fact['subject'])
            predicate_term = Term(fact['predicate'], [subject_term])
            return predicate_term
        elif fact['type'] == 'has':
            subject_term = Constant(fact['subject'])
            property_term = Constant(fact['property'])
            return Term('has', [subject_term, property_term])
        else:
            return Term('unknown', [])
            
    def _query_who(self, predicate: str) -> Dict[str, Any]:
        """Query who has a predicate."""
        goal = Term(predicate, [Variable('X')])
        results = list(self.interpreter.query(goal))
        
        return {
            'type': 'query',
            'query': f'who is {predicate}',
            'results': results
        }
        
    def _query_what(self, subject: str) -> Dict[str, Any]:
        """Query what a subject is."""
        # Find all predicates for this subject
        results = []
        
        # Check common predicates
        predicates = ['human', 'mortal', 'smart', 'adult', 'child']
        for pred in predicates:
            goal = Term(pred, [Constant(subject)])
            if list(self.interpreter.query(goal)):
                results.append({pred: subject})
                
        return {
            'type': 'query',
            'query': f'what is {subject}',
            'results': results
        }
        
    def _query_is(self, subject: str, predicate: str) -> Dict[str, Any]:
        """Query if subject is predicate."""
        goal = Term(predicate, [Constant(subject)])
        results = list(self.interpreter.query(goal))
        
        return {
            'type': 'query',
            'query': f'is {subject} {predicate}',
            'results': results
        }
        
    def _query_does_have(self, subject: str, property_: str) -> Dict[str, Any]:
        """Query if subject has property."""
        goal = Term('has', [Constant(subject), Constant(property_)])
        results = list(self.interpreter.query(goal))
        
        return {
            'type': 'query',
            'query': f'does {subject} have {property_}',
            'results': results
        }
        
    def _show_facts(self) -> Dict[str, Any]:
        """Show all facts."""
        stats = self.engine.stats()
        return {
            'type': 'command',
            'command': 'facts',
            'count': stats['num_facts']
        }
        
    def _show_rules(self) -> Dict[str, Any]:
        """Show all rules."""
        stats = self.engine.stats()
        return {
            'type': 'command',
            'command': 'rules',
            'count': stats['num_rules']
        }
        
    def _reset(self) -> Dict[str, Any]:
        """Reset the engine."""
        self.engine = RulesEngine()
        self.interpreter = Interpreter(self.engine)
        return {
            'type': 'command',
            'command': 'reset',
            'message': 'Engine reset'
        }
        
    def _show_stats(self) -> Dict[str, Any]:
        """Show statistics."""
        stats = self.engine.stats()
        return {
            'type': 'command',
            'command': 'stats',
            'stats': stats
        }
        
    def _show_help(self) -> Dict[str, Any]:
        """Show help."""
        return {
            'type': 'command',
            'command': 'help',
            'help': [
                'Facts: "Alice is a human"',
                'Rules: "if X is a human then X is mortal"',
                'Queries: "who is mortal?", "is Alice human?"',
                'Commands: ":facts", ":rules", ":reset", ":stats"'
            ]
        }


class SimpleREPL:
    """Simple REPL for the DSL."""
    
    def __init__(self):
        self.dsl = SimpleDSL()
        
    def run(self):
        """Run the REPL."""
        print("🤖 Egdol Simple DSL Assistant")
        print("Type 'help' for examples, 'quit' to exit")
        print()
        
        while True:
            try:
                line = input("egdol> ").strip()
                
                if not line:
                    continue
                    
                if line.lower() in ['quit', 'exit']:
                    print("Goodbye! 👋")
                    break
                    
                if line.lower() == 'help':
                    self._show_help()
                    continue
                    
                # Execute the line
                result = self.dsl.execute(line)
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
                continue
            except EOFError:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    def _display_result(self, result: Dict[str, Any]):
        """Display execution result."""
        if 'error' in result:
            print(f"❌ {result['error']}")
        elif result['type'] == 'fact':
            print(f"✅ Added fact: {result['description']}")
        elif result['type'] == 'rule':
            print(f"✅ Added rule: {result['description']}")
        elif result['type'] == 'query':
            results = result['results']
            if results:
                print(f"🔍 {result['query']}:")
                for i, res in enumerate(results, 1):
                    print(f"  {i}. {res}")
            else:
                print(f"🔍 {result['query']}: No results")
        elif result['type'] == 'command':
            if result['command'] == 'help':
                for line in result['help']:
                    print(f"  {line}")
            elif result['command'] == 'stats':
                stats = result['stats']
                print(f"📊 Facts: {stats['num_facts']}, Rules: {stats['num_rules']}")
            else:
                print(f"✅ {result.get('message', 'Command executed')}")
                
    def _show_help(self):
        """Show help information."""
        print("Egdol DSL Examples:")
        print("  Facts: Alice is a human")
        print("  Rules: if X is a human then X is mortal")
        print("  Queries: who is mortal?, is Alice human?")
        print("  Commands: :facts, :rules, :reset, :stats")


def main():
    """Main entry point."""
    repl = SimpleREPL()
    repl.run()


if __name__ == '__main__':
    main()
