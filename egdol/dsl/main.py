"""
Main entry point for the Egdol DSL Interactive Assistant.
"""

import sys
import argparse
from .repl import DSLREPL
from ..rules_engine import RulesEngine
from .translator import DSLExecutor


def main():
    """Main entry point for the DSL assistant."""
    parser = argparse.ArgumentParser(
        description="Egdol Interactive DSL Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  egdol-dsl                    # Start interactive REPL
  egdol-dsl --file script.dsl  # Execute DSL script
  egdol-dsl --web              # Start web interface
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        help='Execute DSL script from file'
    )
    
    parser.add_argument(
        '--web', '-w',
        action='store_true',
        help='Start web interface'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.web:
        start_web_interface()
    elif args.file:
        execute_script(args.file, args.verbose)
    else:
        start_repl()


def start_repl():
    """Start the interactive REPL."""
    repl = DSLREPL()
    repl.run()


def execute_script(filename: str, verbose: bool = False):
    """Execute a DSL script from file."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        engine = RulesEngine()
        executor = DSLExecutor(engine)
        
        if verbose:
            print(f"Executing script: {filename}")
            
        results = executor.execute(content)
        
        # Display results
        if results['facts']:
            print(f"Added {len(results['facts'])} facts")
        if results['rules']:
            print(f"Added {len(results['rules'])} rules")
        if results['queries']:
            print(f"Executed {len(results['queries'])} queries")
            for query in results['queries']:
                print(f"Query: {query['description']}")
                if query['results']:
                    for result in query['results']:
                        print(f"  Result: {result}")
                else:
                    print("  No results")
                    
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing script: {e}")
        sys.exit(1)


def start_web_interface():
    """Start the web interface (placeholder)."""
    print("Web interface not yet implemented.")
    print("Use the REPL interface for now: egdol-dsl")


if __name__ == '__main__':
    main()
