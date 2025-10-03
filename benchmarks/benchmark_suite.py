#!/usr/bin/env python3
"""
Performance benchmark suite for egdol.
Run with: python benchmarks/benchmark_suite.py
"""

import time
import statistics
from egdol.rules_engine import RulesEngine
from egdol.interpreter import Interpreter
from egdol.parser import Term, Variable, Constant, Rule


def benchmark_fact_queries():
    """Benchmark simple fact queries."""
    print("=== Fact Query Benchmarks ===")
    
    engine = RulesEngine()
    interp = Interpreter(engine)
    
    # Add facts
    for i in range(1000):
        engine.add_fact(Term('fact', [Constant(str(i))]))
    
    # Benchmark queries
    times = []
    for _ in range(100):
        start = time.perf_counter()
        results = list(interp.query(Term('fact', [Variable('X')])))
        end = time.perf_counter()
        times.append(end - start)
    
    print(f"Fact queries (1000 facts, 100 queries):")
    print(f"  Mean: {statistics.mean(times):.6f}s")
    print(f"  Median: {statistics.median(times):.6f}s")
    print(f"  Results: {len(results)}")


def benchmark_rule_chains():
    """Benchmark rule chain performance."""
    print("\n=== Rule Chain Benchmarks ===")
    
    engine = RulesEngine()
    interp = Interpreter(engine)
    
    # Create chain: a(1), b(1), c(1), d(1), e(1)
    # chain(X) :- a(X), b(X), c(X), d(X), e(X).
    for i in range(1, 6):
        engine.add_fact(Term('a', [Constant(str(i))]))
        engine.add_fact(Term('b', [Constant(str(i))]))
        engine.add_fact(Term('c', [Constant(str(i))]))
        engine.add_fact(Term('d', [Constant(str(i))]))
        engine.add_fact(Term('e', [Constant(str(i))]))
    
    engine.add_rule(Rule(
        Term('chain', [Variable('X')]),
        [Term('a', [Variable('X')]), Term('b', [Variable('X')]), 
         Term('c', [Variable('X')]), Term('d', [Variable('X')]), 
         Term('e', [Variable('X')])]
    ))
    
    times = []
    for _ in range(50):
        start = time.perf_counter()
        results = list(interp.query(Term('chain', [Variable('X')])))
        end = time.perf_counter()
        times.append(end - start)
    
    print(f"Rule chain queries (5 facts × 5 predicates, 50 queries):")
    print(f"  Mean: {statistics.mean(times):.6f}s")
    print(f"  Median: {statistics.median(times):.6f}s")
    print(f"  Results: {len(results)}")


def benchmark_recursive_rules():
    """Benchmark recursive rule performance."""
    print("\n=== Recursive Rule Benchmarks ===")
    
    engine = RulesEngine()
    interp = Interpreter(engine)
    
    # Create parent relationships
    for i in range(1, 21):
        engine.add_fact(Term('parent', [Constant(str(i)), Constant(str(i+1))]))
    
    # ancestor(X,Y) :- parent(X,Y).
    engine.add_rule(Rule(
        Term('ancestor', [Variable('X'), Variable('Y')]),
        [Term('parent', [Variable('X'), Variable('Y')])]
    ))
    
    # ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z).
    engine.add_rule(Rule(
        Term('ancestor', [Variable('X'), Variable('Z')]),
        [Term('parent', [Variable('X'), Variable('Y')]), 
         Term('ancestor', [Variable('Y'), Variable('Z')])]
    ))
    
    times = []
    for _ in range(20):
        start = time.perf_counter()
        results = list(interp.query(Term('ancestor', [Constant('1'), Variable('Y')])))
        end = time.perf_counter()
        times.append(end - start)
    
    print(f"Recursive ancestor queries (20 levels, 20 queries):")
    print(f"  Mean: {statistics.mean(times):.6f}s")
    print(f"  Median: {statistics.median(times):.6f}s")
    print(f"  Results: {len(results)}")


def benchmark_unification():
    """Benchmark unification performance."""
    print("\n=== Unification Benchmarks ===")
    
    engine = RulesEngine()
    interp = Interpreter(engine)
    
    # Add complex terms
    for i in range(100):
        engine.add_fact(Term('complex', [
            Term('nested', [Constant(str(i)), Constant(str(i*2))]),
            Term('list', [Constant(str(i)), Constant(str(i+1)), Constant(str(i+2))])
        ]))
    
    times = []
    for _ in range(50):
        start = time.perf_counter()
        results = list(interp.query(Term('complex', [Variable('X'), Variable('Y')])))
        end = time.perf_counter()
        times.append(end - start)
    
    print(f"Complex unification (100 facts, 50 queries):")
    print(f"  Mean: {statistics.mean(times):.6f}s")
    print(f"  Median: {statistics.median(times):.6f}s")
    print(f"  Results: {len(results)}")


def benchmark_memory_usage():
    """Benchmark memory usage with large datasets."""
    print("\n=== Memory Usage Benchmarks ===")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    engine = RulesEngine()
    interp = Interpreter(engine)
    
    # Add many facts
    for i in range(10000):
        engine.add_fact(Term('data', [Constant(str(i)), Constant(str(i*2))]))
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Memory usage with 10,000 facts:")
    print(f"  Initial: {initial_memory:.1f} MB")
    print(f"  Peak: {peak_memory:.1f} MB")
    print(f"  Increase: {peak_memory - initial_memory:.1f} MB")


if __name__ == '__main__':
    print("egdol Performance Benchmark Suite")
    print("=" * 50)
    
    try:
        benchmark_fact_queries()
        benchmark_rule_chains()
        benchmark_recursive_rules()
        benchmark_unification()
        benchmark_memory_usage()
        
        print("\n" + "=" * 50)
        print("Benchmark suite completed successfully!")
        
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
