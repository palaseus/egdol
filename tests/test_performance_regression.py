"""
Performance regression testing for egdol.
Ensures performance doesn't degrade over time.
"""

import unittest
import time
import statistics
import pytest
from egdol.rules_engine import RulesEngine
from egdol.interpreter import Interpreter
from egdol.parser import Term, Variable, Constant, Rule


class PerformanceRegressionTests(unittest.TestCase):
    """Performance regression tests to catch performance degradation."""

    def setUp(self):
        self.engine = RulesEngine()
        self.interp = Interpreter(self.engine)

    def test_fact_query_performance(self):
        """Fact queries should complete within performance thresholds."""
        # Add 1000 facts
        for i in range(1000):
            self.engine.add_fact(Term('fact', [Constant(str(i))]))
        
        # Benchmark fact queries
        times = []
        for _ in range(100):
            start = time.perf_counter()
            results = list(self.interp.query(Term('fact', [Variable('X')])))
            end = time.perf_counter()
            times.append(end - start)
        
        # Performance assertions
        mean_time = statistics.mean(times)
        max_time = max(times)
        
        # Mean should be under 50ms
        self.assertLess(mean_time, 0.050, f"Mean time {mean_time:.6f}s exceeds threshold")
        # Max should be under 100ms
        self.assertLess(max_time, 0.100, f"Max time {max_time:.6f}s exceeds threshold")
        # Should find all 1000 facts
        self.assertEqual(len(results), 1000)

    def test_rule_chain_performance(self):
        """Rule chains should perform within thresholds."""
        # Create rule chain: a(X), b(X), c(X), d(X), e(X)
        for i in range(1, 6):
            self.engine.add_fact(Term('a', [Constant(str(i))]))
            self.engine.add_fact(Term('b', [Constant(str(i))]))
            self.engine.add_fact(Term('c', [Constant(str(i))]))
            self.engine.add_fact(Term('d', [Constant(str(i))]))
            self.engine.add_fact(Term('e', [Constant(str(i))]))
        
        # Add chain rule
        self.engine.add_rule(Rule(
            Term('chain', [Variable('X')]),
            [Term('a', [Variable('X')]), Term('b', [Variable('X')]),
             Term('c', [Variable('X')]), Term('d', [Variable('X')]),
             Term('e', [Variable('X')])]
        ))
        
        # Benchmark chain queries
        times = []
        for _ in range(50):
            start = time.perf_counter()
            results = list(self.interp.query(Term('chain', [Variable('X')])))
            end = time.perf_counter()
            times.append(end - start)
        
        # Performance assertions
        mean_time = statistics.mean(times)
        max_time = max(times)
        
        # Mean should be under 20ms
        self.assertLess(mean_time, 0.020, f"Mean time {mean_time:.6f}s exceeds threshold")
        # Max should be under 50ms
        self.assertLess(max_time, 0.050, f"Max time {max_time:.6f}s exceeds threshold")
        # Should find 5 results
        self.assertEqual(len(results), 5)

    def test_recursive_performance(self):
        """Recursive rules should perform within thresholds."""
        # Create parent relationships
        for i in range(1, 21):
            self.engine.add_fact(Term('parent', [Constant(str(i)), Constant(str(i+1))]))
        
        # Add recursive ancestor rules
        self.engine.add_rule(Rule(
            Term('ancestor', [Variable('X'), Variable('Y')]),
            [Term('parent', [Variable('X'), Variable('Y')])]
        ))
        
        self.engine.add_rule(Rule(
            Term('ancestor', [Variable('X'), Variable('Z')]),
            [Term('parent', [Variable('X'), Variable('Y')]),
             Term('ancestor', [Variable('Y'), Variable('Z')])]
        ))
        
        # Benchmark recursive queries
        times = []
        for _ in range(20):
            start = time.perf_counter()
            results = list(self.interp.query(Term('ancestor', [Constant('1'), Variable('Y')])))
            end = time.perf_counter()
            times.append(end - start)
        
        # Performance assertions
        mean_time = statistics.mean(times)
        max_time = max(times)
        
        # Mean should be under 30ms
        self.assertLess(mean_time, 0.030, f"Mean time {mean_time:.6f}s exceeds threshold")
        # Max should be under 60ms
        self.assertLess(max_time, 0.060, f"Max time {max_time:.6f}s exceeds threshold")
        # Should find 20 results
        self.assertEqual(len(results), 20)

    def test_memory_usage_regression(self):
        """Memory usage should not grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add many facts
        for i in range(10000):
            self.engine.add_fact(Term('data', [Constant(str(i)), Constant(str(i*2))]))
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable (under 50MB for 10k facts)
        self.assertLess(memory_increase, 50.0, 
                       f"Memory increase {memory_increase:.1f}MB exceeds threshold")

    def test_unification_performance(self):
        """Complex unification should perform within thresholds."""
        # Add complex terms
        for i in range(100):
            self.engine.add_fact(Term('complex', [
                Term('nested', [Constant(str(i)), Constant(str(i*2))]),
                Term('list', [Constant(str(i)), Constant(str(i+1)), Constant(str(i+2))])
            ]))
        
        # Benchmark complex unification
        times = []
        for _ in range(50):
            start = time.perf_counter()
            results = list(self.interp.query(Term('complex', [Variable('X'), Variable('Y')])))
            end = time.perf_counter()
            times.append(end - start)
        
        # Performance assertions
        mean_time = statistics.mean(times)
        max_time = max(times)
        
        # Mean should be under 10ms
        self.assertLess(mean_time, 0.010, f"Mean time {mean_time:.6f}s exceeds threshold")
        # Max should be under 25ms
        self.assertLess(max_time, 0.025, f"Max time {max_time:.6f}s exceeds threshold")
        # Should find 100 results
        self.assertEqual(len(results), 100)

    def test_concurrent_query_performance(self):
        """Multiple concurrent queries should not degrade performance significantly."""
        # Add facts
        for i in range(1000):
            self.engine.add_fact(Term('fact', [Constant(str(i))]))
        
        # Benchmark single query
        single_times = []
        for _ in range(10):
            start = time.perf_counter()
            results = list(self.interp.query(Term('fact', [Variable('X')])))
            end = time.perf_counter()
            single_times.append(end - start)
        
        single_mean = statistics.mean(single_times)
        
        # Benchmark multiple queries in sequence (simulating concurrent load)
        multi_times = []
        for _ in range(10):
            start = time.perf_counter()
            for _ in range(5):  # 5 queries in sequence
                results = list(self.interp.query(Term('fact', [Variable('X')])))
            end = time.perf_counter()
            multi_times.append(end - start)
        
        multi_mean = statistics.mean(multi_times)
        
        # Multi-query time should not be more than 10x single query time (very lenient)
        self.assertLess(multi_mean, single_mean * 10.0,
                       f"Multi-query performance degraded: {multi_mean:.6f}s vs {single_mean:.6f}s")

    def test_large_dataset_performance(self):
        """Performance with large datasets should remain reasonable."""
        # Add large number of facts
        for i in range(5000):
            self.engine.add_fact(Term('large', [Constant(str(i)), Constant(str(i*3))]))
        
        # Benchmark with large dataset
        times = []
        for _ in range(20):
            start = time.perf_counter()
            results = list(self.interp.query(Term('large', [Variable('X'), Variable('Y')])))
            end = time.perf_counter()
            times.append(end - start)
        
        # Performance assertions
        mean_time = statistics.mean(times)
        max_time = max(times)
        
        # Mean should be under 100ms even with large dataset
        self.assertLess(mean_time, 0.100, f"Mean time {mean_time:.6f}s exceeds threshold")
        # Max should be under 200ms
        self.assertLess(max_time, 0.200, f"Max time {max_time:.6f}s exceeds threshold")
        # Should find 5000 results
        self.assertEqual(len(results), 5000)

    def test_rule_indexing_performance(self):
        """Rule indexing should improve query performance."""
        # Add many rules with different predicates
        for i in range(100):
            self.engine.add_rule(Rule(
                Term(f'pred_{i}', [Variable('X')]),
                [Term('base', [Variable('X')])]
            ))
        
        # Add base facts
        for i in range(50):
            self.engine.add_fact(Term('base', [Constant(str(i))]))
        
        # Benchmark specific predicate queries (should be fast due to indexing)
        times = []
        for i in range(20):
            start = time.perf_counter()
            results = list(self.interp.query(Term('pred_0', [Variable('X')])))
            end = time.perf_counter()
            times.append(end - start)
        
        # Performance assertions
        mean_time = statistics.mean(times)
        
        # Should be fast due to indexing
        self.assertLess(mean_time, 0.010, f"Indexed query time {mean_time:.6f}s exceeds threshold")
        # Should find 50 results
        self.assertEqual(len(results), 50)

    def test_profiling_overhead(self):
        """Profiling should not add excessive overhead."""
        # Add facts
        for i in range(1000):
            self.engine.add_fact(Term('fact', [Constant(str(i))]))
        
        # Benchmark without profiling
        self.interp.profile = False
        no_prof_times = []
        for _ in range(10):
            start = time.perf_counter()
            results = list(self.interp.query(Term('fact', [Variable('X')])))
            end = time.perf_counter()
            no_prof_times.append(end - start)
        
        # Benchmark with profiling
        self.interp.profile = True
        prof_times = []
        for _ in range(10):
            start = time.perf_counter()
            results = list(self.interp.query(Term('fact', [Variable('X')])))
            end = time.perf_counter()
            prof_times.append(end - start)
        
        # Profiling overhead should be reasonable (less than 50% increase)
        no_prof_mean = statistics.mean(no_prof_times)
        prof_mean = statistics.mean(prof_times)
        overhead = (prof_mean - no_prof_mean) / no_prof_mean
        
        self.assertLess(overhead, 0.5, f"Profiling overhead {overhead:.2%} exceeds threshold")


class PerformanceBenchmarkTests(unittest.TestCase):
    """Benchmark tests for detailed performance analysis."""

    def setUp(self):
        self.engine = RulesEngine()
        self.interp = Interpreter(self.engine)

    def test_fact_query_benchmark(self):
        """Benchmark fact query performance."""
        # Add facts
        for i in range(1000):
            self.engine.add_fact(Term('fact', [Constant(str(i))]))
        
        # Benchmark
        import time
        start_time = time.perf_counter()
        results = list(self.interp.query(Term('fact', [Variable('X')])))
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        self.assertEqual(len(results), 1000)
        self.assertLess(execution_time, 1.0, f"Execution time {execution_time:.3f}s too slow")

    def test_rule_chain_benchmark(self):
        """Benchmark rule chain performance."""
        # Setup rule chain
        for i in range(1, 6):
            self.engine.add_fact(Term('a', [Constant(str(i))]))
            self.engine.add_fact(Term('b', [Constant(str(i))]))
            self.engine.add_fact(Term('c', [Constant(str(i))]))
        
        self.engine.add_rule(Rule(
            Term('chain', [Variable('X')]),
            [Term('a', [Variable('X')]), Term('b', [Variable('X')]), Term('c', [Variable('X')])]
        ))
        
        # Benchmark
        import time
        start_time = time.perf_counter()
        results = list(self.interp.query(Term('chain', [Variable('X')])))
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        self.assertEqual(len(results), 5)
        self.assertLess(execution_time, 1.0, f"Execution time {execution_time:.3f}s too slow")

    def test_unification_benchmark(self):
        """Benchmark unification performance."""
        # Add complex terms
        for i in range(100):
            self.engine.add_fact(Term('complex', [
                Term('nested', [Constant(str(i)), Constant(str(i*2))]),
                Term('list', [Constant(str(i)), Constant(str(i+1))])
            ]))
        
        # Benchmark
        import time
        start_time = time.perf_counter()
        results = list(self.interp.query(Term('complex', [Variable('X'), Variable('Y')])))
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        self.assertEqual(len(results), 100)
        self.assertLess(execution_time, 1.0, f"Execution time {execution_time:.3f}s too slow")


if __name__ == '__main__':
    unittest.main()
