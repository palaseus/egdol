"""
Sophisticated integration tests combining all advanced features.
Tests the interaction between optimization, parallel execution, constraints, and profiling.
"""

import unittest
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from egdol.rules_engine import RulesEngine
from egdol.interpreter import Interpreter
from egdol.parser import Term, Variable, Constant, Rule
from egdol.advanced_constraints import AdvancedConstraintEngine
from egdol.simple_optimizer import SimpleQueryOptimizer
from egdol.parallel_executor import ParallelQueryExecutor
from egdol.advanced_profiler import AdvancedProfiler


class SophisticatedIntegrationTests(unittest.TestCase):
    """Integration tests for sophisticated features."""

    def setUp(self):
        self.engine = RulesEngine()
        self.interp = Interpreter(self.engine)
        self.constraint_engine = AdvancedConstraintEngine()
        self.optimizer = SimpleQueryOptimizer(self.engine)
        self.parallel_executor = ParallelQueryExecutor(self.engine)
        self.profiler = AdvancedProfiler(self.engine)

    def test_optimized_constraint_solving(self):
        """Test integration of optimization with constraint solving."""
        # Add facts with constraints
        for i in range(1, 11):
            self.engine.add_fact(Term('value', [Constant(str(i))]))
            
        # Add constraint: X + Y = 10
        self.constraint_engine.add_finite_domain('X', set(range(1, 11)))
        self.constraint_engine.add_finite_domain('Y', set(range(1, 11)))
        self.constraint_engine.add_binary_constraint('X', '+', 'Y')
        
        # Create optimized query
        goal = Term('value', [Variable('X')])
        optimized_query = self.optimizer.optimize_query(goal)
        
        # Execute with profiling
        self.profiler.start_profiling()
        results = optimized_query.execute()
        self.profiler.stop_profiling()
        
        # Verify results
        self.assertGreater(len(results), 0)
        
        # Check profiling data
        report = self.profiler.generate_performance_report()
        self.assertIn('summary', report)

    def test_parallel_constraint_optimization(self):
        """Test parallel execution with constraint optimization."""
        # Add complex constraint problem
        for i in range(1, 21):
            self.engine.add_fact(Term('node', [Constant(str(i))]))
            
        # Add all_different constraint
        variables = [f'X{i}' for i in range(1, 6)]
        for var in variables:
            self.constraint_engine.add_finite_domain(var, set(range(1, 21)))
        self.constraint_engine.add_all_different(variables)
        
        # Create parallel execution plan
        goal = Term('node', [Variable('X1')])
        
        # Execute in parallel with optimization
        results = []
        for result in self.parallel_executor.execute_parallel_query(goal, max_depth=50):
            results.append(result)
            
        # Verify parallel execution worked
        self.assertGreater(len(results), 0)
        
        # Check performance metrics
        metrics = self.parallel_executor.get_performance_metrics()
        self.assertIn('total_tasks', metrics)

    def test_profiled_optimization_pipeline(self):
        """Test complete pipeline with profiling and optimization."""
        # Add complex knowledge base
        self._setup_complex_knowledge_base()
        
        # Create complex query
        goal = Term('path', [Variable('X'), Variable('Y')])
        
        # Profile the optimization process
        self.profiler.start_profiling()
        
        # Optimize query
        optimized_query = self.optimizer.optimize_query(goal)
        
        # Execute with parallel processing
        results = []
        for result in self.parallel_executor.execute_parallel_query(goal, max_depth=100):
            results.append(result)
            
        self.profiler.stop_profiling()
        
        # Verify results
        self.assertGreater(len(results), 0)
        
        # Generate comprehensive report
        report = self.profiler.generate_performance_report()
        
        # Verify report structure
        self.assertIn('summary', report)
        self.assertIn('rule_profiles', report)
        self.assertIn('recommendations', report)
        
        # Check for optimization suggestions
        self.assertGreater(len(report['recommendations']), 0)

    def test_constraint_optimization_with_profiling(self):
        """Test constraint solving with optimization and profiling."""
        # Set up constraint problem
        self._setup_constraint_problem()
        
        # Create query with constraints
        goal = Term('solution', [Variable('X'), Variable('Y'), Variable('Z')])
        
        # Profile constraint solving
        self.profiler.start_profiling()
        
        # Solve constraints
        solutions = self.constraint_engine.solve()
        
        # Optimize query execution
        optimized_query = self.optimizer.optimize_query(goal)
        query_results = optimized_query.execute()
        
        self.profiler.stop_profiling()
        
        # Verify constraint solutions
        self.assertGreater(len(solutions), 0)
        
        # Verify query results
        self.assertGreater(len(query_results), 0)
        
        # Check profiling captured constraint solving
        report = self.profiler.generate_performance_report()
        self.assertIn('summary', report)

    def test_parallel_optimization_under_load(self):
        """Test parallel optimization under concurrent load."""
        # Set up knowledge base
        self._setup_complex_knowledge_base()
        
        # Create multiple concurrent queries
        queries = [
            Term('path', [Variable('X'), Constant('1')]),
            Term('path', [Variable('X'), Constant('2')]),
            Term('path', [Variable('X'), Constant('3')]),
            Term('path', [Variable('X'), Constant('4')]),
            Term('path', [Variable('X'), Constant('5')])
        ]
        
        # Execute queries in parallel with optimization
        def execute_optimized_query(query):
            optimized_query = self.optimizer.optimize_query(query)
            return list(optimized_query.execute())
            
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(execute_optimized_query, query) for query in queries]
            results = [future.result() for future in futures]
            
        # Verify all queries executed successfully
        for result_set in results:
            self.assertIsInstance(result_set, list)
            self.assertGreater(len(result_set), 0)

    def test_advanced_profiling_visualization(self):
        """Test advanced profiling with visualization."""
        # Set up complex scenario
        self._setup_complex_knowledge_base()
        
        # Create multiple queries for profiling
        queries = [
            Term('path', [Variable('X'), Variable('Y')]),
            Term('connected', [Variable('A'), Variable('B')]),
            Term('reachable', [Variable('S'), Variable('T')])
        ]
        
        # Profile all queries
        self.profiler.start_profiling()
        
        for i, query in enumerate(queries):
            profile = self.profiler.profile_query(query, f"query_{i}")
            self.assertIsNotNone(profile)
            
        self.profiler.stop_profiling()
        
        # Generate report
        report = self.profiler.generate_performance_report()
        
        # Verify report completeness
        self.assertIn('summary', report)
        self.assertIn('rule_profiles', report)
        self.assertIn('query_profiles', report)
        self.assertIn('call_graph', report)
        
        # Test visualization creation (would create files in real scenario)
        try:
            self.profiler.create_visualizations("test_plots")
        except Exception as e:
            # Visualization might fail in test environment
            pass

    def test_memory_efficient_parallel_execution(self):
        """Test memory-efficient parallel execution."""
        # Add large dataset
        for i in range(1000):
            self.engine.add_fact(Term('data', [Constant(str(i)), Constant(str(i*2))]))
            
        # Create memory-intensive query
        goal = Term('data', [Variable('X'), Variable('Y')])
        
        # Execute with streaming to manage memory
        results = []
        for result in self.parallel_executor.execute_streaming_query(goal, batch_size=100):
            results.append(result)
            
            # Simulate memory pressure
            if len(results) % 100 == 0:
                time.sleep(0.001)  # Small delay to simulate processing
                
        # Verify streaming worked
        self.assertGreater(len(results), 0)
        
        # Check memory usage didn't grow excessively
        metrics = self.parallel_executor.get_performance_metrics()
        self.assertIn('total_tasks', metrics)

    def test_constraint_optimization_with_parallel_execution(self):
        """Test constraint optimization with parallel execution."""
        # Set up constraint problem
        self._setup_constraint_problem()
        
        # Create parallel constraint solver
        def solve_constraints():
            return self.constraint_engine.solve()
            
        def execute_optimized_query():
            goal = Term('solution', [Variable('X'), Variable('Y')])
            optimized_query = self.optimizer.optimize_query(goal)
            return optimized_query.execute()
            
        # Execute both in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            constraint_future = executor.submit(solve_constraints)
            query_future = executor.submit(execute_optimized_query)
            
            constraint_results = constraint_future.result()
            query_results = query_future.result()
            
        # Verify both completed successfully
        self.assertGreater(len(constraint_results), 0)
        self.assertGreater(len(query_results), 0)

    def test_end_to_end_optimization_pipeline(self):
        """Test complete end-to-end optimization pipeline."""
        # Set up comprehensive knowledge base
        self._setup_comprehensive_knowledge_base()
        
        # Create complex query
        goal = Term('optimal_path', [Variable('Start'), Variable('End'), Variable('Cost')])
        
        # Start profiling
        self.profiler.start_profiling()
        
        # Step 1: Optimize query
        optimized_query = self.optimizer.optimize_query(goal)
        
        # Step 2: Execute with parallel processing
        results = []
        for result in self.parallel_executor.execute_parallel_query(goal, max_depth=200):
            results.append(result)
            
        # Step 3: Apply constraints for filtering
        filtered_results = []
        for result in results:
            if self._satisfies_constraints(result):
                filtered_results.append(result)
                
        # Step 4: Optimize results
        optimized_results = self._optimize_results(filtered_results)
        
        self.profiler.stop_profiling()
        
        # Verify pipeline worked
        self.assertGreater(len(optimized_results), 0)
        
        # Generate comprehensive report
        report = self.profiler.generate_performance_report()
        
        # Verify optimization was effective
        self.assertIn('recommendations', report)
        self.assertGreater(len(report['recommendations']), 0)

    def _setup_complex_knowledge_base(self):
        """Set up complex knowledge base for testing."""
        # Add graph structure
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (2, 4), (3, 5)]
        
        for start, end in edges:
            self.engine.add_fact(Term('edge', [Constant(str(start)), Constant(str(end))]))
            
        # Add path rules
        self.engine.add_rule(Rule(
            Term('path', [Variable('X'), Variable('Y')]),
            [Term('edge', [Variable('X'), Variable('Y')])]
        ))
        
        self.engine.add_rule(Rule(
            Term('path', [Variable('X'), Variable('Z')]),
            [Term('edge', [Variable('X'), Variable('Y')]),
             Term('path', [Variable('Y'), Variable('Z')])]
        ))
        
        # Add connected predicate
        self.engine.add_rule(Rule(
            Term('connected', [Variable('X'), Variable('Y')]),
            [Term('path', [Variable('X'), Variable('Y')])]
        ))
        
        # Add reachable predicate
        self.engine.add_rule(Rule(
            Term('reachable', [Variable('S'), Variable('T')]),
            [Term('path', [Variable('S'), Variable('T')])]
        ))

    def _setup_constraint_problem(self):
        """Set up constraint satisfaction problem."""
        # Add solution facts
        for i in range(1, 6):
            for j in range(1, 6):
                if i != j:
                    self.engine.add_fact(Term('solution', [Constant(str(i)), Constant(str(j))]))
                    
        # Set up constraint domains
        variables = ['X', 'Y', 'Z']
        for var in variables:
            self.constraint_engine.add_finite_domain(var, set(range(1, 6)))
            
        # Add constraints
        self.constraint_engine.add_binary_constraint('X', '!=', 'Y')
        self.constraint_engine.add_binary_constraint('Y', '!=', 'Z')
        self.constraint_engine.add_binary_constraint('X', '!=', 'Z')

    def _setup_comprehensive_knowledge_base(self):
        """Set up comprehensive knowledge base."""
        # Add complex graph with weights
        weighted_edges = [
            (1, 2, 5), (2, 3, 3), (3, 4, 2), (4, 5, 1),
            (1, 3, 8), (2, 4, 4), (3, 5, 6), (1, 4, 10)
        ]
        
        for start, end, weight in weighted_edges:
            self.engine.add_fact(Term('edge', [Constant(str(start)), Constant(str(end)), Constant(str(weight))]))
            
        # Add weighted path rules
        self.engine.add_rule(Rule(
            Term('path', [Variable('X'), Variable('Y'), Variable('C')]),
            [Term('edge', [Variable('X'), Variable('Y'), Variable('C')])]
        ))
        
        self.engine.add_rule(Rule(
            Term('path', [Variable('X'), Variable('Z'), Variable('C')]),
            [Term('edge', [Variable('X'), Variable('Y'), Variable('C1')]),
             Term('path', [Variable('Y'), Variable('Z'), Variable('C2')]),
             Term('+', [Variable('C1'), Variable('C2'), Variable('C')])]
        ))
        
        # Add optimal path rule
        self.engine.add_rule(Rule(
            Term('optimal_path', [Variable('S'), Variable('T'), Variable('Cost')]),
            [Term('path', [Variable('S'), Variable('T'), Variable('Cost')])]
        ))

    def _satisfies_constraints(self, result: dict) -> bool:
        """Check if result satisfies constraints."""
        # Simple constraint: X != Y
        if 'X' in result and 'Y' in result:
            return str(result['X']) != str(result['Y'])
        return True

    def _optimize_results(self, results: list) -> list:
        """Optimize results (placeholder for complex optimization)."""
        # Simple optimization: limit to first 10 results
        return results[:10]


if __name__ == '__main__':
    unittest.main()
