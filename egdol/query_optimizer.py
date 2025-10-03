"""
Advanced query optimization engine for egdol.
Implements cost-based optimization, rule reordering, and indexing strategies.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import heapq
import time
from .parser import Term, Variable, Rule
from .rules_engine import RulesEngine


# Forward declarations for classes defined later in the file
class OptimizedQuery:
    pass


class QueryOptimizer:
    """Advanced query optimizer with cost-based optimization."""
    
    def __init__(self, engine: RulesEngine):
        self.engine = engine
        self.statistics = QueryStatistics()
        self.cost_model = CostModel()
        self.index_manager = IndexManager()
        
    def optimize_query(self, goal: Term):
        """Optimize a query using cost-based optimization."""
        # Analyze query structure
        analysis = self._analyze_query(goal)
        
        # Generate optimization strategies
        strategies = self._generate_strategies(analysis)
        
        # Select best strategy
        best_strategy = self._select_best_strategy(strategies)
        
        # Create optimized query (simplified for now)
        return SimpleOptimizedQuery(goal, best_strategy, self.engine)
        
    def _analyze_query(self, goal: Term) -> QueryAnalysis:
        """Analyze query structure and characteristics."""
        variables = self._extract_variables(goal)
        predicates = self._extract_predicates(goal)
        
        # Estimate selectivity for each predicate
        selectivity = {}
        for pred in predicates:
            selectivity[pred] = self._estimate_selectivity(pred)
            
        # Analyze variable dependencies
        dependencies = self._analyze_dependencies(goal)
        
        # Estimate cardinality
        cardinality = self._estimate_cardinality(goal, selectivity)
        
        return QueryAnalysis(
            variables=variables,
            predicates=predicates,
            selectivity=selectivity,
            dependencies=dependencies,
            cardinality=cardinality
        )
        
    def _extract_variables(self, goal: Term) -> Set[str]:
        """Extract all variables from goal."""
        variables = set()
        self._extract_variables_recursive(goal, variables)
        return variables
        
    def _extract_variables_recursive(self, term: Term, variables: Set[str]) -> None:
        """Recursively extract variables."""
        for arg in term.args:
            if isinstance(arg, Variable):
                variables.add(arg.name)
            elif isinstance(arg, Term):
                self._extract_variables_recursive(arg, variables)
                
    def _extract_predicates(self, goal: Term) -> List[str]:
        """Extract predicate names from goal."""
        predicates = []
        self._extract_predicates_recursive(goal, predicates)
        return predicates
        
    def _extract_predicates_recursive(self, term: Term, predicates: List[str]) -> None:
        """Recursively extract predicates."""
        predicates.append(term.name)
        for arg in term.args:
            if isinstance(arg, Term):
                self._extract_predicates_recursive(arg, predicates)
                
    def _estimate_selectivity(self, predicate: str) -> float:
        """Estimate selectivity of predicate (0.0 to 1.0)."""
        # Get statistics for predicate
        stats = self.statistics.get_predicate_stats(predicate)
        
        if stats.total_facts == 0:
            return 1.0
            
        # Estimate based on number of facts and rules
        fact_selectivity = 1.0 / max(1, stats.total_facts)
        rule_selectivity = 1.0 / max(1, stats.total_rules)
        
        # Combine selectivities
        return min(fact_selectivity, rule_selectivity)
        
    def _analyze_dependencies(self, goal: Term) -> Dict[str, Set[str]]:
        """Analyze variable dependencies in goal."""
        dependencies = defaultdict(set)
        
        # Find shared variables between predicates
        predicate_vars = {}
        self._collect_predicate_variables(goal, predicate_vars)
        
        for pred1, vars1 in predicate_vars.items():
            for pred2, vars2 in predicate_vars.items():
                if pred1 != pred2:
                    shared_vars = vars1.intersection(vars2)
                    if shared_vars:
                        dependencies[pred1].add(pred2)
                        
        return dict(dependencies)
        
    def _collect_predicate_variables(self, term: Term, predicate_vars: Dict[str, Set[str]]) -> None:
        """Collect variables for each predicate."""
        vars_in_term = set()
        for arg in term.args:
            if isinstance(arg, Variable):
                vars_in_term.add(arg.name)
            elif isinstance(arg, Term):
                self._collect_predicate_variables(arg, predicate_vars)
                
        if vars_in_term:
            predicate_vars[term.name] = vars_in_term
            
    def _estimate_cardinality(self, goal: Term, selectivity: Dict[str, float]) -> int:
        """Estimate result cardinality."""
        if not selectivity:
            return 1
            
        # Use product of selectivities
        total_selectivity = 1.0
        for pred, sel in selectivity.items():
            total_selectivity *= sel
            
        # Estimate based on total facts in engine
        total_facts = len(self.engine.facts)
        estimated_cardinality = int(total_facts * total_selectivity)
        
        return max(1, estimated_cardinality)
        
    def _generate_strategies(self, analysis: QueryAnalysis) -> List[OptimizationStrategy]:
        """Generate optimization strategies."""
        strategies = []
        
        # Strategy 1: Predicate reordering based on selectivity
        strategies.append(self._create_selectivity_strategy(analysis))
        
        # Strategy 2: Index-based optimization
        strategies.append(self._create_index_strategy(analysis))
        
        # Strategy 3: Join order optimization
        strategies.append(self._create_join_order_strategy(analysis))
        
        # Strategy 4: Parallel execution strategy
        strategies.append(self._create_parallel_strategy(analysis))
        
        return strategies
        
    def _create_selectivity_strategy(self, analysis: QueryAnalysis) -> OptimizationStrategy:
        """Create strategy based on predicate selectivity."""
        # Sort predicates by selectivity (most selective first)
        sorted_predicates = sorted(
            analysis.predicates,
            key=lambda p: analysis.selectivity.get(p, 1.0)
        )
        
        return OptimizationStrategy(
            name="selectivity_based",
            predicate_order=sorted_predicates,
            cost_estimate=self._estimate_selectivity_cost(analysis, sorted_predicates)
        )
        
    def _create_index_strategy(self, analysis: QueryAnalysis) -> OptimizationStrategy:
        """Create strategy using available indexes."""
        indexed_predicates = []
        non_indexed_predicates = []
        
        for pred in analysis.predicates:
            if self.index_manager.has_index(pred):
                indexed_predicates.append(pred)
            else:
                non_indexed_predicates.append(pred)
                
        # Use indexed predicates first
        predicate_order = indexed_predicates + non_indexed_predicates
        
        return OptimizationStrategy(
            name="index_based",
            predicate_order=predicate_order,
            cost_estimate=self._estimate_index_cost(analysis, predicate_order)
        )
        
    def _create_join_order_strategy(self, analysis: QueryAnalysis) -> OptimizationStrategy:
        """Create strategy optimizing join order."""
        # Use dynamic programming for optimal join order
        optimal_order = self._find_optimal_join_order(analysis)
        
        return OptimizationStrategy(
            name="join_order",
            predicate_order=optimal_order,
            cost_estimate=self._estimate_join_cost(analysis, optimal_order)
        )
        
    def _create_parallel_strategy(self, analysis: QueryAnalysis) -> OptimizationStrategy:
        """Create strategy for parallel execution."""
        # Identify independent predicates that can be executed in parallel
        parallel_groups = self._identify_parallel_groups(analysis)
        
        return OptimizationStrategy(
            name="parallel",
            predicate_order=analysis.predicates,  # Order doesn't matter for parallel
            parallel_groups=parallel_groups,
            cost_estimate=self._estimate_parallel_cost(analysis, parallel_groups)
        )
        
    def _find_optimal_join_order(self, analysis: QueryAnalysis) -> List[str]:
        """Find optimal join order using dynamic programming."""
        n = len(analysis.predicates)
        if n <= 1:
            return analysis.predicates
            
        # Use dynamic programming to find optimal order
        memo = {}
        
        def dp(mask: int) -> Tuple[float, List[str]]:
            if mask == (1 << n) - 1:
                return 0.0, []
                
            if mask in memo:
                return memo[mask]
                
            best_cost = float('inf')
            best_order = []
            
            for i in range(n):
                if not (mask & (1 << i)):
                    new_mask = mask | (1 << i)
                    cost, order = dp(new_mask)
                    
                    # Add cost of joining with current predicate
                    join_cost = self._estimate_join_cost_single(
                        analysis.predicates[i], 
                        [analysis.predicates[j] for j in range(n) if mask & (1 << j)]
                    )
                    
                    total_cost = cost + join_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_order = [analysis.predicates[i]] + order
                        
            memo[mask] = (best_cost, best_order)
            return best_cost, best_order
            
        _, optimal_order = dp(0)
        return optimal_order
        
    def _identify_parallel_groups(self, analysis: QueryAnalysis) -> List[List[str]]:
        """Identify predicates that can be executed in parallel."""
        groups = []
        remaining = set(analysis.predicates)
        
        while remaining:
            # Find predicates with no dependencies
            independent = []
            for pred in remaining:
                if not any(dep in remaining for dep in analysis.dependencies.get(pred, [])):
                    independent.append(pred)
                    
            if independent:
                groups.append(independent)
                remaining -= set(independent)
            else:
                # No independent predicates, add remaining one by one
                groups.append([remaining.pop()])
                
        return groups
        
    def _select_best_strategy(self, strategies: List[OptimizationStrategy]) -> OptimizationStrategy:
        """Select the best optimization strategy."""
        return min(strategies, key=lambda s: s.cost_estimate)
        
    def _estimate_selectivity_cost(self, analysis: QueryAnalysis, order: List[str]) -> float:
        """Estimate cost for selectivity-based strategy."""
        cost = 0.0
        cumulative_selectivity = 1.0
        
        for pred in order:
            selectivity = analysis.selectivity.get(pred, 1.0)
            cost += self.cost_model.estimate_predicate_cost(pred) * cumulative_selectivity
            cumulative_selectivity *= selectivity
            
        return cost
        
    def _estimate_index_cost(self, analysis: QueryAnalysis, order: List[str]) -> float:
        """Estimate cost for index-based strategy."""
        cost = 0.0
        
        for pred in order:
            if self.index_manager.has_index(pred):
                cost += self.cost_model.estimate_indexed_cost(pred)
            else:
                cost += self.cost_model.estimate_scan_cost(pred)
                
        return cost
        
    def _estimate_join_cost(self, analysis: QueryAnalysis, order: List[str]) -> float:
        """Estimate cost for join order strategy."""
        cost = 0.0
        intermediate_size = 1.0
        
        for pred in order:
            pred_cost = self.cost_model.estimate_predicate_cost(pred)
            cost += pred_cost * intermediate_size
            
            # Update intermediate result size
            selectivity = analysis.selectivity.get(pred, 1.0)
            intermediate_size *= selectivity
            
        return cost
        
    def _estimate_parallel_cost(self, analysis: QueryAnalysis, groups: List[List[str]]) -> float:
        """Estimate cost for parallel strategy."""
        max_group_cost = 0.0
        
        for group in groups:
            group_cost = 0.0
            for pred in group:
                group_cost += self.cost_model.estimate_predicate_cost(pred)
            max_group_cost = max(max_group_cost, group_cost)
            
        return max_group_cost


class QueryStatistics:
    """Collects and maintains query execution statistics."""
    
    def __init__(self):
        self.predicate_stats: Dict[str, PredicateStats] = {}
        self.query_history: List[QueryExecution] = []
        
    def get_predicate_stats(self, predicate: str) -> PredicateStats:
        """Get statistics for a predicate."""
        if predicate not in self.predicate_stats:
            self.predicate_stats[predicate] = PredicateStats()
        return self.predicate_stats[predicate]
        
    def record_query_execution(self, query: Term, execution_time: float, 
                              result_count: int, strategy: str) -> None:
        """Record query execution statistics."""
        execution = QueryExecution(
            query=query,
            execution_time=execution_time,
            result_count=result_count,
            strategy=strategy,
            timestamp=time.time()
        )
        self.query_history.append(execution)
        
    def get_average_execution_time(self, predicate: str) -> float:
        """Get average execution time for predicate."""
        times = []
        for execution in self.query_history:
            if predicate in str(execution.query):
                times.append(execution.execution_time)
                
        return sum(times) / len(times) if times else 0.0


class CostModel:
    """Cost model for query optimization."""
    
    def __init__(self):
        self.base_costs = {
            'fact_lookup': 1.0,
            'rule_evaluation': 5.0,
            'unification': 2.0,
            'index_lookup': 0.5,
            'full_scan': 10.0
        }
        
    def estimate_predicate_cost(self, predicate: str) -> float:
        """Estimate cost for evaluating a predicate."""
        # Base cost for predicate evaluation
        base_cost = self.base_costs['rule_evaluation']
        
        # Add unification cost
        unification_cost = self.base_costs['unification']
        
        return base_cost + unification_cost
        
    def estimate_indexed_cost(self, predicate: str) -> float:
        """Estimate cost for indexed predicate."""
        return self.base_costs['index_lookup']
        
    def estimate_scan_cost(self, predicate: str) -> float:
        """Estimate cost for full table scan."""
        return self.base_costs['full_scan']


class IndexManager:
    """Manages database indexes for optimization."""
    
    def __init__(self):
        self.indexes: Dict[str, Any] = {}
        
    def has_index(self, predicate: str) -> bool:
        """Check if predicate has an index."""
        return predicate in self.indexes
        
    def create_index(self, predicate: str, index_type: str = 'btree') -> None:
        """Create index for predicate."""
        self.indexes[predicate] = {'type': index_type, 'created_at': time.time()}
        
    def drop_index(self, predicate: str) -> None:
        """Drop index for predicate."""
        if predicate in self.indexes:
            del self.indexes[predicate]


class SimpleOptimizedQuery:
    """Simplified optimized query for testing."""
    
    def __init__(self, goal: Term, strategy, engine: RulesEngine):
        self.goal = goal
        self.strategy = strategy
        self.engine = engine
        
    def execute(self) -> List[Dict[str, Any]]:
        """Execute optimized query."""
        # For now, just use standard engine query
        from .interpreter import Interpreter
        interp = Interpreter(self.engine)
        return list(interp.query(self.goal))


# Data classes for optimization
class QueryAnalysis:
    def __init__(self, variables: Set[str], predicates: List[str], 
                 selectivity: Dict[str, float], dependencies: Dict[str, Set[str]], 
                 cardinality: int):
        self.variables = variables
        self.predicates = predicates
        self.selectivity = selectivity
        self.dependencies = dependencies
        self.cardinality = cardinality


class OptimizationStrategy:
    def __init__(self, name: str, predicate_order: List[str], 
                 cost_estimate: float, parallel_groups: Optional[List[List[str]]] = None):
        self.name = name
        self.predicate_order = predicate_order
        self.cost_estimate = cost_estimate
        self.parallel_groups = parallel_groups


class PredicateStats:
    def __init__(self):
        self.total_facts = 0
        self.total_rules = 0
        self.avg_execution_time = 0.0
        self.selectivity = 1.0


class QueryExecution:
    def __init__(self, query: Term, execution_time: float, result_count: int, 
                 strategy: str, timestamp: float):
        self.query = query
        self.execution_time = execution_time
        self.result_count = result_count
        self.strategy = strategy
        self.timestamp = timestamp
