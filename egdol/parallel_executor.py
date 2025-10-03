"""
Parallel query execution engine for egdol.
Implements multi-threaded and multi-process query execution.
"""

import threading
import multiprocessing
import queue
import time
from typing import Dict, List, Set, Optional, Any, Generator, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import pickle
from .parser import Term, Variable
from .rules_engine import RulesEngine
from .interpreter import Interpreter


@dataclass
class ParallelTask:
    """Represents a task for parallel execution."""
    task_id: str
    goal: Term
    subst: Dict[str, Any]
    depth: int
    priority: int = 0


@dataclass
class ParallelResult:
    """Represents a result from parallel execution."""
    task_id: str
    result: Dict[str, Any]
    execution_time: float
    worker_id: str


class ParallelQueryExecutor:
    """Advanced parallel query executor with load balancing and optimization."""
    
    def __init__(self, engine: RulesEngine, max_workers: Optional[int] = None):
        self.engine = engine
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.active_tasks: Dict[str, ParallelTask] = {}
        self.completed_tasks: Dict[str, ParallelResult] = {}
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.load_balancer = LoadBalancer()
        
    def execute_parallel_query(self, goal: Term, max_depth: int = 100, 
                             timeout: Optional[float] = None) -> Generator[Dict[str, Any], None, None]:
        """Execute query in parallel with multiple workers."""
        start_time = time.perf_counter()
        
        # Create initial task
        initial_task = ParallelTask(
            task_id="root",
            goal=goal,
            subst={},
            depth=0,
            priority=0
        )
        
        # Submit to thread pool for parallel execution
        future = self.thread_pool.submit(self._execute_task_parallel, initial_task, max_depth)
        
        # Collect results as they become available
        try:
            while True:
                if timeout and (time.perf_counter() - start_time) > timeout:
                    break
                    
                try:
                    result = future.result(timeout=0.1)
                    if result:
                        yield result
                    else:
                        break
                except queue.Empty:
                    continue
                except Exception as e:
                    if "timeout" not in str(e).lower():
                        raise
                    break
                    
        finally:
            future.cancel()
            
    def _execute_task_parallel(self, task: ParallelTask, max_depth: int) -> Optional[Dict[str, Any]]:
        """Execute a single task in parallel."""
        if task.depth > max_depth:
            return None
            
        # Create interpreter for this task
        interp = Interpreter(self.engine)
        interp.max_depth = max_depth - task.depth
        
        try:
            # Execute the goal
            results = list(interp.query(task.goal))
            
            # Return first result (or None if no results)
            return results[0] if results else None
            
        except Exception:
            return None
            
    def execute_distributed_query(self, goal: Term, worker_nodes: List[str], 
                                 max_depth: int = 100) -> Generator[Dict[str, Any], None, None]:
        """Execute query across multiple worker nodes."""
        # This would require network communication in a real implementation
        # For now, simulate with local process pool
        
        tasks = self._partition_query(goal, len(worker_nodes))
        futures = []
        
        for i, task in enumerate(tasks):
            future = self.process_pool.submit(
                self._execute_distributed_task, 
                task, 
                max_depth,
                worker_nodes[i % len(worker_nodes)]
            )
            futures.append(future)
            
        # Collect results from all workers
        for future in as_completed(futures):
            try:
                results = future.result()
                for result in results:
                    yield result
            except Exception as e:
                # Log error but continue with other workers
                print(f"Worker error: {e}")
                
    def _partition_query(self, goal: Term, num_partitions: int) -> List[ParallelTask]:
        """Partition query into multiple tasks."""
        tasks = []
        
        # Create different variations of the goal for parallel execution
        for i in range(num_partitions):
            task = ParallelTask(
                task_id=f"partition_{i}",
                goal=goal,
                subst={},
                depth=0,
                priority=i
            )
            tasks.append(task)
            
        return tasks
        
    def _execute_distributed_task(self, task: ParallelTask, max_depth: int, 
                                 worker_node: str) -> List[Dict[str, Any]]:
        """Execute task on a specific worker node."""
        # In a real implementation, this would communicate with remote workers
        # For now, execute locally with different strategies
        
        interp = Interpreter(self.engine)
        interp.max_depth = max_depth
        
        # Apply worker-specific optimizations
        if "fast" in worker_node:
            interp.trace_level = 0
        elif "thorough" in worker_node:
            interp.trace_level = 2
            
        try:
            results = list(interp.query(task.goal))
            return results
        except Exception:
            return []
            
    def execute_pipeline_query(self, goals: List[Term], max_depth: int = 100) -> Generator[Dict[str, Any], None, None]:
        """Execute a pipeline of queries with parallel processing."""
        if not goals:
            return
            
        # Create pipeline stages (simplified)
        stages = []
        for i, goal in enumerate(goals):
            stage = {
                'stage_id': i,
                'goal': goal,
                'max_depth': max_depth
            }
            stages.append(stage)
            
        # Execute pipeline
        current_results = [{}]  # Start with empty substitution
        
        for stage in stages:
            next_results = []
            
            # Process current results in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for result in current_results:
                    future = executor.submit(self._execute_pipeline_stage, stage, result)
                    futures.append(future)
                    
                # Collect results
                for future in as_completed(futures):
                    try:
                        stage_results = future.result()
                        next_results.extend(stage_results)
                    except Exception:
                        continue
                        
            current_results = next_results
            
            # Yield intermediate results
            for result in current_results:
                yield result
                
    def _execute_pipeline_stage(self, stage, 
                                   input_subst: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a single pipeline stage."""
        interp = Interpreter(self.engine)
        interp.max_depth = stage['max_depth']
        
        try:
            # Apply input substitution to goal
            goal = self._apply_substitution(stage['goal'], input_subst)
            
            # Execute goal
            results = list(interp.query(goal))
            
            # Merge with input substitution
            merged_results = []
            for result in results:
                merged = {**input_subst, **result}
                merged_results.append(merged)
                
            return merged_results
            
        except Exception:
            return []
            
    def _apply_substitution(self, goal: Term, subst: Dict[str, Any]) -> Term:
        """Apply substitution to a goal."""
        # This is a simplified implementation
        # In practice, would need to handle complex term substitution
        return goal
        
    def execute_streaming_query(self, goal: Term, batch_size: int = 100, 
                               max_depth: int = 100) -> Generator[Dict[str, Any], None, None]:
        """Execute query with streaming results."""
        interp = Interpreter(self.engine)
        interp.max_depth = max_depth
        
        batch = []
        batch_count = 0
        
        try:
            for result in interp.query(goal):
                batch.append(result)
                
                if len(batch) >= batch_size:
                    # Process batch in parallel
                    yield from self._process_batch_parallel(batch, batch_count)
                    batch = []
                    batch_count += 1
                    
            # Process remaining results
            if batch:
                yield from self._process_batch_parallel(batch, batch_count)
                
        except Exception as e:
            # Yield any partial results
            for result in batch:
                yield result
                
    def _process_batch_parallel(self, batch: List[Dict[str, Any]], 
                               batch_id: int) -> Generator[Dict[str, Any], None, None]:
        """Process a batch of results in parallel."""
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
            futures = []
            
            for i, result in enumerate(batch):
                future = executor.submit(self._process_single_result, result, batch_id, i)
                futures.append(future)
                
            # Yield results as they complete
            for future in as_completed(futures):
                try:
                    processed_result = future.result()
                    if processed_result:
                        yield processed_result
                except Exception:
                    continue
                    
    def _process_single_result(self, result: Dict[str, Any], 
                              batch_id: int, result_id: int) -> Optional[Dict[str, Any]]:
        """Process a single result (placeholder for complex processing)."""
        # Add metadata
        result['_batch_id'] = batch_id
        result['_result_id'] = result_id
        result['_processed_at'] = time.time()
        
        return result
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for parallel execution."""
        total_tasks = len(self.completed_tasks)
        total_time = sum(result.execution_time for result in self.completed_tasks.values())
        
        metrics = {
            'total_tasks': total_tasks,
            'total_execution_time': total_time,
            'average_task_time': total_time / max(1, total_tasks),
            'worker_utilization': self._calculate_worker_utilization(),
            'throughput': total_tasks / max(0.001, total_time),
            'load_balance_score': self.load_balancer.get_balance_score()
        }
        
        return metrics
        
    def _calculate_worker_utilization(self) -> float:
        """Calculate worker utilization percentage."""
        if not self.worker_stats:
            return 0.0
            
        total_work_time = sum(stats.total_work_time for stats in self.worker_stats.values())
        total_idle_time = sum(stats.total_idle_time for stats in self.worker_stats.values())
        
        if total_work_time + total_idle_time == 0:
            return 0.0
            
        return total_work_time / (total_work_time + total_idle_time)
        
    def shutdown(self):
        """Shutdown parallel executor."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class LoadBalancer:
    """Load balancer for distributing tasks across workers."""
    
    def __init__(self):
        self.worker_loads: Dict[str, float] = {}
        self.task_distribution: Dict[str, List[str]] = {}
        
    def select_worker(self, task: ParallelTask) -> str:
        """Select best worker for task."""
        if not self.worker_loads:
            return "worker_0"
            
        # Select worker with lowest load
        best_worker = min(self.worker_loads.keys(), key=lambda w: self.worker_loads[w])
        
        # Update load
        self.worker_loads[best_worker] += self._estimate_task_load(task)
        
        return best_worker
        
    def _estimate_task_load(self, task: ParallelTask) -> float:
        """Estimate load for a task."""
        # Simple estimation based on goal complexity
        complexity = len(str(task.goal))
        return complexity / 100.0  # Normalize
        
    def get_balance_score(self) -> float:
        """Get load balance score (0.0 to 1.0, higher is better)."""
        if not self.worker_loads:
            return 1.0
            
        loads = list(self.worker_loads.values())
        if not loads:
            return 1.0
            
        # Calculate coefficient of variation (lower is better)
        mean_load = sum(loads) / len(loads)
        if mean_load == 0:
            return 1.0
            
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        std_dev = variance ** 0.5
        
        # Convert to balance score (1.0 = perfect balance)
        cv = std_dev / mean_load
        return max(0.0, 1.0 - cv)


class PipelineStage:
    """Represents a stage in a query pipeline."""
    
    def __init__(self, stage_id: int, goal: Term, max_depth: int):
        self.stage_id = stage_id
        self.goal = goal
        self.max_depth = max_depth


class WorkerStats:
    """Statistics for a worker."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.total_tasks = 0
        self.total_work_time = 0.0
        self.total_idle_time = 0.0
        self.average_task_time = 0.0
        self.error_count = 0
        
    def record_task(self, execution_time: float, success: bool = True):
        """Record a completed task."""
        self.total_tasks += 1
        self.total_work_time += execution_time
        self.average_task_time = self.total_work_time / self.total_tasks
        
        if not success:
            self.error_count += 1
            
    def record_idle(self, idle_time: float):
        """Record idle time."""
        self.total_idle_time += idle_time


class ParallelQueryOptimizer:
    """Optimizer for parallel query execution."""
    
    def __init__(self):
        self.cost_model = ParallelCostModel()
        self.partition_strategies = [
            'round_robin',
            'hash_based',
            'load_aware',
            'affinity_based'
        ]
        
    def optimize_parallel_execution(self, goal: Term, num_workers: int) -> ParallelExecutionPlan:
        """Create optimized parallel execution plan."""
        # Analyze goal for parallelization opportunities
        analysis = self._analyze_parallelization_potential(goal)
        
        # Select best partitioning strategy
        strategy = self._select_partitioning_strategy(analysis, num_workers)
        
        # Create execution plan
        plan = ParallelExecutionPlan(
            goal=goal,
            num_workers=num_workers,
            partitioning_strategy=strategy,
            estimated_cost=self._estimate_parallel_cost(analysis, num_workers)
        )
        
        return plan
        
    def _analyze_parallelization_potential(self, goal: Term) -> ParallelizationAnalysis:
        """Analyze goal for parallelization opportunities."""
        # Extract variables and predicates
        variables = self._extract_variables(goal)
        predicates = self._extract_predicates(goal)
        
        # Estimate independence
        independence_score = self._calculate_independence_score(predicates)
        
        # Estimate data distribution
        distribution_score = self._calculate_distribution_score(goal)
        
        return ParallelizationAnalysis(
            variables=variables,
            predicates=predicates,
            independence_score=independence_score,
            distribution_score=distribution_score,
            parallelization_potential=min(independence_score, distribution_score)
        )
        
    def _extract_variables(self, goal: Term) -> Set[str]:
        """Extract variables from goal."""
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
        """Extract predicates from goal."""
        predicates = []
        self._extract_predicates_recursive(goal, predicates)
        return predicates
        
    def _extract_predicates_recursive(self, term: Term, predicates: List[str]) -> None:
        """Recursively extract predicates."""
        predicates.append(term.name)
        for arg in term.args:
            if isinstance(arg, Term):
                self._extract_predicates_recursive(arg, predicates)
                
    def _calculate_independence_score(self, predicates: List[str]) -> float:
        """Calculate how independent the predicates are."""
        # Simple heuristic: more predicates = more independence
        return min(1.0, len(predicates) / 5.0)
        
    def _calculate_distribution_score(self, goal: Term) -> float:
        """Calculate how well the goal can be distributed."""
        # Simple heuristic based on goal complexity
        complexity = len(str(goal))
        return min(1.0, complexity / 100.0)
        
    def _select_partitioning_strategy(self, analysis: ParallelizationAnalysis, 
                                     num_workers: int) -> str:
        """Select best partitioning strategy."""
        if analysis.independence_score > 0.8:
            return 'round_robin'
        elif analysis.distribution_score > 0.6:
            return 'hash_based'
        else:
            return 'load_aware'
            
    def _estimate_parallel_cost(self, analysis: ParallelizationAnalysis, 
                               num_workers: int) -> float:
        """Estimate cost of parallel execution."""
        base_cost = 1.0
        parallelization_overhead = 0.1 * num_workers
        communication_cost = 0.05 * analysis.parallelization_potential
        
        return base_cost + parallelization_overhead + communication_cost


class ParallelCostModel:
    """Cost model for parallel execution."""
    
    def __init__(self):
        self.base_costs = {
            'task_creation': 0.001,
            'worker_communication': 0.005,
            'result_aggregation': 0.002,
            'load_balancing': 0.001
        }
        
    def estimate_parallel_cost(self, num_workers: int, task_count: int) -> float:
        """Estimate cost of parallel execution."""
        base_cost = self.base_costs['task_creation'] * task_count
        communication_cost = self.base_costs['worker_communication'] * num_workers
        aggregation_cost = self.base_costs['result_aggregation'] * task_count
        
        return base_cost + communication_cost + aggregation_cost


@dataclass
class ParallelExecutionPlan:
    """Plan for parallel query execution."""
    goal: Term
    num_workers: int
    partitioning_strategy: str
    estimated_cost: float


@dataclass
class ParallelizationAnalysis:
    """Analysis of parallelization potential."""
    variables: Set[str]
    predicates: List[str]
    independence_score: float
    distribution_score: float
    parallelization_potential: float
