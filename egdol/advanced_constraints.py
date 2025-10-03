"""
Advanced constraint solving capabilities for egdol.
Implements sophisticated constraint propagation and solving algorithms.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, deque
import heapq
from .parser import Term, Variable, Constant


class ConstraintPropagator:
    """Advanced constraint propagator with arc consistency and domain reduction."""
    
    def __init__(self):
        self.domains: Dict[str, Set[Any]] = {}
        self.constraints: List[Tuple[str, str, str]] = []  # (var1, op, var2)
        self.arcs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.support: Dict[Tuple[str, str, Any], Set[Any]] = defaultdict(set)
        
    def add_domain(self, var: str, domain: Set[Any]) -> None:
        """Add or update domain for a variable."""
        self.domains[var] = set(domain)
        
    def add_constraint(self, var1: str, op: str, var2: str) -> None:
        """Add binary constraint between variables."""
        self.constraints.append((var1, op, var2))
        self.arcs[var1].append((var2, op))
        self.arcs[var2].append((var1, self._reverse_op(op)))
        
    def _reverse_op(self, op: str) -> str:
        """Get reverse operation for constraint."""
        reverse_ops = {
            '=': '=',
            '!=': '!=',
            '<': '>',
            '>': '<',
            '<=': '>=',
            '>=': '<='
        }
        return reverse_ops.get(op, op)
        
    def propagate_arc_consistency(self) -> bool:
        """Apply arc consistency propagation."""
        queue = deque()
        
        # Initialize queue with all arcs
        for var1, var2, op in self.constraints:
            queue.append((var1, var2, op))
            queue.append((var2, var1, self._reverse_op(op)))
            
        while queue:
            var1, var2, op = queue.popleft()
            
            if self._revise_domain(var1, var2, op):
                if not self.domains[var1]:
                    return False  # Inconsistent
                    
                # Add all arcs involving var1 back to queue
                for neighbor, neighbor_op in self.arcs[var1]:
                    if neighbor != var2:
                        queue.append((neighbor, var1, self._reverse_op(neighbor_op)))
                        
        return True
        
    def _revise_domain(self, var1: str, var2: str, op: str) -> bool:
        """Revise domain of var1 based on constraint with var2."""
        if var1 not in self.domains or var2 not in self.domains:
            return False
            
        revised = False
        domain1 = self.domains[var1].copy()
        
        for val1 in domain1:
            if not self._has_support(val1, var2, op):
                self.domains[var1].discard(val1)
                revised = True
                
        return revised
        
    def _has_support(self, val1: Any, var2: str, op: str) -> bool:
        """Check if val1 has support in var2's domain."""
        if var2 not in self.domains:
            return False
            
        for val2 in self.domains[var2]:
            if self._satisfies_constraint(val1, val2, op):
                return True
        return False
        
    def _satisfies_constraint(self, val1: Any, val2: Any, op: str) -> bool:
        """Check if constraint is satisfied."""
        try:
            if op == '=':
                return val1 == val2
            elif op == '!=':
                return val1 != val2
            elif op == '<':
                return val1 < val2
            elif op == '>':
                return val1 > val2
            elif op == '<=':
                return val1 <= val2
            elif op == '>=':
                return val1 >= val2
            else:
                return False
        except TypeError:
            return False
            
    def get_solutions(self) -> List[Dict[str, Any]]:
        """Get all solutions using backtracking with constraint propagation."""
        if not self.propagate_arc_consistency():
            return []
            
        solutions = []
        self._backtrack_solve({}, solutions)
        return solutions
        
    def _backtrack_solve(self, assignment: Dict[str, Any], solutions: List[Dict[str, Any]]) -> None:
        """Backtracking search with constraint propagation."""
        if len(assignment) == len(self.domains):
            solutions.append(assignment.copy())
            return
            
        # Select unassigned variable with smallest domain
        unassigned = [var for var in self.domains if var not in assignment]
        if not unassigned:
            return
            
        var = min(unassigned, key=lambda v: len(self.domains[v]))
        
        # Try each value in domain
        for value in self.domains[var]:
            assignment[var] = value
            
            # Check if assignment is consistent
            if self._is_consistent(assignment):
                # Forward check: propagate constraints
                old_domains = {v: self.domains[v].copy() for v in self.domains}
                
                if self._forward_check(assignment):
                    self._backtrack_solve(assignment, solutions)
                    
                # Restore domains
                self.domains = old_domains
                
            del assignment[var]
            
    def _is_consistent(self, assignment: Dict[str, Any]) -> bool:
        """Check if current assignment is consistent."""
        for var1, var2, op in self.constraints:
            if var1 in assignment and var2 in assignment:
                if not self._satisfies_constraint(assignment[var1], assignment[var2], op):
                    return False
        return True
        
    def _forward_check(self, assignment: Dict[str, Any]) -> bool:
        """Forward checking to reduce domains of unassigned variables."""
        for var1, var2, op in self.constraints:
            if var1 in assignment and var2 not in assignment:
                # Remove values from var2's domain that don't support var1's value
                domain2 = self.domains[var2].copy()
                for val2 in domain2:
                    if not self._satisfies_constraint(assignment[var1], val2, op):
                        self.domains[var2].discard(val2)
                        if not self.domains[var2]:
                            return False
                            
            elif var2 in assignment and var1 not in assignment:
                # Remove values from var1's domain that don't support var2's value
                domain1 = self.domains[var1].copy()
                for val1 in domain1:
                    if not self._satisfies_constraint(val1, assignment[var2], op):
                        self.domains[var1].discard(val1)
                        if not self.domains[var1]:
                            return False
                            
        return True


class GlobalConstraintSolver:
    """Solver for global constraints like all_different, alldistinct, etc."""
    
    def __init__(self):
        self.variables: List[str] = []
        self.domains: Dict[str, Set[Any]] = {}
        self.global_constraints: List[Tuple[str, List[str]]] = []
        
    def add_variable(self, var: str, domain: Set[Any]) -> None:
        """Add variable with domain."""
        self.variables.append(var)
        self.domains[var] = set(domain)
        
    def add_all_different(self, variables: List[str]) -> None:
        """Add all_different constraint."""
        self.global_constraints.append(('all_different', variables))
        
    def solve_all_different(self, variables: List[str]) -> List[Dict[str, Any]]:
        """Solve all_different constraint using matching algorithm."""
        if not variables:
            return [{}]
            
        # Build bipartite graph: variables -> values
        graph = defaultdict(list)
        for var in variables:
            if var in self.domains:
                for value in self.domains[var]:
                    graph[var].append(value)
                    
        # Find maximum matching
        matching = self._find_maximum_matching(graph, variables)
        
        if len(matching) == len(variables):
            return [matching]
        else:
            return []
            
    def _find_maximum_matching(self, graph: Dict[str, List[Any]], variables: List[str]) -> Dict[str, Any]:
        """Find maximum matching using Hungarian algorithm (simplified)."""
        matching = {}
        used_values = set()
        
        # Greedy matching
        for var in variables:
            if var in graph:
                for value in graph[var]:
                    if value not in used_values:
                        matching[var] = value
                        used_values.add(value)
                        break
                        
        return matching
        
    def solve_with_global_constraints(self) -> List[Dict[str, Any]]:
        """Solve with all global constraints."""
        solutions = []
        
        # Handle all_different constraints
        all_diff_vars = []
        for constraint_type, variables in self.global_constraints:
            if constraint_type == 'all_different':
                all_diff_vars.extend(variables)
                
        if all_diff_vars:
            all_diff_solutions = self.solve_all_different(all_diff_vars)
            for solution in all_diff_solutions:
                # Extend to other variables
                extended_solutions = self._extend_solution(solution)
                solutions.extend(extended_solutions)
        else:
            # No global constraints, return all combinations
            solutions = self._generate_all_combinations()
            
        return solutions
        
    def _extend_solution(self, partial_solution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extend partial solution to all variables."""
        remaining_vars = [var for var in self.variables if var not in partial_solution]
        
        if not remaining_vars:
            return [partial_solution]
            
        solutions = []
        self._extend_recursive(partial_solution, remaining_vars, solutions)
        return solutions
        
    def _extend_recursive(self, solution: Dict[str, Any], remaining_vars: List[str], solutions: List[Dict[str, Any]]) -> None:
        """Recursively extend solution."""
        if not remaining_vars:
            solutions.append(solution.copy())
            return
            
        var = remaining_vars[0]
        if var in self.domains:
            for value in self.domains[var]:
                solution[var] = value
                self._extend_recursive(solution, remaining_vars[1:], solutions)
                del solution[var]
                
    def _generate_all_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible combinations."""
        solutions = []
        self._generate_recursive({}, self.variables, solutions)
        return solutions
        
    def _generate_recursive(self, solution: Dict[str, Any], variables: List[str], solutions: List[Dict[str, Any]]) -> None:
        """Recursively generate all combinations."""
        if not variables:
            solutions.append(solution.copy())
            return
            
        var = variables[0]
        if var in self.domains:
            for value in self.domains[var]:
                solution[var] = value
                self._generate_recursive(solution, variables[1:], solutions)
                del solution[var]


class ConstraintOptimizer:
    """Optimizer for constraint satisfaction problems."""
    
    def __init__(self):
        self.objective_function = None
        self.constraints = []
        
    def set_objective(self, func: callable) -> None:
        """Set objective function for optimization."""
        self.objective_function = func
        
    def add_constraint(self, constraint: callable) -> None:
        """Add constraint function."""
        self.constraints.append(constraint)
        
    def optimize(self, variables: List[str], domains: Dict[str, Set[Any]]) -> Optional[Dict[str, Any]]:
        """Find optimal solution using branch and bound."""
        if not self.objective_function:
            return None
            
        best_solution = None
        best_value = float('inf')
        
        # Generate all solutions and find optimal
        solutions = self._generate_solutions(variables, domains)
        
        for solution in solutions:
            if self._satisfies_constraints(solution):
                value = self.objective_function(solution)
                if value < best_value:
                    best_value = value
                    best_solution = solution
                    
        return best_solution
        
    def _generate_solutions(self, variables: List[str], domains: Dict[str, Set[Any]]) -> List[Dict[str, Any]]:
        """Generate all possible solutions."""
        solutions = []
        self._generate_recursive({}, variables, domains, solutions)
        return solutions
        
    def _generate_recursive(self, solution: Dict[str, Any], variables: List[str], 
                          domains: Dict[str, Set[Any]], solutions: List[Dict[str, Any]]) -> None:
        """Recursively generate solutions."""
        if not variables:
            solutions.append(solution.copy())
            return
            
        var = variables[0]
        if var in domains:
            for value in domains[var]:
                solution[var] = value
                self._generate_recursive(solution, variables[1:], domains, solutions)
                del solution[var]
                
    def _satisfies_constraints(self, solution: Dict[str, Any]) -> bool:
        """Check if solution satisfies all constraints."""
        for constraint in self.constraints:
            if not constraint(solution):
                return False
        return True


class AdvancedConstraintEngine:
    """Advanced constraint engine combining all constraint solving techniques."""
    
    def __init__(self):
        self.propagator = ConstraintPropagator()
        self.global_solver = GlobalConstraintSolver()
        self.optimizer = ConstraintOptimizer()
        
    def add_finite_domain(self, var: str, domain: Set[Any]) -> None:
        """Add finite domain constraint."""
        self.propagator.add_domain(var, domain)
        self.global_solver.add_variable(var, domain)
        
    def add_binary_constraint(self, var1: str, op: str, var2: str) -> None:
        """Add binary constraint."""
        self.propagator.add_constraint(var1, op, var2)
        
    def add_all_different(self, variables: List[str]) -> None:
        """Add all_different global constraint."""
        self.global_solver.add_all_different(variables)
        
    def solve(self) -> List[Dict[str, Any]]:
        """Solve constraint satisfaction problem."""
        # First try arc consistency propagation
        if not self.propagator.propagate_arc_consistency():
            return []
            
        # Get solutions from propagator
        solutions = self.propagator.get_solutions()
        
        # Filter solutions that satisfy global constraints
        filtered_solutions = []
        for solution in solutions:
            if self._satisfies_global_constraints(solution):
                filtered_solutions.append(solution)
                
        return filtered_solutions
        
    def _satisfies_global_constraints(self, solution: Dict[str, Any]) -> bool:
        """Check if solution satisfies global constraints."""
        for constraint_type, variables in self.global_solver.global_constraints:
            if constraint_type == 'all_different':
                values = [solution.get(var) for var in variables if var in solution]
                if len(values) != len(set(values)):
                    return False
        return True
        
    def optimize(self, objective_func: callable) -> Optional[Dict[str, Any]]:
        """Find optimal solution."""
        self.optimizer.set_objective(objective_func)
        
        # Get all variables and domains
        variables = list(self.propagator.domains.keys())
        domains = self.propagator.domains
        
        return self.optimizer.optimize(variables, domains)
