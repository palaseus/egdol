"""
Advanced profiling and visualization tools for egdol.
Provides detailed performance analysis, call graphs, and optimization recommendations.
"""

import time
import threading
import json
import statistics
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import networkx as nx
from .parser import Term, Rule
from .rules_engine import RulesEngine
from .interpreter import Interpreter


@dataclass
class ProfilingEvent:
    """Represents a profiling event."""
    event_type: str
    timestamp: float
    duration: float
    thread_id: int
    call_stack: List[str]
    metadata: Dict[str, Any]


@dataclass
class RuleProfile:
    """Detailed profile for a rule."""
    rule_name: str
    arity: int
    total_calls: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    unify_calls: int
    unify_time: float
    avg_unify_time: float
    hot_paths: List[str]
    dependencies: List[str]


@dataclass
class QueryProfile:
    """Profile for a complete query."""
    query_id: str
    query_text: str
    total_time: float
    result_count: int
    rules_executed: List[str]
    call_graph: Dict[str, List[str]]
    bottlenecks: List[str]
    optimization_suggestions: List[str]


class AdvancedProfiler:
    """Advanced profiler with detailed analysis and visualization."""
    
    def __init__(self, engine: RulesEngine):
        self.engine = engine
        self.events: List[ProfilingEvent] = []
        self.rule_profiles: Dict[str, RuleProfile] = {}
        self.query_profiles: Dict[str, QueryProfile] = {}
        self.call_graph = nx.DiGraph()
        self.performance_metrics = PerformanceMetrics()
        self.thread_local = threading.local()
        self.profiling_active = False
        
    def start_profiling(self):
        """Start profiling session."""
        self.profiling_active = True
        self.events.clear()
        self.performance_metrics.reset()
        
    def stop_profiling(self):
        """Stop profiling session."""
        self.profiling_active = False
        self._analyze_events()
        
    def profile_query(self, query: Term, query_id: str = None) -> QueryProfile:
        """Profile a single query with detailed analysis."""
        if not query_id:
            query_id = f"query_{int(time.time())}"
            
        start_time = time.perf_counter()
        
        # Record query start
        self._record_event("query_start", {
            "query_id": query_id,
            "query": str(query)
        })
        
        # Execute query with profiling
        interp = Interpreter(self.engine)
        interp.profile = True
        
        results = []
        try:
            for result in interp.query(query):
                results.append(result)
        except Exception as e:
            self._record_event("query_error", {
                "query_id": query_id,
                "error": str(e)
            })
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Record query end
        self._record_event("query_end", {
            "query_id": query_id,
            "total_time": total_time,
            "result_count": len(results)
        })
        
        # Analyze query performance
        profile = self._analyze_query_performance(query_id, query, total_time, len(results))
        self.query_profiles[query_id] = profile
        
        return profile
        
    def _record_event(self, event_type: str, metadata: Dict[str, Any]):
        """Record a profiling event."""
        if not self.profiling_active:
            return
            
        event = ProfilingEvent(
            event_type=event_type,
            timestamp=time.perf_counter(),
            duration=0.0,
            thread_id=threading.get_ident(),
            call_stack=self._get_call_stack(),
            metadata=metadata
        )
        
        self.events.append(event)
        
    def _get_call_stack(self) -> List[str]:
        """Get current call stack."""
        import traceback
        stack = traceback.extract_stack()
        return [f"{frame.filename}:{frame.lineno}:{frame.name}" for frame in stack[-10:]]
        
    def _analyze_events(self):
        """Analyze collected events."""
        # Group events by type
        event_groups = defaultdict(list)
        for event in self.events:
            event_groups[event.event_type].append(event)
            
        # Analyze rule execution
        self._analyze_rule_execution(event_groups.get("rule_execution", []))
        
        # Analyze unification
        self._analyze_unification(event_groups.get("unification", []))
        
        # Build call graph
        self._build_call_graph()
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
    def _analyze_rule_execution(self, rule_events: List[ProfilingEvent]):
        """Analyze rule execution events."""
        rule_stats = defaultdict(list)
        
        for event in rule_events:
            rule_name = event.metadata.get("rule_name", "unknown")
            duration = event.metadata.get("duration", 0.0)
            rule_stats[rule_name].append(duration)
            
        # Create rule profiles
        for rule_name, durations in rule_stats.items():
            if durations:
                profile = RuleProfile(
                    rule_name=rule_name,
                    arity=0,  # Would need to extract from metadata
                    total_calls=len(durations),
                    total_time=sum(durations),
                    avg_time=statistics.mean(durations),
                    min_time=min(durations),
                    max_time=max(durations),
                    std_dev=statistics.stdev(durations) if len(durations) > 1 else 0.0,
                    unify_calls=0,  # Would need to track separately
                    unify_time=0.0,
                    avg_unify_time=0.0,
                    hot_paths=[],
                    dependencies=[]
                )
                self.rule_profiles[rule_name] = profile
                
    def _analyze_unification(self, unify_events: List[ProfilingEvent]):
        """Analyze unification events."""
        # Track unification performance
        unify_times = [event.metadata.get("duration", 0.0) for event in unify_events]
        
        if unify_times:
            self.performance_metrics.avg_unify_time = statistics.mean(unify_times)
            self.performance_metrics.total_unify_time = sum(unify_times)
            self.performance_metrics.unify_count = len(unify_times)
            
    def _build_call_graph(self):
        """Build call graph from events."""
        self.call_graph.clear()
        
        # Extract call relationships from events
        for event in self.events:
            if event.event_type == "rule_call":
                caller = event.metadata.get("caller", "unknown")
                callee = event.metadata.get("callee", "unknown")
                self.call_graph.add_edge(caller, callee)
                
    def _calculate_performance_metrics(self):
        """Calculate overall performance metrics."""
        if not self.events:
            return
            
        # Calculate timing metrics
        start_times = [e.timestamp for e in self.events if e.event_type == "query_start"]
        end_times = [e.timestamp for e in self.events if e.event_type == "query_end"]
        
        if start_times and end_times:
            self.performance_metrics.total_execution_time = max(end_times) - min(start_times)
            self.performance_metrics.query_count = len(start_times)
            
        # Calculate rule execution metrics
        rule_events = [e for e in self.events if e.event_type == "rule_execution"]
        self.performance_metrics.rule_execution_count = len(rule_events)
        
        if rule_events:
            durations = [e.metadata.get("duration", 0.0) for e in rule_events]
            self.performance_metrics.avg_rule_time = statistics.mean(durations)
            self.performance_metrics.total_rule_time = sum(durations)
            
    def _analyze_query_performance(self, query_id: str, query: Term, 
                                  total_time: float, result_count: int) -> QueryProfile:
        """Analyze performance of a specific query."""
        # Extract rules executed for this query
        query_events = [e for e in self.events if e.metadata.get("query_id") == query_id]
        rules_executed = list(set(e.metadata.get("rule_name") for e in query_events 
                                 if e.event_type == "rule_execution"))
        
        # Build call graph for this query
        call_graph = defaultdict(list)
        for event in query_events:
            if event.event_type == "rule_call":
                caller = event.metadata.get("caller")
                callee = event.metadata.get("callee")
                if caller and callee:
                    call_graph[caller].append(callee)
                    
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(query_events)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(query, bottlenecks)
        
        return QueryProfile(
            query_id=query_id,
            query_text=str(query),
            total_time=total_time,
            result_count=result_count,
            rules_executed=rules_executed,
            call_graph=dict(call_graph),
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions
        )
        
    def _identify_bottlenecks(self, events: List[ProfilingEvent]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Find slow rules
        rule_times = defaultdict(list)
        for event in events:
            if event.event_type == "rule_execution":
                rule_name = event.metadata.get("rule_name", "unknown")
                duration = event.metadata.get("duration", 0.0)
                rule_times[rule_name].append(duration)
                
        # Identify rules that are significantly slower than average
        all_times = [t for times in rule_times.values() for t in times]
        if all_times:
            avg_time = statistics.mean(all_times)
            threshold = avg_time * 2  # 2x average time
            
            for rule_name, times in rule_times.items():
                if any(t > threshold for t in times):
                    bottlenecks.append(f"Slow rule: {rule_name}")
                    
        # Find frequently called rules
        rule_counts = Counter(e.metadata.get("rule_name") for e in events 
                             if e.event_type == "rule_execution")
        if rule_counts:
            max_calls = max(rule_counts.values())
            for rule_name, count in rule_counts.items():
                if count > max_calls * 0.5:  # Called more than 50% of max
                    bottlenecks.append(f"Frequently called rule: {rule_name}")
                    
        return bottlenecks
        
    def _generate_optimization_suggestions(self, query: Term, bottlenecks: List[str]) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Rule reordering suggestions
        if "Slow rule" in str(bottlenecks):
            suggestions.append("Consider reordering rules to put faster rules first")
            
        # Indexing suggestions
        if "Frequently called rule" in str(bottlenecks):
            suggestions.append("Consider adding indexes for frequently accessed predicates")
            
        # Cut operator suggestions
        if len(bottlenecks) > 2:
            suggestions.append("Consider using cut operator (!) to prune search space")
            
        # Constraint suggestions
        if "unification" in str(bottlenecks).lower():
            suggestions.append("Consider adding constraints to reduce unification attempts")
            
        return suggestions
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "summary": {
                "total_queries": len(self.query_profiles),
                "total_execution_time": self.performance_metrics.total_execution_time,
                "average_query_time": self.performance_metrics.total_execution_time / max(1, len(self.query_profiles)),
                "total_rules_executed": self.performance_metrics.rule_execution_count
            },
            "rule_profiles": {name: asdict(profile) for name, profile in self.rule_profiles.items()},
            "query_profiles": {qid: asdict(profile) for qid, profile in self.query_profiles.items()},
            "performance_metrics": {
                "total_execution_time": self.performance_metrics.total_execution_time,
                "query_count": self.performance_metrics.query_count,
                "rule_execution_count": self.performance_metrics.rule_execution_count,
                "avg_unify_time": self.performance_metrics.avg_unify_time,
                "total_unify_time": self.performance_metrics.total_unify_time,
                "unify_count": self.performance_metrics.unify_count,
                "avg_rule_time": self.performance_metrics.avg_rule_time,
                "total_rule_time": self.performance_metrics.total_rule_time
            },
            "call_graph": {
                "nodes": list(self.call_graph.nodes()),
                "edges": list(self.call_graph.edges())
            },
            "recommendations": self._generate_recommendations()
        }
        
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Rule optimization recommendations
        if self.rule_profiles:
            slowest_rule = max(self.rule_profiles.values(), key=lambda p: p.avg_time)
            if slowest_rule.avg_time > 0.01:  # 10ms threshold
                recommendations.append(f"Optimize slow rule: {slowest_rule.rule_name}")
                
        # Memory optimization recommendations
        if self.performance_metrics.total_execution_time > 1.0:
            recommendations.append("Consider adding memory limits for long-running queries")
            
        # Indexing recommendations
        if self.performance_metrics.avg_unify_time > 0.001:  # 1ms threshold
            recommendations.append("Consider adding indexes for frequently unified predicates")
            
        return recommendations
        
    def export_report(self, filename: str):
        """Export performance report to file."""
        report = self.generate_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
    def create_visualizations(self, output_dir: str = "profiling_plots"):
        """Create visualization plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Rule execution time plot
        self._plot_rule_execution_times(f"{output_dir}/rule_times.png")
        
        # Call graph visualization
        self._plot_call_graph(f"{output_dir}/call_graph.png")
        
        # Performance timeline
        self._plot_performance_timeline(f"{output_dir}/timeline.png")
        
    def _plot_rule_execution_times(self, filename: str):
        """Plot rule execution times."""
        if not self.rule_profiles:
            return
            
        rules = list(self.rule_profiles.keys())
        avg_times = [self.rule_profiles[rule].avg_time for rule in rules]
        
        plt.figure(figsize=(12, 6))
        plt.bar(rules, avg_times)
        plt.title("Average Rule Execution Times")
        plt.xlabel("Rules")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def _plot_call_graph(self, filename: str):
        """Plot call graph."""
        if not self.call_graph.nodes():
            return
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.call_graph)
        nx.draw(self.call_graph, pos, with_labels=True, node_size=1000, 
                node_color='lightblue', font_size=8)
        plt.title("Rule Call Graph")
        plt.savefig(filename)
        plt.close()
        
    def _plot_performance_timeline(self, filename: str):
        """Plot performance timeline."""
        if not self.events:
            return
            
        # Extract timeline data
        timestamps = [e.timestamp for e in self.events]
        event_types = [e.event_type for e in self.events]
        
        plt.figure(figsize=(12, 6))
        plt.scatter(timestamps, range(len(timestamps)), c=range(len(timestamps)), 
                   cmap='viridis', alpha=0.6)
        plt.title("Event Timeline")
        plt.xlabel("Time")
        plt.ylabel("Event Index")
        plt.colorbar(label="Event Index")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


class PerformanceMetrics:
    """Performance metrics collector."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.total_execution_time = 0.0
        self.query_count = 0
        self.rule_execution_count = 0
        self.avg_rule_time = 0.0
        self.total_rule_time = 0.0
        self.unify_count = 0
        self.avg_unify_time = 0.0
        self.total_unify_time = 0.0
        self.memory_usage = 0.0
        self.cpu_usage = 0.0


class ProfilingDecorator:
    """Decorator for profiling functions."""
    
    def __init__(self, profiler: AdvancedProfiler, event_type: str):
        self.profiler = profiler
        self.event_type = event_type
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if not self.profiler.profiling_active:
                return func(*args, **kwargs)
                
            start_time = time.perf_counter()
            
            # Record function start
            self.profiler._record_event(f"{self.event_type}_start", {
                "function": func.__name__,
                "args": str(args)[:100],  # Truncate long args
                "kwargs": str(kwargs)[:100]
            })
            
            try:
                result = func(*args, **kwargs)
                
                # Record function end
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                self.profiler._record_event(f"{self.event_type}_end", {
                    "function": func.__name__,
                    "duration": duration,
                    "success": True
                })
                
                return result
                
            except Exception as e:
                # Record error
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                self.profiler._record_event(f"{self.event_type}_error", {
                    "function": func.__name__,
                    "duration": duration,
                    "error": str(e)
                })
                
                raise
                
        return wrapper


def profile_rule(profiler: AdvancedProfiler):
    """Decorator for profiling rule execution."""
    return ProfilingDecorator(profiler, "rule_execution")


def profile_unification(profiler: AdvancedProfiler):
    """Decorator for profiling unification."""
    return ProfilingDecorator(profiler, "unification")


def profile_query(profiler: AdvancedProfiler):
    """Decorator for profiling queries."""
    return ProfilingDecorator(profiler, "query")
