"""
Evaluation Engine for OmniMind Meta-Intelligence
Measures the performance, novelty, and usefulness of new architectures, policies, and skills.
"""

import uuid
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class MetricType(Enum):
    """Types of metrics that can be evaluated."""
    PERFORMANCE = auto()
    NOVELTY = auto()
    USEFULNESS = auto()
    EFFICIENCY = auto()
    ROBUSTNESS = auto()
    SCALABILITY = auto()
    ADAPTABILITY = auto()
    COMPATIBILITY = auto()


class EvaluationStatus(Enum):
    """Status of an evaluation."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class EvaluationResult:
    """Represents the result of an evaluation."""
    id: str
    target_id: str
    target_type: str
    metric_type: MetricType
    score: float
    confidence: float
    evaluation_method: str
    test_cases_passed: int
    total_test_cases: int
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: EvaluationStatus = EvaluationStatus.COMPLETED
    evaluation_notes: List[str] = field(default_factory=list)


class EvaluationEngine:
    """Measures the performance, novelty, and usefulness of new architectures, policies, and skills."""
    
    def __init__(self, network, memory_manager, knowledge_graph):
        self.network = network
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph
        self.evaluation_results: Dict[str, EvaluationResult] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        self.benchmark_data: Dict[str, List[float]] = {}
        self.evaluation_criteria: Dict[str, Dict[str, Any]] = {}
        self.performance_baselines: Dict[str, float] = {}
        
        # Initialize evaluation criteria
        self._initialize_evaluation_criteria()
    
    def _initialize_evaluation_criteria(self):
        """Initialize evaluation criteria for different target types."""
        self.evaluation_criteria = {
            'architecture': {
                'performance_weight': 0.3,
                'novelty_weight': 0.4,
                'usefulness_weight': 0.3,
                'min_score_threshold': 0.7
            },
            'skill': {
                'performance_weight': 0.4,
                'novelty_weight': 0.3,
                'usefulness_weight': 0.3,
                'min_score_threshold': 0.75
            },
            'policy': {
                'performance_weight': 0.35,
                'novelty_weight': 0.25,
                'usefulness_weight': 0.4,
                'min_score_threshold': 0.8
            },
            'algorithm': {
                'performance_weight': 0.5,
                'novelty_weight': 0.2,
                'usefulness_weight': 0.3,
                'min_score_threshold': 0.85
            }
        }
    
    def evaluate_target(self, target_id: str, target_type: str, 
                       target_data: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a target (architecture, skill, policy, etc.)."""
        evaluation_id = str(uuid.uuid4())
        
        # Create evaluation result
        result = EvaluationResult(
            id=evaluation_id,
            target_id=target_id,
            target_type=target_type,
            metric_type=MetricType.PERFORMANCE,
            score=0.0,
            confidence=0.0,
            evaluation_method="comprehensive",
            test_cases_passed=0,
            total_test_cases=0,
            status=EvaluationStatus.IN_PROGRESS
        )
        
        try:
            # Perform evaluation
            evaluation_success = self._perform_evaluation(result, target_data)
            
            if evaluation_success:
                result.status = EvaluationStatus.COMPLETED
                self.evaluation_results[evaluation_id] = result
                self.evaluation_history.append({
                    'evaluation_id': evaluation_id,
                    'target_id': target_id,
                    'target_type': target_type,
                    'completed_at': datetime.now(),
                    'success': True
                })
            else:
                result.status = EvaluationStatus.FAILED
                result.evaluation_notes.append("Evaluation failed")
            
            return result
            
        except Exception as e:
            result.status = EvaluationStatus.FAILED
            result.evaluation_notes.append(f"Evaluation error: {str(e)}")
            return result
    
    def _perform_evaluation(self, result: EvaluationResult, target_data: Dict[str, Any]) -> bool:
        """Perform the actual evaluation."""
        try:
            # Get evaluation criteria for target type
            criteria = self.evaluation_criteria.get(result.target_type, self.evaluation_criteria['skill'])
            
            # Evaluate different aspects
            performance_score = self._evaluate_performance(result, target_data)
            novelty_score = self._evaluate_novelty(result, target_data)
            usefulness_score = self._evaluate_usefulness(result, target_data)
            
            # Calculate weighted score
            weighted_score = (
                performance_score * criteria['performance_weight'] +
                novelty_score * criteria['novelty_weight'] +
                usefulness_score * criteria['usefulness_weight']
            )
            
            result.score = weighted_score
            result.confidence = self._calculate_confidence(result, target_data)
            
            # Generate performance metrics
            result.performance_metrics = {
                'performance_score': performance_score,
                'novelty_score': novelty_score,
                'usefulness_score': usefulness_score,
                'weighted_score': weighted_score,
                'efficiency': random.uniform(0.7, 1.0),
                'robustness': random.uniform(0.6, 0.95),
                'scalability': random.uniform(0.5, 0.9),
                'adaptability': random.uniform(0.6, 0.9)
            }
            
            # Generate recommendations and issues
            result.recommendations = self._generate_recommendations(result, target_data)
            result.issues_found = self._identify_issues(result, target_data)
            
            # Simulate test case execution
            result.test_cases_passed = random.randint(8, 12)
            result.total_test_cases = 12
            
            return True
            
        except Exception as e:
            result.evaluation_notes.append(f"Evaluation error: {str(e)}")
            return False
    
    def _evaluate_performance(self, result: EvaluationResult, target_data: Dict[str, Any]) -> float:
        """Evaluate performance metrics."""
        # Simulate performance evaluation
        base_performance = random.uniform(0.6, 0.9)
        
        # Adjust based on target type
        if result.target_type == 'architecture':
            base_performance += random.uniform(0.0, 0.1)
        elif result.target_type == 'algorithm':
            base_performance += random.uniform(0.0, 0.15)
        
        return min(1.0, base_performance)
    
    def _evaluate_novelty(self, result: EvaluationResult, target_data: Dict[str, Any]) -> float:
        """Evaluate novelty metrics."""
        # Simulate novelty evaluation
        base_novelty = random.uniform(0.5, 0.9)
        
        # Adjust based on target type
        if result.target_type == 'architecture':
            base_novelty += random.uniform(0.0, 0.1)
        elif result.target_type == 'skill':
            base_novelty += random.uniform(0.0, 0.05)
        
        return min(1.0, base_novelty)
    
    def _evaluate_usefulness(self, result: EvaluationResult, target_data: Dict[str, Any]) -> float:
        """Evaluate usefulness metrics."""
        # Simulate usefulness evaluation
        base_usefulness = random.uniform(0.6, 0.9)
        
        # Adjust based on target type
        if result.target_type == 'policy':
            base_usefulness += random.uniform(0.0, 0.1)
        elif result.target_type == 'skill':
            base_usefulness += random.uniform(0.0, 0.05)
        
        return min(1.0, base_usefulness)
    
    def _calculate_confidence(self, result: EvaluationResult, target_data: Dict[str, Any]) -> float:
        """Calculate confidence in the evaluation."""
        # Base confidence
        base_confidence = random.uniform(0.7, 0.95)
        
        # Adjust based on test cases
        if result.test_cases_passed > 0:
            test_confidence = result.test_cases_passed / result.total_test_cases
            base_confidence = (base_confidence + test_confidence) / 2
        
        return min(1.0, base_confidence)
    
    def _generate_recommendations(self, result: EvaluationResult, target_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation."""
        recommendations = []
        
        if result.score < 0.8:
            recommendations.append("Consider improving performance before deployment")
        
        if result.performance_metrics.get('efficiency', 0) < 0.8:
            recommendations.append("Optimize for better efficiency")
        
        if result.performance_metrics.get('robustness', 0) < 0.8:
            recommendations.append("Add more robust error handling")
        
        if result.performance_metrics.get('scalability', 0) < 0.7:
            recommendations.append("Improve scalability for larger workloads")
        
        if result.performance_metrics.get('adaptability', 0) < 0.8:
            recommendations.append("Enhance adaptability for different scenarios")
        
        if not recommendations:
            recommendations.append("Target meets all quality standards")
        
        return recommendations
    
    def _identify_issues(self, result: EvaluationResult, target_data: Dict[str, Any]) -> List[str]:
        """Identify issues found during evaluation."""
        issues = []
        
        if result.score < 0.7:
            issues.append("Overall score below acceptable threshold")
        
        if result.performance_metrics.get('efficiency', 0) < 0.7:
            issues.append("Low efficiency detected")
        
        if result.performance_metrics.get('robustness', 0) < 0.7:
            issues.append("Robustness concerns identified")
        
        if result.performance_metrics.get('scalability', 0) < 0.6:
            issues.append("Scalability limitations found")
        
        if result.performance_metrics.get('adaptability', 0) < 0.7:
            issues.append("Adaptability issues detected")
        
        return issues
    
    def compare_targets(self, target_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple targets."""
        if len(target_ids) < 2:
            return {}
        
        comparison_results = {
            'targets': target_ids,
            'scores': {},
            'rankings': [],
            'best_target': None,
            'comparison_metrics': {}
        }
        
        # Get scores for each target
        for target_id in target_ids:
            target_results = [r for r in self.evaluation_results.values() if r.target_id == target_id]
            if target_results:
                avg_score = sum(r.score for r in target_results) / len(target_results)
                comparison_results['scores'][target_id] = avg_score
        
        # Create rankings
        sorted_targets = sorted(comparison_results['scores'].items(), key=lambda x: x[1], reverse=True)
        comparison_results['rankings'] = [target_id for target_id, _ in sorted_targets]
        
        if sorted_targets:
            comparison_results['best_target'] = sorted_targets[0][0]
        
        # Calculate comparison metrics
        if len(sorted_targets) >= 2:
            best_score = sorted_targets[0][1]
            second_best_score = sorted_targets[1][1]
            comparison_results['comparison_metrics'] = {
                'score_difference': best_score - second_best_score,
                'improvement_percentage': ((best_score - second_best_score) / second_best_score) * 100
            }
        
        return comparison_results
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about evaluations."""
        total_evaluations = len(self.evaluation_results)
        
        if total_evaluations == 0:
            return {
                'total_evaluations': 0,
                'average_score': 0,
                'success_rate': 0,
                'target_type_distribution': {},
                'metric_distribution': {}
            }
        
        # Calculate average score
        avg_score = sum(r.score for r in self.evaluation_results.values()) / total_evaluations
        
        # Calculate success rate
        successful_evaluations = len([r for r in self.evaluation_results.values() if r.status == EvaluationStatus.COMPLETED])
        success_rate = successful_evaluations / total_evaluations
        
        # Target type distribution
        target_type_counts = {}
        for result in self.evaluation_results.values():
            target_type_counts[result.target_type] = target_type_counts.get(result.target_type, 0) + 1
        
        # Metric distribution
        metric_counts = {}
        for result in self.evaluation_results.values():
            metric_counts[result.metric_type.name] = metric_counts.get(result.metric_type.name, 0) + 1
        
        return {
            'total_evaluations': total_evaluations,
            'average_score': avg_score,
            'success_rate': success_rate,
            'target_type_distribution': target_type_counts,
            'metric_distribution': metric_counts,
            'evaluation_history_count': len(self.evaluation_history)
        }
    
    def get_high_performing_targets(self, threshold: float = 0.8) -> List[EvaluationResult]:
        """Get targets with high performance scores."""
        return [r for r in self.evaluation_results.values() if r.score >= threshold]
    
    def get_target_evaluation_history(self, target_id: str) -> List[EvaluationResult]:
        """Get evaluation history for a specific target."""
        return [r for r in self.evaluation_results.values() if r.target_id == target_id]
    
    def update_benchmark_data(self, target_type: str, scores: List[float]) -> None:
        """Update benchmark data for a target type."""
        if target_type not in self.benchmark_data:
            self.benchmark_data[target_type] = []
        
        self.benchmark_data[target_type].extend(scores)
        
        # Keep only recent data (last 100 scores)
        if len(self.benchmark_data[target_type]) > 100:
            self.benchmark_data[target_type] = self.benchmark_data[target_type][-100:]
    
    def get_benchmark_comparison(self, target_type: str, score: float) -> Dict[str, Any]:
        """Compare a score against benchmark data."""
        if target_type not in self.benchmark_data or not self.benchmark_data[target_type]:
            return {
                'benchmark_available': False,
                'score': score,
                'percentile': None,
                'comparison': 'No benchmark data available'
            }
        
        benchmark_scores = self.benchmark_data[target_type]
        
        # Calculate percentile
        percentile = (sum(1 for s in benchmark_scores if s < score) / len(benchmark_scores)) * 100
        
        # Calculate statistics
        benchmark_mean = statistics.mean(benchmark_scores)
        benchmark_std = statistics.stdev(benchmark_scores) if len(benchmark_scores) > 1 else 0
        
        return {
            'benchmark_available': True,
            'score': score,
            'percentile': percentile,
            'benchmark_mean': benchmark_mean,
            'benchmark_std': benchmark_std,
            'comparison': 'Above average' if score > benchmark_mean else 'Below average',
            'z_score': (score - benchmark_mean) / benchmark_std if benchmark_std > 0 else 0
        }
