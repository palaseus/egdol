"""
Result Analyzer for OmniMind Experimental Intelligence
Analyzes experiment results and provides confidence scoring and insights.
"""

import statistics
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class ResultType(Enum):
    """Types of experiment results."""
    SUCCESS = auto()
    PARTIAL_SUCCESS = auto()
    FAILURE = auto()
    INCONCLUSIVE = auto()
    INVALID = auto()


class ConfidenceLevel(Enum):
    """Levels of confidence in results."""
    VERY_LOW = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    VERY_HIGH = auto()


@dataclass
class ExperimentResult:
    """Represents the analysis of an experiment result."""
    experiment_id: str
    hypothesis_id: str
    result_type: ResultType
    confidence_level: ConfidenceLevel
    confidence_score: float
    success_metrics: Dict[str, float]
    failure_metrics: Dict[str, float]
    insights: List[str]
    implications: List[str]
    recommendations: List[str]
    analyzed_at: datetime = field(default_factory=datetime.now)
    statistical_significance: float = 0.0
    effect_size: float = 0.0
    reproducibility_score: float = 0.0


class ResultAnalyzer:
    """Analyzes experiment results and provides insights."""
    
    def __init__(self, network, memory_manager, knowledge_graph):
        self.network = network
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph
        self.analysis_history: List[ExperimentResult] = []
        self.pattern_insights: Dict[str, List[str]] = {}
        self.confidence_thresholds: Dict[str, float] = {
            'very_high': 0.9,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'very_low': 0.2
        }
        
    def analyze_experiment_result(self, experiment_id: str, experiment_data: Dict[str, Any]) -> ExperimentResult:
        """Analyze the results of an experiment."""
        # Extract key metrics
        metrics = experiment_data.get('metrics', {})
        results = experiment_data.get('results', {})
        errors = experiment_data.get('errors', [])
        
        # Determine result type
        result_type = self._determine_result_type(metrics, results, errors)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(metrics, results, errors)
        confidence_level = self._determine_confidence_level(confidence_score)
        
        # Analyze success and failure metrics
        success_metrics = self._extract_success_metrics(metrics, results)
        failure_metrics = self._extract_failure_metrics(metrics, results, errors)
        
        # Generate insights
        insights = self._generate_insights(experiment_data, result_type, confidence_score)
        
        # Generate implications
        implications = self._generate_implications(experiment_data, result_type, insights)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(experiment_data, result_type, insights)
        
        # Calculate statistical measures
        statistical_significance = self._calculate_statistical_significance(metrics, results)
        effect_size = self._calculate_effect_size(metrics, results)
        reproducibility_score = self._calculate_reproducibility_score(experiment_data)
        
        # Create result analysis
        analysis = ExperimentResult(
            experiment_id=experiment_id,
            hypothesis_id=experiment_data.get('hypothesis_id', ''),
            result_type=result_type,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            success_metrics=success_metrics,
            failure_metrics=failure_metrics,
            insights=insights,
            implications=implications,
            recommendations=recommendations,
            statistical_significance=statistical_significance,
            effect_size=effect_size,
            reproducibility_score=reproducibility_score
        )
        
        # Store analysis
        self.analysis_history.append(analysis)
        
        # Update pattern insights
        self._update_pattern_insights(analysis)
        
        return analysis
    
    def _determine_result_type(self, metrics: Dict[str, float], results: Dict[str, Any], errors: List[str]) -> ResultType:
        """Determine the type of result based on metrics and errors."""
        if errors:
            return ResultType.INVALID
        
        # Check if experiment was successful
        success_indicators = self._get_success_indicators(metrics, results)
        failure_indicators = self._get_failure_indicators(metrics, results)
        
        if success_indicators > failure_indicators * 2:
            return ResultType.SUCCESS
        elif success_indicators > failure_indicators:
            return ResultType.PARTIAL_SUCCESS
        elif failure_indicators > success_indicators * 2:
            return ResultType.FAILURE
        else:
            return ResultType.INCONCLUSIVE
    
    def _calculate_confidence_score(self, metrics: Dict[str, float], results: Dict[str, Any], errors: List[str]) -> float:
        """Calculate confidence score for the experiment result."""
        if errors:
            return 0.1  # Very low confidence if there are errors
        
        # Base confidence on metric consistency
        confidence_factors = []
        
        # Check metric consistency
        if metrics:
            metric_values = list(metrics.values())
            if len(metric_values) > 1:
                consistency = 1.0 - statistics.stdev(metric_values) / statistics.mean(metric_values)
                confidence_factors.append(max(0, consistency))
        
        # Check result quality
        if results:
            quality_score = self._assess_result_quality(results)
            confidence_factors.append(quality_score)
        
        # Check statistical significance
        if 'significance' in metrics:
            confidence_factors.append(metrics['significance'])
        
        # Check reproducibility
        if 'reproducibility' in metrics:
            confidence_factors.append(metrics['reproducibility'])
        
        # Calculate final confidence score
        if confidence_factors:
            return statistics.mean(confidence_factors)
        else:
            return 0.5  # Default moderate confidence
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level based on score."""
        if confidence_score >= self.confidence_thresholds['very_high']:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= self.confidence_thresholds['high']:
            return ConfidenceLevel.HIGH
        elif confidence_score >= self.confidence_thresholds['medium']:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= self.confidence_thresholds['low']:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _extract_success_metrics(self, metrics: Dict[str, float], results: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics that indicate success."""
        success_metrics = {}
        
        # Look for positive indicators
        for key, value in metrics.items():
            if any(indicator in key.lower() for indicator in ['accuracy', 'efficiency', 'improvement', 'success']):
                success_metrics[key] = value
        
        # Add result-based success metrics
        for key, value in results.items():
            if isinstance(value, (int, float)) and value > 0:
                if any(indicator in key.lower() for indicator in ['rate', 'score', 'performance']):
                    success_metrics[key] = float(value)
        
        return success_metrics
    
    def _extract_failure_metrics(self, metrics: Dict[str, float], results: Dict[str, Any], errors: List[str]) -> Dict[str, float]:
        """Extract metrics that indicate failure."""
        failure_metrics = {}
        
        # Look for negative indicators
        for key, value in metrics.items():
            if any(indicator in key.lower() for indicator in ['error', 'failure', 'loss', 'degradation']):
                failure_metrics[key] = value
        
        # Add error count
        if errors:
            failure_metrics['error_count'] = len(errors)
            failure_metrics['error_severity'] = len([e for e in errors if 'critical' in e.lower()])
        
        return failure_metrics
    
    def _generate_insights(self, experiment_data: Dict[str, Any], result_type: ResultType, confidence_score: float) -> List[str]:
        """Generate insights from the experiment results."""
        insights = []
        
        # Result type insights
        if result_type == ResultType.SUCCESS:
            insights.append("Experiment achieved its primary objectives with high confidence")
        elif result_type == ResultType.PARTIAL_SUCCESS:
            insights.append("Experiment showed mixed results with some objectives met")
        elif result_type == ResultType.FAILURE:
            insights.append("Experiment failed to meet primary objectives")
        elif result_type == ResultType.INCONCLUSIVE:
            insights.append("Experiment results are inconclusive and require further investigation")
        
        # Confidence-based insights
        if confidence_score >= 0.8:
            insights.append("High confidence in results - suitable for integration")
        elif confidence_score >= 0.6:
            insights.append("Moderate confidence - results may need validation")
        else:
            insights.append("Low confidence - results require additional verification")
        
        # Metric-specific insights
        metrics = experiment_data.get('metrics', {})
        if 'accuracy' in metrics:
            if metrics['accuracy'] > 0.9:
                insights.append("High accuracy achieved in experimental measurements")
            elif metrics['accuracy'] < 0.7:
                insights.append("Accuracy concerns detected - measurement reliability questionable")
        
        if 'efficiency' in metrics:
            if metrics['efficiency'] > 0.8:
                insights.append("Efficient resource utilization demonstrated")
            elif metrics['efficiency'] < 0.6:
                insights.append("Resource efficiency below optimal levels")
        
        return insights
    
    def _generate_implications(self, experiment_data: Dict[str, Any], result_type: ResultType, insights: List[str]) -> List[str]:
        """Generate implications of the experiment results."""
        implications = []
        
        # Result type implications
        if result_type == ResultType.SUCCESS:
            implications.append("Hypothesis validated - can be integrated into network policies")
            implications.append("Successful approach can be replicated in similar scenarios")
        elif result_type == ResultType.PARTIAL_SUCCESS:
            implications.append("Partial validation suggests need for refinement")
            implications.append("Mixed results indicate complex interactions requiring further study")
        elif result_type == ResultType.FAILURE:
            implications.append("Hypothesis rejected - alternative approaches needed")
            implications.append("Failed approach provides valuable negative knowledge")
        elif result_type == ResultType.INCONCLUSIVE:
            implications.append("Inconclusive results suggest experimental design issues")
            implications.append("Additional controlled experiments required")
        
        # Network implications
        if 'collaboration' in str(experiment_data).lower():
            implications.append("Collaboration patterns may need adjustment based on results")
        
        if 'optimization' in str(experiment_data).lower():
            implications.append("Optimization strategies may require revision")
        
        return implications
    
    def _generate_recommendations(self, experiment_data: Dict[str, Any], result_type: ResultType, insights: List[str]) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        # Result type recommendations
        if result_type == ResultType.SUCCESS:
            recommendations.append("Integrate successful approach into network operations")
            recommendations.append("Scale up successful methodology to broader applications")
        elif result_type == ResultType.PARTIAL_SUCCESS:
            recommendations.append("Refine experimental approach based on partial success")
            recommendations.append("Investigate factors that led to mixed results")
        elif result_type == ResultType.FAILURE:
            recommendations.append("Develop alternative hypotheses and approaches")
            recommendations.append("Analyze failure modes to prevent similar issues")
        elif result_type == ResultType.INCONCLUSIVE:
            recommendations.append("Redesign experiment with better controls")
            recommendations.append("Increase sample size or experimental duration")
        
        # Metric-based recommendations
        metrics = experiment_data.get('metrics', {})
        if 'reproducibility' in metrics and metrics['reproducibility'] < 0.8:
            recommendations.append("Improve experimental reproducibility through better controls")
        
        if 'significance' in metrics and metrics['significance'] < 0.7:
            recommendations.append("Increase statistical power through larger sample sizes")
        
        return recommendations
    
    def _get_success_indicators(self, metrics: Dict[str, float], results: Dict[str, Any]) -> int:
        """Count success indicators in metrics and results."""
        success_count = 0
        
        # Check metrics for success indicators
        for key, value in metrics.items():
            if any(indicator in key.lower() for indicator in ['accuracy', 'efficiency', 'improvement']):
                if value > 0.7:  # Threshold for success
                    success_count += 1
        
        # Check results for success indicators
        for key, value in results.items():
            if isinstance(value, (int, float)) and value > 0.7:
                success_count += 1
        
        return success_count
    
    def _get_failure_indicators(self, metrics: Dict[str, float], results: Dict[str, Any]) -> int:
        """Count failure indicators in metrics and results."""
        failure_count = 0
        
        # Check metrics for failure indicators
        for key, value in metrics.items():
            if any(indicator in key.lower() for indicator in ['error', 'failure', 'loss']):
                failure_count += 1
        
        # Check for low performance metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value < 0.5:
                failure_count += 1
        
        return failure_count
    
    def _assess_result_quality(self, results: Dict[str, Any]) -> float:
        """Assess the quality of experimental results."""
        if not results:
            return 0.0
        
        quality_factors = []
        
        # Check for completeness
        completeness = len([v for v in results.values() if v is not None]) / len(results)
        quality_factors.append(completeness)
        
        # Check for consistency
        numeric_values = [v for v in results.values() if isinstance(v, (int, float))]
        if len(numeric_values) > 1:
            consistency = 1.0 - statistics.stdev(numeric_values) / statistics.mean(numeric_values)
            quality_factors.append(max(0, consistency))
        
        return statistics.mean(quality_factors) if quality_factors else 0.5
    
    def _calculate_statistical_significance(self, metrics: Dict[str, float], results: Dict[str, Any]) -> float:
        """Calculate statistical significance of results."""
        if 'significance' in metrics:
            return metrics['significance']
        
        # Estimate significance based on result consistency
        numeric_values = [v for v in results.values() if isinstance(v, (int, float))]
        if len(numeric_values) > 1:
            # Simple significance estimation based on variance
            mean_val = statistics.mean(numeric_values)
            std_val = statistics.stdev(numeric_values)
            return min(1.0, max(0.0, 1.0 - std_val / mean_val)) if mean_val > 0 else 0.0
        
        return 0.5  # Default moderate significance
    
    def _calculate_effect_size(self, metrics: Dict[str, float], results: Dict[str, Any]) -> float:
        """Calculate effect size of the experimental intervention."""
        if 'effect_size' in metrics:
            return metrics['effect_size']
        
        # Estimate effect size based on improvement metrics
        improvement_metrics = [v for k, v in metrics.items() if 'improvement' in k.lower()]
        if improvement_metrics:
            return statistics.mean(improvement_metrics)
        
        return 0.0  # No effect size available
    
    def _calculate_reproducibility_score(self, experiment_data: Dict[str, Any]) -> float:
        """Calculate reproducibility score for the experiment."""
        if 'reproducibility' in experiment_data.get('metrics', {}):
            return experiment_data['metrics']['reproducibility']
        
        # Estimate reproducibility based on result consistency
        results = experiment_data.get('results', {})
        numeric_values = [v for v in results.values() if isinstance(v, (int, float))]
        
        if len(numeric_values) > 1:
            # Higher consistency = higher reproducibility
            mean_val = statistics.mean(numeric_values)
            std_val = statistics.stdev(numeric_values)
            return min(1.0, max(0.0, 1.0 - std_val / mean_val)) if mean_val > 0 else 0.0
        
        return 0.5  # Default moderate reproducibility
    
    def _update_pattern_insights(self, analysis: ExperimentResult) -> None:
        """Update pattern insights based on analysis."""
        pattern_key = f"{analysis.result_type.name}_{analysis.confidence_level.name}"
        
        if pattern_key not in self.pattern_insights:
            self.pattern_insights[pattern_key] = []
        
        # Add insights to pattern
        self.pattern_insights[pattern_key].extend(analysis.insights)
        
        # Keep only recent insights (limit to 10 per pattern)
        if len(self.pattern_insights[pattern_key]) > 10:
            self.pattern_insights[pattern_key] = self.pattern_insights[pattern_key][-10:]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about result analyses."""
        total_analyses = len(self.analysis_history)
        
        if total_analyses == 0:
            return {'total_analyses': 0}
        
        # Result type distribution
        result_type_counts = {}
        for analysis in self.analysis_history:
            result_type = analysis.result_type.name
            result_type_counts[result_type] = result_type_counts.get(result_type, 0) + 1
        
        # Confidence level distribution
        confidence_counts = {}
        for analysis in self.analysis_history:
            confidence = analysis.confidence_level.name
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        
        # Average confidence score
        avg_confidence = statistics.mean([a.confidence_score for a in self.analysis_history])
        
        # Average statistical significance
        avg_significance = statistics.mean([a.statistical_significance for a in self.analysis_history])
        
        return {
            'total_analyses': total_analyses,
            'result_type_distribution': result_type_counts,
            'confidence_distribution': confidence_counts,
            'average_confidence': avg_confidence,
            'average_significance': avg_significance,
            'pattern_insights_count': len(self.pattern_insights)
        }
    
    def get_insights_by_pattern(self, pattern: str) -> List[str]:
        """Get insights for a specific result pattern."""
        return self.pattern_insights.get(pattern, [])
    
    def get_high_confidence_results(self, threshold: float = 0.8) -> List[ExperimentResult]:
        """Get results with confidence above threshold."""
        return [a for a in self.analysis_history if a.confidence_score >= threshold]
    
    def get_successful_results(self) -> List[ExperimentResult]:
        """Get all successful experiment results."""
        return [a for a in self.analysis_history if a.result_type == ResultType.SUCCESS]
