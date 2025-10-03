"""
Performance Forecaster for OmniMind
Predicts network performance trends and adjusts resource allocation proactively.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict, deque
import statistics
import math


class TrendType(Enum):
    """Types of performance trends."""
    IMPROVING = auto()
    DECLINING = auto()
    STABLE = auto()
    VOLATILE = auto()
    CYCLICAL = auto()


@dataclass
class ForecastResult:
    """Result of a performance forecast."""
    id: str
    metric_name: str
    current_value: float
    predicted_value: float
    confidence: float
    trend_type: TrendType
    forecast_horizon: float
    created_at: float
    recommendations: List[str] = None
    risk_factors: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.risk_factors is None:
            self.risk_factors = []
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'id': self.id,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'predicted_value': self.predicted_value,
            'confidence': self.confidence,
            'trend_type': self.trend_type.name,
            'forecast_horizon': self.forecast_horizon,
            'created_at': self.created_at,
            'recommendations': self.recommendations,
            'risk_factors': self.risk_factors
        }


class PerformanceForecaster:
    """Forecasts network performance trends and provides proactive recommendations."""
    
    def __init__(self, network, monitor, resource_manager):
        self.network = network
        self.monitor = monitor
        self.resource_manager = resource_manager
        self.forecasts: Dict[str, ForecastResult] = {}
        self.forecast_history: List[Dict[str, Any]] = []
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        self.forecast_models: Dict[str, Any] = {}
        
    def forecast_network_performance(self, horizon_hours: float = 24.0) -> List[ForecastResult]:
        """Forecast network performance over a time horizon."""
        forecasts = []
        
        # Forecast network efficiency
        efficiency_forecast = self._forecast_network_efficiency(horizon_hours)
        forecasts.append(efficiency_forecast)
        
        # Forecast communication patterns
        comm_forecast = self._forecast_communication_patterns(horizon_hours)
        forecasts.append(comm_forecast)
        
        # Forecast resource utilization
        resource_forecast = self._forecast_resource_utilization(horizon_hours)
        forecasts.append(resource_forecast)
        
        # Forecast learning effectiveness
        learning_forecast = self._forecast_learning_effectiveness(horizon_hours)
        forecasts.append(learning_forecast)
        
        # Forecast collaboration patterns
        collab_forecast = self._forecast_collaboration_patterns(horizon_hours)
        forecasts.append(collab_forecast)
        
        # Store forecasts
        for forecast in forecasts:
            self.forecasts[forecast.id] = forecast
            
        # Log forecasting
        self._log_forecast_event('performance_forecasted', {
            'total_forecasts': len(forecasts),
            'horizon_hours': horizon_hours,
            'metrics': [f.metric_name for f in forecasts]
        })
        
        return forecasts
        
    def _forecast_network_efficiency(self, horizon_hours: float) -> ForecastResult:
        """Forecast network efficiency."""
        # Get current efficiency
        current_efficiency = self.network.get_network_statistics().get('network_efficiency', 0.5)
        
        # Analyze historical efficiency data
        efficiency_data = self.performance_data.get('network_efficiency', [])
        if len(efficiency_data) < 3:
            # Not enough data for forecasting
            predicted_efficiency = current_efficiency
            confidence = 0.3
            trend_type = TrendType.STABLE
        else:
            # Calculate trend
            trend = self._calculate_trend(efficiency_data)
            predicted_efficiency = current_efficiency + (trend * horizon_hours / 24.0)
            confidence = min(0.9, len(efficiency_data) / 10.0)
            trend_type = self._determine_trend_type(trend)
            
        # Generate recommendations
        recommendations = self._generate_efficiency_recommendations(
            current_efficiency, predicted_efficiency, trend_type
        )
        
        # Ensure we always have at least one recommendation
        if not recommendations:
            recommendations = ["Monitor network performance trends"]
        
        # Identify risk factors
        risk_factors = self._identify_efficiency_risks(predicted_efficiency, trend_type)
        
        forecast = ForecastResult(
            id=str(uuid.uuid4()),
            metric_name='network_efficiency',
            current_value=current_efficiency,
            predicted_value=predicted_efficiency,
            confidence=confidence,
            trend_type=trend_type,
            forecast_horizon=horizon_hours,
            created_at=time.time(),
            recommendations=recommendations,
            risk_factors=risk_factors
        )
        
        return forecast
        
    def _forecast_communication_patterns(self, horizon_hours: float) -> ForecastResult:
        """Forecast communication patterns."""
        # Get current communication volume
        current_volume = 0
        for agent in self.network.agents.values():
            if hasattr(agent, 'communication_history'):
                current_volume += len(agent.communication_history)
                
        # Analyze historical communication data
        comm_data = self.performance_data.get('communication_volume', [])
        if len(comm_data) < 3:
            predicted_volume = current_volume
            confidence = 0.3
            trend_type = TrendType.STABLE
        else:
            trend = self._calculate_trend(comm_data)
            predicted_volume = current_volume + (trend * horizon_hours / 24.0)
            confidence = min(0.9, len(comm_data) / 10.0)
            trend_type = self._determine_trend_type(trend)
            
        # Generate recommendations
        recommendations = self._generate_communication_recommendations(
            current_volume, predicted_volume, trend_type
        )
        
        # Ensure we always have at least one recommendation
        if not recommendations:
            recommendations = ["Monitor communication patterns"]
        
        # Identify risk factors
        risk_factors = self._identify_communication_risks(predicted_volume, trend_type)
        
        forecast = ForecastResult(
            id=str(uuid.uuid4()),
            metric_name='communication_volume',
            current_value=current_volume,
            predicted_value=predicted_volume,
            confidence=confidence,
            trend_type=trend_type,
            forecast_horizon=horizon_hours,
            created_at=time.time(),
            recommendations=recommendations,
            risk_factors=risk_factors
        )
        
        return forecast
        
    def _forecast_resource_utilization(self, horizon_hours: float) -> ForecastResult:
        """Forecast resource utilization."""
        # Get current resource utilization
        resource_stats = self.resource_manager.get_resource_statistics()
        current_utilization = 0.0
        if resource_stats.get('resource_usage'):
            total_allocated = sum(
                resource.get('allocated', 0) for resource in resource_stats['resource_usage'].values()
            )
            total_available = sum(
                resource.get('available', 0) for resource in resource_stats['resource_usage'].values()
            )
            current_utilization = total_allocated / total_available if total_available > 0 else 0
            
        # Analyze historical resource data
        resource_data = self.performance_data.get('resource_utilization', [])
        if len(resource_data) < 3:
            predicted_utilization = current_utilization
            confidence = 0.3
            trend_type = TrendType.STABLE
        else:
            trend = self._calculate_trend(resource_data)
            predicted_utilization = current_utilization + (trend * horizon_hours / 24.0)
            confidence = min(0.9, len(resource_data) / 10.0)
            trend_type = self._determine_trend_type(trend)
            
        # Generate recommendations
        recommendations = self._generate_resource_recommendations(
            current_utilization, predicted_utilization, trend_type
        )
        
        # Ensure we always have at least one recommendation
        if not recommendations:
            recommendations = ["Monitor resource utilization"]
        
        # Identify risk factors
        risk_factors = self._identify_resource_risks(predicted_utilization, trend_type)
        
        forecast = ForecastResult(
            id=str(uuid.uuid4()),
            metric_name='resource_utilization',
            current_value=current_utilization,
            predicted_value=predicted_utilization,
            confidence=confidence,
            trend_type=trend_type,
            forecast_horizon=horizon_hours,
            created_at=time.time(),
            recommendations=recommendations,
            risk_factors=risk_factors
        )
        
        return forecast
        
    def _forecast_learning_effectiveness(self, horizon_hours: float) -> ForecastResult:
        """Forecast learning effectiveness."""
        # Get current learning effectiveness
        current_effectiveness = 0.5  # Placeholder
        
        # Analyze historical learning data
        learning_data = self.performance_data.get('learning_effectiveness', [])
        if len(learning_data) < 3:
            predicted_effectiveness = current_effectiveness
            confidence = 0.3
            trend_type = TrendType.STABLE
        else:
            trend = self._calculate_trend(learning_data)
            predicted_effectiveness = current_effectiveness + (trend * horizon_hours / 24.0)
            confidence = min(0.9, len(learning_data) / 10.0)
            trend_type = self._determine_trend_type(trend)
            
        # Generate recommendations
        recommendations = self._generate_learning_recommendations(
            current_effectiveness, predicted_effectiveness, trend_type
        )
        
        # Ensure we always have at least one recommendation
        if not recommendations:
            recommendations = ["Monitor learning effectiveness"]
        
        # Identify risk factors
        risk_factors = self._identify_learning_risks(predicted_effectiveness, trend_type)
        
        forecast = ForecastResult(
            id=str(uuid.uuid4()),
            metric_name='learning_effectiveness',
            current_value=current_effectiveness,
            predicted_value=predicted_effectiveness,
            confidence=confidence,
            trend_type=trend_type,
            forecast_horizon=horizon_hours,
            created_at=time.time(),
            recommendations=recommendations,
            risk_factors=risk_factors
        )
        
        return forecast
        
    def _forecast_collaboration_patterns(self, horizon_hours: float) -> ForecastResult:
        """Forecast collaboration patterns."""
        # Get current collaboration effectiveness
        current_collaboration = 0.5  # Placeholder
        
        # Analyze historical collaboration data
        collab_data = self.performance_data.get('collaboration_effectiveness', [])
        if len(collab_data) < 3:
            predicted_collaboration = current_collaboration
            confidence = 0.3
            trend_type = TrendType.STABLE
        else:
            trend = self._calculate_trend(collab_data)
            predicted_collaboration = current_collaboration + (trend * horizon_hours / 24.0)
            confidence = min(0.9, len(collab_data) / 10.0)
            trend_type = self._determine_trend_type(trend)
            
        # Generate recommendations
        recommendations = self._generate_collaboration_recommendations(
            current_collaboration, predicted_collaboration, trend_type
        )
        
        # Ensure we always have at least one recommendation
        if not recommendations:
            recommendations = ["Monitor collaboration patterns"]
        
        # Identify risk factors
        risk_factors = self._identify_collaboration_risks(predicted_collaboration, trend_type)
        
        forecast = ForecastResult(
            id=str(uuid.uuid4()),
            metric_name='collaboration_effectiveness',
            current_value=current_collaboration,
            predicted_value=predicted_collaboration,
            confidence=confidence,
            trend_type=trend_type,
            forecast_horizon=horizon_hours,
            created_at=time.time(),
            recommendations=recommendations,
            risk_factors=risk_factors
        )
        
        return forecast
        
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend in data using linear regression."""
        if len(data) < 2:
            return 0.0
            
        n = len(data)
        x = list(range(n))
        y = data
        
        # Calculate slope using least squares
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
        
    def _determine_trend_type(self, trend: float) -> TrendType:
        """Determine trend type based on slope."""
        if trend > 0.1:
            return TrendType.IMPROVING
        elif trend < -0.1:
            return TrendType.DECLINING
        else:
            return TrendType.STABLE
            
    def _generate_efficiency_recommendations(self, current: float, predicted: float, 
                                           trend_type: TrendType) -> List[str]:
        """Generate recommendations for network efficiency."""
        recommendations = []
        
        if trend_type == TrendType.DECLINING:
            recommendations.append("Network efficiency is declining, consider optimization")
        elif trend_type == TrendType.IMPROVING:
            recommendations.append("Network efficiency is improving, maintain current strategies")
        else:
            recommendations.append("Network efficiency is stable, monitor for changes")
            
        if predicted < 0.5:
            recommendations.append("Low predicted efficiency, implement proactive measures")
            
        return recommendations
        
    def _generate_communication_recommendations(self, current: float, predicted: float,
                                              trend_type: TrendType) -> List[str]:
        """Generate recommendations for communication patterns."""
        recommendations = []
        
        if trend_type == TrendType.IMPROVING:
            recommendations.append("Communication volume is increasing, ensure capacity")
        elif trend_type == TrendType.DECLINING:
            recommendations.append("Communication volume is decreasing, check for issues")
            
        if predicted > 100:  # High volume threshold
            recommendations.append("High predicted communication volume, optimize routing")
            
        return recommendations
        
    def _generate_resource_recommendations(self, current: float, predicted: float,
                                         trend_type: TrendType) -> List[str]:
        """Generate recommendations for resource utilization."""
        recommendations = []
        
        if trend_type == TrendType.IMPROVING:
            recommendations.append("Resource utilization is increasing, monitor capacity")
        elif trend_type == TrendType.DECLINING:
            recommendations.append("Resource utilization is decreasing, check allocation")
            
        if predicted > 0.8:  # High utilization threshold
            recommendations.append("High predicted resource utilization, consider scaling")
            
        return recommendations
        
    def _generate_learning_recommendations(self, current: float, predicted: float,
                                        trend_type: TrendType) -> List[str]:
        """Generate recommendations for learning effectiveness."""
        recommendations = []
        
        if trend_type == TrendType.DECLINING:
            recommendations.append("Learning effectiveness is declining, review protocols")
        elif trend_type == TrendType.IMPROVING:
            recommendations.append("Learning effectiveness is improving, maintain strategies")
            
        if predicted < 0.5:
            recommendations.append("Low predicted learning effectiveness, enhance training")
            
        return recommendations
        
    def _generate_collaboration_recommendations(self, current: float, predicted: float,
                                             trend_type: TrendType) -> List[str]:
        """Generate recommendations for collaboration patterns."""
        recommendations = []
        
        if trend_type == TrendType.DECLINING:
            recommendations.append("Collaboration effectiveness is declining, improve protocols")
        elif trend_type == TrendType.IMPROVING:
            recommendations.append("Collaboration effectiveness is improving, maintain strategies")
            
        if predicted < 0.5:
            recommendations.append("Low predicted collaboration effectiveness, enhance coordination")
            
        return recommendations
        
    def _identify_efficiency_risks(self, predicted: float, trend_type: TrendType) -> List[str]:
        """Identify risks for network efficiency."""
        risks = []
        
        if predicted < 0.3:
            risks.append("Low predicted efficiency may cause bottlenecks")
        if trend_type == TrendType.DECLINING:
            risks.append("Declining efficiency trend may lead to system degradation")
            
        return risks
        
    def _identify_communication_risks(self, predicted: float, trend_type: TrendType) -> List[str]:
        """Identify risks for communication patterns."""
        risks = []
        
        if predicted > 200:  # High volume threshold
            risks.append("High predicted communication volume may cause overload")
        if trend_type == TrendType.IMPROVING:
            risks.append("Increasing communication volume may require capacity scaling")
            
        return risks
        
    def _identify_resource_risks(self, predicted: float, trend_type: TrendType) -> List[str]:
        """Identify risks for resource utilization."""
        risks = []
        
        if predicted > 0.9:  # High utilization threshold
            risks.append("High predicted resource utilization may cause bottlenecks")
        if trend_type == TrendType.IMPROVING:
            risks.append("Increasing resource utilization may require capacity scaling")
            
        return risks
        
    def _identify_learning_risks(self, predicted: float, trend_type: TrendType) -> List[str]:
        """Identify risks for learning effectiveness."""
        risks = []
        
        if predicted < 0.3:
            risks.append("Low predicted learning effectiveness may impact knowledge growth")
        if trend_type == TrendType.DECLINING:
            risks.append("Declining learning effectiveness may lead to knowledge stagnation")
            
        return risks
        
    def _identify_collaboration_risks(self, predicted: float, trend_type: TrendType) -> List[str]:
        """Identify risks for collaboration patterns."""
        risks = []
        
        if predicted < 0.3:
            risks.append("Low predicted collaboration effectiveness may impact coordination")
        if trend_type == TrendType.DECLINING:
            risks.append("Declining collaboration effectiveness may lead to coordination issues")
            
        return risks
        
    def update_performance_data(self, metric_name: str, value: float):
        """Update performance data for forecasting."""
        self.performance_data[metric_name].append(value)
        
        # Keep only recent data (last 100 points)
        if len(self.performance_data[metric_name]) > 100:
            self.performance_data[metric_name] = self.performance_data[metric_name][-100:]
            
    def get_forecast_statistics(self) -> Dict[str, Any]:
        """Get forecasting statistics."""
        total_forecasts = len(self.forecasts)
        
        # Calculate average confidence
        confidence_scores = [f.confidence for f in self.forecasts.values()]
        average_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        # Calculate trend distribution
        trend_distribution = defaultdict(int)
        for forecast in self.forecasts.values():
            trend_distribution[forecast.trend_type.name] += 1
            
        # Calculate metric distribution
        metric_distribution = defaultdict(int)
        for forecast in self.forecasts.values():
            metric_distribution[forecast.metric_name] += 1
            
        return {
            'total_forecasts': total_forecasts,
            'average_confidence': average_confidence,
            'trend_distribution': dict(trend_distribution),
            'metric_distribution': dict(metric_distribution)
        }
        
    def _log_forecast_event(self, event_type: str, data: Dict[str, Any]):
        """Log a forecast event."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        self.forecast_history.append(event)
        
    def get_forecast_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get forecast history."""
        return list(self.forecast_history[-limit:])
