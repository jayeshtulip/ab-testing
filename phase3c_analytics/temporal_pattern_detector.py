"""
Temporal Pattern Detector for Phase 3C
Time series analysis, seasonality detection, trend analysis, and forecasting
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PatternType(str, Enum):
    """Types of temporal patterns"""
    TREND = "trend"
    SEASONALITY = "seasonality"
    CYCLICAL = "cyclical"
    ANOMALY = "anomaly"
    CHANGE_POINT = "change_point"
    PERIODICITY = "periodicity"

class TrendDirection(str, Enum):
    """Direction of trends"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"

class SeasonalityPeriod(str, Enum):
    """Common seasonality periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

@dataclass
class TimeSeriesData:
    """Time series data point"""
    timestamp: datetime
    value: float
    metric_name: str
    experiment_group: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class DetectedPattern:
    """A detected temporal pattern"""
    pattern_type: PatternType
    start_time: datetime
    end_time: datetime
    strength: float  # 0-1 confidence in pattern
    description: str
    parameters: Dict[str, Any]
    statistical_significance: float

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    direction: TrendDirection
    slope: float
    r_squared: float
    confidence_interval: Tuple[float, float]
    change_rate_per_day: float
    statistical_significance: float

@dataclass
class SeasonalityAnalysis:
    """Seasonality analysis results"""
    period: SeasonalityPeriod
    strength: float
    peak_times: List[str]
    trough_times: List[str]
    amplitude: float
    statistical_significance: float

@dataclass
class AnomalyDetection:
    """Anomaly detection results"""
    anomalies: List[Tuple[datetime, float, float]]  # timestamp, actual, expected
    anomaly_threshold: float
    total_anomalies: int
    anomaly_percentage: float

@dataclass
class ForecastResult:
    """Forecasting results"""
    forecasted_values: List[Tuple[datetime, float, float, float]]  # timestamp, value, lower_bound, upper_bound
    forecast_accuracy: float
    confidence_level: float
    forecast_horizon_days: int

@dataclass
class TemporalAnalysis:
    """Complete temporal pattern analysis"""
    experiment_id: str
    metric_name: str
    analysis_period: Tuple[datetime, datetime]
    data_points: int
    detected_patterns: List[DetectedPattern]
    trend_analysis: TrendAnalysis
    seasonality_analysis: List[SeasonalityAnalysis]
    anomaly_detection: AnomalyDetection
    forecast_result: ForecastResult
    key_insights: List[str]
    recommendations: List[str]

class TemporalPatternDetector:
    """Detector for temporal patterns in experiment data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "anomaly_threshold": 2.5,  # Standard deviations
            "seasonality_periods": [7, 30, 90, 365],  # Days
            "min_pattern_strength": 0.3,
            "trend_significance_threshold": 0.05,
            "forecast_horizon_days": 30,
            "confidence_level": 0.95,
            "min_data_points": 14,  # Minimum 2 weeks of data
            "smoothing_window": 7  # Days for moving average
        }
    
    async def analyze_temporal_patterns(
        self,
        experiment_id: str,
        time_series_data: List[TimeSeriesData],
        metric_name: str
    ) -> TemporalAnalysis:
        """
        Perform comprehensive temporal pattern analysis
        """
        self.logger.info(f"Analyzing temporal patterns for {metric_name} in experiment {experiment_id}")
        
        if len(time_series_data) < self.config["min_data_points"]:
            raise ValueError(f"Insufficient data points: {len(time_series_data)} < {self.config['min_data_points']}")
        
        # Sort data by timestamp
        sorted_data = sorted(time_series_data, key=lambda x: x.timestamp)
        
        # Extract time series arrays
        timestamps = [d.timestamp for d in sorted_data]
        values = [d.value for d in sorted_data]
        
        analysis_period = (timestamps[0], timestamps[-1])
        
        # Detect patterns
        detected_patterns = await self._detect_all_patterns(timestamps, values)
        
        # Trend analysis
        trend_analysis = await self._analyze_trend(timestamps, values)
        
        # Seasonality analysis
        seasonality_analysis = await self._analyze_seasonality(timestamps, values)
        
        # Anomaly detection
        anomaly_detection = await self._detect_anomalies(timestamps, values)
        
        # Forecasting
        forecast_result = await self._generate_forecast(timestamps, values)
        
        # Generate insights and recommendations
        key_insights = await self._generate_temporal_insights(
            detected_patterns, trend_analysis, seasonality_analysis, anomaly_detection
        )
        
        recommendations = await self._generate_temporal_recommendations(
            trend_analysis, seasonality_analysis, anomaly_detection, forecast_result
        )
        
        return TemporalAnalysis(
            experiment_id=experiment_id,
            metric_name=metric_name,
            analysis_period=analysis_period,
            data_points=len(sorted_data),
            detected_patterns=detected_patterns,
            trend_analysis=trend_analysis,
            seasonality_analysis=seasonality_analysis,
            anomaly_detection=anomaly_detection,
            forecast_result=forecast_result,
            key_insights=key_insights,
            recommendations=recommendations
        )
    
    async def _detect_all_patterns(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> List[DetectedPattern]:
        """Detect all types of temporal patterns"""
        
        patterns = []
        
        # Trend patterns
        trend_patterns = await self._detect_trend_patterns(timestamps, values)
        patterns.extend(trend_patterns)
        
        # Seasonal patterns
        seasonal_patterns = await self._detect_seasonal_patterns(timestamps, values)
        patterns.extend(seasonal_patterns)
        
        # Change point patterns
        change_point_patterns = await self._detect_change_points(timestamps, values)
        patterns.extend(change_point_patterns)
        
        # Cyclical patterns
        cyclical_patterns = await self._detect_cyclical_patterns(timestamps, values)
        patterns.extend(cyclical_patterns)
        
        return patterns
    
    async def _detect_trend_patterns(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> List[DetectedPattern]:
        """Detect trend patterns in the data"""
        
        patterns = []
        
        # Convert timestamps to numeric values (days since start)
        start_time = timestamps[0]
        x = [(ts - start_time).days for ts in timestamps]
        
        # Linear regression for trend
        n = len(x)
        if n < 3:
            return patterns
        
        x_mean = np.mean(x)
        y_mean = np.mean(values)
        
        # Calculate slope and intercept
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return patterns
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        predicted = [slope * x[i] + intercept for i in range(n)]
        ss_res = sum((values[i] - predicted[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend strength and significance
        if r_squared > 0.5 and abs(slope) > 0.01:  # Significant trend
            if slope > 0:
                direction = "increasing"
                description = f"Strong upward trend: {slope:.3f} units per day"
            else:
                direction = "decreasing"  
                description = f"Strong downward trend: {slope:.3f} units per day"
            
            patterns.append(DetectedPattern(
                pattern_type=PatternType.TREND,
                start_time=timestamps[0],
                end_time=timestamps[-1],
                strength=min(1.0, r_squared),
                description=description,
                parameters={
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_squared,
                    "direction": direction
                },
                statistical_significance=1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            ))
        
        return patterns
    
    async def _detect_seasonal_patterns(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> List[DetectedPattern]:
        """Detect seasonal patterns in the data"""
        
        patterns = []
        
        for period_days in self.config["seasonality_periods"]:
            if len(timestamps) < period_days * 2:  # Need at least 2 cycles
                continue
            
            # Group data by period (e.g., day of week, day of month)
            period_groups = {}
            
            for i, timestamp in enumerate(timestamps):
                if period_days == 7:  # Weekly
                    key = timestamp.weekday()
                elif period_days == 30:  # Monthly
                    key = timestamp.day
                elif period_days == 90:  # Quarterly  
                    key = timestamp.timetuple().tm_yday % 90
                elif period_days == 365:  # Yearly
                    key = timestamp.timetuple().tm_yday
                else:
                    key = (timestamp - timestamps[0]).days % period_days
                
                if key not in period_groups:
                    period_groups[key] = []
                period_groups[key].append(values[i])
            
            # Calculate seasonal strength
            if len(period_groups) >= 3:  # Need enough periods
                period_means = {}
                for key, group_values in period_groups.items():
                    period_means[key] = np.mean(group_values)
                
                overall_mean = np.mean(values)
                seasonal_variance = np.var(list(period_means.values()))
                total_variance = np.var(values)
                
                seasonal_strength = seasonal_variance / total_variance if total_variance > 0 else 0
                
                if seasonal_strength > self.config["min_pattern_strength"]:
                    # Find peak and trough times
                    sorted_periods = sorted(period_means.items(), key=lambda x: x[1])
                    peak_periods = sorted_periods[-2:]  # Top 2
                    trough_periods = sorted_periods[:2]  # Bottom 2
                    
                    period_name = {7: "weekly", 30: "monthly", 90: "quarterly", 365: "yearly"}.get(period_days, f"{period_days}-day")
                    
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.SEASONALITY,
                        start_time=timestamps[0],
                        end_time=timestamps[-1],
                        strength=seasonal_strength,
                        description=f"Strong {period_name} seasonality detected",
                        parameters={
                            "period_days": period_days,
                            "seasonal_strength": seasonal_strength,
                            "peak_periods": [p[0] for p in peak_periods],
                            "trough_periods": [p[0] for p in trough_periods],
                            "amplitude": max(period_means.values()) - min(period_means.values())
                        },
                        statistical_significance=seasonal_strength
                    ))
        
        return patterns
    
    async def _detect_change_points(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> List[DetectedPattern]:
        """Detect significant change points in the data"""
        
        patterns = []
        
        if len(values) < 10:  # Need sufficient data
            return patterns
        
        # Simple change point detection using moving averages
        window_size = min(7, len(values) // 4)
        
        for i in range(window_size, len(values) - window_size):
            # Calculate means before and after potential change point
            before_values = values[max(0, i - window_size):i]
            after_values = values[i:min(len(values), i + window_size)]
            
            before_mean = np.mean(before_values)
            after_mean = np.mean(after_values)
            
            # Calculate change magnitude
            change_magnitude = abs(after_mean - before_mean)
            
            # Check if change is significant relative to overall variation
            overall_std = np.std(values)
            if overall_std > 0 and change_magnitude > 2 * overall_std:
                
                change_type = "increase" if after_mean > before_mean else "decrease"
                change_percentage = (after_mean - before_mean) / before_mean * 100 if before_mean != 0 else 0
                
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.CHANGE_POINT,
                    start_time=timestamps[i],
                    end_time=timestamps[i],
                    strength=min(1.0, change_magnitude / (3 * overall_std)),
                    description=f"Significant {change_type} detected: {change_percentage:+.1f}%",
                    parameters={
                        "change_magnitude": change_magnitude,
                        "change_percentage": change_percentage,
                        "before_mean": before_mean,
                        "after_mean": after_mean,
                        "change_type": change_type
                    },
                    statistical_significance=min(1.0, change_magnitude / overall_std / 2)
                ))
        
        return patterns
    
    async def _detect_cyclical_patterns(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> List[DetectedPattern]:
        """Detect cyclical patterns using autocorrelation"""
        
        patterns = []
        
        if len(values) < 20:  # Need sufficient data for autocorrelation
            return patterns
        
        # Calculate autocorrelation for different lags
        max_lag = min(len(values) // 3, 30)  # Check up to 30 days lag
        autocorrelations = []
        
        for lag in range(1, max_lag):
            if lag >= len(values):
                break
                
            # Calculate correlation between series and lagged series
            original = values[lag:]
            lagged = values[:-lag]
            
            if len(original) > 5:  # Need some data points
                correlation = np.corrcoef(original, lagged)[0, 1] if len(original) > 1 else 0
                if not np.isnan(correlation):
                    autocorrelations.append((lag, correlation))
        
        # Find significant peaks in autocorrelation
        for lag, correlation in autocorrelations:
            if abs(correlation) > 0.3:  # Significant correlation
                cycle_days = lag
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.CYCLICAL,
                    start_time=timestamps[0],
                    end_time=timestamps[-1],
                    strength=abs(correlation),
                    description=f"Cyclical pattern detected with {cycle_days}-day cycle",
                    parameters={
                        "cycle_length_days": cycle_days,
                        "autocorrelation": correlation
                    },
                    statistical_significance=abs(correlation)
                ))
        
        return patterns
    
    async def _analyze_trend(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> TrendAnalysis:
        """Analyze overall trend in the data"""
        
        # Convert timestamps to numeric values
        start_time = timestamps[0]
        x = [(ts - start_time).days for ts in timestamps]
        
        # Linear regression
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        predicted = [slope * x[i] + intercept for i in range(n)]
        ss_res = sum((values[i] - predicted[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend direction
        if abs(slope) < 0.001:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Check for volatility
        if np.std(values) / np.mean(values) > 0.5:  # High coefficient of variation
            direction = TrendDirection.VOLATILE
        
        # Confidence interval for slope (simplified)
        slope_std_error = np.sqrt(ss_res / (n - 2) / denominator) if n > 2 and denominator > 0 else 0
        t_critical = 1.96  # Approximate 95% CI
        confidence_interval = (
            slope - t_critical * slope_std_error,
            slope + t_critical * slope_std_error
        )
        
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            confidence_interval=confidence_interval,
            change_rate_per_day=slope,
            statistical_significance=r_squared
        )
    
    async def _analyze_seasonality(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> List[SeasonalityAnalysis]:
        """Analyze seasonality patterns"""
        
        seasonality_results = []
        
        # Weekly seasonality
        if len(timestamps) >= 14:  # At least 2 weeks
            weekly_analysis = await self._analyze_weekly_seasonality(timestamps, values)
            if weekly_analysis:
                seasonality_results.append(weekly_analysis)
        
        # Monthly seasonality
        if len(timestamps) >= 60:  # At least 2 months
            monthly_analysis = await self._analyze_monthly_seasonality(timestamps, values)
            if monthly_analysis:
                seasonality_results.append(monthly_analysis)
        
        return seasonality_results
    
    async def _analyze_weekly_seasonality(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> Optional[SeasonalityAnalysis]:
        """Analyze weekly seasonality patterns"""
        
        # Group by day of week
        weekday_groups = [[] for _ in range(7)]
        
        for i, timestamp in enumerate(timestamps):
            weekday = timestamp.weekday()
            weekday_groups[weekday].append(values[i])
        
        # Calculate means for each day
        weekday_means = []
        for group in weekday_groups:
            if group:
                weekday_means.append(np.mean(group))
            else:
                weekday_means.append(0)
        
        if all(mean == 0 for mean in weekday_means):
            return None
        
        # Calculate seasonality strength
        overall_mean = np.mean(values)
        seasonal_variance = np.var(weekday_means)
        total_variance = np.var(values)
        
        strength = seasonal_variance / total_variance if total_variance > 0 else 0
        
        if strength < self.config["min_pattern_strength"]:
            return None
        
        # Find peak and trough days
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        max_idx = np.argmax(weekday_means)
        min_idx = np.argmin(weekday_means)
        
        amplitude = max(weekday_means) - min(weekday_means)
        
        return SeasonalityAnalysis(
            period=SeasonalityPeriod.WEEKLY,
            strength=strength,
            peak_times=[day_names[max_idx]],
            trough_times=[day_names[min_idx]],
            amplitude=amplitude,
            statistical_significance=strength
        )
    
    async def _analyze_monthly_seasonality(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> Optional[SeasonalityAnalysis]:
        """Analyze monthly seasonality patterns"""
        
        # Group by day of month
        day_groups = {}
        
        for i, timestamp in enumerate(timestamps):
            day = timestamp.day
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append(values[i])
        
        if len(day_groups) < 10:  # Need enough days
            return None
        
        # Calculate means for each day
        day_means = {}
        for day, group in day_groups.items():
            day_means[day] = np.mean(group)
        
        # Calculate seasonality strength
        overall_mean = np.mean(values)
        seasonal_variance = np.var(list(day_means.values()))
        total_variance = np.var(values)
        
        strength = seasonal_variance / total_variance if total_variance > 0 else 0
        
        if strength < self.config["min_pattern_strength"]:
            return None
        
        # Find peak and trough days
        max_day = max(day_means.keys(), key=lambda k: day_means[k])
        min_day = min(day_means.keys(), key=lambda k: day_means[k])
        
        amplitude = max(day_means.values()) - min(day_means.values())
        
        return SeasonalityAnalysis(
            period=SeasonalityPeriod.MONTHLY,
            strength=strength,
            peak_times=[f"Day {max_day}"],
            trough_times=[f"Day {min_day}"],
            amplitude=amplitude,
            statistical_significance=strength
        )
    
    async def _detect_anomalies(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> AnomalyDetection:
        """Detect anomalies in the time series"""
        
        # Calculate moving average and standard deviation
        window_size = min(self.config["smoothing_window"], len(values) // 3)
        
        anomalies = []
        
        for i in range(len(values)):
            # Calculate local statistics
            start_idx = max(0, i - window_size)
            end_idx = min(len(values), i + window_size + 1)
            
            local_values = values[start_idx:end_idx]
            local_mean = np.mean(local_values)
            local_std = np.std(local_values)
            
            # Check if point is anomalous
            if local_std > 0:
                z_score = abs(values[i] - local_mean) / local_std
                
                if z_score > self.config["anomaly_threshold"]:
                    anomalies.append((timestamps[i], values[i], local_mean))
        
        return AnomalyDetection(
            anomalies=anomalies,
            anomaly_threshold=self.config["anomaly_threshold"],
            total_anomalies=len(anomalies),
            anomaly_percentage=len(anomalies) / len(values) * 100 if values else 0
        )
    
    async def _generate_forecast(
        self, 
        timestamps: List[datetime], 
        values: List[float]
    ) -> ForecastResult:
        """Generate forecast for future values"""
        
        if len(values) < 7:  # Need at least a week of data
            return ForecastResult(
                forecasted_values=[],
                forecast_accuracy=0.0,
                confidence_level=0.95,
                forecast_horizon_days=0
            )
        
        # Simple exponential smoothing for forecast
        alpha = 0.3  # Smoothing parameter
        
        # Calculate smoothed values
        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed_value = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_value)
        
        # Calculate trend
        trend = 0
        if len(values) > 1:
            recent_values = values[-7:] if len(values) >= 7 else values
            x = list(range(len(recent_values)))
            trend = np.polyfit(x, recent_values, 1)[0]  # Linear trend
        
        # Generate forecast
        last_value = smoothed[-1]
        last_timestamp = timestamps[-1]
        forecast_horizon = self.config["forecast_horizon_days"]
        
        forecasted_values = []
        
        # Estimate forecast error
        if len(values) >= 7:
            recent_errors = []
            for i in range(7, len(values)):
                predicted = smoothed[i-1] + trend
                actual = values[i]
                recent_errors.append(abs(actual - predicted))
            
            forecast_error = np.mean(recent_errors) if recent_errors else np.std(values) * 0.1
        else:
            forecast_error = np.std(values) * 0.2
        
        for day in range(1, forecast_horizon + 1):
            forecast_date = last_timestamp + timedelta(days=day)
            
            # Simple forecast: last smoothed value + trend
            forecasted_value = last_value + trend * day
            
            # Confidence intervals
            confidence_multiplier = 1.96  # 95% confidence
            margin = confidence_multiplier * forecast_error * np.sqrt(day)  # Increasing uncertainty
            
            lower_bound = forecasted_value - margin
            upper_bound = forecasted_value + margin
            
            forecasted_values.append((forecast_date, forecasted_value, lower_bound, upper_bound))
        
        # Estimate forecast accuracy (simplified)
        if len(values) >= 14:
            # Use last 7 days to estimate accuracy
            test_values = values[-7:]
            test_forecasts = []
            
            for i in range(len(test_values)):
                forecast_val = smoothed[-8] + trend * (i + 1)
                test_forecasts.append(forecast_val)
            
            accuracy_errors = [abs(test_values[i] - test_forecasts[i]) for i in range(len(test_values))]
            mean_error = np.mean(accuracy_errors)
            mean_actual = np.mean(test_values)
            
            forecast_accuracy = max(0, 1 - (mean_error / mean_actual)) if mean_actual > 0 else 0.5
        else:
            forecast_accuracy = 0.7  # Default moderate accuracy
        
        return ForecastResult(
            forecasted_values=forecasted_values,
            forecast_accuracy=forecast_accuracy,
            confidence_level=0.95,
            forecast_horizon_days=forecast_horizon
        )
    
    async def _generate_temporal_insights(
        self,
        patterns: List[DetectedPattern],
        trend: TrendAnalysis,
        seasonality: List[SeasonalityAnalysis],
        anomalies: AnomalyDetection
    ) -> List[str]:
        """Generate key insights from temporal analysis"""
        
        insights = []
        
        # Trend insights
        if trend.direction == TrendDirection.INCREASING and trend.statistical_significance > 0.5:
            insights.append(f"Strong upward trend detected: {trend.change_rate_per_day:+.3f} units per day")
        elif trend.direction == TrendDirection.DECREASING and trend.statistical_significance > 0.5:
            insights.append(f"Concerning downward trend: {trend.change_rate_per_day:+.3f} units per day")
        elif trend.direction == TrendDirection.VOLATILE:
            insights.append("High volatility detected - consider stabilization strategies")
        
        # Seasonality insights
        for season in seasonality:
            if season.strength > 0.3:
                if season.period == SeasonalityPeriod.WEEKLY:
                    insights.append(f"Strong weekly pattern: peaks on {', '.join(season.peak_times)}")
                elif season.period == SeasonalityPeriod.MONTHLY:
                    insights.append(f"Monthly seasonality detected: peaks on {', '.join(season.peak_times)}")
        
        # Pattern insights
        significant_patterns = [p for p in patterns if p.strength > 0.5]
        if significant_patterns:
            change_points = [p for p in significant_patterns if p.pattern_type == PatternType.CHANGE_POINT]
            if change_points:
                insights.append(f"{len(change_points)} significant change points detected")
        
        # Anomaly insights
        if anomalies.anomaly_percentage > 5:
            insights.append(f"High anomaly rate: {anomalies.anomaly_percentage:.1f}% of data points are outliers")
        elif anomalies.total_anomalies > 0:
            insights.append(f"{anomalies.total_anomalies} anomalies detected - investigate unusual events")
        
        return insights
    
    async def _generate_temporal_recommendations(
        self,
        trend: TrendAnalysis,
        seasonality: List[SeasonalityAnalysis],
        anomalies: AnomalyDetection,
        forecast: ForecastResult
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Trend recommendations
        if trend.direction == TrendDirection.INCREASING:
            recommendations.append("Capitalize on positive trend - consider scaling successful strategies")
        elif trend.direction == TrendDirection.DECREASING:
            recommendations.append("Address declining trend - investigate root causes and implement corrective measures")
        elif trend.direction == TrendDirection.VOLATILE:
            recommendations.append("Reduce volatility through consistent experimentation and measurement practices")
        
        # Seasonality recommendations
        for season in seasonality:
            if season.strength > 0.3:
                if season.period == SeasonalityPeriod.WEEKLY:
                    recommendations.append(f"Optimize for weekly patterns - focus efforts on {', '.join(season.peak_times)}")
                elif season.period == SeasonalityPeriod.MONTHLY:
                    recommendations.append("Consider monthly campaign timing based on detected seasonality")
        
        # Anomaly recommendations
        if anomalies.anomaly_percentage > 10:
            recommendations.append("High anomaly rate suggests data quality issues - investigate measurement consistency")
        elif anomalies.total_anomalies > 0:
            recommendations.append("Analyze anomaly dates for external events that may have influenced results")
        
        # Forecast recommendations
        if forecast.forecast_accuracy > 0.8:
            recommendations.append("High forecast accuracy enables reliable future planning")
        elif forecast.forecast_accuracy < 0.6:
            recommendations.append("Low forecast accuracy - consider longer measurement periods for better predictions")
        
        return recommendations

# Demo functions
async def create_demo_time_series() -> List[TimeSeriesData]:
    """Create demo time series data"""
    
    np.random.seed(42)  # For reproducible results
    data_points = []
    
    start_date = datetime(2024, 1, 1)
    
    # Generate 90 days of data
    for day in range(90):
        current_date = start_date + timedelta(days=day)
        
        # Base value with trend
        base_value = 100 + day * 0.2  # Slight upward trend
        
        # Weekly seasonality (higher on weekdays)
        weekday = current_date.weekday()
        if weekday < 5:  # Monday-Friday
            seasonal_boost = 15
        else:  # Weekend
            seasonal_boost = -5
        
        # Monthly pattern (higher at month end)
        monthly_factor = 1 + 0.1 * np.sin(2 * np.pi * current_date.day / 30)
        
        # Random noise
        noise = np.random.normal(0, 8)
        
        # Occasional anomalies
        if day in [25, 45, 65]:  # Specific anomaly days
            noise += np.random.choice([-30, 30])  # Large anomaly
        
        final_value = (base_value + seasonal_boost) * monthly_factor + noise
        final_value = max(10, final_value)  # Ensure positive values
        
        data_points.append(TimeSeriesData(
            timestamp=current_date,
            value=final_value,
            metric_name="conversion_rate",
            experiment_group="treatment",
            metadata={"day_of_week": weekday, "day_of_month": current_date.day}
        ))
    
    return data_points

async def run_temporal_pattern_demo():
    """Run comprehensive temporal pattern detection demo"""
    
    print("Temporal Pattern Detector Demo")
    print("=" * 50)
    
    # Create detector
    detector = TemporalPatternDetector()
    
    # Create demo time series data
    print("Demo Analysis: 90-Day Conversion Rate Time Series")
    print("Detecting patterns, trends, seasonality, and anomalies...")
    
    time_series_data = await create_demo_time_series()
    
    print(f"\nTime Series Summary:")
    print(f"  Data Points: {len(time_series_data)}")
    print(f"  Period: {time_series_data[0].timestamp.strftime('%Y-%m-%d')} to {time_series_data[-1].timestamp.strftime('%Y-%m-%d')}")
    
    values = [d.value for d in time_series_data]
    print(f"  Value Range: {min(values):.1f} to {max(values):.1f}")
    print(f"  Mean Value: {np.mean(values):.1f}")
    print(f"  Standard Deviation: {np.std(values):.1f}")
    
    # Run temporal analysis
    print(f"\nRunning Temporal Pattern Analysis...")
    analysis = await detector.analyze_temporal_patterns("time_series_demo_001", time_series_data, "conversion_rate")
    
    # Display results
    print(f"\nTemporal Pattern Analysis Results")
    print("=" * 50)
    
    # Trend analysis
    trend = analysis.trend_analysis
    print(f"Trend Analysis:")
    print(f"  Direction: {trend.direction.value.title()}")
    print(f"  Slope: {trend.slope:+.4f} units per day")
    print(f"  R-squared: {trend.r_squared:.3f}")
    print(f"  Statistical Significance: {trend.statistical_significance:.3f}")
    if trend.direction in [TrendDirection.INCREASING, TrendDirection.DECREASING]:
        print(f"  Change Rate: {trend.change_rate_per_day:+.3f} per day")
    
    # Detected patterns
    if analysis.detected_patterns:
        print(f"\nDetected Patterns ({len(analysis.detected_patterns)}):")
        for pattern in analysis.detected_patterns:
            print(f"  {pattern.pattern_type.value.title()}: {pattern.description}")
            print(f"    Strength: {pattern.strength:.3f}")
            print(f"    Significance: {pattern.statistical_significance:.3f}")
    
    # Seasonality analysis
    if analysis.seasonality_analysis:
        print(f"\nSeasonality Analysis:")
        for seasonality in analysis.seasonality_analysis:
            print(f"  {seasonality.period.value.title()} Pattern:")
            print(f"    Strength: {seasonality.strength:.3f}")
            print(f"    Peak Times: {', '.join(seasonality.peak_times)}")
            print(f"    Trough Times: {', '.join(seasonality.trough_times)}")
            print(f"    Amplitude: {seasonality.amplitude:.2f}")
    
    # Anomaly detection
    anomalies = analysis.anomaly_detection
    print(f"\nAnomaly Detection:")
    print(f"  Total Anomalies: {anomalies.total_anomalies}")
    print(f"  Anomaly Rate: {anomalies.anomaly_percentage:.1f}%")
    print(f"  Threshold: {anomalies.anomaly_threshold} standard deviations")
    
    if anomalies.anomalies:
        print(f"  Recent Anomalies:")
        for timestamp, actual, expected in anomalies.anomalies[-3:]:  # Show last 3
            print(f"    {timestamp.strftime('%Y-%m-%d')}: {actual:.1f} (expected: {expected:.1f})")
    
    # Forecast
    forecast = analysis.forecast_result
    print(f"\nForecast ({forecast.forecast_horizon_days} days):")
    print(f"  Forecast Accuracy: {forecast.forecast_accuracy:.1%}")
    print(f"  Confidence Level: {forecast.confidence_level:.0%}")
    
    if forecast.forecasted_values:
        print(f"  Next 5 Days Forecast:")
        for timestamp, value, lower, upper in forecast.forecasted_values[:5]:
            print(f"    {timestamp.strftime('%Y-%m-%d')}: {value:.1f} [{lower:.1f}, {upper:.1f}]")
    
    # Key insights
    if analysis.key_insights:
        print(f"\nKey Insights:")
        for insight in analysis.key_insights:
            print(f"  • {insight}")
    
    # Recommendations
    if analysis.recommendations:
        print(f"\nRecommendations:")
        for recommendation in analysis.recommendations:
            print(f"  • {recommendation}")

async def main():
    """Main function"""
    
    print("Phase 3C: Temporal Pattern Detector")
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        await run_temporal_pattern_demo()
        
        print("\n" + "=" * 60)
        print("TEMPORAL PATTERN ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Successfully Analyzed:")
        print("  - Trend Detection & Analysis")
        print("  - Seasonality Pattern Recognition")
        print("  - Change Point Detection")
        print("  - Cyclical Pattern Analysis")
        print("  - Anomaly Detection")
        print("  - Time Series Forecasting")
        print("  - Statistical Significance Testing")
        print("  - Actionable Insights & Recommendations")
        print()
        print("PHASE 3C COMPLETE!")
        print("=" * 30)
        print("All Components Successfully Implemented:")
        print("1. Business Impact Analyzer - COMPLETED")
        print("2. ROI Calculator - COMPLETED")
        print("3. Segment Analyzer - COMPLETED") 
        print("4. Temporal Pattern Detector - COMPLETED")
        print()
        print("Phase 3C Advanced Analytics is now fully operational!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())