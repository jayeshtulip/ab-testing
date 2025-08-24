"""
Segment Analyzer for Phase 3C
Customer segmentation and segment-specific performance analysis
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

class SegmentationType(str, Enum):
    """Types of customer segmentation"""
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    GEOGRAPHIC = "geographic"
    PSYCHOGRAPHIC = "psychographic"
    VALUE_BASED = "value_based"
    LIFECYCLE = "lifecycle"

class SegmentMetric(str, Enum):
    """Metrics for segment analysis"""
    CONVERSION_RATE = "conversion_rate"
    REVENUE_PER_USER = "revenue_per_user"
    ENGAGEMENT_SCORE = "engagement_score"
    RETENTION_RATE = "retention_rate"
    LIFETIME_VALUE = "lifetime_value"
    ACQUISITION_COST = "acquisition_cost"

@dataclass
class CustomerSegment:
    """Definition of a customer segment"""
    segment_id: str
    segment_name: str
    segmentation_type: SegmentationType
    criteria: Dict[str, Any]
    size: int
    description: str

@dataclass
class SegmentPerformance:
    """Performance metrics for a segment"""
    segment_id: str
    segment_name: str
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    sample_sizes: Dict[str, int]  # control and treatment sample sizes
    statistical_significance: Dict[str, bool]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class SegmentInsight:
    """Insights discovered from segment analysis"""
    insight_type: str
    segment_id: str
    metric: str
    description: str
    impact_magnitude: float
    confidence_level: float
    recommendation: str

@dataclass
class SegmentationAnalysis:
    """Complete segmentation analysis results"""
    experiment_id: str
    segmentation_strategy: str
    segments: List[CustomerSegment]
    segment_performances: List[SegmentPerformance]
    key_insights: List[SegmentInsight]
    personalization_opportunities: List[Dict[str, Any]]
    overall_summary: Dict[str, Any]

class SegmentAnalyzer:
    """Analyzer for customer segmentation and segment performance"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "min_segment_size": 100,
            "max_segments": 10,
            "significance_threshold": 0.05,
            "effect_size_threshold": 0.05,  # 5% minimum effect
            "confidence_level": 0.95,
            "auto_segmentation": True,
            "segment_metrics": ["conversion_rate", "revenue_per_user", "engagement_score"]
        }
    
    async def analyze_segments(
        self,
        experiment_id: str,
        customer_data: List[Dict[str, Any]],
        segmentation_strategy: Optional[str] = None
    ) -> SegmentationAnalysis:
        """
        Perform comprehensive segment analysis for an experiment
        """
        self.logger.info(f"Starting segment analysis for experiment {experiment_id}")
        
        # Create segments based on strategy
        if not segmentation_strategy:
            segments = await self._auto_create_segments(customer_data)
            strategy = "auto_behavioral_demographic"
        else:
            segments = await self._create_predefined_segments(customer_data, segmentation_strategy)
            strategy = segmentation_strategy
        
        # Analyze performance for each segment
        segment_performances = []
        for segment in segments:
            performance = await self._analyze_segment_performance(segment, customer_data)
            segment_performances.append(performance)
        
        # Generate insights
        key_insights = await self._generate_segment_insights(segments, segment_performances)
        
        # Identify personalization opportunities
        personalization_opportunities = await self._identify_personalization_opportunities(
            segments, segment_performances
        )
        
        # Generate overall summary
        overall_summary = await self._generate_overall_summary(segments, segment_performances)
        
        return SegmentationAnalysis(
            experiment_id=experiment_id,
            segmentation_strategy=strategy,
            segments=segments,
            segment_performances=segment_performances,
            key_insights=key_insights,
            personalization_opportunities=personalization_opportunities,
            overall_summary=overall_summary
        )
    
    async def _auto_create_segments(self, customer_data: List[Dict[str, Any]]) -> List[CustomerSegment]:
        """Automatically create segments using clustering and business logic"""
        
        segments = []
        
        # Demographic segments
        age_segments = self._create_age_segments(customer_data)
        segments.extend(age_segments)
        
        # Behavioral segments
        engagement_segments = self._create_engagement_segments(customer_data)
        segments.extend(engagement_segments)
        
        # Value-based segments
        value_segments = self._create_value_segments(customer_data)
        segments.extend(value_segments)
        
        # Geographic segments
        geo_segments = self._create_geographic_segments(customer_data)
        segments.extend(geo_segments)
        
        # Filter segments by minimum size
        valid_segments = [s for s in segments if s.size >= self.config["min_segment_size"]]
        
        # Limit number of segments
        if len(valid_segments) > self.config["max_segments"]:
            # Sort by size and take the largest segments
            valid_segments.sort(key=lambda x: x.size, reverse=True)
            valid_segments = valid_segments[:self.config["max_segments"]]
        
        return valid_segments
    
    def _create_age_segments(self, customer_data: List[Dict[str, Any]]) -> List[CustomerSegment]:
        """Create age-based demographic segments"""
        
        ages = [customer.get("age", 0) for customer in customer_data if customer.get("age")]
        
        if not ages:
            return []
        
        segments = []
        age_ranges = [
            ("18-25", 18, 25),
            ("26-35", 26, 35),
            ("36-45", 36, 45),
            ("46-55", 46, 55),
            ("56+", 56, 100)
        ]
        
        for range_name, min_age, max_age in age_ranges:
            segment_customers = [
                c for c in customer_data 
                if c.get("age", 0) >= min_age and c.get("age", 0) <= max_age
            ]
            
            if len(segment_customers) >= self.config["min_segment_size"]:
                segments.append(CustomerSegment(
                    segment_id=f"age_{range_name.replace('-', '_').replace('+', '_plus')}",
                    segment_name=f"Age {range_name}",
                    segmentation_type=SegmentationType.DEMOGRAPHIC,
                    criteria={"age_min": min_age, "age_max": max_age},
                    size=len(segment_customers),
                    description=f"Customers aged {range_name} years"
                ))
        
        return segments
    
    def _create_engagement_segments(self, customer_data: List[Dict[str, Any]]) -> List[CustomerSegment]:
        """Create engagement-based behavioral segments"""
        
        engagement_scores = [
            customer.get("engagement_score", 0) for customer in customer_data 
            if customer.get("engagement_score") is not None
        ]
        
        if not engagement_scores:
            return []
        
        # Calculate percentiles
        p33 = np.percentile(engagement_scores, 33)
        p66 = np.percentile(engagement_scores, 66)
        
        segments = []
        engagement_ranges = [
            ("high_engagement", p66, float('inf'), "Highly engaged users"),
            ("medium_engagement", p33, p66, "Moderately engaged users"),
            ("low_engagement", 0, p33, "Low engagement users")
        ]
        
        for segment_name, min_score, max_score, description in engagement_ranges:
            segment_customers = [
                c for c in customer_data 
                if c.get("engagement_score", 0) >= min_score and c.get("engagement_score", 0) < max_score
            ]
            
            if len(segment_customers) >= self.config["min_segment_size"]:
                segments.append(CustomerSegment(
                    segment_id=segment_name,
                    segment_name=segment_name.replace('_', ' ').title(),
                    segmentation_type=SegmentationType.BEHAVIORAL,
                    criteria={"engagement_min": min_score, "engagement_max": max_score},
                    size=len(segment_customers),
                    description=description
                ))
        
        return segments
    
    def _create_value_segments(self, customer_data: List[Dict[str, Any]]) -> List[CustomerSegment]:
        """Create value-based segments using customer lifetime value"""
        
        clv_values = [
            customer.get("lifetime_value", 0) for customer in customer_data 
            if customer.get("lifetime_value") is not None
        ]
        
        if not clv_values:
            return []
        
        # Calculate percentiles
        p20 = np.percentile(clv_values, 20)
        p50 = np.percentile(clv_values, 50)
        p80 = np.percentile(clv_values, 80)
        
        segments = []
        value_ranges = [
            ("vip_customers", p80, float('inf'), "High-value VIP customers"),
            ("valuable_customers", p50, p80, "Valuable customers"),
            ("regular_customers", p20, p50, "Regular customers"),
            ("low_value_customers", 0, p20, "Low-value customers")
        ]
        
        for segment_name, min_value, max_value, description in value_ranges:
            segment_customers = [
                c for c in customer_data 
                if c.get("lifetime_value", 0) >= min_value and c.get("lifetime_value", 0) < max_value
            ]
            
            if len(segment_customers) >= self.config["min_segment_size"]:
                segments.append(CustomerSegment(
                    segment_id=segment_name,
                    segment_name=segment_name.replace('_', ' ').title(),
                    segmentation_type=SegmentationType.VALUE_BASED,
                    criteria={"clv_min": min_value, "clv_max": max_value},
                    size=len(segment_customers),
                    description=description
                ))
        
        return segments
    
    def _create_geographic_segments(self, customer_data: List[Dict[str, Any]]) -> List[CustomerSegment]:
        """Create geographic segments"""
        
        locations = {}
        for customer in customer_data:
            location = customer.get("location", "unknown")
            if location != "unknown":
                locations[location] = locations.get(location, 0) + 1
        
        segments = []
        for location, count in locations.items():
            if count >= self.config["min_segment_size"]:
                segments.append(CustomerSegment(
                    segment_id=f"geo_{location.lower().replace(' ', '_')}",
                    segment_name=f"Location: {location}",
                    segmentation_type=SegmentationType.GEOGRAPHIC,
                    criteria={"location": location},
                    size=count,
                    description=f"Customers from {location}"
                ))
        
        return segments
    
    async def _create_predefined_segments(
        self, 
        customer_data: List[Dict[str, Any]], 
        strategy: str
    ) -> List[CustomerSegment]:
        """Create segments based on predefined strategy"""
        
        if strategy == "age_based":
            return self._create_age_segments(customer_data)
        elif strategy == "engagement_based":
            return self._create_engagement_segments(customer_data)
        elif strategy == "value_based":
            return self._create_value_segments(customer_data)
        elif strategy == "geographic":
            return self._create_geographic_segments(customer_data)
        else:
            return await self._auto_create_segments(customer_data)
    
    async def _analyze_segment_performance(
        self, 
        segment: CustomerSegment, 
        customer_data: List[Dict[str, Any]]
    ) -> SegmentPerformance:
        """Analyze performance metrics for a specific segment"""
        
        # Filter customers for this segment
        segment_customers = self._filter_customers_by_segment(customer_data, segment)
        
        # Split into control and treatment groups
        control_customers = [c for c in segment_customers if c.get("experiment_group") == "control"]
        treatment_customers = [c for c in segment_customers if c.get("experiment_group") == "treatment"]
        
        # Calculate metrics for control and treatment
        control_metrics = self._calculate_segment_metrics(control_customers)
        treatment_metrics = self._calculate_segment_metrics(treatment_customers)
        
        # Calculate statistical significance for each metric
        statistical_significance = {}
        effect_sizes = {}
        confidence_intervals = {}
        
        for metric in self.config["segment_metrics"]:
            if metric in control_metrics and metric in treatment_metrics:
                # Perform statistical test
                significance, effect_size, ci = await self._test_metric_significance(
                    control_customers, treatment_customers, metric
                )
                statistical_significance[metric] = significance
                effect_sizes[metric] = effect_size
                confidence_intervals[metric] = ci
        
        return SegmentPerformance(
            segment_id=segment.segment_id,
            segment_name=segment.segment_name,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            sample_sizes={
                "control": len(control_customers),
                "treatment": len(treatment_customers)
            },
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals
        )
    
    def _filter_customers_by_segment(
        self, 
        customer_data: List[Dict[str, Any]], 
        segment: CustomerSegment
    ) -> List[Dict[str, Any]]:
        """Filter customers that belong to a specific segment"""
        
        filtered_customers = []
        
        for customer in customer_data:
            matches = True
            
            # Check each criteria
            for key, value in segment.criteria.items():
                if key == "age_min":
                    if customer.get("age", 0) < value:
                        matches = False
                        break
                elif key == "age_max":
                    if customer.get("age", 0) > value:
                        matches = False
                        break
                elif key == "engagement_min":
                    if customer.get("engagement_score", 0) < value:
                        matches = False
                        break
                elif key == "engagement_max":
                    if customer.get("engagement_score", 0) >= value:
                        matches = False
                        break
                elif key == "clv_min":
                    if customer.get("lifetime_value", 0) < value:
                        matches = False
                        break
                elif key == "clv_max":
                    if customer.get("lifetime_value", 0) >= value:
                        matches = False
                        break
                elif key == "location":
                    if customer.get("location") != value:
                        matches = False
                        break
                else:
                    if customer.get(key) != value:
                        matches = False
                        break
            
            if matches:
                filtered_customers.append(customer)
        
        return filtered_customers
    
    def _calculate_segment_metrics(self, customers: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics for a group of customers"""
        
        if not customers:
            return {}
        
        metrics = {}
        
        # Conversion rate
        conversions = sum(1 for c in customers if c.get("converted", False))
        metrics["conversion_rate"] = conversions / len(customers)
        
        # Revenue per user
        total_revenue = sum(c.get("revenue", 0) for c in customers)
        metrics["revenue_per_user"] = total_revenue / len(customers)
        
        # Engagement score
        engagement_scores = [c.get("engagement_score", 0) for c in customers]
        metrics["engagement_score"] = np.mean(engagement_scores) if engagement_scores else 0
        
        # Retention rate
        retained = sum(1 for c in customers if c.get("retained", False))
        metrics["retention_rate"] = retained / len(customers)
        
        # Lifetime value
        clv_values = [c.get("lifetime_value", 0) for c in customers]
        metrics["lifetime_value"] = np.mean(clv_values) if clv_values else 0
        
        return metrics
    
    async def _test_metric_significance(
        self,
        control_customers: List[Dict[str, Any]],
        treatment_customers: List[Dict[str, Any]],
        metric: str
    ) -> Tuple[bool, float, Tuple[float, float]]:
        """Test statistical significance of a metric between control and treatment"""
        
        if not control_customers or not treatment_customers:
            return False, 0.0, (0.0, 0.0)
        
        # Get metric values
        control_values = []
        treatment_values = []
        
        for customer in control_customers:
            if metric == "conversion_rate":
                control_values.append(1 if customer.get("converted", False) else 0)
            elif metric == "revenue_per_user":
                control_values.append(customer.get("revenue", 0))
            elif metric == "engagement_score":
                control_values.append(customer.get("engagement_score", 0))
            elif metric == "retention_rate":
                control_values.append(1 if customer.get("retained", False) else 0)
            elif metric == "lifetime_value":
                control_values.append(customer.get("lifetime_value", 0))
        
        for customer in treatment_customers:
            if metric == "conversion_rate":
                treatment_values.append(1 if customer.get("converted", False) else 0)
            elif metric == "revenue_per_user":
                treatment_values.append(customer.get("revenue", 0))
            elif metric == "engagement_score":
                treatment_values.append(customer.get("engagement_score", 0))
            elif metric == "retention_rate":
                treatment_values.append(1 if customer.get("retained", False) else 0)
            elif metric == "lifetime_value":
                treatment_values.append(customer.get("lifetime_value", 0))
        
        if not control_values or not treatment_values:
            return False, 0.0, (0.0, 0.0)
        
        # Perform t-test (simplified)
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        control_std = np.std(control_values, ddof=1) if len(control_values) > 1 else 0
        treatment_std = np.std(treatment_values, ddof=1) if len(treatment_values) > 1 else 0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_values) - 1) * control_std**2 + 
                             (len(treatment_values) - 1) * treatment_std**2) / 
                            (len(control_values) + len(treatment_values) - 2))
        
        if pooled_std > 0:
            effect_size = (treatment_mean - control_mean) / pooled_std
        else:
            effect_size = 0
        
        # Simple significance test (t-test approximation)
        se_control = control_std / np.sqrt(len(control_values)) if control_std > 0 else 0
        se_treatment = treatment_std / np.sqrt(len(treatment_values)) if treatment_std > 0 else 0
        se_diff = np.sqrt(se_control**2 + se_treatment**2)
        
        if se_diff > 0:
            t_stat = (treatment_mean - control_mean) / se_diff
            # Simplified p-value approximation
            p_value = 2 * (1 - self._approximate_t_cdf(abs(t_stat), len(control_values) + len(treatment_values) - 2))
        else:
            p_value = 1.0
        
        is_significant = p_value < self.config["significance_threshold"]
        
        # Confidence interval for difference
        if se_diff > 0:
            margin_error = 1.96 * se_diff  # Approximate 95% CI
            diff = treatment_mean - control_mean
            ci = (diff - margin_error, diff + margin_error)
        else:
            ci = (0.0, 0.0)
        
        return is_significant, effect_size, ci
    
    def _approximate_t_cdf(self, t, df):
        """Approximate t-distribution CDF"""
        if df > 30:
            # Use normal approximation for large df
            return self._approximate_norm_cdf(t)
        else:
            # Simple approximation for t-distribution
            return self._approximate_norm_cdf(t * np.sqrt(df / (df + t*t)))
    
    def _approximate_norm_cdf(self, x):
        """Approximate standard normal CDF"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x * x / np.pi)))
    
    async def _generate_segment_insights(
        self,
        segments: List[CustomerSegment],
        performances: List[SegmentPerformance]
    ) -> List[SegmentInsight]:
        """Generate key insights from segment analysis"""
        
        insights = []
        
        for performance in performances:
            for metric in self.config["segment_metrics"]:
                if (metric in performance.statistical_significance and 
                    performance.statistical_significance[metric]):
                    
                    control_value = performance.control_metrics.get(metric, 0)
                    treatment_value = performance.treatment_metrics.get(metric, 0)
                    
                    if control_value > 0:
                        impact = (treatment_value - control_value) / control_value
                        
                        if abs(impact) >= self.config["effect_size_threshold"]:
                            insight_type = "positive_impact" if impact > 0 else "negative_impact"
                            
                            insights.append(SegmentInsight(
                                insight_type=insight_type,
                                segment_id=performance.segment_id,
                                metric=metric,
                                description=f"{performance.segment_name} shows {abs(impact):.1%} {'improvement' if impact > 0 else 'decline'} in {metric.replace('_', ' ')}",
                                impact_magnitude=abs(impact),
                                confidence_level=0.95,
                                recommendation=self._generate_segment_recommendation(performance, metric, impact)
                            ))
        
        # Sort insights by impact magnitude
        insights.sort(key=lambda x: x.impact_magnitude, reverse=True)
        
        return insights
    
    def _generate_segment_recommendation(
        self, 
        performance: SegmentPerformance, 
        metric: str, 
        impact: float
    ) -> str:
        """Generate recommendation for a segment"""
        
        if impact > 0.1:  # >10% improvement
            return f"Strongly recommend targeting {performance.segment_name} - shows significant {metric.replace('_', ' ')} improvement"
        elif impact > 0.05:  # >5% improvement
            return f"Consider prioritizing {performance.segment_name} for this optimization"
        elif impact < -0.1:  # >10% decline
            return f"Avoid deploying to {performance.segment_name} - shows significant negative impact"
        elif impact < -0.05:  # >5% decline
            return f"Use caution with {performance.segment_name} - shows declining performance"
        else:
            return f"Monitor {performance.segment_name} performance closely"
    
    async def _identify_personalization_opportunities(
        self,
        segments: List[CustomerSegment],
        performances: List[SegmentPerformance]
    ) -> List[Dict[str, Any]]:
        """Identify opportunities for personalization based on segment performance"""
        
        opportunities = []
        
        # Find segments with strong positive performance
        high_impact_segments = []
        for performance in performances:
            for metric in ["conversion_rate", "revenue_per_user"]:
                if (metric in performance.statistical_significance and 
                    performance.statistical_significance[metric]):
                    
                    control_value = performance.control_metrics.get(metric, 0)
                    treatment_value = performance.treatment_metrics.get(metric, 0)
                    
                    if control_value > 0:
                        impact = (treatment_value - control_value) / control_value
                        if impact > 0.1:  # >10% improvement
                            high_impact_segments.append({
                                "segment": performance.segment_name,
                                "segment_id": performance.segment_id,
                                "metric": metric,
                                "impact": impact,
                                "size": performance.sample_sizes["control"] + performance.sample_sizes["treatment"]
                            })
        
        # Generate personalization opportunities
        for segment_info in high_impact_segments:
            opportunities.append({
                "opportunity_type": "segment_personalization",
                "segment": segment_info["segment"],
                "description": f"Create personalized experience for {segment_info['segment']} segment",
                "expected_impact": f"{segment_info['impact']:.1%} improvement in {segment_info['metric'].replace('_', ' ')}",
                "potential_reach": segment_info["size"],
                "priority": "high" if segment_info["impact"] > 0.2 else "medium"
            })
        
        # Identify cross-segment patterns
        segment_types = {}
        for performance in performances:
            segment = next(s for s in segments if s.segment_id == performance.segment_id)
            seg_type = segment.segmentation_type
            if seg_type not in segment_types:
                segment_types[seg_type] = []
            segment_types[seg_type].append(performance)
        
        # Look for patterns within segmentation types
        for seg_type, seg_performances in segment_types.items():
            if len(seg_performances) > 1:
                opportunities.append({
                    "opportunity_type": "cross_segment_analysis",
                    "segmentation_type": seg_type.value,
                    "description": f"Analyze patterns across {seg_type.value} segments for deeper insights",
                    "segments_involved": [p.segment_name for p in seg_performances],
                    "priority": "medium"
                })
        
        return opportunities
    
    async def _generate_overall_summary(
        self,
        segments: List[CustomerSegment],
        performances: List[SegmentPerformance]
    ) -> Dict[str, Any]:
        """Generate overall summary of segment analysis"""
        
        total_customers = sum(s.size for s in segments)
        significant_segments = sum(
            1 for p in performances 
            if any(p.statistical_significance.get(m, False) for m in self.config["segment_metrics"])
        )
        
        # Calculate weighted average improvements
        weighted_improvements = {}
        total_weight = 0
        
        for performance in performances:
            segment_weight = performance.sample_sizes["control"] + performance.sample_sizes["treatment"]
            total_weight += segment_weight
            
            for metric in self.config["segment_metrics"]:
                if metric in performance.control_metrics and metric in performance.treatment_metrics:
                    control_value = performance.control_metrics[metric]
                    treatment_value = performance.treatment_metrics[metric]
                    
                    if control_value > 0:
                        improvement = (treatment_value - control_value) / control_value
                        if metric not in weighted_improvements:
                            weighted_improvements[metric] = 0
                        weighted_improvements[metric] += improvement * segment_weight
        
        # Normalize by total weight
        for metric in weighted_improvements:
            if total_weight > 0:
                weighted_improvements[metric] /= total_weight
        
        return {
            "total_segments_analyzed": len(segments),
            "total_customers": total_customers,
            "statistically_significant_segments": significant_segments,
            "significance_rate": significant_segments / len(segments) if segments else 0,
            "weighted_average_improvements": weighted_improvements,
            "best_performing_segment": self._find_best_performing_segment(performances),
            "segmentation_coverage": total_customers,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _find_best_performing_segment(self, performances: List[SegmentPerformance]) -> Optional[str]:
        """Find the best performing segment overall"""
        
        best_segment = None
        best_score = -float('inf')
        
        for performance in performances:
            # Calculate composite score based on multiple metrics
            score = 0
            metric_count = 0
            
            for metric in ["conversion_rate", "revenue_per_user"]:
                if (metric in performance.control_metrics and 
                    metric in performance.treatment_metrics and
                    performance.statistical_significance.get(metric, False)):
                    
                    control_value = performance.control_metrics[metric]
                    treatment_value = performance.treatment_metrics[metric]
                    
                    if control_value > 0:
                        improvement = (treatment_value - control_value) / control_value
                        score += improvement
                        metric_count += 1
            
            if metric_count > 0:
                average_score = score / metric_count
                if average_score > best_score:
                    best_score = average_score
                    best_segment = performance.segment_name
        
        return best_segment

# Demo functions
async def create_demo_customer_data() -> List[Dict[str, Any]]:
    """Create demo customer data for segment analysis"""
    
    np.random.seed(42)  # For reproducible results
    customers = []
    
    # Generate 2000 demo customers
    for i in range(2000):
        # Basic demographics
        age = np.random.normal(35, 12)
        age = max(18, min(70, age))  # Constrain age
        
        # Location based on realistic distribution
        locations = ["New York", "California", "Texas", "Florida", "Illinois"]
        location_weights = [0.25, 0.20, 0.15, 0.15, 0.25]
        location = np.random.choice(locations, p=location_weights)
        
        # Engagement score (correlated with age and location)
        base_engagement = 3.0
        if age < 30:
            base_engagement += 0.5
        if location in ["New York", "California"]:
            base_engagement += 0.3
        
        engagement_score = max(1, min(5, np.random.normal(base_engagement, 0.8)))
        
        # Lifetime value (correlated with engagement and age)
        base_clv = 200 + engagement_score * 50
        if age > 40:
            base_clv += 100
        clv = max(50, np.random.normal(base_clv, base_clv * 0.3))
        
        # Experiment group assignment
        experiment_group = "control" if i < 1000 else "treatment"
        
        # Performance metrics (treatment group gets boost for some segments)
        base_conversion_rate = 0.12
        base_revenue = 85
        
        # Boost for certain segments in treatment group
        if experiment_group == "treatment":
            if age < 35:  # Younger users respond better
                base_conversion_rate *= 1.25
                base_revenue *= 1.15
            if engagement_score > 3.5:  # High engagement users respond better
                base_conversion_rate *= 1.20
                base_revenue *= 1.12
            if location in ["New York", "California"]:  # Urban areas respond better
                base_conversion_rate *= 1.15
                base_revenue *= 1.08
        
        # Generate actual metrics with noise
        converted = np.random.random() < base_conversion_rate
        revenue = np.random.normal(base_revenue, base_revenue * 0.25) if converted else 0
        revenue = max(0, revenue)
        
        retained = np.random.random() < (0.7 + engagement_score * 0.05)
        
        customers.append({
            "customer_id": f"cust_{i:04d}",
            "age": int(age),
            "location": location,
            "engagement_score": engagement_score,
            "lifetime_value": clv,
            "experiment_group": experiment_group,
            "converted": converted,
            "revenue": revenue,
            "retained": retained
        })
    
    return customers

async def run_segment_analyzer_demo():
    """Run comprehensive segment analyzer demo"""
    
    print("Segment Analyzer Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = SegmentAnalyzer()
    
    # Create demo customer data
    print("Demo Experiment: Mobile App Redesign A/B Test")
    print("Analyzing customer segments and performance...")
    
    customer_data = await create_demo_customer_data()
    
    print(f"\nCustomer Data Summary:")
    print(f"  Total Customers: {len(customer_data):,}")
    
    control_customers = [c for c in customer_data if c["experiment_group"] == "control"]
    treatment_customers = [c for c in customer_data if c["experiment_group"] == "treatment"]
    
    print(f"  Control Group: {len(control_customers):,}")
    print(f"  Treatment Group: {len(treatment_customers):,}")
    
    # Overall performance
    control_conv = np.mean([c["converted"] for c in control_customers])
    treatment_conv = np.mean([c["converted"] for c in treatment_customers])
    print(f"  Overall Conversion - Control: {control_conv:.1%}, Treatment: {treatment_conv:.1%}")
    
    # Run segment analysis
    print(f"\nRunning Segment Analysis...")
    analysis = await analyzer.analyze_segments("mobile_app_redesign_001", customer_data)
    
    # Display results
    print(f"\nSegment Analysis Results")
    print("=" * 50)
    
    summary = analysis.overall_summary
    print(f"Analysis Summary:")
    print(f"  Total Segments: {summary['total_segments_analyzed']}")
    print(f"  Customers Analyzed: {summary['total_customers']:,}")
    print(f"  Significant Segments: {summary['statistically_significant_segments']}")
    print(f"  Significance Rate: {summary['significance_rate']:.1%}")
    if summary["best_performing_segment"]:
        print(f"  Best Performing Segment: {summary['best_performing_segment']}")
    
    print(f"\nWeighted Average Improvements:")
    for metric, improvement in summary["weighted_average_improvements"].items():
        print(f"  {metric.replace('_', ' ').title()}: {improvement:+.1%}")
    
    print(f"\nSegment Performance Details:")
    for performance in analysis.segment_performances:
        print(f"\n  {performance.segment_name}:")
        print(f"    Sample Size: Control={performance.sample_sizes['control']}, Treatment={performance.sample_sizes['treatment']}")
        
        for metric in ["conversion_rate", "revenue_per_user"]:
            if metric in performance.control_metrics:
                control_val = performance.control_metrics[metric]
                treatment_val = performance.treatment_metrics[metric]
                is_sig = performance.statistical_significance.get(metric, False)
                effect_size = performance.effect_sizes.get(metric, 0)
                
                if control_val > 0:
                    improvement = (treatment_val - control_val) / control_val
                    sig_marker = "[**]" if is_sig else "[ ]"
                    
                    print(f"    {metric.replace('_', ' ').title()}: {control_val:.3f} -> {treatment_val:.3f} ({improvement:+.1%}) {sig_marker}")
                    print(f"      Effect Size: {effect_size:.3f}")
    
    print(f"\nKey Insights:")
    for insight in analysis.key_insights[:5]:  # Show top 5 insights
        impact_emoji = "📈" if insight.insight_type == "positive_impact" else "📉"
        print(f"  {impact_emoji} {insight.description}")
        print(f"    Recommendation: {insight.recommendation}")
    
    print(f"\nPersonalization Opportunities:")
    for opportunity in analysis.personalization_opportunities:
        priority_emoji = "🔥" if opportunity.get("priority") == "high" else "💡"
        print(f"  {priority_emoji} {opportunity['description']}")
        if "expected_impact" in opportunity:
            print(f"    Expected Impact: {opportunity['expected_impact']}")
        if "potential_reach" in opportunity:
            print(f"    Potential Reach: {opportunity['potential_reach']:,} customers")

async def main():
    """Main function"""
    
    print("Phase 3C: Segment Analyzer")
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        await run_segment_analyzer_demo()
        
        print("\n" + "=" * 60)
        print("SEGMENT ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Successfully Analyzed:")
        print("  - Customer Segmentation")
        print("  - Segment-Specific Performance")
        print("  - Statistical Significance Testing")
        print("  - Cross-Segment Insights")
        print("  - Personalization Opportunities")
        print("  - Behavioral Pattern Detection")
        print("  - Value-Based Segmentation")
        print("  - Geographic Analysis")
        print()
        print("Next Steps:")
        print("1. Business Impact Analyzer - COMPLETED")
        print("2. ROI Calculator - COMPLETED") 
        print("3. Segment Analyzer - COMPLETED")
        print("4. Temporal Pattern Detector - FINAL")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())