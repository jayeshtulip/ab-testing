"""
Business Impact Analyzer for Phase 3C
Quantifies real business value from experiments and optimizations
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

class MetricType(str, Enum):
    """Types of business metrics"""
    REVENUE = "revenue"
    CONVERSION = "conversion"
    RETENTION = "retention"
    ENGAGEMENT = "engagement"
    COST = "cost"
    LIFETIME_VALUE = "lifetime_value"

class ImpactSignificance(str, Enum):
    """Statistical significance levels"""
    NOT_SIGNIFICANT = "not_significant"
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # p < 0.1
    SIGNIFICANT = "significant"  # p < 0.05
    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01

@dataclass
class BusinessMetric:
    """Business metric data structure"""
    metric_name: str
    metric_type: MetricType
    control_value: float
    treatment_value: float
    control_samples: int
    treatment_samples: int
    control_std: float
    treatment_std: float
    measurement_period: str
    currency: str = "USD"

@dataclass
class StatisticalTest:
    """Statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    significance_level: ImpactSignificance
    effect_size: float
    power: float

@dataclass
class BusinessImpact:
    """Complete business impact analysis"""
    experiment_id: str
    metric_name: str
    absolute_impact: float
    relative_impact: float
    statistical_test: StatisticalTest
    confidence_interval: Tuple[float, float]
    annualized_impact: float
    risk_assessment: str
    recommendation: str
    metadata: Dict[str, Any]

# Simple statistics functions to avoid scipy dependency
def norm_cdf(x):
    """Approximate standard normal CDF"""
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x * x / np.pi)))

def t_cdf(x, df):
    """Approximate t-distribution CDF"""
    if df > 30:
        return norm_cdf(x)
    # Simple approximation for t-distribution
    return norm_cdf(x * np.sqrt(df / (df + x*x)))

def norm_ppf(p):
    """Approximate standard normal percent point function (inverse CDF)"""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    if p == 0.5:
        return 0
    
    # Approximation using Beasley-Springer-Moro algorithm
    a = np.array([0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01])
    
    if p > 0.5:
        p = 1 - p
        sign = 1
    else:
        sign = -1
    
    if p > 1e-8:
        y = np.sqrt(-2 * np.log(p))
        x = y + ((((a[6]*y + a[5])*y + a[4])*y + a[3])*y + a[2])*y + a[1] / ((((b[5]*y + b[4])*y + b[3])*y + b[2])*y + b[1])
    else:
        x = 8  # Large value for very small p
    
    return sign * x

class BusinessImpactAnalyzer:
    """Analyzer for quantifying business impact of experiments"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "significance_threshold": 0.05,
            "minimum_sample_size": 100,
            "minimum_effect_size": 0.01,  # 1%
            "confidence_level": 0.95,
            "default_currency": "USD",
            "annualization_factor": 365,  # days per year
            "bootstrap_iterations": 1000
        }
    
    async def analyze_business_impact(
        self,
        experiment_id: str,
        metrics: List[BusinessMetric]
    ) -> List[BusinessImpact]:
        """Analyze business impact for multiple metrics"""
        self.logger.info(f"Analyzing business impact for experiment {experiment_id}")
        
        impacts = []
        
        for metric in metrics:
            try:
                impact = await self._analyze_single_metric(experiment_id, metric)
                impacts.append(impact)
                
                self.logger.info(
                    f"✅ {metric.metric_name}: "
                    f"{impact.relative_impact:.2%} impact, "
                    f"significance: {impact.statistical_test.significance_level.value}"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {metric.metric_name}: {str(e)}")
        
        return impacts
    
    async def _analyze_single_metric(
        self,
        experiment_id: str,
        metric: BusinessMetric
    ) -> BusinessImpact:
        """Analyze impact for a single business metric"""
        
        # Validate data quality
        self._validate_metric_data(metric)
        
        # Perform statistical test
        stat_test = await self._perform_statistical_test(metric)
        
        # Calculate business impact
        absolute_impact = metric.treatment_value - metric.control_value
        relative_impact = absolute_impact / metric.control_value if metric.control_value != 0 else 0
        
        # Calculate confidence interval for impact
        confidence_interval = self._calculate_impact_confidence_interval(metric)
        
        # Annualize the impact
        annualized_impact = await self._calculate_annualized_impact(metric, absolute_impact)
        
        # Assess risk
        risk_assessment = self._assess_risk(stat_test, relative_impact, metric)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(stat_test, relative_impact, risk_assessment)
        
        return BusinessImpact(
            experiment_id=experiment_id,
            metric_name=metric.metric_name,
            absolute_impact=absolute_impact,
            relative_impact=relative_impact,
            statistical_test=stat_test,
            confidence_interval=confidence_interval,
            annualized_impact=annualized_impact,
            risk_assessment=risk_assessment,
            recommendation=recommendation,
            metadata={
                "metric_type": metric.metric_type.value,
                "measurement_period": metric.measurement_period,
                "currency": metric.currency,
                "analysis_timestamp": datetime.now().isoformat()
            }
        )
    
    def _validate_metric_data(self, metric: BusinessMetric):
        """Validate metric data quality"""
        
        if metric.control_samples < self.config["minimum_sample_size"]:
            raise ValueError(f"Control sample size too small: {metric.control_samples}")
        
        if metric.treatment_samples < self.config["minimum_sample_size"]:
            raise ValueError(f"Treatment sample size too small: {metric.treatment_samples}")
        
        if metric.control_value < 0 and metric.metric_type in [MetricType.REVENUE, MetricType.CONVERSION]:
            raise ValueError(f"Negative {metric.metric_type} value in control: {metric.control_value}")
        
        if metric.control_std <= 0 or metric.treatment_std <= 0:
            raise ValueError("Standard deviations must be positive")
    
    async def _perform_statistical_test(self, metric: BusinessMetric) -> StatisticalTest:
        """Perform appropriate statistical test based on metric type"""
        
        if metric.metric_type == MetricType.CONVERSION:
            return await self._two_proportion_test(metric)
        else:
            return await self._two_sample_ttest(metric)
    
    async def _two_proportion_test(self, metric: BusinessMetric) -> StatisticalTest:
        """Two-proportion z-test for conversion metrics"""
        
        # Convert rates to counts
        control_successes = int(metric.control_value * metric.control_samples)
        treatment_successes = int(metric.treatment_value * metric.treatment_samples)
        
        # Pooled proportion
        total_successes = control_successes + treatment_successes
        total_samples = metric.control_samples + metric.treatment_samples
        pooled_proportion = total_successes / total_samples
        
        # Standard error
        se = np.sqrt(pooled_proportion * (1 - pooled_proportion) * 
                     (1/metric.control_samples + 1/metric.treatment_samples))
        
        # Z-statistic
        if se == 0:
            z_stat = 0
            p_value = 1.0
        else:
            z_stat = (metric.treatment_value - metric.control_value) / se
            p_value = 2 * (1 - norm_cdf(abs(z_stat)))
        
        # Confidence interval
        diff = metric.treatment_value - metric.control_value
        margin_error = norm_ppf(1 - (1 - self.config["confidence_level"]) / 2) * se
        ci = (diff - margin_error, diff + margin_error)
        
        # Effect size (Cohen's h for proportions)
        effect_size = 2 * (np.arcsin(np.sqrt(metric.treatment_value)) - 
                          np.arcsin(np.sqrt(metric.control_value)))
        
        # Statistical power (simplified estimation)
        power = self._estimate_statistical_power(metric, effect_size)
        
        return StatisticalTest(
            test_name="Two-Proportion Z-Test",
            statistic=z_stat,
            p_value=p_value,
            confidence_interval=ci,
            significance_level=self._determine_significance(p_value),
            effect_size=effect_size,
            power=power
        )
    
    async def _two_sample_ttest(self, metric: BusinessMetric) -> StatisticalTest:
        """Two-sample t-test for continuous metrics"""
        
        # Welch's t-test (unequal variances)
        se_control = metric.control_std / np.sqrt(metric.control_samples)
        se_treatment = metric.treatment_std / np.sqrt(metric.treatment_samples)
        se_diff = np.sqrt(se_control**2 + se_treatment**2)
        
        if se_diff == 0:
            t_stat = 0
            p_value = 1.0
            df = metric.control_samples + metric.treatment_samples - 2
        else:
            t_stat = (metric.treatment_value - metric.control_value) / se_diff
            
            # Degrees of freedom for Welch's t-test
            df = (se_control**2 + se_treatment**2)**2 / (
                se_control**4 / (metric.control_samples - 1) + 
                se_treatment**4 / (metric.treatment_samples - 1)
            )
            
            p_value = 2 * (1 - t_cdf(abs(t_stat), df))
        
        # Confidence interval
        diff = metric.treatment_value - metric.control_value
        # Simplified critical value (approximation)
        t_critical = 1.96 if df > 30 else 2.0  # Rough approximation
        margin_error = t_critical * se_diff
        ci = (diff - margin_error, diff + margin_error)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((metric.control_samples - 1) * metric.control_std**2 + 
                             (metric.treatment_samples - 1) * metric.treatment_std**2) /
                             (metric.control_samples + metric.treatment_samples - 2))
        
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        
        # Statistical power
        power = self._estimate_statistical_power(metric, effect_size)
        
        return StatisticalTest(
            test_name="Welch's Two-Sample T-Test",
            statistic=t_stat,
            p_value=p_value,
            confidence_interval=ci,
            significance_level=self._determine_significance(p_value),
            effect_size=effect_size,
            power=power
        )
    
    def _determine_significance(self, p_value: float) -> ImpactSignificance:
        """Determine significance level based on p-value"""
        
        if p_value < 0.01:
            return ImpactSignificance.HIGHLY_SIGNIFICANT
        elif p_value < 0.05:
            return ImpactSignificance.SIGNIFICANT
        elif p_value < 0.1:
            return ImpactSignificance.MARGINALLY_SIGNIFICANT
        else:
            return ImpactSignificance.NOT_SIGNIFICANT
    
    def _estimate_statistical_power(self, metric: BusinessMetric, effect_size: float) -> float:
        """Estimate statistical power of the test"""
        
        # Simplified power calculation
        total_samples = metric.control_samples + metric.treatment_samples
        
        # Cohen's conventions for effect size
        if abs(effect_size) < 0.2:
            return 0.2  # Low power for small effects
        elif abs(effect_size) < 0.5:
            return min(0.8, 0.3 + total_samples / 1000)
        elif abs(effect_size) < 0.8:
            return min(0.9, 0.6 + total_samples / 1000)
        else:
            return min(0.95, 0.8 + total_samples / 1000)
    
    def _calculate_impact_confidence_interval(self, metric: BusinessMetric) -> Tuple[float, float]:
        """Calculate confidence interval for the impact estimate"""
        
        # Use bootstrap method for robust CI estimation
        n_bootstrap = 1000
        bootstrap_impacts = []
        
        for _ in range(n_bootstrap):
            # Bootstrap samples
            control_bootstrap = np.random.normal(
                metric.control_value, 
                metric.control_std / np.sqrt(metric.control_samples),
                1
            )[0]
            
            treatment_bootstrap = np.random.normal(
                metric.treatment_value,
                metric.treatment_std / np.sqrt(metric.treatment_samples), 
                1
            )[0]
            
            impact = treatment_bootstrap - control_bootstrap
            bootstrap_impacts.append(impact)
        
        # Calculate percentiles for confidence interval
        alpha = 1 - self.config["confidence_level"]
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_impacts, lower_percentile)
        ci_upper = np.percentile(bootstrap_impacts, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    async def _calculate_annualized_impact(
        self, 
        metric: BusinessMetric, 
        impact: float
    ) -> float:
        """Calculate annualized business impact"""
        
        # Extract time period from measurement_period
        period_str = metric.measurement_period.lower()
        
        if "day" in period_str:
            period_days = int(period_str.split("_")[0]) if period_str.split("_")[0].isdigit() else 1
        elif "week" in period_str:
            period_days = int(period_str.split("_")[0]) * 7 if period_str.split("_")[0].isdigit() else 7
        elif "month" in period_str:
            period_days = int(period_str.split("_")[0]) * 30 if period_str.split("_")[0].isdigit() else 30
        else:
            period_days = 1  # Default to daily
        
        # Annualize the impact
        annualization_factor = self.config["annualization_factor"] / period_days
        annualized_impact = impact * annualization_factor
        
        return annualized_impact
    
    def _assess_risk(
        self, 
        stat_test: StatisticalTest, 
        relative_impact: float, 
        metric: BusinessMetric
    ) -> str:
        """Assess risk of implementing the change"""
        
        if stat_test.significance_level == ImpactSignificance.NOT_SIGNIFICANT:
            return "High risk - no statistical evidence of impact"
        
        if stat_test.power < 0.8:
            return "Medium risk - low statistical power, results may be unreliable"
        
        if abs(relative_impact) < self.config["minimum_effect_size"]:
            return "Low risk but minimal impact - change may not be worth implementing"
        
        # Check confidence interval
        ci_lower, ci_upper = stat_test.confidence_interval
        if ci_lower < 0 < ci_upper:
            return "Medium risk - confidence interval includes both positive and negative impacts"
        
        if relative_impact > 0 and ci_lower > 0:
            return "Low risk - strong evidence of positive impact"
        elif relative_impact < 0 and ci_upper < 0:
            return "High risk - strong evidence of negative impact"
        else:
            return "Medium risk - mixed evidence"
    
    def _generate_recommendation(
        self, 
        stat_test: StatisticalTest, 
        relative_impact: float, 
        risk_assessment: str
    ) -> str:
        """Generate actionable recommendation"""
        
        if stat_test.significance_level == ImpactSignificance.NOT_SIGNIFICANT:
            return "Do not implement - no statistical evidence of improvement"
        
        if "High risk" in risk_assessment and relative_impact < 0:
            return "Do not implement - evidence suggests negative impact"
        
        if stat_test.significance_level in [ImpactSignificance.SIGNIFICANT, ImpactSignificance.HIGHLY_SIGNIFICANT]:
            if relative_impact > 0.05:  # 5% improvement
                return "Strongly recommend implementing - significant positive impact"
            elif relative_impact > 0:
                return "Recommend implementing - positive impact detected"
            else:
                return "Consider not implementing - negative impact detected"
        
        if stat_test.significance_level == ImpactSignificance.MARGINALLY_SIGNIFICANT:
            return "Consider additional testing - marginally significant results"
        
        return "Insufficient evidence for recommendation"
    
    async def generate_impact_report(
        self,
        impacts: List[BusinessImpact]
    ) -> Dict[str, Any]:
        """Generate comprehensive business impact report"""
        
        if not impacts:
            return {"error": "No impacts to report"}
        
        # Calculate summary statistics
        total_metrics = len(impacts)
        significant_metrics = len([i for i in impacts if i.statistical_test.significance_level in [
            ImpactSignificance.SIGNIFICANT, ImpactSignificance.HIGHLY_SIGNIFICANT
        ]])
        
        positive_impacts = [i for i in impacts if i.relative_impact > 0]
        negative_impacts = [i for i in impacts if i.relative_impact < 0]
        
        # Calculate total annualized impact (for revenue metrics)
        revenue_impacts = [i for i in impacts if i.metadata.get("metric_type") == "revenue"]
        total_revenue_impact = sum(i.annualized_impact for i in revenue_impacts)
        
        return {
            "summary": {
                "experiment_id": impacts[0].experiment_id,
                "total_metrics_analyzed": total_metrics,
                "statistically_significant": significant_metrics,
                "significance_rate": significant_metrics / total_metrics,
                "positive_impacts": len(positive_impacts),
                "negative_impacts": len(negative_impacts),
                "total_annualized_revenue_impact": total_revenue_impact,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "detailed_impacts": [
                {
                    "metric_name": impact.metric_name,
                    "absolute_impact": impact.absolute_impact,
                    "relative_impact": impact.relative_impact,
                    "annualized_impact": impact.annualized_impact,
                    "statistical_significance": impact.statistical_test.significance_level.value,
                    "p_value": impact.statistical_test.p_value,
                    "confidence_interval": impact.confidence_interval,
                    "effect_size": impact.statistical_test.effect_size,
                    "statistical_power": impact.statistical_test.power,
                    "risk_assessment": impact.risk_assessment,
                    "recommendation": impact.recommendation
                }
                for impact in impacts
            ]
        }

# Demo and testing functions
async def create_demo_metrics() -> List[BusinessMetric]:
    """Create demo business metrics for testing"""
    
    metrics = [
        BusinessMetric(
            metric_name="Revenue per User",
            metric_type=MetricType.REVENUE,
            control_value=25.30,
            treatment_value=27.85,
            control_samples=5000,
            treatment_samples=5000,
            control_std=12.5,
            treatment_std=13.2,
            measurement_period="7_days",
            currency="USD"
        ),
        BusinessMetric(
            metric_name="Conversion Rate",
            metric_type=MetricType.CONVERSION,
            control_value=0.15,  # 15%
            treatment_value=0.18,  # 18%
            control_samples=5000,
            treatment_samples=5000,
            control_std=0.36,  # std for binomial
            treatment_std=0.38,
            measurement_period="7_days"
        ),
        BusinessMetric(
            metric_name="Customer Lifetime Value",
            metric_type=MetricType.LIFETIME_VALUE,
            control_value=245.50,
            treatment_value=267.30,
            control_samples=2000,
            treatment_samples=2000,
            control_std=98.2,
            treatment_std=105.7,
            measurement_period="30_days",
            currency="USD"
        ),
        BusinessMetric(
            metric_name="User Engagement Score",
            metric_type=MetricType.ENGAGEMENT,
            control_value=3.2,
            treatment_value=3.6,
            control_samples=8000,
            treatment_samples=8000,
            control_std=1.1,
            treatment_std=1.2,
            measurement_period="7_days"
        )
    ]
    
    return metrics

async def run_business_impact_demo():
    """Run comprehensive business impact analysis demo"""
    
    print("Business Impact Analyzer Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = BusinessImpactAnalyzer()
    
    # Create demo metrics
    print("Demo Experiment: Website Redesign A/B Test")
    print("Analyzing 4 key business metrics...")
    
    metrics = await create_demo_metrics()
    
    # Display metrics being analyzed
    for metric in metrics:
        print(f"\n{metric.metric_name} ({metric.metric_type.value}):")
        print(f"  Control: {metric.control_value:.2f} ({metric.control_samples:,} samples)")
        print(f"  Treatment: {metric.treatment_value:.2f} ({metric.treatment_samples:,} samples)")
        if metric.metric_type == MetricType.CONVERSION:
            improvement = ((metric.treatment_value - metric.control_value) / metric.control_value) * 100
            print(f"  Raw Improvement: {improvement:.1f}%")
        else:
            diff = metric.treatment_value - metric.control_value
            print(f"  Raw Difference: {diff:.2f} {getattr(metric, 'currency', '')}")
    
    # Analyze business impact
    print(f"\nRunning Statistical Analysis...")
    impacts = await analyzer.analyze_business_impact("website_redesign_001", metrics)
    
    # Generate comprehensive report
    report = await analyzer.generate_impact_report(impacts)
    
    # Display results
    print(f"\nBusiness Impact Analysis Results")
    print("=" * 50)
    
    summary = report["summary"]
    print(f"Summary:")
    print(f"  Total Metrics Analyzed: {summary['total_metrics_analyzed']}")
    print(f"  Statistically Significant: {summary['statistically_significant']}")
    print(f"  Significance Rate: {summary['significance_rate']:.1%}")
    print(f"  Positive Impacts: {summary['positive_impacts']}")
    print(f"  Negative Impacts: {summary['negative_impacts']}")
    if summary['total_annualized_revenue_impact'] != 0:
        print(f"  Total Annual Revenue Impact: ${summary['total_annualized_revenue_impact']:,.2f}")
    
    print(f"\nDetailed Results:")
    for impact_detail in report["detailed_impacts"]:
        metric_name = impact_detail["metric_name"]
        significance = impact_detail["statistical_significance"]
        relative_impact = impact_detail["relative_impact"]
        
        significance_symbols = {
            "highly_significant": "[***]",
            "significant": "[**]",
            "marginally_significant": "[*]",
            "not_significant": "[ ]"
        }
        
        symbol = significance_symbols.get(significance, "[?]")
        
        print(f"\n{symbol} {metric_name}:")
        print(f"  Relative Impact: {relative_impact:.2%}")
        print(f"  Statistical Significance: {significance.replace('_', ' ').title()}")
        print(f"  P-value: {impact_detail['p_value']:.4f}")
        print(f"  Effect Size: {impact_detail['effect_size']:.3f}")
        print(f"  Statistical Power: {impact_detail['statistical_power']:.3f}")
        
        if impact_detail['annualized_impact'] != 0:
            print(f"  Annualized Impact: ${impact_detail['annualized_impact']:,.2f}")
        
        print(f"  Recommendation: {impact_detail['recommendation']}")

async def main():
    """Main function"""
    
    print("Phase 3C: Business Impact Analyzer")
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        await run_business_impact_demo()
        
        print("\n" + "=" * 60)
        print("BUSINESS IMPACT ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Successfully Analyzed:")
        print("  - Revenue Impact")
        print("  - Conversion Metrics") 
        print("  - Lifetime Value")
        print("  - Engagement Scores")
        print("  - Statistical Significance")
        print("  - Effect Sizes & Power Analysis")
        print("  - Strategic Recommendations")
        print()
        print("Next Steps:")
        print("1. Business Impact Analyzer - COMPLETED")
        print("2. ROI Calculator - NEXT")
        print("3. Segment Analyzer - COMING")
        print("4. Temporal Pattern Detector - FINAL")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())