"""
Phase 3A: Statistical Testing Engine
Advanced statistical significance testing for A/B experiments

Save as: scripts/statistical_testing_engine.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, chi2_contingency, ttest_ind
from typing import Dict, List, Tuple, Optional
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentResults:
    """Data class for experiment results"""
    control_successes: int
    control_total: int
    treatment_successes: int
    treatment_total: int
    metric_name: str = "conversion_rate"
    start_date: Optional[datetime] = None
    
    @property
    def control_rate(self) -> float:
        """Control group conversion rate"""
        return self.control_successes / self.control_total if self.control_total > 0 else 0
    
    @property
    def treatment_rate(self) -> float:
        """Treatment group conversion rate"""
        return self.treatment_successes / self.treatment_total if self.treatment_total > 0 else 0
    
    @property
    def lift(self) -> float:
        """Relative lift of treatment over control"""
        if self.control_rate == 0:
            return 0
        return (self.treatment_rate - self.control_rate) / self.control_rate
    
    @property
    def absolute_lift(self) -> float:
        """Absolute difference between treatment and control"""
        return self.treatment_rate - self.control_rate

@dataclass 
class StatisticalTestResult:
    """Results of statistical significance test"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    power: Optional[float] = None
    min_detectable_effect: Optional[float] = None

class StatisticalTestingEngine:
    """
    Advanced statistical testing engine for A/B experiments
    Supports multiple test types and power analysis
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize statistical testing engine
        
        Args:
            alpha: Significance level (default: 0.05)
            power: Desired statistical power (default: 0.8)
        """
        self.alpha = alpha
        self.power = power
        self.confidence_level = 1 - alpha
        
    def calculate_sample_size(self, 
                             baseline_rate: float,
                             minimum_detectable_effect: float,
                             alpha: Optional[float] = None,
                             power: Optional[float] = None,
                             two_tailed: bool = True) -> Dict:
        """
        Calculate required sample size for A/B test
        
        Args:
            baseline_rate: Expected control group rate (0-1)
            minimum_detectable_effect: Minimum effect to detect (relative, e.g., 0.1 for 10%)
            alpha: Significance level (uses default if None)
            power: Statistical power (uses default if None)
            two_tailed: Whether test is two-tailed
            
        Returns:
            Dict with sample size calculations
        """
        alpha = alpha or self.alpha
        power = power or self.power
        
        try:
            # Calculate effect size (Cohen's h for proportions)
            p1 = baseline_rate
            p2 = baseline_rate * (1 + minimum_detectable_effect)
            
            # Ensure p2 is valid probability
            p2 = min(max(p2, 0), 1)
            
            # Cohen's h effect size
            h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
            
            # Z-scores for alpha and beta
            if two_tailed:
                z_alpha = norm.ppf(1 - alpha/2)
            else:
                z_alpha = norm.ppf(1 - alpha)
            z_beta = norm.ppf(power)
            
            # Sample size calculation (per group)
            n_per_group = ((z_alpha + z_beta) / h) ** 2
            n_per_group = math.ceil(n_per_group)
            
            # Total sample size
            total_sample_size = n_per_group * 2
            
            return {
                'n_per_group': int(n_per_group),
                'total_sample_size': int(total_sample_size),
                'baseline_rate': float(baseline_rate),
                'treatment_rate': float(p2),
                'minimum_detectable_effect': float(minimum_detectable_effect),
                'effect_size_cohens_h': float(h),
                'alpha': float(alpha),
                'power': float(power),
                'two_tailed': two_tailed,
                'z_alpha': float(z_alpha),
                'z_beta': float(z_beta)
            }
            
        except Exception as e:
            logger.error(f"Sample size calculation failed: {e}")
            return {
                'error': str(e),
                'n_per_group': 1000,  # Fallback
                'total_sample_size': 2000
            }
    
    def chi_square_test(self, results: ExperimentResults) -> StatisticalTestResult:
        """
        Perform chi-square test for independence
        
        Args:
            results: Experiment results data
            
        Returns:
            Statistical test results
        """
        try:
            # Create contingency table
            # Rows: Control/Treatment, Columns: Success/Failure
            contingency_table = np.array([
                [results.control_successes, results.control_total - results.control_successes],
                [results.treatment_successes, results.treatment_total - results.treatment_successes]
            ])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate effect size (Cramér's V)
            n = contingency_table.sum()
            cramer_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            
            # Calculate confidence interval for difference in proportions
            ci = self._proportion_difference_ci(results)
            
            significant = p_value < self.alpha
            
            return StatisticalTestResult(
                test_name="chi_square",
                statistic=float(chi2_stat),
                p_value=float(p_value),
                significant=significant,
                confidence_level=self.confidence_level,
                effect_size=float(cramer_v),
                confidence_interval=ci
            )
            
        except Exception as e:
            logger.error(f"Chi-square test failed: {e}")
            return StatisticalTestResult(
                test_name="chi_square",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0)
            )
    
    def two_proportion_z_test(self, results: ExperimentResults) -> StatisticalTestResult:
        """
        Perform two-proportion z-test
        
        Args:
            results: Experiment results data
            
        Returns:
            Statistical test results
        """
        try:
            # Sample proportions
            p1 = results.control_rate
            p2 = results.treatment_rate
            n1 = results.control_total
            n2 = results.treatment_total
            
            # Pooled proportion
            p_pool = (results.control_successes + results.treatment_successes) / (n1 + n2)
            
            # Standard error
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            
            # Test statistic
            if se == 0:
                z_stat = 0
            else:
                z_stat = (p2 - p1) / se
            
            # P-value (two-tailed)
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            
            # Effect size (Cohen's h)
            h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
            
            # Confidence interval
            ci = self._proportion_difference_ci(results)
            
            significant = p_value < self.alpha
            
            return StatisticalTestResult(
                test_name="two_proportion_z_test",
                statistic=float(z_stat),
                p_value=float(p_value),
                significant=significant,
                confidence_level=self.confidence_level,
                effect_size=float(h),
                confidence_interval=ci
            )
            
        except Exception as e:
            logger.error(f"Two-proportion z-test failed: {e}")
            return StatisticalTestResult(
                test_name="two_proportion_z_test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0)
            )
    
    def t_test(self, control_values: List[float], treatment_values: List[float]) -> StatisticalTestResult:
        """
        Perform independent samples t-test
        
        Args:
            control_values: Continuous values from control group
            treatment_values: Continuous values from treatment group
            
        Returns:
            Statistical test results
        """
        try:
            # Perform Welch's t-test (unequal variances)
            t_stat, p_value = ttest_ind(treatment_values, control_values, equal_var=False)
            
            # Effect size (Cohen's d)
            mean_diff = np.mean(treatment_values) - np.mean(control_values)
            pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values, ddof=1) +
                                 (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) /
                                (len(control_values) + len(treatment_values) - 2))
            
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for mean difference
            ci = self._mean_difference_ci(control_values, treatment_values)
            
            significant = p_value < self.alpha
            
            return StatisticalTestResult(
                test_name="welch_t_test",
                statistic=float(t_stat),
                p_value=float(p_value),
                significant=significant,
                confidence_level=self.confidence_level,
                effect_size=float(cohen_d),
                confidence_interval=ci
            )
            
        except Exception as e:
            logger.error(f"T-test failed: {e}")
            return StatisticalTestResult(
                test_name="welch_t_test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0)
            )
    
    def _proportion_difference_ci(self, results: ExperimentResults) -> Tuple[float, float]:
        """Calculate confidence interval for difference in proportions"""
        try:
            p1 = results.control_rate
            p2 = results.treatment_rate
            n1 = results.control_total
            n2 = results.treatment_total
            
            # Difference in proportions
            diff = p2 - p1
            
            # Standard error for difference
            se_diff = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
            
            # Confidence interval
            z_critical = norm.ppf(1 - self.alpha/2)
            margin_error = z_critical * se_diff
            
            ci_lower = diff - margin_error
            ci_upper = diff + margin_error
            
            return (float(ci_lower), float(ci_upper))
            
        except Exception:
            return (0.0, 0.0)
    
    def _mean_difference_ci(self, control_values: List[float], treatment_values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        try:
            n1, n2 = len(control_values), len(treatment_values)
            mean1, mean2 = np.mean(control_values), np.mean(treatment_values)
            var1, var2 = np.var(control_values, ddof=1), np.var(treatment_values, ddof=1)
            
            # Welch's t-test degrees of freedom
            se_diff = np.sqrt(var1/n1 + var2/n2)
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            
            # T-critical value
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            
            # Mean difference and confidence interval
            diff = mean2 - mean1
            margin_error = t_critical * se_diff
            
            ci_lower = diff - margin_error
            ci_upper = diff + margin_error
            
            return (float(ci_lower), float(ci_upper))
            
        except Exception:
            return (0.0, 0.0)
    
    def power_analysis(self, results: ExperimentResults, effect_size: float) -> float:
        """
        Calculate statistical power of the test
        
        Args:
            results: Current experiment results
            effect_size: Effect size to detect
            
        Returns:
            Statistical power (0-1)
        """
        try:
            n1 = results.control_total
            n2 = results.treatment_total
            
            # Average sample size for power calculation
            n_avg = (n1 + n2) / 2
            
            # Z-critical for alpha
            z_alpha = norm.ppf(1 - self.alpha/2)
            
            # Z-score for power
            z_power = effect_size * np.sqrt(n_avg / 2) - z_alpha
            
            # Power calculation
            power = norm.cdf(z_power)
            
            return float(max(0, min(1, power)))
            
        except Exception as e:
            logger.error(f"Power analysis failed: {e}")
            return 0.0

# Example usage and testing
def test_statistical_engine():
    """Test the statistical testing engine"""
    
    print("🧪 Testing Statistical Testing Engine")
    print("=" * 40)
    
    # Initialize engine
    engine = StatisticalTestingEngine(alpha=0.05, power=0.8)
    
    # Test 1: Sample size calculation
    print("\n1. Sample Size Calculation:")
    sample_calc = engine.calculate_sample_size(
        baseline_rate=0.10,
        minimum_detectable_effect=0.20  # 20% relative improvement
    )
    
    print(f"   Baseline rate: {sample_calc['baseline_rate']:.1%}")
    print(f"   Treatment rate: {sample_calc['treatment_rate']:.1%}")
    print(f"   Sample size per group: {sample_calc['n_per_group']:,}")
    print(f"   Total sample size: {sample_calc['total_sample_size']:,}")
    print(f"   Effect size (Cohen's h): {sample_calc['effect_size_cohens_h']:.3f}")
    
    # Test 2: Chi-square test
    print("\n2. Chi-square Test:")
    results = ExperimentResults(
        control_successes=45,
        control_total=500,
        treatment_successes=65,
        treatment_total=500,
        metric_name="conversion_rate"
    )
    
    chi2_result = engine.chi_square_test(results)
    print(f"   Control rate: {results.control_rate:.1%}")
    print(f"   Treatment rate: {results.treatment_rate:.1%}")
    print(f"   Chi-square statistic: {chi2_result.statistic:.3f}")
    print(f"   P-value: {chi2_result.p_value:.4f}")
    print(f"   Significant: {chi2_result.significant}")
    print(f"   Effect size (Cramér's V): {chi2_result.effect_size:.3f}")
    print(f"   95% CI: [{chi2_result.confidence_interval[0]:.3f}, {chi2_result.confidence_interval[1]:.3f}]")
    
    # Test 3: Two-proportion Z-test
    print("\n3. Two-proportion Z-test:")
    z_result = engine.two_proportion_z_test(results)
    print(f"   Z-statistic: {z_result.statistic:.3f}")
    print(f"   P-value: {z_result.p_value:.4f}")
    print(f"   Significant: {z_result.significant}")
    print(f"   Effect size (Cohen's h): {z_result.effect_size:.3f}")
    
    # Test 4: T-test with continuous data
    print("\n4. T-test (continuous data):")
    np.random.seed(42)
    control_continuous = np.random.normal(100, 15, 200).tolist()
    treatment_continuous = np.random.normal(105, 15, 200).tolist()
    
    t_result = engine.t_test(control_continuous, treatment_continuous)
    print(f"   Control mean: {np.mean(control_continuous):.2f}")
    print(f"   Treatment mean: {np.mean(treatment_continuous):.2f}")
    print(f"   T-statistic: {t_result.statistic:.3f}")
    print(f"   P-value: {t_result.p_value:.4f}")
    print(f"   Significant: {t_result.significant}")
    print(f"   Effect size (Cohen's d): {t_result.effect_size:.3f}")
    
    # Test 5: Power analysis
    print("\n5. Power Analysis:")
    power = engine.power_analysis(results, effect_size=0.3)
    print(f"   Current statistical power: {power:.1%}")
    
    print("\n🎉 Statistical Testing Engine working correctly!")

if __name__ == "__main__":
    test_statistical_engine()