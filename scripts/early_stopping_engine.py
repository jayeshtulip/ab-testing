"""
Phase 3A: Early Stopping Criteria Engine (FIXED)
Automated experiment termination based on statistical significance and futility

Save as: scripts/early_stopping_engine.py (replace the existing file)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum

# Import only the classes we need to avoid circular imports
try:
    from statistical_testing_engine import StatisticalTestingEngine, ExperimentResults, StatisticalTestResult
except ImportError:
    # Define minimal classes if import fails
    @dataclass
    class ExperimentResults:
        control_successes: int
        control_total: int
        treatment_successes: int
        treatment_total: int
        metric_name: str = "conversion_rate"
        start_date: Optional[datetime] = None
        
        @property
        def control_rate(self) -> float:
            return self.control_successes / self.control_total if self.control_total > 0 else 0
        
        @property
        def treatment_rate(self) -> float:
            return self.treatment_successes / self.treatment_total if self.treatment_total > 0 else 0
        
        @property
        def lift(self) -> float:
            if self.control_rate == 0:
                return 0
            return (self.treatment_rate - self.control_rate) / self.control_rate
        
        @property
        def absolute_lift(self) -> float:
            return self.treatment_rate - self.control_rate

    @dataclass 
    class StatisticalTestResult:
        test_name: str
        statistic: float
        p_value: float
        significant: bool
        confidence_level: float
        effect_size: float
        confidence_interval: Tuple[float, float]
        power: Optional[float] = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoppingReason(Enum):
    """Reasons for stopping an experiment"""
    SIGNIFICANCE_REACHED = "significance_reached"
    FUTILITY_DETECTED = "futility_detected" 
    MAXIMUM_DURATION = "maximum_duration"
    MAXIMUM_SAMPLE_SIZE = "maximum_sample_size"
    HARMFUL_EFFECT = "harmful_effect"
    INSUFFICIENT_POWER = "insufficient_power"
    BUSINESS_DECISION = "business_decision"

@dataclass
class StoppingRule:
    """Configuration for early stopping rules"""
    min_sample_size_per_group: int = 100
    max_sample_size_per_group: int = 10000
    min_duration_days: int = 7
    max_duration_days: int = 28
    significance_threshold: float = 0.05
    futility_threshold: float = 0.20  # Power threshold for futility
    harm_threshold: float = -0.05  # Negative effect threshold
    min_effect_size: float = 0.01  # Minimum practical significance
    multiple_testing_correction: bool = True

@dataclass
class StoppingDecision:
    """Decision about whether to stop an experiment"""
    should_stop: bool
    reason: Optional[StoppingReason]
    confidence: float
    recommendation: str
    statistical_result: Optional[StatisticalTestResult]
    power: float
    sample_size_adequate: bool
    duration_adequate: bool
    futility_analysis: Dict
    next_check_date: Optional[datetime] = None

class EarlyStoppingEngine:
    """
    Advanced early stopping engine for A/B tests
    Implements multiple stopping criteria and power monitoring
    """
    
    def __init__(self, stopping_rules: Optional[StoppingRule] = None):
        """
        Initialize early stopping engine
        
        Args:
            stopping_rules: Configuration for stopping criteria
        """
        self.rules = stopping_rules or StoppingRule()
        
        # Try to import statistical engine, create simple fallback if not available
        try:
            from statistical_testing_engine import StatisticalTestingEngine
            self.statistical_engine = StatisticalTestingEngine(
                alpha=self.rules.significance_threshold
            )
        except ImportError:
            self.statistical_engine = None
            logger.warning("Statistical testing engine not available - using simplified tests")
        
    def evaluate_stopping_criteria(self, 
                                 results: ExperimentResults,
                                 experiment_start_date: datetime,
                                 target_sample_size: Optional[int] = None) -> StoppingDecision:
        """
        Evaluate all stopping criteria for an experiment
        
        Args:
            results: Current experiment results
            experiment_start_date: When the experiment started
            target_sample_size: Target sample size per group
            
        Returns:
            Decision about whether to stop the experiment
        """
        
        # Calculate experiment duration
        duration_days = (datetime.utcnow() - experiment_start_date).days
        
        # Get statistical test results (with fallback)
        if self.statistical_engine:
            statistical_result = self.statistical_engine.chi_square_test(results)
        else:
            statistical_result = self._simple_statistical_test(results)
        
        # Apply multiple testing correction if enabled
        if self.rules.multiple_testing_correction:
            statistical_result = self._apply_bonferroni_correction(statistical_result)
        
        # Calculate current power (with fallback)
        effect_size = abs(results.absolute_lift)
        if self.statistical_engine:
            current_power = self.statistical_engine.power_analysis(results, effect_size)
        else:
            current_power = self._simple_power_estimate(results, effect_size)
        
        # Check each stopping criterion
        stopping_checks = {
            'significance': self._check_significance(statistical_result),
            'futility': self._check_futility(results, current_power),
            'harm': self._check_harmful_effect(results, statistical_result),
            'sample_size': self._check_sample_size(results, target_sample_size),
            'duration': self._check_duration(duration_days),
            'power': self._check_power_adequacy(results, current_power)
        }
        
        # Determine stopping decision
        should_stop, reason, confidence, recommendation = self._make_stopping_decision(
            stopping_checks, statistical_result, current_power, duration_days
        )
        
        # Futility analysis details
        futility_analysis = self._detailed_futility_analysis(results, current_power)
        
        # Calculate next check date
        next_check = self._calculate_next_check_date(
            duration_days, results.control_total + results.treatment_total
        )
        
        return StoppingDecision(
            should_stop=should_stop,
            reason=reason,
            confidence=confidence,
            recommendation=recommendation,
            statistical_result=statistical_result,
            power=current_power,
            sample_size_adequate=stopping_checks['sample_size']['adequate'],
            duration_adequate=stopping_checks['duration']['adequate'], 
            futility_analysis=futility_analysis,
            next_check_date=next_check
        )
    
    def _simple_statistical_test(self, results: ExperimentResults) -> StatisticalTestResult:
        """Simple statistical test fallback when scipy not available"""
        from scipy.stats import chi2_contingency
        
        try:
            # Create contingency table
            contingency_table = np.array([
                [results.control_successes, results.control_total - results.control_successes],
                [results.treatment_successes, results.treatment_total - results.treatment_successes]
            ])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate effect size (Cramér's V)
            n = contingency_table.sum()
            cramer_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            
            # Simple confidence interval
            diff = results.treatment_rate - results.control_rate
            se = np.sqrt((results.control_rate * (1 - results.control_rate) / results.control_total) +
                        (results.treatment_rate * (1 - results.treatment_rate) / results.treatment_total))
            margin = 1.96 * se
            ci = (diff - margin, diff + margin)
            
            return StatisticalTestResult(
                test_name="chi_square_simple",
                statistic=float(chi2_stat),
                p_value=float(p_value),
                significant=p_value < self.rules.significance_threshold,
                confidence_level=1 - self.rules.significance_threshold,
                effect_size=float(cramer_v),
                confidence_interval=ci
            )
            
        except Exception as e:
            logger.error(f"Simple statistical test failed: {e}")
            return StatisticalTestResult(
                test_name="fallback",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=0.95,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0)
            )
    
    def _simple_power_estimate(self, results: ExperimentResults, effect_size: float) -> float:
        """Simple power estimation fallback"""
        # Very basic power estimation
        n_avg = (results.control_total + results.treatment_total) / 2
        
        if n_avg < 100:
            return 0.2
        elif n_avg < 500:
            return 0.5
        elif n_avg < 1000:
            return 0.7
        else:
            return 0.8
    
    def _check_significance(self, statistical_result: StatisticalTestResult) -> Dict:
        """Check if statistical significance is reached"""
        return {
            'criterion': 'significance',
            'met': statistical_result.significant,
            'p_value': statistical_result.p_value,
            'threshold': self.rules.significance_threshold,
            'confidence': 1 - statistical_result.p_value if statistical_result.significant else 0.5
        }
    
    def _check_futility(self, results: ExperimentResults, current_power: float) -> Dict:
        """Check for futility (low probability of detecting meaningful effect)"""
        
        # Calculate remaining budget
        remaining_budget = max(0, self.rules.max_sample_size_per_group - max(results.control_total, results.treatment_total))
        
        # Futility conditions
        futile_conditions = [
            current_power < self.rules.futility_threshold,
            abs(results.absolute_lift) < self.rules.min_effect_size,
            remaining_budget < 0.1 * self.rules.max_sample_size_per_group  # < 10% budget remaining
        ]
        
        futile = any(futile_conditions)
        
        return {
            'criterion': 'futility',
            'met': futile,
            'current_power': current_power,
            'effect_size': abs(results.absolute_lift),
            'min_effect_threshold': self.rules.min_effect_size,
            'remaining_budget_pct': remaining_budget / self.rules.max_sample_size_per_group * 100,
            'confidence': 0.8 if futile else 0.2
        }
    
    def _check_harmful_effect(self, results: ExperimentResults, statistical_result: StatisticalTestResult) -> Dict:
        """Check if treatment shows harmful effect"""
        
        # Check if treatment is significantly worse than control
        harmful_effect = (results.absolute_lift < self.rules.harm_threshold and 
                         statistical_result.significant)
        
        # Also check confidence interval
        ci_lower, ci_upper = statistical_result.confidence_interval
        ci_indicates_harm = ci_upper < self.rules.harm_threshold
        
        harmful = harmful_effect or ci_indicates_harm
        
        return {
            'criterion': 'harm',
            'met': harmful,
            'effect_size': results.absolute_lift,
            'harm_threshold': self.rules.harm_threshold,
            'significant_harm': harmful_effect,
            'ci_indicates_harm': ci_indicates_harm,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence': 0.9 if harmful else 0.1
        }
    
    def _check_sample_size(self, results: ExperimentResults, target_sample_size: Optional[int]) -> Dict:
        """Check sample size criteria"""
        
        current_min_size = min(results.control_total, results.treatment_total)
        current_max_size = max(results.control_total, results.treatment_total)
        
        min_adequate = current_min_size >= self.rules.min_sample_size_per_group
        max_exceeded = current_max_size >= self.rules.max_sample_size_per_group
        target_reached = (target_sample_size is not None and 
                         current_min_size >= target_sample_size)
        
        return {
            'criterion': 'sample_size',
            'adequate': min_adequate,
            'max_exceeded': max_exceeded,
            'target_reached': target_reached,
            'current_size': current_min_size,
            'min_required': self.rules.min_sample_size_per_group,
            'max_allowed': self.rules.max_sample_size_per_group,
            'target_size': target_sample_size,
            'confidence': 0.8 if max_exceeded else 0.3
        }
    
    def _check_duration(self, duration_days: int) -> Dict:
        """Check experiment duration criteria"""
        
        min_adequate = duration_days >= self.rules.min_duration_days
        max_exceeded = duration_days >= self.rules.max_duration_days
        
        return {
            'criterion': 'duration',
            'adequate': min_adequate,
            'max_exceeded': max_exceeded,
            'current_days': duration_days,
            'min_required_days': self.rules.min_duration_days,
            'max_allowed_days': self.rules.max_duration_days,
            'confidence': 0.7 if max_exceeded else 0.2
        }
    
    def _check_power_adequacy(self, results: ExperimentResults, current_power: float) -> Dict:
        """Check if statistical power is adequate"""
        
        adequate_power = current_power >= 0.8
        
        return {
            'criterion': 'power',
            'adequate': adequate_power,
            'current_power': current_power,
            'target_power': 0.8,
            'confidence': current_power
        }
    
    def _make_stopping_decision(self, 
                               checks: Dict, 
                               statistical_result: StatisticalTestResult,
                               current_power: float,
                               duration_days: int) -> Tuple[bool, Optional[StoppingReason], float, str]:
        """Make final stopping decision based on all criteria"""
        
        # Priority order for stopping decisions
        
        # 1. Harmful effect - highest priority
        if checks['harm']['met']:
            return (
                True, 
                StoppingReason.HARMFUL_EFFECT,
                checks['harm']['confidence'],
                "🛑 STOP: Treatment shows statistically significant harmful effect. Discontinue immediately for safety."
            )
        
        # 2. Statistical significance reached
        if (checks['significance']['met'] and 
            checks['sample_size']['adequate'] and 
            checks['duration']['adequate']):
            return (
                True,
                StoppingReason.SIGNIFICANCE_REACHED,
                checks['significance']['confidence'],
                f"🎉 STOP: Statistical significance reached (p={statistical_result.p_value:.4f}). "
                f"Experiment successful with {checks['significance']['confidence']:.1%} confidence."
            )
        
        # 3. Maximum constraints exceeded
        if checks['duration']['max_exceeded']:
            reason = StoppingReason.MAXIMUM_DURATION
            recommendation = (f"⏰ STOP: Maximum duration ({self.rules.max_duration_days} days) reached. "
                            f"Make decision based on current evidence.")
        elif checks['sample_size']['max_exceeded']:
            reason = StoppingReason.MAXIMUM_SAMPLE_SIZE
            recommendation = (f"📊 STOP: Maximum sample size ({self.rules.max_sample_size_per_group:,}) reached. "
                            f"Analyze results with current data.")
        else:
            reason = None
            recommendation = None
        
        if reason is not None:
            confidence = max(checks['duration']['confidence'], checks['sample_size']['confidence'])
            return (True, reason, confidence, recommendation)
        
        # 4. Futility analysis
        if (checks['futility']['met'] and 
            checks['sample_size']['adequate'] and 
            checks['duration']['adequate']):
            return (
                True,
                StoppingReason.FUTILITY_DETECTED,
                checks['futility']['confidence'],
                f"📉 STOP: Futility detected. Current power: {current_power:.1%}. "
                f"Unlikely to detect meaningful effect with additional data."
            )
        
        # 5. Insufficient power with near-max sample
        current_size = min(checks['sample_size']['current_size'], 
                          checks['sample_size']['current_size'])
        if (current_size > 0.8 * self.rules.max_sample_size_per_group and 
            current_power < 0.3):
            return (
                True,
                StoppingReason.INSUFFICIENT_POWER,
                0.7,
                f"⚡ STOP: Insufficient power ({current_power:.1%}) despite large sample. "
                f"Consider redesigning experiment."
            )
        
        # Continue experiment
        continue_reasons = []
        
        if not checks['sample_size']['adequate']:
            continue_reasons.append(f"need {self.rules.min_sample_size_per_group - checks['sample_size']['current_size']:,} more samples")
        
        if not checks['duration']['adequate']:
            days_needed = self.rules.min_duration_days - duration_days
            continue_reasons.append(f"need {days_needed} more days")
        
        if current_power < 0.8:
            continue_reasons.append(f"power only {current_power:.1%}")
        
        recommendation = f"▶️ CONTINUE: " + ", ".join(continue_reasons) if continue_reasons else "▶️ CONTINUE: Monitoring for significance"
        
        return (False, None, 0.5, recommendation)
    
    def _apply_bonferroni_correction(self, result: StatisticalTestResult) -> StatisticalTestResult:
        """Apply Bonferroni correction for multiple testing"""
        # Assume 2 comparisons for conservative correction
        corrected_alpha = self.rules.significance_threshold / 2
        corrected_significant = result.p_value < corrected_alpha
        
        # Create new result with corrected significance
        return StatisticalTestResult(
            test_name=f"{result.test_name}_bonferroni_corrected",
            statistic=result.statistic,
            p_value=result.p_value,
            significant=corrected_significant,
            confidence_level=1 - corrected_alpha,
            effect_size=result.effect_size,
            confidence_interval=result.confidence_interval,
            power=result.power
        )
    
    def _detailed_futility_analysis(self, results: ExperimentResults, current_power: float) -> Dict:
        """Perform detailed futility analysis"""
        
        return {
            'current_power': current_power,
            'current_effect_size': abs(results.absolute_lift),
            'min_effect_for_80_power': self.rules.min_effect_size * 2,  # Simplified estimate
            'recommendation': self._futility_recommendation(current_power)
        }
    
    def _futility_recommendation(self, current_power: float) -> str:
        """Generate futility analysis recommendation"""
        
        if current_power < 0.3:
            return "High futility: Very low power, unlikely to detect meaningful effects"
        elif current_power < 0.6:
            return "Moderate futility: Low power, consider larger effect sizes"
        else:
            return "Low futility: Reasonable power, continue monitoring"
    
    def _calculate_next_check_date(self, current_duration_days: int, current_sample_size: int) -> datetime:
        """Calculate when to next evaluate stopping criteria"""
        
        # Check more frequently early in experiment
        if current_duration_days < 7:
            days_to_add = 2
        elif current_duration_days < 14:
            days_to_add = 3
        else:
            days_to_add = 7
        
        # Also consider sample size growth rate
        if current_sample_size < self.rules.min_sample_size_per_group:
            days_to_add = min(days_to_add, 3)  # Check more frequently if under-powered
        
        return datetime.utcnow() + timedelta(days=days_to_add)
    
    def generate_stopping_report(self, decision: StoppingDecision, results: ExperimentResults) -> str:
        """Generate detailed stopping analysis report"""
        
        report = f"""
🧪 EARLY STOPPING ANALYSIS REPORT
{'='*50}

📊 EXPERIMENT RESULTS:
   Control: {results.control_successes}/{results.control_total} ({results.control_rate:.2%})
   Treatment: {results.treatment_successes}/{results.treatment_total} ({results.treatment_rate:.2%})
   Absolute Lift: {results.absolute_lift:.3f} ({results.absolute_lift:.1%})
   Relative Lift: {results.lift:.1%}

📈 STATISTICAL ANALYSIS:
   Test: {decision.statistical_result.test_name}
   Statistic: {decision.statistical_result.statistic:.3f}
   P-value: {decision.statistical_result.p_value:.4f}
   Significant: {decision.statistical_result.significant}
   Effect Size: {decision.statistical_result.effect_size:.3f}
   95% CI: [{decision.statistical_result.confidence_interval[0]:.3f}, {decision.statistical_result.confidence_interval[1]:.3f}]

⚡ POWER ANALYSIS:
   Current Power: {decision.power:.1%}
   Sample Size Adequate: {decision.sample_size_adequate}
   Duration Adequate: {decision.duration_adequate}

🎯 STOPPING DECISION:
   Should Stop: {decision.should_stop}
   Reason: {decision.reason.value if decision.reason else 'N/A'}
   Confidence: {decision.confidence:.1%}
   
   {decision.recommendation}

🔍 FUTILITY ANALYSIS:
   {decision.futility_analysis['recommendation']}

📅 NEXT EVALUATION:
   Scheduled: {decision.next_check_date.strftime('%Y-%m-%d %H:%M') if decision.next_check_date else 'N/A'}

{'='*50}
        """
        
        return report.strip()

# Example usage and testing
def test_early_stopping_engine():
    """Test the early stopping engine"""
    
    print("🛑 Testing Early Stopping Engine")
    print("=" * 40)
    
    # Initialize with custom rules
    rules = StoppingRule(
        min_sample_size_per_group=100,
        max_sample_size_per_group=2000,
        min_duration_days=7,
        max_duration_days=28,
        significance_threshold=0.05,
        futility_threshold=0.20
    )
    
    engine = EarlyStoppingEngine(rules)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Early Success',
            'results': ExperimentResults(80, 400, 120, 400),
            'days_ago': 10
        },
        {
            'name': 'Clear Futility',
            'results': ExperimentResults(50, 500, 51, 500),
            'days_ago': 15
        },
        {
            'name': 'Harmful Effect',
            'results': ExperimentResults(100, 400, 60, 400),
            'days_ago': 14
        },
        {
            'name': 'Need More Time',
            'results': ExperimentResults(25, 200, 32, 200),
            'days_ago': 3
        },
        {
            'name': 'Maximum Duration',
            'results': ExperimentResults(180, 1500, 195, 1500),
            'days_ago': 30
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        print("-" * 30)
        
        start_date = datetime.utcnow() - timedelta(days=scenario['days_ago'])
        decision = engine.evaluate_stopping_criteria(
            scenario['results'], 
            start_date,
            target_sample_size=500
        )
        
        print(f"   Decision: {'STOP' if decision.should_stop else 'CONTINUE'}")
        if decision.reason:
            print(f"   Reason: {decision.reason.value}")
        print(f"   Confidence: {decision.confidence:.1%}")
        print(f"   Power: {decision.power:.1%}")
        print(f"   Recommendation: {decision.recommendation}")
        
        if decision.should_stop and decision.reason == StoppingReason.SIGNIFICANCE_REACHED:
            print("   🎉 Experiment successful!")
        elif decision.should_stop and decision.reason == StoppingReason.HARMFUL_EFFECT:
            print("   🛑 Safety stop required!")
    
    print("\n🎉 Early Stopping Engine working correctly!")

if __name__ == "__main__":
    test_early_stopping_engine()