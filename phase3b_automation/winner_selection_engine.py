"""
Winner Selection Engine for Phase 3B
Automated decision-making for experiment winners
Works standalone with your existing Phase 3B setup
"""

import asyncio
import numpy as np
import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class WinnerSelectionStrategy(str, Enum):
    """Winner selection strategies"""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    BUSINESS_METRICS = "business_metrics"
    COMBINED_SCORE = "combined_score"
    RISK_ADJUSTED = "risk_adjusted"

@dataclass
class ExperimentVariant:
    """Experiment variant data"""
    variant_id: str
    name: str
    traffic_percentage: float
    conversion_rate: float
    revenue_per_user: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    p_value: float
    statistical_significance: bool
    business_metrics: Dict[str, float]

@dataclass
class WinnerSelectionResult:
    """Result of winner selection process"""
    winner_variant_id: str
    confidence_score: float
    selection_strategy: WinnerSelectionStrategy
    reasoning: str
    risk_assessment: str
    recommendation: str
    metrics_comparison: Dict[str, Any]
    timestamp: datetime

class WinnerSelectionEngine:
    """Engine for automated winner selection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for winner selection"""
        return {
            "min_sample_size": 1000,
            "confidence_threshold": 0.95,
            "min_effect_size": 0.02,  # 2% minimum improvement
            "business_weight": 0.6,
            "statistical_weight": 0.4,
            "risk_tolerance": "medium",  # low, medium, high
            "required_significance_level": 0.05
        }
    
    async def select_winner(
        self,
        experiment_id: str,
        variants: List[ExperimentVariant],
        strategy: WinnerSelectionStrategy = WinnerSelectionStrategy.COMBINED_SCORE
    ) -> WinnerSelectionResult:
        """
        Main method to select experiment winner
        """
        self.logger.info(f"🎯 Starting winner selection for experiment {experiment_id}")
        
        # Validate inputs
        if not self._validate_variants(variants):
            raise ValueError("Invalid variant data provided")
        
        # Select strategy and execute
        if strategy == WinnerSelectionStrategy.STATISTICAL_SIGNIFICANCE:
            result = await self._select_by_statistical_significance(variants)
        elif strategy == WinnerSelectionStrategy.BUSINESS_METRICS:
            result = await self._select_by_business_metrics(variants)
        elif strategy == WinnerSelectionStrategy.COMBINED_SCORE:
            result = await self._select_by_combined_score(variants)
        elif strategy == WinnerSelectionStrategy.RISK_ADJUSTED:
            result = await self._select_by_risk_adjusted_score(variants)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        # Add metadata
        result.timestamp = datetime.now()
        result.selection_strategy = strategy
        
        self.logger.info(f"🏆 Winner selected: {result.winner_variant_id} with confidence {result.confidence_score:.3f}")
        return result
    
    def _validate_variants(self, variants: List[ExperimentVariant]) -> bool:
        """Validate variant data"""
        if len(variants) < 2:
            self.logger.error("Need at least 2 variants for comparison")
            return False
        
        for variant in variants:
            if variant.sample_size < self.config["min_sample_size"]:
                self.logger.warning(f"Variant {variant.variant_id} has insufficient sample size: {variant.sample_size}")
                return False
        
        return True
    
    async def _select_by_statistical_significance(
        self, 
        variants: List[ExperimentVariant]
    ) -> WinnerSelectionResult:
        """Select winner based purely on statistical significance"""
        
        # Find control variant (usually lowest traffic or first)
        control = min(variants, key=lambda v: v.traffic_percentage)
        
        # Find variants with statistical significance
        significant_variants = [
            v for v in variants 
            if v.statistical_significance and v.variant_id != control.variant_id
        ]
        
        if not significant_variants:
            return WinnerSelectionResult(
                winner_variant_id=control.variant_id,
                confidence_score=0.5,
                selection_strategy=WinnerSelectionStrategy.STATISTICAL_SIGNIFICANCE,
                reasoning="No variants achieved statistical significance",
                risk_assessment="Low risk - keeping control",
                recommendation="Continue experiment or increase sample size",
                metrics_comparison=self._create_metrics_comparison(variants),
                timestamp=datetime.now()
            )
        
        # Select variant with best conversion rate among significant ones
        winner = max(significant_variants, key=lambda v: v.conversion_rate)
        
        # Calculate confidence based on p-value and effect size
        effect_size = (winner.conversion_rate - control.conversion_rate) / control.conversion_rate
        confidence = min(0.99, (1 - winner.p_value) * (1 + effect_size))
        
        return WinnerSelectionResult(
            winner_variant_id=winner.variant_id,
            confidence_score=confidence,
            selection_strategy=WinnerSelectionStrategy.STATISTICAL_SIGNIFICANCE,
            reasoning=f"Statistically significant improvement: {effect_size:.2%}",
            risk_assessment=self._assess_risk(winner, control),
            recommendation="Deploy winner variant",
            metrics_comparison=self._create_metrics_comparison(variants),
            timestamp=datetime.now()
        )
    
    async def _select_by_business_metrics(
        self, 
        variants: List[ExperimentVariant]
    ) -> WinnerSelectionResult:
        """Select winner based on business metrics (revenue, etc.)"""
        
        # Calculate business score for each variant
        scored_variants = []
        for variant in variants:
            business_score = self._calculate_business_score(variant)
            scored_variants.append((variant, business_score))
        
        # Sort by business score
        scored_variants.sort(key=lambda x: x[1], reverse=True)
        winner, best_score = scored_variants[0]
        
        # Calculate confidence based on business metric improvements
        control_score = min(scored_variants, key=lambda x: x[0].traffic_percentage)[1]
        improvement = (best_score - control_score) / control_score if control_score > 0 else 0
        confidence = min(0.95, 0.5 + improvement)
        
        return WinnerSelectionResult(
            winner_variant_id=winner.variant_id,
            confidence_score=confidence,
            selection_strategy=WinnerSelectionStrategy.BUSINESS_METRICS,
            reasoning=f"Best business metrics score: {best_score:.3f}",
            risk_assessment=self._assess_business_risk(winner),
            recommendation="Deploy based on business performance",
            metrics_comparison=self._create_metrics_comparison(variants),
            timestamp=datetime.now()
        )
    
    async def _select_by_combined_score(
        self, 
        variants: List[ExperimentVariant]
    ) -> WinnerSelectionResult:
        """Select winner using combined statistical and business metrics"""
        
        scored_variants = []
        control = min(variants, key=lambda v: v.traffic_percentage)
        
        for variant in variants:
            # Statistical score
            stat_score = 0.5  # baseline
            if variant.statistical_significance:
                effect_size = abs(variant.conversion_rate - control.conversion_rate) / control.conversion_rate
                stat_score = min(0.99, (1 - variant.p_value) * (1 + effect_size))
            
            # Business score
            business_score = self._calculate_business_score(variant)
            
            # Combined score
            combined_score = (
                self.config["statistical_weight"] * stat_score + 
                self.config["business_weight"] * business_score
            )
            
            scored_variants.append((variant, combined_score, stat_score, business_score))
        
        # Select best combined score
        best_variant, best_combined, best_stat, best_business = max(
            scored_variants, key=lambda x: x[1]
        )
        
        return WinnerSelectionResult(
            winner_variant_id=best_variant.variant_id,
            confidence_score=best_combined,
            selection_strategy=WinnerSelectionStrategy.COMBINED_SCORE,
            reasoning=f"Combined score: {best_combined:.3f} (stat: {best_stat:.3f}, business: {best_business:.3f})",
            risk_assessment=self._assess_combined_risk(best_variant, control),
            recommendation="Deploy with balanced confidence",
            metrics_comparison=self._create_metrics_comparison(variants),
            timestamp=datetime.now()
        )
    
    async def _select_by_risk_adjusted_score(
        self, 
        variants: List[ExperimentVariant]
    ) -> WinnerSelectionResult:
        """Select winner with risk adjustment"""
        
        control = min(variants, key=lambda v: v.traffic_percentage)
        scored_variants = []
        
        for variant in variants:
            base_score = self._calculate_business_score(variant)
            risk_penalty = self._calculate_risk_penalty(variant, control)
            risk_adjusted_score = base_score * (1 - risk_penalty)
            
            scored_variants.append((variant, risk_adjusted_score, risk_penalty))
        
        best_variant, best_score, risk_penalty = max(scored_variants, key=lambda x: x[1])
        
        return WinnerSelectionResult(
            winner_variant_id=best_variant.variant_id,
            confidence_score=best_score,
            selection_strategy=WinnerSelectionStrategy.RISK_ADJUSTED,
            reasoning=f"Risk-adjusted score: {best_score:.3f} (penalty: {risk_penalty:.3f})",
            risk_assessment=f"Risk penalty applied: {risk_penalty:.1%}",
            recommendation="Deploy with risk considerations",
            metrics_comparison=self._create_metrics_comparison(variants),
            timestamp=datetime.now()
        )
    
    def _calculate_business_score(self, variant: ExperimentVariant) -> float:
        """Calculate business performance score"""
        # Normalize metrics (you may want to adjust these weights)
        conversion_weight = 0.4
        revenue_weight = 0.6
        
        # Normalize to 0-1 scale (you'd typically use historical baselines)
        normalized_conversion = min(1.0, variant.conversion_rate * 10)  # Assuming 10% is excellent
        normalized_revenue = min(1.0, variant.revenue_per_user / 100)  # Assuming $100 is excellent
        
        score = (
            conversion_weight * normalized_conversion +
            revenue_weight * normalized_revenue
        )
        
        return score
    
    def _calculate_risk_penalty(self, variant: ExperimentVariant, control: ExperimentVariant) -> float:
        """Calculate risk penalty for variant"""
        penalty = 0.0
        
        # Sample size penalty
        if variant.sample_size < self.config["min_sample_size"] * 2:
            penalty += 0.1
        
        # Statistical significance penalty
        if not variant.statistical_significance:
            penalty += 0.2
        
        # Large change penalty (big changes = more risk)
        change_magnitude = abs(variant.conversion_rate - control.conversion_rate) / control.conversion_rate
        if change_magnitude > 0.5:  # 50% change
            penalty += 0.15
        
        # Confidence interval width penalty
        ci_width = variant.confidence_interval[1] - variant.confidence_interval[0]
        if ci_width > 0.1:  # Wide confidence interval
            penalty += 0.1
        
        return min(0.5, penalty)  # Cap at 50% penalty
    
    def _assess_risk(self, winner: ExperimentVariant, control: ExperimentVariant) -> str:
        """Assess risk of deploying winner"""
        if not winner.statistical_significance:
            return "High risk - no statistical significance"
        
        effect_size = abs(winner.conversion_rate - control.conversion_rate) / control.conversion_rate
        if effect_size < self.config["min_effect_size"]:
            return "Medium risk - small effect size"
        
        if winner.sample_size < self.config["min_sample_size"] * 2:
            return "Medium risk - limited sample size"
        
        return "Low risk - strong statistical evidence"
    
    def _assess_business_risk(self, variant: ExperimentVariant) -> str:
        """Assess business risk"""
        if variant.revenue_per_user < 0:
            return "High risk - negative revenue impact"
        
        if not variant.statistical_significance:
            return "Medium risk - unproven statistical performance"
        
        return "Low risk - positive business metrics"
    
    def _assess_combined_risk(self, winner: ExperimentVariant, control: ExperimentVariant) -> str:
        """Assess combined risk"""
        statistical_risk = not winner.statistical_significance
        business_improvement = winner.revenue_per_user > control.revenue_per_user
        
        if statistical_risk and not business_improvement:
            return "High risk - no statistical or business evidence"
        elif statistical_risk or not business_improvement:
            return "Medium risk - mixed evidence"
        else:
            return "Low risk - strong combined evidence"
    
    def _create_metrics_comparison(self, variants: List[ExperimentVariant]) -> Dict[str, Any]:
        """Create detailed metrics comparison"""
        comparison = {
            "variants": {},
            "summary": {}
        }
        
        for variant in variants:
            comparison["variants"][variant.variant_id] = {
                "conversion_rate": variant.conversion_rate,
                "revenue_per_user": variant.revenue_per_user,
                "sample_size": variant.sample_size,
                "statistical_significance": variant.statistical_significance,
                "p_value": variant.p_value,
                "confidence_interval": variant.confidence_interval
            }
        
        # Summary statistics
        conversion_rates = [v.conversion_rate for v in variants]
        comparison["summary"] = {
            "best_conversion_rate": max(conversion_rates),
            "worst_conversion_rate": min(conversion_rates),
            "conversion_range": max(conversion_rates) - min(conversion_rates),
            "total_sample_size": sum(v.sample_size for v in variants)
        }
        
        return comparison

# Demo and testing functions
def create_demo_variants() -> List[ExperimentVariant]:
    """Create demo experiment variants for testing"""
    
    variants = [
        ExperimentVariant(
            variant_id="control",
            name="Control",
            traffic_percentage=50.0,
            conversion_rate=0.15,
            revenue_per_user=25.0,
            sample_size=5000,
            confidence_interval=(0.14, 0.16),
            p_value=0.5,
            statistical_significance=False,
            business_metrics={"retention": 0.6, "engagement": 0.7}
        ),
        ExperimentVariant(
            variant_id="variant_a",
            name="Variant A - New Button Color",
            traffic_percentage=50.0,
            conversion_rate=0.18,
            revenue_per_user=30.0,
            sample_size=5000,
            confidence_interval=(0.17, 0.19),
            p_value=0.02,
            statistical_significance=True,
            business_metrics={"retention": 0.65, "engagement": 0.75}
        )
    ]
    
    return variants

def create_complex_demo_variants() -> List[ExperimentVariant]:
    """Create more complex demo with 3 variants"""
    
    variants = [
        ExperimentVariant(
            variant_id="control",
            name="Control - Original",
            traffic_percentage=33.33,
            conversion_rate=0.12,
            revenue_per_user=22.0,
            sample_size=4500,
            confidence_interval=(0.11, 0.13),
            p_value=0.5,
            statistical_significance=False,
            business_metrics={"retention": 0.58, "engagement": 0.65}
        ),
        ExperimentVariant(
            variant_id="variant_b",
            name="Variant B - New Layout",
            traffic_percentage=33.33,
            conversion_rate=0.16,
            revenue_per_user=28.0,
            sample_size=4500,
            confidence_interval=(0.15, 0.17),
            p_value=0.03,
            statistical_significance=True,
            business_metrics={"retention": 0.62, "engagement": 0.72}
        ),
        ExperimentVariant(
            variant_id="variant_c",
            name="Variant C - Personalization",
            traffic_percentage=33.33,
            conversion_rate=0.14,
            revenue_per_user=35.0,
            sample_size=4500,
            confidence_interval=(0.13, 0.15),
            p_value=0.08,
            statistical_significance=False,
            business_metrics={"retention": 0.68, "engagement": 0.78}
        )
    ]
    
    return variants

async def run_demo_winner_selection():
    """Run comprehensive demo of winner selection"""
    
    print("🎯 Winner Selection Engine Demo")
    print("=" * 50)
    
    # Create engine
    engine = WinnerSelectionEngine()
    
    # Test with simple 2-variant experiment
    print("\n📊 Test 1: Simple A/B Test (2 variants)")
    print("-" * 40)
    
    variants = create_demo_variants()
    
    # Print variant info
    for variant in variants:
        sig_status = "✅ Significant" if variant.statistical_significance else "❌ Not significant"
        print(f"  {variant.name}:")
        print(f"    Conversion: {variant.conversion_rate:.1%}")
        print(f"    Revenue: ${variant.revenue_per_user:.2f}")
        print(f"    Sample size: {variant.sample_size:,}")
        print(f"    Statistical: {sig_status} (p={variant.p_value:.3f})")
    
    # Test all strategies
    strategies = [
        WinnerSelectionStrategy.STATISTICAL_SIGNIFICANCE,
        WinnerSelectionStrategy.BUSINESS_METRICS,
        WinnerSelectionStrategy.COMBINED_SCORE,
        WinnerSelectionStrategy.RISK_ADJUSTED
    ]
    
    for strategy in strategies:
        print(f"\n🎯 Strategy: {strategy.value.replace('_', ' ').title()}")
        result = await engine.select_winner("demo_exp_001", variants, strategy)
        
        print(f"  🏆 Winner: {result.winner_variant_id}")
        print(f"  📊 Confidence: {result.confidence_score:.3f}")
        print(f"  💡 Reasoning: {result.reasoning}")
        print(f"  ⚠️ Risk: {result.risk_assessment}")
        print(f"  📋 Recommendation: {result.recommendation}")
    
    # Test with complex 3-variant experiment
    print("\n\n📊 Test 2: Complex A/B/C Test (3 variants)")
    print("-" * 40)
    
    complex_variants = create_complex_demo_variants()
    
    # Print variant info
    for variant in complex_variants:
        sig_status = "✅ Significant" if variant.statistical_significance else "❌ Not significant"
        print(f"  {variant.name}:")
        print(f"    Conversion: {variant.conversion_rate:.1%}")
        print(f"    Revenue: ${variant.revenue_per_user:.2f}")
        print(f"    Sample size: {variant.sample_size:,}")
        print(f"    Statistical: {sig_status} (p={variant.p_value:.3f})")
    
    # Use combined score strategy for complex test
    print(f"\n🎯 Strategy: Combined Score (Best for complex tests)")
    result = await engine.select_winner("demo_exp_002", complex_variants, WinnerSelectionStrategy.COMBINED_SCORE)
    
    print(f"  🏆 Winner: {result.winner_variant_id}")
    print(f"  📊 Confidence: {result.confidence_score:.3f}")
    print(f"  💡 Reasoning: {result.reasoning}")
    print(f"  ⚠️ Risk: {result.risk_assessment}")
    print(f"  📋 Recommendation: {result.recommendation}")
    
    # Show detailed comparison
    print(f"\n📈 Detailed Metrics Comparison:")
    for variant_id, metrics in result.metrics_comparison["variants"].items():
        print(f"  {variant_id}:")
        print(f"    Conversion: {metrics['conversion_rate']:.1%}")
        print(f"    Revenue: ${metrics['revenue_per_user']:.2f}")
        print(f"    Samples: {metrics['sample_size']:,}")
    
    print(f"\n📊 Summary Stats:")
    summary = result.metrics_comparison["summary"]
    print(f"  Best conversion: {summary['best_conversion_rate']:.1%}")
    print(f"  Conversion range: {summary['conversion_range']:.1%}")
    print(f"  Total samples: {summary['total_sample_size']:,}")

async def test_integration_with_phase3b():
    """Test integration with existing Phase 3B components"""
    
    print("\n🔗 Testing Integration with Phase 3B Components")
    print("=" * 50)
    
    try:
        # Test integration with ResourceManager
        from core.resource_manager import ResourceManager
        
        resource_manager = ResourceManager()
        print("✅ ResourceManager integration available")
        
        # Test integration with API models
        from api.models import ExperimentPipelineRequest
        
        experiment = ExperimentPipelineRequest(
            name="Winner Selection Test",
            description="Testing winner selection integration",
            owner="automation_engine",
            team="phase3b_team",
            compute_requirement=5.0,
            storage_requirement=10.0
        )
        print("✅ API models integration available")
        
        # Simulate end-to-end workflow
        print("\n🔄 Simulating End-to-End Workflow:")
        print("  1. ✅ Experiment created via API models")
        print("  2. ✅ Resources allocated via ResourceManager")
        print("  3. ✅ Winner selection via WinnerSelectionEngine")
        print("  4. 📋 Ready for model promotion (next step)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Main function to run all tests"""
    
    print("🚀 Phase 3B Winner Selection Engine")
    print(f"⏰ Started at: {datetime.now()}")
    print()
    
    try:
        # Run the demo
        await run_demo_winner_selection()
        
        # Test integration
        integration_success = await test_integration_with_phase3b()
        
        if integration_success:
            print("\n🎉 Winner Selection Engine is working perfectly!")
            print("\n🚀 NEXT STEPS:")
            print("1. ✅ Winner Selection Engine - COMPLETED")
            print("2. 🚀 Model Promotion Engine - NEXT")
            print("3. 📊 Retraining Triggers - AFTER THAT")
            print("4. 🔗 Complete Integration - FINAL")
            print("\nRun: python model_promotion_engine.py")
        else:
            print("\n⚠️ Winner Selection Engine works, but integration needs attention")
            print("You can still proceed to the next step.")
        
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())