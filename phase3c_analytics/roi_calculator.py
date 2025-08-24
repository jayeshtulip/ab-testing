"""
ROI Calculator for Phase 3C
Comprehensive financial analysis and ROI calculations for experiments
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

class CostCategory(str, Enum):
    """Categories of experiment costs"""
    DEVELOPMENT = "development"
    INFRASTRUCTURE = "infrastructure"
    PERSONNEL = "personnel"
    TOOLING = "tooling"
    MARKETING = "marketing"
    OPPORTUNITY_COST = "opportunity_cost"

class RevenueModel(str, Enum):
    """Revenue attribution models"""
    LINEAR = "linear"
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    TIME_DECAY = "time_decay"
    DATA_DRIVEN = "data_driven"

@dataclass
class ExperimentCost:
    """Individual cost component"""
    category: CostCategory
    description: str
    amount: float
    currency: str
    is_recurring: bool
    frequency_days: Optional[int] = None  # For recurring costs
    allocation_percentage: float = 1.0  # % allocated to this experiment

@dataclass
class RevenueStream:
    """Revenue stream from experiment"""
    stream_name: str
    baseline_revenue: float
    uplift_revenue: float
    confidence_interval: Tuple[float, float]
    attribution_model: RevenueModel
    time_to_realize_days: int
    revenue_duration_days: int
    currency: str

@dataclass
class ROIAnalysis:
    """Complete ROI analysis results"""
    experiment_id: str
    total_costs: float
    total_revenue: float
    net_benefit: float
    roi_percentage: float
    payback_period_days: int
    npv: float
    irr: float
    break_even_point: datetime
    confidence_interval: Tuple[float, float]
    risk_metrics: Dict[str, float]
    sensitivity_analysis: Dict[str, Any]

@dataclass
class FinancialProjection:
    """Financial projection over time"""
    period_start: datetime
    period_end: datetime
    cumulative_costs: float
    cumulative_revenue: float
    net_cash_flow: float
    roi_to_date: float

class ROICalculator:
    """Calculator for experiment ROI and financial analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "discount_rate": 0.08,  # 8% annual discount rate
            "time_horizon_days": 365,  # 1 year analysis period
            "confidence_level": 0.95,
            "monte_carlo_simulations": 10000,
            "default_currency": "USD",
            "risk_free_rate": 0.02,  # 2% risk-free rate
            "cost_inflation_rate": 0.03  # 3% annual cost inflation
        }
    
    async def calculate_roi(
        self,
        experiment_id: str,
        costs: List[ExperimentCost],
        revenue_streams: List[RevenueStream],
        analysis_period_days: Optional[int] = None
    ) -> ROIAnalysis:
        """
        Calculate comprehensive ROI analysis for an experiment
        """
        self.logger.info(f"Calculating ROI for experiment {experiment_id}")
        
        analysis_period = analysis_period_days or self.config["time_horizon_days"]
        
        # Calculate total costs
        total_costs = await self._calculate_total_costs(costs, analysis_period)
        
        # Calculate total revenue
        total_revenue, revenue_confidence = await self._calculate_total_revenue(revenue_streams, analysis_period)
        
        # Calculate basic ROI metrics
        net_benefit = total_revenue - total_costs
        roi_percentage = (net_benefit / total_costs * 100) if total_costs > 0 else 0
        
        # Calculate payback period
        payback_period = await self._calculate_payback_period(costs, revenue_streams)
        
        # Calculate NPV and IRR
        npv = await self._calculate_npv(costs, revenue_streams, analysis_period)
        irr = await self._calculate_irr(costs, revenue_streams, analysis_period)
        
        # Calculate break-even point
        break_even_point = await self._calculate_break_even_point(costs, revenue_streams)
        
        # Risk analysis
        risk_metrics = await self._calculate_risk_metrics(costs, revenue_streams)
        
        # Sensitivity analysis
        sensitivity_analysis = await self._perform_sensitivity_analysis(costs, revenue_streams)
        
        # ROI confidence interval
        roi_confidence = await self._calculate_roi_confidence_interval(costs, revenue_streams)
        
        return ROIAnalysis(
            experiment_id=experiment_id,
            total_costs=total_costs,
            total_revenue=total_revenue,
            net_benefit=net_benefit,
            roi_percentage=roi_percentage,
            payback_period_days=payback_period,
            npv=npv,
            irr=irr,
            break_even_point=break_even_point,
            confidence_interval=roi_confidence,
            risk_metrics=risk_metrics,
            sensitivity_analysis=sensitivity_analysis
        )
    
    async def _calculate_total_costs(
        self, 
        costs: List[ExperimentCost], 
        analysis_period_days: int
    ) -> float:
        """Calculate total costs over the analysis period"""
        
        total_cost = 0.0
        
        for cost in costs:
            allocated_cost = cost.amount * cost.allocation_percentage
            
            if not cost.is_recurring:
                # One-time cost
                total_cost += allocated_cost
            else:
                # Recurring cost
                if cost.frequency_days:
                    recurrences = analysis_period_days / cost.frequency_days
                    # Apply cost inflation for future periods
                    for i in range(int(recurrences)):
                        inflation_factor = (1 + self.config["cost_inflation_rate"]) ** (i / 365)
                        total_cost += allocated_cost * inflation_factor
        
        return total_cost
    
    async def _calculate_total_revenue(
        self, 
        revenue_streams: List[RevenueStream], 
        analysis_period_days: int
    ) -> Tuple[float, Tuple[float, float]]:
        """Calculate total revenue and confidence interval"""
        
        total_revenue = 0.0
        revenue_variances = []
        
        for stream in revenue_streams:
            # Calculate revenue for this stream over analysis period
            revenue_days = min(stream.revenue_duration_days, 
                             analysis_period_days - stream.time_to_realize_days)
            
            if revenue_days > 0:
                # Apply time decay if specified
                if stream.attribution_model == RevenueModel.TIME_DECAY:
                    decay_factor = np.exp(-0.1 * stream.time_to_realize_days / 30)  # Monthly decay
                    effective_uplift = stream.uplift_revenue * decay_factor
                else:
                    effective_uplift = stream.uplift_revenue
                
                # Calculate revenue over the period
                daily_revenue = effective_uplift / max(1, stream.revenue_duration_days)
                stream_revenue = daily_revenue * revenue_days
                
                total_revenue += stream_revenue
                
                # Calculate variance for confidence intervals
                ci_lower, ci_upper = stream.confidence_interval
                variance = ((ci_upper - ci_lower) / 4) ** 2  # Rough approximation
                revenue_variances.append(variance)
        
        # Calculate overall confidence interval
        total_variance = sum(revenue_variances)
        total_std = np.sqrt(total_variance)
        
        # 95% confidence interval
        z_score = 1.96
        confidence_interval = (
            total_revenue - z_score * total_std,
            total_revenue + z_score * total_std
        )
        
        return total_revenue, confidence_interval
    
    async def _calculate_payback_period(
        self, 
        costs: List[ExperimentCost], 
        revenue_streams: List[RevenueStream]
    ) -> int:
        """Calculate payback period in days"""
        
        # Simulate cash flows day by day
        cumulative_costs = 0.0
        cumulative_revenue = 0.0
        
        # Calculate initial costs
        for cost in costs:
            if not cost.is_recurring:
                cumulative_costs += cost.amount * cost.allocation_percentage
        
        # Find when cumulative revenue exceeds cumulative costs
        for day in range(1, 3650):  # Check up to 10 years
            # Add daily costs
            for cost in costs:
                if cost.is_recurring and cost.frequency_days:
                    if day % cost.frequency_days == 0:
                        cumulative_costs += cost.amount * cost.allocation_percentage
            
            # Add daily revenue
            for stream in revenue_streams:
                if day >= stream.time_to_realize_days:
                    revenue_day = day - stream.time_to_realize_days
                    if revenue_day < stream.revenue_duration_days:
                        daily_revenue = stream.uplift_revenue / stream.revenue_duration_days
                        cumulative_revenue += daily_revenue
            
            # Check if we've reached payback
            if cumulative_revenue >= cumulative_costs:
                return day
        
        return -1  # Payback not achieved within 10 years
    
    async def _calculate_npv(
        self, 
        costs: List[ExperimentCost], 
        revenue_streams: List[RevenueStream], 
        analysis_period_days: int
    ) -> float:
        """Calculate Net Present Value"""
        
        npv = 0.0
        daily_discount_rate = (1 + self.config["discount_rate"]) ** (1/365) - 1
        
        # Calculate NPV by discounting daily cash flows
        for day in range(analysis_period_days):
            daily_cash_flow = 0.0
            
            # Subtract daily costs
            for cost in costs:
                if not cost.is_recurring and day == 0:
                    daily_cash_flow -= cost.amount * cost.allocation_percentage
                elif cost.is_recurring and cost.frequency_days and day % cost.frequency_days == 0:
                    daily_cash_flow -= cost.amount * cost.allocation_percentage
            
            # Add daily revenue
            for stream in revenue_streams:
                if day >= stream.time_to_realize_days:
                    revenue_day = day - stream.time_to_realize_days
                    if revenue_day < stream.revenue_duration_days:
                        daily_revenue = stream.uplift_revenue / stream.revenue_duration_days
                        daily_cash_flow += daily_revenue
            
            # Discount cash flow
            discount_factor = 1 / (1 + daily_discount_rate) ** day
            npv += daily_cash_flow * discount_factor
        
        return npv
    
    async def _calculate_irr(
        self, 
        costs: List[ExperimentCost], 
        revenue_streams: List[RevenueStream], 
        analysis_period_days: int
    ) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method"""
        
        # Generate cash flow series
        cash_flows = []
        for day in range(analysis_period_days):
            daily_cash_flow = 0.0
            
            # Subtract costs
            for cost in costs:
                if not cost.is_recurring and day == 0:
                    daily_cash_flow -= cost.amount * cost.allocation_percentage
                elif cost.is_recurring and cost.frequency_days and day % cost.frequency_days == 0:
                    daily_cash_flow -= cost.amount * cost.allocation_percentage
            
            # Add revenue
            for stream in revenue_streams:
                if day >= stream.time_to_realize_days:
                    revenue_day = day - stream.time_to_realize_days
                    if revenue_day < stream.revenue_duration_days:
                        daily_revenue = stream.uplift_revenue / stream.revenue_duration_days
                        daily_cash_flow += daily_revenue
            
            cash_flows.append(daily_cash_flow)
        
        # Check if we have positive and negative cash flows
        has_negative = any(cf < 0 for cf in cash_flows)
        has_positive = any(cf > 0 for cf in cash_flows)
        
        if not (has_negative and has_positive):
            return 0.0  # Cannot calculate IRR without both positive and negative cash flows
        
        # Use Newton-Raphson method to find IRR
        def npv_function(rate):
            try:
                if rate <= -1:
                    return float('inf')
                return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
            except (OverflowError, ZeroDivisionError):
                return float('inf')
        
        def npv_derivative(rate):
            try:
                if rate <= -1:
                    return float('inf')
                return sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
            except (OverflowError, ZeroDivisionError):
                return float('inf')
        
        # Initial guess - use a reasonable starting point
        irr = 0.1  # 10% daily rate as starting guess
        
        for iteration in range(50):  # Reduced iterations
            try:
                npv_val = npv_function(irr)
                npv_deriv = npv_derivative(irr)
                
                if abs(npv_val) < 1e-6 or abs(irr) > 10:  # Convergence or unreasonable rate
                    break
                
                if abs(npv_deriv) < 1e-12:  # Avoid division by very small numbers
                    break
                
                irr_new = irr - npv_val / npv_deriv
                
                # Constrain the IRR to reasonable bounds
                irr_new = max(-0.99, min(irr_new, 10))
                
                if abs(irr_new - irr) < 1e-8:
                    break
                
                irr = irr_new
                
            except (OverflowError, ZeroDivisionError, ValueError):
                # If calculation fails, use a simple approximation
                total_investment = sum(cf for cf in cash_flows if cf < 0)
                total_return = sum(cf for cf in cash_flows if cf > 0)
                if total_investment < 0:
                    simple_return = (total_return / abs(total_investment)) - 1
                    irr = simple_return / (analysis_period_days / 365)  # Annualize
                else:
                    irr = 0.0
                break
        
        # Convert to annual percentage with safety checks
        try:
            if abs(irr) > 5:  # If daily rate is too high, cap it
                annual_irr = min(1000, max(-100, irr * 365))  # Linear approximation
            else:
                annual_irr = (1 + irr) ** 365 - 1
            
            # Cap at reasonable bounds
            annual_irr = min(1000, max(-99, annual_irr))
            return annual_irr * 100
            
        except (OverflowError, ValueError):
            # Fallback to simple calculation
            total_investment = sum(cf for cf in cash_flows if cf < 0)
            total_return = sum(cf for cf in cash_flows if cf > 0)
            if total_investment < 0:
                return ((total_return / abs(total_investment)) - 1) * 100
            return 0.0
    
    async def _calculate_break_even_point(
        self, 
        costs: List[ExperimentCost], 
        revenue_streams: List[RevenueStream]
    ) -> datetime:
        """Calculate when the experiment breaks even"""
        
        payback_days = await self._calculate_payback_period(costs, revenue_streams)
        
        if payback_days > 0:
            return datetime.now() + timedelta(days=payback_days)
        else:
            return datetime.now() + timedelta(days=3650)  # 10 years in future
    
    async def _calculate_risk_metrics(
        self, 
        costs: List[ExperimentCost], 
        revenue_streams: List[RevenueStream]
    ) -> Dict[str, float]:
        """Calculate various risk metrics"""
        
        # Value at Risk (VaR) - simplified calculation
        total_cost = sum(cost.amount * cost.allocation_percentage for cost in costs)
        total_expected_revenue = sum(stream.uplift_revenue for stream in revenue_streams)
        
        # Calculate downside scenario (5th percentile)
        downside_revenue = sum(stream.confidence_interval[0] for stream in revenue_streams)
        downside_loss = total_cost - downside_revenue
        var_95 = max(0, downside_loss)
        
        # Calculate Sharpe ratio (risk-adjusted return)
        expected_return = (total_expected_revenue - total_cost) / total_cost if total_cost > 0 else 0
        risk_free_return = self.config["risk_free_rate"]
        
        # Estimate volatility from confidence intervals
        revenue_volatilities = []
        for stream in revenue_streams:
            ci_width = stream.confidence_interval[1] - stream.confidence_interval[0]
            volatility = ci_width / (4 * stream.uplift_revenue) if stream.uplift_revenue > 0 else 0
            revenue_volatilities.append(volatility)
        
        portfolio_volatility = np.sqrt(np.mean([v**2 for v in revenue_volatilities]))
        sharpe_ratio = (expected_return - risk_free_return) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            "value_at_risk_95": var_95,
            "sharpe_ratio": sharpe_ratio,
            "downside_scenario_roi": ((downside_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0,
            "upside_scenario_roi": ((sum(stream.confidence_interval[1] for stream in revenue_streams) - total_cost) / total_cost * 100) if total_cost > 0 else 0
        }
    
    async def _perform_sensitivity_analysis(
        self, 
        costs: List[ExperimentCost], 
        revenue_streams: List[RevenueStream]
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on key variables"""
        
        base_total_cost = sum(cost.amount * cost.allocation_percentage for cost in costs)
        base_total_revenue = sum(stream.uplift_revenue for stream in revenue_streams)
        base_roi = ((base_total_revenue - base_total_cost) / base_total_cost * 100) if base_total_cost > 0 else 0
        
        sensitivity_results = {}
        
        # Test different scenarios
        scenarios = {
            "cost_increase_10": {"cost_multiplier": 1.1, "revenue_multiplier": 1.0},
            "cost_increase_25": {"cost_multiplier": 1.25, "revenue_multiplier": 1.0},
            "revenue_decrease_10": {"cost_multiplier": 1.0, "revenue_multiplier": 0.9},
            "revenue_decrease_25": {"cost_multiplier": 1.0, "revenue_multiplier": 0.75},
            "optimistic": {"cost_multiplier": 0.9, "revenue_multiplier": 1.2},
            "pessimistic": {"cost_multiplier": 1.2, "revenue_multiplier": 0.8}
        }
        
        for scenario_name, multipliers in scenarios.items():
            adjusted_cost = base_total_cost * multipliers["cost_multiplier"]
            adjusted_revenue = base_total_revenue * multipliers["revenue_multiplier"]
            scenario_roi = ((adjusted_revenue - adjusted_cost) / adjusted_cost * 100) if adjusted_cost > 0 else 0
            
            sensitivity_results[scenario_name] = {
                "roi": scenario_roi,
                "roi_change": scenario_roi - base_roi,
                "net_benefit": adjusted_revenue - adjusted_cost
            }
        
        return sensitivity_results
    
    async def _calculate_roi_confidence_interval(
        self, 
        costs: List[ExperimentCost], 
        revenue_streams: List[RevenueStream]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for ROI using Monte Carlo simulation"""
        
        n_simulations = 1000  # Simplified for performance
        roi_simulations = []
        
        base_total_cost = sum(cost.amount * cost.allocation_percentage for cost in costs)
        
        for _ in range(n_simulations):
            # Simulate revenue with uncertainty
            simulated_revenue = 0.0
            for stream in revenue_streams:
                # Sample from normal distribution within confidence interval
                ci_lower, ci_upper = stream.confidence_interval
                std = (ci_upper - ci_lower) / 4  # Approximation
                simulated_stream_revenue = np.random.normal(stream.uplift_revenue, std)
                simulated_revenue += max(0, simulated_stream_revenue)  # No negative revenue
            
            # Calculate ROI for this simulation
            if base_total_cost > 0:
                sim_roi = ((simulated_revenue - base_total_cost) / base_total_cost) * 100
                roi_simulations.append(sim_roi)
        
        # Calculate confidence interval
        if roi_simulations:
            roi_5th = np.percentile(roi_simulations, 2.5)
            roi_95th = np.percentile(roi_simulations, 97.5)
            return (roi_5th, roi_95th)
        else:
            return (0, 0)
    
    async def generate_financial_projections(
        self,
        costs: List[ExperimentCost],
        revenue_streams: List[RevenueStream],
        projection_months: int = 12
    ) -> List[FinancialProjection]:
        """Generate month-by-month financial projections"""
        
        projections = []
        cumulative_costs = 0.0
        cumulative_revenue = 0.0
        
        for month in range(projection_months):
            month_start = datetime.now() + timedelta(days=month * 30)
            month_end = month_start + timedelta(days=30)
            
            # Calculate monthly costs
            monthly_costs = 0.0
            for cost in costs:
                if not cost.is_recurring and month == 0:
                    monthly_costs += cost.amount * cost.allocation_percentage
                elif cost.is_recurring and cost.frequency_days:
                    monthly_recurrences = 30 / cost.frequency_days
                    monthly_costs += cost.amount * cost.allocation_percentage * monthly_recurrences
            
            # Calculate monthly revenue
            monthly_revenue = 0.0
            for stream in revenue_streams:
                if month * 30 >= stream.time_to_realize_days:
                    revenue_month = (month * 30) - stream.time_to_realize_days
                    if revenue_month < stream.revenue_duration_days:
                        days_in_month = min(30, stream.revenue_duration_days - revenue_month)
                        daily_revenue = stream.uplift_revenue / stream.revenue_duration_days
                        monthly_revenue += daily_revenue * days_in_month
            
            cumulative_costs += monthly_costs
            cumulative_revenue += monthly_revenue
            
            net_cash_flow = cumulative_revenue - cumulative_costs
            roi_to_date = (net_cash_flow / cumulative_costs * 100) if cumulative_costs > 0 else 0
            
            projections.append(FinancialProjection(
                period_start=month_start,
                period_end=month_end,
                cumulative_costs=cumulative_costs,
                cumulative_revenue=cumulative_revenue,
                net_cash_flow=net_cash_flow,
                roi_to_date=roi_to_date
            ))
        
        return projections
    
    async def generate_roi_report(
        self,
        roi_analysis: ROIAnalysis,
        financial_projections: Optional[List[FinancialProjection]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive ROI report"""
        
        return {
            "executive_summary": {
                "experiment_id": roi_analysis.experiment_id,
                "roi_percentage": roi_analysis.roi_percentage,
                "net_benefit": roi_analysis.net_benefit,
                "payback_period_days": roi_analysis.payback_period_days,
                "break_even_date": roi_analysis.break_even_point.strftime("%Y-%m-%d"),
                "recommendation": self._generate_investment_recommendation(roi_analysis),
                "confidence_interval": roi_analysis.confidence_interval
            },
            "financial_metrics": {
                "total_investment": roi_analysis.total_costs,
                "total_expected_return": roi_analysis.total_revenue,
                "net_present_value": roi_analysis.npv,
                "internal_rate_of_return": roi_analysis.irr,
                "payback_period_months": roi_analysis.payback_period_days / 30.44
            },
            "risk_analysis": roi_analysis.risk_metrics,
            "sensitivity_analysis": roi_analysis.sensitivity_analysis,
            "monthly_projections": [
                {
                    "month": i + 1,
                    "cumulative_costs": proj.cumulative_costs,
                    "cumulative_revenue": proj.cumulative_revenue,
                    "net_cash_flow": proj.net_cash_flow,
                    "roi_to_date": proj.roi_to_date
                }
                for i, proj in enumerate(financial_projections or [])
            ] if financial_projections else [],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_investment_recommendation(self, roi_analysis: ROIAnalysis) -> str:
        """Generate investment recommendation based on analysis"""
        
        if roi_analysis.roi_percentage > 100 and roi_analysis.payback_period_days < 365:
            return "STRONG BUY - Excellent ROI with fast payback"
        elif roi_analysis.roi_percentage > 50 and roi_analysis.payback_period_days < 730:
            return "BUY - Good ROI with reasonable payback period"
        elif roi_analysis.roi_percentage > 20 and roi_analysis.payback_period_days < 1095:
            return "CONSIDER - Positive ROI but longer payback"
        elif roi_analysis.roi_percentage > 0:
            return "MARGINAL - Positive ROI but low returns"
        else:
            return "DO NOT INVEST - Negative expected returns"

# Demo functions
async def create_demo_costs() -> List[ExperimentCost]:
    """Create demo experiment costs"""
    return [
        ExperimentCost(
            category=CostCategory.DEVELOPMENT,
            description="Frontend development",
            amount=25000,
            currency="USD",
            is_recurring=False,
            allocation_percentage=1.0
        ),
        ExperimentCost(
            category=CostCategory.DEVELOPMENT,
            description="Backend development",
            amount=15000,
            currency="USD", 
            is_recurring=False,
            allocation_percentage=1.0
        ),
        ExperimentCost(
            category=CostCategory.INFRASTRUCTURE,
            description="Additional server capacity",
            amount=2000,
            currency="USD",
            is_recurring=True,
            frequency_days=30,
            allocation_percentage=0.5
        ),
        ExperimentCost(
            category=CostCategory.PERSONNEL,
            description="Data analyst time",
            amount=8000,
            currency="USD",
            is_recurring=False,
            allocation_percentage=0.3
        ),
        ExperimentCost(
            category=CostCategory.TOOLING,
            description="A/B testing platform",
            amount=500,
            currency="USD",
            is_recurring=True,
            frequency_days=30,
            allocation_percentage=1.0
        )
    ]

async def create_demo_revenue_streams() -> List[RevenueStream]:
    """Create demo revenue streams"""
    return [
        RevenueStream(
            stream_name="Increased conversion revenue",
            baseline_revenue=100000,
            uplift_revenue=18000,  # 18% increase
            confidence_interval=(12000, 24000),
            attribution_model=RevenueModel.LINEAR,
            time_to_realize_days=7,
            revenue_duration_days=365,
            currency="USD"
        ),
        RevenueStream(
            stream_name="Higher customer lifetime value",
            baseline_revenue=250000,
            uplift_revenue=25000,  # 10% increase
            confidence_interval=(15000, 35000),
            attribution_model=RevenueModel.TIME_DECAY,
            time_to_realize_days=30,
            revenue_duration_days=365,
            currency="USD"
        ),
        RevenueStream(
            stream_name="Reduced churn impact",
            baseline_revenue=50000,
            uplift_revenue=8000,  # 16% reduction in churn
            confidence_interval=(4000, 12000),
            attribution_model=RevenueModel.LINEAR,
            time_to_realize_days=60,
            revenue_duration_days=365,
            currency="USD"
        )
    ]

async def run_roi_calculator_demo():
    """Run comprehensive ROI calculator demo"""
    
    print("ROI Calculator Demo")
    print("=" * 50)
    
    # Create calculator
    calculator = ROICalculator()
    
    # Create demo data
    print("Demo Experiment: E-commerce Checkout Optimization")
    print("Calculating comprehensive ROI analysis...")
    
    costs = await create_demo_costs()
    revenue_streams = await create_demo_revenue_streams()
    
    # Display investment summary
    total_investment = sum(cost.amount * cost.allocation_percentage for cost in costs)
    total_expected_return = sum(stream.uplift_revenue for stream in revenue_streams)
    
    print(f"\nInvestment Summary:")
    print(f"  Total Investment: ${total_investment:,.2f}")
    print(f"  Expected Annual Return: ${total_expected_return:,.2f}")
    print(f"  Expected ROI: {((total_expected_return - total_investment) / total_investment * 100):.1f}%")
    
    print(f"\nCost Breakdown:")
    for cost in costs:
        allocated_amount = cost.amount * cost.allocation_percentage
        recurring_text = f" (${cost.amount}/month)" if cost.is_recurring else ""
        print(f"  {cost.description}: ${allocated_amount:,.2f}{recurring_text}")
    
    print(f"\nRevenue Streams:")
    for stream in revenue_streams:
        ci_lower, ci_upper = stream.confidence_interval
        print(f"  {stream.stream_name}: ${stream.uplift_revenue:,.2f}")
        print(f"    Confidence Interval: ${ci_lower:,.2f} - ${ci_upper:,.2f}")
        print(f"    Time to Realize: {stream.time_to_realize_days} days")
    
    # Calculate ROI
    print(f"\nRunning ROI Analysis...")
    roi_analysis = await calculator.calculate_roi("checkout_optimization_001", costs, revenue_streams)
    
    # Generate projections
    projections = await calculator.generate_financial_projections(costs, revenue_streams, 12)
    
    # Generate comprehensive report
    report = await calculator.generate_roi_report(roi_analysis, projections)
    
    # Display results
    print(f"\nROI Analysis Results")
    print("=" * 50)
    
    summary = report["executive_summary"]
    print(f"Executive Summary:")
    print(f"  ROI: {summary['roi_percentage']:.1f}%")
    print(f"  Net Benefit: ${summary['net_benefit']:,.2f}")
    print(f"  Payback Period: {summary['payback_period_days']} days ({summary['payback_period_days']/30.44:.1f} months)")
    print(f"  Break-even Date: {summary['break_even_date']}")
    print(f"  Investment Recommendation: {summary['recommendation']}")
    
    roi_ci = summary['confidence_interval']
    print(f"  ROI Confidence Interval: {roi_ci[0]:.1f}% to {roi_ci[1]:.1f}%")
    
    financial = report["financial_metrics"]
    print(f"\nFinancial Metrics:")
    print(f"  Net Present Value: ${financial['net_present_value']:,.2f}")
    print(f"  Internal Rate of Return: {financial['internal_rate_of_return']:.1f}%")
    print(f"  Total Investment: ${financial['total_investment']:,.2f}")
    print(f"  Expected Return: ${financial['total_expected_return']:,.2f}")
    
    risk = report["risk_analysis"]
    print(f"\nRisk Analysis:")
    print(f"  Value at Risk (95%): ${risk['value_at_risk_95']:,.2f}")
    print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
    print(f"  Downside Scenario ROI: {risk['downside_scenario_roi']:.1f}%")
    print(f"  Upside Scenario ROI: {risk['upside_scenario_roi']:.1f}%")
    
    print(f"\nSensitivity Analysis:")
    sensitivity = report["sensitivity_analysis"]
    for scenario, results in sensitivity.items():
        scenario_name = scenario.replace("_", " ").title()
        print(f"  {scenario_name}: {results['roi']:.1f}% ROI ({results['roi_change']:+.1f}% change)")
    
    # Show monthly projections for first 6 months
    if report["monthly_projections"]:
        print(f"\nMonthly Financial Projections (First 6 months):")
        print(f"  Month | Cumulative Costs | Cumulative Revenue | Net Cash Flow | ROI")
        print(f"  ------|------------------|--------------------|--------------|---------")
        for proj in report["monthly_projections"][:6]:
            print(f"    {proj['month']:2d}  |    ${proj['cumulative_costs']:8,.0f}    |     ${proj['cumulative_revenue']:8,.0f}     |  ${proj['net_cash_flow']:8,.0f}  | {proj['roi_to_date']:6.1f}%")

async def main():
    """Main function"""
    
    print("Phase 3C: ROI Calculator")
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        await run_roi_calculator_demo()
        
        print("\n" + "=" * 60)
        print("ROI ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Successfully Calculated:")
        print("  - Comprehensive Cost Analysis")
        print("  - Multi-Stream Revenue Attribution")
        print("  - Net Present Value (NPV)")
        print("  - Internal Rate of Return (IRR)")
        print("  - Payback Period Analysis")
        print("  - Risk Metrics & Value at Risk")
        print("  - Sensitivity Analysis")
        print("  - Monthly Financial Projections")
        print("  - Investment Recommendations")
        print()
        print("Next Steps:")
        print("1. Business Impact Analyzer - COMPLETED")
        print("2. ROI Calculator - COMPLETED")
        print("3. Segment Analyzer - NEXT")
        print("4. Temporal Pattern Detector - FINAL")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())