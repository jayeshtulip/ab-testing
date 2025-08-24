"""
Fixed data generator for A/B testing
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
import os

class SyntheticDataGenerator:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        # Set the global numpy seed
        np.random.seed(random_seed)
    
    def generate_baseline_data(self, n_samples: int = 1000, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate baseline synthetic data"""
        
        # Always reset the seed at the beginning of data generation
        np.random.seed(self.random_seed)
        
        # Create realistic loan default features
        loan_amount = np.random.lognormal(10, 1, n_samples)
        income = np.random.lognormal(11, 0.8, n_samples)
        credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
        debt_to_income = np.random.beta(2, 5, n_samples).clip(0, 1)
        employment_years = np.random.exponential(5, n_samples).clip(0, 40)
        
        X = np.column_stack([loan_amount, income, credit_score, debt_to_income, employment_years])
        
        # Generate target based on features
        risk_score = (
            (X[:, 0] / X[:, 1]) * 0.3 +  # loan-to-income ratio
            ((850 - X[:, 2]) / 100) * 0.4 +  # credit score (inverted)
            (X[:, 3]) * 0.2 +  # debt-to-income ratio
            np.random.random(n_samples) * 0.1  # random noise
        )
        
        # Convert to binary classification (15% default rate)
        threshold = np.percentile(risk_score, 85)
        y = (risk_score > threshold).astype(int)
        
        feature_names = ['loan_amount', 'income', 'credit_score', 'debt_to_income', 'employment_years']
        
        return X, y, feature_names

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    X, y, features = generator.generate_baseline_data(1000, 5)
    print(f"Generated {len(X)} samples with {len(features)} features")
    print(f"Target distribution: 0={sum(y==0)}, 1={sum(y==1)}")
