import pandas as pd
import numpy as np
import os

def analyze_data():
    """Analyze the German Credit data to understand its structure"""
    
    x_path = "data/raw/X.csv"
    y_path = "data/raw/y.csv"
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"❌ Data files not found:")
        print(f"   Looking for: {x_path}")
        print(f"   Looking for: {y_path}")
        print("Please make sure the data files exist in the correct location.")
        return False
    
    print("🔍 Analyzing German Credit Data...")
    print("="*50)
    
    # Load the data
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    
    print(f"📊 Features (X) Shape: {X.shape}")
    print(f"📊 Target (y) Shape: {y.shape}")
    
    # Combine for analysis
    df = X.copy()
    df['target'] = y.iloc[:, 0] if y.shape[1] == 1 else y
    
    print(f"📊 Combined Dataset Shape: {df.shape}")
    print(f"📊 Feature Columns: {list(X.columns)}")
    print(f"📊 Target Column: {list(y.columns)}")
    print()
    
    print("📈 Feature Data Types:")
    print(X.dtypes.value_counts())
    print()
    
    print("📊 Sample Feature Data (first 5 rows):")
    print(X.head())
    print()
    
    print("📊 Target Data:")
    print(f"Target shape: {y.shape}")
    print(f"Target unique values: {y.iloc[:, 0].unique() if y.shape[1] == 1 else 'Multiple columns'}")
    print(f"Target distribution: {y.iloc[:, 0].value_counts().to_dict() if y.shape[1] == 1 else 'Multiple columns'}")
    print()
    
    print("🔍 Missing Values in Features:")
    missing_values_x = X.isnull().sum()
    if missing_values_x.sum() > 0:
        print(missing_values_x[missing_values_x > 0])
    else:
        print("✅ No missing values found in features!")
    
    print("🔍 Missing Values in Target:")
    missing_values_y = y.isnull().sum()
    if missing_values_y.sum() > 0:
        print(missing_values_y[missing_values_y > 0])
    else:
        print("✅ No missing values found in target!")
    print()
    
    # Analyze categorical columns in features
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"📊 Categorical Columns ({len(categorical_columns)}):")
    for col in categorical_columns:
        unique_values = X[col].nunique()
        print(f"   {col}: {unique_values} unique values")
        if unique_values <= 10:  # Show values if not too many
            print(f"      Values: {list(X[col].unique())}")
    print()
    
    print(f"📊 Numerical Columns ({len(numerical_columns)}):")
    for col in numerical_columns:
        print(f"   {col}: {X[col].dtype}")
        print(f"      Range: {X[col].min()} to {X[col].max()}")
        print(f"      Mean: {X[col].mean():.2f}")
    print()
    
    # Analyze target column
    print("🎯 Target Analysis:")
    target_col = y.columns[0] if y.shape[1] == 1 else 'multiple_targets'
    if y.shape[1] == 1:
        target_values = y.iloc[:, 0]
        print(f"   Target column: {target_col}")
        print(f"   Data type: {target_values.dtype}")
        print(f"   Unique values: {target_values.unique()}")
        print(f"   Value counts: {target_values.value_counts().to_dict()}")
    else:
        print(f"   Multiple target columns: {list(y.columns)}")
    
    print()
    print("✅ Data analysis complete!")
    print("\n💡 Data Structure Summary:")
    print(f"   Features: {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"   Target: {y.shape[1]} column(s), {y.shape[0]} rows")
    print(f"   Categorical features: {len(categorical_columns)}")
    print(f"   Numerical features: {len(numerical_columns)}")
    print("\n💡 Next Steps:")
    print("1. Run the fixed training script to handle categorical variables")
    print("2. The script will automatically encode categorical variables")
    print("3. After successful training, run the deployment script")
    
    return True

if __name__ == '__main__':
    analyze_data()