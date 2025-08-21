# Ka Data Preprocessing with DVC Integration
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import json
import os
from pathlib import Path

def load_params():
    """Load parameters from model_params.yaml"""
    try:
        with open('model_params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        return params
    except FileNotFoundError:
        print("⚠️ model_params.yaml not found, using defaults")
        return {
            'evaluate': {
                'test_size': 0.2,
                'random_state': 42,
                'stratify': True
            }
        }

def preprocess_ka_data():
    """Enhanced Ka data preprocessing with DVC tracking"""
    print('⚙️ Ka Data Preprocessing with DVC Integration')
    print('=' * 50)
    
    # Load parameters
    params = load_params()
    eval_params = params.get('evaluate', {})
    
    # Load raw data
    try:
        print('📥 Loading raw Ka dataset...')
        df = pd.read_csv('data/raw/ka_lending_club_dataset.csv')
        print(f'📊 Raw data shape: {df.shape}')
    except FileNotFoundError:
        print('❌ Raw data not found!')
        print('🔧 Run data generation first: python scripts/ka_scripts/generate_training_data.py')
        return False
    
    # Data quality checks
    print('\n🔍 Data Quality Assessment:')
    print(f'   Total samples: {len(df):,}')
    print(f'   Missing values: {df.isnull().sum().sum()}')
    print(f'   Duplicate rows: {df.duplicated().sum()}')
    
    # Target distribution
    target_dist = df['loan_status'].value_counts()
    default_rate = (df['loan_status'] == 'Charged Off').mean()
    print(f'   Default rate: {default_rate:.1%}')
    print(f'   Target distribution: {dict(target_dist)}')
    
    # Feature engineering
    print('\n🔧 Feature Engineering:')
    
    # Select features
    numerical_features = [
        'loan_amnt', 'int_rate', 'annual_inc', 'dti', 
        'fico_range_low', 'fico_range_high', 'installment',
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 
        'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'mort_acc', 'pub_rec_bankruptcies'
    ]
    
    categorical_features = [
        'grade', 'term', 'home_ownership', 
        'verification_status', 'purpose'
    ]
    
    # Prepare feature matrix
    available_numerical = [col for col in numerical_features if col in df.columns]
    available_categorical = [col for col in categorical_features if col in df.columns]
    
    print(f'   Numerical features: {len(available_numerical)}')
    print(f'   Categorical features: {len(available_categorical)}')
    
    # Create feature matrix
    X = df[available_numerical + available_categorical].copy()
    
    # Handle missing values
    print('\n🔧 Handling Missing Values:')
    for col in available_numerical:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f'   {col}: filled {X[col].isnull().sum()} missing values with median {median_val:.2f}')
    
    # Encode categorical features
    print('\n🔧 Encoding Categorical Features:')
    label_encoders = {}
    for col in available_categorical:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna('Unknown')
        
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f'   {col}: {len(le.classes_)} unique values encoded')
    
    # Create target variable
    y = (df['loan_status'] == 'Charged Off').astype(int)
    
    # Train-test split
    print('\n📊 Creating Train-Test Split:')
    test_size = eval_params.get('test_size', 0.2)
    random_state = eval_params.get('random_state', 42)
    stratify = y if eval_params.get('stratify', True) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify
    )
    
    print(f'   Training set: {X_train.shape[0]:,} samples')
    print(f'   Test set: {X_test.shape[0]:,} samples')
    print(f'   Train default rate: {y_train.mean():.1%}')
    print(f'   Test default rate: {y_test.mean():.1%}')
    
    # Save processed data
    print('\n💾 Saving Processed Data:')
    output_dir = Path('data/processed/ka_files')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(output_dir / 'ka_X_train.csv', index=False)
    X_test.to_csv(output_dir / 'ka_X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['loan_status']).to_csv(output_dir / 'ka_y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['loan_status']).to_csv(output_dir / 'ka_y_test.csv', index=False)
    
    # Save preprocessing metadata
    preprocessing_info = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'raw_data_shape': df.shape,
        'processed_features': list(X.columns),
        'numerical_features': available_numerical,
        'categorical_features': available_categorical,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'default_rate_train': float(y_train.mean()),
        'default_rate_test': float(y_test.mean()),
        'test_size': test_size,
        'random_state': random_state,
        'label_encoders': {col: list(le.classes_) for col, le in label_encoders.items()}
    }
    
    with open(output_dir / 'preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    
    print(f'   ✅ X_train.csv: {X_train.shape}')
    print(f'   ✅ X_test.csv: {X_test.shape}')
    print(f'   ✅ y_train.csv: {y_train.shape}')
    print(f'   ✅ y_test.csv: {y_test.shape}')
    print(f'   ✅ preprocessing_info.json')
    
    print('\n🎉 Ka Data Preprocessing Complete!')
    return True

def main():
    """Main preprocessing function"""
    try:
        success = preprocess_ka_data()
        if success:
            print('✅ Preprocessing completed successfully!')
            return 0
        else:
            print('❌ Preprocessing failed!')
            return 1
    except Exception as e:
        print(f'❌ Error during preprocessing: {e}')
        return 1

if __name__ == "__main__":
    exit(main())