# Ka-MLOps Data Preprocessing Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class KaLendingClubPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Define feature columns based on our Ka dataset
        self.numerical_features = [
            'loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low', 
            'fico_range_high', 'installment', 'delinq_2yrs', 'inq_last_6mths', 
            'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 
            'mort_acc', 'pub_rec_bankruptcies'
        ]
        
        self.categorical_features = [
            'grade', 'term', 'emp_length', 'home_ownership', 
            'verification_status', 'purpose', 'addr_state'
        ]
        
    def load_and_clean_data(self, file_path):
        '''Load and clean Ka Lending Club dataset'''
        print(f'🔄 Loading Ka dataset from: {file_path}')
        df = pd.read_csv(file_path)
        
        print(f' Loaded dataset shape: {df.shape}')
        
        # Create binary target variable
        df['ka_target'] = (df['loan_status'] == 'Charged Off').astype(int)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        print(f'✅ Ka dataset processed: {len(df):,} samples')
        return df
    
    def _handle_missing_values(self, df):
        '''Handle missing values in Ka dataset'''
        print(' Handling missing values...')
        
        # Fill numerical missing values with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Fill categorical missing values with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def _feature_engineering(self, df):
        '''Create new features for Ka model'''
        print(' Engineering new features...')
        
        # FICO average
        df['ka_fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        
        # Income to loan ratio
        df['ka_income_loan_ratio'] = df['annual_inc'] / df['loan_amnt']
        
        # Debt burden (installment as % of monthly income)
        monthly_income = df['annual_inc'] / 12
        df['ka_debt_burden'] = df['installment'] / monthly_income
        
        # Credit utilization risk score
        df['ka_credit_risk'] = df['revol_util'] / 100 * df['revol_bal'] / 10000
        
        # Add engineered features to numerical list
        self.numerical_features.extend([
            'ka_fico_avg', 'ka_income_loan_ratio', 'ka_debt_burden', 'ka_credit_risk'
        ])
        
        return df
    
    def prepare_features(self, df, is_training=True):
        '''Prepare features for Ka model training/prediction'''
        print(' Preparing features for modeling...')
        
        # Select features that exist in the dataset
        available_numerical = [col for col in self.numerical_features if col in df.columns]
        available_categorical = [col for col in self.categorical_features if col in df.columns]
        
        # Create feature matrix
        X_numerical = df[available_numerical].copy()
        X_categorical = df[available_categorical].copy()
        
        # Encode categorical variables
        for col in available_categorical:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                X_categorical[col] = self.label_encoders[col].fit_transform(X_categorical[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    known_categories = self.label_encoders[col].classes_
                    X_categorical[col] = X_categorical[col].astype(str).apply(
                        lambda x: x if x in known_categories else known_categories[0]
                    )
                    X_categorical[col] = self.label_encoders[col].transform(X_categorical[col])
        
        # Combine numerical and categorical features
        X = pd.concat([X_numerical, X_categorical], axis=1)
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Return as DataFrame
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def save_preprocessor(self, path):
        '''Save Ka preprocessor objects'''
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }, path)
        print(f' Saved Ka preprocessor to: {path}')
    
    def load_preprocessor(self, path):
        '''Load Ka preprocessor objects'''
        objects = joblib.load(path)
        self.scaler = objects['scaler']
        self.label_encoders = objects['label_encoders']
        self.numerical_features = objects['numerical_features']
        self.categorical_features = objects['categorical_features']
        print(f' Loaded Ka preprocessor from: {path}')

def main():
    '''Main Ka preprocessing pipeline'''
    print(' Ka-MLOps Data Preprocessing Pipeline')
    print('=' * 50)
    
    # Initialize Ka preprocessor
    ka_preprocessor = KaLendingClubPreprocessor()
    
    # Load and process Ka data
    ka_df = ka_preprocessor.load_and_clean_data('data/raw/ka_lending_club_dataset.csv')
    
    # Prepare features and target
    X = ka_preprocessor.prepare_features(ka_df, is_training=True)
    y = ka_df['ka_target']
    
    print(f' Feature matrix shape: {X.shape}')
    print(f' Target distribution: {y.value_counts().to_dict()}')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create directories
    Path('data/processed/ka_files').mkdir(parents=True, exist_ok=True)
    Path('models/ka_models').mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    X_train.to_csv('data/processed/ka_files/ka_X_train.csv', index=False)
    X_test.to_csv('data/processed/ka_files/ka_X_test.csv', index=False)
    y_train.to_csv('data/processed/ka_files/ka_y_train.csv', index=False)
    y_test.to_csv('data/processed/ka_files/ka_y_test.csv', index=False)
    
    # Save preprocessor
    ka_preprocessor.save_preprocessor('models/ka_models/ka_preprocessor.pkl')
    
    # Show results
    train_samples = len(X_train)
    test_samples = len(X_test)
    feature_count = X.shape[1]
    default_rate = y.mean()
    
    print(' Ka preprocessing completed!')
    print(f' Training samples: {train_samples:,}')
    print(f' Test samples: {test_samples:,}')
    print(f' Total features: {feature_count}')
    print(f' Default rate: {default_rate:.1%}')
    print(' Files saved to: data/processed/ka_files/')

if __name__ == '__main__':
    main()
