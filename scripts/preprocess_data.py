import pandas as pd
import numpy as np
import json
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from datetime import datetime

def load_params():
    """Load parameters from params.yaml"""
    if os.path.exists('params.yaml'):
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        return params
    else:
        # Default parameters if params.yaml doesn't exist
        return {
            'data_preprocessing': {
                'test_size': 0.2,
                'random_state': 42,
                'scale_features': True
            }
        }

def load_raw_data():
    """Load raw data from DVC tracked files"""
    print("📊 Loading raw data...")
    
    try:
        X = pd.read_csv('data/raw/X.csv')
        y = pd.read_csv('data/raw/y.csv')
        
        print(f"   ✅ Features loaded: {X.shape}")
        print(f"   ✅ Target loaded: {y.shape}")
        
        # Handle target - take first column if multiple columns
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            print(f"   ⚠️ Multiple target columns found, using first: {y.columns[0]}")
            y = y.iloc[:, 0]
        
        return X, y
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def preprocess_data(X, y, params):
    """Preprocess the data according to parameters"""
    print("🔧 Preprocessing data...")
    
    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"   📊 Categorical columns: {len(categorical_columns)}")
    print(f"   📊 Numerical columns: {len(numerical_columns)}")
    
    # Create label encoders for categorical variables
    label_encoders = {}
    X_processed = X.copy()
    
    for col in categorical_columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le
        print(f"   🔄 Encoded {col}: {len(le.classes_)} unique values")
    
    # Encode target if categorical
    target_encoder = None
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y_processed = target_encoder.fit_transform(y)
        print(f"   🎯 Target encoded: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
    else:
        y_processed = y.copy()
    
    # Split data
    test_size = params['data_preprocessing']['test_size']
    random_state = params['data_preprocessing']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_processed
    )
    
    print(f"   📚 Training set: {X_train.shape[0]} samples")
    print(f"   🧪 Test set: {X_test.shape[0]} samples")
    
    # Scale features if specified
    scaler = None
    if params['data_preprocessing']['scale_features'] and numerical_columns:
        scaler = StandardScaler()
        
        # Get numerical column indices
        numerical_indices = [X_processed.columns.get_loc(col) for col in numerical_columns if col in X_processed.columns]
        
        if numerical_indices:
            # Create copies for scaling
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            # Scale only numerical columns
            X_train_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X_train.iloc[:, numerical_indices])
            X_test_scaled.iloc[:, numerical_indices] = scaler.transform(X_test.iloc[:, numerical_indices])
            
            print(f"   📏 Scaled {len(numerical_indices)} numerical features")
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Create preprocessing metadata
    preprocessing_metadata = {
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'feature_columns': list(X_processed.columns),
        'label_encoders_info': {col: list(le.classes_) for col, le in label_encoders.items()},
        'target_encoder_info': list(target_encoder.classes_) if target_encoder else None,
        'scaling_applied': params['data_preprocessing']['scale_features'],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'preprocessing_timestamp': datetime.now().isoformat(),
        'target_distribution': dict(zip(*np.unique(y_processed, return_counts=True)))
    }
    
    # Save preprocessing objects
    preprocessing_objects = {
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'scaler': scaler,
        'metadata': preprocessing_metadata
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessing_objects

def save_processed_data(X_train, X_test, y_train, y_test, preprocessing_objects):
    """Save processed data and metadata"""
    print("💾 Saving processed data...")
    
    # Ensure output directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Combine train and test data for saving
    X_processed = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_processed = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0, ignore_index=True)
    
    # Add split indicator
    split_indicator = ['train'] * len(X_train) + ['test'] * len(X_test)
    X_processed['split'] = split_indicator
    
    # Save processed features
    X_processed.to_csv('data/processed/X_processed.csv', index=False)
    print(f"   ✅ Saved X_processed.csv: {X_processed.shape}")
    
    # Save processed target
    y_df = pd.DataFrame({
        'target': y_processed,
        'split': split_indicator
    })
    y_df.to_csv('data/processed/y_processed.csv', index=False)
    print(f"   ✅ Saved y_processed.csv: {y_df.shape}")
    
    # Save preprocessing metadata
    with open('data/processed/preprocessing_metadata.json', 'w') as f:
        # Convert non-serializable objects to serializable format
        metadata_to_save = preprocessing_objects['metadata'].copy()
        
        # Convert numpy types to native Python types for JSON serialization
        if 'target_distribution' in metadata_to_save:
            metadata_to_save['target_distribution'] = {
                str(k): int(v) for k, v in metadata_to_save['target_distribution'].items()
            }
        
        # Convert other numpy types
        for key, value in metadata_to_save.items():
            if isinstance(value, (np.integer, np.int64)):
                metadata_to_save[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                metadata_to_save[key] = float(value)
        
        json.dump(metadata_to_save, f, indent=2)
    print("   ✅ Saved preprocessing_metadata.json")
    
    # Save preprocessing objects (joblib for sklearn objects)
    preprocessing_pipeline = {
        'label_encoders': preprocessing_objects['label_encoders'],
        'target_encoder': preprocessing_objects['target_encoder'],
        'scaler': preprocessing_objects['scaler'],
        'metadata': preprocessing_objects['metadata']
    }
    
    joblib.dump(preprocessing_pipeline, 'data/processed/preprocessing_pipeline.joblib')
    print("   ✅ Saved preprocessing_pipeline.joblib")

def main():
    """Main preprocessing function"""
    print("🚀 Starting DVC Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Load parameters
    params = load_params()
    print(f"📋 Loaded parameters: {params['data_preprocessing']}")
    
    # Load raw data
    X, y = load_raw_data()
    if X is None or y is None:
        print("❌ Failed to load raw data")
        return False
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessing_objects = preprocess_data(X, y, params)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, preprocessing_objects)
    
    print("=" * 60)
    print("✅ Data preprocessing completed successfully!")
    print(f"📊 Processed {len(X_train) + len(X_test)} samples")
    print(f"📊 Features: {len(X_train.columns)} columns")
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)