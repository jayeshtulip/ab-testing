import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from datetime import datetime

def load_and_preprocess_data():
    """Load and preprocess the German Credit data"""
    print("📊 Loading and preprocessing training data...")
    
    # Load the data
    x_path = "data/raw/X.csv"
    y_path = "data/raw/y.csv"
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"❌ Data files not found:")
        print(f"   Looking for: {x_path}")
        print(f"   Looking for: {y_path}")
        return None, None, None, None, None
    
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    
    print(f"   📈 Features shape: {X.shape}")
    print(f"   📈 Target shape: {y.shape}")
    
    # Handle target - take first column if multiple columns
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    else:
        print(f"   ⚠️ Multiple target columns found, using first column: {y.columns[0]}")
        y = y.iloc[:, 0]
    
    # Display basic info about the data
    print(f"   📊 Feature columns: {list(X.columns)}")
    print(f"   📊 Data types: {X.dtypes.value_counts().to_dict()}")
    
    # Identify categorical columns
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
        print(f"   🔄 Encoded column '{col}': {len(le.classes_)} unique values")
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        label_encoders['target'] = target_encoder
        print(f"   🎯 Target encoded: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
    
    print(f"   📈 Final preprocessed shape: {X_processed.shape}")
    print(f"   📊 Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X_processed, y, label_encoders, categorical_columns, numerical_columns

def train_model_with_mlflow():
    """Train model with MLflow tracking and proper data preprocessing"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"loan_default_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print("🚀 Starting MLflow tracked training...")
        print(f"📊 MLflow Run ID: {run.info.run_id}")
        
        # Load and preprocess data
        X, y, label_encoders, categorical_columns, numerical_columns = load_and_preprocess_data()
        
        if X is None:
            return False
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   📚 Training set: {X_train.shape[0]} samples")
        print(f"   🧪 Test set: {X_test.shape[0]} samples")
        
        # Log data info
        mlflow.log_param("total_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("categorical_columns", len(categorical_columns))
        mlflow.log_param("numerical_columns", len(numerical_columns))
        
        # Scale numerical features (optional, but often helpful)
        scaler = StandardScaler()
        if numerical_columns:
            numerical_indices = [X.columns.get_loc(col) for col in numerical_columns if col in X.columns]
            if numerical_indices:
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                
                X_train_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X_train.iloc[:, numerical_indices])
                X_test_scaled.iloc[:, numerical_indices] = scaler.transform(X_test.iloc[:, numerical_indices])
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            scaler = None
        
        # Train Random Forest model
        print("🤖 Training Random Forest model...")
        
        # Model parameters
        model_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
        model = RandomForestClassifier(**model_params)
        
        # Log model parameters
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        try:
            # Fit the model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            y_test_prob = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Calculate AUC (handle both binary and multiclass)
            try:
                if len(np.unique(y)) == 2:
                    auc_score = roc_auc_score(y_test, y_test_prob[:, 1])
                else:
                    auc_score = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
            except:
                auc_score = 0.0
                print("⚠️ Could not calculate AUC score")
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("accuracy", test_accuracy)
            mlflow.log_metric("auc_score", auc_score)
            
            print(f"✅ Training completed!")
            print(f"   📈 Training Accuracy: {train_accuracy:.4f}")
            print(f"   📈 Test Accuracy: {test_accuracy:.4f}")
            print(f"   📈 AUC Score: {auc_score:.4f}")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("📊 Top 10 Feature Importances:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            # Save preprocessing objects
            preprocessing_artifacts = {
                'label_encoders': label_encoders,
                'scaler': scaler,
                'feature_columns': list(X.columns),
                'categorical_columns': categorical_columns,
                'numerical_columns': numerical_columns
            }
            
            # Save preprocessing pipeline
            preprocessing_path = "artifacts/preprocessing_pipeline.joblib"
            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(preprocessing_artifacts, preprocessing_path)
            
            # Log preprocessing artifacts
            mlflow.log_artifact(preprocessing_path)
            
            # Register the model
            model_name = "loan-default-model"
            
            # Log and register model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name,
                input_example=X_train_scaled.head(1),
                signature=mlflow.models.signature.infer_signature(X_train_scaled, y_train_pred)
            )
            
            print(f"✅ Model registered as '{model_name}'")
            print(f"📊 MLflow Run: {run.info.run_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            mlflow.log_param("error", str(e))
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main training function"""
    try:
        success = train_model_with_mlflow()
        if success:
            print("🎉 Training pipeline completed successfully!")
            return True
        else:
            print("❌ Training pipeline failed!")
            return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)