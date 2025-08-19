import mlflow
import mlflow.sklearn

def simple_deployment_test():
    print("🚀 Starting simple deployment test...")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://ab124afa4840a4f8298398f9c7fd7c7e-306571921.ap-south-1.elb.amazonaws.com")
    print("✅ MLflow URI set")
    
    try:
        # Test connection
        client = mlflow.tracking.MlflowClient()
        print("✅ MLflow client created")
        
        # Get model
        model_name = "loan-default-model"
        print(f"🔍 Looking for model: {model_name}")
        
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        
        if latest_versions:
            version = latest_versions[0].version
            print(f"✅ Found model version: {version}")
            
            # Get model details
            model_version = client.get_model_version(model_name, version)
            print(f"✅ Model version details retrieved")
            
            # Get metrics
            run_id = model_version.run_id
            run = client.get_run(run_id)
            metrics = run.data.metrics
            
            accuracy = metrics.get('accuracy', 0)
            auc_score = metrics.get('auc_score', 0)
            
            print(f"📊 Model Performance:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   AUC Score: {auc_score:.4f}")
            
            # Check if it meets criteria
            if accuracy >= 0.70 and auc_score >= 0.60:
                print("✅ Model meets deployment criteria!")
                return True
            else:
                print("❌ Model does not meet deployment criteria")
                return False
            
        else:
            print(f"❌ No versions found for model {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 50)
    success = simple_deployment_test()
    print("=" * 50)
    if success:
        print("🎉 Simple test passed! Model is ready for deployment.")
    else:
        print("❌ Simple test failed!")