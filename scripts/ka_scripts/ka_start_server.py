# Ka-MLOps Server Startup Script
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Start the Ka server
if __name__ == "__main__":
    import uvicorn
    from src.ka_api.ka_main import app
    
    print("🚀 Starting Ka-MLOps API Server...")
    print(" Server will be available at: http://localhost:8000")
    print(" API docs at: http://localhost:8000/ka-docs")
    print("  Health check at: http://localhost:8000/ka-health")
    print(" Predictions at: http://localhost:8000/ka-predict")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
        log_level="info"
    )
