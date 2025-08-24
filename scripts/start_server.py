#!/usr/bin/env python3
"""
Server start with detailed logging
"""
import sys
from pathlib import Path
import os

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set environment
os.environ["PYTHONPATH"] = str(src_path)

if __name__ == "__main__":
    try:
        print("🚀 Starting A/B Testing API Server")
        print("=" * 40)
        
        print("Importing modules...")
        from ab_testing.ab_testing_api import app
        print(" API imported successfully")
        
        print("Starting uvicorn server...")
        print("Server will be available at: http://localhost:8000")
        print("API docs will be at: http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        print("=" * 40)
        
        import uvicorn
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            reload=False  # Disable reload for stability
        )
        
    except KeyboardInterrupt:
        print("\n Server stopped by user")
    except Exception as e:
        print(f" Server start failed: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to continue...")
