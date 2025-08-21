from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Ka-MLOps API", version="1.0.0")

@app.get("/")
def root():
    return {"message": "Ka-MLOps API Working!", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
