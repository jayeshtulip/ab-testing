import time
from fastapi import FastAPI
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="Loan API")
requests = Counter("api_requests_total", "Total requests", ["endpoint"])

@app.get("/")
def root():
    requests.labels(endpoint="root").inc()
    return {"message": "Loan Default API", "status": "running", "version": "k8s-1.0"}

@app.get("/health")  
def health():
    requests.labels(endpoint="health").inc()
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
