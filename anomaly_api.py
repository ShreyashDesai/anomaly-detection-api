from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import IsolationForest
import sqlite3
from datetime import datetime
import logging
from typing import List
import os

# ---------- LOGGING SETUP ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- DATABASE SETUP ----------
DB_FILE = "predictions.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                features TEXT,
                prediction TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database init error: {e}")

init_db()

# ---------- FASTAPI APP ----------
app = FastAPI(
    title="Anomaly Detection API",
    description="Real-time anomaly detection using Isolation Forest",
    version="2.0.0"
)

# ---------- CORS (Allow frontend to call) ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- TRAIN MODEL ----------
logger.info("Training Isolation Forest model...")
model = IsolationForest(contamination=0.05, random_state=42)
sample_data = np.random.randn(1000, 5)
model.fit(sample_data)
logger.info("✅ Model trained successfully")

# ---------- REQUEST MODEL ----------
class SensorData(BaseModel):
    features: List[float]  # exactly 5 numbers

class PredictResponse(BaseModel):
    prediction: str
    timestamp: str

# ---------- ENDPOINTS ----------
@app.get("/")
def root():
    return {
        "message": "Anomaly Detection API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": ["/health", "/predict", "/logs", "/metrics"]
    }

@app.get("/health")
def health():
    """Check API health and database connectivity"""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("SELECT 1")
        conn.close()
        return {
            "status": "ok",
            "model_loaded": True,
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "model_loaded": True,
            "database": "disconnected",
            "error": str(e)
        }

@app.post("/predict", response_model=PredictResponse)
def predict(data: SensorData):
    """
    Predict if a given set of features is normal or anomalous.
    
    - **features**: List of 5 float values (e.g., [0.5, 0.3, -0.2, 1.2, -0.5])
    """
    if len(data.features) != 5:
        raise HTTPException(400, "Need exactly 5 features")
    
    try:
        arr = np.array(data.features).reshape(1, -1)
        pred = model.predict(arr)[0]
        result = "anomaly" if pred == -1 else "normal"
        
        # Log to SQLite
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (features, prediction) VALUES (?, ?)",
            (str(data.features), result)
        )
        conn.commit()
        conn.close()
        
        logger.info(f"Prediction: {result} for features: {data.features}")
        
        return {
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.get("/logs")
def get_logs(limit: int = 20):
    """Get recent predictions"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, features, prediction, timestamp FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cur.fetchall()
        conn.close()
        
        return {
            "count": len(rows),
            "logs": [
                {
                    "id": r[0],
                    "features": r[1],
                    "prediction": r[2],
                    "timestamp": r[3]
                }
                for r in rows
            ]
        }
    except Exception as e:
        logger.error(f"Logs error: {e}")
        return {"error": str(e)}

@app.get("/metrics")
def get_metrics():
    """Get usage statistics"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'anomaly'")
        anomalies = cur.fetchone()[0]
        conn.close()
        
        return {
            "total_predictions": total,
            "anomalies": anomalies,
            "normals": total - anomalies,
            "anomaly_rate": round(anomalies / total * 100, 2) if total > 0 else 0,
            "model": "Isolation Forest",
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {"status": "error", "message": str(e)}

# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)