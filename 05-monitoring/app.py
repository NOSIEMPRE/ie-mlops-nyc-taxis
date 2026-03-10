"""FastAPI service for NYC Taxi Duration Prediction.

Loads a single sklearn Pipeline artifact (DictVectorizer + XGBRegressor) from MLflow
using the run id in run_id.txt and exposes /predict.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc  # load unified pipeline
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
RUN_ID: Optional[str] = None
model = None  # sklearn Pipeline (pyfunc)

# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------
class RideRequest(BaseModel):
    """Input payload for a duration prediction."""
    PULocationID: int = Field(..., ge=1, description="Pickup Location ID")
    DOLocationID: int = Field(..., ge=1, description="Dropoff Location ID")
    trip_distance: float = Field(..., gt=0, description="Trip distance in miles")

    class Config:
        json_schema_extra = {
            "example": {
                "PULocationID": 138,
                "DOLocationID": 236,
                "trip_distance": 2.5,
            }
        }


class PredictionResponse(BaseModel):
    duration: float
    model_version: str


# ---------------------------------------------------------------------------
# Lifespan: load artifacts once at startup
# ---------------------------------------------------------------------------
def _load_model(run_id: str):
    """Load model: prefer local path (avoids MLflow server fetch hang on macOS)."""
    model_path_file = "model_path.txt"
    if os.path.exists(model_path_file):
        with open(model_path_file, "r") as f:
            local_path = f.read().strip()
        if os.path.exists(local_path):
            print(f"[startup] Loading from local: {local_path}")
            return mlflow.pyfunc.load_model(local_path)
    print(f"[startup] Loading from MLflow: runs:/{run_id}/model")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUN_ID, model

    with open("run_id.txt", "r") as f:
        RUN_ID = f.read().strip()
    model = _load_model(RUN_ID)
    print("[startup] Loaded Pipeline artifact 'model'.")
    yield
    # (No teardown needed)


app = FastAPI(
    title="NYC Taxi Duration Predictor",
    description="Predict taxi trip duration (minutes).",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the NYC Taxi Duration prediction API"}


@app.get("/health")
def health():
    return {"status": "ok", "run_id": RUN_ID}


@app.post("/predict", response_model=PredictionResponse)
def predict(ride: RideRequest):
    feature_dict = {
        "PU_DO": f"{ride.PULocationID}_{ride.DOLocationID}",
        "trip_distance": ride.trip_distance,
    }
    pred = model.predict([feature_dict])[0]
    return PredictionResponse(duration=float(pred), model_version=RUN_ID or "unknown")


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)

