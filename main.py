import os
import logging
import time
import pickle
import joblib  # Added for better scikit-learn compatibility
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("churn-api")

# ── Config ────────────────────────────────────────────────────────────────────
# Ensure this matches your renamed file
MODEL_FILE = os.getenv("MODEL_FILE", "model.pkl")
PORT = int(os.getenv("PORT", 7860))

ml = {}


class ChurnRequest(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use absolute path to avoid "File Not Found" in Docker
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, MODEL_FILE)

    logger.info("Loading model from '%s' ...", full_path)
    try:
        # joblib is generally preferred for scikit-learn models
        ml["model"] = joblib.load(full_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        ml["model"] = None
    yield
    ml.clear()
    logger.info("Server shutting down.")


app = FastAPI(
    title="Telecom Churn Prediction API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    root_path="/",  # Important for Hugging Face Proxy
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ... [Timing middleware and GET routes remain the same] ...

@app.middleware("http")
async def timing_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


@app.get("/", include_in_schema=False)
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Churn API v2.0.0 is live. POST to /predict."}


@app.get("/health", tags=["System"])
async def health():
    ready = ml.get("model") is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ok" if ready else "degraded", "model_loaded": ready}
    )


@app.post("/predict", tags=["Inference"])
async def predict(req: ChurnRequest):
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check logs.")

    try:
        t0 = time.perf_counter()
        data = req.model_dump()
        input_df = pd.DataFrame([data])

        # Inference
        raw_pred = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        churn_prob = round(float(probabilities[1]) * 100, 2)

        # Robust handling of different label types (string vs int)
        if isinstance(raw_pred, (str, np.str_)):
            is_churn = raw_pred.lower() in ["yes", "1", "churn"]
        else:
            is_churn = int(raw_pred) == 1

        risk_level = "High" if churn_prob > 70 else "Medium" if churn_prob > 30 else "Low"
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "prediction": 1 if is_churn else 0,
            "probability": churn_prob,
            "risk_level": risk_level,
            "result": "Customer Will Churn" if is_churn else "Customer Will Stay",
            "latency_ms": latency_ms,
        }

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)