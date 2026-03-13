import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING LOGIC ---
model = None
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model file: {e}")
else:
    print(f"❌ Critical Error: {MODEL_PATH} not found in current directory: {os.getcwd()}")

class ChurnData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
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
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
async def root():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(item: ChurnData):
    # Check if model exists before trying to use it
    if model is None:
        raise HTTPException(status_code=500, detail="Machine Learning model is not initialized on the server.")

    try:
        input_dict = item.model_dump() if hasattr(item, 'model_dump') else item.dict()
        input_df = pd.DataFrame([input_dict])

        # Standardizing types
        input_df['MonthlyCharges'] = input_df['MonthlyCharges'].astype(float)
        input_df['TotalCharges'] = input_df['TotalCharges'].astype(float)
        
        # Get prediction and probabilities
        raw_prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        churn_probability = round(float(probabilities[1]) * 100, 2)
        
        if isinstance(raw_prediction, str):
            is_churn = raw_prediction.lower() in ['yes', '1', 'churn']
        else:
            is_churn = int(raw_prediction) == 1

        return {
            "prediction": 1 if is_churn else 0,
            "probability": churn_probability,
            "risk_level": "High" if churn_probability > 70 else "Medium" if churn_probability > 30 else "Low",
            "result": "Customer Will Churn" if is_churn else "Customer Will Stay"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
