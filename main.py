import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Define the Input Schema
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

app = FastAPI()

# Enable CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

# IMPORTANT: Add a Health Check route for Render
@app.get("/")
async def root():
    return {"message": "API is online and ready for predictions!"}

@app.post("/predict")
async def predict(item: ChurnData):
    try:
        input_dict = item.model_dump() if hasattr(item, 'model_dump') else item.dict()
        input_df = pd.DataFrame([input_dict])

        # Ensure numeric columns are strictly typed
        input_df['MonthlyCharges'] = input_df['MonthlyCharges'].astype(float)
        input_df['TotalCharges'] = input_df['TotalCharges'].astype(float)
        input_df['tenure'] = input_df['tenure'].astype(int)
        input_df['SeniorCitizen'] = input_df['SeniorCitizen'].astype(int)

        column_order = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
        input_df = input_df[column_order]

        # Get raw prediction
        raw_prediction = model.predict(input_df)[0]

        # SMART CHECK: Handle if model returns "Yes"/"No" or 1/0
        if isinstance(raw_prediction, str):
            is_churn = raw_prediction.lower() in ['yes', '1', 'true', 'churn']
        else:
            is_churn = int(raw_prediction) == 1

        # Get probability
        probabilities = model.predict_proba(input_df)[0]
        # Most models: [0] is Stay, [1] is Churn
        churn_probability = round(float(probabilities[1]) * 100, 2)

        risk_level = "High" if churn_probability > 70 else "Medium" if churn_probability > 30 else "Low"

        return {
            "prediction": 1 if is_churn else 0,
            "probability": churn_probability,
            "risk_level": risk_level,
            "result": "Customer Will Churn" if is_churn else "Customer Will Stay"
        }

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))




