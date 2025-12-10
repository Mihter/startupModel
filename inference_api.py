from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import datetime
import xgboost as xgb

MODEL_PATH = "model_2_target_2.pkl"
with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    feature_names = saved["feature_names"]

app = FastAPI(title="Period Prediction API")

class PredictRequest(BaseModel):
    MeanCycleLength: float
    LengthofMenses: float
    MeanMensesLength: float
    Age: float
    Height: float
    Weight: float
    BMI: float
    last_period_start: str | None = None  # "YYYY-MM-DD"


@app.post("/predict")
def predict(req: PredictRequest):

    row = {
        "MeanCycleLength": req.MeanCycleLength,
        "LengthofMenses": req.LengthofMenses,
        "MeanMensesLength": req.MeanMensesLength,
        "Age": req.Age,
        "Height": req.Height,
        "Weight": req.Weight,
        "BMI": req.BMI,
    }

    df = pd.DataFrame([row])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    pred = model.predict(df)[0]
    predicted_cycle_length = float(pred[0])
    predicted_menses_length = float(pred[1])

    result = {
        "predicted_cycle_length": predicted_cycle_length,
        "predicted_menses_length": predicted_menses_length
    }

    if req.last_period_start:
        last_date = datetime.date.fromisoformat(req.last_period_start)
        next_period = last_date + datetime.timedelta(days=int(predicted_cycle_length))
        result["predicted_start_date"] = next_period.isoformat()

    return result