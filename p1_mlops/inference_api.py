import joblib
from fastapi import FastAPI
import pandas as pd
from schema import FEATURE_COLUMNS

app = FastAPI()
model = joblib.load("model.joblib")



@app.post("/predict")
def predict(payload:dict):

    data = {col:0.0 for col in FEATURE_COLUMNS}

    for key, value in payload.items():
        if key in data:
            data[key] = value

    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)


    prob = model.predict_proba(df)[0,1]
    return {"fraud_probability": float(prob)}


