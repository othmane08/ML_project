from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os


PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "churn_pipeline.pkl")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "feature_columns.pkl")

pipeline = joblib.load(PIPELINE_PATH)
feature_columns = joblib.load(FEATURES_PATH)  

app = FastAPI(title="Churn Prediction API")

class Customer(BaseModel):
    # define the expected fields (example; adapt to your features)
    customerID: str | None = None
    gender: str | None = None
    SeniorCitizen: int | None = None
    Partner: int | None = None
    Dependents: int | None = None
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    TotalServices: int
    TenureNorm: float
    # plus the one-hot columns you used; alternatively accept a free dict below
    # To keep it flexible, we accept additional fields via extra model later

class RawCustomer(BaseModel):
    data: dict  # expects a dict mapping feature names -> values

@app.get("/")
def root():
    return {"status": "ok", "message": "Churn Prediction API"}

@app.post("/predict")
def predict(payload: RawCustomer):
    try:
        row = payload.data
        # Build dataframe in the same column order as training
        # Start with empty DF with feature_columns
        X = pd.DataFrame([row], columns=feature_columns)
        # Fill missing columns with 0 (for dummy columns) or NaN for numeric
        for c in feature_columns:
            if c not in X.columns:
                # default for dummies = 0, numeric = 0
                X[c] = 0
        # Reorder
        X = X[feature_columns]

        # Ensure numeric types where possible
        X = X.apply(pd.to_numeric, errors="ignore")

        # predict
        proba = pipeline.predict_proba(X)[:, 1][0]
        pred = int(pipeline.predict(X)[0])

        return {"prediction": pred, "probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_bulk")
def predict_bulk(payload: dict):
    # expects {"instances": [ {feature: val, ...}, {...} ]}
    try:
        instances = payload.get("instances", [])
        X = pd.DataFrame(instances)
        # ensure all columns present
        for c in feature_columns:
            if c not in X.columns:
                X[c] = 0
        X = X[feature_columns]
        X = X.apply(pd.to_numeric, errors="ignore")
        probas = pipeline.predict_proba(X)[:, 1].tolist()
        preds = pipeline.predict(X).astype(int).tolist()
        return {"predictions": preds, "probabilities": probas}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))