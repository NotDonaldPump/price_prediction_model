# main.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best_student_housing_model.pkl"  # <- ton nouveau modèle
SCHEMA_PATH = APP_DIR / "model_input_schema.json"

# Chargement modèle et schéma
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

try:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        input_schema = json.load(f)
    EXPECTED_COLS: List[str] = input_schema["expected_columns"]
except Exception as e:
    raise RuntimeError(f"Failed to load schema at {SCHEMA_PATH}: {e}")

# ---- API ----
app = FastAPI(title="Student Housing Price API", version="1.0.0")

class PredictionItem(BaseModel):
    surface_m2: float
    num_rooms: float
    type: str
    is_furnished: bool
    wifi_incl: bool
    charges_incl: bool
    car_park: bool
    dist_public_transport_km: float
    proxim_hesso_km: float

class PredictRequest(BaseModel):
    items: List[PredictionItem] = Field(..., description="List of items to score")

class PredictResponse(BaseModel):
    predictions: List[float]
    clipped_to_zero: bool = False

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name}

@app.get("/schema")
def schema():
    return {"expected_columns": EXPECTED_COLS}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # DataFrame dans l'ordre attendu
        df = pd.DataFrame([i.model_dump() for i in req.items])
        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        X = df[EXPECTED_COLS].copy()

        # Assurer les booléens (utile si l’appelant envoie 0/1)
        bool_cols = ["is_furnished", "wifi_incl", "charges_incl", "car_park"]
        for c in bool_cols:
            if c in X.columns:
                X[c] = X[c].astype(bool)

        # Prédiction
        y_pred = model.predict(X)

        # Clamp à >= 0 pour la robustesse métier (prix)
        y_pred = np.array(y_pred, dtype=float)
        clipped = False
        if (y_pred < 0).any():
            y_pred = np.maximum(y_pred, 0.0)
            clipped = True

        return PredictResponse(predictions=y_pred.round(2).tolist(),
                               clipped_to_zero=clipped)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
