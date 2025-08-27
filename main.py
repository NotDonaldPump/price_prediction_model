from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

# -------- Config API --------
app = FastAPI(title="Student Housing Price API", version="1.0")

# CORS (autorise ton app Flutter à appeler l'API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # mets l’URL de prod de ton app si tu veux restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Chargement du modèle --------
MODEL_PATH = os.getenv("MODEL_PATH", "model/student_housing_price_model.pkl")
model = joblib.load(MODEL_PATH)

# -------- Schéma d’entrée (doit matcher les colonnes d’entraînement) --------
class PredictInput(BaseModel):
    surface_m2: float = Field(..., ge=0)
    num_rooms: float = Field(..., ge=0)
    type: str = Field(..., description="room ou entire_home")
    is_furnished: bool
    wifi_incl: bool
    charges_incl: bool
    car_park: bool
    dist_public_transport_km: float = Field(..., ge=0)
    proxim_hesso_km: float = Field(..., ge=0)

class PredictOutput(BaseModel):
    predicted_price: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOutput)
def predict(payload: PredictInput):
    # Convertir en DataFrame (une seule ligne)
    df = pd.DataFrame([payload.model_dump()])
    y_pred = model.predict(df)[0]
    return {"predicted_price": float(y_pred)}
