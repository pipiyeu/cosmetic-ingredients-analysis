from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path


# ==========================================
# FastAPI
# ==========================================

app = FastAPI(
    title="Mandali API",
    description="API for Multi-Label Cosmetic Effects Classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Model Directory
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ==========================================
# Load Models
# ==========================================

model = joblib.load(
    MODEL_DIR / "binary_relevance.pkl"
)

tfidf = joblib.load(
    MODEL_DIR / "tfidf.pkl"
)

selected_indices = joblib.load(
    MODEL_DIR / "selected_idx.pkl"
)

mlb = joblib.load(
    MODEL_DIR / "mlb.pkl"
)

label_names = joblib.load(
    MODEL_DIR / "label_names.pkl"
)


# ==========================================
# Request Schema
# ==========================================

class IngredientRequest(BaseModel):
    ingredients: str


# ==========================================
# Root Endpoint
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Mandali API is running"
    }


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
def predict(data: IngredientRequest):

    if not data.ingredients.strip():
        raise HTTPException(
            status_code=400,
            detail="Ingredients cannot be empty."
        )

    ingredients = data.ingredients.strip()

    # ======================================
    # 1. TF-IDF Transformation
    # ======================================

    X_tfidf = tfidf.transform([ingredients])

    # ======================================
    # 2. Feature Selection
    # ======================================

    X_selected = X_tfidf[:, selected_indices]

    # ======================================
    # 3. Binary Relevance Prediction
    # ======================================

    prediction = model.predict(X_selected)

    # BinaryRelevance menghasilkan sparse matrix
    prediction = prediction.toarray()

    # ======================================
    # 4. Convert Prediction to Labels
    # ======================================

    predicted_labels = []

    for i, value in enumerate(prediction[0]):
        if value == 1:
            predicted_labels.append(label_names[i])

    # ======================================
    # 5. Response
    # ======================================

    return {
        "ingredients": ingredients,
        "predictions": predicted_labels
    }