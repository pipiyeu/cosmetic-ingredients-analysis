from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(
    title="Mandali API",
    description="API for Multi-Label Cosmetic Effects Classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Lazy Loading Container
_model = None
_tfidf = None
_selected_features = None
_mlb = None
_label_names = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_DIR / "binary_relevance.pkl")
    return _model

def get_tfidf():
    global _tfidf
    if _tfidf is None:
        _tfidf = joblib.load(MODEL_DIR / "tfidf.pkl")
    return _tfidf

def get_selected_features():
    global _selected_features
    if _selected_features is None:
        _selected_features = joblib.load(MODEL_DIR / "selected_features.pkl")
    return _selected_features

def get_mlb():
    global _mlb
    if _mlb is None:
        _mlb = joblib.load(MODEL_DIR / "mlb.pkl")
    return _mlb

def get_label_names():
    global _label_names
    if _label_names is None:
        _label_names = joblib.load(MODEL_DIR / "label_names.pkl")
    return _label_names

class IngredientRequest(BaseModel):
    ingredients: str

@app.get("/api")
def root():
    return {"message": "Mandali API is running"}

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Mandali API"}

@app.post("/api/predict")
def predict(data: IngredientRequest):
    if not data.ingredients or not data.ingredients.strip():
        raise HTTPException(status_code=400, detail="Ingredients cannot be empty.")

    ingredients = data.ingredients.strip()

    try:
        tfidf_inst = get_tfidf()
        selected_features_inst = get_selected_features()
        model_inst = get_model()
        label_names_inst = get_label_names()

        X_tfidf = tfidf_inst.transform([ingredients])
        X_selected = X_tfidf[:, selected_features_inst]

        prediction = model_inst.predict(X_selected)
        if hasattr(prediction, "toarray"):
            prediction = prediction.toarray()

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {error}")

    predicted_labels = []
    for index, value in enumerate(prediction[0]):
        if value == 1:
            if index < len(label_names_inst):
                predicted_labels.append(str(label_names_inst[index]))

    return {
        "ingredients": ingredients,
        "predictions": predicted_labels
    }