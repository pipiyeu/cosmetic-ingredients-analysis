from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from pathlib import Path
import urllib.request


# ==========================================
# FastAPI
# ==========================================

app = FastAPI(
    title="Mandali API",
    description="API for Multi-Label Cosmetic Effects Classification",
    version="1.0.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Model Directory
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Download Binary Relevance Model
# ==========================================

MODEL_URL = (
    "https://huggingface.co/fiyaw/Mandali-model-XGBOOST/"
    "resolve/main/binary_relevance.pkl"
)

MODEL_PATH = MODEL_DIR / "binary_relevance.pkl"


if not MODEL_PATH.exists():
    print("Downloading binary_relevance.pkl...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("binary_relevance.pkl downloaded.")


# ==========================================
# Load Models
# ==========================================

print("Loading models...")

try:

    model = joblib.load(
        MODEL_PATH
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

    print("All models loaded successfully.")

except Exception as e:

    print(f"Model loading error: {e}")

    raise


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
        "message": "Mandali API is running",
        "status": "success"
    }


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
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

    try:

        # ==================================
        # 1. TF-IDF
        # ==================================

        X_tfidf = tfidf.transform(
            [ingredients]
        )


        # ==================================
        # 2. Feature Selection
        # ==================================

        X_selected = X_tfidf[
            :,
            selected_indices
        ]


        # ==================================
        # 3. Prediction
        # ==================================

        prediction = model.predict(
            X_selected
        )


        # ==================================
        # 4. Convert Sparse Matrix
        # ==================================

        if hasattr(prediction, "toarray"):

            prediction = prediction.toarray()


        # ==================================
        # 5. Convert to Labels
        # ==================================

        predicted_labels = []

        for i, value in enumerate(prediction[0]):

            if value == 1:

                predicted_labels.append(
                    label_names[i]
                )


        # ==================================
        # 6. Response
        # ==================================

        return {
            "ingredients": ingredients,
            "predictions": predicted_labels
        }

    except Exception as e:

        print(f"Prediction error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )