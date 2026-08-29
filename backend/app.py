from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from pathlib import Path
import urllib.request
import tempfile
import gradio as gr


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Mandali API",
    description="API for Multi-Label Cosmetic Effects Classification",
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://cosmetic-ingredients-analysis.vercel.app",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# =========================================================
# MODEL URL
# =========================================================

MODEL_URL = (
    "https://huggingface.co/fiyaw/Mandali-model-XGBOOST/"
    "resolve/main/binary_relevance.pkl"
)


# =========================================================
# LOAD BINARY RELEVANCE MODEL
# =========================================================

MODEL_PATH = MODEL_DIR / "binary_relevance.pkl"


def load_binary_relevance_model():

    # Jika model tersedia di repository
    if MODEL_PATH.exists():
        print("Loading binary_relevance.pkl from local models folder...")
        return joblib.load(MODEL_PATH)

    # Jika tidak tersedia, download ke temporary directory
    temp_path = Path(tempfile.gettempdir()) / "binary_relevance.pkl"

    if not temp_path.exists():
        print("Downloading binary_relevance.pkl...")

        try:
            urllib.request.urlretrieve(
                MODEL_URL,
                temp_path
            )

            print("binary_relevance.pkl downloaded.")

        except Exception as error:
            raise RuntimeError(
                f"Failed to download binary_relevance.pkl: {error}"
            )

    return joblib.load(temp_path)


# =========================================================
# LOAD ALL MODELS
# =========================================================

print("Loading Mandali models...")

model = load_binary_relevance_model()

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


# =========================================================
# REQUEST SCHEMA
# =========================================================

class IngredientRequest(BaseModel):
    ingredients: str


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/api")
def root():
    return {
        "message": "Mandali API is running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "Mandali API"
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/api/predict")
def predict(data: IngredientRequest):

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not data.ingredients or not data.ingredients.strip():
        raise HTTPException(
            status_code=400,
            detail="Ingredients cannot be empty."
        )

    ingredients = data.ingredients.strip()


    # -----------------------------------------------------
    # 1. TF-IDF
    # -----------------------------------------------------

    try:

        X_tfidf = tfidf.transform(
            [ingredients]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"TF-IDF transformation failed: {error}"
        )


    # -----------------------------------------------------
    # 2. Feature Selection
    # -----------------------------------------------------

    try:

        X_selected = X_tfidf[
            :,
            selected_indices
        ]

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Feature selection failed: {error}"
        )


    # -----------------------------------------------------
    # 3. Binary Relevance Prediction
    # -----------------------------------------------------

    try:

        prediction = model.predict(
            X_selected
        )

        if hasattr(prediction, "toarray"):
            prediction = prediction.toarray()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )


    # -----------------------------------------------------
    # 4. Convert Prediction to Labels
    # -----------------------------------------------------

    predicted_labels = []

    for index, value in enumerate(prediction[0]):

        if value == 1:

            if index < len(label_names):

                predicted_labels.append(
                    str(label_names[index])
                )


    # -----------------------------------------------------
    # 5. Response
    # -----------------------------------------------------

    return {
        "ingredients": ingredients,
        "predictions": predicted_labels
    }


# =========================================================
# GRADIO MOUNT (WAJIB UNTUK HUGGINGFACE SPACES)
# =========================================================

with gr.Blocks(title="Mandali API") as demo:
    gr.Markdown("# 🚀 Mandali API is Live!")
    gr.Markdown("FastAPI backend aktif dan siap menerima request dari Next.js pada endpoint `/api/predict`.")

app = gr.mount_gradio_app(app, demo, path="/")