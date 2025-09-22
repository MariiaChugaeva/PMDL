# code/deployment/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

# Paths
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
MODEL_PATH = REPO_ROOT / "models" / "digit_recognizer_cnn.h5"

# FastAPI app
app = FastAPI(title="MNIST Digit Recognition API")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    pixels: List[float]  # flattened 28x28 = 784 floats

class PredictResponse(BaseModel):
    predicted: int
    probabilities: List[float]

# Load model
try:
    model = load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    model = None
    print("Failed to load model:", e)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        pixels = np.array(req.pixels, dtype=np.float32)
        if pixels.size != 784:
            raise HTTPException(status_code=400, detail=f"Expected 784 pixels (28x28), got {pixels.size}")

        # Reshape input for Keras
        img = pixels.reshape(1, 28, 28)

        # If model expects channel dimension (28,28,1)
        if len(model.input_shape) == 4:  # e.g. (None, 28, 28, 1)
            img = img[..., np.newaxis]

        probs = model.predict(img)[0].tolist()
        pred = int(np.argmax(probs))

        return PredictResponse(predicted=pred, probabilities=probs)

    except Exception as e:
        # Print traceback to server logs
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

