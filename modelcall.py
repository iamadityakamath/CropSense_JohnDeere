import numpy as np
import base64
import json
from io import BytesIO
from pathlib import Path
from PIL import Image
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

BASE_DIR = Path(__file__).resolve().parent

# Rebuild model architecture
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(42, activation='softmax')
])

# Load trained weights
model.load_weights(str(BASE_DIR / "cropdoctor_weights.weights.h5"))

# Load config
with open(BASE_DIR / "cropdoctor_config.json", "r") as f:
    config = json.load(f)
class_labels = config["class_labels"]
region_map = config["region_map"]

# FastAPI app
app = FastAPI()

class ImageRequest(BaseModel):
    image_base64: str

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    confidence_pct: str
    region: str

def predict(image_base64):
    payload = image_base64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    payload = "".join(payload.split())
    padding = (-len(payload)) % 4
    if padding:
        payload += "=" * padding
    img_data = base64.b64decode(payload)
    img = Image.open(BytesIO(img_data)).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    disease = class_labels[idx]
    region = region_map.get(disease, "Unknown")
    return {
        "disease": disease,
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.2f}%",
        "region": region
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: ImageRequest):
    return predict(req.image_base64)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)