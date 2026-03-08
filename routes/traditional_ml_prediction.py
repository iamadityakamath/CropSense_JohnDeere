import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["traditional-ml"])

PREDICT_URL = "https://noninflectionally-nonchivalrous-jacki.ngrok-free.dev/predict"


class TraditionalMLPredictionRequest(BaseModel):
    image_base64: str


@router.post("/traditional_ml_prediction")
def traditional_ml_prediction(payload: TraditionalMLPredictionRequest) -> dict:
    request_body = {"image_base64": payload.image_base64}

    try:
        response = requests.post(PREDICT_URL, json=request_body, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Traditional ML API failed: {exc}") from exc

    try:
        response_body = response.json()
    except ValueError:
        response_body = {"raw": response.text}

    return {
        "status_code": response.status_code,
        "response": response_body,
    }
