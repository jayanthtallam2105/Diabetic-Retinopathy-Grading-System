from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import build_models, get_device, preprocess_image_bytes, run_inference_all
from image_validation import validate_image_quality


app = FastAPI(
    title="Diabetic Retinopathy Grading API",
    version="1.0.0",
    description="Research prototype: runs 5 deep learning models in parallel and compares predictions. Not for clinical use.",
)

# CORS for local dev. Tighten this for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

DEVICE: torch.device = get_device()
MODELS: Dict[str, torch.nn.Module] = {}


@app.on_event("startup")
async def on_startup() -> None:
    """
    Load all models ONCE at startup (strict=True).
    """
    global MODELS
    if MODELS:
        # Avoid re-loading if the startup hook fires again in reload scenarios.
        return

    if not MODELS_DIR.exists():
        raise RuntimeError(f"Models directory not found: {MODELS_DIR}")

    MODELS = build_models(MODELS_DIR, DEVICE)
    print(f"[Startup] Loaded {len(MODELS)} models on {DEVICE}: {list(MODELS.keys())}")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": list(MODELS.keys()),
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Input: multipart/form-data with field 'image'
    Output:
      {
        "efficientnet":   {grade, confidence, probs} OR {grade:null,...,error:"..."},
        "resnet50":       {...},
        "vit":            {...},
        "hybrid_effvit":  {...},
        "hybrid_resvit":  {...}
      }
    """
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file.")

    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    # Image quality validation BEFORE inference
    quality_info = validate_image_quality(data)
    if not quality_info["accepted"]:
        raise HTTPException(
            status_code=400,
            detail=quality_info["message"],
        )

    try:
        x = preprocess_image_bytes(data, DEVICE)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}")

    try:
        results = await run_inference_all(MODELS, x)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # Ensure all expected keys are present (even if a model didn't load for some reason).
    expected = ["efficientnet", "resnet50", "vit", "hybrid_effvit", "hybrid_resvit"]
    for k in expected:
        results.setdefault(k, {"grade": None, "confidence": None, "probs": None, "error": "Model missing on server."})

    # Include quality info in response
    return {
        "quality": quality_info,
        "predictions": results,
    }

