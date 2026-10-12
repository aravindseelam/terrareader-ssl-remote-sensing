"""
FastAPI inference server for the SSL-pretrained land-cover classifier.

Local:   uvicorn deployment.app:app --reload --port 8000
Online:  served by the Dockerfile (port 7860 on Hugging Face Spaces)

Endpoints
  GET  /health        liveness + whether the model loaded
  GET  /api/info      model metadata (classes, architecture, input size)
  GET  /api/samples   sample tiles available for the one-click demo
  POST /predict       multipart image upload -> ranked class probabilities
"""

import io
import os
import sys
import time
from contextlib import asynccontextmanager

# Make the project root importable no matter where uvicorn is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from finetune.classifier import RemoteSensingClassifier
from ssl_pretrain.augmentations import get_eval_transform

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "checkpoints", "classifier_final.pt"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = {"model": None, "class_names": [], "image_size": 64, "transform": None, "arch": None}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[startup] WARNING: no checkpoint at {MODEL_PATH}; /predict will return 503.")
        return
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = RemoteSensingClassifier(arch=ckpt["arch"], num_classes=ckpt["num_classes"], small_image=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    state.update(
        model=model,
        class_names=ckpt["class_names"],
        image_size=ckpt.get("image_size", 64),
        transform=get_eval_transform(ckpt.get("image_size", 64)),
        arch=ckpt["arch"],
    )
    print(f"[startup] Loaded {ckpt['arch']} from {MODEL_PATH} on {device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="TerraReader - SSL Land-Cover Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None, "device": str(device)}


@app.get("/api/info")
def info():
    return {
        "architecture": state["arch"],
        "classes": state["class_names"],
        "input_size": state["image_size"],
        "pretraining": "SimCLR (self-supervised) on unlabeled EuroSAT tiles",
        "dataset": "EuroSAT (Sentinel-2 RGB, 10 classes)",
    }


@app.get("/api/samples")
def samples():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return [{"name": os.path.splitext(f)[0], "url": f"/samples/{f}"} for f in files]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(503, f"Model not loaded. Place a trained checkpoint at {MODEL_PATH}.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB).")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read the upload as an image.")

    tensor = state["transform"](image).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = F.softmax(state["model"](tensor), dim=1).squeeze(0).cpu()
    latency_ms = (time.perf_counter() - t0) * 1000

    names = state["class_names"]
    ranked = sorted(
        ({"label": names[i], "probability": float(probs[i])} for i in range(len(names))),
        key=lambda d: d["probability"],
        reverse=True,
    )
    return JSONResponse(
        {
            "predicted_label": ranked[0]["label"],
            "confidence": ranked[0]["probability"],
            "all_probabilities": ranked,
            "latency_ms": round(latency_ms, 1),
            "input_size": [image.width, image.height],
        }
    )


# Frontend (must be mounted last so API routes above take priority)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
