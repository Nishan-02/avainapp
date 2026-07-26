# backend/app/main.py

import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from . import ml_handler

app = FastAPI(title="Avian Weather Net API")

# FIXED: allow_credentials=True cannot be used with allow_origins=["*"].
# Browsers block this combination. We remove allow_credentials and keep
# allow_origins=["*"] so any frontend (Vercel, localhost) can connect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Avian Weather Net API 🦅"}

# /ping — lightweight health check for UptimeRobot keep-alive
# This endpoint responds instantly and keeps Render from sleeping.
@app.get("/ping")
def ping():
    return {"status": "ok"}

# NEW: Endpoint to get the list of available models
@app.get("/models")
def get_models():
    return {"models": ml_handler.get_loaded_model_names()}

# Ensemble prediction endpoint
@app.post("/predict/ensemble")
async def handle_ensemble_prediction(file: UploadFile = File(...)):
    # Accept any file — let librosa handle format errors gracefully
    audio_bytes = await file.read()
    # Preserve original extension so librosa identifies the format correctly
    ext = os.path.splitext(file.filename or "")[1] or ".wav"
    try:
        prediction = ml_handler.predict_weather(audio_bytes, ext)
        return {"weather_prediction": prediction, "model_used": "Ensemble (All Models)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Single-model prediction endpoint
@app.post("/predict/single")
async def handle_single_prediction(model_name: str = Form(...), file: UploadFile = File(...)):
    # Accept any file — let librosa handle format errors gracefully
    audio_bytes = await file.read()
    # Preserve original extension so librosa identifies the format correctly
    ext = os.path.splitext(file.filename or "")[1] or ".wav"
    try:
        prediction = ml_handler.predict_with_single_model(audio_bytes, model_name, ext)
        return {"weather_prediction": prediction, "model_used": model_name}
    except ValueError as e:
        # Model not found — 400 so the frontend shows a meaningful message
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")