# backend/app/main.py (Updated)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from . import ml_handler

app = FastAPI(title="Avian Weather Net API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Avian Weather Net API 🦅"}

# NEW: Endpoint to get the list of available models
@app.get("/models")
def get_models():
    return {"models": ml_handler.get_loaded_model_names()}

# UPDATED: This is your endpoint for ensemble prediction
@app.post("/predict/ensemble")
async def handle_ensemble_prediction(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    audio_bytes = await file.read()
    try:
        prediction = ml_handler.predict_weather(audio_bytes)
        # We now return the model_used for clarity on the frontend
        return {"weather_prediction": prediction, "model_used": "Ensemble (All Models)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

# NEW: Endpoint for single model prediction
@app.post("/predict/single")
async def handle_single_prediction(model_name: str = Form(...), file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
        
    audio_bytes = await file.read()
    try:
        prediction = ml_handler.predict_with_single_model(audio_bytes, model_name)
        # We return the specific model_name used
        return {"weather_prediction": prediction, "model_used": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")