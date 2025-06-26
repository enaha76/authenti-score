from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from transformers import (
    PreTrainedTokenizerFast,
    AutoModelForImageClassification,
    AutoImageProcessor,
)
import torch
import os
from PIL import Image
import base64
from io import BytesIO
from watermark import detect_c2pa

# Initialize FastAPI app
app = FastAPI(
    title="AI Text Detection API",
    description="API for detecting whether text is AI-generated or human-written",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths to model and tokenizer files
MODEL_PATH = "../ml_models/onnx/model.onnx"
TOKENIZER_PATH = "../ml_models/tokenizer.json"
IMAGE_MODEL_PATH = "../ml_models/onnx/image_model.onnx"
SMOGY_MODEL_DIR = "../models/smogy"

# Request model
class TextRequest(BaseModel):
    text: str

# Response model
class PredictionResponse(BaseModel):
    text: str
    prediction: str
    is_ai_generated: bool
    confidence: float


class ImagePredictionResponse(BaseModel):
    prediction: str
    is_ai_generated: bool
    confidence: float

# Global variables
ort_session = None
tokenizer_vocab = None
image_session = None
smogy_model = None
smogy_processor = None
MAX_LENGTH = 256
IMAGE_SIZE = 224

# Load model on startup
@app.on_event("startup")
async def load_model():
    global ort_session, tokenizer_vocab, image_session, smogy_model, smogy_processor
    
    try:
        print(f"Loading ONNX model from {MODEL_PATH}")
        # Create ONNX Runtime session with CPU provider
        ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        
        # Initialize tokenizer
        if os.path.exists(TOKENIZER_PATH):
            tokenizer_vocab = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
            print("Tokenizer loaded successfully")
        else:
            print(f"Warning: Tokenizer file not found at {TOKENIZER_PATH}")
        
        print("Model loaded successfully")
        if os.path.exists(IMAGE_MODEL_PATH):
            image_session = ort.InferenceSession(
                IMAGE_MODEL_PATH, providers=["CPUExecutionProvider"]
            )
            print("Image model loaded successfully")
        else:
            print(f"Warning: Image model not found at {IMAGE_MODEL_PATH}")

        if os.path.isdir(SMOGY_MODEL_DIR):
            smogy_model = AutoModelForImageClassification.from_pretrained(
                SMOGY_MODEL_DIR
            )
            smogy_processor = AutoImageProcessor.from_pretrained(SMOGY_MODEL_DIR)
            print("Smogy model loaded successfully")
        else:
            print(f"Warning: Smogy model directory not found at {SMOGY_MODEL_DIR}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    global ort_session, tokenizer_vocab
    
    if ort_session is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    if tokenizer_vocab is None:
        raise HTTPException(
            status_code=503,
            detail="Tokenizer not loaded. Please check server logs."
        )
    
    try:
        # Tokenize input text using the same settings as training
        inputs = tokenizer_vocab(
            request.text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        
        # Get input names from model
        input_names = [x.name for x in ort_session.get_inputs()]
        
        # Prepare inputs dictionary
        ort_inputs = {}
        if "input_ids" in input_names:
            ort_inputs["input_ids"] = inputs["input_ids"]
        if "attention_mask" in input_names:
            ort_inputs["attention_mask"] = inputs["attention_mask"]
        
        # Run inference
        output_names = [x.name for x in ort_session.get_outputs()]
        ort_outputs = ort_session.run(output_names, ort_inputs)
        
        # Get logits (assuming the first output is logits)
        logits = ort_outputs[0]
        
        # Calculate probabilities with softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Get prediction and confidence
        prediction = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][prediction]
        
        # Prepare response
        is_ai_generated = prediction == 1
        prediction_label = "AI-generated" if is_ai_generated else "Human-written"
        
        return PredictionResponse(
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            prediction=prediction_label,
            is_ai_generated=is_ai_generated,
            confidence=float(confidence)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(None), image_base64: str = Form(None)):
    global smogy_model, smogy_processor

    if smogy_model is None or smogy_processor is None:
        raise HTTPException(status_code=503, detail="Smogy model not loaded")

    if file is not None:
        image_bytes = await file.read()
    elif image_base64:
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 data")
    else:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Check for embedded watermarks before running the model
    detected, generator = detect_c2pa(img)
    if detected:
        result = {"prediction": "AI-generated (watermark detected)"}
        if generator:
            result["generator"] = generator
        return result

    inputs = smogy_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = smogy_model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    prediction = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(probabilities[0][prediction])
    is_ai = prediction == 1
    label = "AI-generated" if is_ai else "Real"

    return ImagePredictionResponse(
        prediction=label,
        is_ai_generated=is_ai,
        confidence=confidence,
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    if ort_session is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "API is operational"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Text Detection API",
        "endpoints": {
            "/predict": "POST - Submit text for AI detection",
            "/predict-image": "POST - Submit image for AI detection",
            "/health": "GET - Check API health"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
