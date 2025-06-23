from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from transformers import PreTrainedTokenizerFast
import os

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

# Request model
class TextRequest(BaseModel):
    text: str

# Response model
class PredictionResponse(BaseModel):
    text: str
    prediction: str
    is_ai_generated: bool
    confidence: float

# Global variables
ort_session = None
tokenizer_vocab = None
MAX_LENGTH = 256

# Load model on startup
@app.on_event("startup")
async def load_model():
    global ort_session, tokenizer_vocab
    
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
            "/health": "GET - Check API health"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
