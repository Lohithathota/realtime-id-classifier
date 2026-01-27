"""
FastAPI application for real-time image classification.

Endpoints:
    GET /           - Health check and status
    POST /predict   - Image classification endpoint
    GET /docs       - Interactive Swagger UI documentation

Usage:
    uvicorn app:app --reload
    
Then access:
    - API: http://localhost:8000
    - Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import logging
from typing import Dict, Any
import io

from utils import load_model, preprocess_image, predict, check_filename_match, DISPLAY_NAMES


# Configuration
MODEL_PATH = "model.pth"
CONFIDENCE_THRESHOLD_LOW = 0.50  # Below this is strictly Unknown
CONFIDENCE_THRESHOLD_HIGH = 0.70 # Below this triggers a low confidence warning

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="Real-time image classification for Aadhaar and PAN documents",
    version="1.0.0"
)

# Global model variable
model = None


def is_valid_image(file_content: bytes) -> bool:
    """
    Check if file content is a valid image by checking magic bytes.
    
    Args:
        file_content (bytes): File content to validate
        
    Returns:
        bool: True if file appears to be a valid image
    """
    if len(file_content) < 8:
        return False
    
    # Magic bytes for common image formats
    image_signatures = {
        b'\xFF\xD8\xFF': 'jpeg',  # JPEG
        b'\x89PNG': 'png',         # PNG
        b'GIF8': 'gif',            # GIF
        b'BM': 'bmp',              # BMP
        b'RIFF': 'webp',           # WebP (RIFF format)
    }
    
    for signature in image_signatures:
        if file_content.startswith(signature):
            return True
    
    return False


@app.on_event("startup")
async def startup_event():
    """
    Initialize model on application startup.
    Loads the trained model from model.pth.
    """
    global model
    
    try:
        if not Path(MODEL_PATH).exists():
            logger.error(f"Model file not found at {MODEL_PATH}")
            logger.info("Please run 'python train.py' to train the model first")
            return
        
        model = load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint that returns API status and model availability.
    
    Returns:
        dict: Status message and model availability
    """
    return {
        "status": "ok",
        "service": "Image Classification API",
        "model_loaded": model is not None,
        "message": "API is running. Use POST /predict to classify images."
    }


@app.post("/predict", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Classify an uploaded image into four categories: aadhaar, pan, payment_receipt, or other.
    
    Args:
        file (UploadFile): Image file to classify (jpg, png, etc.)
        
    Returns:
        dict: Contains:
            - predicted_class: "aadhaar", "pan", "payment_receipt", or "other"
            - confidence: Confidence score as percentage (0-100)
            - uploaded_filename: Name of the uploaded file
            - filename_matches_prediction: Boolean indicating if filename text matches prediction
            - is_document: Boolean indicating if image is a valid document (not 'other')
            
    Raises:
        HTTPException: If model is not loaded or image processing fails
    """
    
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please check server logs."
        )
    
    try:
        # Read the file into memory first
        contents = await file.read()
        
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        # Check file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum size: 10MB"
            )
        
        # Validate file is actually an image using magic bytes
        if not is_valid_image(contents):
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Please upload a valid image (JPG, PNG, GIF, BMP, or WebP)"
            )
        
        image_file = io.BytesIO(contents)
        
        # Preprocess the image
        image_tensor = preprocess_image(image_file)
        
        # Perform prediction
        predicted_class, class_idx, confidence = predict(model, image_tensor)
        
        # Get display name for the predicted class
        display_name = DISPLAY_NAMES.get(class_idx, predicted_class)
        
        # Determine if it's a valid document (not "other")
        # Determine if it's a valid document (not "other")
        is_document = predicted_class != "other"
        
        # Check if filename matches prediction
        filename_match = check_filename_match(file.filename if file.filename else "unknown", predicted_class)
        
        message = ""
        
        # --- CONFIDENCE THRESHOLD LOGIC ---
        if confidence < CONFIDENCE_THRESHOLD_LOW:
            # Case 1: Confidence is too low (< 50%). Treat as Unknown.
            predicted_class = "other"
            display_name = DISPLAY_NAMES[1] # "Unknown / Not Supported Document"
            is_document = False
            message = "Confidence too low. Classified as Unknown/Unsupported."
            
        elif confidence < CONFIDENCE_THRESHOLD_HIGH:
            # Case 2: Confidence is medium (50% - 70%). Keep class but warn user.
            low_conf_warning = " (Low Confidence - Verify Manually)"
            
            if predicted_class == "other":
                message = "Unknown or unsupported document type" + low_conf_warning
            elif predicted_class == "aadhaar":
                message = "Valid Aadhaar document detected" + low_conf_warning
            elif predicted_class == "pan":
                message = "Valid PAN document detected" + low_conf_warning
            elif predicted_class == "payment_receipt":
                message = "Valid Payment Receipt detected" + low_conf_warning
                
        else:
            # Case 3: High Confidence (>= 70%). Standard success messages.
            if predicted_class == "other":
                message = "Unknown or unsupported document type"
            elif predicted_class == "aadhaar":
                message = "Valid Aadhaar document detected"
            elif predicted_class == "pan":
                message = "Valid PAN document detected"
            elif predicted_class == "payment_receipt":
                message = "Valid Payment Receipt detected"
        
        # Convert confidence to percentage
        confidence_percentage = round(confidence * 100, 2)
        
        logger.info(
            f"Prediction: {predicted_class} (display: {display_name}, confidence: {confidence_percentage:.2f}%) "
            f"for file: {file.filename if file.filename else 'unknown'}"
        )
        
        return {
            "predicted_class": display_name,
            "confidence": confidence_percentage,
            "uploaded_filename": file.filename if file.filename else "unknown",
            "filename_matches_prediction": filename_match,
            "is_document": is_document,
            "message": message
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# --- OCR ENDPOINT ---
from ocr_engine import process_document

@app.post("/predict_with_ocr", tags=["Prediction"])
async def predict_image_with_ocr(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Classify and perform OCR on an uploaded image.
    Only runs OCR if the image is classified as 'aadhaar' or 'pan'.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # 1. Read File
        contents = await file.read()
        if not contents or len(contents) > 10 * 1024 * 1024:
             raise HTTPException(status_code=400, detail="Invalid file size or empty")
        
        if not is_valid_image(contents):
             raise HTTPException(status_code=400, detail="Invalid image format")
             
        image_file = io.BytesIO(contents)
        
        # 2. Classification
        image_tensor = preprocess_image(io.BytesIO(contents)) # Create fresh stream
        predicted_class, class_idx, confidence = predict(model, image_tensor)
        
        # Basic response structure
        response = {
            "classification": {
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "display_name": DISPLAY_NAMES.get(class_idx, predicted_class)
            },
            "ocr_data": None,
            "message": "OCR not performed for this document type"
        }

        # 3. OCR if applicable
        if confidence >= CONFIDENCE_THRESHOLD_LOW and predicted_class in ['aadhaar', 'pan']:
            # Reset stream for OCR
            image_file.seek(0)
            ocr_result = process_document(image_file, predicted_class)
            response['ocr_data'] = ocr_result
            response['message'] = f"OCR performed for {predicted_class}"
            
            # If classification was high confidence but OCR failed completely, warning?
            if ocr_result['ocr_status'] == 'FAIL':
                 response['message'] += " (OCR Failed)"
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/docs", include_in_schema=False)
async def get_docs():
    """
    Swagger UI documentation endpoint.
    Access at http://localhost:8000/docs
    """
    pass


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Image Classification API...")
    logger.info("Visit http://localhost:8000/docs for interactive documentation")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
