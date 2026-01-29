"""
FastAPI application for real-time Aadhaar/PAN OCR extraction.
Fixed: BytesIO handling, error boundaries, OCR integration.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import io
import json

# FIXED: Import from ocr_engine
from ocr_engine import process_document

# Configuration
CONFIDENCE_THRESHOLD_LOW = 0.50

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Aadhaar & PAN OCR API",
    description="Production-ready OCR for Indian ID documents",
    version="2.1.0"
)

@app.get("/", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "ID OCR API v2.1.0",
        "timestamp": "2026-01-28"
    }


@app.post("/process_document", tags=["OCR Processing"])
async def process_document_endpoint(
    file: UploadFile = File(..., description="Upload Aadhaar or PAN card image")
) -> Dict[str, Any]:
    """
    Unified endpoint for Aadhaar/PAN OCR processing.
    
    Steps:
    1. Validate image format & size
    2. Auto-detect document type via OCR
    3. Extract & validate all fields
    4. Return structured JSON
    """
    try:
        # 1. Read & validate file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large (>10MB)")
        
        # 2. FIXED: Direct OCR processing with BytesIO
        logger.info(f"Processing {file.filename} ({len(contents)} bytes)")
        
        # Call OCR engine - it will auto-detect document type!
        result =process_document(io.BytesIO(contents))
        
        logger.info(f"Processing complete: {result.get('overall_status', 'UNKNOWN')}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
