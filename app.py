"""
FastAPI application for identity document extraction and validation.

Endpoints:
    GET /           - Web UI for document upload
    POST /extract   - Strict rule-based identification and extraction
    GET /docs       - Interactive Swagger UI documentation

Usage:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import logging
from typing import Dict, Any
import io

from ocr_engine import identify_and_extract


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ID Shield Pro API",
    description="High-precision identification and data extraction for identity documents",
    version="2.1.0"
)


def is_valid_image(file_content: bytes) -> bool:
    """Check if file content is a valid image."""
    image_signatures = [b'\xFF\xD8\xFF', b'\x89PNG', b'GIF8', b'BM', b'RIFF']
    return any(file_content.startswith(sig) for sig in image_signatures)


@app.get("/", tags=["UI"])
async def get_ui():
    return FileResponse("index.html")


@app.post("/extract", tags=["Extraction"])
async def extract_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Perform high-precision rule-based identification and extraction.
    Returns: document_type, validation_status, extracted_fields.
    """
    try:
        contents = await file.read()
        if not contents or not is_valid_image(contents):
             raise HTTPException(status_code=400, detail="Invalid image file format.")
             
        # Rule Engine Processing
        result = identify_and_extract(io.BytesIO(contents))
        
        logger.info(f"Analyzed: {result['document_type']} | Status: {result['validation_status']}")
        return result

    except Exception as e:
        logger.error(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs", include_in_schema=False)
async def get_docs():
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
