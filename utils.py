"""
Utility functions for model loading, image preprocessing, and inference.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import io
from pathlib import Path
from typing import Tuple, Dict, Any


# Device configuration (CPU-only for compatibility)
DEVICE = torch.device("cpu")

# Image preprocessing transforms (ImageNet normalization)
PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Class names mapping (must match alphabetical order from ImageFolder)
# ImageFolder sorts classes alphabetically: aadhaar, other, pan, payment_receipt
CLASS_NAMES = {
    0: "aadhaar",
    1: "other",
    2: "pan",
    3: "payment_receipt"
}

# Display names for API responses
DISPLAY_NAMES = {
    0: "Aadhaar",
    1: "Unknown / Not Supported Document",
    2: "PAN",
    3: "Payment Receipt"
}   

# Inverse mapping for optional filename matching
CLASS_TO_IDX = {v: k for k, v in CLASS_NAMES.items()}


def load_model(model_path: str) -> torch.nn.Module:
    """
    Load a pre-trained ResNet18 model from the saved checkpoint.
    
    Args:
        model_path (str): Path to the saved model.pth file
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode on CPU
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        from torchvision import models
        import torch.nn as nn
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # Check if it's a state_dict or full model
        if isinstance(checkpoint, dict) and 'fc.weight' in checkpoint:
            # It's a state dict, need to create model first
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(512, 4)  # 4 classes: aadhaar, other, pan, payment_receipt
            model.load_state_dict(checkpoint)
        else:
            # It's a full model
            model = checkpoint
        
        model.eval()  # Set to evaluation mode
        model.to(DEVICE)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def preprocess_image(image_file) -> torch.Tensor:
    """
    Preprocess an uploaded image file into a tensor ready for inference.
    
    Args:
        image_file: PIL Image or file-like object containing image data
        
    Returns:
        torch.Tensor: Preprocessed image tensor (1, 3, 224, 224)
        
    Raises:
        ValueError: If image preprocessing fails
    """
    try:
        # If it's a file-like object, open it as PIL Image
        if isinstance(image_file, Image.Image):
            image = image_file
        else:
            image = Image.open(image_file).convert("RGB")
        
        # Apply preprocessing transforms
        image_tensor = PREPROCESS_TRANSFORM(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")


def predict(model: torch.nn.Module, image_tensor: torch.Tensor) -> Tuple[str, int, float]:
    """
    Perform prediction on a preprocessed image tensor.
    
    Args:
        model (torch.nn.Module): Loaded PyTorch model
        image_tensor (torch.Tensor): Preprocessed image tensor (1, 3, 224, 224)
        
    Returns:
        Tuple[str, int, float]: Predicted class name, class index, and confidence score
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        return predicted_class, predicted_idx.item(), confidence_score


def check_filename_match(filename: str, predicted_class: str) -> bool:
    """
    Check if the filename contains text matching the predicted class.
    This is a simple string comparison (case-insensitive).
    
    Args:
        filename (str): Name of the uploaded file
        predicted_class (str): Predicted class label
        
    Returns:
        bool: True if filename contains the predicted class name
    """
    filename_lower = filename.lower()
    return predicted_class.lower() in filename_lower


def is_blue_dominant(image_file, threshold: float = 0.05) -> bool:
    """
    Analyze the image to detect if blue is a significant color.
    Used to verify PAN card classification (PAN cards are typically blue).
    
    Args:
        image_file: PIL Image, file-like object, or bytes
        threshold: Minimum percentage of blue pixels to return True
        
    Returns:
        bool: True if blue color exceeds the threshold
    """
    try:
        # Load image into OpenCV format (BGR)
        if isinstance(image_file, Image.Image):
            pil_img = image_file.convert('RGB')
        elif isinstance(image_file, (io.BytesIO, bytes)):
            if isinstance(image_file, io.BytesIO):
                image_file.seek(0)
                file_bytes = image_file.read()
            else:
                file_bytes = image_file
            pil_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        else:
            # Assume it's a file path or similar
            pil_img = Image.open(image_file).convert('RGB')
            
        # Convert PIL (RGB) to OpenCV (BGR)
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Define range of blue color in HSV
        # Note: Tuned for typical blue PAN backgrounds
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Calculate percentage of blue pixels
        blue_pixel_count = cv2.countNonZero(mask)
        total_pixels = hsv.shape[0] * hsv.shape[1]
        blue_ratio = blue_pixel_count / total_pixels
        
        from ocr_engine import logger
        logger.info(f"Blue color analysis: {blue_ratio:.2%} blue pixels (threshold: {threshold:.2%})")
        
        return blue_ratio >= threshold
        
    except Exception as e:
        from ocr_engine import logger
        logger.error(f"Color analysis failed: {e}")
        return True # Default to True on failure to avoid false rejections
