
# Real-Time Image Classification API

A production-ready FastAPI application for real-time classification of Aadhaar and PAN documents using PyTorch and ResNet18.

## Features

- **Offline Model Training**: Train a ResNet18 model on custom dataset
- **Online Inference**: FastAPI server for real-time image classification
- **CPU-Compatible**: Works on CPU (no GPU required)
- **Type-Safe**: Full type hints and docstrings
- **Production-Ready**: Proper error handling, logging, and validation
- **Interactive API Docs**: Swagger UI at `/docs` endpoint
- **Filename Matching**: Optional feature to compare filename with prediction

---

## Project Structure

```
realtime-id-classifier/
├── app.py                    # FastAPI application with inference endpoints
├── utils.py                  # Model loading, preprocessing, and prediction functions
├── train.py                  # Training script
├── model.pth                 # Trained model (generated after training)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── dataset/
    ├── train/
    │   ├── aadhaar/          # Training images of Aadhaar documents
    │   └── pan/              # Training images of PAN documents
    └── val/
        ├── aadhaar/          # Validation images of Aadhaar documents
        └── pan/              # Validation images of PAN documents
```

---

## Installation

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

### Directory Structure

Create the following directory structure and populate with images:

```
dataset/
├── train/
│   ├── aadhaar/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ... (more images)
│   └── pan/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ... (more images)
└── val/
    ├── aadhaar/
    │   ├── image1.jpg
    │   └── ... (more images)
    └── pan/
        ├── image1.jpg
        └── ... (more images)
```

### Image Requirements

- **Format**: JPG, PNG, GIF, BMP, or WebP
- **Size**: Any size (will be resized to 224×224)
- **Count**: At least 10-20 images per class for training
- **Validation Split**: 20-30% of data in validation set, rest in training set

### Example Dataset Creation

```bash
# Create directories
mkdir -p dataset/train/aadhaar dataset/train/pan
mkdir -p dataset/val/aadhaar dataset/val/pan

# Copy your images into respective folders
# dataset/train/aadhaar - Aadhaar training images
# dataset/train/pan     - PAN training images
# dataset/val/aadhaar   - Aadhaar validation images
# dataset/val/pan       - PAN validation images
```

---

## Model Training

### Train the Model

Run the training script to train a ResNet18 model on your dataset:

```bash
python train.py
```

### Training Output

The script will:
1. Load images from `dataset/train/` and `dataset/val/`
2. Apply data augmentation to training images
3. Train ResNet18 for 10 epochs
4. Display training/validation loss and accuracy
5. Save the best model as `model.pth`

### Training Parameters

You can modify these in `train.py`:

```python
BATCH_SIZE = 32           # Batch size for training
LEARNING_RATE = 0.001     # Learning rate for optimizer
NUM_EPOCHS = 10           # Number of training epochs
```

### Example Output

```
============================================================
Starting Image Classification Model Training
============================================================
Loading training data from dataset\train...
Loading validation data from dataset\val...
Found 2 classes: ['aadhaar', 'pan']
Training samples: 80, Validation samples: 20
Loading pre-trained ResNet18 model...
Modified model for 2 classes

============================================================
Beginning Training
============================================================

--- Epoch 1/10 ---
Epoch [1], Batch [1], Loss: 0.6931
Epoch [1], Batch [10], Loss: 0.6842
Training Loss: 0.6854
Validation Loss: 0.6721, Accuracy: 52.50%
Model saved to model.pth (Accuracy: 52.50%)

... (more epochs) ...

============================================================
Training Complete!
============================================================
Total training time: 245.32 seconds
Best validation accuracy: 85.00%
Model saved as: model.pth
============================================================
```

---

## Running the FastAPI Server

### Start the Server

```bash
uvicorn app:app --reload
```

Or with custom host/port:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Server Output

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete
```

---

## API Documentation

### Interactive Swagger UI

Once the server is running, visit:

```
http://localhost:8000/docs
```

All endpoints are documented with examples.

### Endpoints

#### 1. Health Check

**GET** `/`

Check if the API is running and the model is loaded.

**Response:**
```json
{
  "status": "ok",
  "service": "Image Classification API",
  "model_loaded": true,
  "message": "API is running. Use POST /predict to classify images."
}
```

---

#### 2. Image Prediction

**POST** `/predict`

Classify an uploaded image.

**Parameters:**
- `file` (file, required): Image file (jpg, png, gif, bmp, webp)

**Response:**
```json
{
  "predicted_class": "aadhaar",
  "confidence": 0.9542,
  "uploaded_filename": "my_document.jpg",
  "filename_matches_prediction": true
}
```

**Response Fields:**
- `predicted_class`: Predicted class ("aadhaar" or "pan")
- `confidence`: Confidence score (0.0 to 1.0)
- `uploaded_filename`: Name of the uploaded file
- `filename_matches_prediction`: Boolean indicating if the filename text contains the predicted class name

---

### Example API Calls

#### Using cURL

```bash
# Test health check
curl http://localhost:8000/

# Upload image for classification
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@/path/to/image.jpg"
```

#### Using Python Requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Classify image
with open("my_document.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

#### Using JavaScript Fetch

```javascript
// Health check
fetch('http://localhost:8000/')
  .then(response => response.json())
  .then(data => console.log(data));

// Classify image
const formData = new FormData();
formData.append('file', document.getElementById('fileInput').files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => console.log(data));
```

---

## Testing

### Test the API with Swagger UI

1. Start the server: `uvicorn app:app --reload`
2. Open browser: `http://localhost:8000/docs`
3. Click on the **POST /predict** endpoint
4. Click "Try it out"
5. Click "Choose File" and select an image
6. Click "Execute"
7. View the response

---

## Troubleshooting

### Model Not Found Error

**Error:**
```
Error: Model file not found at model.pth
Please run 'python train.py' to train the model first
```

**Solution:**
1. Ensure you have dataset images in `dataset/train/` and `dataset/val/`
2. Run `python train.py` to train and generate `model.pth`
3. Start the server again

### No Dataset Found

**Error:**
```
FileNotFoundError: Dataset directories not found.
```

**Solution:**
1. Create dataset directories: `mkdir -p dataset/train/aadhaar dataset/train/pan dataset/val/aadhaar dataset/val/pan`
2. Add sample images to respective folders
3. Run `python train.py` again

### Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Use a different port
uvicorn app:app --port 8001
```

Or kill the process using port 8000.

### Invalid Image File

**Error:**
```
Invalid file type. Supported: .jpg, .jpeg, .png, .gif, .bmp, .webp
```

**Solution:**
Ensure you're uploading a supported image format.

---

## Performance Notes

- **Model Size**: ~45 MB (ResNet18)
- **Inference Time**: ~100-500ms per image (CPU)
- **Memory Usage**: ~300-500 MB
- **Batch Processing**: Single image per API call (modify for batch support if needed)

---

## Production Deployment

For production use, consider:

1. **Use Gunicorn with multiple workers:**
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
   ```

2. **Docker containerization:**
   ```dockerfile
   FROM python:3.9
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

3. **Nginx reverse proxy:** For load balancing

4. **HTTPS/SSL:** Enable for secure communications

5. **Rate limiting:** Add request throttling

6. **Caching:** Implement response caching for repeated requests

---

## License

This project is provided as-is for educational and commercial use.

---

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review API logs in the console
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Verify dataset structure matches the requirements

---

## File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | FastAPI application with HTTP endpoints |
| `utils.py` | Model loading, preprocessing, and inference utilities |
| `train.py` | Training script using ResNet18 |
| `requirements.txt` | Python package dependencies |
| `model.pth` | Trained model (generated after training) |

---

## Version History

- **v1.0.0** (2024-01-20): Initial release with training and inference capabilities

