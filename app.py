from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import traceback
import logging
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from typing import Dict, Any

# Import utility functions
from utils.detection import check_and_rotate_input_image, detect_cards_from_image
from utils.classification import classify_cards_from_board_image
from utils.set_finder import find_sets
from utils.drawing import draw_sets_on_image

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Set Game Detector API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a dictionary for models instead of separate global variables
# This improves organization and error handling
models: Dict[str, Any] = {
    "shape_model": None,
    "fill_model": None,
    "shape_detection_model": None,
    "card_detection_model": None
}

@app.on_event("startup")
async def startup_event():
    """
    Load all required models on application startup with optimizations for CPU.
    Uses structured error handling and model warmup for better performance.
    """
    model_paths = {
        "shape_model": './models/Characteristics/11022025/shape_model.keras',
        "fill_model": './models/Characteristics/11022025/fill_model.keras',
        "shape_detection_model": {
            "path": './models/Shape/15052024/best.pt',
            "yaml": './models/Shape/15052024/data.yaml',
            "conf": 0.5
        },
        "card_detection_model": {
            "path": './models/Card/16042024/best.pt',
            "yaml": './models/Card/16042024/data.yaml',
            "conf": 0.5
        }
    }
    
    logger.info("Loading models...")
    
    try:
        # Load TensorFlow/Keras models
        models["shape_model"] = load_model(model_paths["shape_model"])
        models["fill_model"] = load_model(model_paths["fill_model"])
        
        # Optimize Keras models with warmup prediction
        # This triggers model compilation optimized for the current hardware
        dummy_shape = np.zeros((1, *models["shape_model"].input_shape[1:]), dtype=np.float32)
        dummy_fill = np.zeros((1, *models["fill_model"].input_shape[1:]), dtype=np.float32)
        models["shape_model"](dummy_shape)
        models["fill_model"](dummy_fill)
        
        # Load YOLO models with CPU optimizations
        models["shape_detection_model"] = YOLO(model_paths["shape_detection_model"]["path"])
        models["shape_detection_model"].yaml = model_paths["shape_detection_model"]["yaml"]
        models["shape_detection_model"].conf = model_paths["shape_detection_model"]["conf"]
        
        models["card_detection_model"] = YOLO(model_paths["card_detection_model"]["path"])
        models["card_detection_model"].yaml = model_paths["card_detection_model"]["yaml"]
        models["card_detection_model"].conf = model_paths["card_detection_model"]["conf"]
        
        # Warm up YOLO models with a dummy prediction for faster first inference
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        models["shape_detection_model"](dummy_img)
        models["card_detection_model"](dummy_img)
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        # The app will start but health check will fail until models are loaded

@app.get("/")
async def root():
    """Root endpoint that confirms the API is running."""
    return {"message": "Set Game Detector API is running"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint that verifies all models are loaded correctly.
    """
    if any(model is None for model in models.values()):
        missing_models = [name for name, model in models.items() if model is None]
        raise HTTPException(
            status_code=503, 
            detail=f"Models not loaded: {', '.join(missing_models)}"
        )
    return {"status": "healthy"}

@app.post("/detect_sets")
async def detect_sets(file: UploadFile = File(...)):
    """
    Process an uploaded image to detect and annotate valid sets of cards.
    
    Optimized for CPU performance with image size checks, efficient memory
    management, and appropriate quality settings.
    
    Args:
        file (UploadFile): The uploaded image file
        
    Returns:
        Response: JPEG image with annotated sets
    """
    # Validate file is an image
    if not (file.content_type and file.content_type.startswith('image/')):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image with memory efficiency
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        input_image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Free memory immediately after use
        del contents
        del nparr
        
        # Resize large images for faster processing
        max_dim = 1920  # Balance between quality and speed
        h, w = input_image_cv.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            input_image_cv = cv2.resize(input_image_cv, new_size, interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {w}x{h} to {new_size[0]}x{new_size[1]}")
        
        # Check orientation and rotate if needed
        corrected_image, was_rotated = check_and_rotate_input_image(
            input_image_cv, 
            models["card_detection_model"]
        )
        
        # Classify cards with all models passed directly
        card_df = classify_cards_from_board_image(
            corrected_image, 
            models["card_detection_model"], 
            models["shape_detection_model"], 
            models["fill_model"], 
            models["shape_model"]
        )
        
        # Find sets (already CPU-optimized)
        sets_found = find_sets(card_df)
        
        # Draw sets on image
        annotated_image = draw_sets_on_image(corrected_image.copy(), sets_found)
        
        # Restore original orientation if needed
        if was_rotated:
            annotated_image = cv2.rotate(annotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Optimize image encoding for better response time
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]  # 90% quality is good balance
        success, encoded_image = cv2.imencode('.jpg', annotated_image, encode_params)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        return Response(content=encoded_image.tobytes(), media_type="image/jpeg")
    
    except Exception as e:
        error_details = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_details)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    # Use workers=1 to avoid loading models multiple times
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
