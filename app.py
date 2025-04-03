from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import gc
import traceback
import time
import logging
from contextlib import asynccontextmanager
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Import utility functions from optimized modules
from utils import (
    check_and_rotate_input_image,
    classify_cards_from_board_image,
    find_sets,
    draw_sets_on_image
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application state management
class ModelManager:
    def __init__(self):
        self.shape_model = None
        self.fill_model = None
        self.shape_detection_model = None
        self.card_detection_model = None
        self.is_loaded = False
        
    def load_models(self):
        """Load all required models into memory."""
        if self.is_loaded:
            return
            
        start_time = time.time()
        logger.info("Loading models...")
        
        try:
            # Configure TensorFlow for CPU optimizations (optional, if not set in environment)
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Reduce TF logging
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")  # Enable Intel MKL-DNN
            
            # Load classification models
            self.shape_model = load_model('./models/Characteristics/11022025/shape_model.keras', compile=False)
            self.fill_model = load_model('./models/Characteristics/11022025/fill_model.keras', compile=False)
            
            # Load YOLO detection models - configure for CPU performance
            self.shape_detection_model = YOLO('./models/Shape/15052024/best.pt')
            self.shape_detection_model.yaml = './models/Shape/15052024/data.yaml'
            self.shape_detection_model.conf = 0.5
            self.shape_detection_model.fuse()  # Fuse model layers for better inference speed
            
            self.card_detection_model = YOLO('./models/Card/16042024/best.pt')
            self.card_detection_model.yaml = './models/Card/16042024/data.yaml'
            self.card_detection_model.conf = 0.5
            self.card_detection_model.fuse()  # Fuse model layers for better inference speed
            
            # Run model warmup for faster first inference
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.card_detection_model(dummy_img)
            self.shape_detection_model(dummy_img)
            
            self.is_loaded = True
            elapsed = time.time() - start_time
            logger.info(f"Models loaded successfully in {elapsed:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def get_models(self):
        """Get the loaded models, ensuring they're available."""
        if not self.is_loaded:
            self.load_models()
        return (
            self.shape_model,
            self.fill_model, 
            self.shape_detection_model, 
            self.card_detection_model
        )
        
    def run_gc(self):
        """Explicitly run garbage collection after heavy operations."""
        gc.collect()

# Create model manager
model_manager = ModelManager()

# Define application startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    model_manager.load_models()
    yield
    # Shutdown: nothing specific needed

# Create FastAPI app
app = FastAPI(
    title="Set Game Detector API",
    description="API for detecting and annotating card sets in images",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"message": "Set Game Detector API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify models are loaded properly."""
    try:
        # Check if models are loaded
        if not model_manager.is_loaded:
            model_manager.load_models()
            
        return {
            "status": "healthy",
            "models_loaded": model_manager.is_loaded
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/detect_sets")
async def detect_sets(file: UploadFile = File(...)):
    """
    Process an uploaded image to detect and annotate sets.
    
    The endpoint performs the following steps:
    1. Read and decode the uploaded image
    2. Check for correct orientation and rotate if needed
    3. Detect and classify cards on the image
    4. Find valid sets among the detected cards
    5. Annotate the image with detected sets
    6. Return the annotated image
    
    Args:
        file: The uploaded image file
        
    Returns:
        Annotated JPEG image with detected sets
    """
    # Validate input file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    
    try:
        # Get models
        shape_model, fill_model, shape_detection_model, card_detection_model = model_manager.get_models()
        
        # Read image efficiently with proper error handling
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        input_image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if input_image_cv is None or input_image_cv.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        # Free memory from raw image data
        del contents
        del nparr
        
        # Check orientation and rotate if needed
        corrected_image, was_rotated = check_and_rotate_input_image(input_image_cv, card_detection_model)
        
        # Classify cards
        logger.info("Classifying cards...")
        card_df = classify_cards_from_board_image(
            corrected_image, 
            card_detection_model, 
            shape_detection_model, 
            fill_model, 
            shape_model
        )
        
        # Find sets
        logger.info(f"Found {len(card_df)} cards, finding sets...")
        sets_found = find_sets(card_df)
        logger.info(f"Found {len(sets_found)} sets")
        
        # Draw sets on image
        annotated_image = draw_sets_on_image(corrected_image, sets_found)
        
        # Restore original orientation if needed
        if was_rotated:
            annotated_image = cv2.rotate(annotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Return processed image with optimized encoding
        # Use quality 90 for better compression with minimal quality loss
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        success, encoded_image = cv2.imencode('.jpg', annotated_image, encode_params)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        # Get output bytes
        result = encoded_image.tobytes()
        
        # Clean up to free memory
        del annotated_image
        del encoded_image
        model_manager.run_gc()
        
        elapsed = time.time() - start_time
        logger.info(f"Processed image in {elapsed:.2f} seconds")
        
        return Response(content=result, media_type="image/jpeg")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log detailed error and return simplified message to user
        error_details = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_details)
        model_manager.run_gc()  # Try to recover memory
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment with fallback
    port = int(os.environ.get("PORT", 8080))
    
    # Configure Uvicorn for production
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port,
        workers=2,  # Use 2 workers for better CPU utilization without overload
        log_level="info",
        timeout_keep_alive=120
    )
