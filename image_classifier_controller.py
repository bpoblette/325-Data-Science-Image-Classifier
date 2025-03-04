from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from image_classifier import ImageClassifier
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS from all origins (you can replace "*" with specific origins if necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):  
    image_classifier = ImageClassifier()

    # Read the image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get predictions
    predictions = image_classifier.predict(image=image)
    
    # Draw bounding boxes on the image
    for pred in predictions:
        bbox = pred['bbox']
        cv2.rectangle(image, 
                      (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[2]), int(bbox[3])), 
                      (255, 0, 0), 2)
        cv2.putText(image, 
                    f"{pred['class']} ({pred['confidence']:.2f})", 
                    (int(bbox[0]), int(bbox[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (255, 0, 0), 2)

    # Convert the image to a base64 string
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return {
        "filename": file.filename,
        "predictions": predictions,
        "image": f"data:image/jpeg;base64,{base64_image}"
    }

