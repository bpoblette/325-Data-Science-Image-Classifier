from fastapi import FastAPI, UploadFile, File
from image_classifier import ImageClassifier
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):  
    image_classifier = ImageClassifier()

    # Read the image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get prediction
    result = image_classifier.predict(image=image)
    return {"filename": file.filename, "predictions": result}
