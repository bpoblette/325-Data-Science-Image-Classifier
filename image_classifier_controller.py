from fastapi import FastAPI, File, UploadFile
from image_classifier import ImageClassifier
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

# to do: Need to take image
@app.get("/predict/")
async def predict(file = UploadFile()):
    image_classifier = ImageClassifier()

    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes).convert("RGB"))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    result = image_classifier.predict(image=image)
    return {"filename": file.filename, "predictions": result}
    
