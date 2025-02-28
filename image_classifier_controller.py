from fastapi import FastAPI
from image_classifier import ImageClassifier

app = FastAPI()

@app.get("/predict/{image}")
async def root():
    image_classifier = ImageClassifier()
    image_classifier.predict(image={image})
    
