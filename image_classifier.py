from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

class ImageClassifier:
    def __init__(self):
        self.rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        self.project = self.rf.workspace("zombieimageclassification").project("zombiedetection0.1-lcsaw")
        self.version = self.project.version(2)
        self.dataset = self.version.download("yolov8")
        self.model = YOLO("yolov8n.pt")


    def train(self, training_set, epochs=20, imgsz=640):
        self.model.train(
            data=training_set, 
            epochs=epochs,
            imgsz=imgsz,
            lr0=0.0005,            # Base learning rate
            lrf=0.05,              # Final LR fraction (cosine schedule)
            cos_lr=True,          # Cosine annealing
            weight_decay=0.001,  # Helps with regularization
            batch=8,              # -1 allows YOLO to auto-adjust, or 8 due to no GPU support

            # Augmentations
            mosaic=0.1,
            mixup=0.0,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.2,
            flipud=0.1,
            fliplr=0.2,
            scale=0.3,
            translate=0.05,
            shear=1,
            perspective=0.00005,


            # Detection tweaks
            conf=0.25, 
            iou=0.45,
        )



    def test(self, test_images_folder):
        # Run detection on test images folder
        results = self.model.predict(source=test_images_folder, conf=0.5, save=True, save_txt=True)

        # Loop through results and display images
        for result in results:
            img = result.plot()  # Draw bounding boxes
            cv2.imshow("Detection", img)
            cv2.waitKey(0)  # Wait for key press
            cv2.destroyAllWindows()
    
    def predict(self, image: np.ndarray):
        result = self.model.predict(source=image,conf=0.6, save=True, save_txt=True)
        predictions = []

        for r in result:
            for box in r.boxes:
                predictions.append({
                    "class": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]), 
                    "bbox": box.xyxy[0].tolist()
                })
        return predictions



def main():
    classifier = ImageClassifier()

    # Path to dataset YAML
    dataset_yaml = "/home/bpoblette/325-Data-Science-Image-Classifier/ZombieDetection0.1-2/data.yaml"
    # Train model
    training_start_time = time.time()
    classifier.train(training_set=dataset_yaml, epochs=50, imgsz=800)
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    # Testing the model
    test_images_folder = "/home/bpoblette/325-Data-Science-Image-Classifier/ZombieDetection0.1-2/test/images"
    classifier.test(test_images_folder)
    print(f"The Total testing time was: {total_training_time} seconds")

if __name__ == "__main__":
    main()

# To do: Create a service which will take pictures using a computers camera. 

# Graham advice: for Hyperparameters in the learning, mostly just play with learning rate and batch size

# for confidence of 0.5 and above try to improve the iou of .5/95 