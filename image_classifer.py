from roboflow import Roboflow
from ultralytics import YOLO
import cv2

class ImageClassifier:
    def __init__(self):
        self.rf = Roboflow(api_key="ujauwGnQ8zCBEuqgI3O7")
        self.project = self.rf.workspace("zombieimageclassification").project("zombiedetection0.1-lcsaw")
        self.version = self.project.version(1)
        self.dataset = self.version.download("yolov8")
        self.model = YOLO("runs/detect/train18/weights/best.pt")


    def train(self, training_set, epochs=10, imgsz=640):
        self.model.train(data=training_set, epochs=epochs, imgsz=imgsz)

    def test(self, test_images_folder):
        # Run detection on test images folder
        results = self.model.predict(source=test_images_folder, conf=0.1, save=True, save_txt=True)

        # Loop through results and display images
        for result in results:
            img = result.plot()  # Draw bounding boxes
            cv2.imshow("Detection", img)
            cv2.waitKey(0)  # Wait for key press
            cv2.destroyAllWindows()

def main():
    classifier = ImageClassifier()

    # Path to dataset YAML
    dataset_yaml = "/home/bpoblette/325-Data-Science-Image-Classifier/ZombieDetection0.1-1/data.yaml"

    # Train model
    classifier.train(training_set=dataset_yaml, epochs=10, imgsz=640)

    # Testing the model
    test_images_folder = "/home/bpoblette/325-Data-Science-Image-Classifier/ZombieDetection0.1-1/test/images"
    classifier.test(test_images_folder)

if __name__ == "__main__":
    main()

# Todo: Create a service which will process video files into frames that can be processed by the yolo model.
# Use bittorrent for getting the video files for season 1 of the walking dead
