import os
import shutil
from icrawler.builtin import GoogleImageCrawler
from ultralytics import YOLO
from PIL import Image
import uuid

# === CONFIG ===
keyword = "zombie apocalypse"
max_images = 10000
confidence_threshold = 0.7
model_path = "runs/detect/train27/weights/best.pt"
dataset_dir = "/home/bpoblette/325-Data-Science-Image-Classifier/ZombieDetection0.1-2"

# === Folder setup ===
temp_image_dir = "autodata_tmp/images"
os.makedirs(temp_image_dir, exist_ok=True)

img_train_dir = os.path.join(dataset_dir, "train/images")
lbl_train_dir = os.path.join(dataset_dir, "train/labels")
os.makedirs(img_train_dir, exist_ok=True)
os.makedirs(lbl_train_dir, exist_ok=True)

# === Step 1: Scrape Google Images ===
print(f"🔍 Downloading images for: '{keyword}'...")
crawler = GoogleImageCrawler(storage={'root_dir': temp_image_dir})
crawler.crawl(keyword=keyword, max_num=max_images)

# === Step 2: Load YOLO model ===
model = YOLO(model_path)
# === Step 3: Predict, filter, and save ===
for filename in os.listdir(temp_image_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(temp_image_dir, filename)
    try:
        results = model.predict(image_path, conf=confidence_threshold, save=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            print(f"🚫 No detections: {filename}")
            continue

        img = Image.open(image_path)
        w, h = img.size

        # Generate unique file name
        base_id = str(uuid.uuid4())[:8]
        img_name = f"{base_id}.jpg"
        label_name = f"{base_id}.txt"

        # Save image to train folder
        dest_img_path = os.path.join(img_train_dir, img_name)
        shutil.copy(image_path, dest_img_path)

        # Save YOLO-formatted label
        valid_label = False
        with open(os.path.join(lbl_train_dir, label_name), "w") as f:
            for box in results.boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                valid_label = True

        # If nothing valid was written, remove the image and label
        if not valid_label:
            os.remove(dest_img_path)
            os.remove(os.path.join(lbl_train_dir, label_name))
        else:
            print(f"✅ Appended: {img_name}")

    except Exception as e:
        print(f"❌ Skipped {filename}: {e}")


# === Cleanup ===
shutil.rmtree("autodata_tmp")
print("🧹 Temporary files cleaned up.")
