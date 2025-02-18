import csv
import os

# Directory where the result CSV files are located
directory = 'runs/detect'

# This will store the best mAP50(B) score and the corresponding file path
best_score = -1
best_file = ""

# Function to read CSV and extract mAP50(B)
def get_map50(file_path):
    try:
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'metrics/mAP50(B)' in row:  # Check if the column exists
                    return float(row['metrics/mAP50(B)'])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# Loop through the directories and files to find the highest mAP50(B)
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('results.csv'):
            file_path = os.path.join(subdir, file)
            score = get_map50(file_path)
            if score is not None and score > best_score:
                best_score = score
                best_file = file_path

# Output the result
if best_file:
    print(f"Best model found in: {best_file}")
    print(f"Highest mAP50(B) score: {best_score}")
else:
    print("No valid mAP50(B) scores found.")
