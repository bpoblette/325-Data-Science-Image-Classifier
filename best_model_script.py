import csv
import os

# Directory where the result CSV files are located
directory = 'runs/detect'

# Dictionary to store the best scores and file paths
best_metrics = {
    'metrics/mAP50(B)': {'score': -1, 'file': ''},
    'metrics/mAP50-95(B)': {'score': -1, 'file': ''},
    'metrics/precision(B)': {'score': -1, 'file': ''},
    'metrics/recall(B)': {'score': -1, 'file': ''},
}

# Function to read CSV and extract metrics
def get_metrics(file_path):
    try:
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                return {
                    metric: float(row[metric]) for metric in best_metrics if metric in row
                }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return {}

# Loop through the directories and files to find the highest values
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('results.csv'):
            file_path = os.path.join(subdir, file)
            scores = get_metrics(file_path)
            for metric, value in scores.items():
                if value > best_metrics[metric]['score']:
                    best_metrics[metric]['score'] = value
                    best_metrics[metric]['file'] = file_path

# Output the results
for metric, data in best_metrics.items():
    if data['file']:
        print(f"Best {metric} found in: {data['file']}")
        print(f"Highest {metric} score: {data['score']}\n")
    else:
        print(f"No valid {metric} scores found.\n")
