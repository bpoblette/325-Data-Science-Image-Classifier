FROM python:3.13.3-slim

WORKDIR /app

# Install system dependencies FIRST (important!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Start API server
CMD ["uvicorn", "image_classifier_controller:app", "--host", "0.0.0.0", "--port", "8000"]
