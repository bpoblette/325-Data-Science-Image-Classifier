#!/bin/bash

echo "Pulling latest changes from Git..."
git pull

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete!"
