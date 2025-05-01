# 325-Data-Science-Image-Classifier

How to install requirement.txt:
  pip install -r requirements.txt

How to push download updates:
  pip freeze > requirements.txt

# Running the Project
I setup the project to run with a docker-compose file
In the terminal:
To build the images for docker, copy and paste this command
* docker-compose build
This will create the images neccessary for docker as well as download all dependencies and node modules
Once that command is done:
* docker-compose start

This will start the app locally. Python will start the fastapi and the vue development server.

# Routes:
## Predict Route:
* http://127.0.0.1:8000/predict

# Python Environement:
## To start Python environment
* myenv\Scripts\activate

# Hosting:
## Frontend: Vercel
https://325-data-science-image-classifier.vercel.app/

## Backend: Render
https://three25-data-science-image-classifier.onrender.com
