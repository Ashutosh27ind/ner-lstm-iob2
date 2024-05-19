#!/bin/bash

# Build the Docker image
docker build -t ner-flask-app:latest .

# Run the Docker container
docker run -p 4000:5000 ner-flask-app

