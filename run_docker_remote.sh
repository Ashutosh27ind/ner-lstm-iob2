#!/bin/bash

# Define the ECR repository URL
ECR_REPO_URL="your-ecr-repo-url"

# Build the Docker image
docker build -t ner-flask-app:latest .

# Tag the Docker image
docker tag ner-flask-app:latest $ECR_REPO_URL:latest

# run aws ecr get-login-password --region region | docker login --username AWS --password-stdin your-ecr-repo-url

# Push the Docker image to the ECR repository
docker push $ECR_REPO_URL:latest

# Run the Docker container
docker run -p 4000:80 ner-flask-app
