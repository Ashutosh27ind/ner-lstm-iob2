# Use the official TensorFlow Docker image
FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /app/src

# Copy the project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download Spacy model
RUN python -m spacy download en_core_web_sm

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

