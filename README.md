# NER with LSTM using Keras and TensorFlow

**Named Entity Recognition (NER)** is a task in natural language processing (NLP) that involves identifying named entities in text and classifying them into predefined categories such as names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. NER is widely used in various NLP applications such as information extraction, question answering, text summarization, and more.  
  
The **IOB2 (Inside, Outside, Beginning)** tagging scheme is commonly used for NER tasks. In this scheme, each token in a sentence is labeled with one of three tags: B (Beginning), I (Inside), or O (Outside).  
  
**B-<entity_type>**: Represents the beginning of an entity of type <entity_type>.  
**I-<entity_type>**: Represents a token inside an entity of type <entity_type>.  
**O**: Represents a token outside any entity.      
For example, consider the sentence: "New York is a city in the United States." In IOB2 format, it would be annotated as:    
    
```mathematics
New    B-LOC
York   I-LOC
is     O
a      O
city   O
in     O
the    O
United B-LOC
States I-LOC
.      O
```  
  
Here, "New York" and "United States" are entities of type Location (LOC).
The dataset used in this project, ner_dataset.csv, follows the IOB2 tagging scheme and contains labeled sentences for training and testing the NER model. The dataset includes words and their corresponding part-of-speech (POS) tags and NER labels.  
   
This project implements a Named Entity Recognition (NER) model using LSTM with Keras and TensorFlow, leveraging a GPU for training. It also includes a Flask web application for serving predictions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Project](#running-the-project)
- [API Endpoints](#api-endpoints)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Logging](#logging)
- [Docker](#docker)

## Prerequisites

- Python 3.9+
- TensorFlow 2.x
- Keras
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Spacy


## Project Structure
```
.
├── data
│   └── ner_dataset.csv
├── docs
│   └── model.png
├── logs
│   └── app.log
├── models
│   ├── ner_lstm.h5
│   └── model.keras
├── notebooks
│   └── ner-using-lstm-with-iob2.ipynb
├── src
│   └── predict_lstm.py
└── README.md

```


## Setup and Installation

1. **Clone the repository:**
   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```
   
2. **Create a virtual environment:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
    
3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

1. **Train the model:**
Make sure the dataset is in the data directory. Run the training script to train and save the model:
    ```sh
    cd src
    python predict_lstm.py
    ```

2. **Run the Flask application:**
   ```sh
   flask run --host=0.0.0.0 --port=5000
   ```
   
## API Endpoints

**POST /predict**
   ```json
   {
  "sentence": ["Sample", "sentence", "for", "NER", "prediction"]
   }
   ```

**Response:**
   ```json
   {
  "prediction": [[0.1, 0.7, 0.2, 0.2], 0.1]
   }
   ```

## Training the Model 
The model is trained using LSTM layers to perform NER. The script app.py handles data preprocessing, model training, and evaluation. Modify the parameters as needed before training.  

## Evaluation 
After training, the model is evaluated using the test set. The F1 score is calculated, and a confusion matrix is plotted for further analysis.  

## Logging
Logs are saved to the logs directory, with detailed information about the training process, errors, and other runtime information.  
  
## Docker   
To containerize the application, create a Dockerfile in the project root      
   
Build and run the Docker container:   
      
1. Build the Docker image:    
   ```sh
   docker build -t ner-flask-app .
   ```
   
2. Run the Docker container:    
   ```sh
   docker run -p 4000:5000 ner-flask-app
   ```
  
## License   
This project is licensed under the MIT License. See the LICENSE file for details.  
