# Used GPU T4 (Google Colab Kernel)

# Import necessary libraries
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from flask import Flask, request, jsonify
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
import spacy
from spacy import displacy

# Determine the parent directory of the current working directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Configure logging
log_dir = os.path.join(parent_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'app.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Enable Console Logging:
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available!!")
    device = '/gpu:0'
else:
    print("GPU not available, using CPU instead!")
    device = '/cpu:0'

# Create a data directory if it doesn't exist
data_directory = os.path.join(parent_dir, 'data/')
os.makedirs(data_directory, exist_ok=True)

# Load the dataset from the specified directory
dataset_path = os.path.join(data_directory, 'ner_dataset.csv')
if os.path.exists(dataset_path):
    df_ner = pd.read_csv(dataset_path, encoding="latin1")
else:
    logging.error(f"Dataset not found at: {dataset_path}")
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

# Drop unnecessary columns
df_ner = df_ner.drop(columns=["POS"])

# Function to preprocess the data into sentences and corresponding labels
def preprocess_data(data):
    """
    Preprocesses the NER dataset into sentences and corresponding labels.

    Args:
    data: pandas DataFrame containing the NER dataset

    Returns:
    sentences: list of lists containing words for each sentence
    labels: list of lists containing NER labels for each sentence
    """
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    for index, row in data.iterrows():
        if pd.isnull(row['Sentence #']):
            if current_sentence:  # Check if the sentence is not empty
                sentences.append(current_sentence)
                labels.append(current_labels)
            current_sentence = []
            current_labels = []
        else:
            current_sentence.append(row['Word'])
            current_labels.append(row['Tag'])

    return sentences, labels

# Preprocess the data
sentences, labels = preprocess_data(df_ner)

# Split the data into train and test sets (80% train, 20% test)
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Further split the train set into train and validation sets (80% train, 20% validation)
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_sentences, train_labels, test_size=0.2, random_state=42)

# Tokenize words and labels using only the training data to avoid data leakage
words = list(set([word for sentence in train_sentences for word in sentence]))
n_words = len(words)

tags = list(set(df_ner["Tag"].values))
n_tags = len(tags)

# Create mappings for words and tags
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

# Padding sequences
max_len = 50
X_train = [[word2idx[w] for w in s] for s in train_sentences]
X_train = pad_sequences(maxlen=max_len, sequences=X_train, padding="post", value=n_words-1)

y_train = [[tag2idx[t] for t in l] for l in train_labels]
y_train = pad_sequences(maxlen=max_len, sequences=y_train, padding="post", value=tag2idx["O"])
y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]

# Function to train the model
def train_model(X_train, y_train, n_words, n_tags, device):
    """
    Trains the LSTM model.

    Args:
    X_train: Training data
    y_train: Labels for the training data
    n_words: Total number of unique words
    n_tags: Total number of unique tags
    device: Device to run the model on ('/cpu:0' or '/gpu:0')

    Returns:
    model: Trained model
    """
    with tf.device(device):

        # Define the model architecture
        model = Sequential()
        model.add(Embedding(input_dim=n_words, output_dim=50, input_length=max_len))
        model.add(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))
        model.add(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))
        model.add(Dropout(0.5))
        model.add(Dense(n_tags, activation="softmax"))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        
        # Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        
        # Define model checkpointing
        checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        batch_size = 256
        # Train the model
        model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=10, validation_split=0.25, verbose=1, callbacks=[early_stopping, checkpoint])

    return model

# Train the model
model = train_model(X_train, y_train, n_words, n_tags, device)

# Save the model:
model.save('models/model.keras')

# Print model summary
print(model.summary())
plot_model(model)

# Prepare the test data
X_test = [[word2idx.get(w, n_words-1) for w in s] for s in test_sentences]
X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=n_words-1)

y_test = [[tag2idx.get(t, tag2idx["O"]) for t in l] for l in test_labels]
y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx["O"])
y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]

# Predict on the test data
y_pred = model.predict(X_test)

# Convert the index to tag
idx2tag = {i: w for w, i in tag2idx.items()}

# Function to convert predictions to labels
def pred2label(pred):
    """
    Converts predictions to labels.

    Args:
    pred: Predictions from the model

    Returns:
    out: List of predicted labels
    """
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

# Convert predictions to labels
pred_labels = pred2label(y_pred)
test_labels = pred2label(y_test)

# Flatten the lists of labels and predictions
flat_test_labels = [label for sublist in test_labels for label in sublist]
flat_pred_labels = [label for sublist in pred_labels for label in sublist]

# Calculate the F1 score
f1 = f1_score(flat_test_labels, flat_pred_labels, average='weighted')

print(f'Weighted F1 Score: {f1}')

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, normalize=False, cmap='Blues'):
    """
    Plots a confusion matrix using seaborn.

    Args:
    y_true: True labels
    y_pred: Predicted labels
    labels: List of unique labels
    normalize: Whether to normalize the confusion matrix
    cmap: Color map for the heatmap
    """
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, 
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error occurred while plotting confusion matrix: {str(e)}")
        traceback.print_exc()

# Plot the confusion matrix
unique_tags = list(tag2idx.keys())
plot_confusion_matrix(flat_test_labels, flat_pred_labels, unique_tags, normalize=True)

# Create a Flask App:
app = Flask(__name__)
model = load_model('models/model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data['sentence'])])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

