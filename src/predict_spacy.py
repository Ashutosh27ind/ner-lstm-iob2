# !pip install pandas numpy scikit-learn keras tensorflow thinc==8.2.3
# Tested on GPU P100 (Kaggle Kernel)

import logging
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.scorer import Scorer
from thinc.api import require_gpu  # 3rd party API

# Use GPU if available
require_gpu()

# Determine the parent directory of the current working directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Create the log directory if it doesn't exist
log_dir = os.path.join(parent_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure the logging settings
logging.basicConfig(
    filename=os.path.join(log_dir, 'app.log'),  # Log file path
    level=logging.INFO,  # Set the desired log level
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Enable Console Logging:
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

df_ner = pd.read_csv('/kaggle/input/ner-data/ner_dataset.csv', encoding="latin1")

# Drop POS since its not needed:
df_ner = df_ner.drop(columns=["POS"])

# Display the first few rows of the dataset
print(df_ner.head())

# Preprocess the data
df_ner = df_ner.ffill()
# df_ner = df_ner.fillna(method='ffill')
words = list(set(df_ner['Word'].values))
n_words = len(words)


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s['Word'].tolist(), s['Tag'].tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]


getter = SentenceGetter(df_ner)
sentences = getter.sentences

# Split the data into train and test
train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)

'''
# Create blank Language class
nlp = spacy.blank('en', disable=['tagger', 'parser'])

# Create the built-in pipeline components and add them to the pipeline
ner = nlp.create_pipe('ner')
nlp.add_pipe('ner')
'''

# Create blank Language class
nlp = spacy.blank('en')

# Disable tagger and parser
# nlp.disable_pipes('tagger', 'parser')

# Create the built-in pipeline components and add them to the pipeline
ner = nlp.create_pipe('ner')
nlp.add_pipe('ner')

TRAIN_DATA = []
for sentence in train_sentences:
    ents = []
    start = 0
    for word, tag in sentence:
        end = start + len(word)
        if tag != 'O':
            ents.append((start, end, tag))
        start = end + 1  # +1 for the space between words
    sentence = " ".join(word for word, tag in sentence)
    TRAIN_DATA.append((sentence, {"entities": ents}))

# Only train NER
optimizer = nlp.begin_training()
for itn in range(10):
    print("Starting iteration " + str(itn))
    random.shuffle(TRAIN_DATA)
    losses = {}
    # batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    batches = minibatch(TRAIN_DATA, size=256)
    for batch in batches:
        texts, annotations = zip(*batch)
        example = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
        nlp.update(example, drop=0.5, sgd=optimizer, losses=losses)
    print("Model losses are:")
    print(losses)


# Function to evaluate the model
def evaluate(ner_model, examples):
    scorer = Scorer(ner_model)
    scores = {}
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        example = Example.from_dict(doc_gold_text, annot)
        scores = scorer.score([example])  # Pass a list of Example objects
    return scores


# Test the trained model
test_data = []
for test_sentence in test_sentences:
    sentence = []
    entities = []
    start = 0
    for word_tag_pair in test_sentence:
        word, tag = word_tag_pair
        sentence.append(word)
        if tag != 'O':
            end = start + len(word)
            entities.append((start, end, tag))
        start += len(word) + 1
    test_data.append((' '.join(sentence), {"entities": entities}))

# Get the evaluation results
results = evaluate(nlp, test_data)
print("Test Results are :")
print(results)