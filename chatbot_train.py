"""
Training the chatbot.
"""


import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from utils.utils import count_unique_words


"""
Loading the intents file to train the chatbot into training sentences and labels
"""

with open('intents.json') as file:
    data = json.load(file)

train_sentences = []
train_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        train_sentences.append(pattern)
        train_labels.append(intent['tag'])
    responses.append(intent['responses'])
    labels.append(intent['tag'])


# Use label encoder to encode labels for the model
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)
train_labels = label_encoder.transform(train_labels)

# Tokenize training data for the model
num_unique_words = count_unique_words(train_sentences)
embedding_dim = 20
max_length = 20

tokenizer = Tokenizer(num_words = num_unique_words)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_length)
num_classes = len(labels)

"""
Creating the model.
Embedding layer and 3 dense layers with 20 neurons, 10 neurons and number of labels.
Using GlobalAveragePooling1D layer for its usefulness capturing the meaning of a sentence, and since
this will be a kind of classification.
"""
model = Sequential()
model.add(Embedding(num_unique_words, embedding_dim, input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

# Number of epochs set to 500, diminishing returns with 600
epochs = 500
history = model.fit(padded_sequences, np.array(train_labels), epochs=epochs)

# Saving trained model
model.save("chatbot_trained_model")