"""
Simple chat to interact with the chatbot through terminal.
"""
import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.utils import count_unique_words

#TODO: Reuse the tokenizer and label encoder from the model training


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


# Importing the trained saved model
model = keras.models.load_model('chatbot_trained_model')

# Tokenizer
tokenizer = Tokenizer(num_words=count_unique_words(train_sentences))
tokenizer.fit_on_texts(train_sentences)

# Label encoder
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)

max_length = 20

# Chat function
while True:

    print("You:")
    text = input()

    if text == 'quit':
        break

    input_data = pad_sequences(tokenizer.texts_to_sequences([text]), truncating='post', maxlen=max_length)

    prediction = model.predict(input_data, verbose=0)


    tag = label_encoder.inverse_transform([np.argmax(prediction)])

    for i in data['intents']:
        if i['tag'] == tag:
            print("Chatbot: " + i['responses'][0])

    print()
