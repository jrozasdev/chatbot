"""
Simple chat to interact with the chatbot through terminal.
"""

import numpy as np
import pickle
import json

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importing the trained saved model
model = keras.models.load_model('chatbot_trained_model')

# Load the tokenizer from disk
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder from disk
with open('encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

with open("intents.json") as file:
    data = json.load(file)

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
