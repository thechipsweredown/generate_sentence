import pickle
from keras.models import load_model
import keras
import train
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('model/sequences_digit', 'rb') as f:
    sequences_digit = pickle.load(f)

model = load_model('model/51_acc_language_model.h5')

import numpy as np


def preprocess_input(doc):
    tokens = train.clean_document(doc)
    tokens = tokenizer.texts_to_sequences(tokens)
    tokens = keras.preprocessing.sequence([tokens], maxlen=50, truncating='pre')
    return np.reshape(tokens, (1,50))


def generate_text(text_input, n_words):
    tokens = preprocess_input(text_input)
    for _ in range(n_words):
        next_digit = model.predict_classes(tokens)
        tokens = np.append(tokens, next_digit)
        tokens = np.delete(tokens, 0)
        tokens = np.reshape(tokens, (1, 50))

    # Mapping to text
    tokens = np.reshape(tokens, (50))
    out_word = []
    for token in tokens:
        for word, index in tokenizer.word_index.items():
            if index == token:
                out_word.append(word)
                break

    return ' '.join(out_word)

print(generate_text("đường phố ở việt nam", 100))