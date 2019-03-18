import pickle
from keras.models import load_model
from pyvi import ViTokenizer
import string
import keras

def clean_document(doc):
    doc = ViTokenizer.tokenize(doc) #Pyvi Vitokenizer library
    doc = doc.lower() #Lower
    tokens = doc.split() #Split in_to words
    table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('model/sequences_digit.pkl', 'rb') as f:
    sequences_digit = pickle.load(f)

model = load_model('model/51_acc_language_model.h5')

import numpy as np


def preprocess_input(doc):
    tokens = clean_document(doc)
    tokens = tokenizer.texts_to_sequences(tokens)
    tokens = keras.preprocessing.sequence.pad_sequences([tokens], maxlen=50, truncating='pre')
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

    return text_input+' '+' '.join(out_word)

print(generate_text("‘‘ Thôi kệ, giờ lo tìm việc kiếm tiền trả tiền nhà đã’’.", 100))