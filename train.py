from pyvi import ViTokenizer
import string
import os
import keras
from numpy import array

def clean_document(doc):
    doc = ViTokenizer.tokenize(doc) #Pyvi Vitokenizer library
    doc = doc.lower() #Lower
    tokens = doc.split() #Split in_to words
    table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens

INPUT_LENGTH = 50
sequences = []
for f in os.listdir("Train_Full/Doi song"):
    f1 = open("Train_Full/Doi song/"+f, encoding='utf-16')
    doc = f1.read()
    tokens = clean_document(doc)

    for i in range(INPUT_LENGTH + 1, len(tokens)):
        seq = tokens[i-INPUT_LENGTH-1:i]
        line = ' '.join(seq)
        sequences.append(line)

tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ ')
tokenizer.fit_on_texts(sequences)

sequences_digit = tokenizer.texts_to_sequences(sequences)

# separate into input and output
sequences_digit = array(sequences_digit)
vocab_size = len(tokenizer.word_index) + 1
X, y = sequences_digit[:,:-1], sequences_digit[:,-1]
y = keras.utils.to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

from keras.models import Sequential
from keras.layers import Embedding,BatchNormalization,LSTM,Dense,Dropout
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=50))
model.add(BatchNormalization())
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=512, epochs=100)
import pickle

model.save('51_acc_language_model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('sequences_digit.pkl', 'wb') as f:
    pickle.dump(sequences_digit, f)
