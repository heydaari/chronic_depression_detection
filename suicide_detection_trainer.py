
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Bidirectional, Embedding
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

def data_preprocessing(path):
    data = pd.read_csv(path)


    texts, labels = list(data['text']), list(data['class'])

    del data


    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>', lower=True)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    padded = pad_sequences(sequences, maxlen=100, truncating='post', padding='post')

    x_train, y_train = np.array(padded), np.array(labels)

    del padded, sequences, labels, tokenizer

    return x_train[:20000], y_train[:20000]



x_train, y_train = data_preprocessing('data/Suicide_Detection.csv')


model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=100),
    Bidirectional(LSTM(units = 256, return_sequences=True, activation='tanh')),
    Dropout(0.2),
    Bidirectional(LSTM(units = 64, return_sequences=False, activation='tanh')),
    Dropout(0.1),
    Flatten(),
    Dense(units = 256, activation='relu'),
    Dropout(0.2),
    Dense(units = 128, activation='relu'),
    Dropout(0.1),
    Dense(units=32, activation='relu'),
    Dropout(0.2),
    Dense(units = 1, activation='sigmoid')
])

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3)

model.save('model/suicide_detection.h5')