# Developer: Mohammad Hassan Heydari

# Importing necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Bidirectional, Embedding
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

def data_preprocessing(path): # path is the path to the dataset
    data = pd.read_csv(path)


    texts, labels = list(data['text']), list(data['class']) # extracting texts and labels from the pandas dataframe

    del data # deleting the data variable to free up some memory


    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>', lower=True) # creating a tokenizer object
    tokenizer.fit_on_texts(texts) # fitting the tokenizer on the texts
    sequences = tokenizer.texts_to_sequences(texts) # converting the texts to sequences

    padded = pad_sequences(sequences, maxlen=100, truncating='post', padding='post') # padding the sequences

    x_train, y_train = np.array(padded), np.array(labels) # converting the padded sequences and labels to numpy arrays

    del padded, sequences, labels, tokenizer # deleting the variables to free up some memory

    return x_train[:20000], y_train[:20000] # returning the training data



# load data
x_train, y_train = data_preprocessing('data/chronic_depression.csv')


model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=100), # embedding layer to convert the sequences to vectors that can be fed to the LSTM layer
    Bidirectional(LSTM(units = 256, return_sequences=True, activation='tanh')), # Bidirectional LSTM layer which is a type of LSTM layer that takes into account the past and the future
    Dropout(0.2), # dropout layer to prevent overfitting
    Bidirectional(LSTM(units = 64, return_sequences=False, activation='tanh')), # Bidirectional LSTM layer which is a type of LSTM layer that takes into account the past and the future
    Dropout(0.1), # dropout layer to prevent overfitting
    Flatten(), # flattening the output of the LSTM layer
    Dense(units = 256, activation='relu'), # dense layer with 256 neurons
    Dropout(0.2), # dropout layer to prevent overfitting
    Dense(units = 128, activation='relu'), # dense layer with 128 neurons
    Dropout(0.1), # dropout layer to prevent overfitting
    Dense(units=32, activation='relu'), # dense layer with 32 neurons
    Dropout(0.2), # dropout layer to prevent overfitting
    Dense(units = 1, activation='sigmoid') # dense layer with 1 neuron and sigmoid activation function to get the output
])

print(model.summary()) # printing the summary of the model which shows the layers and the number of parameters in each layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # compiling the model , details : https://keras.io/api/models/model_training_apis/#compile-method
history = model.fit(x_train, y_train, epochs=3) # training the model , details : https://keras.io/api/models/model_training_apis/#fit-method

model.save('model/suicide_detection.h5')

