# Developer: Mohammad Hassan Heydari

# Importing necessary libraries
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, Input, Concatenate
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.keras.models import Model

def data_preprocessing(path): # path is the path to the dataset
    data = pd.read_csv(path)

    texts, labels = list(data['text']), list(data['class']) # extracting texts and labels from the pandas dataframe


    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>', lower=True) # creating a tokenizer object
    tokenizer.fit_on_texts(texts) # fitting the tokenizer on the texts
    sequences = tokenizer.texts_to_sequences(texts) # converting the texts to sequences

    padded = pad_sequences(sequences, maxlen=100, truncating='post', padding='post') # padding the sequences

    x_train, y_train = np.array(padded), np.array(labels) # converting the padded sequences and labels to numpy arrays



    return x_train , y_train # returning the training data



# load data
x_train, y_train = data_preprocessing('data/chronic_depression.csv')

#Model : using tf.keras functional API to create more flexible architecture

input_layer = Input(shape=(100,))

embedding_layer = Embedding(input_dim=10000, output_dim=16, input_length=100)(input_layer)
lstm1 = Bidirectional(LSTM(units = 128, return_sequences=True, activation='tanh'))(embedding_layer) # LSTM recurrent layer
dropout = Dropout(0.2)(lstm1)
lstm2 = Bidirectional(LSTM(units = 64, return_sequences=False, activation='tanh'))(dropout)
dense1 = Dense(256, activation='relu')(lstm2)
dense2_1 = Dense(32, activation='elu')(dense1)
dense2_2 = Dense(32, activation='tanh')(dense1)
dense2_3 = Dense(32, activation='relu')(dense1)
merge = Concatenate()([dense2_1, dense2_2, dense2_3]) # merging three parallel densley connected layers
dense3 = Dense(64, activation='relu')(merge)

predictions = Dense(1, activation='sigmoid')(dense3)




model = Model(inputs = input_layer, outputs = predictions)
print(model.summary()) # printing the summary of the model which shows the layers and the number of parameters in each layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # compiling the model , details : https://keras.io/api/models/model_training_apis/#compile-method
history = model.fit(x_train, y_train, epochs=10) # training the model , details : https://keras.io/api/models/model_training_apis/#fit-method

model.save('model/DD_rnn_parallel.h5')

