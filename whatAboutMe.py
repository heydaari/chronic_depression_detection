from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def classifier(sentence) :

    test_sequence = tokenizer.texts_to_sequences([sentence])
    test_padded = pad_sequences(test_sequence, maxlen=100, padding='post', truncating='post')

    predicted = model.predict(test_padded, verbose = 0)[0][0]

    return predicted

with open('data/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)


tokenizer = Tokenizer(10000, oov_token='<OOV>')
tokenizer.word_index = word_index

model = load_model('model/DD_rnn_parallel.h5')

user_input = input("Enter your statement : ")


while user_input != 'exit':

    predicted_class = classifier(user_input)
    if predicted_class >= 0.55 :
        print('This statement is classified as Depression : 1')

    else:
        print('This statement is classified as Non-depression : 0')


    user_input = input("Enter your statement : ")



