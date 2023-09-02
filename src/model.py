# src/model.py

import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def create_model(voc_size, vector_len, sent_maxlen):
    model = Sequential()
    model.add(Embedding(voc_size, vector_len, input_length=sent_maxlen))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    return accuracy, classification_rep

def preprocess_and_pad(corpus, voc_size, sent_maxlen):
    onehot_rep = [one_hot(word, voc_size) for word in corpus]
    padding_data = pad_sequences(onehot_rep, padding='post', maxlen=sent_maxlen)
    return padding_data

def split_train_test(X_final, y_final, test_size=0.33, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
 
