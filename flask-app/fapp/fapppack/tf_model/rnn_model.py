# read training data
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from tensorflow import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import asarray
from numpy import zeros
#import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
import seaborn as sns


class RnnModel:
    train_hp_file = '';

    def __init__(self, train_file):
        self.train_hp_file = train_file

    def main(self):
        # df = pd.Dataframe()
        df_train = pd.read_csv(self.train_hp_file, encoding='utf-8')
        df_train = df_train[['Description', 'Is_Response']]
        # select only rows with happy values and reduce happy values to same amount as not happy values
        df_happy = df_train.loc[df_train['Is_Response'] == 'happy']
        df_not_happy = df_train.loc[df_train['Is_Response'] == 'not happy']
        # only get the first 12411 rows of happy dataframe
        df_happy = df_happy.head(12411)
        frames = [df_happy, df_not_happy]
        df_train = pd.concat(frames)
        # Preprocessing
        X = []
        sentences = list(df_train['Description'])
        for sen in sentences:
            X.append(self.preprocess_text(sen))
            
        # binary classification for happy and not_happy
        y = df_train['Is_Response']
        y = np.array(list(map(lambda x: 1 if x == "happy" else 0, y)))

        self.simple_knn()
        self.cnn()
        self.rnn()
            
    # Data Preprocessing
    def preprocess_text(self, sen):
        # Removing html tags
        sentence = self.remove_tags(self, sen)
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence


    def remove_tags(self, text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

    

    def simple_knn(self):
        # 1) Simple Neural Network
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
        # Prepare embedding layer
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)
    
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
    
        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1
    
        maxlen = 100
    
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
   
        embeddings_dictionary = dict()
        glove_file = open('./glove.twitter.27B/glove.twitter.27B.100d.txt', encoding="utf8")

        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
        glove_file.close()

        embedding_matrix = zeros((vocab_size, 100))
        for word, index in tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        model = Sequential()
        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
        model.add(embedding_layer)

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        # # Model training

        history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

        # # Evaluation of model

        score = model.evaluate(X_test, y_test, verbose=1)


    def cnn(self):
        #2) Convolutional Neural Network
        model = Sequential()

        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
        model.add(embedding_layer)

        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # # Model Training & Evaluation

        history = model.fit(_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
        score = model.evaluate(X_test, y_test, verbose=1)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def rnn(self):
    #3) Recurrent Neural Network: LSTM (Long Short Term Memory network)
        model = Sequential()
        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
        model.add(embedding_layer)
        model.add(LSTM(128))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # # Model Training & Evaluation
        history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
        score = model.evaluate(X_test, y_test, verbose=1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


