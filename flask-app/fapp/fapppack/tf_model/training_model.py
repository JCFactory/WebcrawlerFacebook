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
from keras.layers import MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split as skl_tt_split
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import asarray
from numpy import zeros
# import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
import seaborn as sns


class RnnModel:
    maxlen = 50
    vocab_size = 0

    train_hp_file = ''
    test_hp_file = ''
    glove_file = ''
    glove_files = []
    model_X = []
    model_Y = []
    model_X_train = []
    model_Y_train = []
    model_X_test = []
    model_Y_test = []
    model_history = None
    model_score = None
    model = None
    embedding_matrix = None
    tokenizer = None

    def __init__(self, train_file, test_file, glove_file, glove_files=None):
        self.train_hp_file = train_file
        self.test_hp_file = test_file
        self.glove_file = glove_file
        self.glove_files = glove_files

    def tokenize(self, text):
        return self.tokenizer.texts_to_sequences(text)

    def predict(self, post):
        post = self.preprocess_text(post)

        el = self.tokenize([post])
        el = pad_sequences(el, padding='post', maxlen=self.maxlen)
        return self.model.predict(el)

    def run(self):
        # np.random.seed(1)
        # tf.compat.v1.set_random_seed (1)
        # df = pd.Dataframe()
        df_train = pd.read_csv(self.train_hp_file, encoding='utf-8')
        df_train = df_train[['Description', 'Is_Response']]
        # select only rows with happy values and reduce happy values to same amount as not happy values
        df_happy = df_train.loc[df_train['Is_Response'] == 'happy']
        df_not_happy = df_train.loc[df_train['Is_Response'] == 'not happy']
        # only get the first 12411 rows of happy dataframe
        df_happy = df_happy.head(12411)
        frames_train = [df_happy[:10000], df_not_happy[:10000]]
        frames_test = [df_happy[11001:], df_not_happy[11001:]]
        df_train = pd.concat(frames_train)
        df_test = pd.concat(frames_test)

        # df_train = df_train[len(df_train['Description'].split()) < 120]
        # df_test = df_test[len(df_test.Description.split()) < 120]
        print(df_train.count())
        # Preprocessing

        sentences = list(df_train['Description'])
        for sen in sentences:
            self.model_X_train.append(self.preprocess_text(sen))
        sentences = list(df_test['Description'])
        for sen in sentences:
            self.model_X_test.append(self.preprocess_text(sen))

        # binary classification for happy and not_happy
        y = df_train['Is_Response']
        print(df_train.loc[df_train['Is_Response'] != 'happy'])
        self.model_Y_train = np.array(list(map(lambda x: 1 if x == "happy" else 0, y)))

        print(self.model_Y_train[self.model_Y_train == 1])
        y = df_test['Is_Response']
        self.model_Y_test = np.array(list(map(lambda x: 1 if x == "happy" else 0, y)))
        self.do_embedding()
        self.build_model()

    # Data Preprocessing
    def preprocess_text(self, sen):
        # Removing html tags
        sentence = self.remove_tags(sen)
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence.lower()

    def remove_tags(self, text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

    def do_embedding(self):
        # , self.model_Y, test_size=0.20, random_state=42)
        # Prepare embedding layer
        self.tokenizer = Tokenizer(num_words=20000)
        self.tokenizer.fit_on_texts(self.model_X_train + self.model_X_test)

        self.model_X_train = self.tokenizer.texts_to_sequences(self.model_X_train)
        self.model_X_test = self.tokenizer.texts_to_sequences(self.model_X_test)

        # Adding 1 because of reserved 0 index
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.model_X_train = pad_sequences(self.model_X_train, padding='post', maxlen=self.maxlen)
        self.model_X_test = pad_sequences(self.model_X_test, padding='post', maxlen=self.maxlen)

        embeddings_dictionary = dict()
        for glove_file in self.glove_files:
            glove_file = open(glove_file, encoding="utf8")

            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimensions = asarray(records[1:], dtype='float32')
                embeddings_dictionary[word] = vector_dimensions
            glove_file.close()

        self.embedding_matrix = zeros((self.vocab_size, self.maxlen))
        for word, index in self.tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index] = embedding_vector

    def build_model(self):
        init = keras.initializers.glorot_uniform(seed=1)

        model = Sequential()
        model.add(Embedding(self.vocab_size, self.maxlen, weights=[self.embedding_matrix], input_length=self.maxlen,
                            trainable=False))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128, activation='relu'))
        # model.add(Dense(16, activation='relu'))
        model.add(Dense(units=1, kernel_initializer=init,
                        activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        print(model.summary())

        bat_size = 32
        max_epochs = 10  # 10
        print("\nStarting training ")
        model.fit(self.model_X_train, self.model_Y_train, epochs=max_epochs,
                  batch_size=bat_size, shuffle=True, verbose=1)
        print("Training complete \n")
        loss_acc = model.evaluate(self.model_X_test, self.model_Y_test, verbose=0)
        print("Test data: loss = %0.6f  accuracy = %0.2f%% " % \
              (loss_acc[0], loss_acc[1] * 100))

        self.model = model
