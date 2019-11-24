
class RnnModel:
    def test(self):
        return "test"

    # !/usr/bin/env python
    # coding: utf-8

    # Build neural network

    # read training data
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import re
    import nltk
    from nltk.corpus import stopwords

    from numpy import array
    from keras.preprocessing.text import one_hot
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers.core import Activation, Dropout, Dense
    from keras.layers import Flatten
    from keras.layers import GlobalMaxPooling1D
    from keras.layers.embeddings import Embedding
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.text import Tokenizer

    # df = pd.Dataframe()
    df_train = pd.read_csv('./measuring-customer-happiness/train_hp.csv', encoding='utf-8')
    print(df_train.head(3))

    # df_test = pd.read_csv('./measuring-customer-happiness/test_hp.csv', encoding='utf-8')
    # print(df_test.head(3))

    df_train = df_train[['Description', 'Is_Response']]
    print(df_train.head())

    # select only rows with happy values and reduce happy values to same amount as not happy values
    df_happy = df_train.loc[df_train['Is_Response'] == 'happy']
    df_not_happy = df_train.loc[df_train['Is_Response'] == 'not happy']
    print(df_happy)


    # only get the first 12411 rows of happy dataframe
    df_happy = df_happy.head(12411)
    print(df_happy)

    frames = [df_happy, df_not_happy]
    df_train = pd.concat(frames)
    print(df_train)

    import seaborn as sns

    sns.countplot(x='Is_Response', data=df_train)

    # In[15]:

    # Data Preprocessing

    def preprocess_text(sen):
        # Removing html tags
        sentence = remove_tags(sen)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence

    TAG_RE = re.compile(r'<[^>]+>')

    def remove_tags(text):
        return TAG_RE.sub('', text)

    X = []
    sentences = list(df_train['Description'])
    for sen in sentences:
        X.append(preprocess_text(sen))

    # In[16]:

    X[3]

    # In[17]:

    # binary classification for happy and not_happy

    y = df_train['Is_Response']

    y = np.array(list(map(lambda x: 1 if x == "happy" else 0, y)))

    # # 1) Simple Neural Network

    # In[18]:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # In[19]:

    # Prepare embedding layer
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # In[20]:

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # In[21]:

    from numpy import array
    from numpy import asarray
    from numpy import zeros

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

    print(model.summary())

    # # Model training

    history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

    # # Evaluation of model

    score = model.evaluate(X_test, y_test, verbose=1)

    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    # In[28]:

    import matplotlib.pyplot as plt

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

    # # 2) Convolutional Neural Network

    # In[29]:

    from keras.layers.convolutional import Conv1D
    model = Sequential()

    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # In[30]:

    print(model.summary())

    # # Model Training & Evaluation

    # In[31]:

    history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

    score = model.evaluate(X_test, y_test, verbose=1)

    # In[32]:

    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    # In[33]:

    import matplotlib.pyplot as plt

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

    # # 3) Recurrent Neural Network: LSTM (Long Short Term Memory network)

    # In[34]:

    from keras.layers.recurrent import LSTM
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(128))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # In[35]:

    print(model.summary())

    # # Model Training & Evaluation

    # In[36]:

    history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

    score = model.evaluate(X_test, y_test, verbose=1)

    # In[37]:

    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    # In[38]:

    import matplotlib.pyplot as plt

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

    # # Load facebook comments

    import json
    import requests

    # api-endpoint
    burghard_api_url = "https://graph.facebook.com/v4.0/me?fields=id%2Cname%2Cposts%7Bcomments%2Clikes%2Creactions%7Bname%2Ctype%2Cid%7D%7D%2Cfeed&access_token=EAAlcIv35CUUBACAL7BAOaT1Fm4mn84wsn0yqOr2UU9R5FG59beXoymQAdA47G4jQqtcW9iZBhCrwlW0VxOQNEUTNgWK4ZBM6vjTA9BthBw7jwM55RKKVgYRt0V4tFcQZAthLHPMZBAZBZBnoHRpD3PXeZBZBilgnaxHJuHoMg7E4dpYdhZAbioeaA33hviyR6DZAj6ZBWh3896phZAo8ft9yV26Y"

    lachmann_cruises_api_url = "https://graph.facebook.com/v5.0/me?fields=id%2Cname%2Cposts%7Bcomments%2Csharedposts%7D&access_token=EAAlcIv35CUUBAKDaXZA3zrldcprNX4RosQpbZBAVn5WvMmzp6fwh3Q17zB6qiHMf3OZCiLdx3GSrhGma0fDkYwPGQzj6zYqQw3pdpv5ZBAUoWZAsO2OiCUKHlBvMNOp2lDx2On9q0jWZC7pC2XynX6OigHn9TswWQfpfaxGS2tIRYp1Dm2JYEHTKAlbf5juQIkoXmnKqk1wsIDFUqYebjj"

    new_url = "https://graph.facebook.com/v5.0/me?fields=id%2Cname%2Clikes%2Cposts&access_token=EAAlcIv35CUUBAEXpcPy1a1OtHIrEw3qD79EZBZCk0oAPBYO8F7tvg5xKvV3XcbevYdgwvjE9vTXVAc4cW014I5FkGlpIQgjAZAi8c0C0cIJSNqYzqJfP78Bq1Cdc6u8mITjpdRZCx73dsrjYyfQsLiKRZCdAShWSpHrFHj7CXbvim8hYSjFbwWreIt2mOUZAmATaXqTHNUhKdXsO9cNHxS"

    response = requests.get(new_url)
    json_data = json.loads(response.text)
    posts = json_data['posts']

    for i in posts['data']:
        message = i['message']
        print(message)

    # # Load facebook comments

    # # Predict sentiments

    # In[39]:

    instance = X[57]
    print(instance)

    # ####  - convert review into numeric form (using the tokenizer)
    # ####  - text_to_sequences method will convert the sentence into its numeric counter part
    # ####  - positive = 1, negative = 0
    # ####  - sigmoid function predicts floating value between 0 and 1.
    # ####  - value < 0.5 = negative sentiment
    # ####  - value > 0.5 = positive sentiment
    #


    instance = tokenizer.texts_to_sequences(instance)

    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]

    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

    model.predict(instance)
