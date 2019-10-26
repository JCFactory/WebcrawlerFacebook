# Build neural network

# read training data

import tensorflow as tf
import pandas as pd
import numpy as np
df = pd.read_csv('../measuring-customer-happiness/train_hp.csv', encoding='utf-8')
df.head(3)

import tensorflow as tf

from tf.python.keras.preprocessing.text import Tokenizer
from tf.python.keras.preprocessing.sequencing import pad_sequences

tokenizer_obj = Tokenizer()
total_reviews = X_train + X_test
tokenizer_obj.fit_on_texts(total_reviews)

# pad sequences
max_length = max([len(s.split()) for s in total_reviews])

# define vocabulary size
vocab_size = len(tokenizer_obj.word_index) + 1

X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')
