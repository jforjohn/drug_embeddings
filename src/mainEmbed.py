from input_output.parser import Parser
from input_output.writer import Writer
from input_output.load_config import load_config_file
from preprocessing.tokenizer import tokenize, tokens2sent, tokenize_embed, labelEncode
from preprocessing.transformations import lemmas, sentClean, removeEmptyRows, removeSpecialCases, padList
from preprocessing.transformations import CRF_get_tag
from keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
tf.reset_default_graph()

import pandas as pd
import numpy as np
import keras
print(keras.__version__)

def preprocess_steps(base_folder, tk=None, tk_class=None):
  df = Parser('../'+ base_folder).call()
  df['sentence'] = df['sentence'].apply(sentClean)#.apply(lambda x: tokens2sent(x, removals=False))
  df['tokens'] = df['sentence'].apply(tokenize)
  df['crf_tags'] = df[['tokens', 'parsed_drugs']].apply(CRF_get_tag, axis=1)

  if tk and tk_class:
    df['tokens_emb'] = tk.texts_to_sequences(df['tokens'])
    df['labels'] = tk_class.texts_to_sequences(df['crf_tags'])
  else:
    tk, df['tokens_emb'] = tokenize_embed(df['tokens'])
    tk_class, df['labels'] = labelEncode(df['crf_tags'])

  n_words = len(tk.word_index)
  n_tags = len(tk_class.word_index)

  df = removeEmptyRows(df)
  df = removeSpecialCases(df)

  df['tokens_emb'] = padList(df['tokens_emb'], 0,config_preprocess)

  df['labels'] = padList(df['labels'], tk_class.word_index['o'], config_preprocess)

  return tk, tk_class, n_words, n_tags, df

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 1000)

config = load_config_file('config', './')
config_data = config['data']
config_preprocess = config['preprocessing']

train_base_folder = config_data.get('train_dir')
test_base_folder = config_data.get('test_dir')

tk, tk_class, n_words, n_tags, train = preprocess_steps(train_base_folder)

_,_,_,_, test = preprocess_steps(test_base_folder)

X_train = train['tokens_emb'].apply(lambda x: pd.Series(x))
X_test = test['tokens_emb'].apply(lambda x: pd.Series(x))
y_train = np.array([to_categorical(i, num_classes=n_tags) for i in train['labels']])

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(train['tokens_emb'].values.shape)

from keras.models import Model, Input, Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Conv1D, Dense, Flatten, MaxPooling1D
from keras_contrib.layers import CRF
'''
model = Sequential()
model.add(Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=config_preprocess['MAX_LEN'], mask_zero=False))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
'''
input = Input(shape=(100,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=config_preprocess['MAX_LEN'], mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1, verbose=1)
print('keys', history.keys())
test_pred = model.predict(test, verbose=1)

loss = history.history['loss']
val_loss = history.history['val_loss']
#Loss plot
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('loss.png')
plt.close()

test_pred = model.predict(test, verbose=1)
print(test_pred)