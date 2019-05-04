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

from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from models.dl import architecture
from keras.models import load_model
from keras.utils import plot_model

from structs import DrugEntity

import tensorflow as tf
#tf.reset_default_graph()

from time import time
import pandas as pd
import numpy as np
from os import path
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
config_arch = config['arch']
config_training = config['training']

output_dir = config_data['output_dir']
max_len = config_preprocess['MAX_LEN']
emb_dim = config_preprocess['EMB_DIM']

train_base_folder = config_data.get('train_dir')
test_base_folder = config_data.get('test_dir')

tk, tk_class, n_words, n_tags, train = preprocess_steps(train_base_folder)

_,_,_,_, test = preprocess_steps(test_base_folder)

X_train = train['tokens_emb'].apply(lambda x: pd.Series(x))
X_test = test['tokens_emb'].apply(lambda x: pd.Series(x))
y_train = np.array([to_categorical(i, num_classes=n_tags) for i in train['labels']])
y_test = np.array([to_categorical(i, num_classes=n_tags) for i in test['labels']])

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(train['tokens_emb'].shape)

crf, model = architecture(config_arch, n_words, n_tags, max_len, emb_dim)
plot_model(model, to_file=path(output_dir,'model.png'), show_shapes=True)


# Training
optimizer = config_training['optimizer']

if optimizer == 'rmsprop':
    if 'lrate' in config_training:
        optimizer = RMSprop(lr=config_training['lrate'],
                    decay=config_training['lrate']//config_training['epochs'])
    else:
        optimizer = RMSprop(lr=0.001)
else:
    optimizer = Adam(lr=config_training['lrate'],
                    decay=config_training['lrate']//config_training['epochs'])

cbacks = []
tensorboard = TensorBoard(log_dir=output_dir+"/{}".format(time()))
cbacks.append(tensorboard)

modfile = path(output_dir, 'model.h5')
mcheck = ModelCheckpoint(filepath=modfile,
                         monitor='val_loss',
                         verbose=0,
                         save_best_only=True,
                         save_weights_only=False,
                         mode='auto',
                         period=1)
cbacks.append(mcheck)

model.compile(optimizer=optimizer, loss=crf.loss_function, metrics=[crf.accuracy])


start = time()
history = model.fit(X_train, y_train, 
            batch_size=config_training['BATCH_SIZE'],
            epochs=config_training['EPOCHS'],
            validation_data=(X_test, y_test),
            callbacks=cbacks,
            shuffle=False,
            verbose=1)
train_duration = time() - start

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
plt.savefig(path.join(output_dir,'loss.png'))
plt.close()

acc = history.history['crf_viterbi_accuracy']
val_acc = history.history['val_crf_viterbi_accuracy']
#Acc plot
plt.plot(acc)
plt.plot(val_acc)
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(path.join(output_dir,'acc.png'))
plt.close()

model = load_model(modfile)

# Evaluation
score_trn = model.evaluate(X_train, y_train, batch_size=32, verbose=0)
score_tst = model.evaluate(X_test, y_test, batch_size=32, verbose=0)

print('score', score_trn, score_tst)

test_pred = model.predict(X_test, verbose=1)

idx2tag = dict(map(reversed, tk_class.word_index.items()))

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out
    
test['pred_labels'] = pred2label(test_pred)
def keepOnlyTags(row):
    labels = row[0]
    crf_tags = row[1]
    return labels[:len(crf_tags)]
test['preds'] = test[['pred_labels','crf_tags']].apply(keepOnlyTags, axis=1)

drugs = []
for tokens, crf_tags in zip(test['tokens'], test['preds']):
    current_drugs = []
    current_token = None
    for token, crf_tag in zip(tokens, crf_tags):
        if crf_tag == 'o':
            if current_token is not None:
                current_drugs.append(current_token)
                current_token = None
        else:
            if current_token == None:
                current_token = DrugEntity(
                    offsets=token['char_offset'],
                    de_type=crf_tag.split('-')[-1],
                    text=token['text']
                )
            else:
                current_token.offsets = [current_token.offsets[0], token['char_offset'][1]]
                current_token.text = current_token.text + ' ' + token['text']

    drugs.append(current_drugs)

test['drugs'] = drugs
out_file = path(output_dir,'task9.2_CRF1_1.txt')
tmp = Writer(out_file).call(test, col_names=['drugs'])

from os import system
bank_type = 'NER'
bank_name = 'DrugBank'
test_dir = f'../resources/Test-{bank_type}/{bank_name}/'
results = system(f'java -jar ../bin/evaluateNER.jar {test_dir} {out_file}')
#!rm {out_folder}*.log *.txt
print('\n'.join(results[-5:-2]))
