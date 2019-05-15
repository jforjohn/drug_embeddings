#from os import chdir; chdir('/content/gdrive/My Drive/drug_embeddings/src')
import sys; sys.path.append('../src')
from input_output.parser import Parser
from input_output.writer import Writer
from preprocessing.tokenizer import tokenize
from preprocessing.transformations import removeEmptyRows
from preprocessing.transformations import CRF_get_tag
from structs import DrugEntity
from models.dl import architecture
from models.dl import Metrics
from models.dl import embedding_weights

from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_contrib import metrics, losses
from keras.models import load_model
from keras.utils import plot_model
from seqeval.metrics import f1_score, classification_report
import tensorflow as tf
#tf.reset_default_graph()

from time import time
import pandas as pd
import numpy as np
from os import path
from os import mkdir
import keras
print(keras.__version__)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 1000)

def pred2label(pred, idx2tag):
  out = []
  for pred_i in pred:
      out_i = []
      for p in pred_i:
          p_i = np.argmax(p)
          out_i.append(idx2tag[p_i].replace("PAD", "O"))
      out.append(out_i)
  return out

def preprocess_steps(base_folder):
  df = Parser('../'+ base_folder).call()
  #df['sentence'] = df['sentence'].apply(sentClean)#.apply(lambda x: tokens2sent(x, removals=False))
  df['tokens'] = df['sentence'].apply(tokenize)
  df['crf_tags'] = df[['tokens', 'parsed_drugs']].apply(CRF_get_tag, axis=1)
  return df
    
def embedding_step(config):
  config_data = config['data']
  #config_preprocess = config['preprocessing']
  config_emb = config['embeddings']

  output_dir = config_data['output_dir']
  pretrained_emb_dir = config_data['pretrained_emb_dir']
        
  train_base_folder = config_data.get('train_dir')
  test_base_folder = config_data.get('test_dir')

  df_train = preprocess_steps(train_base_folder)
  df_test = preprocess_steps(test_base_folder)

  emb_dim = config_emb['emb_dim']
  emb_type = config_emb['emb_type']
  emb_window = config_emb['window']
  max_len = df_train['tokens'].apply(len).max()

  print('emb type, dim:', emb_type, emb_dim, emb_window)
    
  words = df_train['tokens'].apply(
      lambda el_lst: pd.Series([el['text'] for el in el_lst])).stack().unique().tolist()
  words.append("ENDPAD")

  word2idx = {w: i + 1 for i, w in enumerate(words)}

  tags = df_train['crf_tags'].apply(lambda el_lst: pd.Series(el_lst)).stack().unique()
  tag2idx = {t: i for i, t in enumerate(tags)}

  n_words = len(words)
  n_tags = len(tags)

  # Train
  X_train = [[word2idx[w['text']] for w in s] for s in df_train['tokens']]
  X_train = pad_sequences(maxlen=max_len, sequences=X_train, padding="post", value=word2idx['ENDPAD'])

  y_train = [[tag2idx[t] for t in s] for s in df_train['crf_tags']]
  y_train_pad = pad_sequences(maxlen=max_len, sequences=y_train, padding="post", value=tag2idx["O"])
  y_train = [to_categorical(i, num_classes=n_tags) for i in y_train_pad]

  # Test
  X_test = [[word2idx.get(w['text'], 0) for w in s] for s in df_test['tokens']]
  X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=0)

  y_test = [[tag2idx[t] for t in s] for s in df_test['crf_tags']]
  y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx["O"])
  y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]

  start = time()
  weights_file, weights = embedding_weights(
              X_train, y_train_pad,
              n_words, n_tags,
              max_len, emb_dim,
              output_dir,
              pretrained_emb_dir,
              word2idx,
              config_emb)
  emb_duration = time() - start

  return (X_train, y_train, X_test, y_test, df_train, df_test, word2idx, tag2idx, emb_duration, weights_file, weights, n_words, n_tags, max_len)


def run_model_eval(X_train, y_train, X_test, y_test, df_test,
                   config, tag2idx,
                   n_words, n_tags, max_len, weights=None):

  config_arch = config['arch']
  config_training = config['training']
  config_data = config['data']
  config_emb = config['embeddings']

  output_dir = config_data['output_dir']
  emb_dim = config_emb['emb_dim']

  print(output_dir)

  model = architecture(config_arch, n_words, n_tags, max_len, emb_dim, emb_weights=weights)
  #plot_model(model, to_file=path.join(output_dir,'model.png'), show_shapes=True)

  # Training
  optimizer = config_training['optimizer']

  if optimizer == 'rmsprop':
      if 'lrate' in config_training:
          optimizer = RMSprop(lr=config_training['lrate'],
                      decay=config_training['lrate']//config_training['EPOCHS'])
      else:
          optimizer = RMSprop(lr=0.001)
  else:
      optimizer = Adam(lr=config_training['lrate'],
                      decay=config_training['lrate']//config_training['EPOCHS'])

  cbacks = []
  tensorboard = TensorBoard(log_dir=output_dir+"/{}".format(time()),histogram_freq=0, write_graph=True, write_images=True)
  cbacks.append(tensorboard)

  modfile = path.join(output_dir, 'model.h5')

  model.compile(optimizer=optimizer, loss=losses.crf_loss, metrics=[metrics.crf_accuracy])

  idx2tag = dict(map(reversed, tag2idx.items()))
  metrics_per_epoch = Metrics(idx2tag=idx2tag)
  cbacks.append(metrics_per_epoch)

  x_tr, x_tst, y_tr, y_tst = train_test_split(
              X_train , np.array(y_train), test_size=0.1)
  start = time()
  history = model.fit(x_tr, y_tr, 
              batch_size=config_training['BATCH_SIZE'],
              epochs=config_training['EPOCHS'],
              validation_data=(x_tst, y_tst),
              callbacks=cbacks,
              #shuffle=False,
              verbose=1)
  duration_train = time() - start
  output_dir = config_data['output_dir']
  '''
  try:
      # Create target Directory
      mkdir(output_dir)
  except FileExistsError:
      print("Directory " , output_dir ,  " already exists")
  '''

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

  acc = history.history['crf_accuracy']
  val_acc = history.history['val_crf_accuracy']
  #Acc plot
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('model acc')
  plt.ylabel('acc')
  plt.xlabel('epoch')
  plt.legend(['train','val'], loc='upper left')
  plt.savefig(path.join(output_dir,'acc.png'))
  plt.close()

  emb_type = config_emb['emb_type']
  emb_alpha = config_emb['alpha']
  emb_dim = config_emb['emb_dim']
  emb_window = config_emb['window']

  np.save(f'{output_dir}/acc_{emb_type}_{emb_dim}_{emb_window}_{emb_alpha}.npy', acc)
  np.save(f'{output_dir}/val_acc_{emb_type}_{emb_dim}_{emb_window}_{emb_alpha}.npy', val_acc)
  np.save(f'{output_dir}/loss_{emb_type}_{emb_dim}_{emb_window}_{emb_alpha}.npy', loss)
  np.save(f'{output_dir}/val_loss_{emb_type}_{emb_dim}_{emb_window}_{emb_alpha}.npy', val_loss)

  f1s = metrics_per_epoch.val_f1s
  #F1 score plot
  plt.plot(f1s)
  plt.title('model f1 score')
  plt.ylabel('f1 score')
  plt.xlabel('epoch')
  plt.legend(['val'], loc='upper left')
  plt.savefig(path.join(output_dir,'f1s.png'))
  plt.close()

  np.save(f'{output_dir}/f1s_{emb_type}_{emb_dim}_{emb_window}_{emb_alpha}.npy', np.array(f1s))

  # Evaluation
  score_trn = model.evaluate(X_train, np.array(y_train), batch_size=config_training['BATCH_SIZE_TST'], verbose=0)
  score_tst = model.evaluate(X_test, np.array(y_test), batch_size=config_training['BATCH_SIZE_TST'], verbose=0)

  print('score', score_trn, score_tst)

  test_pred = model.predict(X_test, verbose=1)

  #pred_labels = pred2label(test_pred)    
  pred_labels = pred2label(test_pred, idx2tag)
  test_labels = pred2label(y_test, idx2tag)
  f1s_model = f1_score(test_labels, pred_labels)
  print("F1-score: {:.1%}".format(f1s_model))
  print(classification_report(test_labels, pred_labels))
  
  with open(f'{output_dir}/{classification_report}', 'w') as fd:
    fd.write(classification_report(test_labels, pred_labels))

  # Evaluation of SemEval
  df_test['pred_labels'] = pred_labels
  def keepOnlyTags(row):
      labels = row[0]
      crf_tags = row[1]
      return labels[:len(crf_tags)]
  df_test['preds'] = df_test[['pred_labels','crf_tags']].apply(keepOnlyTags, axis=1)

  drugs = []
  for tokens, crf_tags in zip(df_test['tokens'], df_test['preds']):
      current_drugs = []
      current_token = None
      for token, crf_tag in zip(tokens, crf_tags):
          if crf_tag == 'O':
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

  df_test['drugs'] = drugs
  out_file = path.join(output_dir,'task9.2_CRF1_1.txt')
  _ = Writer(out_file).call(df_test, col_names=['drugs'])

  return f1s_model, score_trn, score_tst
