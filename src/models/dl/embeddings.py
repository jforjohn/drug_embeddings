from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.optimizers import RMSprop, Adam

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from .MyGensimCallbacks import LossEpochSaver

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os import path, listdir

def reverseX(X_ind, word2idx):
  idx2word = dict(map(reversed, word2idx.items()))
  out = []
  for x_i in X_ind:
      out_i = []
      for p in x_i:
          #p_i = np.argmax(p)
          out_i.append(idx2word[p])
      out.append(out_i)
  return out

def w2v_get_weights(word2idx, n_words, emb_dim, w2v_model):
  embedding_matrix = np.zeros((n_words+1, emb_dim))
  for word, i in word2idx.items():
    try:
      embedding_vector = w2v_model.wv[word]
    except KeyError:
      embedding_vector = None
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  return embedding_matrix

def embedding_weights(X_train, y_train,
                      n_words, n_tags,
                      max_len, emb_dim,
                      output_dir, pretrained_emb_dir,
                      word2idx, 
                      config_emb):
  emb_weights = None
  weights_file = ''
  emb_type = config_emb['emb_type']
  alpha = config_emb['alpha']
  print('pame')
  if emb_type == 'simple':
    weights_file = f'weights_emb_simple_{emb_dim}_{alpha}.npy'
    if weights_file not in listdir(pretrained_emb_dir):
      model = Sequential()
      model.add(Embedding(input_dim=n_words+1,
                          output_dim=emb_dim,
                          input_length=max_len))
      model.add(Flatten())
      model.add(Dense(max_len, activation='sigmoid'))

      print(model.summary())
      plot_model(model, to_file=path.join(output_dir,'emb_model.png'), show_shapes=True)
                
      optimizer = Adam(lr=config_emb['alpha'],
                    decay=config_emb['alpha']//config_emb['epochs'])
      model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

      cbacks = []
      tensorboard = TensorBoard(
        log_dir=path.join(output_dir, 'tb_simple_embedding'),
        histogram_freq=0,
        write_graph=True,
        write_images=True)
      cbacks.append(tensorboard)

      history = model.fit(X_train, y_train,
                epochs=config_emb['epochs'],
                callbacks=cbacks,
                verbose=1)

      loss = history.history['loss']
      #Loss plot
      plt.plot(loss)
      plt.title('Embedding Loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.savefig(path.join(output_dir,'loss_emb.png'))
      plt.close()

      acc = history.history['acc']
      #Acc plot
      plt.plot(acc)
      plt.title('embedding acc')
      plt.ylabel('acc')
      plt.xlabel('epoch')
      plt.savefig(path.join(output_dir,'acc_emb.png'))
      plt.close()

      np.save(f'{output_dir}/simple_acc_{emb_dim}_{alpha}.npy', acc)
      np.save(f'{output_dir}/simple_loss_{emb_dim}_{alpha}.npy', loss)

      emb_weights = np.array(model.layers[0].get_weights())[0,:,:]
      np.save(f'{pretrained_emb_dir}/{weights_file}', emb_weights)
    else:
      print('weights file %s exists' %(weights_file))
      emb_weights = np.load(f'{pretrained_emb_dir}/{weights_file}')

  elif emb_type == 'w2v_sg':
    window = config_emb['window']
    negative = config_emb['negative']
    weights_file = f'weights_emb_sg_{emb_dim}_{window}_{alpha}_{negative}.npy'

    if weights_file not in listdir(pretrained_emb_dir):
      loss_saver = LossEpochSaver()
      x2_train = reverseX(X_train, word2idx)
      w2v_model = Word2Vec(x2_train,
                      min_count=1,
                      window=config_emb['window'],
                      size=emb_dim,
                      alpha=config_emb['alpha'],
                      iter=config_emb['epochs'],
                      #min_alpha=, 
                      #max_alpha=0.5, 
                      negative=config_emb['negative'],
                      compute_loss=True,
                      sg=1,
                      callbacks=[loss_saver])
      
      loss = loss_saver.losses
      #Emb loss plot
      plt.plot(loss)
      plt.title('Embedding loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train'], loc='upper left')
      plt.savefig(path.join(output_dir,'emb_loss_sg.png'))
      plt.close()

      np.save(f'{output_dir}/sg_loss_{emb_dim}_{window}_{alpha}_{negative}.npy', np.array(loss))
      
      emb_weights = w2v_get_weights(word2idx,
                                    n_words,
                                    emb_dim,
                                    w2v_model)
      del w2v_model
      np.save(f'{pretrained_emb_dir}/{weights_file}', emb_weights)
    else:
      emb_weights = np.load(f'{pretrained_emb_dir}/{weights_file}')

  elif emb_type == 'w2v_cb':
    window = config_emb['window']
    negative = config_emb['negative']
    weights_file = f'weights_emb_cb_{emb_dim}_{window}_{alpha}_{negative}.npy'
    if weights_file not in listdir(pretrained_emb_dir):
      loss_saver = LossEpochSaver()
      x2_train = reverseX(X_train, word2idx)
      w2v_model = Word2Vec(x2_train,
                      min_count=1,
                      window=config_emb['window'],
                      size=emb_dim,
                      alpha=config_emb['alpha'],
                      iter=config_emb['epochs'],
                      #min_alpha=, 
                      #max_alpha=0.5, 
                      negative=config_emb['negative'],
                      compute_loss=True,
                      sg=0,
                      callbacks=[loss_saver])
      loss = loss_saver.losses
      #Emb loss plot
      plt.plot(loss)
      plt.title('Embedding loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train'], loc='upper left')
      plt.savefig(path.join(output_dir,'emb_loss_cb.png'))
      plt.close()

      np.save(f'{output_dir}/cb_loss_{emb_dim}_{window}_{alpha}_{negative}.npy', np.array(loss))

      emb_weights = w2v_get_weights(word2idx,
                                    n_words,
                                    emb_dim,
                                    w2v_model)
      
      del w2v_model
      np.save(f'{pretrained_emb_dir}/{weights_file}', emb_weights)
    else:
      emb_weights = np.load(f'{pretrained_emb_dir}/{weights_file}')

  elif emb_type == 'pretrained_glove_50':
    weights_file = 'weights_emb_glove50.npy'
    if weights_file not in listdir(pretrained_emb_dir):
      glove_file = path.join(pretrained_emb_dir, config_emb['emb_file'])
      w2v_glove_file = path.join(pretrained_emb_dir, 'w2v'+config_emb['emb_file'])
      glove2word2vec(glove_file, w2v_glove_file)
      w2v_model = KeyedVectors.load_word2vec_format(w2v_glove_file, binary=False)

      emb_weights = w2v_get_weights(word2idx,
                                    n_words,
                                    emb_dim,
                                    w2v_model)
      del w2v_model
      np.save(f'{pretrained_emb_dir}/{weights_file}', emb_weights)
    else:
      emb_weights = np.load(f'{pretrained_emb_dir}/{weights_file}')
  
  elif emb_type == 'pretrained_glove_200':
    weights_file = 'weights_emb_glove200.npy'
    if weights_file not in listdir(pretrained_emb_dir):
      glove_file = path.join(pretrained_emb_dir, config_emb['emb_file'])
      w2v_glove_file = path.join(pretrained_emb_dir, 'w2v'+config_emb['emb_file'])
      glove2word2vec(glove_file, w2v_glove_file)
      w2v_model = KeyedVectors.load_word2vec_format(w2v_glove_file, binary=False)

      emb_weights = w2v_get_weights(word2idx,
                                    n_words,
                                    emb_dim,
                                    w2v_model)
      del w2v_model
      np.save(f'{pretrained_emb_dir}/{weights_file}', emb_weights)
    else:
      emb_weights = np.load(f'{pretrained_emb_dir}/{weights_file}')

  elif emb_type == 'pretrained_pub':
    weights_file = 'weights_emb_pub.npy'
    if weights_file not in listdir(pretrained_emb_dir):
      w2v_glove_file = path.join(pretrained_emb_dir, config_emb['emb_file'])
      w2v_model = KeyedVectors.load_word2vec_format(w2v_glove_file, binary=True)

      emb_weights = w2v_get_weights(word2idx,
                                    n_words,
                                    emb_dim,
                                    w2v_model)
      del w2v_model
      np.save(f'{pretrained_emb_dir}/{weights_file}', emb_weights)
    else:
      emb_weights = np.load(f'{pretrained_emb_dir}/{weights_file}')
  return weights_file, emb_weights