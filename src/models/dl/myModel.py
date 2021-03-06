from keras.layers import Dense
from keras.layers import LSTM, GRU, Dropout
from keras.regularizers import L1L2
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras.constraints import maxnorm


def architecture(config, n_words, n_tags, max_len, emb_dim, emb_weights=None):
  arch_type = config['arch_type']
  neurons_rnn = config['neurons_rnn']
  neurons_dense = config['neurons_dense']
  rec_drop = config['rec_drop']
  impl = config['impl']

  if arch_type == 'BLSTM':
    input_model = Input(shape=(max_len,))
    if emb_weights is not None:
      print('-------Using pretrained weights--------')
      model = Embedding(input_dim=n_words + 1,
                        output_dim=emb_dim,
                        input_length=max_len,
                        weights=[emb_weights],
                        trainable=False,
                        mask_zero=True)(input_model)
    else:
      model = Embedding(input_dim=n_words + 1,
                        output_dim=emb_dim,
                        input_length=max_len,
                        mask_zero=True)(input_model)  # 20-dim embedding
    model = Bidirectional(LSTM(units=neurons_rnn,
                          return_sequences=True,
                          recurrent_dropout=rec_drop,
                          dropout=0.4,
                          implementation=impl))(model)  # variational biLSTM
    #model = Dropout(0.4)(model)
    model = TimeDistributed(
      Dense(neurons_dense, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    # kernel_constraint=maxnorm(3)
    #model = Dropout(0.2)(model)
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output

    model = Model(input_model, out)

  #print(model.summary())

  return model
