import numpy as np
from keras.callbacks import Callback
from seqeval.metrics import f1_score, precision_score, recall_score

class Metrics(Callback):
  def __init__(self, idx2tag={}, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.idx2tag = idx2tag
  
  def on_train_begin(self, logs={}):
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []
    
  def _pred2label(self,pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(self.idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

  def on_epoch_end(self, epoch, logs={}):
    super().on_epoch_end(epoch, logs)
    val_predict = self._pred2label(self.model.predict(self.validation_data[0]))
    val_targ = self._pred2label(self.validation_data[1])
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    #logs['val_f1'] = _val_f1
    print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
    
    return
