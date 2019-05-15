from gensim.models.callbacks import CallbackAny2Vec
class LossEpochSaver(CallbackAny2Vec):
  '''Callback to save model after each epoch and show training parameters '''

  def __init__(self):
    self.epoch = 0
    self.last_loss = 0
    self.losses = []

  def on_epoch_end(self, model):
    loss_cur = model.get_latest_training_loss()
    actual_loss = loss_cur - self.last_loss
    self.losses.append(actual_loss)
    self.last_loss = loss_cur
    print('Loss after epoch {}: {}'.format(self.epoch, actual_loss))
    self.epoch += 1
