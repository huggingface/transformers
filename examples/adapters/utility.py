import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np
import os

def spearman(labels, preds):
    return (tf.py_function(spearmanr, [tf.cast(labels, tf.float32), 
                           tf.cast(preds, tf.float32)], Tout = tf.float32))
def f1(labels, preds):
    preds = tf.math.argmax(preds, axis=1)
    return (tf.py_function(f1_score, [tf.cast(labels, tf.float32), 
                           tf.cast(preds, tf.float32)], Tout = tf.float32))
def matthews_cc(labels, preds):
    preds = tf.math.argmax(preds, axis=1)
    return (tf.py_function(matthews_corrcoef, [tf.cast(labels, tf.float32), 
                           tf.cast(preds, tf.float32)], Tout = tf.float32))
    

class ModelCheckpoint(tf.keras.callbacks.Callback):
  def __init__(self, monitor, save_path):
    super(ModelCheckpoint, self).__init__()
    self.monitor = monitor
    self.save_path = save_path
    self.bestScore = -np.Inf
    self.bestLoss = np.Inf

  def on_epoch_end(self, epoch, logs):
    score = logs.get(self.monitor)
    loss = logs.get("val_loss")
    if score > self.bestScore or (score == self.bestScore and loss < self.bestLoss):
      path = os.path.join(self.save_path, str(epoch+1))
      os.makedirs(path)
      self.model.save_weights(path+'/best_weights.h5')
      self.bestScore = score
      self.bestLoss = loss
      print("\nModel saved as the best model")

