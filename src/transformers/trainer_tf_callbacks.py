import tensorflow as tf


class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        it = self.model.optimizer.iterations
        lr = self.model.optimizer.lr(it)
        tf.summary.scalar("learning rate", data=lr, step=self.model.optimizer.iterations)
