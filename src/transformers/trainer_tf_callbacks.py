import os
import operator

import tensorflow as tf


class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        it = self.model.optimizer.iterations
        lr = self.model.optimizer.lr(it)
        tf.summary.scalar("learning rate", data=lr, step=self.model.optimizer.iterations)


class KeepNCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, checkpoints_dir, num_keep=5):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir
        self.num_keep = num_keep

    def on_epoch_end(self, epoch, logs=None):
        checkpoints = self.sort_checkpoints(self.checkpoints_dir)

        if self.num_keep > 0:
            checkpoints = self.sort_checkpoints(self.checkpoints_dir)

            for checkpoint in checkpoints[self.num_keep:]:
                checkpoint_files = tf.io.gfile.glob(checkpoint + '*')

                for file in checkpoint_files:
                    tf.io.gfile.remove(file)

    
    def sort_checkpoints(self, model_dir: str):
        """
        Get all checkpoints in descending order sorted by epoch.
        Args:
            model_dir (:obj:`str`):
                The location of the checkpoints.
        Returns:
            sorted list of checkpoints
        """
        checkpoints = tf.io.gfile.glob(os.path.join(model_dir, 'weights-*.index'))
        checkpoints = map(lambda s: s.strip('.index'), checkpoints)
        by_epoch = []

        for checkpoint in checkpoints:
            checkpoint_name = os.path.basename(checkpoint)
            epoch = checkpoint_name.split('-')[1]

            by_epoch.append((epoch, checkpoint))

        by_epoch = sorted(by_epoch, key=operator.itemgetter(0))
        by_epoch = by_epoch[::-1]

        return list(map(operator.itemgetter(1), by_epoch))