import tensorflow as tf

class QALoss(tf.keras.losses.Loss):
    def __init__(self, name="qa_loss"):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
    
    @tf.function
    def call(self, y_true, y_pred):
        start_positions = y_true["start_position"]
        end_positions = y_true["end_position"]
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
        start_loss = loss_fn(start_positions, y_pred[0])
        end_loss = loss_fn(end_positions, y_pred[1])
        total_loss = (start_loss + end_loss) / 2

        return total_loss
