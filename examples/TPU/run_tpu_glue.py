from tpu_utils import get_tpu
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, glue_convert_examples_to_features
from time import time
import tensorflow_datasets
print("TF version: {}".format(tf.__version__))

num_epochs = 3
max_seq_length = 128

# The number of replicas should be obtained from the get_tpu() method, but the dataset pre-processing crashes if the
# TPU is loaded beforehand
num_replicas = 8

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
data = tensorflow_datasets.load('glue/mrpc')
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_seq_length, 'mrpc')
test_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_seq_length, 'mrpc')

total_train_batch_size = 32
train_batch_size_per_replica = total_train_batch_size / num_replicas
train_dataset = train_dataset.batch(total_train_batch_size)
assert train_batch_size_per_replica.is_integer()

total_test_batch_size = 64
test_batch_size_per_replica = total_test_batch_size / num_replicas
test_dataset = test_dataset.batch(total_test_batch_size)
assert test_batch_size_per_replica.is_integer()
print('Fetched & created dataset.')

strategy, _ = get_tpu()
print('TPUStrategy obtained.')

dataset_options = tf.data.Options()
dataset_options.experimental_distribute.auto_shard = False

train_dataset = train_dataset.with_options(dataset_options)
test_dataset = test_dataset.with_options(dataset_options)

with strategy.scope():
    # Crashes here
    train_distributed_dataset = strategy.experimental_distribute_dataset(train_dataset)








# with strategy.scope():
#     model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")
#     optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True,
#         reduction=tf.keras.losses.Reduction.NONE
#     )
#     metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
#     test_loss = tf.keras.metrics.Mean(name='test_loss')
#     train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#     test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#
#     print('Hyper-parameters set.')
#
#     def compute_loss(model, x, y):
#         y_ = model(x)[0]
#         per_example_loss = loss_object(y, y_)
#         return tf.nn.compute_average_loss(per_example_loss, global_batch_size=total_train_batch_size)
#
#
#     def grad(model, inputs, targets):
#         with tf.GradientTape() as tape:
#             loss_value = compute_loss(model, inputs, targets)
#         return loss_value, tape.gradient(loss_value, model.trainable_variables)
#
#
#     train_loss_results = []
#     train_accuracy_results = []
#
#     def train_step(inputs):
#         features, labels = inputs
#
#         with tf.GradientTape() as tape:
#             predictions = model(features, training=True)
#             loss = compute_loss(labels, predictions)
#
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#         train_accuracy.update_state(labels, predictions)
#         return loss
#
#     def test_step(inputs):
#         images, labels = inputs
#
#         predictions = model(images, training=False)
#         t_loss = loss_object(labels, predictions)
#
#         test_loss.update_state(t_loss)
#         test_accuracy.update_state(labels, predictions)