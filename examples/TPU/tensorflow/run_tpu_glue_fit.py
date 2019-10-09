# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" === Under active development === Script to fine-tune GLUE on a TPU using keras' fit method
"""


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
valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_seq_length, 'mrpc')

total_train_batch_size = 32
train_batch_size_per_replica = total_train_batch_size / num_replicas
train_dataset = train_dataset.batch(total_train_batch_size)
assert train_batch_size_per_replica.is_integer()

total_valid_batch_size = 64
valid_batch_size_per_replica = total_valid_batch_size / num_replicas
valid_dataset = valid_dataset.batch(total_valid_batch_size)
assert valid_batch_size_per_replica.is_integer()
print('Fetched & created dataset.')

tpu, num_replicas = get_tpu()

with tpu.scope():
    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

final_stats = model.evaluate(valid_dataset, steps=1)
print("Validation accuracy: ", final_stats[1])