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
""" === Under active development === Script to fine-tune GLUE on a TPU

Adapted from https://github.com/tensorflow/models
Especially https://github.com/tensorflow/models/blob/master/official/modeling/model_training_utils.py
"""

from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig
from tpu_utils import get_tpu
from tpu_dataset import create_dataset
import functools
import tensorflow as tf
import logging
import json
import math


def _get_input_iterator(input_fn, strategy):
    """Returns distributed dataset iterator."""

    # When training with TPU pods, datasets needs to be cloned across
    # workers. Since Dataset instance cannot be cloned in eager mode, we instead
    # pass callable that returns a dataset.
    input_data = input_fn()
    if callable(input_data):
        iterator = iter(
            strategy.experimental_distribute_datasets_from_function(input_data))
    else:
        iterator = iter(strategy.experimental_distribute_dataset(input_data))
    return iterator


def _steps_to_run(current_step, steps_per_epoch, steps_per_loop):
    """Calculates steps to run on device."""
    if steps_per_loop <= 0:
        raise ValueError('steps_per_loop should be positive integer.')
    if steps_per_loop == 1:
        return steps_per_loop
    remainder_in_epoch = current_step % steps_per_epoch
    if remainder_in_epoch != 0:
        return min(steps_per_epoch - remainder_in_epoch, steps_per_loop)
    else:
        return steps_per_loop


def get_loss_fn(num_classes, loss_factor=1.0):
    """Gets the classification loss function."""

    def classification_loss_fn(labels, logits):
        """Classification loss."""
        labels = tf.squeeze(labels)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(
            tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        loss *= loss_factor
        return loss

    return classification_loss_fn


def run_customized_training_loop(
        strategy,
        model_fn,
        loss_fn,
        train_input_fn,
        steps_per_loop,
        steps_per_epoch,
        epochs,
        eval_input_fn,
        eval_steps,
        metric_fn,
):
    total_training_steps = steps_per_epoch * epochs

    train_input_data = train_input_fn()
    train_iterator = iter(strategy.experimental_distribute_dataset(train_input_data))

    with strategy.scope():
        model = model_fn()
        optimizer = model.optimizer
        train_loss_metric = tf.keras.metrics.Mean(
            'training_loss', dtype=tf.float32
        )

        eval_metrics = [metric_fn()]

        train_metrics = [
            metric.__class__.from_config(metric.get_config())
            for metric in eval_metrics
        ]

        # Collects training variables.
        training_vars = model.trainable_variables

        def _replicated_step(inputs):
            """Replicated training step."""

            inputs, labels = inputs
            with tf.GradientTape() as tape:
                model_outputs = model(inputs, training=True)[0]
                loss = loss_fn(labels, model_outputs)

            grads = tape.gradient(loss, training_vars)
            optimizer.apply_gradients(zip(grads, training_vars))
            # For reporting, the metric takes the mean of losses.
            train_loss_metric.update_state(loss)
            for metric in train_metrics:
                metric.update_state(labels, model_outputs)


        @tf.function
        def train_steps(iterator, steps):
            for _ in tf.range(steps):
                strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))

        def train_single_step(iterator):
            strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))

        def test_step(iterator):
            def _test_step_fn(inputs):
                inputs, labels = inputs
                model_outputs = model(inputs, training=False)
                for metric in eval_metrics:
                    metric.update_state(labels, model_outputs)
            strategy.experimental_run_v2(_test_step_fn, args=(next(iterator),))

        train_single_step = tf.function(train_single_step)
        test_step = tf.function(test_step)

        def _run_evaluation(current_training_step, test_iterator):
            for _ in range(eval_steps):
                test_step(test_iterator)

        current_step = optimizer.iterations.numpy()

        while current_step < total_training_steps:
            train_loss_metric.reset_states()
            for metric in train_metrics + model.metrics:
                metric.reset_states()

            steps = _steps_to_run(current_step, steps_per_epoch, steps_per_loop)

            if steps == 1:
                train_single_step(train_iterator)
            else:
                train_steps(train_iterator, tf.convert_to_tensor(steps, dtype=tf.int32))

            current_step += steps
            train_loss = train_loss_metric.result().numpy().astype(float)
            training_status = 'Train Step: %d/%d  / loss = %s' % (current_step, total_training_steps, train_loss)
            print(training_status)

    return model


def run_customized_training(
        tokenizer,
        strategy,
        dataset_path,
        max_sequence_length,
        train_batch_size,
        eval_batch_size,

        num_classes=10,
        num_replicas=8,
        steps_per_loop=10,
        steps_per_epoch=10,
        epochs=10,
        eval_steps=10
):
    train_input_fn = functools.partial(
        create_dataset,
        tokenizer,
        dataset_path,
        max_sequence_length,
        train_batch_size
    )

    eval_input_fn = functools.partial(
        create_dataset,
        tokenizer,
        dataset_path,
        max_sequence_length,
        eval_batch_size,
        evaluate=True
    )

    def model_fn():
        config = BertConfig.from_pretrained("bert-base-cased")
        config.num_labels = 3
        model = TFBertForSequenceClassification.from_pretrained("bert-base-cased", config=config)
        optimizer = tf.keras.optimizers.Adam()
        model.optimizer = optimizer
        return model

    loss_fn = get_loss_fn(num_classes, loss_factor=1.0/num_replicas)

    def metric_fn():
        return tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy', dtype=tf.float32)

    return run_customized_training_loop(
        strategy=strategy,
        model_fn=model_fn,
        loss_fn=loss_fn,
        train_input_fn=train_input_fn,
        steps_per_loop=steps_per_loop,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        eval_input_fn=eval_input_fn,
        eval_steps=eval_steps,
        metric_fn=metric_fn,
    )


if __name__ == "__main__":
    strategy, num_replicas = get_tpu()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    input_meta_data_path = "gs://huggingface-bucket/transformers/outputs/MNLI_meta_data"
    dataset_path = "/home/lysandre/transformers/examples/TPU/glue_data/MNLI"

    with tf.io.gfile.GFile(input_meta_data_path, 'rb') as reader:
        input_meta_data = json.loads(reader.read().decode('utf-8'))

    max_sequence_length = input_meta_data["max_seq_length"]
    num_classes = input_meta_data['num_labels']

    epochs = 3
    train_batch_size = 32
    eval_batch_size = 32

    train_data_size = input_meta_data["train_data_size"]
    steps_per_epoch = int(train_data_size / train_batch_size)
    steps_per_loop = 200
    warmup_steps = int(epochs * train_data_size * 0.1 / train_batch_size)
    eval_steps = int(math.ceil(input_meta_data['eval_data_size'] / eval_batch_size))

    trained_model = run_customized_training(
        tokenizer,
        strategy,
        dataset_path,
        max_sequence_length,
        train_batch_size,
        eval_batch_size,

        num_classes=num_classes,
        num_replicas=num_replicas,
        steps_per_loop=steps_per_loop,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        eval_steps=eval_steps
    )
