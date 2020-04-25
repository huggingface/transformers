"""Tensorflow trainer class."""

import itertools
import logging
import math
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from data_processors import DatasetInfo
from tf_training_args import TFTrainingArguments
from transformers import (
    AdamWeightDecay,
    GradientAccumulator,
    TFPreTrainedModel,
    WarmUp,
)


logger = logging.getLogger(__name__)


class TFTrainer:
    model: TFPreTrainedModel
    args: TFTrainingArguments
    train_dataset: Optional[tf.data.Dataset]
    eval_dataset: Optional[tf.data.Dataset]
    test_dataset: Optional[tf.data.Dataset]
    dataset_info: DatasetInfo

    def __init__(
        self,
        model: TFPreTrainedModel,
        args: TFTrainingArguments,
        train_dataset: Optional[tf.data.Dataset] = None,
        eval_dataset: Optional[tf.data.Dataset] = None,
        test_dataset: Optional[tf.data.Dataset] = None,
        dataset_info: Optional[DatasetInfo] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.dataset_info = dataset_info
        self.gradient_accumulator = GradientAccumulator()

        self._setup_training()

    def _setup_training(self) -> None:
        """
        Setup the different steps to train a model:
          - check if all the data are given
          - create the proper strategy
          - create the features
          - prepare the model settings
        """
        self._prepare_dataset()

        with self.args.strategy.scope():
            self._create_optimizer()
            _ = self.optimizer.iterations
            self._set_loss_and_metric()
            self._create_checkpoint_manager(self.args.output_dir)
            self._create_summary_writer(self.args.logging_dir)

    def _set_loss_and_metric(self) -> None:
        """
        Create the training loss and metric with their name. Allowed names are those listed
        in the Tensorflow documentation and those contained in the transformers library.
        """
        try:
            self.loss = tf.keras.losses.get({"class_name": self.args.loss_name, "config": {"from_logits": True, "reduction": tf.keras.losses.Reduction.NONE}})
        except TypeError:
            self.loss = tf.keras.losses.get({"class_name": self.args.loss_name, "config": {"reduction": tf.keras.losses.Reduction.NONE}})

        self.train_acc_metric = tf.keras.metrics.get({"class_name": self.args.metric_name, "config": {"name": "train_accuracy"}})
        self.test_acc_metric = tf.keras.metrics.get({"class_name": self.args.metric_name, "config": {"name": "test_accuracy"}})

    def _create_summary_writer(self) -> None:
        """
        Create a summary writer to be able to read the logs in Tensorboard.
        """
        self.train_writer = tf.summary.create_file_writer(self.args.logging_dir + "/train")
        self.test_writer = tf.summary.create_file_writer(self.args.logging_dir + "/test")

    def _prepare_dataset(self) -> None:
        """
        Prepare the training, validation and test data.
        """
        test_batch = self.args.per_gpu_eval_batch_size
        self.train_steps = math.ceil(self.dataset_info.sizes["train"] / self.args.train_batch_size)
        self.train_dataset = self.train_dataset.shuffle(128).batch(self.args.train_batch_size).repeat(-1)
        self.train_dataset = self.args.strategy.experimental_distribute_dataset(self.train_dataset)
        self.validation_steps = math.ceil(self.dataset_info.sizes["validation"] / self.args.eval_batch_size)
        self.eval_dataset = self.eval_dataset.batch(self.args.eval_batch_size)
        self.eval_dataset = self.args.strategy.experimental_distribute_dataset(self.eval_dataset)
        self.test_steps = math.ceil(self.dataset_info.sizes["test"] / test_batch)
        self.test_dataset = self.test_dataset.batch(test_batch)

    def _create_optimizer(self) -> None:
        """
        Create the training optimizer with its name. Allowed names are those listed
        in the Tensorflow documentation and those contained in the transformers library.
        """
        if self.args.optimizer_name == "adamw":
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=self.args.learning_rate,
                                                                             decay_steps=self.train_steps,
                                                                             end_learning_rate=0.0)
            if self.args.warmup_steps:
                learning_rate_fn = WarmUp(initial_learning_rate=self.args.learning_rate, decay_schedule_fn=learning_rate_fn,
                                          warmup_steps=self.args.warmup_steps)

            self.optimizer = AdamWeightDecay(learning_rate=learning_rate_fn, weight_decay_rate=0.01, epsilon=self.args.adam_epsilon,
                                             exclude_from_weight_decay=["layer_norm", "bias"])
        else:
            try:
                self.optimizer = tf.keras.optimizers.get({"class_name": self.args.optimizer_name, "config" : {"learning_rate": self.args.learning_rate, "epsilon": self.args.adam_epsilon}})
            except TypeError:
                # This is for the case where the optimizer is not Adam-like such as SGD
                self.optimizer = tf.keras.optimizers.get({"class_name": self.args.optimizer_name, "config" : {"learning_rate": self.args.learning_rate}})

    def _create_checkpoint_manager(self, max_to_keep: int = 5, load_model: bool = True) -> None:
        """
        Create a checkpoint manager in order to be able to make the training
        fault-tolerant.
        Args:
          max_to_keep: the maximum number of checkpoints to keep in the checkpoint path.
          load_model: if we want to start the training from the latest checkpoint.
        """
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, self.args.output_dir, max_to_keep=max_to_keep)

        if load_model:
            ckpt.restore(self.model.ckpt_manager.latest_checkpoint)

    def _evaluate_steps(self, per_replica_features, per_replica_labels):
        """
        One step evaluation across replica.
        Args:
          per_replica_features: the batched features.
          per_replica_labels: the batched labels.
        Returns:
          The loss corresponding to the given batch.
        """
        per_replica_loss = self.args.strategy.experimental_run_v2(self._run_model, args=(per_replica_features, per_replica_labels, False))

        return self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

    def _evaluate(self) -> None:
        """
        Evaluate the model during the training at the end of each epoch.
        """
        step = 1
        loss = 0.0

        for features, labels in self.eval_dataset:
            step = tf.convert_to_tensor(step, dtype=tf.int64)
            loss = self._evaluate_steps(features, labels)
            loss = tf.reduce_mean(loss)

            with self.test_writer.as_default():
                tf.summary.scalar("loss", loss, step=step)

            if step % self.validation_steps == 0:
                break

            step += 1

        return loss

    def train(self) -> None:
        """
        Train method to train the model.
        """
        tf.summary.trace_on(graph=True, profiler=True)
        self.gradient_accumulator.reset()

        iterations = self.optimizer.iterations
        tf.summary.experimental.set_step(iterations)

        for epoch in range(int(self.args.num_train_epochs)):
            for training_loss in self._training_steps():
                step = iterations.numpy()
                training_loss = tf.reduce_mean(training_loss)

                with self.train_writer.as_default():
                    tf.summary.scalar("loss", training_loss, step=step)

                if step == 1:
                    with self.train_writer.as_default():
                        tf.summary.trace_export(name="training", step=step, profiler_outdir=self.args.logging_dir)

                if step % 10 == 0:
                    logger.info("Epoch {} Step {} Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step, training_loss.numpy(), self.train_acc_metric.result()))

                if step % 100 == 0:
                    ckpt_save_path = self.model.ckpt_manager.save()
                    logger.info("Saving checkpoint for step {} at {}".format(step, ckpt_save_path))

                if step % self.train_steps == 0:
                    break

            test_loss = self._evaluate()

            logger.info("Epoch {} Step {} Train Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step, training_loss.numpy(), self.train_acc_metric.result()))
            logger.info("Epoch {} Validation Loss {:.4f} Validation Accuracy {:.4f}".format(epoch, test_loss.numpy(), self.test_acc_metric.result()))

            self.train_acc_metric.reset_states()
            self.test_acc_metric.reset_states()

    def _training_steps(self):
        """
        Returns a generator over training steps (i.e. parameters update).
        """
        for i, loss in enumerate(self._accumulate_next_gradients()):
            if i % self.accum_steps == 0:
                self._apply_gradients()
                yield loss

    @tf.function
    def _apply_gradients(self):
        """Applies the gradients (cross-replica)."""
        self.args.strategy.experimental_run_v2(self._step)

    def _step(self):
        """Applies gradients and resets accumulation."""
        gradient_scale = self.gradient_accumulator.step * self.args.strategy.num_replicas_in_sync
        gradients = [gradient / tf.cast(gradient_scale, gradient.dtype) for gradient in self.gradient_accumulator.gradients]
        gradients = [(tf.clip_by_value(grad, -self.args.max_grad_norm, self.args.max_grad_norm)) for grad in gradients]
        vars = self.model.trainable_variables

        if self.args.mode == "labelling":
            vars = [var for var in self.model.trainable_variables if "pooler" not in var.name]

        self.optimizer.apply_gradients(list(zip(gradients, vars)))
        self.gradient_accumulator.reset()

    def _accumulate_next_gradients(self):
        """Accumulates the gradients from the next element in dataset."""
        iterator = iter(self.train_dataset)

        @tf.function
        def _accumulate_next():
            per_replica_features, per_replica_labels = next(iterator)

            return self._accumulate_gradients(per_replica_features, per_replica_labels)

        while True:
            try:
                yield _accumulate_next()
            except tf.errors.OutOfRangeError:
                break

    def _accumulate_gradients(self, per_replica_features, per_replica_labels):
        """Accumulates the gradients across all the replica."""
        per_replica_loss = self.args.strategy.experimental_run_v2(self._forward, args=(per_replica_features, per_replica_labels))

        return self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

    def _forward(self, features, labels):
        """Forwards a training example and accumulates the gradients."""
        per_example_loss = self._run_model(features, labels, True)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.args.train_batch_size)
        vars = self.model.trainable_variables

        if self.args.mode == "labelling":
            vars = [var for var in self.model.trainable_variables if "pooler" not in var.name]

        gradients = self.optimizer.get_gradients(loss, vars)

        self.gradient_accumulator(gradients)

        return per_example_loss

    def _run_model(self, features, labels, training):
        """
        Computes the loss of the given features and labels pair.
        Args:
          features: the batched features.
          labels: the batched labels.
        """
        if self.args.mode == "classification" or self.args.mode == "labelling":
            logits = self.model(features, training=training)[0]
        else:
            logits = self.model(features, training=training)

        if self.args.mode == "labelling":
            active_loss = tf.reshape(labels, (-1,)) != -1
            logits = tf.boolean_mask(tf.reshape(logits, (-1, len(self.dataset_info.labels))), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

        loss = self.loss(labels, logits)

        if training:
            self.train_acc_metric(labels, logits)
        else:
            self.test_acc_metric(labels, logits)

        return loss

    def test(self) -> None:
        """
        Test the model over the test dataset and print a report.
        """
        y_true = []
        results = self.model.predict(self.test_dataset, steps=self.test_steps)

        if self.args.mode == "classification":
            for batch in self.test_dataset:
                y_true.extend(batch[1].numpy().tolist())

            y_pred = np.reshape(np.argmax(results, axis=-1), (-1, 1)).tolist()
            y_true = list(itertools.chain.from_iterable(y_true))
            y_pred = list(itertools.chain.from_iterable(y_pred))

            logger.info(classification_report(y_true, y_pred, target_names=self.dataset_info.labels))

    def save_model(self) -> None:
        """
        Save the pretrained model and create a Tensorflow saved model.
        """
        logger.info("Saving model in {}".format(self.args.output_dir))

        path = os.path.join(self.args.output_dir, "saved_model")

        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(self.args.output_dir)
        tf.saved_model.save(self.model, path)
