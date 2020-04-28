"""Tensorflow trainer class."""

import logging
import math
import os
from typing import Callable, Dict, Optional

import numpy as np
import tensorflow as tf

from .modeling_tf_utils import TFPreTrainedModel, shape_list
from .optimization_tf import GradientAccumulator, create_optimizer
from .trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from .training_tf_args import TFTrainingArguments


logger = logging.getLogger(__name__)


class TFDataset:
    TrainOutput,
    """
    Fake superclass to partially imitate the PT Dataset class.
    """

    def get_dataset(self):
        return self.dataset


class TFTrainer:
    model: TFPreTrainedModel
    args: TFTrainingArguments
    # something similar to a PT Dataset.
    # This is just temporary before to have
    # a framework-agnostic approach for datasets.
    train_dataset: Optional[TFDataset]
    eval_dataset: Optional[TFDataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool

    def __init__(
        self,
        model: TFPreTrainedModel,
        args: TFTrainingArguments,
        train_dataset: Optional[TFDataset] = None,
        eval_dataset: Optional[TFDataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
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
            self._create_checkpoint_manager()
            self._create_summary_writer()

    def _set_loss_and_metric(self) -> None:
        """
        Create the training loss and metric with their name. Allowed names are those listed
        in the Tensorflow documentation and those contained in the transformers library.
        """
        try:
            self.loss = tf.keras.losses.get(
                {
                    "class_name": self.args.loss_name,
                    "config": {"from_logits": True, "reduction": tf.keras.losses.Reduction.NONE},
                }
            )
        except TypeError:
            self.loss = tf.keras.losses.get(
                {"class_name": self.args.loss_name, "config": {"reduction": tf.keras.losses.Reduction.NONE}}
            )

    def _create_summary_writer(self) -> None:
        """
        Create a summary writer to be able to read the logs in Tensorboard.
        """
        self.writer = tf.summary.create_file_writer(self.args.logging_dir)

    def _prepare_dataset(self) -> None:
        """
        Prepare the training, validation and test data.
        """
        if self.train_dataset is not None:
            if self.args.max_steps > 0:
                self.train_steps = self.args.max_steps
            else:
                self.train_steps: int = math.ceil(len(self.train_dataset) / self.args.train_batch_size)

            self.tf_train_dataset: tf.data.Dataset = self.train_dataset.get_dataset().shuffle(128).batch(
                self.args.train_batch_size
            ).repeat(-1)
            self.tf_train_dataset: tf.data.Dataset = self.args.strategy.experimental_distribute_dataset(
                self.tf_train_dataset
            )
        else:
            self.train_steps = 0

        if self.eval_dataset is not None:
            self.tf_eval_dataset: tf.data.Dataset = self.eval_dataset.get_dataset().batch(self.args.eval_batch_size)
            self.tf_eval_dataset: tf.data.Dataset = self.args.strategy.experimental_distribute_dataset(self.tf_eval_dataset)

    def _create_optimizer(self) -> None:
        """
        Create the training optimizer with its name. Allowed names are those listed
        in the Tensorflow documentation and those contained in the transformers library.
        """
        if self.args.optimizer_name == "adamw":
            self.optimizer = create_optimizer(self.args.learning_rate, self.train_steps, self.args.warmup_steps)
        else:
            try:
                self.optimizer = tf.keras.optimizers.get(
                    {
                        "class_name": self.args.optimizer_name,
                        "config": {"learning_rate": self.args.learning_rate, "epsilon": self.args.adam_epsilon},
                    }
                )
            except TypeError:
                # This is for the case where the optimizer is not Adam-like such as SGD
                self.optimizer = tf.keras.optimizers.get(
                    {"class_name": self.args.optimizer_name, "config": {"learning_rate": self.args.learning_rate}}
                )

    def _create_checkpoint_manager(self, max_to_keep: int = 5, load_model: bool = True) -> None:
        """
        Create a checkpoint manager in order to be able to make the training
        fault-tolerant.
        Args:
          max_to_keep: the maximum number of checkpoints to keep in the checkpoint path.
          load_model: if we want to start the training from the latest checkpoint.
        """
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, PREFIX_CHECKPOINT_DIR, max_to_keep=max_to_keep)

        if load_model:
            ckpt.restore(self.model.ckpt_manager.latest_checkpoint).expect_partial()

    @tf.function
    def _evaluate_steps(self, per_replica_features, per_replica_labels):
        """
        One step evaluation across replica.
        Args:
          per_replica_features: the batched features.
          per_replica_labels: the batched labels.
        Returns:
          The loss corresponding to the given batch.
        """
        per_replica_loss, per_replica_logits = self.args.strategy.experimental_run_v2(
            self._run_model, args=(per_replica_features, per_replica_labels, False)
        )

        return self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=0), per_replica_logits

    def _prediction_loop(
        self, dataset: tf.data.Dataset, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        logger.info("***** Running %s *****", description)
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        label_ids: np.ndarray = None
        preds: np.ndarray = None

        step: int = 1
        loss: float = 0.0

        for features, labels in dataset:
            step = tf.convert_to_tensor(step, dtype=tf.int64)
            loss, logits = self._evaluate_steps(features, labels)
            loss = tf.reduce_mean(loss)

            if not prediction_loss_only:
                if self.args.n_gpu > 1:
                    for val in logits.values:
                        if preds is None:
                            preds = val.numpy()
                        else:
                            preds = np.append(preds, val.numpy(), axis=0)

                    for val in labels.values:
                        if label_ids is None:
                            label_ids = val.numpy()
                        else:
                            label_ids = np.append(label_ids, val.numpy(), axis=0)
                else:
                    if preds is None:
                        preds = logits.numpy()
                    else:
                        preds = np.append(preds, logits.numpy(), axis=0)

                    if label_ids is None:
                        label_ids = labels.numpy()
                    else:
                        label_ids = np.append(label_ids, labels.numpy(), axis=0)

            step += 1

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        metrics["loss"] = loss.numpy()

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def evaluate(
        self, eval_dataset: Optional[tf.data.Dataset] = None, prediction_loss_only: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        """
        if eval_dataset is None:
            eval_dataset = self.tf_eval_dataset

        output = self._prediction_loop(eval_dataset, description="Evaluation")

        return output.metrics

    def train(self) -> None:
        """
        Train method to train the model.
        """
        tf.summary.trace_on(graph=True, profiler=True)
        self.gradient_accumulator.reset()

        iterations = self.optimizer.iterations
        tf.summary.experimental.set_step(iterations)

        epochs = 1 if self.args.max_steps > 0 else self.args.num_train_epochs

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total optimization steps = %d", self.train_steps)

        for epoch in range(1, int(epochs + 1)):
            for training_loss in self._training_steps():
                step = iterations.numpy()
                training_loss = tf.reduce_mean(training_loss)

                with self.writer.as_default():
                    tf.summary.scalar("loss", training_loss, step=step)

                if step == 1:
                    with self.writer.as_default():
                        tf.summary.trace_export(name="training", step=step, profiler_outdir=self.args.logging_dir)

                if step % self.args.logging_steps == 0:
                    logger.info("Epoch {} Step {} Train Loss {:.4f}".format(epoch, step, training_loss.numpy()))

                    if self.args.evaluate_during_training and step % self.args.eval_steps == 0:
                        logs = {}
                        results = self.evaluate()

                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                        if callable(self.optimizer.learning_rate):
                            logs["learning_rate"] = self.optimizer.learning_rate(step).numpy()
                        else:
                            logs["learning_rate"] = self.optimizer.learning_rate.numpy()

                        logger.info("Epoch {} Step {} Validation Metrics {}".format(epoch, step, logs))

                        with self.writer.as_default():
                            for k, v in logs.items():
                                tf.summary.scalar(k, v, step=step)

                if step % self.args.save_steps == 0:
                    ckpt_save_path = self.model.ckpt_manager.save()
                    logger.info("Saving checkpoint for step {} at {}".format(step, ckpt_save_path))

                if step % self.train_steps == 0:
                    break

    def _training_steps(self):
        """
        Returns a generator over training steps (i.e. parameters update).
        """
        for i, loss in enumerate(self._accumulate_next_gradients()):
            if i % self.args.gradient_accumulation_steps == 0:
                self._apply_gradients()
                yield loss

    @tf.function
    def _apply_gradients(self):
        """Applies the gradients (cross-replica)."""
        self.args.strategy.experimental_run_v2(self._step)

    def _step(self):
        """Applies gradients and resets accumulation."""
        gradient_scale = self.gradient_accumulator.step * self.args.strategy.num_replicas_in_sync
        gradients = [
            gradient / tf.cast(gradient_scale, gradient.dtype) for gradient in self.gradient_accumulator.gradients
        ]
        gradients = [(tf.clip_by_value(grad, -self.args.max_grad_norm, self.args.max_grad_norm)) for grad in gradients]
        vars = self.model.trainable_variables

        if self.args.mode == "token-classification":
            vars = [var for var in self.model.trainable_variables if "pooler" not in var.name]

        self.optimizer.apply_gradients(list(zip(gradients, vars)))
        self.gradient_accumulator.reset()

    def _accumulate_next_gradients(self):
        """Accumulates the gradients from the next element in dataset."""
        iterator = iter(self.tf_train_dataset)

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
        per_replica_loss = self.args.strategy.experimental_run_v2(
            self._forward, args=(per_replica_features, per_replica_labels)
        )

        return self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=0)

    def _forward(self, features, labels):
        """Forwards a training example and accumulates the gradients."""
        per_example_loss, _ = self._run_model(features, labels, True)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.args.train_batch_size)
        vars = self.model.trainable_variables

        if self.args.mode == "token-classification":
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
          training: run the model in training mode or not
        """
        if self.args.mode == "sequence-classification" or self.args.mode == "token-classification":
            logits = self.model(features, training=training)[0]
        else:
            logits = self.model(features, training=training)

        if self.args.mode == "token-classification":
            active_loss = tf.reshape(labels, (-1,)) != -1
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

        loss = self.loss(labels, reduced_logits)

        return loss, logits

    def predict(self, test_dataset: TFDataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        Args:
          test_dataset: something similar to a PT Dataset. This is just
            temporary before to have a framework-agnostic approach for datasets.
        """
        tf_test_dataset = test_dataset.get_dataset().batch(self.args.eval_batch_size)
        tf_test_dataset = self.args.strategy.experimental_distribute_dataset(tf_test_dataset)

        return self._prediction_loop(tf_test_dataset, description="Prediction")

    def save_model(self) -> None:
        """
        Save the pretrained model and create a Tensorflow saved model.
        """
        logger.info("Saving model in {}".format(self.args.output_dir))

        path = os.path.join(self.args.output_dir, "saved_model")

        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(self.args.output_dir)
