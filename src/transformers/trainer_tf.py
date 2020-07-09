"""Tensorflow trainer class."""

import logging
import math
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .modeling_tf_utils import TFPreTrainedModel
from .optimization_tf import GradientAccumulator, create_optimizer
from .trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, is_wandb_available, set_seed
from .training_args_tf import TFTrainingArguments


if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)


class TFTrainer:
    """
    TFTrainer is a simple but feature-complete training and eval loop for TensorFlow,
    optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.TFPreTrainedModel`):
            The model to train, evaluate or use for predictions.
        args (:class:`~transformers.TFTrainingArguments`):
            The arguments to tweak training.
        train_dataset (:class:`~tf.data.Dataset`, `optional`):
            The dataset to use for training.
        eval_dataset (:class:`~tf.data.Dataset`, `optional`):
            The dataset to use for evaluation.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
            When performing evaluation and predictions, only returns the loss.
        tb_writer (:obj:`tf.summary.SummaryWriter`, `optional`):
            Object to write to TensorBoard.
        optimizers (:obj:`Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule]`, `optional`):
            A tuple containing the optimizer and the scheduler to use. The optimizer default to an instance of
            :class:`tf.keras.optimizers.Adam` if :obj:`args.weight_decay_rate` is 0 else an instance of
            :class:`~transformers.AdamWeightDecay`. The scheduler will default to an instance of
            :class:`tf.keras.optimizers.schedules.PolynomialDecay` if :obj:`args.num_warmup_steps` is 0 else
            an instance of :class:`~transformers.WarmUp`.
    """

    model: TFPreTrainedModel
    args: TFTrainingArguments
    train_dataset: Optional[tf.data.Dataset]
    eval_dataset: Optional[tf.data.Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional[tf.summary.SummaryWriter] = None
    optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = None
    global_step: Optional[int] = None
    epoch_logging: Optional[float] = None

    def __init__(
        self,
        model: TFPreTrainedModel,
        args: TFTrainingArguments,
        train_dataset: Optional[tf.data.Dataset] = None,
        eval_dataset: Optional[tf.data.Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional[tf.summary.SummaryWriter] = None,
        optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        self.gradient_accumulator = GradientAccumulator()
        self.global_step = 0
        self.epoch_logging = 0

        if tb_writer is not None:
            self.tb_writer = tb_writer
        else:
            self.tb_writer = tf.summary.create_file_writer(self.args.logging_dir)

        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )

        set_seed(self.args.seed)

    def get_train_tfdataset(self) -> tf.data.Dataset:
        """
        Returns the training :class:`~tf.data.Dataset`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        self.num_train_examples = self.train_dataset.reduce(tf.constant(0), lambda x, _: x + 1).numpy()

        if self.args.max_steps > 0:
            self.train_steps = self.args.max_steps
        else:
            self.train_steps: int = math.ceil(self.num_train_examples / self.args.train_batch_size)

        ds = (
            self.train_dataset.cache()
            .shuffle(self.num_train_examples)
            .batch(self.args.train_batch_size, drop_remainder=self.args.dataloader_drop_last)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        if self.args.max_steps > 0:
            self.train_dataset = self.train_dataset.repeat(-1)

        return self.args.strategy.experimental_distribute_dataset(ds)

    def get_eval_tfdataset(self, eval_dataset: Optional[tf.data.Dataset] = None) -> tf.data.Dataset:
        """
        Returns the evaluation :class:`~tf.data.Dataset`.

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                If provided, will override `self.eval_dataset`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        ds = (
            eval_dataset.cache()
            .batch(self.args.eval_batch_size, drop_remainder=self.args.dataloader_drop_last)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return self.args.strategy.experimental_distribute_dataset(ds)

    def get_test_tfdataset(self, test_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Returns a test :class:`~tf.data.Dataset`.

        Args:
            test_dataset (:class:`~tf.data.Dataset`): The dataset to use.
        """
        ds = test_dataset.batch(self.args.eval_batch_size, drop_remainder=self.args.dataloader_drop_last)

        return self.args.strategy.experimental_distribute_dataset(ds)

    def get_optimizers(
        self, num_training_steps: int,
    ) -> Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        TFTrainer's init through :obj:`optimizers`, or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers

        optimizer, scheduler = create_optimizer(
            self.args.learning_rate,
            num_training_steps,
            self.args.warmup_steps,
            adam_epsilon=self.args.adam_epsilon,
            weight_decay_rate=self.args.weight_decay,
        )

        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))

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

        try:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=0)
        except ValueError:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

        return reduced_loss, per_replica_logits

    def _prediction_loop(
        self, dataset: tf.data.Dataset, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        logger.info("***** Running %s *****", description)
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        label_ids: np.ndarray = None
        preds: np.ndarray = None

        step: int = 1

        # Reset the past mems state at the beginning of the evaluation if necessary.
        if self.args.past_index >= 0:
            self._past = None

        for features, labels in dataset:
            step = tf.convert_to_tensor(step, dtype=tf.int64)
            loss, logits = self._evaluate_steps(features, labels)
            loss = tf.reduce_mean(loss)

            if not prediction_loss_only:
                if isinstance(logits, tuple):
                    logits = logits[0]

                if isinstance(labels, tuple):
                    labels = labels[0]

                if self.args.n_replicas > 1:
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

        metrics["eval_loss"] = loss.numpy()

        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def _log(self, logs: Dict[str, float]) -> None:
        logs["epoch"] = self.epoch_logging

        if self.tb_writer:
            with self.tb_writer.as_default():
                for k, v in logs.items():
                    tf.summary.scalar(k, v, step=self.global_step)
            self.tb_writer.flush()

        if is_wandb_available():
            wandb.log(logs, step=self.global_step)

        output = {**logs, **{"step": self.global_step}}

        logger.info(output)

    def evaluate(self, eval_dataset: Optional[tf.data.Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        eval_ds = self.get_eval_tfdataset(eval_dataset)

        output = self._prediction_loop(eval_ds, description="Evaluation")

        logs = {**output.metrics}
        logs["epoch"] = self.epoch_logging

        self._log(logs)

        return output.metrics

    def train(self) -> None:
        """
        Train method to train the model.
        """
        train_ds = self.get_train_tfdataset()

        if self.args.debug:
            tf.summary.trace_on(graph=True, profiler=True)

        self.gradient_accumulator.reset()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            steps_per_epoch = self.args.max_steps
        else:
            if self.args.dataloader_drop_last:
                approx = math.floor
            else:
                approx = math.ceil

            steps_per_epoch = approx(
                self.num_train_examples / (self.args.train_batch_size * self.args.gradient_accumulation_steps)
            )
            t_total = steps_per_epoch * self.args.num_train_epochs

        with self.args.strategy.scope():
            optimizer, lr_scheduler = self.get_optimizers(num_training_steps=t_total)
            iterations = optimizer.iterations
            self.global_step = iterations.numpy()
            folder = os.path.join(self.args.output_dir, PREFIX_CHECKPOINT_DIR)
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=self.model)
            self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=self.args.save_total_limit)

            if self.model.ckpt_manager.latest_checkpoint:
                epochs_trained = self.global_step // (self.num_train_examples // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    self.num_train_examples // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
                logger.info(
                    "Checkpoint file %s found and restoring from checkpoint", self.model.ckpt_manager.latest_checkpoint
                )

                ckpt.restore(self.model.ckpt_manager.latest_checkpoint).expect_partial()
            else:
                epochs_trained = 1

        tf.summary.experimental.set_step(iterations)

        epochs = 1 if self.args.max_steps > 0 else self.args.num_train_epochs

        if self.args.fp16:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
            tf.keras.mixed_precision.experimental.set_policy(policy)

        with self.tb_writer.as_default():
            tf.summary.text("args", self.args.to_json_string())

        self.tb_writer.flush()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_train_examples)
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d", self.args.train_batch_size
        )
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        for epoch_iter in range(epochs_trained, int(epochs + 1)):
            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None
            for step, training_loss in enumerate(self._training_steps(train_ds, optimizer)):
                self.global_step = iterations.numpy()
                self.epoch_logging = epoch_iter - 1 + (step + 1) / steps_per_epoch

                if self.args.debug:
                    logs = {}
                    logs["loss"] = training_loss.numpy()
                    logs["epoch"] = self.epoch_logging

                    self._log(logs)

                if self.global_step == 1 and self.args.debug:
                    with self.tb_writer.as_default():
                        tf.summary.trace_export(
                            name="training", step=self.global_step, profiler_outdir=self.args.logging_dir
                        )

                if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                    self.evaluate()

                if (
                    self.global_step % self.args.logging_steps == 0
                    or self.global_step == 1
                    and self.args.logging_first_step
                ):
                    logs = {}
                    logs["loss"] = training_loss.numpy()
                    logs["learning_rate"] = lr_scheduler(self.global_step).numpy()
                    logs["epoch"] = self.epoch_logging

                    self._log(logs)

                if self.global_step % self.args.save_steps == 0:
                    ckpt_save_path = self.model.ckpt_manager.save()
                    logger.info("Saving checkpoint for step {} at {}".format(self.global_step, ckpt_save_path))

                if self.args.max_steps > 0 and self.global_step % self.args.max_steps == 0:
                    break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

    def _training_steps(self, ds, optimizer):
        """
        Returns a generator over training steps (i.e. parameters update).
        """
        for i, loss in enumerate(self._accumulate_next_gradients(ds)):
            if i % self.args.gradient_accumulation_steps == 0:
                self._apply_gradients(optimizer)
                yield loss

    @tf.function
    def _apply_gradients(self, optimizer):
        """Applies the gradients (cross-replica)."""
        self.args.strategy.experimental_run_v2(self._step, args=(optimizer,))

    def _step(self, optimizer):
        """Applies gradients and resets accumulation."""
        gradient_scale = self.gradient_accumulator.step * self.args.strategy.num_replicas_in_sync
        gradients = [
            gradient / tf.cast(gradient_scale, gradient.dtype) for gradient in self.gradient_accumulator.gradients
        ]
        gradients = [(tf.clip_by_value(grad, -self.args.max_grad_norm, self.args.max_grad_norm)) for grad in gradients]

        optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))
        self.gradient_accumulator.reset()

    def _accumulate_next_gradients(self, ds):
        """Accumulates the gradients from the next element in dataset."""
        iterator = iter(ds)

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

        try:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=0)
        except ValueError:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

        return reduced_loss

    def _forward(self, features, labels):
        """Forwards a training example and accumulates the gradients."""
        per_example_loss, _ = self._run_model(features, labels, True)
        gradients = tf.gradients(per_example_loss, self.model.trainable_variables)
        gradients = [
            g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.model.trainable_variables)
        ]

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
        if self.args.past_index >= 0 and getattr(self, "_past", None) is not None:
            features["mems"] = self._past
        if isinstance(labels, (dict)):
            outputs = self.model(features, training=training, **labels)[:2]
        else:
            outputs = self.model(features, labels=labels, training=training)[:2]
        loss, logits = outputs[:2]
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        loss += sum(self.model.losses) * (1.0 / self.args.n_replicas)

        return loss, logits

    def predict(self, test_dataset: tf.data.Dataset) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:class:`~tf.data.Dataset`):
                Dataset to run the predictions on.
        Returns:
            `NamedTuple`:
            predictions (:obj:`np.ndarray`):
                The predictions on :obj:`test_dataset`.
            label_ids (:obj:`np.ndarray`, `optional`):
                The labels (if the dataset contained some).
            metrics (:obj:`Dict[str, float]`, `optional`):
                The potential dictionary of metrics (if the dataset contained labels).
        """
        test_ds = self.get_test_tfdataset(test_dataset)

        return self._prediction_loop(test_ds, description="Prediction")

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        logger.info("Saving model in {}".format(output_dir))

        if not isinstance(self.model, TFPreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        self.model.save_pretrained(self.args.output_dir)
