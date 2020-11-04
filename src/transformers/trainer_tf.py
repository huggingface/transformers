# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""Tensorflow trainer class."""

import math
import os
import sys
import warnings
from typing import Dict, List, Optional, Union

from .file_utils import ENV_VARS_TRUE_VALUES


# Integrations must be imported before ML frameworks:
from .integrations import (  # isort: split
    is_comet_available,
    is_wandb_available,
)

import numpy as np
import tensorflow as tf

from .modeling_tf_utils import TFPreTrainedModel
from .optimization_tf import create_optimizer
from .trainer_tf_callbacks import LearningRateLoggingCallback
from .trainer_utils import PREFIX_CHECKPOINT_DIR, PredictionOutput, set_seed
from .training_args_tf import TFTrainingArguments
from .utils import logging


if is_comet_available():
    import comet_ml

if is_wandb_available():
    import wandb
    from wandb.keras import WandbCallback

logger = logging.get_logger(__name__)


class TFTrainer:
    """
    TFTrainer is a simple but feature-complete training and eval loop for TensorFlow, optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.TFPreTrainedModel`):
            The model to train, evaluate or use for predictions.
        args (:class:`~transformers.TFTrainingArguments`):
            The arguments to tweak training.
        train_dataset (:class:`~tf.data.Dataset`, `optional`):
            The dataset to use for training. The dataset should yield tuples of ``(features, labels)`` where
            ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the loss
            is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict, such as
            when using a QuestionAnswering head model with multiple targets, the loss is instead calculated by calling
            ``model(features, **labels)``.
        eval_dataset (:class:`~tf.data.Dataset`, `optional`):
            The dataset to use for evaluation. The dataset should yield tuples of ``(features, labels)`` where
            ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the loss
            is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict, such as
            when using a QuestionAnswering head model with multiple targets, the loss is instead calculated by calling
            ``model(features, **labels)``.
        compute_metrics (:obj:`List[Union[tf.keras.metrics.Metric, str]]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        optimizers (:obj:`tf.keras.optimizers.Optimizer`, `optional`):
            The optimizer to use. The optimizer default to an instance of :class:`~transformers.AccumulationOptimizer`
            with a the following wrapped optimizer: :class:`tf.keras.optimizers.Adam` if :obj:`args.weight_decay_rate`
            is 0 else an instance of :class:`~transformers.AdamWeightDecay`. The scheduler will default to an instance
            of :class:`tf.keras.optimizers.schedules.PolynomialDecay` if :obj:`args.num_warmup_steps` is 0 else an
            instance of :class:`~transformers.WarmUp`.
    """

    def __init__(
        self,
        model: TFPreTrainedModel,
        args: TFTrainingArguments,
        train_dataset: Optional[tf.data.Dataset] = None,
        eval_dataset: Optional[tf.data.Dataset] = None,
        compute_metrics: Optional[List[Union[tf.keras.metrics.Metric, str]]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = [],
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        **kwargs,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizer = optimizer
        self.global_step = 0
        self.epoch_logging = 0
        self.eval_loss = tf.keras.metrics.Sum()

        if is_wandb_available():
            self.setup_wandb()
        elif os.getenv("WANDB_DISABLED", "").upper() not in ENV_VARS_TRUE_VALUES:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )

        if is_comet_available():
            self.setup_comet()
        elif os.environ.get("COMET_MODE") != "DISABLED":
            logger.info(
                "To use comet_ml logging, run `pip/conda install comet_ml` "
                "see https://www.comet.ml/docs/python-sdk/huggingface/"
            )

        set_seed(self.args.seed)

    def get_train_tfdataset(self) -> tf.data.Dataset:
        """
        Returns the training :class:`~tf.data.Dataset`.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # As the accumulation is done directly inside the optimizer we don't batch the dataset to "train_batch_size * gradient_accumulation_steps"
        # but to usual train_batch_size. The "gradient_accumulation_steps" value is used in the decay of the optimizer where we divide the total number
        # of steps by the "gradient_accumulation_steps" value.
        self.total_train_batch_size = self.args.train_batch_size
        self.num_train_examples = self.train_dataset.cardinality().numpy()

        if self.num_train_examples < 0:
            raise ValueError(
                "The training dataset must have an asserted cardinality. Use the tf.data.experimental.assert_cardinality method when creating your dataset."
            )

        ds = (
            self.train_dataset.repeat()
            .shuffle(self.num_train_examples, seed=self.args.seed)
            .batch(self.args.train_batch_size, drop_remainder=self.args.dataloader_drop_last)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return self.args.strategy.experimental_distribute_dataset(ds)

    def get_test_eval_tfdataset(self, test_eval_dataset: Optional[tf.data.Dataset] = None) -> tf.data.Dataset:
        """
        Returns a test :class:`~tf.data.Dataset`. If ``test_eval_dataset`` is empty with run the test/evaluation over
        ``self.eval_dataset``.

        Args:
            test_eval_dataset (:class:`~tf.data.Dataset`):
                The dataset to use. The dataset should yield tuples of ``(features, labels)`` where ``features`` is a
                dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the loss is calculated
                by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict, such as when using
                a QuestionAnswering head model with multiple targets, the loss is instead calculated by calling
                ``model(features, **labels)``.

        Subclass and override this method if you want to inject some custom behavior.
        """

        if test_eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation/test requires a dataset.")

        dataset = test_eval_dataset if test_eval_dataset is not None else self.eval_dataset
        num_examples = tf.data.experimental.cardinality(dataset).numpy()

        if num_examples < 0:
            raise ValueError("The test dataset must have an asserted cardinality")

        approx = math.floor if self.args.dataloader_drop_last else math.ceil
        steps = approx(num_examples / self.args.eval_batch_size)
        ds = (
            dataset.repeat()
            .batch(self.args.eval_batch_size, drop_remainder=self.args.dataloader_drop_last)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return self.args.strategy.experimental_distribute_dataset(ds), steps, num_examples

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        TFTrainer's init through :obj:`optimizers`, or subclass and override this method.
        """
        if not self.optimizer:
            self.optimizer = create_optimizer(
                self.args.learning_rate,
                num_training_steps,
                self.args.warmup_steps,
                adam_beta1=self.args.adam_beta1,
                adam_beta2=self.args.adam_beta2,
                adam_epsilon=self.args.adam_epsilon,
                weight_decay_rate=self.args.weight_decay,
                power=self.args.poly_power,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                max_grad_norm=self.args.max_grad_norm,
            )

    def setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.com/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different
                project.
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely.
        """

        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        combined_dict = {**self.model.config.to_dict(), **self.args.to_sanitized_dict()}
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=combined_dict, name=self.args.run_name)

    def setup_comet(self):
        """
        Setup the optional Comet.ml integration.

        Environment:
            COMET_MODE:
                (Optional): str - "OFFLINE", "ONLINE", or "DISABLED"
            COMET_PROJECT_NAME:
                (Optional): str - Comet.ml project name for experiments
            COMET_OFFLINE_DIRECTORY:
                (Optional): str - folder to use for saving offline experiments when `COMET_MODE` is "OFFLINE"

        For a number of configurable items in the environment, see `here
        <https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables>`__
        """
        comet_mode = os.getenv("COMET_MODE", "ONLINE").upper()
        args = {"project_name": os.getenv("COMET_PROJECT_NAME", "huggingface")}
        experiment = None
        if comet_mode == "ONLINE":
            experiment = comet_ml.Experiment(**args)
            logger.info("Automatic Comet.ml online logging enabled")
        elif comet_mode == "OFFLINE":
            args["offline_directory"] = os.getenv("COMET_OFFLINE_DIRECTORY", "./")
            experiment = comet_ml.OfflineExperiment(**args)
            logger.info("Automatic Comet.ml offline logging enabled; use `comet upload` when finished")
        if experiment is not None:
            experiment._log_parameters(self.args, prefix="args/")
            experiment._log_parameters(self.model.config, prefix="config/")

    def evaluate(self, eval_dataset: Optional[tf.data.Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. The dataset should yield tuples of
                ``(features, labels)`` where ``features`` is a dict of input features and ``labels`` is the labels. If
                ``labels`` is a tensor, the loss is calculated by the model by calling ``model(features,
                labels=labels)``. If ``labels`` is a dict, such as when using a QuestionAnswering head model with
                multiple targets, the loss is instead calculated by calling ``model(features, **labels)``.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        eval_ds, steps, num_examples = self.get_test_eval_tfdataset(eval_dataset)

        logger.info("***** Running Evaluate *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        return self.model.evaluate(eval_ds, steps=steps, return_dict=True)

    def train(self) -> None:
        """
        Train method to train the model.
        """
        train_ds = self.get_train_tfdataset()
        num_update_steps_per_epoch = self.num_train_examples / self.total_train_batch_size

        # In fact, ``self.args.dataloader_drop_last`` has no effect in `trainer_tf.py`, because
        # the dataset is repeated before being batched.
        # It has the effect only when TPU is used which requires explicit tensor shape in order to make
        # the gradient accumulation implementation work.
        approx = math.floor if self.args.dataloader_drop_last else math.ceil
        num_update_steps_per_epoch = approx(num_update_steps_per_epoch)

        # At least one update for each epoch.
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        self.steps_per_epoch = num_update_steps_per_epoch

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            epochs = (self.args.max_steps // self.steps_per_epoch) + int(
                self.args.max_steps % self.steps_per_epoch > 0
            )
        else:
            t_total = self.steps_per_epoch * self.args.num_train_epochs
            epochs = self.args.num_train_epochs

        # Since ``self.args.num_train_epochs`` can be `float`, we make ``epochs`` be a `int` always.
        epochs = int(epochs)

        with self.args.strategy.scope():
            self.create_optimizer_and_scheduler(num_training_steps=t_total)
            self.model.compile(
                loss=self.model.compute_loss,
                optimizer=self.optimizer,
                experimental_steps_per_execution=np.gcd.reduce([self.steps_per_epoch, self.args.logging_steps]),
                metrics=self.compute_metrics,
            )

            folder = os.path.join(self.args.output_dir, PREFIX_CHECKPOINT_DIR)
            epochs_trained = 0
            latest = tf.train.latest_checkpoint(folder)

            if latest is not None:
                logger.info("Checkpoint file %s found and restoring from checkpoint", latest)
                self.model.load_weights(latest)

                # Here we take the total number of iterations (with optimizer._iterations) and not the total number of updates (with optimizer.iterations)
                epochs_trained = self.optimizer._iterations // self.steps_per_epoch

                logger.info("  Continuing training from checkpoint, will skip to saved epoch")
                logger.info("  Continuing training from epoch %d", epochs_trained)

            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", self.num_train_examples)
            logger.info("  Num Epochs = %d", epochs)
            logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
            logger.info("  Total train batch size (w. parallel, distributed) = %d", self.total_train_batch_size)
            logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
            logger.info("  Steps per epoch = %d", self.steps_per_epoch)
            logger.info(
                "  Optimization steps per epoch (w. gradient accumulation) = %d",
                self.steps_per_epoch // self.args.gradient_accumulation_steps,
            )
            logger.info(
                "  Total optimization steps (w. gradient accumulation) = %d",
                t_total // self.args.gradient_accumulation_steps,
            )
            logger.info("  Total training steps = %d", t_total)

            eval_ds, steps, _ = self.get_test_eval_tfdataset()

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(folder, "weights.{epoch:04d}.ckpt"),
                save_weights_only=True,
            )

            self.callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir=self.args.logging_dir, update_freq=self.args.logging_steps)
            )
            self.callbacks.append(LearningRateLoggingCallback())
            self.callbacks.append(model_checkpoint_callback)

            if is_wandb_available():
                self.callbacks.append(WandbCallback(save_model=False, log_weights=True, log_batch_frequency=self.args.logging_steps))

            self.model.fit(
                train_ds,
                validation_data=eval_ds,
                epochs=epochs,
                initial_epoch=epochs_trained,
                validation_steps=steps,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=self.callbacks,
            )

    def predict(self, test_dataset: tf.data.Dataset) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:class:`~tf.data.Dataset`):
                Dataset to run the predictions on. The dataset should yield tuples of ``(features, labels)`` where
                ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the
                loss is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict,
                such as when using a QuestionAnswering head model with multiple targets, the loss is instead calculated
                by calling ``model(features, **labels)``

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        test_ds, steps, num_examples = self.get_test_eval_tfdataset(test_dataset)

        # return self.prediction_loop(test_ds, steps, num_examples, description="Prediction")

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        logger.info("Saving model in {}".format(output_dir))

        if not isinstance(self.model, TFPreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        self.model.save_pretrained(output_dir)
