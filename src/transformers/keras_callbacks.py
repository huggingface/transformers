import logging
import os
from pathlib import Path
from time import sleep
from typing import Callable, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from huggingface_hub import Repository

from . import IntervalStrategy, PreTrainedTokenizerBase
from .file_utils import get_full_repo_name
from .modelcard import TrainingSummary


logger = logging.getLogger(__name__)


class KerasMetricCallback(Callback):
    """
    Callback to prompt metrics at the end of every epoch.

    Args:
        metric_fn: Metric function provided by the user.
        eval_dataset: Validation data to be used to evaluate the model at
        the end of the epoch.
        batch_size: Batch size.
        labels: Labels.
    """

    def __init__(
        self,
        metric_fn: Callable,
        tokenizer: PreTrainedTokenizerBase,
        eval_dataset: Union[tf.data.Dataset, np.ndarray, tf.Tensor, tuple, dict],
        output_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        predict_with_generate: Optional[bool] = False,
    ):
        super().__init__()
        self.metric_fn = metric_fn
        self.batch_size = batch_size
        if not isinstance(eval_dataset, tf.data.Dataset):
            if batch_size is None:
                raise ValueError(
                    "When passing data to KerasMetricCallback that is not a pre-batched tf.data.Dataset "
                    "the batch_size argument must be set."
                )
            # Wrap a tf.data.Dataset around it
            eval_dataset = tf.data.Dataset.from_tensor_slices(eval_dataset).batch(batch_size, drop_remainder=False)
        self.eval_dataset = eval_dataset
        self.predict_with_generate = predict_with_generate
        self.output_cols = output_cols
        self.model_input_names = tokenizer.model_input_names

        # This next block attempts to parse out which elements of the dataset should be appended to the labels list
        # that is passed to the metric_fn
        if isinstance(eval_dataset.element_spec, tuple) and len(eval_dataset.element_spec) == 2:
            input_spec, label_spec = eval_dataset.element_spec
        else:
            input_spec = eval_dataset.element_spec
            label_spec = None
        if label_cols is not None:
            for label in label_cols:
                if label not in input_spec:
                    raise ValueError(f"Label {label} is in label_cols but could not be found in the dataset inputs!")
            self.label_cols = label_cols
            self.use_keras_label = False
        elif label_spec is not None:
            # If the dataset inputs are split into a 2-tuple of inputs and labels,
            # assume the second element is the labels
            self.label_cols = None
            self.use_keras_label = True
        elif "labels" in input_spec:
            self.label_cols = ["labels"]
            self.use_keras_label = False
            logging.warning("No label_cols specified for KerasMetricCallback, assuming you want the 'labels' key.")
        else:
            raise ValueError("Could not autodetect label_cols for KerasMetricCallback, please specify them!")

    @staticmethod
    def _concatenate_batches(batches):
        # Flattens Numpy array batches into a list of single samples, where each sample is still np.ndarray
        return [sample for batch in batches for sample in batch]

    def _postprocess_predictions_or_labels(self, inputs):
        if isinstance(inputs[0], dict):
            outputs = dict()
            for key in inputs[0].keys():
                outputs[key] = self._concatenate_batches(batch[key] for batch in inputs)
        elif isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
            outputs = []
            for input_list in zip(*inputs):
                outputs.append(self._concatenate_batches(input_list))
        elif isinstance(inputs[0], np.ndarray):
            outputs = self._concatenate_batches(inputs)
        else:
            raise TypeError(f"Couldn't handle batch of type {type(inputs[0])}!")
        return outputs

    def on_epoch_end(self, epoch, logs=None):
        prediction_list = []
        label_list = []

        # The whole predict/generate loop is handled inside this method
        for batch in self.eval_dataset:
            if isinstance(batch, tuple):
                batch, labels = batch
            else:
                labels = None
            if isinstance(batch, dict):
                batch = {key: array for key, array in batch.items() if key in self.model_input_names}
            if self.predict_with_generate:
                predictions = self.model.generate(batch)
            else:
                predictions = self.model.predict(batch)
            predictions = dict(predictions)
            if self.output_cols is not None:
                predictions = {key: predictions[key] for key in self.output_cols}
            prediction_list.append(predictions)
            if not self.use_keras_label:
                labels = {key: batch[key].numpy() for key in self.label_cols}
            elif isinstance(labels, dict):
                labels = {key: array.numpy() for key, array in labels.items()}
            elif isinstance(labels, list) or isinstance(labels, tuple):
                labels = [array.numpy() for array in labels]
            elif isinstance(labels, tf.Tensor):
                labels = labels.numpy()
            else:
                raise TypeError(f"Confused by labels of type {type(labels)}")
            label_list.append(labels)

        prediction_list = self._postprocess_predictions_or_labels(prediction_list)
        label_list = self._postprocess_predictions_or_labels(label_list)

        metric_output = self.metric_fn(prediction_list, label_list)
        if not isinstance(metric_output, dict):
            raise TypeError(
                f"metric_fn should return a dict mapping metric names to values but instead returned {metric_output}"
            )
        # This is the critical bit - Keras passes a dict containing the loss and standard metric values for this epoch
        # in the logs argument. Ordinarily, this is so the callback can read them, but in this case we write a bunch of
        # new keys in there, which will then get read by the History callback and treated like any other metric value.
        # I promise that I have it in writing from Chollet that this is okay.
        logs.update(metric_output)


class PushToHubCallback(Callback):
    def __init__(
        self,
        output_dir: Union[str, Path],
        save_strategy: Union[str, IntervalStrategy] = "epoch",
        save_steps: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        checkpoint: bool = False,
        **model_card_args
    ):
        """
        Callback for pushing the model to the Hub after training.

        Args:
            output_dir (:obj:`str`):
                The output directory where the model predictions and checkpoints will be written and synced with the
                repository on the Hub.
            save_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"epoch"`):
                The checkpoint save strategy to adopt during training. Possible values are:

                    * :obj:`"no"`: No save is done during training.
                    * :obj:`"epoch"`: Save is done at the end of each epoch.
                    * :obj:`"steps"`: Save is done every :obj:`save_steps`
            save_steps (:obj:`int`, `optional`):
                The number of steps between saves when using the "steps" save_strategy.
            tokenizer (:obj:`PreTrainedTokenizerBase`, `optional`):
                The tokenizer used by the model. If supplied, will be uploaded to the repo alongside the weights.
            hub_model_id (:obj:`str`, `optional`):
                The name of the repository to keep in sync with the local `output_dir`. It can be a simple model ID in
                which case the model will be pushed in your namespace. Otherwise it should be the whole repository
                name, for instance :obj:`"user_name/model"`, which allows you to push to an organization you are a
                member of with :obj:`"organization_name/model"`.

                Will default to to the name of :obj:`output_dir`.
            hub_token (:obj:`str`, `optional`):
                The token to use to push the model to the Hub. Will default to the token in the cache folder obtained
                with :obj:`huggingface-cli login`.
            checkpoint (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to save full training checkpoints (including epoch and optimizer state) to allow training to be
                resumed. Only usable when `save_strategy` is `epoch`.
        """
        super().__init__()
        if checkpoint and save_strategy != "epoch":
            raise ValueError("Cannot save checkpoints when save_strategy is not 'epoch'!")
        if isinstance(save_strategy, str):
            save_strategy = IntervalStrategy(save_strategy.lower())
        self.save_strategy = save_strategy
        if self.save_strategy == IntervalStrategy.STEPS and (not isinstance(save_steps, int) or save_steps <= 0):
            raise ValueError("Please supply a positive integer argument for save_steps when save_strategy == 'steps'!")
        self.save_steps = save_steps
        output_dir = Path(output_dir)
        if hub_model_id is None:
            hub_model_id = output_dir.absolute().name
        if "/" not in hub_model_id:
            hub_model_id = get_full_repo_name(hub_model_id, token=hub_token)

        self.output_dir = output_dir
        self.hub_model_id = hub_model_id
        self.repo = Repository(
            str(self.output_dir),
            clone_from=self.hub_model_id,
            use_auth_token=hub_token if hub_token else True,
        )
        self.tokenizer = tokenizer
        self.last_job = None
        self.checkpoint = checkpoint
        self.training_history = None
        self.model_card_args = model_card_args

    def on_train_begin(self, logs=None):
        # Although we can access model.history, we have no guarantees that the History callback will fire before this
        # one, so we keep track of it here too
        self.training_history = []

    def on_train_batch_end(self, batch, logs=None):
        if self.save_strategy == IntervalStrategy.STEPS and batch + 1 % self.save_steps == 0:
            if self.last_job is not None and not self.last_job.is_done:
                return  # The last upload is still running, don't start another
            self.model.save_pretrained(self.output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress steps {batch}", blocking=False
            )

    def on_epoch_end(self, epoch, logs=None):
        if "epoch" not in logs:
            logs["epoch"] = epoch
        self.training_history.append(logs)
        if self.save_strategy == IntervalStrategy.EPOCH:
            if self.last_job is not None and not self.last_job.is_done:
                return  # The last upload is still running, don't start another
            self.model.save_pretrained(self.output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
            if self.checkpoint:
                checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
                self.model._save_checkpoint(checkpoint_dir, epoch)
            train_summary = TrainingSummary.from_keras(
                model=self.model,
                model_name=self.hub_model_id,
                keras_history=self.training_history,
                **self.model_card_args,
            )
            model_card = train_summary.to_model_card()
            with (self.output_dir / "README.md").open("w") as f:
                f.write(model_card)
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )

    def on_train_end(self, logs=None):
        if self.last_job is not None and not self.last_job.is_done:
            logger.info("Waiting for existing upload to finish...")
            while not self.last_job.is_done:
                sleep(1)
        self.model.save_pretrained(self.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.output_dir)
        train_summary = TrainingSummary.from_keras(
            model=self.model, model_name=self.hub_model_id, keras_history=self.training_history, **self.model_card_args
        )
        model_card = train_summary.to_model_card()
        with (self.output_dir / "README.md").open("w") as f:
            f.write(model_card)
        self.repo.push_to_hub(commit_message="End of training", blocking=True)
