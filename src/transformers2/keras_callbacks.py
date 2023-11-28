import logging
import os
from pathlib import Path
from time import sleep
from typing import Callable, List, Optional, Union

import numpy as np
import tensorflow as tf
from huggingface_hub import Repository, create_repo
from packaging.version import parse
from tensorflow.keras.callbacks import Callback

from . import IntervalStrategy, PreTrainedTokenizerBase
from .modelcard import TrainingSummary


logger = logging.getLogger(__name__)


class KerasMetricCallback(Callback):
    """
    Callback to compute metrics at the end of every epoch. Unlike normal Keras metrics, these do not need to be
    compilable by TF. It is particularly useful for common NLP metrics like BLEU and ROUGE that require string
    operations or generation loops that cannot be compiled. Predictions (or generations) will be computed on the
    `eval_dataset` before being passed to the `metric_fn` in `np.ndarray` format. The `metric_fn` should compute
    metrics and return a dict mapping metric names to metric values.

    We provide an example of a suitable metric_fn that computes ROUGE scores for a summarization model below. Note that
    this example skips some post-processing for readability and simplicity, and should probably not be used as-is!

    ```py
    from datasets import load_metric

    rouge_metric = load_metric("rouge")


    def rouge_fn(predictions, labels):
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge_metric.compute(predictions=decoded_predictions, references=decoded_labels)
        return {key: value.mid.fmeasure * 100 for key, value in result.items()}
    ```

    The above function will return a dict containing values which will be logged like any other Keras metric:

    ```
    {'rouge1': 37.4199, 'rouge2': 13.9768, 'rougeL': 34.361, 'rougeLsum': 35.0781
    ```

    Args:
        metric_fn (`Callable`):
            Metric function provided by the user. It will be called with two arguments - `predictions` and `labels`.
            These contain the model's outputs and matching labels from the dataset. It should return a dict mapping
            metric names to numerical values.
        eval_dataset (`tf.data.Dataset` or `dict` or `tuple` or `np.ndarray` or `tf.Tensor`):
            Validation data to be used to generate predictions for the `metric_fn`.
        output_cols (`List[str], *optional*):
            A list of columns to be retained from the model output as the predictions. Defaults to all.
        label_cols ('`List[str]`, *optional*'):
            A list of columns to be retained from the input dataset as the labels. Will be autodetected if this is not
            supplied.
        batch_size (`int`, *optional*):
            Batch size. Only used when the data is not a pre-batched `tf.data.Dataset`.
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            Whether we should use `model.generate()` to get outputs for the model.
        use_xla_generation (`bool`, *optional*, defaults to `False`):
            If we're generating, whether to compile model generation with XLA. This can massively increase the speed of
            generation (up to 100X speedup) but will require a new XLA compilation for each input shape. When using XLA
            generation, it's a good idea to pad your inputs to the same size, or to use the `pad_to_multiple_of`
            argument in your `tokenizer` or `DataCollator`, which will reduce the number of unique input shapes and
            save a lot of compilation time. This option has no effect is `predict_with_generate` is `False`.
        generate_kwargs (`dict`, *optional*):
            Keyword arguments to pass to `model.generate()` when generating. Has no effect if `predict_with_generate`
            is `False`.

    """

    def __init__(
        self,
        metric_fn: Callable,
        eval_dataset: Union[tf.data.Dataset, np.ndarray, tf.Tensor, tuple, dict],
        output_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        predict_with_generate: bool = False,
        use_xla_generation: bool = False,
        generate_kwargs: Optional[dict] = None,
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
        elif "start_positions" in input_spec and "end_positions" in input_spec:
            self.label_cols = ["start_positions", "end_positions"]
            self.use_keras_label = False
            logging.warning(
                "No label_cols specified for KerasMetricCallback, assuming you want the "
                "start_positions and end_positions keys."
            )
        else:
            raise ValueError("Could not autodetect label_cols for KerasMetricCallback, please specify them!")
        if parse(tf.__version__) < parse("2.7"):
            logging.warning("TF versions less than 2.7 may encounter issues with KerasMetricCallback!")

        self.use_xla_generation = use_xla_generation
        self.generate_kwargs = {} if generate_kwargs is None else generate_kwargs

        self.generation_function = None

    @staticmethod
    def _concatenate_batches(batches, padding_index=-100):
        # If all batches are unidimensional or same length, do a simple concatenation
        if batches[0].ndim == 1 or all(batch.shape[1] == batches[0].shape[1] for batch in batches):
            return np.concatenate(batches, axis=0)

        # Welp, they're not the same length. Let's do some padding
        max_len = max([batch.shape[1] for batch in batches])
        num_samples = sum([batch.shape[0] for batch in batches])
        output = np.full_like(
            batches[0], fill_value=padding_index, shape=[num_samples, max_len] + list(batches[0].shape[2:])
        )
        # i keeps track of which part of the concatenated array we're writing the next batch to
        i = 0
        for batch in batches:
            output[i : i + len(batch), : batch.shape[1]] = batch
            i += len(batch)
        return output

    def _postprocess_predictions_or_labels(self, inputs):
        if isinstance(inputs[0], dict):
            outputs = {}
            for key in inputs[0].keys():
                outputs[key] = self._concatenate_batches([batch[key] for batch in inputs])
            # If it's a dict with only one key, just return the array
            if len(outputs) == 1:
                outputs = list(outputs.values())[0]
        elif isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
            outputs = []
            for input_list in zip(*inputs):
                outputs.append(self._concatenate_batches(input_list))
            if len(outputs) == 1:
                outputs = outputs[0]  # If it's a list with only one element, just return the array
        elif isinstance(inputs[0], np.ndarray):
            outputs = self._concatenate_batches(inputs)
        elif isinstance(inputs[0], tf.Tensor):
            outputs = self._concatenate_batches([tensor.numpy() for tensor in inputs])
        else:
            raise TypeError(f"Couldn't handle batch of type {type(inputs[0])}!")
        return outputs

    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.model, "config"):
            ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
        else:
            ignore_keys = []

        main_input_name = None
        if self.predict_with_generate:
            # This dense conditional recognizes the case where we have an encoder-decoder model, but
            # avoids getting tangled up when we just have a model with a layer called 'encoder'
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "main_input_name"):
                main_input_name = self.model.encoder.main_input_name
            else:
                main_input_name = getattr(self.model, "main_input_name", "input_ids")

            if self.use_xla_generation and self.generation_function is None:

                def generation_function(inputs, attention_mask):
                    return self.model.generate(inputs, attention_mask=attention_mask, **self.generate_kwargs)

                self.generation_function = tf.function(generation_function, jit_compile=True)

        prediction_list = []
        label_list = []

        # The whole predict/generate loop is handled inside this method
        for batch in self.eval_dataset:
            if isinstance(batch, tuple):
                batch, labels = batch
            else:
                labels = None
            if self.predict_with_generate:
                if isinstance(batch, dict):
                    generation_inputs = batch[main_input_name]
                    attention_mask = batch.get("attention_mask", None)
                else:
                    generation_inputs = batch
                    attention_mask = None
                if self.use_xla_generation:
                    predictions = self.generation_function(generation_inputs, attention_mask=attention_mask)
                else:
                    predictions = self.model.generate(
                        generation_inputs, attention_mask=attention_mask, **self.generate_kwargs
                    )
            else:
                predictions = self.model.predict_on_batch(batch)
                if isinstance(predictions, dict):
                    # This converts any dict-subclass to a regular dict
                    # Keras REALLY doesn't like it when we pass around a BatchEncoding or other derived class
                    predictions = dict(predictions)
                    if self.output_cols is not None:
                        predictions = {key: predictions[key] for key in self.output_cols}
                    else:
                        predictions = {
                            key: val for key, val in predictions.items() if key not in ignore_keys + ["loss"]
                        }
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

        all_preds = self._postprocess_predictions_or_labels(prediction_list)
        all_labels = self._postprocess_predictions_or_labels(label_list)

        metric_output = self.metric_fn((all_preds, all_labels))
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
    """
    Callback that will save and push the model to the Hub regularly. By default, it pushes once per epoch, but this can
    be changed with the `save_strategy` argument. Pushed models can be accessed like any other model on the hub, such
    as with the `from_pretrained` method.

    ```py
    from transformers.keras_callbacks import PushToHubCallback

    push_to_hub_callback = PushToHubCallback(
        output_dir="./model_save",
        tokenizer=tokenizer,
        hub_model_id="gpt5-7xlarge",
    )

    model.fit(train_dataset, callbacks=[push_to_hub_callback])
    ```

    Args:
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written and synced with the
            repository on the Hub.
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"epoch"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: Save is done at the end of training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`
        save_steps (`int`, *optional*):
            The number of steps between saves when using the "steps" `save_strategy`.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            The tokenizer used by the model. If supplied, will be uploaded to the repo alongside the weights.
        hub_model_id (`str`, *optional*):
            The name of the repository to keep in sync with the local `output_dir`. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
            for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
            `"organization_name/model"`.

            Will default to the name of `output_dir`.
        hub_token (`str`, *optional*):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            `huggingface-cli login`.
        checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to save full training checkpoints (including epoch and optimizer state) to allow training to be
            resumed. Only usable when `save_strategy` is `"epoch"`.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        save_strategy: Union[str, IntervalStrategy] = "epoch",
        save_steps: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        checkpoint: bool = False,
        **model_card_args,
    ):
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

        # Create repo and retrieve repo_id
        if hub_model_id is None:
            hub_model_id = output_dir.absolute().name
        self.hub_model_id = create_repo(repo_id=hub_model_id, exist_ok=True, token=hub_token).repo_id

        self.output_dir = output_dir
        self.repo = Repository(str(self.output_dir), clone_from=self.hub_model_id, token=hub_token)

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
        if self.save_strategy == IntervalStrategy.STEPS and (batch + 1) % self.save_steps == 0:
            if self.last_job is not None and not self.last_job.is_done:
                return  # The last upload is still running, don't start another
            self.model.save_pretrained(self.output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress steps {batch}", blocking=False
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs.copy()  # Don't accidentally write things that Keras will read later
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
        # Makes sure the latest version of the model is uploaded
        if self.last_job is not None and not self.last_job.is_done:
            logging.info("Pushing the last epoch to the Hub, this may take a while...")
            while not self.last_job.is_done:
                sleep(1)
        else:
            self.model.save_pretrained(self.output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
            train_summary = TrainingSummary.from_keras(
                model=self.model,
                model_name=self.hub_model_id,
                keras_history=self.training_history,
                **self.model_card_args,
            )
            model_card = train_summary.to_model_card()
            with (self.output_dir / "README.md").open("w") as f:
                f.write(model_card)
            self.repo.push_to_hub(commit_message="End of training", blocking=True)
