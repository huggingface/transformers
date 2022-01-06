#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library vision-encoder-decoder models for image captioning.
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import Dataset, load_dataset, load_metric
from PIL import Image
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from filelock import FileLock
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    FlaxVisionEncoderDecoderModel,
    HfArgumentParser,
    is_tensorboard_available,
)
from transformers.file_utils import get_full_repo_name, is_offline_mode


logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    _block_size_doc = """
        The default value `0` will preprocess (tokenization + feature extraction) the whole dataset before training and
        cache the results. This uses more disk space, but avoids (repeated) processing time during training. This is a
        good option if your disk space is large enough to store the whole processed dataset.
        If a positive value is given, the captions in the dataset will be tokenized before training and the results are
        cached. During training, it iterates the dataset in chunks of size `block_size`. On each block, images are
        transformed by the feature extractor with the results being kept in memory (no cache), and batches of size
        `batch_size` are yielded before processing the next block. This could avoid the heavy disk usage when the
        dataset is large.
        """
    block_size: int = field(default=0, metadata={"help": _block_size_doc})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory of the dataset to use (via the datasets library)."}
    )
    image_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input predict data file to do prediction on (a text file)."},
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the `max_length` param of `model.generate`, which is used "
            "during evaluation."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`, "
            "which is used during evaluation."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


image_captioning_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    steps = len(dataset) // batch_size  # Skip incomplete batch.

    # We use `numpy.ndarray` to interact with `datasets.Dataset`, since using `jax.numpy.array` to index into a
    # dataset is significantly slow. Using JAX array at the 1st place is only to keep JAX's PRNGs generation
    # mechanism, which works differently from NumPy/SciPy.
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    for idx in range(steps):

        start_idx = batch_size * idx
        end_idx = batch_size * (idx + 1)

        selected_indices = batch_idx[start_idx:end_idx]
        batch = dataset[selected_indices]
        batch = shard(batch)

        yield batch


def write_metric(summary_writer, metrics, train_time, step, metric_key_prefix="train"):

    if train_time:
        summary_writer.scalar("train_time", train_time, step)

        metrics = get_metrics(metrics)
        for key, vals in metrics.items():
            tag = f"{metric_key_prefix}_{key}"
            for i, val in enumerate(vals):
                summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    else:
        for metric_name, value in metrics.items():
            summary_writer.scalar(f"{metric_key_prefix}_{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name, token=training_args.hub_token
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    model = FlaxVisionEncoderDecoderModel.from_pretrained(
        model_args.model_name_or_path,
        seed=training_args.seed,
        dtype=getattr(jnp, model_args.dtype),
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
    )
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(model.config.pad_token_id)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = image_captioning_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        assert dataset_columns is not None
        image_column = dataset_columns[0]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.caption_column is None:
        assert dataset_columns is not None
        caption_column = dataset_columns[1]
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # In Flax, for seq2seq models we need to pass `decoder_input_ids`
    # as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
    # for that dynamically import the `shift_tokens_right` function from the model file
    model_module = __import__(model.__module__, fromlist=["shift_tokens_right"])
    shift_tokens_right_fn = getattr(model_module, "shift_tokens_right", shift_tokens_right)

    def filter_fn(examples):
        """remove problematic images"""

        bools = []
        for image_file in examples[image_column]:
            try:
                image = Image.open(image_file)
                feature_extractor(images=image, return_tensors="np")
                bools.append(True)
            except Exception:
                bools.append(False)

        return bools

    # Setting padding="max_length" as we need fixed length inputs for jitted functions
    def tokenization_fn(examples, max_target_length):
        """Run tokenization on captions."""

        captions = []
        for caption in examples[caption_column]:
            captions.append(caption.lower() + " " + tokenizer.eos_token)
        targets = captions

        model_inputs = {}
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="np"
            )
        model_inputs["labels"] = labels["input_ids"]
        decoder_input_ids = shift_tokens_right_fn(
            labels["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id
        )
        model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)
        # We need decoder_attention_mask so we can ignore pad tokens from loss
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]
        model_inputs[image_column] = examples[image_column]

        return model_inputs

    def feature_extraction_fn(examples, check_image=True):
        """
        Run feature extraction on images

        If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
        Otherwise, an exception will be thrown.
        """

        model_inputs = {}

        if check_image:
            images = []
            to_keep = []
            for image_file in examples[image_column]:
                try:
                    img = Image.open(image_file)
                    images.append(img)
                    to_keep.append(True)
                except Exception:
                    to_keep.append(False)

            for k, v in examples.items():
                if k != image_column:
                    model_inputs[k] = v[to_keep]
        else:
            images = [Image.open(image_file) for image_file in examples[image_column]]

        encoder_inputs = feature_extractor(images=images, return_tensors="np")
        model_inputs["pixel_values"] = encoder_inputs.pixel_values

        return model_inputs

    def preprocess_fn(examples, max_target_length, check_image=True):
        """Run tokenization + image feature extraction"""

        model_inputs = {}
        # This contains image path column
        model_inputs.update(tokenization_fn(examples, max_target_length))
        model_inputs.update(feature_extraction_fn(model_inputs, check_image=check_image))
        # Remove image path column
        model_inputs.pop(image_column)

        return model_inputs

    features = datasets.Features(
        {
            "pixel_values": datasets.Array3D(
                shape=(
                    getattr(model.config.encoder, "num_channels", 3),
                    model.config.encoder.image_size,
                    model.config.encoder.image_size,
                ),
                dtype="float32",
            ),
            "labels": datasets.Sequence(feature=datasets.Value(dtype="int32", id=None), length=-1, id=None),
            "decoder_input_ids": datasets.Sequence(feature=datasets.Value(dtype="int32", id=None), length=-1, id=None),
            "decoder_attention_mask": datasets.Sequence(
                feature=datasets.Value(dtype="int32", id=None), length=-1, id=None
            ),
        }
    )

    # If `block_size` is `0`, tokenization & image feature extraction is done at the beginning
    run_feat_ext_at_beginning = training_args.block_size == 0
    # Used in .map() below
    function_kwarg = preprocess_fn if run_feat_ext_at_beginning else tokenization_fn
    # `features` is used only for the final preprocessed dataset (for the performance purpose).
    features_kwarg = features if run_feat_ext_at_beginning else None
    # Keep `image_column` if the feature extraction is done during training
    remove_columns_kwarg = [x for x in column_names if x != image_column or run_feat_ext_at_beginning]
    processor_names = "tokenizer and feature extractor" if run_feat_ext_at_beginning else "tokenizer"

    # Store some constant
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    if training_args.block_size % train_batch_size > 0 or training_args.block_size % eval_batch_size > 0:
        raise ValueError(
            f"`training_args.block_size` needs to be a multiple of the global train/eval batch size."
            f"Got {training_args.block_size}, {train_batch_size} and {eval_batch_size} respectively instead."
        )

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # remove problematic examples
        # (if feature extraction is performed at the beginning, the filtering is done during preprocessing below
        # instead here.)
        if not run_feat_ext_at_beginning:
            train_dataset = train_dataset.filter(filter_fn, batched=True, num_proc=data_args.preprocessing_num_workers)
        train_dataset = train_dataset.map(
            function=function_kwarg,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            # kept image paths
            remove_columns=remove_columns_kwarg,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Running {processor_names} on train dataset",
            fn_kwargs={"max_target_length": data_args.max_target_length},
            features=features_kwarg,
        )
        if run_feat_ext_at_beginning:
            # set format (for performance) since the dataset is ready to be used
            train_dataset = train_dataset.with_format("numpy")

        steps_per_epoch = len(train_dataset) // train_batch_size
        num_train_examples_per_epoch = steps_per_epoch * train_batch_size
        num_epochs = int(training_args.num_train_epochs)
        total_train_steps = steps_per_epoch * num_epochs
    else:
        num_train_examples_per_epoch = 0

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        # remove problematic examples
        # (if feature extraction is performed at the beginning, the filtering is done during preprocessing below
        # instead here.)
        if not run_feat_ext_at_beginning:
            eval_dataset = eval_dataset.filter(filter_fn, batched=True, num_proc=data_args.preprocessing_num_workers)
        eval_dataset = eval_dataset.map(
            function=function_kwarg,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            # kept image paths
            remove_columns=remove_columns_kwarg,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Running {processor_names} on validation dataset",
            fn_kwargs={"max_target_length": data_args.val_max_target_length},
            features=features_kwarg,
        )
        if run_feat_ext_at_beginning:
            # set format (for performance) since the dataset is ready to be used
            eval_dataset = eval_dataset.with_format("numpy")

        num_eval_examples = len(eval_dataset)
        eval_steps = num_eval_examples // eval_batch_size

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = dataset["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        # remove problematic examples
        # (if feature extraction is performed at the beginning, the filtering is done during preprocessing below
        # instead here.)
        if not run_feat_ext_at_beginning:
            predict_dataset = predict_dataset.filter(
                filter_fn, batched=True, num_proc=data_args.preprocessing_num_workers
            )
        predict_dataset = predict_dataset.map(
            function=function_kwarg,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            # kept image paths
            remove_columns=remove_columns_kwarg,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Running {processor_names} on prediction dataset",
            fn_kwargs={"max_target_length": data_args.val_max_target_length},
            features=features_kwarg,
        )
        if run_feat_ext_at_beginning:
            # set format (for performance) since the dataset is ready to be used
            predict_dataset = predict_dataset.with_format("numpy")

        num_test_examples = len(predict_dataset)
        test_steps = num_test_examples // eval_batch_size

    def blockwise_data_loader(
        rng: jax.random.PRNGKey,
        ds: Dataset,
        block_size: int,
        batch_size: int,
        shuffle: bool = False,
        keep_in_memory: bool = False,
        split: str = "",
    ):
        """
        Wrap the simple `data_loader` in a block-wise way if `block_size` > 0, else it's the same as `data_loader`.

        If `block_size` > 0, it requires `ds` to have a column that gives image paths in order to perform image feature
        extraction (with the column name being specified by `image_column`). The tokenization should be done before
        training in this case.
        """

        # We use `numpy.ndarray` to interact with `datasets.Dataset`, since using `jax.numpy.array` to index into a
        # dataset is significantly slow. Using JAX array at the 1st place is only to keep JAX's PRNGs generation
        # mechanism, which works differently from NumPy/SciPy.
        if shuffle:
            indices = jax.random.permutation(rng, len(ds))
            indices = np.asarray(indices)
        else:
            indices = np.arange(len(ds))

        _block_size = len(ds) if not block_size else block_size

        steps_per_block = _block_size // batch_size
        num_examples = len(ds)
        steps = num_examples // batch_size
        num_splits = steps // steps_per_block + int(steps % steps_per_block > 0)

        for idx in range(num_splits):

            if not block_size:
                _ds = ds
            else:

                start_idx = block_size * idx
                end_idx = block_size * (idx + 1)

                selected_indices = indices[start_idx:end_idx]

                _ds = ds.select(selected_indices)

                _ds = _ds.map(
                    feature_extraction_fn,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[image_column],
                    load_from_cache_file=not data_args.overwrite_cache,
                    features=features,
                    keep_in_memory=keep_in_memory,
                    # The images are already checked either in `.filter()` or in `preprocess_fn()`
                    fn_kwargs={"check_image": False},
                    desc=f"Running feature extraction on {split} dataset".replace("  ", " "),
                )
                _ds = _ds.with_format("numpy")

            # No need to shuffle here
            loader = data_loader(rng, _ds, batch_size=batch_size, shuffle=False)

            for batch in loader:
                yield batch

    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 6) for k, v in result.items()}

        return result, decoded_preds, decoded_labels

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        num_train_examples_per_epoch,
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxBart.
    # For FlaxT5, one should correct the layer norm parameter naming
    # accordingly - see `run_t5_mlm_flax.py` e.g.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_params = [
            (name, "scale") for name in ["self_attn_layer_norm", "layernorm_embedding", "final_layer_norm"]
        ]
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw, dropout_rng=dropout_rng)

    # label smoothed cross entropy
    def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
        """
        The label smoothing implementation is adapted from Flax's official example:
        https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
        """
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum() / padding_mask.sum()
        return loss

    # Define gradient update step fn
    def train_step(state, batch, label_smoothing_factor=0.0):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch, label_smoothing_factor=0.0):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Define generation function
    max_length = (
        data_args.val_max_target_length if data_args.val_max_target_length is not None else model.config.max_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(batch["pixel_values"], **gen_kwargs)
        return output_ids.sequences

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(
        partial(train_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch", donate_argnums=(0,)
    )
    p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch")
    p_generate_step = jax.pmap(generate_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    if training_args.do_train:
        logger.info("***** Running training *****")
        logger.info(f"  Num train examples = {num_train_examples_per_epoch}")
        logger.info(f"  Num Epochs = {num_epochs}")
        logger.info(f"  Instantaneous train batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
        logger.info(f"  Optimization steps per epoch = {steps_per_epoch}")
        logger.info(f"  Total optimization steps = {total_train_steps}")
    if training_args.do_eval:
        logger.info(f"  Num evaluation examples = {num_eval_examples}")
        logger.info(f"  Instantaneous evaluation batch size per device = {training_args.per_device_eval_batch_size}")
        logger.info(f"  Total evaluation batch size (w. parallel & distributed) = {eval_batch_size}")
        logger.info(f"  Evaluation steps = {eval_steps}")
    if training_args.do_predict:
        logger.info(f"  Num test examples = {num_test_examples}")
        logger.info(f"  Instantaneous test batch size per device = {training_args.per_device_eval_batch_size}")
        logger.info(f"  Total test batch size (w. parallel & distributed) = {eval_batch_size}")
        logger.info(f"  Test steps = {test_steps}")

    # create output directory
    if not os.path.isdir(os.path.join(training_args.output_dir)):
        os.makedirs(os.path.join(training_args.output_dir), exist_ok=True)

    def save_ckpt(ckpt_dir: str, commit_msg: str = ""):
        """save checkpoints and push to Hugging Face Hub if specified"""

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            model.save_pretrained(os.path.join(training_args.output_dir, ckpt_dir), params=params)
            tokenizer.save_pretrained(os.path.join(training_args.output_dir, ckpt_dir))
            if training_args.push_to_hub:
                repo.push_to_hub(commit_message=commit_msg, blocking=False)

    def evaluation_loop(
        rng: jax.random.PRNGKey,
        dataset: Dataset,
        metric_key_prefix: str = "eval",
        ckpt_dir: str = "",
        is_prediction=False,
    ):

        logger.info(f"*** {'Predict' if is_prediction else 'Evaluate'} ***")

        metrics = []
        preds = []
        labels = []

        batches = blockwise_data_loader(
            rng,
            dataset,
            block_size=training_args.block_size,
            batch_size=eval_batch_size,
            keep_in_memory=False,
            shuffle=False,
            split="prediction" if is_prediction else "validation",
        )
        steps = len(dataset) // eval_batch_size
        for _ in tqdm(
            range(steps), desc=f"{'Predicting' if is_prediction else 'Evaluating'}...", position=2, leave=False
        ):
            # Model forward
            batch = next(batches)
            _labels = batch.get("labels", None)
            if not is_prediction and _labels is None:
                raise ValueError("Evaluation requires the validation dataset to have `labels`")

            if _labels is not None:
                _metrics = p_eval_step(state.params, batch)
                metrics.append(_metrics)

            # generation
            if data_args.predict_with_generate:
                generated_ids = p_generate_step(state.params, batch)
                preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
                if _labels is not None:
                    labels.extend(jax.device_get(_labels.reshape(-1, _labels.shape[-1])))

        if metrics:
            # normalize metrics
            metrics = get_metrics(metrics)
            metrics = jax.tree_map(jnp.mean, metrics)

        # compute ROUGE metrics
        generations = []
        rouge_desc = ""
        if data_args.predict_with_generate:
            if labels:
                rouge_metrics, decoded_preds, decoded_labels = compute_metrics(preds, labels)
                metrics.update(rouge_metrics)
                rouge_desc = " ".join(
                    [
                        f"{'Predict' if is_prediction else 'Eval'} {key}: {value} |"
                        for key, value in rouge_metrics.items()
                    ]
                )
                for pred, label in zip(decoded_preds, decoded_labels):
                    pred = pred.replace("\n", " ")
                    label = label.replace("\n", " ")
                    generations.append({"label": label, "pred": pred})
            else:
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                # Some simple post-processing
                decoded_preds = [pred.strip() for pred in decoded_preds]
                # rougeLSum expects newline after each sentence
                decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
                for pred in decoded_preds:
                    pred = pred.replace("\n", " ")
                    generations.append({"pred": pred})

        if metrics:
            # Print metrics and update progress bar
            desc = f"{'Predict' if is_prediction else 'Eval'} Loss: {metrics['loss']} | {rouge_desc})"
            if training_args.do_train and not is_prediction:
                desc = f"Epoch... ({epoch + 1}/{num_epochs} | Step: {cur_step} | " + desc
                epochs.write(desc)
                epochs.desc = desc
            logger.info(desc)

        if jax.process_index() == 0:

            if not os.path.isdir(os.path.join(training_args.output_dir, ckpt_dir)):
                os.makedirs(os.path.join(training_args.output_dir, ckpt_dir), exist_ok=True)

            if metrics:

                # Save metrics (only for the evaluation/prediction being done along with training)
                if has_tensorboard and training_args.do_train:
                    write_metric(
                        summary_writer, metrics, train_time=None, step=cur_step, metric_key_prefix=metric_key_prefix
                    )

                # save final metrics in json
                metrics = {
                    f"{metric_key_prefix}_{metric_name}": round(value.item(), 6)
                    for metric_name, value in metrics.items()
                }
                _path = os.path.join(training_args.output_dir, ckpt_dir, f"{metric_key_prefix}_results.json")
                with open(_path, "w") as f:
                    json.dump(metrics, f, indent=4, sort_keys=True)

                # Update report
                with open(os.path.join(training_args.output_dir, "log"), "a", encoding="UTF-8") as fp:
                    fp.write(desc + "\n")

            # Save generations
            if generations:
                output_file = os.path.join(training_args.output_dir, ckpt_dir, f"{metric_key_prefix}_generation.json")
                with open(output_file, "w", encoding="UTF-8") as fp:
                    json.dump(generations, fp, ensure_ascii=False, indent=4)

    def evaluate(rng: jax.random.PRNGKey, dataset: Dataset, ckpt_dir: str = ""):
        evaluation_loop(rng, dataset, metric_key_prefix="eval", ckpt_dir=ckpt_dir)

    def predict(rng: jax.random.PRNGKey, dataset: Dataset):
        evaluation_loop(rng, dataset, metric_key_prefix="test", is_prediction=True)

    input_rng = None

    if training_args.do_train:

        cur_step = 0
        train_time = 0
        epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

        for epoch in epochs:
            # ======================== Training ================================
            # Create sampling rng
            rng, input_rng = jax.random.split(rng)

            train_metrics = []
            train_batches = blockwise_data_loader(
                input_rng,
                train_dataset,
                block_size=training_args.block_size,
                batch_size=train_batch_size,
                keep_in_memory=True,
                shuffle=True,
                split="train",
            )

            # train
            for (batch_idx, _) in enumerate(tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False)):

                cur_step += 1
                batch = next(train_batches)
                batch_start = time.time()
                state, train_metric = p_train_step(state, batch)
                train_metrics.append(train_metric)
                train_time += time.time() - batch_start
                time_per_step = train_time / cur_step

                # log and save info
                if training_args.logging_steps > 0 and cur_step % training_args.logging_steps == 0:

                    _train_metric = unreplicate(train_metric)
                    desc = f"Epoch... ({epoch + 1}/{num_epochs} | Step: {cur_step} | Loss: {_train_metric['loss']} | Learning Rate: {_train_metric['learning_rate']} | Time per step: {time_per_step})"
                    epochs.desc = desc
                    epochs.write(desc)

                    logger.info(desc)

                    with open(os.path.join(training_args.output_dir, "log"), "a", encoding="UTF-8") as fp:
                        fp.write(desc + "\n")

                    # Save metrics
                    if has_tensorboard and jax.process_index() == 0:
                        write_metric(
                            summary_writer,
                            train_metrics,
                            train_time=train_time,
                            step=cur_step,
                            metric_key_prefix="train",
                        )

                # ======================== Evaluating (inside an epoch) ==============================

                if (
                    training_args.do_eval
                    and (training_args.eval_steps is not None and training_args.eval_steps > 0)
                    and cur_step % training_args.eval_steps == 0
                ):
                    ckpt_dir = f"ckpt_epoch_{epoch + 1}_step_{cur_step}"
                    commit_msg = f"Saving weights and logs of epoch {epoch + 1} - step {cur_step}"
                    evaluate(input_rng, eval_dataset, ckpt_dir)
                    save_ckpt(ckpt_dir=ckpt_dir, commit_msg=commit_msg)

            # ======================== Epoch End ==============================

            # log and save info
            if training_args.logging_steps <= 0:

                logger.info(desc)

                with open(os.path.join(training_args.output_dir, "log"), "a", encoding="UTF-8") as fp:
                    fp.write(desc + "\n")

                # Save metrics
                if has_tensorboard and jax.process_index() == 0:
                    write_metric(
                        summary_writer, train_metrics, train_time=train_time, step=cur_step, metric_key_prefix="train"
                    )

            # ======================== Evaluating (after each epoch) ==============================

            if training_args.do_eval and (training_args.eval_steps is None or training_args.eval_steps <= 0):
                ckpt_dir = f"ckpt_epoch_{epoch + 1}_step_{cur_step}"
                commit_msg = f"Saving weights and logs of epoch {epoch + 1} - step {cur_step}"
                evaluate(input_rng, eval_dataset, ckpt_dir)
                save_ckpt(ckpt_dir=ckpt_dir, commit_msg=commit_msg)

    # ======================== Evaluating | Predicting ==============================

    # Create sampling rng
    if input_rng is None:
        rng, input_rng = jax.random.split(rng)

    # run evaluation without training
    if training_args.do_eval and not training_args.do_train:
        evaluate(input_rng, eval_dataset)

    # run prediction after (or without) training
    if training_args.do_predict:
        predict(input_rng, predict_dataset)


if __name__ == "__main__":
    main()
