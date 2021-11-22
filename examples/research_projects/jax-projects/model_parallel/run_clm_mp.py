#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
Pre-training/Fine-tuning the GPTNeo model for causal language modeling on a text file or a dataset using model parallelism.
"""

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Callable, Optional

import datasets
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.common_utils import onehot, stack_forest
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit
from partitions import set_partitions
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    is_tensorboard_available,
)
from transformers.testing_utils import CaptureLogger


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
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
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        yield batch


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = stack_forest(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir, keep_in_memory=False
        )

        if "validation" not in dataset.keys():
            dataset["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            dataset["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained config and tokenizer
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if training_args.do_train:
        column_names = dataset["train"].column_names
    else:
        column_names = dataset["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

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

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # TODO: weights should be initialized in pjitted fun, this won't work for REALLY large models
    # TODO: when loading from pre-trained model we need to make sure the vocab is divisible by num_partitions
    # GPT2's vocab is odd, we need to resize it for fine-tuning
    model = FlaxAutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
    )

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    optimizer = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    def get_initial_state(params):
        state = optimizer.init(params)
        return tuple(state), params

    # Get PartitionSpec for model params
    param_spec = set_partitions(unfreeze(model.params))

    # Get the PyTree for opt_state, we don't actually initialize the opt_state yet.
    params_shapes = jax.tree_map(lambda x: x.shape, model.params)
    state_shapes = jax.eval_shape(get_initial_state, params_shapes)

    # get PartitionSpec for opt_state, this is very specific to adamw
    # TODO: optax returns different state for different optimizers, how can we handle this generically ?
    # or maybe we don't since in our examples we just use adamw or adafactor
    def get_opt_spec(x):
        if isinstance(x, dict):
            return param_spec
        return None

    opt_state_spec, param_spec = jax.tree_map(
        get_opt_spec, state_shapes, is_leaf=lambda x: isinstance(x, (dict, optax.EmptyState))
    )

    # pjit the get_initial_state function to shard params and init
    # optimizer state in sharded way
    p_get_initial_state = pjit(
        get_initial_state,
        in_axis_resources=None,
        out_axis_resources=(opt_state_spec, param_spec),
    )

    # hack: move the inital params to CPU to free up device memory
    # TODO: allow loading weights on CPU in pre-trained model
    model.params = jax.tree_map(lambda x: np.asarray(x), model.params)

    # mesh defination
    mesh_devices = np.array(jax.devices()).reshape(1, jax.local_device_count())

    # actually initialize the opt_state
    with mesh(mesh_devices, ("dp", "mp")):
        opt_state, params = p_get_initial_state(freeze(model.params))

    # cross-entropy with z loss
    def loss_fn(logits, labels, z_loss=0):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        shift_labels = onehot(shift_labels, shift_logits.shape[-1])

        shift_logits = shift_logits - jax.lax.stop_gradient(shift_logits.max(axis=-1, keepdims=True))
        log_z = jnp.log(jnp.sum(jnp.exp(shift_logits), axis=-1, keepdims=True))
        log_softmax = shift_logits - log_z
        loss = -jnp.sum(shift_labels * log_softmax, axis=-1)

        loss += (1e-4 * jnp.square(log_z.squeeze(-1))) * z_loss

        return loss.mean()

    # Define gradient update step fn
    # TODO: try to use TrainState instead of passing params and opt_state individually
    def train_step(params, opt_state, dropout_rng, batch, step):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = model(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels, z_loss=1.0)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(params)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(step)}
        return new_params, tuple(new_opt_state), new_dropout_rng, metrics, step + 1

    # Define eval fn
    def eval_step(input_ids, labels, params):
        logits = model(input_ids=input_ids, params=params, train=False)[0]
        loss = loss_fn(logits, labels)
        # metrics
        return {"loss": loss}

    p_train_step = pjit(
        train_step,
        in_axis_resources=(param_spec, opt_state_spec, None, None, None),
        out_axis_resources=(param_spec, opt_state_spec, None, None, None),
        donate_argnums=(0, 1),
    )

    p_eval_step = pjit(
        eval_step,
        in_axis_resources=(None, None, param_spec),
        out_axis_resources=None,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    global_step = 0
    # we are not doing 2D parallelism (yet!), this just does model parallelism
    with mesh(mesh_devices, ("dp", "mp")):
        for _ in epochs:
            # ======================== Training ================================
            train_start = time.time()

            # Create sampling rng
            rng, input_rng = jax.random.split(rng)

            # Generate an epoch by shuffling sampling indices from the train dataset
            train_metrics = []
            train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)
            steps_per_epoch = len(train_dataset) // train_batch_size

            # train
            for _ in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
                batch = next(train_loader)
                params, opt_state, dropout_rng, train_metric, global_step = p_train_step(
                    params,
                    opt_state,
                    dropout_rng,
                    batch,
                    global_step,
                )
                train_metrics.append(train_metric)

                cur_step = global_step

                if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                    # Save metrics
                    train_time += time.time() - train_start
                    if has_tensorboard and jax.process_index() == 0:
                        write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                    epochs.write(
                        f"Step... ({cur_step} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
                    )

                    train_metrics = []

                if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                    # ======================== Evaluating ==============================
                    eval_metrics = []
                    eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size)
                    eval_steps = len(eval_dataset) // eval_batch_size

                    for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
                        batch = next(eval_loader)
                        metrics = p_eval_step(batch["input_ids"], batch["labels"], params)
                        eval_metrics.append(metrics)

                    # normalize eval metrics
                    eval_metrics = stack_forest(eval_metrics)
                    eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

                    try:
                        eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
                    except OverflowError:
                        eval_metrics["perplexity"] = float("inf")

                    logger.info(
                        f"Step... ({cur_step} | Eval loss: {eval_metrics['loss']} | Eval Perplexity: {eval_metrics['perplexity']}"
                    )

                if cur_step % training_args.save_steps == 0 and cur_step > 0:
                    # save checkpoint after each epoch and push checkpoint to the hub
                    if jax.process_index() == 0:
                        params = jax.device_get(params)
                        model.save_pretrained(
                            training_args.output_dir,
                            params=params,
                            push_to_hub=training_args.push_to_hub,
                            commit_message=f"Saving weights and logs of step {cur_step}",
                        )


if __name__ == "__main__":
    main()
