# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) with whole word masking on a
text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
import logging
import os
import sys
from dataclasses import dataclass, field

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.optim import Adam
from flax.training import common_utils
from flax.training.common_utils import get_metrics
from jax.nn import log_softmax
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxBertForMaskedLM,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TensorType,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)


# Cache the result
has_tensorboard = is_tensorboard_available()
if has_tensorboard:
    try:
        from flax.metrics.tensorboard import SummaryWriter
    except ImportError as ie:
        has_tensorboard = False
        print(f"Unable to display metrics through TensorBoard because some package are not installed: {ie}")

else:
    print(
        "Unable to display metrics through TensorBoard because the package is not installed: "
        "Please run pip install tensorboard to enable."
    )


MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
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
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
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


# Adapted from transformers/data/data_collator.py
# Letting here for now, let's discuss where it should live
@dataclass
class FlaxDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, examples: List[Dict[str, np.ndarray]], pad_to_multiple_of: int) -> Dict[str, np.ndarray]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, pad_to_multiple_of=pad_to_multiple_of, return_tensors=TensorType.NUMPY)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].copy()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: np.ndarray, special_tokens_mask: Optional[np.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(self.tokenizer.vocab_size, size=labels.shape, dtype="i4")
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_decay",
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000,
):
    """Creates learning rate schedule.
    Interprets factors in the factors string which can consist of:
    * constant: interpreted as the constant value,
    * linear_warmup: interpreted as linear warmup until warmup_steps,
    * rsqrt_decay: divide by square root of max(step, warmup_steps)
    * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
    * decay_every: Every k steps decay the learning rate by decay_factor.
    * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.
    Args:
      factors: string, factors separated by "*" that defines the schedule.
      base_learning_rate: float, the starting constant for the lr schedule.
      warmup_steps: int, how many steps to warm up for in the warmup schedule.
      decay_factor: float, the amount to decay the learning rate by.
      steps_per_decay: int, how often to decay the learning rate.
      steps_per_cycle: int, steps per cycle when using cosine decay.
    Returns:
      a function learning_rate(step): float -> {"learning_rate": float}, the
      step-dependent lr.
    """
    factors = [n.strip() for n in factors.split("*")]

    def step_fn(step):
        """Step to learning rate function."""
        ret = 1.0
        for name in factors:
            if name == "constant":
                ret *= base_learning_rate
            elif name == "linear_warmup":
                ret *= jnp.minimum(1.0, step / warmup_steps)
            elif name == "rsqrt_decay":
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == "rsqrt_normalized_decay":
                ret *= jnp.sqrt(warmup_steps)
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == "decay_every":
                ret *= decay_factor ** (step // steps_per_decay)
            elif name == "cosine_decay":
                progress = jnp.maximum(0.0, (step - warmup_steps) / float(steps_per_cycle))
                ret *= jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
            else:
                raise ValueError("Unknown factor %s." % name)
        return jnp.asarray(ret, dtype=jnp.float32)

    return step_fn


def compute_metrics(logits, labels, weights, label_smoothing=0.0):
    """Compute summary metrics."""
    loss, normalizer = cross_entropy(logits, labels, weights, label_smoothing)
    acc, _ = accuracy(logits, labels, weights)
    metrics = {"loss": loss, "accuracy": acc, "normalizer": normalizer}
    metrics = jax.lax.psum(metrics, axis_name="batch")
    return metrics


def accuracy(logits, targets, weights=None):
    """Compute weighted accuracy for log probs and targets.
    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: None or array of shape [batch, length]
    Returns:
      Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets" % (str(logits.shape), str(targets.shape))
        )

    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    loss *= weights

    return loss.sum(), weights.sum()


def cross_entropy(logits, targets, weights=None, label_smoothing=0.0):
    """Compute cross entropy and entropy for log probs and targets.
    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: None or array of shape [batch, length]
     label_smoothing: label smoothing constant, used to determine the on and off values.
    Returns:
      Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets" % (str(logits.shape), str(targets.shape))
        )

    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_targets = common_utils.onehot(targets, vocab_size, on_value=confidence, off_value=low_confidence)

    loss = -jnp.sum(soft_targets * log_softmax(logits), axis=-1)
    loss = loss - normalizing_constant

    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()
    else:
        normalizing_factor = np.prod(targets.shape)

    return loss.sum(), normalizing_factor


def training_step(optimizer, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        targets = batch.pop("labels")

        # Hide away tokens which doesn't participate in the optimization
        token_mask = jnp.where(targets > 0, 1.0, 0.0)

        pooled, logits = model(**batch, params=params, dropout_rng=dropout_rng, train=True)
        loss, weight_sum = cross_entropy(logits, targets, token_mask)
        return loss / weight_sum

    step = optimizer.state.step
    lr = lr_scheduler_fn(step)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(optimizer.target)
    grad = jax.lax.pmean(grad, "batch")
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    return loss, optimizer, new_dropout_rng


def eval_step(params, batch):
    """
    Calculate evaluation metrics on a batch.
    """
    targets = batch.pop("labels")

    # Hide away tokens which doesn't participate in the optimization
    token_mask = jnp.where(targets > 0, 1.0, 0.0)
    _, logits = model(**batch, params=params, train=False)

    return compute_metrics(logits, targets, token_mask)


def generate_batch_splits(samples_idx: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    nb_samples = len(samples_idx)
    samples_to_remove = nb_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = nb_samples // batch_size
    batch_idx = jnp.split(samples_idx, sections_split)
    return batch_idx


if __name__ == "__main__":
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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level="NOTSET",
        datefmt="[%X]",
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Enable tensorboard only on the master node
    if has_tensorboard and jax.host_id() == 0:
        summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir).joinpath("logs").as_posix())

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = FlaxDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    model = FlaxBertForMaskedLM.from_pretrained("bert-base-cased", dtype=jnp.float32, dropout_rate=0.1)
    model.init(jax.random.PRNGKey(training_args.seed), (training_args.train_batch_size, model.config.max_length))

    # Setup optimizer
    optimizer = Adam(
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        beta1=training_args.adam_beta1,
        beta2=training_args.adam_beta2,
    ).create(model.params)

    # Create learning rate scheduler
    lr_scheduler_fn = create_learning_rate_scheduler(
        base_learning_rate=training_args.learning_rate, warmup_steps=training_args.warmup_steps
    )

    # Create parallel version of the training and evaluation steps
    p_training_step = jax.pmap(training_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    # Replicate the optimizer on each device
    optimizer = jax_utils.replicate(optimizer)

    # Store some constant
    nb_epochs = int(training_args.num_train_epochs)
    batch_size = int(training_args.train_batch_size)
    eval_batch_size = int(training_args.eval_batch_size)

    epochs = tqdm(range(nb_epochs), desc=f"Epoch ... (1/{nb_epochs})", position=0)
    for epoch in epochs:

        # ======================== Training ================================
        # Create sampling rng
        rng, training_rng, eval_rng = jax.random.split(rng, 3)

        # Generate an epoch by shuffling sampling indices from the train dataset
        nb_training_samples = len(tokenized_datasets["train"])
        training_samples_idx = jax.random.permutation(training_rng, jnp.arange(nb_training_samples))
        training_batch_idx = generate_batch_splits(training_samples_idx, batch_size)

        # Gather the indexes for creating the batch and do a training step
        for batch_idx in tqdm(training_batch_idx, desc="Training...", position=1):
            samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples, pad_to_multiple_of=16)

            # Model forward
            model_inputs = common_utils.shard(model_inputs.data)
            loss, optimizer, dropout_rngs = p_training_step(optimizer, model_inputs, dropout_rngs)

        epochs.write(f"Loss: {loss}")

        # ======================== Evaluating ==============================
        nb_eval_samples = len(tokenized_datasets["test"])
        eval_samples_idx = jnp.arange(nb_eval_samples)
        eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

        eval_metrics = []
        for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=2)):
            samples = [tokenized_datasets["test"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples, pad_to_multiple_of=16)

            # Model forward
            model_inputs = common_utils.shard(model_inputs.data)
            metrics = p_eval_step(optimizer.target, model_inputs)
            eval_metrics.append(metrics)

        eval_metrics_np = get_metrics(eval_metrics)
        eval_metrics_np = jax.tree_map(jnp.sum, eval_metrics_np)
        eval_normalizer = eval_metrics_np.pop("normalizer")
        eval_summary = jax.tree_map(lambda x: x / eval_normalizer, eval_metrics_np)

        # Update progress bar
        epochs.desc = (
            f"Epoch... ({epoch + 1}/{nb_epochs} | Loss: {eval_summary['loss']}, Acc: {eval_summary['accuracy']})"
        )

        # Save metrics
        if has_tensorboard and jax.host_id() == 0:
            for name, value in eval_summary.items():
                summary_writer.scalar(name, value, epoch)
