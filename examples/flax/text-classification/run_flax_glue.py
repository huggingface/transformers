#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Flax Transformers model for sequence classification on GLUE."""
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import datasets
from datasets import load_dataset, load_metric

import jax
import jax.numpy as jnp
import optax
import transformers
from flax import linen as nn
from flax import optim, struct, traverse_util
from flax.jax_utils import replicate, unreplicate
from flax.metrics import tensorboard
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from transformers import (
    AutoTokenizer,
    BertConfig,
    FlaxAutoModelForSequenceClassification,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging


logger = logging.getLogger(__name__)

Array = Any
Dataset = datasets.arrow_dataset.Dataset
Tokenizer = PreTrainedTokenizerBase
PRNGKey = Any


"""
Input and label field names for the GLUE tasks.
While it is possible to obtain this information from the dataset directly,
this will make the code harder to follow so we just put it here explicitly.
"""
task_to_fields = {
    "cola": (["sentence"], ["unacceptable", "acceptable"]),
    "sst2": (["sentence"], ["negative", "positive"]),
    "mrpc": (["sentence1", "sentence2"], ["not equivalent", "equivalent"]),
    "stsb": (["sentence1", "sentence2"], ["similarily score"]),
    "qqp": (["question1", "question2"], ["not duplicate", "duplicate"]),
    "mnli": (["premise", "hypothesis"], ["entailment", "neutral", "contradition"]),
    "mnli-mm": (["premise", "hypothesis"], ["entailment", "neutral", "contradition"]),
    "qnli": (["question", "sentence"], ["entailment", "not entailment"]),
    "rte": (["sentence1", "sentence2"], ["entailment", "not entailment"]),
    "wnli": (["sentence1", "sentence2"], ["not entailment", "entailment"]),
}


@dataclass
class GlueTask:
    """A description of a GLUE task.

    Args:
      name: The name of the GLUE task (e.g., `cola`). All names are in `task_to_fields.keys()`.
      inputs: The dataset features used as input for this task.
      labels: The dataset features used as labels for this task.
    """

    name: str
    inputs: Iterable[str]
    labels: Iterable[str]

    @property
    def is_regression(self) -> bool:
        return len(self.labels) == 1

    def datasets_name(self, split) -> str:
        """Returns the name of this task so that it is compatible with the `datasets` library."""
        # Only for `mnli` and `mnli-mm` the dev/test set is renamed.
        if not self.name.startswith("mnli"):
            return self.name
        if split == "train":
            return "mnli"
        # `mnli` has dev/test in `mnli_matched`, and `mnli-mm` in mnli_mismatched.
        if self.name == "mnli":
            return "mnli_matched"
        else:  # self.name == mnli-mm
            return "mnli_mismatched"


def get_glue_task(task: str) -> GlueTask:
    return GlueTask(task, *task_to_fields[task])


class TrainState(train_state.TrainState):
    """Train state with an Optax optimizer.

    The two functions below differ depending on whether the task is classification
    or regression.

    Args:
      logits_fn: Applied to last layer to obtain the logits.
      loss_fn: Function to compute the loss.
    """

    logits_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)


@dataclass
class FlaxTrainingArguments(TrainingArguments):
    # We override the next two properties so we don't have to depend on Pytorch.
    @property
    def train_batch_size(self) -> int:
        return self.per_device_train_batch_size * jax.local_device_count()

    @property
    def eval_batch_size(self) -> int:
        return self.per_device_eval_batch_size * jax.local_device_count()

    per_device_num_predictions: int = field(
        default=4, metadata={"help": "Number of predictions to run per device if --do_predict is true."},
    )
    # Set the seed to 2 by default, which produced the best run (see README.md).
    seed: int = field(default=2, metadata={"help": "Random seed that will be set at the beginning of training."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(task_to_fields.keys())},)
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def setup_logging(training_args: FlaxTrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    hf_logging.set_verbosity_info()
    hf_logging.enable_explicit_format()
    hf_logging.enable_default_handler()

    logger.info(f"Training/evaluation parameters {training_args}")

    return logger


def dataclasses_to_json_dict(inputs: Iterable[Any]) -> Dict[str, Union[str, float, int]]:
    """Converts a list of dataclasses to a JSON serializable dict."""
    d_out = {}
    for inp in inputs:
        d_out.update(asdict(inp))
    # Remove values that aren't JSON serializable.
    d_out = {k: v for k, v in d_out.items() if isinstance(v, (str, float, int))}
    return d_out


def create_learning_rate_fn(config: FlaxTrainingArguments, train_ds_size: int) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // config.train_batch_size
    num_train_steps = steps_per_epoch * config.num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=config.learning_rate, transition_steps=config.warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=config.learning_rate, end_value=0, transition_steps=num_train_steps - config.warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[config.warmup_steps])
    return schedule_fn


def create_optimizer(learning_rate_fn: Callable[[int], float]) -> optax.GradientTransformation:
    """Creates a multi-optimizer consisting of two "Adam with weight decay" optimizers."""
    adamw = lambda wd: optax.adamw(learning_rate=learning_rate_fn, b1=0.9, b2=0.999, eps=1e-6, weight_decay=wd)

    def traverse(fn):
        def mask(data):
            flat = traverse_util.flatten_dict(data)
            return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

        return mask

    # We use Optax's "masking" functionality to create a multi-optimizer, one
    # with weight decay and the other without. Note masking means the optimizer
    # will ignore these paths.
    decay_path = lambda p: not any(x in p for x in ["bias", "LayerNorm.weight"])
    return optax.chain(
        optax.masked(adamw(0.0), mask=traverse(lambda path, _: decay_path(path))),
        optax.masked(adamw(0.01), mask=traverse(lambda path, _: not decay_path(path))),
    )


def create_train_state(
    model: FlaxAutoModelForSequenceClassification, learning_rate_fn: Callable[[int], float], task: GlueTask
) -> TrainState:
    """Create initial training state."""
    tx = create_optimizer(learning_rate_fn)

    def mse_loss(logits, labels):
        return jnp.mean((logits[..., 0] - labels) ** 2)

    def cross_entropy_loss(logits, labels):
        logits = nn.log_softmax(logits)
        xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=len(task.labels)))
        return jnp.mean(xentropy)

    if task.is_regression:
        logits_fn = lambda logits: logits[..., 0]
        loss_fn = mse_loss
    else:  # Classification.
        logits_fn = lambda logits: logits.argmax(-1)
        loss_fn = cross_entropy_loss

    return TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx, logits_fn=logits_fn, loss_fn=loss_fn)


def get_batches(rng: PRNGKey, dataset: Dataset, batch_size: int, task: GlueTask, return_inputs: bool = False):
    """Returns batches of size `batch_size` from `dataset`, sharded over all local devices."""
    train_ds_size = len(dataset)
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        inputs = zip(*[batch.pop(i) for i in task.inputs])
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)

        if return_inputs:
            yield batch, inputs
        yield batch


def get_dataset(task: GlueTask, split: str, cache_dir: str, tokenizer: Tokenizer, max_seq_length: int):
    """Returns a preprocessed dataset. All tokenized features are padded to `max_seq_length`."""
    dataset = load_dataset("glue", task.datasets_name(split), split=split, cache_dir=cache_dir)

    def tokenize_fn(examples):
        args = [examples[s] for s in task.inputs]
        return tokenizer(*args, max_length=max_seq_length, padding="max_length", truncation=True)

    return dataset.map(tokenize_fn, remove_columns=["idx"], batched=True)


def train_step(
    state: TrainState, batch: Dict[str, Array], dropout_rng: PRNGKey, learning_rate_fn: Callable[[int], float]
) -> Tuple[TrainState, float]:
    """Trains model with an optimizer (both in `state`) on `batch`, returning a pair `(new_state, loss)`."""
    targets = batch.pop("label")

    def loss_fn(params):
        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        loss = state.loss_fn(logits, targets)
        return loss, logits

    lr = learning_rate_fn(state.step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    metrics = {"loss": loss, "learning_rate": lr}
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return new_state, metrics


def eval_step(state: TrainState, batch: Dict[str, Any]):
    logits = state.apply_fn(**batch, params=state.params, train=False)[0]
    return state.logits_fn(logits)


def log_prediction(task, inputs, prediction):
    prediction_type = "SIMILARITY (1-5)"
    if not task.is_regression:
        prediction_type = "CLASS"
        prediction = task.labels[prediction]
    for field_name, input_str in zip(task.inputs, inputs):
        logger.info(f"{field_name}:\t{input_str}")
    logger.info(f"PREDICTED {prediction_type}: {prediction}\n")


def write_metrics(train_metrics, eval_metrics, train_time, step, summary_writer):
    summary_writer.scalar("train_time", train_time, step)

    if train_metrics:
        train_metrics = get_metrics(train_metrics)
        for key, vals in train_metrics.items():
            tag = f"train_{key}"
            for i, val in enumerate(vals):
                summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    if eval_metrics:
        for metric_name, value in eval_metrics.items():
            summary_writer.scalar(f"eval_{metric_name}", value, step)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FlaxTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger = setup_logging(training_args)

    set_seed(training_args.seed)
    task = get_glue_task(data_args.task_name)

    # Load pretrained model and tokenizer
    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(task.labels),
        finetuning_task=task.name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir, seed=training_args.seed,
    )

    get_split = partial(
        get_dataset,
        task=task,
        cache_dir=model_args.cache_dir,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )

    num_epochs = int(training_args.num_train_epochs)
    rng = jax.random.PRNGKey(training_args.seed)
    batches = partial(get_batches, task=task)

    summary_writer = tensorboard.SummaryWriter(training_args.output_dir)
    hparams = dataclasses_to_json_dict([model_args, data_args, training_args])
    summary_writer.hparams(hparams)
    write_metric = partial(write_metrics, summary_writer=summary_writer)

    if training_args.do_train:
        train_dataset = get_split(split="train")
        train_batch_size = training_args.train_batch_size
        learning_rate_fn = create_learning_rate_fn(training_args, len(train_dataset))
        state = create_train_state(model, learning_rate_fn, task)
        state = replicate(state)
        p_train_step = jax.pmap(
            partial(train_step, learning_rate_fn=learning_rate_fn), axis_name="batch", donate_argnums=(0,)
        )

    if training_args.do_eval:
        eval_dataset = get_split(split="validation")
        eval_batch_size = training_args.eval_batch_size
        p_eval_step = jax.pmap(eval_step, axis_name="batch")

    if training_args.do_predict:
        predict_dataset = get_split(split="test")

    logger.info(f"===== Starting training ({num_epochs} epochs) =====")

    train_time = 0
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}")
        train_metrics = None
        if training_args.do_train:
            logger.info("  Training...")
            train_start = time.time()
            train_metrics = []
            rng, input_rng, dropout_rng = jax.random.split(rng, 3)
            for batch in batches(input_rng, train_dataset, train_batch_size):
                dropout_rngs = shard_prng_key(dropout_rng)
                state, metrics = p_train_step(state, batch, dropout_rngs)
                train_metrics.append(metrics)
            train_time += time.time() - train_start
            logger.info(f"    Done! Training metrics: {unreplicate(metrics)}")

        eval_metrics = None
        if training_args.do_eval:
            logger.info("  Evaluating...")
            eval_metrics = load_metric("glue", task.datasets_name("validation"))
            rng, input_rng = jax.random.split(rng)
            for batch in batches(input_rng, eval_dataset, eval_batch_size):
                labels = batch.pop("label")
                predictions = p_eval_step(state, batch)
                eval_metrics.add_batch(predictions=chain(*predictions), references=chain(*labels))
            eval_metrics = eval_metrics.compute()
            logger.info(f"    Done! Eval metrics: {eval_metrics}")

        cur_step = epoch * (len(train_dataset) // train_batch_size)
        write_metric(train_metrics, eval_metrics, train_time, cur_step)

    logger.info("===== Finished training =====")
    if training_args.do_predict:
        logger.info("===== Running inference =====")
        rng, input_rng = jax.random.split(rng)
        num_predictions = jax.local_device_count() * 4  # Get 4 predictions from each device.
        batch, inputs = next(batches(input_rng, predict_dataset, num_predictions, return_inputs=True))
        _ = batch.pop("label")  # Test sets do not have labels.
        predictions = p_eval_step(state, batch)
        for inp, prediction in zip(inputs, chain(*predictions)):
            log_prediction(task, inp, prediction)


if __name__ == "__main__":
    main()
